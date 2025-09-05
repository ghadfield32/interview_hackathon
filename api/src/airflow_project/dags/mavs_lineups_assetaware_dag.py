from __future__ import annotations

import sys
import inspect
from pathlib import Path

# Ensure project modules are importable without changing global CWD
THIS = Path(__file__).resolve()
AIRFLOW_PROJECT_ROOT = THIS.parents[1]  # dags -> airflow_project
if str(AIRFLOW_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(AIRFLOW_PROJECT_ROOT))

from utils import config as CFG

from airflow.decorators import dag, task
from airflow.utils.task_group import TaskGroup
from airflow.exceptions import AirflowException


def _supports_deferrable_filesensor() -> bool:
    """Return True if FileSensor accepts 'deferrable' (Airflow >=2.9)."""
    try:
        from airflow.sensors.filesystem import FileSensor  # type: ignore
        return "deferrable" in inspect.signature(FileSensor.__init__).parameters
    except Exception:
        return False


def _file_sensor_partial():
    """Return a FileSensor.partial(...) using config-defined knobs."""
    from airflow.sensors.filesystem import FileSensor  # late import
    kwargs = dict(
        task_id="wait_for_file",
        fs_conn_id=CFG.AIRFLOW_FS_CONN_ID,
        poke_interval=CFG.FILE_SENSOR_POKE_SEC,
        timeout=CFG.FILE_SENSOR_TIMEOUT_SEC,
        soft_fail=False,
    )
    if _supports_deferrable_filesensor():
        kwargs["deferrable"] = True
    return FileSensor.partial(**kwargs)


@dag(
    dag_id="nba_lineups_assetaware_v3",
    # IMPORTANT: cross-version compatible — use 'schedule', not 'timetable'
    schedule=CFG.build_combined_schedule(),
    default_args=CFG.airflow_default_args(),
    catchup=False,
    max_active_runs=1,
    tags=["nba", "lineups", "duckdb", "asset-aware"],
    doc_md="""
### NBA Lineups DAG (Cron + optional Asset/Dataset triggers)
- **Scheduling**: Cron and, when supported by this Airflow, Assets/Datasets.
- **Inputs**: waits on required CSVs via (deferrable) FileSensor mapping.
- **Pipeline**: calls `run_complete_pipeline.run_complete_pipeline()`.
- **Validation**: asserts required CSV exports exist and are non-empty.
""",
)
def lineup_dag():

    required_files = [str(p) for p in CFG.required_input_files()]

    # Wait for all inputs (mapped sensor, deferrable when supported)
    with TaskGroup(group_id="wait_for_inputs", tooltip="Wait for required input CSVs") as wait_for_inputs:
        _file_sensor_partial().expand(filepath=required_files)

    @task
    def print_config() -> None:
        """Emit configuration summary and write column_usage_report.md."""
        CFG.print_configuration_summary()

    @task
    def preflight_checks() -> None:
        """Fail fast if required files are missing or perf config is unsafe."""
        files_ok = CFG.validate_data_files()
        perf_ok = CFG.validate_performance_config()
        if not files_ok:
            raise AirflowException("Preflight failed: required input files missing or empty.")
        if not perf_ok:
            raise AirflowException("Preflight failed: performance configuration invalid.")

    @task
    def run_pipeline() -> dict:
        """Run the complete pipeline; return compact results dict."""
        from run_complete_pipeline import run_complete_pipeline  # late import
        ok, results = run_complete_pipeline(database_path=str(CFG.DUCKDB_PATH))
        if not ok:
            err = (results.get("errors") or ["unknown"])[-1]
            raise AirflowException(f"Pipeline failed: {err}")
        return {
            "outputs": results.get("outputs", {}),
            "total_time": results.get("total_time", 0.0),
            "warnings": results.get("warnings", []),
        }

    @task
    def validate_outputs(results: dict) -> dict:
        """Check required CSVs exist and are non-empty."""
        from pathlib import Path
        outputs = results.get("outputs", {})
        required = [
            "project1_lineups_traditional",
            "project2_players_traditional",
        ]
        missing = [k for k in required if k not in outputs]
        if missing:
            raise AirflowException(f"Missing expected outputs: {missing}")
        zero = []
        for k in required:
            p = Path(outputs[k])
            if not p.exists() or p.stat().st_size == 0:
                zero.append(p.name)
        if zero:
            raise AirflowException(f"Zero-sized outputs: {zero}")
        return {
            "validated": True,
            "export_dir": str(CFG.EXPORTS_DIR),
            "count_required": len(required),
            "warnings": results.get("warnings", []),
            "total_time": float(results.get("total_time", 0.0)),
        }

    @task
    def write_run_report(summary: dict) -> str:
        """Persist a JSON run report and return its path."""
        import json
        from datetime import datetime
        CFG.EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
        report = {
            "validated": summary["validated"],
            "count_required": summary["count_required"],
            "total_time_sec": round(summary["total_time"], 2),
            "warnings": summary.get("warnings", []),
            "generated_at": datetime.utcnow().isoformat() + "Z",
        }
        out = CFG.EXPORTS_DIR / "run_summary.json"
        out.write_text(json.dumps(report, indent=2))
        return str(out)

    wfi = wait_for_inputs
    pc = print_config()
    pf = preflight_checks()
    rp = run_pipeline()
    vo = validate_outputs(rp)
    wr = write_run_report(vo)

    wfi >> pc >> pf >> rp >> vo >> wr


dag = lineup_dag()
