# path: api/src/airflow_project/utils/config.py
from __future__ import annotations
from pathlib import Path
from typing import Dict
import os
from datetime import timedelta


def _find_project_root(anchor: str = "airflow_project") -> Path:
    """Return the project root by locating the given anchor directory."""
    p = Path(__file__).resolve()
    for parent in (p, *p.parents):
        if parent.name == anchor:
            return parent
    return Path.cwd()


# Core project paths
PROJECT_ROOT: Path = _find_project_root()
DATA_DIR: Path = PROJECT_ROOT / "data"
MAVS_DATA_DIR: Path = DATA_DIR / "mavs_data_engineer_2025"
PROCESSED_DIR: Path = MAVS_DATA_DIR / "processed"
EXPORTS_DIR: Path = MAVS_DATA_DIR / "exports"
DUCKDB_DIR: Path = MAVS_DATA_DIR / "duckdb"
LOGS_DIR: Path = PROJECT_ROOT / "logs"

# Ensure directories exist
for directory in (MAVS_DATA_DIR, PROCESSED_DIR, EXPORTS_DIR, DUCKDB_DIR, LOGS_DIR):
    directory.mkdir(parents=True, exist_ok=True)

# Database configuration
DUCKDB_PATH: Path = DUCKDB_DIR / "mavs_enhanced.duckdb"
DUCKDB_CONFIG: Dict[str, str] = {
    "threads": str(min(8, os.cpu_count() or 1)),
    "memory_limit": "6GB",
    "preserve_insertion_order": "false",
    "max_memory": "6GB",
    "temp_directory": str(DUCKDB_DIR / "temp"),
    "checkpoint_threshold": "1GB",
}

# Input data files
BOX_SCORE_FILE: Path = MAVS_DATA_DIR / "box_HOU-DAL.csv"
PBP_FILE: Path = MAVS_DATA_DIR / "pbp_HOU-DAL.csv"
PBP_ACTION_TYPES_FILE: Path = MAVS_DATA_DIR / "pbp_action_types.csv"
PBP_EVENT_MSG_TYPES_FILE: Path = MAVS_DATA_DIR / "pbp_event_msg_types.csv"
PBP_OPTION_TYPES_FILE: Path = MAVS_DATA_DIR / "pbp_option_types.csv"

# === COLUMN SPECIFICATIONS ===

# Box Score Table - Columns used
BOX_SCORE_COLUMNS = {
    "core": [
        "gameId",
        "nbaId",
        "name",
        "nbaTeamId",
        "team",
    ],
    "lineup_tracking": [
        "gs",            # starter flag
        "boxScoreOrder", # sort key for lineup ordering
    ],
    "performance": [
        "secPlayed",
        "pts",
        "reb",
        "ast",
    ],
    "optional": [
        "minDisplay",
        "jerseyNum",
        "startPos",
    ],
}

# Play-by-Play Table - Columns used
PBP_COLUMNS = {
    "core": [
        "gameId",
        "pbpId",
        "period",
        "msgType",
    ],
    "timing": [
        "gameClock",
        "wallClock",
        "wallClockInt",
    ],
    "team_context": [
        "offTeamId",
        "defTeamId",
    ],
    "player_context": [
        "playerId1",
        "playerId2",
        "playerId3",
    ],
    "shot_data": [
        "locX",
        "locY",
        "pts",
    ],
    "event_details": [
        "actionType",
        "option1",
        "option2",
        "option3",
        "description",
    ],
}

# Reference Tables - Full columns needed
PBP_EVENT_TYPES_COLUMNS = {"all": ["EventType", "Description"]}
PBP_ACTION_TYPES_COLUMNS = {"all": ["EventType", "ActionType", "Event", "Description"]}
PBP_OPTION_TYPES_COLUMNS = {
    "all": ["Event", "EventType", "Option1", "Option2", "Option3", "Option4", "Description"]
}

# === OUTPUT TABLE SPECIFICATIONS ===

LINEUPS_OUTPUT_COLUMNS = [
    "Team",
    "Player 1",
    "Player 2",
    "Player 3",
    "Player 4",
    "Player 5",
    "Offensive possessions played",
    "Defensive possessions played",
    "Offensive rating",
    "Defensive rating",
    "Net rating",
]

PLAYERS_OUTPUT_COLUMNS = [
    "Player ID",
    "Player Name",
    "Team",
    "Offensive possessions played",
    "Defensive possessions played",
    "Opponent rim field goal percentage when player is on the court",
    "Opponent rim field goal percentage when player is off the court",
    "Opponent rim field goal percentage on/off difference (on-off)",
]

# === BUSINESS RULES ===

RIM_DISTANCE_FEET: float = 4.0
HOOP_CENTER_X: float = 0.0
HOOP_CENTER_Y: float = 0.0
COORDINATE_SCALE: float = 10.0

MINIMUM_POSSESSIONS_FOR_LINEUP: int = 2
MINIMUM_ATTEMPTS_FOR_RIM_STATS: int = 1
MINIMUM_SECONDS_PLAYED: int = 30

PERFORMANCE_MONITORING: bool = True
MAX_PIPELINE_RUNTIME_SECONDS: int = 120
WARN_IF_SLOWER_THAN_SECONDS: int = 30

EXTREME_DISTANCE_FEET: float = 35.0
TREAT_ZERO_TEAM_AS_ADMIN: bool = True
MAX_CONSECUTIVE_SUBSTITUTION_FAILURES: int = 5

# === NBA SUBSTITUTION CONFIGURATION ===

NBA_SUBSTITUTION_CONFIG = {
    "starter_reset_periods": [1, 3],     # reset to starters at Q1 and Q3
    "lineup_continuity_periods": [2, 4], # continue prior lineups at Q2 and Q4

    "msg_types": {
        "shot_made": 1,
        "shot_missed": 2,
        "rebound": 4,
        "turnover": 5,
        "foul": 6,
        "substitution": 8,
        "start_period": 12,
        "end_period": 13,
    },

    "one_direction": {
        "enabled": True,
        "remove_out_if_present": True,
        "appearance_via_last_name": True,
        "allow_temp_sixth": True,
        "max_lineup_size": 6,
    },

    "validation": {
        "validate_team_membership": True,
        "validate_pre_sub_state": True,
        "min_lineup_size": 5,
        "hard_max_lineup_size": 6,
    },

    "minutes_validation": {
        "enabled": True,
        "tolerance_seconds": 60,
    },

    "recovery": {
        "enable_intelligent_recovery": True,
        "log_recovery_details": True,
        "max_recovery_attempts": 3,
        "prefer_conservative_approach": True,
        "validate_post_recovery": True,
    },

    "debug": {
        "log_all_substitutions": True,
        "log_lineup_state_changes": True,
        "log_period_transitions": True,
        "include_player_context": True,
        "track_recovery_statistics": True,
    },

    "performance": {
        "batch_substitution_processing": False,
        "cache_player_lookups": True,
        "optimize_lineup_comparisons": True,
    },
}

# === AIRFLOW SCHEDULING & SENSOR CONFIG ===

AIRFLOW_OWNER = "nba-analytics"
AIRFLOW_TIMEZONE = "America/New_York"
SCHEDULE_CRON = "0 6 * * *"          # daily at 06:00 NY time
AIRFLOW_RETRIES = 1
AIRFLOW_RETRY_DELAY_MIN = 2

AIRFLOW_FS_CONN_ID = "fs_default"
FILE_SENSOR_POKE_SEC = 30
FILE_SENSOR_TIMEOUT_SEC = 60 * 60     # 1 hour


def airflow_default_args() -> dict:
    """Build default_args with a timezone-aware start_date (uses pendulum if available)."""
    try:
        import pendulum
        start = pendulum.datetime(2025, 1, 1, tz=AIRFLOW_TIMEZONE)
    except Exception:
        from datetime import datetime
        start = datetime(2025, 1, 1)
    return {
        "owner": AIRFLOW_OWNER,
        "depends_on_past": False,
        "start_date": start,
        "email_on_failure": False,
        "email_on_retry": False,
        "retries": AIRFLOW_RETRIES,
        "retry_delay": timedelta(minutes=AIRFLOW_RETRY_DELAY_MIN),
    }


def required_input_files():
    """Absolute paths the DAG must see before running the pipeline."""
    return [
        BOX_SCORE_FILE,
        PBP_FILE,
        PBP_ACTION_TYPES_FILE,
        PBP_EVENT_MSG_TYPES_FILE,
        PBP_OPTION_TYPES_FILE,
    ]


def get_input_assets_or_datasets():
    """
    Prefer Assets (Airflow 3.0+) else Datasets (Airflow 2.9/2.10).
    Returns (objects, kind) where kind is 'asset', 'dataset', or None.
    """
    files = [str(p) for p in required_input_files()]

    # Airflow 3.x Asset API (Task SDK)
    try:
        from airflow.sdk import Asset  # Airflow 3.0+
        return [Asset(f) for f in files], "asset"
    except Exception:
        pass

    # Airflow 2.9/2.10 Dataset API
    try:
        from airflow.datasets import Dataset
        return [Dataset(f) for f in files], "dataset"
    except Exception:
        return [], None


def build_combined_schedule():
    """
    Return a schedule that works across Airflow versions:
      - Prefer AssetOrTimeSchedule (AF >= 3.0) or DatasetOrTimeSchedule (AF 2.9/2.10)
      - Otherwise, return a plain cron string which all versions accept via 'schedule='
    """
    cron_expr = SCHEDULE_CRON
    objects, kind = get_input_assets_or_datasets()

    # AF 3.x: Assets + cron
    if objects and kind == "asset":
        try:
            from airflow.timetables.assets import AssetOrTimeSchedule
            from airflow.timetables.trigger import CronTriggerTimetable
            return AssetOrTimeSchedule(
                timetable=CronTriggerTimetable(cron_expr, timezone=AIRFLOW_TIMEZONE),
                assets=tuple(objects),
            )
        except Exception:
            pass

    # AF 2.9/2.10: Datasets + cron
    if objects and kind == "dataset":
        try:
            from airflow.timetables.datasets import DatasetOrTimeSchedule
            from airflow.timetables.trigger import CronTriggerTimetable
            return DatasetOrTimeSchedule(
                timetable=CronTriggerTimetable(cron_expr, timezone=AIRFLOW_TIMEZONE),
                datasets=tuple(objects),
            )
        except Exception:
            pass

    # Fallback that is universally valid with DAG(schedule=...)
    return cron_expr



def validate_data_files() -> bool:
    """Validate required data files exist and are non-empty."""
    required_files = [
        BOX_SCORE_FILE,
        PBP_FILE,
        PBP_ACTION_TYPES_FILE,
        PBP_EVENT_MSG_TYPES_FILE,
        PBP_OPTION_TYPES_FILE,
    ]
    missing_files, empty_files = [], []

    for file_path in required_files:
        if not file_path.exists():
            missing_files.append(file_path)
        elif file_path.stat().st_size == 0:
            empty_files.append(file_path)

    if missing_files:
        print("Missing required data files:", [str(f.name) for f in missing_files])
        return False

    if empty_files:
        print("Warning: empty data files found:", [str(f.name) for f in empty_files])
    return True


def validate_performance_config() -> bool:
    """Validate DuckDB performance settings."""
    cpu_count = os.cpu_count() or 1
    configured_threads = int(DUCKDB_CONFIG.get("threads", "1"))
    if configured_threads > cpu_count:
        print(f"Warning: configured threads ({configured_threads}) > CPU count ({cpu_count})")

    try:
        import psutil
        available_gb = psutil.virtual_memory().total / (1024**3)
        configured_gb = int(DUCKDB_CONFIG.get("memory_limit", "4GB").rstrip("GB"))
        if configured_gb > available_gb * 0.8:
            print(f"Warning: configured memory ({configured_gb}GB) > 80% of available ({available_gb:.1f}GB)")
    except ImportError:
        print("Note: psutil not available; skipping memory validation")
    return True


def get_column_usage_report() -> str:
    """Generate a column usage report for documentation."""
    report_lines = [
        "# NBA Pipeline Column Usage Report",
        "",
        "## Box Score Table",
        f"- Core: {BOX_SCORE_COLUMNS['core']}",
        f"- Lineup Tracking: {BOX_SCORE_COLUMNS['lineup_tracking']}",
        f"- Performance: {BOX_SCORE_COLUMNS['performance']}",
        f"- Optional: {BOX_SCORE_COLUMNS['optional']}",
        "",
        "## Play-by-Play Table",
        f"- Core: {PBP_COLUMNS['core']}",
        f"- Timing: {PBP_COLUMNS['timing']}",
        f"- Team Context: {PBP_COLUMNS['team_context']}",
        f"- Player Context: {PBP_COLUMNS['player_context']}",
        f"- Shot Data: {PBP_COLUMNS['shot_data']}",
        f"- Event Details: {PBP_COLUMNS['event_details']}",
        "",
        "## Output Tables",
        f"- Lineups Columns: {len(LINEUPS_OUTPUT_COLUMNS)} columns",
        f"- Players Columns: {len(PLAYERS_OUTPUT_COLUMNS)} columns",
        "",
        "## Business Rules",
        f"- Rim Distance: {RIM_DISTANCE_FEET} feet",
        f"- Min Possessions: {MINIMUM_POSSESSIONS_FOR_LINEUP}",
        f"- Min Rim Attempts: {MINIMUM_ATTEMPTS_FOR_RIM_STATS}",
        f"- Performance Target: {MAX_PIPELINE_RUNTIME_SECONDS}s",
    ]
    return "\n".join(report_lines)


def print_configuration_summary() -> None:
    """Print a concise configuration summary and write the column usage report."""
    print("=" * 80)
    print("NBA PIPELINE - CONFIGURATION")
    print("=" * 80)
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Database: {DUCKDB_PATH}")
    print(f"Memory Limit: {DUCKDB_CONFIG.get('memory_limit')}")
    print(f"Threads: {DUCKDB_CONFIG.get('threads')}")

    total_box_cols = sum(len(cols) for cols in BOX_SCORE_COLUMNS.values())
    print(f"\nBox Score Columns: {total_box_cols} total")
    for category, cols in BOX_SCORE_COLUMNS.items():
        print(f"   - {category}: {len(cols)} columns")

    total_pbp_cols = sum(len(cols) for cols in PBP_COLUMNS.values())
    print(f"\nPBP Columns: {total_pbp_cols} total")
    for category, cols in PBP_COLUMNS.items():
        print(f"   - {category}: {len(cols)} columns")

    print("\nBusiness Rules:")
    print(f"   - Rim Distance: {RIM_DISTANCE_FEET} feet")
    print(f"   - Min Lineup Possessions: {MINIMUM_POSSESSIONS_FOR_LINEUP}")
    print(f"   - Performance Target: {MAX_PIPELINE_RUNTIME_SECONDS}s")

    print("\nOutput Tables:")
    print(f"   - Lineups: {len(LINEUPS_OUTPUT_COLUMNS)} columns")
    print(f"   - Players: {len(PLAYERS_OUTPUT_COLUMNS)} columns")
    print("=" * 80)
    report_path = LOGS_DIR / "column_usage_report.md"
    report_path.write_text(get_column_usage_report(), encoding="utf-8")
    print(f"Column usage report: {report_path}")
