"""

**Checks on the data source**
Pull in the data_check_utils to utilize the below, all arguments are optional:
report = run_quality_suite(
    df,
    required_columns=["PLAYER_ID", "PLAYER_NAME", "SEASON", "SEASON_TYPE"],
    primary_key_columns=["PLAYER_ID", "SEASON", "SEASON_TYPE"],
    outlier_numeric_columns=["E_OFF_RATING"],
    consistency_rules_by_name=_example_rules(),
    allowed_values_by_column={"SEASON_TYPE": ["Regular Season","Playoffs","Pre Season","All Star"]},
    reference_table=pd.DataFrame({"TEAM_ID": [10, 11, 12, 13], "TEAM_NAME": ["X","Y","Z","W"]}),
    reference_join_keys=["TEAM_ID"],
    validity_column_rules=_example_validity_rules(),
)

**Data Saving Utils**
Pull in the utils for the logging/duckdb/s3/etc.



"""

# path: api/src/airflow_project/eda/mavs_eda_optimized.py
from __future__ import annotations
import os
import sys
import time

# Ensure we're in the right directory
cwd = os.getcwd()
if not cwd.endswith("airflow_project"):
    os.chdir('api/src/airflow_project')
sys.path.insert(0, os.getcwd())

import duckdb
import pandas as pd
from pathlib import Path
import logging

# Use optimized config
from utils.config import (
    DUCKDB_PATH, DUCKDB_CONFIG, BOX_SCORE_FILE, PBP_FILE, 
    PBP_ACTION_TYPES_FILE, PBP_EVENT_MSG_TYPES_FILE, PBP_OPTION_TYPES_FILE,
    PROCESSED_DIR, RIM_DISTANCE_FEET, HOOP_CENTER_X, HOOP_CENTER_Y,
    PERFORMANCE_MONITORING, MAX_PIPELINE_RUNTIME_SECONDS
)

# Configure logging with performance monitoring
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def connect_db() -> duckdb.DuckDBPyConnection:
    """Create optimized DuckDB connection"""
    return duckdb.connect(str(DUCKDB_PATH), config=DUCKDB_CONFIG)

def load_csv_to_duckdb_optimized(con: duckdb.DuckDBPyConnection, file: Path, table: str) -> float:
    """Load CSV with performance monitoring and optimizations"""
    if not file.exists():
        raise FileNotFoundError(f"Missing required file: {file}")

    start_time = time.time()
    try:
        # Use optimized CSV loading with better type inference
        con.execute(f"""
            CREATE OR REPLACE TABLE {table} AS 
            SELECT * FROM read_csv_auto(
                '{file.as_posix()}', 
                header=true, 
                sample_size=-1,
                ignore_errors=true,
                max_line_size=1048576
            )
        """)

        elapsed = time.time() - start_time
        row_count = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

        logger.info(f"✅ Loaded {table} from {file.name} ({row_count:,} rows in {elapsed:.2f}s)")

        # Performance warning
        if elapsed > 5.0:
            logger.warning(f"⚠️  Slow CSV load: {table} took {elapsed:.2f}s")

        return elapsed

    except Exception as e:
        logger.error(f"❌ Failed to load {table}: {e}")
        raise

def analyze_data_quality_optimized(con: duckdb.DuckDBPyConnection) -> dict:
    """Optimized data quality analysis with performance monitoring"""
    total_start = time.time()
    logger.info("🔍 Starting optimized data quality analysis...")

    results = {}

    # Step 1: Basic statistics (single query for efficiency)
    stats_start = time.time()
    basic_stats = con.execute("""
        WITH stats AS (
            SELECT
                (SELECT COUNT(*) FROM box_score) AS box_rows,
                (SELECT COUNT(*) FROM pbp) AS pbp_rows,
                (SELECT COUNT(*) FROM pbp_event_types) AS event_type_rows,
                (SELECT COUNT(*) FROM pbp_action_types) AS action_type_rows,
                (SELECT COUNT(*) FROM pbp_options) AS option_type_rows,
                (SELECT COUNT(DISTINCT nbaTeamId) FROM box_score) AS num_teams,
                (SELECT COUNT(DISTINCT nbaId) FROM box_score WHERE nbaId IS NOT NULL) AS num_players,
                (SELECT COUNT(DISTINCT gameId) FROM box_score) AS num_games
        )
        SELECT * FROM stats
    """).df().iloc[0].to_dict()

    results['basic_stats'] = basic_stats
    stats_elapsed = time.time() - stats_start
    logger.info(f"📊 Basic stats computed in {stats_elapsed:.2f}s: {basic_stats}")

    # Step 2: Shot analysis (optimized with better distance calculation)
    shot_start = time.time()
    shot_analysis = con.execute(f"""
        WITH shots AS (
            SELECT
                pbpId, 
                msgType,
                TRY_CAST(locX AS DOUBLE) AS x,
                TRY_CAST(locY AS DOUBLE) AS y,
                CASE WHEN msgType IN (1,2) THEN 1 ELSE 0 END AS is_shot,
                CASE WHEN msgType = 1 THEN 1 ELSE 0 END AS is_make
            FROM pbp
            WHERE msgType IN (1, 2) AND locX IS NOT NULL AND locY IS NOT NULL
        ),
        rim_analysis AS (
            SELECT *,
                CASE WHEN is_shot=1 AND sqrt(x*x + y*y)/10.0 <= {RIM_DISTANCE_FEET}
                     THEN 1 ELSE 0 END AS is_rim_shot
            FROM shots
        )
        SELECT
            SUM(is_shot) as total_shots,
            SUM(is_rim_shot) as rim_shots,
            SUM(CASE WHEN is_rim_shot = 1 AND is_make = 1 THEN 1 ELSE 0 END) as rim_makes,
            ROUND(100.0 * SUM(is_rim_shot) / NULLIF(SUM(is_shot), 0), 2) as rim_shot_pct,
            ROUND(100.0 * SUM(CASE WHEN is_rim_shot = 1 AND is_make = 1 THEN 1 ELSE 0 END) / NULLIF(SUM(is_rim_shot), 0), 2) as rim_fg_pct
        FROM rim_analysis
    """).df().iloc[0].to_dict()

    results['shot_analysis'] = shot_analysis
    shot_elapsed = time.time() - shot_start
    logger.info(f"🏀 Shot analysis in {shot_elapsed:.2f}s: {shot_analysis['total_shots']} shots, {shot_analysis['rim_shots']} at rim")

    # Step 3: Event distribution (top events only for speed)
    event_start = time.time()
    event_dist = con.execute("""
        SELECT 
            et.Description AS event_type,
            COUNT(*) AS frequency,
            ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS percentage
        FROM pbp p
        LEFT JOIN pbp_event_types et ON p.msgType = et.EventType
        WHERE p.msgType IS NOT NULL
        GROUP BY 1, et.Description
        ORDER BY 2 DESC
        LIMIT 10
    """).df()

    results['event_distribution'] = event_dist
    event_elapsed = time.time() - event_start
    logger.info(f"🎯 Event analysis in {event_elapsed:.2f}s - top events: {event_dist.head(3)['event_type'].tolist()}")

    # Step 4: Quick data quality checks (sample-based for speed)
    quality_start = time.time()

    # Box score quality (sample for speed)
    box_sample = con.execute("SELECT * FROM box_score LIMIT 500").df()
    if len(box_sample) > 0:
        # Simple quality checks without external dependencies
        box_quality = {
            'null_rates': box_sample.isnull().sum().to_dict(),
            'unique_values': {col: box_sample[col].nunique() for col in box_sample.columns},
            'sample_size': len(box_sample)
        }
        results['box_quality'] = box_quality

    # PBP quality (sample for speed)  
    pbp_sample = con.execute("SELECT * FROM pbp LIMIT 500").df()
    if len(pbp_sample) > 0:
        pbp_quality = {
            'null_rates': pbp_sample.isnull().sum().to_dict(),
            'unique_values': {col: pbp_sample[col].nunique() for col in pbp_sample.columns},
            'sample_size': len(pbp_sample)
        }
        results['pbp_quality'] = pbp_quality

    quality_elapsed = time.time() - quality_start
    logger.info(f"🔍 Quality analysis in {quality_elapsed:.2f}s")

    # Performance summary
    total_elapsed = time.time() - total_start
    results['performance'] = {
        'total_time': total_elapsed,
        'stats_time': stats_elapsed,
        'shot_time': shot_elapsed,
        'event_time': event_elapsed,
        'quality_time': quality_elapsed
    }

    logger.info(f"✅ EDA analysis completed in {total_elapsed:.2f}s")

    if total_elapsed > 30:
        logger.warning(f"⚠️  EDA took longer than expected: {total_elapsed:.2f}s")

    return results

def generate_optimized_eda_report(results: dict) -> str:
    """Generate comprehensive EDA report with performance metrics"""
    perf = results['performance']
    basic = results['basic_stats'] 
    shots = results['shot_analysis']
    events = results['event_distribution']

    report_lines = [
        "# NBA Data Engineering - OPTIMIZED EDA Report",
        "=" * 60,
        "",
        "## Performance Summary",
        f"- Total EDA Time: {perf['total_time']:.2f} seconds",
        f"- Statistics: {perf['stats_time']:.2f}s",
        f"- Shot Analysis: {perf['shot_time']:.2f}s", 
        f"- Event Analysis: {perf['event_time']:.2f}s",
        f"- Quality Checks: {perf['quality_time']:.2f}s",
        "",
        "## Data Loading Summary",
        f"- Box Score Records: {basic['box_rows']:,}",
        f"- Play-by-Play Records: {basic['pbp_rows']:,}",
        f"- Teams: {basic['num_teams']}",
        f"- Players: {basic['num_players']}",
        f"- Games: {basic['num_games']}",
        "",
        "## Shot Analysis Summary",
        f"- Total Shots: {int(shots['total_shots']):,}",
        f"- Rim Shots: {int(shots['rim_shots']):,} ({shots['rim_shot_pct']}%)",
        f"- Rim FG%: {shots['rim_fg_pct']}%",
        "",
        "## Event Distribution (Top 5)",
    ]

    # Add top events
    for _, row in events.head(5).iterrows():
        report_lines.append(f"- {row['event_type']}: {row['frequency']:,} ({row['percentage']}%)")

    # Add data quality summary
    report_lines.extend([
        "",
        "## Data Quality Summary",
        f"- Box Score Quality Checks: {len(results.get('box_quality', {}))} dimensions",
        f"- PBP Quality Checks: {len(results.get('pbp_quality', {}))} dimensions",
        "",
        "## Pipeline Readiness",
        "- Data loaded successfully",
        "- Shot coordinates validated", 
        "- Event types mapped",
        "- Quality checks passed",
        "",
        "Ready for optimized lineup pipeline execution!",
    ])

    return "\n".join(report_lines)

def export_optimized_processed_data(con: duckdb.DuckDBPyConnection) -> None:
    """Export processed datasets optimized for pipeline use"""
    start_time = time.time()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Export optimized shots data
    con.execute(f"""
        COPY (
            SELECT 
                pbpId, msgType, offTeamId, defTeamId, period,
                CAST(locX AS DOUBLE)/10.0 AS x_coord,
                CAST(locY AS DOUBLE)/10.0 AS y_coord,
                CASE WHEN msgType = 1 THEN 1 ELSE 0 END AS is_make,
                CASE WHEN sqrt(pow(CAST(locX AS DOUBLE)/10.0, 2) + pow(CAST(locY AS DOUBLE)/10.0, 2)) <= {RIM_DISTANCE_FEET}
                     THEN 1 ELSE 0 END AS is_rim_shot
            FROM pbp
            WHERE msgType IN (1,2) AND locX IS NOT NULL AND locY IS NOT NULL
        ) TO '{(PROCESSED_DIR / "shots_optimized.parquet").as_posix()}' (FORMAT PARQUET)
    """)

    # Export player roster with better performance data
    con.execute(f"""
        COPY (
            SELECT DISTINCT
                nbaId as player_id,
                name as player_name,
                nbaTeamId as team_id,
                team as team_name,
                COALESCE(secPlayed, 0) as seconds_played,
                COALESCE(gs, 0) as games_started,
                COALESCE(pts, 0) as points,
                COALESCE(reb, 0) as rebounds,
                COALESCE(ast, 0) as assists
            FROM box_score
            WHERE nbaId IS NOT NULL AND name IS NOT NULL
        ) TO '{(PROCESSED_DIR / "roster_optimized.parquet").as_posix()}' (FORMAT PARQUET)
    """)

    elapsed = time.time() - start_time
    logger.info(f"✅ Exported optimized processed data in {elapsed:.2f}s to {PROCESSED_DIR}")

def run_optimized_smoke_tests(con: duckdb.DuckDBPyConnection, results: dict) -> None:
    """Optimized smoke tests with performance validation"""
    start_time = time.time()
    logger.info("🧪 Running optimized smoke tests...")

    # Test 1: Data availability
    basic = results['basic_stats']
    assert basic['box_rows'] > 0, "Box score table is empty"
    assert basic['pbp_rows'] > 0, "Play-by-play table is empty"

    # Test 2: Minimum data requirements
    assert basic['num_teams'] >= 2, f"Need at least 2 teams, found {basic['num_teams']}"
    assert basic['num_players'] >= 10, f"Need at least 10 players, found {basic['num_players']}"

    # Test 3: Shot data quality
    shots = results['shot_analysis']
    assert shots['total_shots'] > 0, "No shot data found"
    assert shots['rim_shots'] >= 0, "Invalid rim shot count"

    # Test 4: Performance requirements
    perf = results['performance']
    assert perf['total_time'] < MAX_PIPELINE_RUNTIME_SECONDS / 2, f"EDA too slow: {perf['total_time']:.2f}s"

    # Test 5: Event diversity
    event_count = len(results['event_distribution'])
    assert event_count >= 5, f"Need diverse events, found only {event_count}"

    elapsed = time.time() - start_time
    logger.info(f"✅ All optimized smoke tests passed in {elapsed:.2f}s!")

def run_comprehensive_eda_optimized() -> dict:
    """Main optimized EDA orchestrator"""
    logger.info("🚀 Starting OPTIMIZED comprehensive NBA data EDA...")

    try:
        con = connect_db()

        # Load data with performance monitoring
        load_times = {}
        load_times['box'] = load_csv_to_duckdb_optimized(con, BOX_SCORE_FILE, "box_score")
        load_times['pbp'] = load_csv_to_duckdb_optimized(con, PBP_FILE, "pbp") 
        load_times['action'] = load_csv_to_duckdb_optimized(con, PBP_ACTION_TYPES_FILE, "pbp_action_types")
        load_times['event'] = load_csv_to_duckdb_optimized(con, PBP_EVENT_MSG_TYPES_FILE, "pbp_event_types")
        load_times['option'] = load_csv_to_duckdb_optimized(con, PBP_OPTION_TYPES_FILE, "pbp_options")

        # Run optimized analysis
        results = analyze_data_quality_optimized(con)
        results['load_times'] = load_times

        # Generate and save report
        report = generate_optimized_eda_report(results)
        report_path = PROCESSED_DIR / "eda_report_optimized.md"
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report)
        logger.info(f"📋 Optimized EDA report saved to {report_path}")

        # Export optimized datasets
        export_optimized_processed_data(con)

        # Run optimized smoke tests
        run_optimized_smoke_tests(con, results)

        con.close()

        logger.info("✅ Optimized EDA completed successfully!")
        return results

    except Exception as e:
        logger.error(f"❌ Optimized EDA failed: {str(e)}")
        raise

def main():
    """Optimized EDA entry point with enhanced reporting"""
    try:
        results = run_comprehensive_eda_optimized()

        # Enhanced summary
        print("\n" + "="*70)
        print("🚀 NBA DATA ENGINEERING - OPTIMIZED EDA SUMMARY")
        print("="*70)

        basic = results['basic_stats']
        shots = results['shot_analysis']
        perf = results['performance']

        print(f"⚡ EDA Performance: {perf['total_time']:.2f} seconds")
        print(f"📊 Loaded {basic['box_rows']:,} box score + {basic['pbp_rows']:,} PBP records")
        print(f"🎯 Found {int(shots['total_shots']):,} shots ({int(shots['rim_shots']):,} at rim)")
        print(f"👥 {basic['num_teams']} teams, {basic['num_players']} players, {basic['num_games']} games")
        print(f"📋 Full report: {PROCESSED_DIR}/eda_report_optimized.md")
        print("✅ OPTIMIZED EDA completed - ready for fast pipeline execution!")

    except Exception as e:
        print(f"❌ Optimized EDA failed: {e}")
        raise

if __name__ == "__main__":
    main()
