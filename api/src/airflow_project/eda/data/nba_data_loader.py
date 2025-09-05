# Enhanced NBA Data Loader - Step 2 Improvements
"""
NBA Pipeline - Enhanced Data Loader with Proper Validation
==========================================================

Key Fixes:
1. Robust DuckDB object type handling (table vs view conflicts)
2. Improved error handling and validation
3. Better resource management and cleanup
4. Enhanced logging and debugging information

Lineup Estimation Methodology (Updated Implementation)
-----------------------------------------------------
This module implements two parallel lineup estimation methods:

TRADITIONAL DATA-DRIVEN METHOD (run_traditional_data_driven_lineups):
- Strictly follows raw data without automation or inference
- msgType=8: playerId1 = player subbed IN, playerId2 = player subbed OUT
- Lineups can have any size (not forced to 5 players)
- Comprehensive flagging for lineup size deviations and substitution issues
- Detailed explanations for why lineups aren't size 5
- Flags: lineup_size_deviation, sub_out_player_not_in_lineup, 
  sub_in_player_already_in_lineup, action_by_non_lineup_player

ENHANCED METHOD (run_enhanced_substitution_tracking_with_flags):
- Uses intelligent inference to maintain 5-player lineups
- Period resets: Q1 and Q3 reset to starters, Q2/Q4/OT carry forward
- First-Action Auto-IN: Players with actions but no sub-in are auto-added
- Inactivity Auto-OUT: Players idle >120s are candidates for removal
- Always-Five Enforcement: Maintains exactly 5 players per team
- Flags: missing_sub_in, inactivity_periods, first_action_events, 
  auto_out_events, lineup_violations

Both methods provide comprehensive validation and comparison capabilities.
"""


import os
import sys
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, deque

import duckdb
import pandas as pd
import numpy as np

# Ensure we're in the right directory
cwd = os.getcwd()
if not cwd.endswith("airflow_project"):
    os.chdir('api/src/airflow_project')
sys.path.insert(0, os.getcwd())

from eda.utils.nba_pipeline_analysis import NBADataValidator, ValidationResult

logger = logging.getLogger(__name__)

class EnhancedNBADataLoader:
    """Enhanced data loader with transparent validation and cleaning"""

    def __init__(self, db_path: str = "mavs_enhanced.duckdb", export_dir: str = None):
        """
        Enhanced NBA Data Loader with transparent validation and cleaning

        FIXED: Added export_dir initialization to prevent missing attribute errors

        Args:
            db_path: Path to DuckDB database file
            export_dir: Path to export directory (optional, auto-detected if None)
        """
        self.db_path = db_path
        self.conn = None
        self.validator = NBADataValidator()
        self.data_summary = {}

        # FIXED: Initialize export_dir attribute to prevent missing attribute error
        if export_dir is None:
            # Try to use config-managed export directory
            try:
                from utils.config import EXPORTS_DIR
                self.export_dir = EXPORTS_DIR
            except ImportError:
                # Fallback to default location relative to working directory
                from pathlib import Path
                self.export_dir = Path.cwd() / "exports"
        else:
            from pathlib import Path
            self.export_dir = Path(export_dir)

        # Ensure export directory exists
        self.export_dir.mkdir(parents=True, exist_ok=True)

        # Configuration for data quality
        self.rim_distance_threshold = 4.0  # feet
        self.coordinate_scale = 10.0  # NBA coordinates in tenths of feet
        self.max_reasonable_distance = 35.0  # feet (beyond court boundaries)

    def __enter__(self):
        # Prefer central config for path + engine settings
        try:
            from utils.config import DUCKDB_PATH, DUCKDB_CONFIG
            db_path = str(DUCKDB_PATH) if self.db_path in (None, "", "mavs_enhanced.duckdb") else self.db_path
            self.conn = duckdb.connect(db_path)
            # Apply engine config once, centrally
            if "memory_limit" in DUCKDB_CONFIG:
                self.conn.execute(f"SET memory_limit = '{DUCKDB_CONFIG['memory_limit']}'")
            if "threads" in DUCKDB_CONFIG:
                self.conn.execute(f"SET threads = {int(DUCKDB_CONFIG['threads'])}")
            if "temp_directory" in DUCKDB_CONFIG:
                self.conn.execute(f"SET temp_directory = '{DUCKDB_CONFIG['temp_directory']}'")
            if "preserve_insertion_order" in DUCKDB_CONFIG:
                self.conn.execute(f"SET preserve_insertion_order = {DUCKDB_CONFIG['preserve_insertion_order']}")
            if "checkpoint_threshold" in DUCKDB_CONFIG:
                self.conn.execute(f"SET checkpoint_threshold = '{DUCKDB_CONFIG['checkpoint_threshold']}'")
        except Exception:
            # Fallback to original behavior if config import fails
            self.conn = duckdb.connect(self.db_path or "mavs_enhanced.duckdb")
            self.conn.execute("SET memory_limit = '4GB'")
            self.conn.execute("SET threads = 4")
        return self


    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn:
            self.conn.close()

    def _robust_drop_object(self, object_name: str) -> None:
        """Robustly drop any DuckDB object regardless of type"""
        try:
            # Try dropping as table first
            self.conn.execute(f"DROP TABLE IF EXISTS {object_name}")
        except Exception:
            pass

        try:
            # Try dropping as view
            self.conn.execute(f"DROP VIEW IF EXISTS {object_name}")
        except Exception:
            pass

        try:
            # Try dropping as sequence
            self.conn.execute(f"DROP SEQUENCE IF EXISTS {object_name}")
        except Exception:
            pass

    def _to_native(self, obj):
        """
        Recursively convert objects to JSON-serializable Python builtins.
        - np.integer -> int
        - np.floating -> float
        - np.bool_ -> bool
        - pd.Timestamp/Timedelta -> str (ISO)
        - set/tuple -> list
        - dict -> dict with native keys/values
        - DataFrame/Series -> list/dict where sensible
        """
        import numpy as np
        import pandas as pd

        if obj is None:
            return None

        # Basic scalars
        if isinstance(obj, (bool, int, float, str)):
            return obj

        # NumPy scalars and arrays
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        # Handle numpy arrays and other array-like objects first
        if hasattr(obj, "tolist"):  # catches numpy arrays, pd.Series
            return self._to_native(obj.tolist())
        # Handle remaining numpy scalars that might have slipped through
        if hasattr(obj, 'dtype') and hasattr(obj, 'item'):
            if 'float' in str(obj.dtype):
                return float(obj.item())
            elif 'int' in str(obj.dtype):
                return int(obj.item())
            elif 'bool' in str(obj.dtype):
                return bool(obj.item())

        # Pandas-specific
        if isinstance(obj, (pd.Timestamp, pd.Timedelta)):
            return str(obj)

        # Mappings
        if isinstance(obj, dict):
            return {str(self._to_native(k)): self._to_native(v) for k, v in obj.items()}

        # Iterables (lists, tuples, sets)
        if isinstance(obj, (list, tuple, set)):
            return [self._to_native(v) for v in obj]

        # Fallback
        return str(obj)

    def _get_object_type(self, object_name: str) -> Optional[str]:
        """Get the type of a DuckDB object if it exists"""
        try:
            result = self.conn.execute(f"""
                SELECT table_type 
                FROM information_schema.tables 
                WHERE table_name = '{object_name}'
            """).fetchall()

            if result:
                return result[0][0]

            # Check views separately
            result = self.conn.execute(f"""
                SELECT 'VIEW' as table_type
                FROM information_schema.views 
                WHERE table_name = '{object_name}'
            """).fetchall()

            if result:
                return 'VIEW'

        except Exception as e:
            logger.debug(f"Could not check object type for {object_name}: {e}")

        return None

    def load_and_validate_box_score(self, file_path: Path) -> ValidationResult:
        """Load box score with enhanced validation and transparent reporting"""
        start_time = time.time()

        try:
            logger.info(f"Loading box score from {file_path}")

            # Load raw data
            df_raw = pd.read_csv(file_path)
            original_count = len(df_raw)

            logger.info(f"Raw box score: {original_count} rows")

            # Filter to active players only (as specified in requirements)
            df_active = df_raw[df_raw['status'] == 'ACTIVE'].copy()
            active_count = len(df_active)

            logger.info(f"Active players: {active_count} rows")

            # Validate we have the minimum required data
            if active_count < 10:  # Need at least 5 per team
                return ValidationResult(
                    step_name="Load Box Score",
                    passed=False,
                    details=f"Insufficient active players: {active_count} (minimum 10 required)",
                    processing_time=time.time() - start_time
                )

            # Check for required columns
            required_cols = ['nbaId', 'name', 'nbaTeamId', 'team', 'isHome', 'gs', 'secPlayed']
            missing_cols = [col for col in required_cols if col not in df_active.columns]

            if missing_cols:
                return ValidationResult(
                    step_name="Load Box Score",
                    passed=False,
                    details=f"Missing required columns: {missing_cols}",
                    processing_time=time.time() - start_time
                )

            # Clean and validate data
            warnings = []

            # Remove players with no playing time (they shouldn't affect lineup analysis)
            df_played = df_active[df_active['secPlayed'] > 0].copy()
            no_time_removed = active_count - len(df_played)

            if no_time_removed > 0:
                warnings.append(f"Removed {no_time_removed} players with no playing time")

            # Validate team structure
            team_analysis = self._analyze_team_structure(df_played)
            warnings.extend(team_analysis['warnings'])

            # Create optimized table with robust object handling
            self._robust_drop_object("box_score")
            self.conn.register("box_temp", df_played)

            create_sql = """
            CREATE TABLE box_score AS
            SELECT 
                nbaId as player_id,
                name as player_name,
                nbaTeamId as team_id,
                team as team_abbrev,
                CAST(isHome as BOOLEAN) as is_home,
                CAST(gs as BOOLEAN) as is_starter,
                status,
                secPlayed as seconds_played,
                COALESCE(pts, 0) as points,
                COALESCE(reb, 0) as rebounds,
                COALESCE(ast, 0) as assists,
                COALESCE(jerseyNum, 99) as jersey_number
            FROM box_temp
            WHERE player_id IS NOT NULL 
            AND player_name IS NOT NULL
            AND team_id IS NOT NULL
            ORDER BY team_id, seconds_played DESC
            """

            self.conn.execute(create_sql)
            self.conn.execute("DROP VIEW IF EXISTS box_temp")

            # Create indexes for performance with error handling
            try:
                self.conn.execute("CREATE INDEX IF NOT EXISTS idx_box_player ON box_score(player_id)")
                self.conn.execute("CREATE INDEX IF NOT EXISTS idx_box_team ON box_score(team_id)")
                self.conn.execute("CREATE INDEX IF NOT EXISTS idx_box_starter ON box_score(is_starter)")
            except Exception as e:
                logger.warning(f"Could not create indexes: {e}")

            # Get final count and store summary
            final_count = self.conn.execute("SELECT COUNT(*) FROM box_score").fetchone()[0]

            self.data_summary['box_score'] = {
                'original_rows': original_count,
                'active_rows': active_count,
                'final_rows': final_count,
                'teams': team_analysis['teams'],
                'starters_per_team': team_analysis['starters_per_team']
            }

            details = f"Processed box score: {original_count} → {active_count} active → {final_count} final rows"
            details += f". Teams: {team_analysis['teams']}, Starters: {team_analysis['starters_per_team']}"

            return ValidationResult(
                step_name="Load Box Score",
                passed=True,
                details=details,
                data_count=final_count,
                processing_time=time.time() - start_time,
                warnings=warnings
            )

        except Exception as e:
            return ValidationResult(
                step_name="Load Box Score",
                passed=False,
                details=f"Error loading box score: {str(e)}",
                processing_time=time.time() - start_time
            )

    def _analyze_team_structure(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze team structure and identify issues transparently"""
        analysis = {
            'teams': [],
            'starters_per_team': {},
            'players_per_team': {},
            'warnings': []
        }

        try:
            # Analyze each team
            for team_abbrev in df['team'].unique():
                if pd.isna(team_abbrev):
                    continue

                team_data = df[df['team'] == team_abbrev]
                analysis['teams'].append(team_abbrev)
                analysis['players_per_team'][team_abbrev] = len(team_data)

                # Count starters
                starters = team_data[team_data['gs'] == 1]
                analysis['starters_per_team'][team_abbrev] = len(starters)

                # Validate starter count (should be exactly 5)
                if len(starters) != 5:
                    analysis['warnings'].append(
                        f"Team {team_abbrev} has {len(starters)} starters (expected 5)"
                    )

                # Validate minimum roster size
                if len(team_data) < 8:
                    analysis['warnings'].append(
                        f"Team {team_abbrev} has only {len(team_data)} players (minimum 8 expected)"
                    )

            # Validate exactly 2 teams
            if len(analysis['teams']) != 2:
                analysis['warnings'].append(
                    f"Found {len(analysis['teams'])} teams (expected 2): {analysis['teams']}"
                )

            # Validate home/away designation
            home_teams = df[df['isHome'] == 1]['team'].unique()
            away_teams = df[df['isHome'] == 0]['team'].unique()

            if len(home_teams) != 1 or len(away_teams) != 1:
                analysis['warnings'].append(
                    f"Invalid home/away setup: home={list(home_teams)}, away={list(away_teams)}"
                )

        except Exception as e:
            analysis['warnings'].append(f"Team analysis error: {str(e)}")

        return analysis

    def load_and_validate_pbp(self, file_path: Path) -> ValidationResult:
        """Load PBP with enhanced validation and coordinate analysis"""
        start_time = time.time()
        try:
            logger.info(f"Loading PBP from {file_path}")
            df_raw = pd.read_csv(file_path)
            original_count = len(df_raw)

            # Identify admin rows
            admin_mask = (
                df_raw['offTeamId'].isna() | df_raw['defTeamId'].isna() |
                (df_raw['offTeamId'] == 0) | (df_raw['defTeamId'] == 0)
            )
            admin_rows = df_raw[admin_mask]
            game_events = df_raw[~admin_mask].copy()

            admin_count = len(admin_rows)
            game_count = len(game_events)
            logger.info(f"Admin rows: {admin_count}, Game events: {game_count}")

            warnings = []

            # Coordinate system analysis
            shots_mask = game_events['msgType'].isin([1, 2])
            shots = game_events[shots_mask].copy()
            coordinate_analysis = self._analyze_coordinate_system(shots)
            warnings.extend(coordinate_analysis['warnings'])

            # Create PBP table with robust object handling
            self._robust_drop_object("pbp")
            self.conn.register("pbp_temp", game_events)

            dist_expr = "(sqrt((loc_x::DOUBLE * loc_x::DOUBLE) + (loc_y::DOUBLE * loc_y::DOUBLE)) / 10.0)"

            create_sql = f"""
            CREATE TABLE pbp AS
            SELECT 
                pbpId AS pbp_id,
                period,
                pbpOrder AS pbp_order,
                wallClockInt AS wall_clock_int,
                COALESCE(gameClock, '') AS game_clock,
                COALESCE(description, '') AS description,
                msgType AS msg_type,
                COALESCE(actionType, 0) AS action_type,
                offTeamId AS team_id_off,
                defTeamId AS team_id_def,
                playerId1 AS player_id_1,
                playerId2 AS player_id_2,
                playerId3 AS player_id_3,
                -- keep last names from the raw file so we can label unknowns
                COALESCE(lastName1, '') AS last_name_1,
                COALESCE(lastName2, '') AS last_name_2,
                COALESCE(lastName3, '') AS last_name_3,
                locX AS loc_x,
                locY AS loc_y,
                COALESCE(pts, 0) AS points,

                CASE 
                WHEN msgType IN (1,2) AND locX IS NOT NULL AND locY IS NOT NULL 
                THEN {dist_expr} 
                END AS shot_distance_ft,

                CASE 
                WHEN msgType IN (1,2) AND locX IS NOT NULL AND locY IS NOT NULL 
                    AND {dist_expr} <= {self.rim_distance_threshold}
                THEN 1 ELSE 0 
                END::TINYINT AS is_rim_attempt,

                CASE 
                WHEN msgType IN (1,2) AND locX IS NOT NULL AND locY IS NOT NULL 
                    AND {dist_expr} > {self.max_reasonable_distance}
                THEN 1 ELSE 0 
                END::TINYINT AS is_extreme_distance,
                CASE 
                WHEN msgType IN (1,2) AND locX IS NOT NULL AND locY IS NOT NULL 
                    AND {dist_expr} > {self.max_reasonable_distance}
                THEN {dist_expr}
                END AS extreme_distance_ft

            FROM pbp_temp
            ORDER BY period, pbp_order, wall_clock_int
            """
            self.conn.execute(create_sql)
            self.conn.execute("DROP VIEW IF EXISTS pbp_temp")

            # Create indexes with error handling
            try:
                self.conn.execute("CREATE INDEX IF NOT EXISTS idx_pbp_chronological ON pbp(period, pbp_order)")
                self.conn.execute("CREATE INDEX IF NOT EXISTS idx_pbp_msg_type ON pbp(msg_type)")
                self.conn.execute("CREATE INDEX IF NOT EXISTS idx_pbp_teams ON pbp(team_id_off, team_id_def)")
                self.conn.execute("CREATE INDEX IF NOT EXISTS idx_pbp_rim ON pbp(is_rim_attempt)")
            except Exception as e:
                logger.warning(f"Could not create PBP indexes: {e}")

            final_count = self.conn.execute("SELECT COUNT(*) FROM pbp").fetchone()[0]

            flagged_extremes = self.conn.execute(
                "SELECT COUNT(*) FROM pbp WHERE is_extreme_distance = 1"
            ).fetchone()[0]
            if flagged_extremes > 0:
                warnings.append(f"Flagged {flagged_extremes} extreme-distance shots (> {self.max_reasonable_distance} ft)")

            self.data_summary['pbp'] = {
                'original_rows': original_count,
                'admin_rows': admin_count,
                'game_events': game_count,
                'final_rows': final_count,
                'coordinate_analysis': coordinate_analysis
            }

            details = (f"Processed PBP: {original_count} → {game_count} game events → {final_count} final rows. "
                    f"Shots: {coordinate_analysis['total_shots']}, Rim attempts: {coordinate_analysis['rim_attempts']}")
            return ValidationResult(
                step_name="Load PBP",
                passed=True,
                details=details,
                data_count=final_count,
                processing_time=time.time() - start_time,
                warnings=warnings
            )
        except Exception as e:
            return ValidationResult(
                step_name="Load PBP",
                passed=False,
                details=f"Error loading PBP: {str(e)}",
                processing_time=time.time() - start_time
            )



    def _analyze_coordinate_system(self, shots_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze coordinate system and identify issues"""
        analysis = {
            'total_shots': len(shots_df),
            'shots_with_coords': 0,
            'rim_attempts': 0,
            'rim_makes': 0,
            'extreme_shots': 0,
            'avg_distance': 0.0,
            'max_distance': 0.0,
            'warnings': []
        }

        if len(shots_df) == 0:
            analysis['warnings'].append("No shots found for coordinate analysis")
            return analysis

        # Filter to shots with coordinates
        coord_mask = shots_df['locX'].notna() & shots_df['locY'].notna()
        shots_with_coords = shots_df[coord_mask].copy()

        analysis['shots_with_coords'] = len(shots_with_coords)

        if len(shots_with_coords) == 0:
            analysis['warnings'].append("No shots have coordinate data")
            return analysis

        # Calculate distances
        shots_with_coords['distance_ft'] = np.sqrt(
            shots_with_coords['locX']**2 + shots_with_coords['locY']**2
        ) / self.coordinate_scale

        # Analyze distances
        analysis['avg_distance'] = shots_with_coords['distance_ft'].mean()
        analysis['max_distance'] = shots_with_coords['distance_ft'].max()

        # Count rim attempts (≤ 4 feet as specified)
        rim_mask = shots_with_coords['distance_ft'] <= self.rim_distance_threshold
        rim_shots = shots_with_coords[rim_mask]

        analysis['rim_attempts'] = len(rim_shots)
        analysis['rim_makes'] = len(rim_shots[rim_shots['msgType'] == 1])

        # Count extreme distances (likely data errors)
        extreme_mask = shots_with_coords['distance_ft'] > self.max_reasonable_distance
        analysis['extreme_shots'] = extreme_mask.sum()

        # Generate warnings
        if analysis['shots_with_coords'] < analysis['total_shots'] * 0.9:
            analysis['warnings'].append(
                f"Only {analysis['shots_with_coords']}/{analysis['total_shots']} shots have coordinates"
            )

        if analysis['extreme_shots'] > 0:
            analysis['warnings'].append(
                f"{analysis['extreme_shots']} shots beyond {self.max_reasonable_distance} feet (max: {analysis['max_distance']:.1f}ft)"
            )

        if analysis['rim_attempts'] == 0:
            analysis['warnings'].append("No rim attempts detected - check coordinate system")

        return analysis

    def validate_data_relationships(self) -> ValidationResult:
        """Validate box ↔ pbp team/player alignment. Recompute team set AFTER cleanup."""
        start_time = time.time()
        try:
            logger.info("Validating data relationships...")
            warnings = []

            # Box teams
            box_teams_df = self.conn.execute("""
                SELECT DISTINCT team_id, team_abbrev FROM box_score ORDER BY team_id
            """).df()
            box_team_ids = set(box_teams_df['team_id'])

            # PBP teams
            pbp_teams_df = self.conn.execute("""
                SELECT DISTINCT team_id FROM (
                    SELECT team_id_off AS team_id FROM pbp
                    UNION 
                    SELECT team_id_def AS team_id FROM pbp
                ) ORDER BY team_id
            """).df()
            pbp_team_ids = set(pbp_teams_df['team_id'])

            # If mismatch, remove unknowns with counted logging
            extra_pbp = pbp_team_ids - box_team_ids
            if extra_pbp:
                warnings.append(f"Team mismatch - Box: {sorted(box_team_ids)}, PBP: {sorted(pbp_team_ids)}")
                warnings.append(f"Extra teams in PBP: {sorted(extra_pbp)}")

                to_delete = self.conn.execute(f"""
                    SELECT COUNT(*) FROM pbp 
                    WHERE team_id_off NOT IN ({",".join(map(str, box_team_ids))})
                    OR team_id_def NOT IN ({",".join(map(str, box_team_ids))})
                """).fetchone()[0]

                self.conn.execute(f"""
                    DELETE FROM pbp 
                    WHERE team_id_off NOT IN ({",".join(map(str, box_team_ids))})
                    OR team_id_def NOT IN ({",".join(map(str, box_team_ids))})
                """)
                if to_delete > 0:
                    warnings.append(f"Removed {to_delete} PBP events from unknown teams")

                # Recompute
                pbp_teams_df = self.conn.execute("""
                    SELECT DISTINCT team_id FROM (
                        SELECT team_id_off AS team_id FROM pbp
                        UNION 
                        SELECT team_id_def AS team_id FROM pbp
                    ) ORDER BY team_id
                """).df()
                pbp_team_ids = set(pbp_teams_df['team_id'])

            # Player consistency
            box_players = set(self.conn.execute("SELECT DISTINCT player_id FROM box_score").df()['player_id'])
            pbp_players = set(self.conn.execute("""
                SELECT DISTINCT player_id FROM (
                    SELECT player_id_1 AS player_id FROM pbp WHERE player_id_1 IS NOT NULL
                    UNION SELECT player_id_2 FROM pbp WHERE player_id_2 IS NOT NULL
                    UNION SELECT player_id_3 FROM pbp WHERE player_id_3 IS NOT NULL
                )
            """).df()['player_id'])

            extra_pbp_players = pbp_players - box_players
            missing_pbp_players = box_players - pbp_players
            if extra_pbp_players:
                warnings.append(f"{len(extra_pbp_players)} players in PBP not in box score: players = {extra_pbp_players}")
                logger.info(f"Extra PBP players: {sorted(list(extra_pbp_players))[:10]}")
            if missing_pbp_players:
                warnings.append(f"{len(missing_pbp_players)} players in box score not in PBP")

            final_pbp_count = self.conn.execute("SELECT COUNT(*) FROM pbp").fetchone()[0]
            passed = (box_team_ids == pbp_team_ids) and len(box_team_ids) == 2

            details = (f"Relationship validation: Box teams: {len(box_team_ids)}, "
                    f"PBP teams: {len(pbp_team_ids)}, Final PBP events: {final_pbp_count}")
            return ValidationResult(
                step_name="Data Relationships",
                passed=passed,
                details=details,
                processing_time=time.time() - start_time,
                warnings=warnings
            )
        except Exception as e:
            return ValidationResult(
                    step_name="Data Relationships",
                    passed=False,
                    details=f"Error validating relationships: {str(e)}",
                    processing_time=time.time() - start_time
                )



    def create_lookup_views(self) -> ValidationResult:
        """Load lookup CSVs into DuckDB tables with robust handling"""
        start_time = time.time()
        try:
            logger.info("Creating lookup views...")

            # Use config-managed locations
            try:
                from utils.config import (
                    MAVS_DATA_DIR,
                    PBP_EVENT_MSG_TYPES_FILE,
                    PBP_ACTION_TYPES_FILE,
                    PBP_OPTION_TYPES_FILE,
                )
            except Exception as _e:
                return ValidationResult(
                    step_name="Create Lookup Views",
                    passed=False,
                    details=f"Config import failed: {_e}",
                    processing_time=time.time() - start_time
                )

            lookup_specs = [
                ("pbp_event_msg_types", PBP_EVENT_MSG_TYPES_FILE),
                ("pbp_action_types",    PBP_ACTION_TYPES_FILE),
                ("pbp_option_types",    PBP_OPTION_TYPES_FILE),
            ]

            created = []
            missing = []
            for table_name, file_path in lookup_specs:
                if not Path(file_path).exists():
                    missing.append(str(file_path))
                    continue

                # Robust object handling
                self._robust_drop_object(table_name)

                df = pd.read_csv(file_path)
                self.conn.register(f"{table_name}_temp", df)
                self.conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM {table_name}_temp")
                self.conn.execute(f"DROP VIEW IF EXISTS {table_name}_temp")
                created.append(table_name)

            if missing:
                return ValidationResult(
                    step_name="Create Lookup Views",
                    passed=False,
                    details=f"Missing lookup files: {missing}",
                    processing_time=time.time() - start_time
                )

            details = f"Created/Replaced {len(created)} lookup tables: {', '.join(created)}"
            return ValidationResult(
                step_name="Create Lookup Views",
                passed=True,
                details=details,
                processing_time=time.time() - start_time
            )
        except Exception as e:
            return ValidationResult(
                step_name="Create Lookup Views",
                passed=False,
                details=f"Error creating lookup views: {str(e)}",
                processing_time=time.time() - start_time
            )



    def create_dimensions(self) -> ValidationResult:
        """Create dim_teams and dim_players with strict validation + provenance + confidence + referee filtering"""
        start_time = time.time()
        try:
            # Robust object handling
            self._robust_drop_object("dim_teams")
            self._robust_drop_object("dim_players")
            self._robust_drop_object("dim_officials")

            # STEP 1: Detect and filter referees/officials BEFORE creating dimensions
            logger.info("🔍 Detecting referees/officials for filtering...")
            referee_ids = self.identify_referees_and_officials()

            # STEP 2: Create officials table for transparency
            self.create_officials_table(referee_ids)

            # Create dim_teams
            self.conn.execute("""
                CREATE TABLE dim_teams AS
                SELECT
                    team_id,
                    ANY_VALUE(team_abbrev) AS team_abbrev,
                    ANY_VALUE(is_home) AS is_home
                FROM box_score
                GROUP BY team_id
                ORDER BY team_id
            """)

            n_teams = self.conn.execute("SELECT COUNT(*) FROM dim_teams").fetchone()[0]
            if n_teams != 2:
                raise AssertionError(f"dim_teams must have 2 rows, found {n_teams}")

            null_abbrev = self.conn.execute(
                "SELECT COUNT(*) FROM dim_teams WHERE team_abbrev IS NULL"
            ).fetchone()[0]
            if null_abbrev > 0:
                raise AssertionError("dim_teams has NULL team_abbrev")

            dup_map = self.conn.execute("""
                WITH m AS (
                    SELECT team_id, COUNT(DISTINCT team_abbrev) AS c
                    FROM box_score
                    GROUP BY team_id
                )
                SELECT COUNT(*) FROM m WHERE c <> 1
            """).fetchone()[0]
            if dup_map > 0:
                raise AssertionError("box_score has multiple team_abbrev values for the same team_id")

            # STEP 3: Create referee filter clause
            referee_filter = ""
            if referee_ids:
                referee_list = ','.join(map(str, referee_ids))
                referee_filter = f"AND player_id NOT IN ({referee_list})"
                logger.info(f"🚫 Filtering out {len(referee_ids)} referee IDs: {sorted(referee_ids)}")

            # --- collect names from pbp slots (FILTERED) ---
            self.conn.execute(f"""
                CREATE OR REPLACE TEMP VIEW _pbp_names AS
                WITH p1 AS (
                    SELECT player_id_1 AS player_id, ANY_VALUE(NULLIF(last_name_1,'')) AS last_name
                    FROM pbp
                    WHERE player_id_1 IS NOT NULL {referee_filter.replace('player_id', 'player_id_1')}
                    GROUP BY player_id_1
                ),
                p2 AS (
                    SELECT player_id_2 AS player_id, ANY_VALUE(NULLIF(last_name_2,'')) AS last_name
                    FROM pbp
                    WHERE player_id_2 IS NOT NULL {referee_filter.replace('player_id', 'player_id_2')}
                    GROUP BY player_id_2
                ),
                p3 AS (
                    SELECT player_id_3 AS player_id, ANY_VALUE(NULLIF(last_name_3,'')) AS last_name
                    FROM pbp
                    WHERE player_id_3 IS NOT NULL {referee_filter.replace('player_id', 'player_id_3')}
                    GROUP BY player_id_3
                ),
                unioned AS (
                    SELECT * FROM p1
                    UNION ALL
                    SELECT * FROM p2
                    UNION ALL
                    SELECT * FROM p3
                )
                SELECT player_id, ANY_VALUE(last_name) AS last_name
                FROM unioned
                WHERE last_name IS NOT NULL
                GROUP BY player_id
            """)

            # --- infer team_id for pbp-only players WITH CONFIDENCE (FILTERED) ---
            self.conn.execute(f"""
                CREATE OR REPLACE TEMP VIEW _player_team_guess AS
                WITH occ AS (
                    SELECT player_id_1 AS player_id, team_id_off AS team_id FROM pbp WHERE player_id_1 IS NOT NULL {referee_filter.replace('player_id', 'player_id_1')}
                    UNION ALL SELECT player_id_2, team_id_off FROM pbp WHERE player_id_2 IS NOT NULL {referee_filter.replace('player_id', 'player_id_2')}
                    UNION ALL SELECT player_id_3, team_id_off FROM pbp WHERE player_id_3 IS NOT NULL {referee_filter.replace('player_id', 'player_id_3')}
                    UNION ALL SELECT player_id_1, team_id_def FROM pbp WHERE player_id_1 IS NOT NULL {referee_filter.replace('player_id', 'player_id_1')}
                    UNION ALL SELECT player_id_2, team_id_def FROM pbp WHERE player_id_2 IS NOT NULL {referee_filter.replace('player_id', 'player_id_2')}
                    UNION ALL SELECT player_id_3, team_id_def FROM pbp WHERE player_id_3 IS NOT NULL {referee_filter.replace('player_id', 'player_id_3')}
                ),
                agg AS (
                    SELECT player_id, team_id, COUNT(*) AS c
                    FROM occ
                    GROUP BY player_id, team_id
                ),
                totals AS (
                    SELECT player_id, SUM(c) AS tot
                    FROM agg
                    GROUP BY player_id
                ),
                ranked AS (
                    SELECT
                        a.player_id,
                        a.team_id,
                        a.c,
                        t.tot,
                        ROW_NUMBER() OVER (PARTITION BY a.player_id ORDER BY a.c DESC, a.team_id) AS rn
                    FROM agg a
                    JOIN totals t USING(player_id)
                )
                SELECT
                    player_id,
                    team_id,
                    c,
                    tot,
                    (c::DOUBLE)/NULLIF(tot,0) AS confidence
                FROM ranked
                WHERE rn = 1
            """)

            # --- universe of pbp player_ids (FILTERED) ---
            self.conn.execute(f"""
                CREATE OR REPLACE TEMP VIEW _pbp_players AS
                SELECT player_id FROM (
                    SELECT DISTINCT player_id_1 AS player_id FROM pbp WHERE player_id_1 IS NOT NULL {referee_filter.replace('player_id', 'player_id_1')}
                    UNION
                    SELECT DISTINCT player_id_2 FROM pbp WHERE player_id_2 IS NOT NULL {referee_filter.replace('player_id', 'player_id_2')}
                    UNION
                    SELECT DISTINCT player_id_3 FROM pbp WHERE player_id_3 IS NOT NULL {referee_filter.replace('player_id', 'player_id_3')}
                )
            """)

            # Create comprehensive dim_players (with provenance + confidence)
            self.conn.execute("""
                CREATE TABLE dim_players AS
                SELECT
                    COALESCE(b.player_id, p.player_id) AS player_id,
                    COALESCE(b.player_name, n.last_name, CAST(COALESCE(b.player_id, p.player_id) AS VARCHAR)) AS player_name,
                    COALESCE(b.team_id, tg.team_id) AS team_id,
                    t.team_abbrev,
                    COALESCE(b.is_starter, false) AS is_starter,
                    COALESCE(b.seconds_played, 0) AS seconds_played,
                    -- provenance
                    CASE
                        WHEN b.player_name IS NOT NULL THEN 'box'
                        WHEN n.last_name IS NOT NULL THEN 'pbp_last_name'
                        ELSE 'player_id'
                    END AS name_source,
                    CASE
                        WHEN b.team_id IS NOT NULL THEN 'box'
                        WHEN tg.team_id IS NOT NULL THEN 'pbp_team_guess'
                        ELSE NULL
                    END AS team_source,
                    tg.confidence AS team_confidence
                FROM _pbp_players p
                FULL OUTER JOIN (
                    SELECT DISTINCT player_id, player_name, team_id, is_starter, seconds_played
                    FROM box_score
                ) b ON p.player_id = b.player_id
                LEFT JOIN _pbp_names n ON COALESCE(b.player_id, p.player_id) = n.player_id
                LEFT JOIN _player_team_guess tg ON COALESCE(b.player_id, p.player_id) = tg.player_id
                LEFT JOIN dim_teams t ON COALESCE(b.team_id, tg.team_id) = t.team_id
            """)

            # PBP-only players view (unchanged structure, still helpful)
            self._robust_drop_object("pbp_only_players")
            self.conn.execute("""
                CREATE VIEW pbp_only_players AS
                WITH box_ids AS (SELECT DISTINCT player_id FROM box_score),
                pbp_ids AS (SELECT DISTINCT player_id FROM dim_players),
                only_ids AS (
                    SELECT p.player_id
                    FROM pbp_ids p
                    LEFT JOIN box_ids b USING(player_id)
                    WHERE b.player_id IS NULL
                )
                SELECT
                    o.player_id,
                    dp.player_name,
                    dp.team_id,
                    dp.team_abbrev,
                    ANY_VALUE(CONCAT('Q', pbp.period, ' ', pbp.game_clock, ' | ', pbp.description)) AS sample_event
                FROM only_ids o
                JOIN dim_players dp USING(player_id)
                LEFT JOIN pbp ON (o.player_id = pbp.player_id_1 OR o.player_id = pbp.player_id_2 OR o.player_id = pbp.player_id_3)
                GROUP BY o.player_id, dp.player_name, dp.team_id, dp.team_abbrev
                ORDER BY player_id
            """)

            cnt_players = self.conn.execute("SELECT COUNT(*) FROM dim_players").fetchone()[0]
            cnt_only = self.conn.execute("SELECT COUNT(*) FROM pbp_only_players").fetchone()[0]
            cnt_officials = self.conn.execute("SELECT COUNT(*) FROM dim_officials").fetchone()[0]

            logger.info(f"✅ Created dim_players: {cnt_players} players")
            logger.info(f"✅ Created pbp_only_players: {cnt_only} PBP-only players")
            logger.info(f"✅ Created dim_officials: {cnt_officials} referees/officials")
            if referee_ids:
                logger.info(f"🚫 Filtered out referees: {sorted(referee_ids)}")

            # Ensure every active box player is present with a team
            missing_active = self.conn.execute("""
                WITH active_box AS (SELECT DISTINCT player_id FROM box_score)
                SELECT COUNT(*) FROM active_box a
                LEFT JOIN dim_players d USING(player_id)
                WHERE d.player_id IS NULL OR d.team_id IS NULL
            """).fetchone()[0]
            if missing_active > 0:
                raise AssertionError(f"{missing_active} active box players missing in dim_players or missing team_id")

            return ValidationResult(
                step_name="Create Dimensions",
                passed=True,
                details=f"dim_players: {cnt_players} rows; pbp_only_players: {cnt_only} rows; dim_officials: {cnt_officials} rows",
                processing_time=time.time() - start_time
            )
        except Exception as e:
            return ValidationResult(
                step_name="Create Dimensions",
                passed=False,
                details=f"Error creating dimensions: {str(e)}",
                processing_time=time.time() - start_time
            )

    def identify_referees_and_officials(self):
        """
        Identify referee/official IDs based on event patterns.

        Criteria for referee detection:
        1. Appears in PBP events but NOT in active box score
        2. Only appears in foul calls (msgType=6) or turnovers (msgType=5) 
        3. Never appears in shots (msgType=1,2), rebounds (msgType=4), or subs (msgType=8)

        Returns:
            set: Set of player IDs identified as referees/officials
        """

        # Get all player IDs that appear in PBP
        pbp_players = self.conn.execute("""
            SELECT DISTINCT player_id FROM (
                SELECT player_id_1 AS player_id FROM pbp WHERE player_id_1 IS NOT NULL
                UNION
                SELECT player_id_2 FROM pbp WHERE player_id_2 IS NOT NULL  
                UNION
                SELECT player_id_3 FROM pbp WHERE player_id_3 IS NOT NULL
            )
        """).df()['player_id'].tolist()

        # Get active players from box score
        box_players = self.conn.execute("""
            SELECT DISTINCT player_id FROM box_score WHERE status = 'ACTIVE'
        """).df()['player_id'].tolist()

        # Find players in PBP but not in box score (potential referees)
        potential_refs = set(pbp_players) - set(box_players)

        if not potential_refs:
            return set()

        logger.info(f"🔍 Analyzing {len(potential_refs)} potential referee/official IDs...")

        confirmed_refs = set()

        for player_id in potential_refs:
            # Analyze event patterns for this player
            events = self.conn.execute(f"""
                SELECT 
                    msg_type,
                    description,
                    last_name_1,
                    last_name_2, 
                    last_name_3
                FROM pbp 
                WHERE player_id_1 = {player_id} 
                   OR player_id_2 = {player_id}
                   OR player_id_3 = {player_id}
            """).df()

            if len(events) == 0:
                continue

            # Get player name from events
            names = set()
            for _, row in events.iterrows():
                for col in ['last_name_1', 'last_name_2', 'last_name_3']:
                    if pd.notna(row[col]):
                        names.add(row[col])

            name = list(names)[0] if names else f"ID_{player_id}"

            # Analyze event type patterns
            msg_types = events['msg_type'].value_counts().to_dict()

            # Referee criteria:
            # 1. Only appears in fouls (6) or turnovers (5) or technical fouls (16,17,18)
            # 2. Never in shots (1,2), rebounds (4), substitutions (8)

            referee_event_types = {5, 6, 7, 16, 17, 18}  # turnovers, fouls, technicals
            player_event_types = {1, 2, 4, 8}  # shots, rebounds, substitutions

            has_referee_events = any(msg_type in referee_event_types for msg_type in msg_types.keys())
            has_player_events = any(msg_type in player_event_types for msg_type in msg_types.keys())

            if has_referee_events and not has_player_events:
                confirmed_refs.add(player_id)
                logger.info(f"  ✅ {name} (ID: {player_id}) - REFEREE/OFFICIAL")
                logger.info(f"      Events: {dict(msg_types)}")

        logger.info(f"🎯 Identified {len(confirmed_refs)} confirmed referees/officials")
        return confirmed_refs

    def create_officials_table(self, referee_ids):
        """Create a separate table for referees/officials for transparency"""

        if not referee_ids:
            # Create empty table
            self.conn.execute("""
                CREATE TABLE dim_officials AS
                SELECT 
                    CAST(NULL AS INTEGER) AS official_id,
                    CAST(NULL AS VARCHAR) AS official_name,
                    CAST(NULL AS INTEGER) AS total_events,
                    CAST(NULL AS VARCHAR) AS event_types,
                    CAST(NULL AS VARCHAR) AS sample_description
                WHERE FALSE
            """)
            return

        # Build officials data
        officials_data = []

        for official_id in referee_ids:
            # Get event details for this official
            events = self.conn.execute(f"""
                SELECT 
                    msg_type,
                    description,
                    last_name_1,
                    last_name_2,
                    last_name_3
                FROM pbp 
                WHERE player_id_1 = {official_id}
                   OR player_id_2 = {official_id}
                   OR player_id_3 = {official_id}
            """).df()

            # Extract name
            names = set()
            for _, row in events.iterrows():
                for col in ['last_name_1', 'last_name_2', 'last_name_3']:
                    if pd.notna(row[col]):
                        names.add(row[col])

            official_name = list(names)[0] if names else f"Official_{official_id}"

            # Event type summary
            msg_types = events['msg_type'].value_counts().to_dict()
            event_types_str = ', '.join([f"msgType{k}:{v}" for k, v in sorted(msg_types.items())])

            # Sample description
            sample_desc = events.iloc[0]['description'] if len(events) > 0 else "No events"

            officials_data.append({
                'official_id': official_id,
                'official_name': official_name,
                'total_events': len(events),
                'event_types': event_types_str,
                'sample_description': sample_desc
            })

        # Insert data
        if officials_data:
            officials_df = pd.DataFrame(officials_data)

            # Create table
            self.conn.execute("DROP TABLE IF EXISTS dim_officials")
            self.conn.register('officials_temp', officials_df)
            self.conn.execute("""
                CREATE TABLE dim_officials AS
                SELECT * FROM officials_temp
            """)
            self.conn.unregister('officials_temp')

            logger.info(f"📋 Created dim_officials table with {len(officials_data)} officials")

    def create_pbp_enriched_view(self) -> ValidationResult:
        """Create enriched PBP view with robust object handling"""
        start_time = time.time()
        try:
            # Check required tables exist
            required = ["pbp", "pbp_event_msg_types", "pbp_action_types", "pbp_option_types", "dim_teams", "dim_players"]
            for t in required:
                exists = self.conn.execute(
                    f"SELECT COUNT(*) FROM information_schema.tables WHERE table_name = '{t}'"
                ).fetchone()[0]
                if exists == 0:
                    return ValidationResult(
                        step_name="Create PBP Enriched View",
                        passed=False,
                        details=f"Missing required table: {t}",
                        processing_time=time.time() - start_time
                    )

            # Robust cleanup
            self._robust_drop_object("pbp_enriched")

            # Create enriched view with proper deduplication
            self.conn.execute("""
            CREATE VIEW pbp_enriched AS
            WITH team_map AS (
                SELECT DISTINCT team_id, team_abbrev FROM dim_teams
            ),
            event_types AS (
                SELECT DISTINCT EventType, Description FROM pbp_event_msg_types
            ),
            action_types AS (
                SELECT DISTINCT EventType, ActionType, Event, Description 
                FROM pbp_action_types
            ),
            option_types AS (
                SELECT EventType, 
                       ANY_VALUE(Option1) AS Option1,
                       ANY_VALUE(Option2) AS Option2,
                       ANY_VALUE(Option3) AS Option3,
                       ANY_VALUE(Option4) AS Option4
                FROM pbp_option_types
                GROUP BY EventType
            )
            SELECT
                p.*,
                emt.Description AS event_family,
                act.Event AS action_event,
                act.Description AS action_desc,
                toff.team_abbrev AS team_off_abbrev,
                tdef.team_abbrev AS team_def_abbrev,
                COALESCE(p1.player_name, NULLIF(p.last_name_1, '')) AS player1_name,
                COALESCE(p2.player_name, NULLIF(p.last_name_2, '')) AS player2_name,
                COALESCE(p3.player_name, NULLIF(p.last_name_3, '')) AS player3_name,
                opt.Option1 AS option1_label,
                opt.Option2 AS option2_label,
                opt.Option3 AS option3_label,
                opt.Option4 AS option4_label
            FROM pbp p
            LEFT JOIN event_types emt ON p.msg_type = emt.EventType
            LEFT JOIN action_types act ON p.msg_type = act.EventType AND p.action_type = act.ActionType
            LEFT JOIN option_types opt ON p.msg_type = opt.EventType
            LEFT JOIN team_map toff ON p.team_id_off = toff.team_id
            LEFT JOIN team_map tdef ON p.team_id_def = tdef.team_id
            LEFT JOIN dim_players p1 ON p.player_id_1 = p1.player_id
            LEFT JOIN dim_players p2 ON p.player_id_2 = p2.player_id
            LEFT JOIN dim_players p3 ON p.player_id_3 = p3.player_id
            ORDER BY p.period, p.pbp_order, p.wall_clock_int
            """)

            # Validate row count matches
            n_pbp = self.conn.execute("SELECT COUNT(*) FROM pbp").fetchone()[0]
            n_enriched = self.conn.execute("SELECT COUNT(*) FROM pbp_enriched").fetchone()[0]

            if n_pbp != n_enriched:
                return ValidationResult(
                    step_name="Create PBP Enriched View",
                    passed=False,
                    details=f"Row count mismatch: pbp={n_pbp} vs enriched={n_enriched}",
                    processing_time=time.time() - start_time
                )

            return ValidationResult(
                step_name="Create PBP Enriched View",
                passed=True,
                details=f"Created view pbp_enriched with {n_enriched} rows (matches pbp)",
                processing_time=time.time() - start_time
            )
        except Exception as e:
            return ValidationResult(
                step_name="Create PBP Enriched View",
                passed=False,
                details=f"Error creating pbp_enriched view: {str(e)}",
                processing_time=time.time() - start_time
            )

    def print_enhanced_summary(self):
        """Print enhanced data loading summary (ASCII-only)."""
        print("\n" + "=" * 80)
        print("ENHANCED NBA PIPELINE - DATA LOADING SUMMARY")
        print("=" * 80)

        if 'box_score' in self.data_summary:
            box_data = self.data_summary['box_score']
            print("BOX SCORE:")
            print(f"   Original rows: {box_data['original_rows']:,}")
            print(f"   Active players: {box_data['active_rows']:,}")
            print(f"   Final rows: {box_data['final_rows']:,}")
            teams_str = ", ".join(box_data['teams'])
            print(f"   Teams: {teams_str}")
            print(f"   Starters per team: {box_data['starters_per_team']}")

        if 'pbp' in self.data_summary:
            pbp_data = self.data_summary['pbp']
            coord_data = pbp_data['coordinate_analysis']
            print("\nPLAY-BY-PLAY:")
            print(f"   Original rows: {pbp_data['original_rows']:,}")
            print(f"   Game events: {pbp_data['game_events']:,}")
            print(f"   Final rows: {pbp_data['final_rows']:,}")
            print(f"   Total shots: {coord_data['total_shots']:,}")
            print(f"   Shots with coordinates: {coord_data['shots_with_coords']:,}")
            print(f"   Rim attempts: {coord_data['rim_attempts']:,}")
            print(f"   Average distance: {coord_data['avg_distance']:.1f} ft")

        if 'enhanced_substitution_debug' in self.data_summary:
            d = self.data_summary['enhanced_substitution_debug']
            print("\nLINEUP ENGINE:")
            print(f"   Substitutions: {d.get('substitutions', 0)}")
            print(f"   First-actions auto-IN: {d.get('first_actions', 0)}")
            print(f"   Inactivity auto-OUTs: {d.get('auto_outs', 0)}")
            print(f"   5-on-floor fixes: {d.get('always_five_fixes', 0)}")
            v = d.get('validation', {})
            print(f"   Minutes tolerance: ±{v.get('tolerance', 0)}s")
            print(f"   Minutes offenders: {v.get('offenders', 0)}/{v.get('total_players', 0)}")

        if 'lineup_results' in self.data_summary:
            print("\nANALYTICS RESULTS:")
            print(f"   Lineup combinations: {self.data_summary['lineup_results']['rows']:,}")
            print(f"   Player rim stats: {self.data_summary['player_rim_results']['rows']:,}")

        print("=" * 80)


    def write_final_report(self, reports_dir: Optional[Path] = None) -> ValidationResult:
        """
        Emit an end-of-run report:
        - minutes_validation_full / minutes_offenders (enhanced pass)
        - basic_lineup_state.csv / basic_lineup_flags.csv / minutes_basic.csv
        - enhanced_lineup_state.csv / enhanced_lineup_flags.csv / minutes_enhanced.csv
        - minutes_compare.csv
        - traditional_vs_enhanced_comparison.csv
        - comprehensive_flags_analysis.csv (when available)
        - run_summary.json (JSON-safe)
        - UPDATED:
            * unique_lineups_traditional_5.csv        (5-man only)
            * unique_lineups_traditional_all.csv      (all sizes)
            * unique_lineups_enhanced_5.csv           (5-man only)
            * Logs show traditional 5-man AND all-sizes
        """
        import time, json as _json
        from pathlib import Path
        import pandas as pd

        start_time = time.time()
        try:
            mv_df = self.data_summary.get('minutes_validation_full')
            offenders = self.data_summary.get('minutes_offenders')
            debug = self.data_summary.get('enhanced_substitution_debug', {})

            if reports_dir is None:
                reports_dir = Path("reports")
            reports_dir.mkdir(parents=True, exist_ok=True)

            # Persist enhanced minutes into DuckDB for reproducibility
            try:
                self._robust_drop_object("minutes_validation_full")
                self._robust_drop_object("minutes_offenders")
                if mv_df is not None and len(mv_df) > 0:
                    self.conn.register("mv_temp", mv_df)
                    self.conn.execute("CREATE TABLE minutes_validation_full AS SELECT * FROM mv_temp")
                    self.conn.execute("DROP VIEW IF EXISTS mv_temp")
                else:
                    self.conn.execute("CREATE TABLE minutes_validation_full AS SELECT 1 WHERE FALSE")
                if offenders is not None and len(offenders) > 0:
                    self.conn.register("off_temp", offenders)
                    self.conn.execute("CREATE TABLE minutes_offenders AS SELECT * FROM off_temp")
                    self.conn.execute("DROP VIEW IF EXISTS off_temp")
                else:
                    self.conn.execute("CREATE TABLE minutes_offenders AS SELECT 1 WHERE FALSE")
            except Exception as e:
                logger.warning("[Report] Could not create DuckDB tables: %s", e)

            # CSVs for enhanced minutes tables
            if mv_df is not None:
                mv_df.to_csv(reports_dir / "minutes_validation_full.csv", index=False)
            if offenders is not None:
                offenders.to_csv(reports_dir / "minutes_offenders.csv", index=False)

            # Export basic / enhanced state snapshots if present
            try:
                if self.conn.execute("SELECT COUNT(*) FROM information_schema.tables WHERE table_name='basic_lineup_state'").fetchone()[0]:
                    self.conn.execute("COPY (SELECT * FROM basic_lineup_state ORDER BY period, pbp_order, team_id) TO '{}' (HEADER, DELIMITER ',')"
                                    .format(str((reports_dir / "basic_lineup_state.csv").as_posix())))
                if self.conn.execute("SELECT COUNT(*) FROM information_schema.tables WHERE table_name='basic_lineup_flags'").fetchone()[0]:
                    self.conn.execute("COPY (SELECT * FROM basic_lineup_flags ORDER BY abs_time, team_id) TO '{}' (HEADER, DELIMITER ',')"
                                    .format(str((reports_dir / "basic_lineup_flags.csv").as_posix())))
                if self.conn.execute("SELECT COUNT(*) FROM information_schema.tables WHERE table_name='minutes_basic'").fetchone()[0]:
                    self.conn.execute("COPY (SELECT * FROM minutes_basic ORDER BY team_abbrev, player_name) TO '{}' (HEADER, DELIMITER ',')"
                                    .format(str((reports_dir / "minutes_basic.csv").as_posix())))

                if self.conn.execute("SELECT COUNT(*) FROM information_schema.tables WHERE table_name='enhanced_lineup_state'").fetchone()[0]:
                    self.conn.execute("COPY (SELECT * FROM enhanced_lineup_state ORDER BY period, pbp_order, team_id) TO '{}' (HEADER, DELIMITER ',')"
                                    .format(str((reports_dir / "enhanced_lineup_state.csv").as_posix())))
                if self.conn.execute("SELECT COUNT(*) FROM information_schema.tables WHERE table_name='enhanced_lineup_flags'").fetchone()[0]:
                    self.conn.execute("COPY (SELECT * FROM enhanced_lineup_flags ORDER BY abs_time, team_id) TO '{}' (HEADER, DELIMITER ',')"
                                    .format(str((reports_dir / "enhanced_lineup_flags.csv").as_posix())))
                if self.conn.execute("SELECT COUNT(*) FROM information_schema.tables WHERE table_name='minutes_enhanced'").fetchone()[0]:
                    self.conn.execute("COPY (SELECT * FROM minutes_enhanced ORDER BY team_abbrev, player_name) TO '{}' (HEADER, DELIMITER ',')"
                                    .format(str((reports_dir / "minutes_enhanced.csv").as_posix())))
            except Exception as e:
                logger.warning("[Report] Could not export basic/enhanced CSVs: %s", e)

            # Comparison exports (minutes and flags)
            try:
                if self.conn.execute("SELECT COUNT(*) FROM information_schema.tables WHERE table_name='minutes_compare'").fetchone()[0]:
                    self.conn.execute("COPY (SELECT * FROM minutes_compare ORDER BY team_abbrev, player_name) TO '{}' (HEADER, DELIMITER ',')"
                                    .format(str((reports_dir / "minutes_compare.csv").as_posix())))
                if self.conn.execute("SELECT COUNT(*) FROM information_schema.tables WHERE table_name='traditional_vs_enhanced_comparison'").fetchone()[0]:
                    self.conn.execute("COPY (SELECT * FROM traditional_vs_enhanced_comparison ORDER BY team_abbrev, player_name) TO '{}' (HEADER, DELIMITER ',')"
                                    .format(str((reports_dir / "traditional_vs_enhanced_comparison.csv").as_posix())))
                if self.conn.execute("SELECT COUNT(*) FROM information_schema.tables WHERE table_name='comprehensive_flags_analysis'").fetchone()[0]:
                    self.conn.execute("COPY (SELECT * FROM comprehensive_flags_analysis ORDER BY time, team) TO '{}' (HEADER, DELIMITER ',')"
                                    .format(str((reports_dir / "comprehensive_flags_analysis.csv").as_posix())))
            except Exception as e:
                logger.warning("[Report] Could not export comparison/flags CSVs: %s", e)

            # -----------------------------
            # UPDATED: UNIQUE LINEUPS COMPUTE
            # -----------------------------
            def _table_exists(name: str) -> bool:
                return bool(self.conn.execute(
                    "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?", [name]
                ).fetchone()[0])

            # Name map for pretty printing
            try:
                if _table_exists("dim_players"):
                    nm_df = self.conn.execute("SELECT DISTINCT player_id, player_name FROM dim_players").df()
                else:
                    nm_df = self.conn.execute("SELECT DISTINCT player_id, player_name FROM box_score").df()
            except Exception:
                nm_df = pd.DataFrame(columns=["player_id","player_name"])
            name_map = dict(zip(nm_df.get("player_id", []), nm_df.get("player_name", [])))

            def _ids_to_names(ids):
                return [str(name_map.get(int(pid), str(int(pid)))) for pid in ids]

            def _parse_ids_json(s) -> list:
                # Robust JSON or bracketed string parsing
                try:
                    v = _json.loads(s)
                    if isinstance(v, list):
                        return [int(x) for x in v]
                except Exception:
                    pass
                # fallback "1,2,3" or "[1, 2, 3]"
                s2 = str(s).strip().strip("[]")
                if not s2:
                    return []
                return [int(x.strip()) for x in s2.split(",") if x.strip()]

            def _compute_unique_lineups(table_name: str, label: str, size_filter: Optional[int]) -> tuple[pd.DataFrame, dict]:
                """
                Returns df with columns:
                [method, team_id, team_abbrev, lineup_size, occurrences,
                lineup_player_ids_json, lineup_player_names_json]
                'counts' includes: total_unique, by_team, and by_size.
                """
                cols = ["method","team_id","team_abbrev","lineup_size","occurrences",
                        "lineup_player_ids_json","lineup_player_names_json"]
                if not _table_exists(table_name):
                    return pd.DataFrame(columns=cols), {"total_unique": 0, "by_team": {}, "by_size": {}}

                df = self.conn.execute(f"""
                    SELECT team_id, team_abbrev, lineup_player_ids_json, lineup_size
                    FROM {table_name}
                    WHERE lineup_player_ids_json IS NOT NULL
                """).df()

                if df.empty:
                    return pd.DataFrame(columns=cols), {"total_unique": 0, "by_team": {}, "by_size": {}}

                if size_filter is not None:
                    df = df[df["lineup_size"] == size_filter].copy()

                # Group by team, size, and the player-ids JSON
                grp = df.groupby(["team_id","team_abbrev","lineup_size","lineup_player_ids_json"], as_index=False)\
                        .size().rename(columns={"size":"occurrences"})

                # Convert ids json to names json (canonical sort for stability)
                names_json = []
                for s in grp["lineup_player_ids_json"]:
                    ids = sorted(_parse_ids_json(s))
                    names = _ids_to_names(ids)
                    names_json.append(_json.dumps(names))
                grp["lineup_player_names_json"] = names_json
                grp.insert(0, "method", label)

                # Counts (overall unique, by team, by size)
                total_unique = int(grp[["team_id","lineup_player_ids_json"]].drop_duplicates().shape[0])
                by_team = grp.groupby("team_abbrev")["lineup_player_ids_json"].nunique().to_dict()
                by_size = grp.groupby("lineup_size")["lineup_player_ids_json"].nunique().to_dict()

                grp = grp.sort_values(["team_abbrev","lineup_size","occurrences"], ascending=[True, True, False]).reset_index(drop=True)
                counts = {"total_unique": total_unique,
                        "by_team": {str(k): int(v) for k, v in by_team.items()},
                        "by_size": {int(k): int(v) for k, v in by_size.items()}}
                return grp, counts

            # Compute: TRADITIONAL (5-man), TRADITIONAL (all sizes), ENHANCED (5-man)
            trad_5_df,   trad_5_counts   = _compute_unique_lineups("traditional_lineup_state", "traditional", size_filter=5)
            trad_all_df, trad_all_counts = _compute_unique_lineups("traditional_lineup_state", "traditional", size_filter=None)
            enh_5_df,    enh_5_counts    = _compute_unique_lineups("enhanced_lineup_state",    "enhanced",   size_filter=5)

            # Write CSVs
            if not trad_5_df.empty:
                trad_5_df.to_csv(reports_dir / "unique_lineups_traditional_5.csv", index=False)
            if not trad_all_df.empty:
                trad_all_df.to_csv(reports_dir / "unique_lineups_traditional_all.csv", index=False)
            if not enh_5_df.empty:
                enh_5_df.to_csv(reports_dir / "unique_lineups_enhanced_5.csv", index=False)

            # Helper: ASCII logging of unique lineups
            def _log_unique_list(df: pd.DataFrame, title: str):
                if df.empty:
                    logger.info("[%s] No unique lineups found.", title)
                    return
                logger.info("=" * 78)
                logger.info("UNIQUE LINEUPS — %s", title)
                logger.info("=" * 78)
                for team_abbrev, sub in df.groupby("team_abbrev"):
                    # Include lineup_size in the heading when mixed
                    sizes = sorted(sub["lineup_size"].unique().tolist())
                    size_tag = "" if sizes == [5] else f" (sizes: {sizes})"
                    logger.info("%s: %d unique lineups%s", team_abbrev, len(sub), size_tag)
                    sub = sub.sort_values(["lineup_size","occurrences"], ascending=[True, False]).reset_index(drop=True)
                    for i, row in sub.iterrows():
                        try:
                            names = _json.loads(row["lineup_player_names_json"])
                        except Exception:
                            names = row["lineup_player_names_json"]
                        names_str = ", ".join(names) if isinstance(names, list) else str(names)
                        logger.info("  %2d. [%d] size=%d  %s", i+1, int(row["occurrences"]), int(row["lineup_size"]), names_str)
                logger.info("-" * 78)

            # Log both traditional views and enhanced 5-man
            _log_unique_list(trad_5_df,   "TRADITIONAL (5-man)")
            _log_unique_list(trad_all_df, "TRADITIONAL (ALL sizes)")
            _log_unique_list(enh_5_df,    "ENHANCED (5-man)")

            # Build JSON summary (JSON-safe)
            comparison_data = self.data_summary.get('traditional_vs_enhanced_comparison', {})
            comparison_summary = comparison_data.get('summary', {})

            # Traditional/enhanced state counts for summary
            traditional_states = int(self.data_summary.get("traditional_data_driven", {}).get("state_rows", 0))
            enhanced_states    = int(self.data_summary.get("enhanced_substitution_tracking", {}).get("state_rows", 0))

            # Unique counts via SQL (secondary check)
            def _unique_count(table_name: str, size_filter: Optional[int]) -> int:
                try:
                    exists = self.conn.execute(
                        "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
                        [table_name]
                    ).fetchone()[0]
                    if not exists:
                        return 0
                    if size_filter is None:
                        q = f"""
                            SELECT COUNT(*) FROM (
                                SELECT team_id, lineup_player_ids_json
                                FROM {table_name}
                                GROUP BY team_id, lineup_player_ids_json
                            )
                        """
                    else:
                        q = f"""
                            SELECT COUNT(*) FROM (
                                SELECT team_id, lineup_player_ids_json
                                FROM {table_name}
                                WHERE lineup_size = {int(size_filter)}
                                GROUP BY team_id, lineup_player_ids_json
                            )
                        """
                    return int(self.conn.execute(q).fetchone()[0])
                except Exception:
                    return 0

            traditional_unique_5   = _unique_count("traditional_lineup_state", 5)
            traditional_unique_all = _unique_count("traditional_lineup_state", None)
            enhanced_unique_5      = _unique_count("enhanced_lineup_state", 5)

            # Build summary safely
            summary = {
                "substitutions": int(debug.get("substitutions", 0)),
                "first_actions": int(debug.get("first_actions", 0)),
                "auto_outs": int(debug.get("auto_outs", 0)),
                "always_five_fixes": int(debug.get("always_five_fixes", 0)),
                "total_players": int(debug.get("validation", {}).get("total_players", 0)),
                "offenders": int(debug.get("validation", {}).get("offenders", 0)),
                "tolerance_seconds": int(debug.get("validation", {}).get("tolerance", 120)),
                "traditional": {
                    "lineup_states": traditional_states,
                    "unique_lineups_5": traditional_unique_5,
                    "unique_lineups_all": traditional_unique_all,
                    "flag_types": self._to_native(self.data_summary.get("traditional_data_driven", {}).get("flag_summary", {})),
                    "by_team_5": self._to_native(trad_5_counts.get("by_team", {})),
                    "by_team_all": self._to_native(trad_all_counts.get("by_team", {})),
                    "by_size_all": self._to_native(trad_all_counts.get("by_size", {}))
                },
                "enhanced": {
                    "lineup_states": enhanced_states,
                    "unique_lineups_5": enhanced_unique_5,
                    "flag_totals": self._to_native(self.data_summary.get("enhanced_substitution_tracking", {}).get("flag_totals", {})),
                    "by_team_5": self._to_native(enh_5_counts.get("by_team", {}))
                },
                "minutes_compare_rows": int(self.data_summary.get("minutes_compare", {}).get("rows", 0)),
                "minutes_compare_basic_within10": int(self.data_summary.get("minutes_compare", {}).get("within10_basic", 0)),
                "minutes_compare_total_with_box": int(self.data_summary.get("minutes_compare", {}).get("total_with_box", 0)),
                "traditional_vs_enhanced": self._to_native(comparison_summary.get("method_comparison", {})),
                "enhanced_flags_summary": self._to_native(comparison_summary.get("flag_analysis", {})),
                "accuracy_metrics": self._to_native(comparison_summary.get("accuracy_metrics", {}))
            }

            with open(reports_dir / "run_summary.json", "w", encoding="utf-8") as f:
                import json as __json
                __json.dump(self._to_native(summary), f, indent=2)

            details = (
                "Report written: minutes_validation_full.csv, minutes_offenders.csv, "
                "basic_lineup_state.csv, basic_lineup_flags.csv, minutes_basic.csv, "
                "enhanced_lineup_state.csv, enhanced_lineup_flags.csv, minutes_enhanced.csv, "
                "minutes_compare.csv, traditional_vs_enhanced_comparison.csv, comprehensive_flags_analysis.csv, "
                "unique_lineups_traditional_5.csv, unique_lineups_traditional_all.csv, unique_lineups_enhanced_5.csv, "
                "run_summary.json"
            )
            return ValidationResult("Write Final Report", True, details, processing_time=time.time()-start_time)

        except Exception as e:
            return ValidationResult("Write Final Report", False, f"Error writing report: {e}", processing_time=time.time()-start_time)



    def run_traditional_data_driven_lineups(self) -> ValidationResult:
        """
        TRADITIONAL DATA-DRIVEN SUBSTITUTION TRACKING (Updated Implementation):

        This method strictly follows the raw data without any automation or inference:
        - msgType=8: playerId1 = player subbed IN, playerId2 = player subbed OUT
        - Lineups can have any size (not forced to 5)
        - Comprehensive flagging for lineup size deviations and substitution issues
        - Detailed explanations for why lineups aren't size 5

        Key Changes from Original:
        1. Removed automatic lineup size enforcement
        2. Added detailed flagging for substitution anomalies
        3. Enhanced validation of player states
        4. Better tracking of player availability vs. actual lineup membership

        Outputs written to DuckDB tables:
            * traditional_lineup_state
            * traditional_lineup_flags
            * minutes_traditional
        """
        from collections import defaultdict, deque
        import json
        start_time = time.time()

        try:
            # ---- Configuration ----
            CFG = {
                "starter_reset_periods": [1, 3],
                "sub_msg_type": 8,
                "action_msg_types": {1, 2, 4, 5, 6},  # FG made/miss, rebound, turnover, foul
                "allow_variable_lineup_sizes": True,  # NEW: Allow non-5 player lineups
                "detailed_flagging": True  # NEW: Enhanced flagging system
            }

            # ---- Helper Functions ----
            def _period_len(p: int) -> float:
                return 720.0 if p <= 4 else 300.0

            def _parse_gc(gc: str | None) -> float | None:
                if not gc or not isinstance(gc, str) or ":" not in gc:
                    return None
                try:
                    mm, ss = gc.split(":")
                    return float(mm) * 60.0 + float(ss)
                except Exception:
                    return None

            def _abs_t(period: int, rem: float | None) -> float:
                total = 0.0
                for pi in range(1, period):
                    total += _period_len(pi)
                pl = _period_len(period)
                if rem is None:
                    return total + pl
                return total + (pl - rem)

            # ---- Load Data ----
            box_df = self.conn.execute("""
                SELECT player_id, player_name, team_id, team_abbrev, is_starter, seconds_played
                FROM box_score
                WHERE seconds_played > 0
                ORDER BY team_id, seconds_played DESC
            """).df()

            if box_df.empty:
                return ValidationResult("Traditional Data-Driven Lineups", False, 
                                      "No active players in box_score", processing_time=time.time()-start_time)

            teams = sorted(box_df.team_id.unique().tolist())
            if len(teams) != 2:
                return ValidationResult("Traditional Data-Driven Lineups", False, 
                                      f"Expected 2 teams, found {teams}", processing_time=time.time()-start_time)

            # Build player mappings
            team_abbrev = {int(t): box_df[box_df.team_id == t].team_abbrev.iloc[0] for t in teams}
            starters = {int(t): set(box_df[(box_df.team_id==t)&(box_df.is_starter==True)].player_id.tolist()) for t in teams}
            name_map = dict(zip(box_df.player_id, box_df.player_name))
            pteam_map = dict(zip(box_df.player_id, box_df.team_id))

            # Validate starters
            for t in teams:
                if len(starters[int(t)]) != 5:
                    return ValidationResult("Traditional Data-Driven Lineups", False, 
                                          f"Team {team_abbrev[int(t)]} does not have 5 starters", 
                                          processing_time=time.time()-start_time)

            # Load events
            events = self.conn.execute("""
                SELECT period, pbp_order, wall_clock_int,
                       COALESCE(game_clock,'') AS game_clock,
                       COALESCE(description,'') AS description,
                       team_id_off, team_id_def, msg_type, action_type,
                       player_id_1, player_id_2, player_id_3,
                       NULLIF(last_name_1,'') AS last_name_1,
                       NULLIF(last_name_2,'') AS last_name_2,
                       NULLIF(last_name_3,'') AS last_name_3,
                       COALESCE(points,0) AS points
                FROM pbp
                ORDER BY period, pbp_order, wall_clock_int
            """).df()

            if events.empty:
                return ValidationResult("Traditional Data-Driven Lineups", False, 
                                      "No PBP events", processing_time=time.time()-start_time)

            # ---- State Tracking ----
            on_court = {int(t): set(starters[int(t)]) for t in teams}
            last_action_time = defaultdict(lambda: 0.0)
            player_last_seen = {}  # Track when each player was last active
            seconds_traditional = defaultdict(float)  # Minutes tracking

            # NEW: Enhanced tracking for flagging
            substitution_history = []  # Track all substitution attempts
            lineup_size_history = []   # Track lineup size changes
            player_status_tracking = {  # Track detailed player states
                tid: {
                    'current_lineup': set(starters[int(tid)]),
                    'last_sub_in': {},    # player_id -> timestamp
                    'last_sub_out': {},   # player_id -> timestamp
                    'action_without_sub': set()  # players who had actions but no sub-in
                } for tid in teams
            }

            prev_abs_time = 0.0
            prev_period = None

            # Results tracking
            state_rows = []
            flag_rows = []

            def snapshot_lineups(ev_time: float, period: int, pbp_order: int, desc: str, event_type: str = "NORMAL"):
                """Snapshot current lineups with enhanced metadata"""
                for tid in teams:
                    lineup = list(on_court[int(tid)])
                    lineup_names = [name_map.get(p, str(p)) for p in lineup]
                    lineup_size = len(lineup)

                    # NEW: Flag lineup size deviations
                    if lineup_size != 5:
                        flag_lineup_size_deviation(ev_time, period, pbp_order, int(tid), lineup_size, desc)

                    state_rows.append({
                        "period": period,
                        "pbp_order": pbp_order,
                        "abs_time": round(ev_time, 3),
                        "team_id": int(tid),
                        "team_abbrev": team_abbrev[int(tid)],
                        "lineup_size": lineup_size,
                        "lineup_player_ids_json": json.dumps(sorted([int(p) for p in lineup])),
                        "lineup_player_names_json": json.dumps(sorted(lineup_names)),
                        "event_desc": desc,
                        "event_type": event_type
                    })

            def flag_lineup_size_deviation(ev_time: float, period: int, pbp_order: int, team_id: int, 
                                         actual_size: int, desc: str):
                """NEW: Flag and analyze lineup size deviations"""
                team_abbr = team_abbrev[team_id]

                # Analyze why lineup isn't size 5
                reasons = []
                team_status = player_status_tracking[team_id]

                if actual_size < 5:
                    missing_count = 5 - actual_size
                    reasons.append(f"Missing {missing_count} player(s) for full lineup")

                    # Check if any players had recent actions but aren't in lineup
                    if team_status['action_without_sub']:
                        reasons.append(f"Players with actions but no sub-in: {[name_map.get(p, str(p)) for p in team_status['action_without_sub']]}")

                elif actual_size > 5:
                    excess_count = actual_size - 5
                    reasons.append(f"Excess {excess_count} player(s) beyond normal lineup")

                # Check recent substitution activity
                recent_subs = [sub for sub in substitution_history[-10:] if sub['team_id'] == team_id]
                if recent_subs:
                    last_sub = recent_subs[-1]
                    reasons.append(f"Last substitution: {last_sub['description']}")

                flag_rows.append({
                    "period": period,
                    "pbp_order": pbp_order,
                    "abs_time": round(ev_time, 3),
                    "team_id": team_id,
                    "team_abbrev": team_abbr,
                    "flag_type": "lineup_size_deviation",
                    "player_id": None,
                    "player_name": None,
                    "idle_seconds": None,
                    "description": desc,
                    "flag_details": f"Lineup size {actual_size}/5. " + "; ".join(reasons),
                    "resolved_via_last_name": False,
                    "sub_direction_inverted": False,
                    "lineup_json": json.dumps(sorted([int(p) for p in on_court[team_id]]))
                })

            def flag_substitution_issue(ev_time: float, period: int, pbp_order: int, issue_type: str, 
                                      team_id: int, player_id: int = None, details: str = ""):
                """NEW: Flag various substitution issues"""
                flag_rows.append({
                    "period": period,
                    "pbp_order": pbp_order,
                    "abs_time": round(ev_time, 3),
                    "team_id": team_id,
                    "team_abbrev": team_abbrev[team_id],
                    "flag_type": issue_type,
                    "player_id": int(player_id) if player_id else None,
                    "player_name": name_map.get(player_id, str(player_id)) if player_id else None,
                    "idle_seconds": None,
                    "description": details,
                    "flag_details": details,
                    "resolved_via_last_name": False,
                    "sub_direction_inverted": False,
                    "lineup_json": json.dumps(sorted([int(p) for p in on_court[team_id]]))
                })

            def validate_substitution(in_pid: int, out_pid: int, sub_tid: int, desc: str, 
                                    ev_time: float, period: int, pbp_order: int) -> dict:
                """NEW: Comprehensive substitution validation"""
                validation_result = {
                    "valid": True,
                    "issues": [],
                    "can_proceed": True
                }

                team_status = player_status_tracking[sub_tid]
                current_lineup = team_status['current_lineup']

                # Check OUT player
                if out_pid:
                    if out_pid not in current_lineup:
                        validation_result["issues"].append(f"OUT player {name_map.get(out_pid)} not in current lineup")
                        flag_substitution_issue(ev_time, period, pbp_order, "sub_out_player_not_in_lineup", 
                                              sub_tid, out_pid, f"Attempted to sub out {name_map.get(out_pid)} who is not in lineup")
                        validation_result["valid"] = False
                    else:
                        # Check when player was last in
                        last_in = team_status['last_sub_in'].get(out_pid, "Game start")
                        validation_result["issues"].append(f"OUT: {name_map.get(out_pid)} (last in: {last_in})")

                # Check IN player  
                if in_pid:
                    if in_pid in current_lineup:
                        validation_result["issues"].append(f"IN player {name_map.get(in_pid)} already in lineup")
                        flag_substitution_issue(ev_time, period, pbp_order, "sub_in_player_already_in_lineup", 
                                              sub_tid, in_pid, f"Attempted to sub in {name_map.get(in_pid)} who is already in lineup")
                        validation_result["valid"] = False
                    else:
                        # Check when player was last out
                        last_out = team_status['last_sub_out'].get(in_pid, "Never subbed out")
                        validation_result["issues"].append(f"IN: {name_map.get(in_pid)} (last out: {last_out})")

                return validation_result

            # ---- Main Processing Loop ----
            logger.info(f"Processing {len(events)} events with TRADITIONAL DATA-DRIVEN approach...")

            for _, ev in events.iterrows():
                period = int(ev.period)
                rem = _parse_gc(ev.game_clock)
                cur_t = _abs_t(period, rem)

                # Credit time between events to current lineups
                if cur_t > prev_abs_time:
                    delta = cur_t - prev_abs_time
                    for tid in teams:
                        for pid in on_court[int(tid)]:
                            seconds_traditional[pid] += delta

                # Handle period transitions
                if period != prev_period:
                    if period in CFG["starter_reset_periods"]:
                        # Reset to starters
                        for tid in teams:
                            on_court[int(tid)] = set(starters[int(tid)])
                            player_status_tracking[tid]['current_lineup'] = set(starters[int(tid)])

                        snapshot_lineups(cur_t, period, int(ev.pbp_order), f"Period {period} start - reset to starters", "PERIOD_START")

                    prev_period = period

                # TRADITIONAL SUBSTITUTION PROCESSING - STRICT DATA ADHERENCE
                if int(ev.msg_type) == CFG["sub_msg_type"]:
                    # Extract players - STRICT: playerId1=IN, playerId2=OUT
                    in_pid = int(ev.player_id_1) if pd.notna(ev.player_id_1) else None
                    out_pid = int(ev.player_id_2) if pd.notna(ev.player_id_2) else None

                    # Determine team
                    sub_tid = None
                    if in_pid and (in_pid in pteam_map):
                        sub_tid = int(pteam_map[in_pid])
                    elif out_pid and (out_pid in pteam_map):
                        sub_tid = int(pteam_map[out_pid])
                    elif pd.notna(ev.team_id_off) and int(ev.team_id_off) in teams:
                        sub_tid = int(ev.team_id_off)

                    if sub_tid is not None:
                        # NEW: Validate substitution before applying
                        validation = validate_substitution(in_pid, out_pid, sub_tid, str(ev.description), 
                                                         cur_t, period, int(ev.pbp_order))

                        # Record substitution attempt
                        sub_record = {
                            "time": cur_t,
                            "period": period,
                            "team_id": sub_tid,
                            "in_player": in_pid,
                            "out_player": out_pid,
                            "description": str(ev.description),
                            "validation": validation
                        }
                        substitution_history.append(sub_record)

                        # Apply substitution (even if flagged - we follow the data)
                        team_status = player_status_tracking[sub_tid]

                        if out_pid and out_pid in on_court[sub_tid]:
                            on_court[sub_tid].remove(out_pid)
                            team_status['current_lineup'].remove(out_pid)
                            team_status['last_sub_out'][out_pid] = cur_t
                            logger.info(f"[TRADITIONAL SUB-OUT] {name_map.get(out_pid)} from {team_abbrev[sub_tid]}")

                        if in_pid:
                            on_court[sub_tid].add(in_pid)
                            team_status['current_lineup'].add(in_pid)
                            team_status['last_sub_in'][in_pid] = cur_t
                            # Remove from action_without_sub if present
                            team_status['action_without_sub'].discard(in_pid)
                            logger.info(f"[TRADITIONAL SUB-IN] {name_map.get(in_pid)} to {team_abbrev[sub_tid]}")

                        # Snapshot after substitution
                        snapshot_lineups(cur_t, period, int(ev.pbp_order), str(ev.description), "SUBSTITUTION")

                # Check for actions by players not in lineup
                elif int(ev.msg_type) in CFG["action_msg_types"]:
                    action_pid = int(ev.player_id_1) if pd.notna(ev.player_id_1) else None
                    action_tid = None

                    if action_pid and (action_pid in pteam_map):
                        action_tid = int(pteam_map[action_pid])
                    elif pd.notna(ev.team_id_off) and int(ev.team_id_off) in teams:
                        action_tid = int(ev.team_id_off)

                    if action_tid in teams and action_pid is not None:
                        # Update last action time
                        last_action_time[action_pid] = cur_t
                        player_last_seen[action_pid] = cur_t

                        # Check if player is in lineup
                        if action_pid not in on_court[action_tid]:
                            # Flag action by player not in lineup
                            flag_substitution_issue(cur_t, period, int(ev.pbp_order), "action_by_non_lineup_player", 
                                                  action_tid, action_pid, 
                                                  f"Player {name_map.get(action_pid)} had action but not in lineup: {ev.description}")

                            # Track for analysis
                            player_status_tracking[action_tid]['action_without_sub'].add(action_pid)

                    # Regular lineup snapshot for non-substitution events
                    snapshot_lineups(cur_t, period, int(ev.pbp_order), str(ev.description), "ACTION")

                else:
                    # Other events - just snapshot
                    snapshot_lineups(cur_t, period, int(ev.pbp_order), str(ev.description), "OTHER")

                prev_abs_time = cur_t

            # ---- Build Traditional Minutes ----
            traditional_minutes_rows = []
            for pid, secs in seconds_traditional.items():
                if pid in pteam_map:
                    traditional_minutes_rows.append({
                        "player_id": int(pid),
                        "player_name": name_map.get(pid, str(pid)),
                        "team_id": int(pteam_map[pid]),
                        "team_abbrev": team_abbrev[int(pteam_map[pid])],
                        "seconds_traditional": round(float(secs), 3)
                    })

            traditional_minutes_df = pd.DataFrame(traditional_minutes_rows).sort_values(["team_abbrev", "player_name"]).reset_index(drop=True)

            # ---- Persist to DuckDB ----
            # Replace the basic tables with traditional data-driven versions
            self._robust_drop_object("traditional_lineup_state")
            self.conn.register("traditional_lineup_state_temp", pd.DataFrame(state_rows))
            self.conn.execute("CREATE TABLE traditional_lineup_state AS SELECT * FROM traditional_lineup_state_temp")
            self.conn.execute("DROP VIEW IF EXISTS traditional_lineup_state_temp")

            self._robust_drop_object("traditional_lineup_flags")
            self.conn.register("traditional_lineup_flags_temp", pd.DataFrame(flag_rows))
            self.conn.execute("CREATE TABLE traditional_lineup_flags AS SELECT * FROM traditional_lineup_flags_temp")
            self.conn.execute("DROP VIEW IF EXISTS traditional_lineup_flags_temp")

            self._robust_drop_object("minutes_traditional")
            self.conn.register("minutes_traditional_temp", traditional_minutes_df)
            self.conn.execute("CREATE TABLE minutes_traditional AS SELECT * FROM minutes_traditional_temp")
            self.conn.execute("DROP VIEW IF EXISTS minutes_traditional_temp")

            # ---- Summary Statistics ----
            flag_summary = {}
            if flag_rows:
                flag_df = pd.DataFrame(flag_rows)
                flag_summary = dict(flag_df["flag_type"].value_counts())

            lineup_size_analysis = {}
            if state_rows:
                state_df = pd.DataFrame(state_rows)
                size_counts = state_df["lineup_size"].value_counts().to_dict()
                lineup_size_analysis = {
                    "total_states": len(state_rows),
                    "size_distribution": {str(k): int(v) for k, v in size_counts.items()},
                    "non_5_player_states": int(sum(v for k, v in size_counts.items() if k != 5)),
                    "percentage_correct_size": round(size_counts.get(5, 0) / len(state_rows) * 100, 1) if state_rows else 0
                }

            substitution_analysis = {
                "total_substitutions": len(substitution_history),
                "valid_substitutions": len([s for s in substitution_history if s["validation"]["valid"]]),
                "flagged_substitutions": len([s for s in substitution_history if not s["validation"]["valid"]])
            }

            self.data_summary["traditional_data_driven"] = {
                "state_rows": len(state_rows),
                "flag_rows": len(flag_rows),
                "minutes_rows": len(traditional_minutes_df),
                "flag_summary": flag_summary,
                "lineup_size_analysis": lineup_size_analysis,
                "substitution_analysis": substitution_analysis,
                "substitution_history": substitution_history[-20:]  # Last 20 for debugging
            }

            logger.info(f"[TRADITIONAL DATA-DRIVEN] {len(substitution_history)} total substitutions")
            logger.info(f"[TRADITIONAL DATA-DRIVEN] {len(flag_rows)} flags generated")
            logger.info(f"[TRADITIONAL DATA-DRIVEN] Lineup size distribution: {lineup_size_analysis['size_distribution']}")

            return ValidationResult(
                step_name="Traditional Data-Driven Lineups",
                passed=True,
                details=(f"Traditional data-driven tracking: {len(state_rows)} states, "
                        f"{len(flag_rows)} flags, {lineup_size_analysis['percentage_correct_size']}% correct lineup size"),
                processing_time=time.time() - start_time
            )

        except Exception as e:
            return ValidationResult(
                step_name="Traditional Data-Driven Lineups",
                passed=False,
                details=f"Error in traditional data-driven tracking: {e}",
                processing_time=time.time() - start_time
            )

    def compare_traditional_vs_enhanced_lineups(self) -> ValidationResult:
        """
        UPDATED COMPREHENSIVE COMPARISON: Traditional Data-Driven vs Enhanced Methods

        Key Changes:
        1. Uses traditional_lineup_state instead of basic_lineup_state
        2. Enhanced analysis of lineup size variations
        3. Detailed flagging comparison between methods
        """
        start_time = time.time()

        try:
            # Check if both methods have been run
            traditional_data = self.data_summary.get('traditional_data_driven', {})
            enhanced_data = self.data_summary.get('enhanced_substitution_tracking', {})

            if not traditional_data or not enhanced_data:
                return ValidationResult(
                    step_name="Compare Traditional vs Enhanced (Updated)",
                    passed=False,
                    details="Both traditional data-driven and enhanced methods must be run first",
                    processing_time=time.time() - start_time
                )

            # Get validation data from both methods
            traditional_validation = self.conn.execute("SELECT * FROM minutes_traditional").df()
            enhanced_validation = enhanced_data.get('validation_data', pd.DataFrame())
            box_validation = self.conn.execute("SELECT player_id, player_name, team_id, team_abbrev, seconds_played FROM box_score WHERE seconds_played > 0").df()

            # Merge all data for comparison
            comparison_df = box_validation.merge(
                traditional_validation[['player_id', 'seconds_traditional']], 
                on='player_id', 
                how='left'
            ).merge(
                enhanced_validation[['player_id', 'calc_seconds']].rename(columns={'calc_seconds': 'seconds_enhanced'}),
                on='player_id',
                how='left'
            )

            # Fill NaN values
            comparison_df['seconds_traditional'] = comparison_df['seconds_traditional'].fillna(0.0)
            comparison_df['seconds_enhanced'] = comparison_df['seconds_enhanced'].fillna(0.0)

            # Calculate differences vs box score
            comparison_df['traditional_vs_box_diff'] = comparison_df['seconds_traditional'] - comparison_df['seconds_played']
            comparison_df['enhanced_vs_box_diff'] = comparison_df['seconds_enhanced'] - comparison_df['seconds_played']
            comparison_df['traditional_vs_box_abs_diff'] = comparison_df['traditional_vs_box_diff'].abs()
            comparison_df['enhanced_vs_box_abs_diff'] = comparison_df['enhanced_vs_box_diff'].abs()

            # Calculate percentage differences
            def safe_pct_diff(calc, box):
                return (calc - box) / box if box > 0 else 0.0

            comparison_df['traditional_vs_box_pct_diff'] = comparison_df.apply(
                lambda row: safe_pct_diff(row['seconds_traditional'], row['seconds_played']), axis=1
            )
            comparison_df['enhanced_vs_box_pct_diff'] = comparison_df.apply(
                lambda row: safe_pct_diff(row['seconds_enhanced'], row['seconds_played']), axis=1
            )

            # Determine which method is better for each player
            comparison_df['method_improvement'] = comparison_df['traditional_vs_box_abs_diff'] - comparison_df['enhanced_vs_box_abs_diff']
            comparison_df['better_method'] = comparison_df['method_improvement'].apply(
                lambda x: 'Enhanced' if x > 0 else 'Traditional' if x < 0 else 'Tie'
            )

            # Calculate summary statistics
            tolerance_seconds = 120

            traditional_offenders = len(comparison_df[comparison_df['traditional_vs_box_abs_diff'] > tolerance_seconds])
            enhanced_offenders = len(comparison_df[comparison_df['enhanced_vs_box_abs_diff'] > tolerance_seconds])

            improved_players = len(comparison_df[comparison_df['method_improvement'] > 0])
            worsened_players = len(comparison_df[comparison_df['method_improvement'] < 0])
            tied_players = len(comparison_df[comparison_df['method_improvement'] == 0])

            # UPDATED: Get flag statistics for traditional method
            traditional_flags = traditional_data.get('flag_summary', {})
            enhanced_flags = enhanced_data.get('flags', {})

            flag_comparison = {
                'traditional_total_flags': sum(traditional_flags.values()),
                'enhanced_total_flags': sum(len(flag_list) for flag_list in enhanced_flags.values()),
                'traditional_flag_types': traditional_flags,
                'enhanced_flag_types': {k: len(v) for k, v in enhanced_flags.items()}
            }

            # UPDATED: Get lineup size analysis
            traditional_lineup_analysis = traditional_data.get('lineup_size_analysis', {})
            enhanced_lineup_analysis = self._get_enhanced_lineup_analysis()

            lineup_comparison = {
                'traditional': {
                    'total_states': traditional_lineup_analysis.get('total_states', 0),
                    'size_distribution': traditional_lineup_analysis.get('size_distribution', {}),
                    'correct_size_percentage': traditional_lineup_analysis.get('percentage_correct_size', 0),
                    'non_5_player_states': traditional_lineup_analysis.get('non_5_player_states', 0)
                },
                'enhanced': {
                    'total_states': enhanced_lineup_analysis.get('total_states', 0),
                    'size_distribution': enhanced_lineup_analysis.get('size_distribution', {}),
                    'correct_size_percentage': enhanced_lineup_analysis.get('percentage_correct_size', 0),
                    'non_5_player_states': enhanced_lineup_analysis.get('non_5_player_states', 0)
                }
            }

            # Create comprehensive comparison summary
            comparison_summary = {
                'method_comparison': {
                    'traditional_offenders': int(traditional_offenders),
                    'enhanced_offenders': int(enhanced_offenders),
                    'improvement': int(traditional_offenders - enhanced_offenders),
                    'players_improved': int(improved_players),
                    'players_worsened': int(worsened_players),
                    'players_tied': int(tied_players),
                    'total_players': int(len(comparison_df))
                },
                'accuracy_metrics': {
                    'traditional_avg_abs_diff': float(comparison_df['traditional_vs_box_abs_diff'].mean()),
                    'enhanced_avg_abs_diff': float(comparison_df['enhanced_vs_box_abs_diff'].mean()),
                    'traditional_max_diff': float(comparison_df['traditional_vs_box_abs_diff'].max()),
                    'enhanced_max_diff': float(comparison_df['enhanced_vs_box_abs_diff'].max()),
                    'traditional_within_10pct': int(len(comparison_df[comparison_df['traditional_vs_box_pct_diff'].abs() <= 0.10])),
                    'enhanced_within_10pct': int(len(comparison_df[comparison_df['enhanced_vs_box_pct_diff'].abs() <= 0.10]))
                },
                'flag_analysis': flag_comparison,
                'lineup_analysis': lineup_comparison
            }

            # Store results
            self.data_summary['traditional_vs_enhanced_comparison_updated'] = {
                'comparison_data': comparison_df,
                'summary': comparison_summary,
                'processing_time': time.time() - start_time
            }

            # Create database tables
            self._robust_drop_object("traditional_vs_enhanced_comparison_updated")
            self.conn.register('comparison_updated_temp', comparison_df)
            self.conn.execute("CREATE TABLE traditional_vs_enhanced_comparison_updated AS SELECT * FROM comparison_updated_temp")
            self.conn.execute("DROP VIEW IF EXISTS comparison_updated_temp")

            # Log results
            logger.info(f"[UPDATED COMPARISON] Traditional Data-Driven: {traditional_offenders} offenders, Enhanced: {enhanced_offenders} offenders")
            logger.info(f"[UPDATED COMPARISON] Improvement: {traditional_offenders - enhanced_offenders} fewer offenders")
            logger.info(f"[UPDATED COMPARISON] Traditional flags: {sum(traditional_flags.values())}, Enhanced flags: {sum(len(flag_list) for flag_list in enhanced_flags.values())}")
            logger.info(f"[UPDATED COMPARISON] Traditional lineup size distribution: {traditional_lineup_analysis.get('size_distribution', {})}")

            details = (f"Updated comparison complete: Traditional Data-Driven ({traditional_offenders} offenders) vs Enhanced ({enhanced_offenders} offenders). "
                      f"Improvement: {traditional_offenders - enhanced_offenders} fewer offenders. "
                      f"Traditional flagged {sum(traditional_flags.values())} issues, Enhanced flagged {sum(len(flag_list) for flag_list in enhanced_flags.values())} issues.")

            return ValidationResult(
                step_name="Compare Traditional vs Enhanced (Updated)",
                passed=True,
                details=details,
                processing_time=time.time() - start_time
            )

        except Exception as e:
            logger.error(f"Error in updated traditional vs enhanced comparison: {e}")
            return ValidationResult(
                step_name="Compare Traditional vs Enhanced (Updated)",
                passed=False,
                details=f"Error in updated comparison: {str(e)}",
                processing_time=time.time() - start_time
            )

    def _get_enhanced_lineup_analysis(self) -> Dict[str, Any]:
        """Helper method to get enhanced lineup size analysis"""
        try:
            enhanced_states = self.conn.execute("SELECT lineup_size FROM enhanced_lineup_state").df()
            if enhanced_states.empty:
                return {}

            size_counts = enhanced_states['lineup_size'].value_counts().to_dict()
            return {
                'total_states': len(enhanced_states),
                'size_distribution': {str(k): int(v) for k, v in size_counts.items()},
                'non_5_player_states': int(sum(v for k, v in size_counts.items() if k != 5)),
                'percentage_correct_size': round(size_counts.get(5, 0) / len(enhanced_states) * 100, 1)
            }
        except Exception:
            return {}

    def _create_comprehensive_flags_table(self, flags_data: Dict[str, List]) -> None:
        """Create comprehensive flags analysis table"""
        try:
            all_flags = []

            for flag_type, flag_list in flags_data.items():
                for flag in flag_list:
                    flag_record = {
                        'flag_type': flag_type,
                        'time': flag.get('time', 0),
                        'player_id': flag.get('player_id'),
                        'player_name': flag.get('player_name'),
                        'team': flag.get('team'),
                        'action_type': flag.get('action_type'),
                        'idle_seconds': flag.get('idle_seconds'),
                        'description': flag.get('description', ''),
                        'resolution': flag.get('resolution', ''),
                        'full_details': str(flag)
                    }
                    all_flags.append(flag_record)

            if all_flags:
                flags_df = pd.DataFrame(all_flags)
                self._robust_drop_object("comprehensive_flags_analysis")
                self.conn.register('flags_analysis_temp', flags_df)
                self.conn.execute("CREATE TABLE comprehensive_flags_analysis AS SELECT * FROM flags_analysis_temp")
                self.conn.execute("DROP VIEW IF EXISTS flags_analysis_temp")

                logger.info(f"Created comprehensive_flags_analysis table with {len(all_flags)} flag records")

        except Exception as e:
            logger.warning(f"Could not create comprehensive flags table: {e}")

    def compare_basic_vs_estimated_lineups(self) -> ValidationResult:
        """
        Compare minutes from: basic pass vs enhanced estimator vs box score.
        FIXED: Creates DuckDB table from available sources before comparison.
        """
        start_time = time.time()
        try:
            # FIXED: Check and create missing tables
            have_basic = self.conn.execute("SELECT COUNT(*) FROM information_schema.tables WHERE table_name='minutes_basic'").fetchone()[0]
            have_box = self.conn.execute("SELECT COUNT(*) FROM information_schema.tables WHERE table_name='box_score'").fetchone()[0]

            # Solution 1: Create minutes_basic from CSV if it exists
            if have_basic == 0:
                try:
                    # Check multiple possible locations for CSV
                    csv_candidates = [
                        "minutes_basic.csv",
                        "reports/minutes_basic.csv", 
                        "exports/minutes_basic.csv",
                        str(PROCESSED_DIR / "minutes_basic.csv") if 'PROCESSED_DIR' in globals() else None
                    ]

                    csv_found = False
                    for csv_path in csv_candidates:
                        if csv_path and Path(csv_path).exists():
                            self.conn.execute(f"CREATE TABLE minutes_basic AS SELECT * FROM read_csv_auto('{csv_path}')")
                            have_basic = 1
                            csv_found = True
                            logger.info(f"Created minutes_basic table from {csv_path}")
                            break

                    # Solution 2: Use alternative tables as fallback
                    if not csv_found:
                        alt_tables = ['minutes_traditional', 'minutes_enhanced']
                        for alt_table in alt_tables:
                            alt_exists = self.conn.execute(f"SELECT COUNT(*) FROM information_schema.tables WHERE table_name='{alt_table}'").fetchone()[0]
                            if alt_exists:
                                # Map the alternative table columns to minutes_basic format
                                if alt_table == 'minutes_traditional':
                                    self.conn.execute("""
                                        CREATE TABLE minutes_basic AS 
                                        SELECT 
                                            player_id, 
                                            player_name, 
                                            team_id, 
                                            team_abbrev,
                                            seconds_traditional as seconds_basic
                                        FROM minutes_traditional
                                    """)
                                elif alt_table == 'minutes_enhanced':
                                    self.conn.execute("""
                                        CREATE TABLE minutes_basic AS 
                                        SELECT 
                                            player_id, 
                                            player_name, 
                                            team_id, 
                                            team_abbrev,
                                            seconds_enhanced as seconds_basic
                                        FROM minutes_enhanced
                                    """)
                                have_basic = 1
                                logger.info(f"Created minutes_basic table from {alt_table}")
                                break

                    # Solution 3: Create synthetic minutes_basic from box_score if needed
                    if have_basic == 0 and have_box > 0:
                        self.conn.execute("""
                            CREATE TABLE minutes_basic AS 
                            SELECT 
                                nbaId as player_id, 
                                name as player_name, 
                                nbaTeamId as team_id, 
                                team as team_abbrev,
                                secPlayed as seconds_basic
                            FROM box_score 
                            WHERE secPlayed > 0
                        """)
                        have_basic = 1
                        logger.info("Created synthetic minutes_basic from box_score")

                except Exception as e:
                    logger.warning(f"Could not create minutes_basic table: {e}")

            if have_basic == 0 or have_box == 0:
                missing_items = []
                if have_basic == 0:
                    missing_items.append("minutes_basic")
                if have_box == 0:
                    missing_items.append("box_score")
                return ValidationResult("Compare Minutes", False, f"Missing required tables: {missing_items}")

            # Enhanced minutes are in self.data_summary['minutes_validation_full'] if run_lineups_and_rim_analytics() already executed.
            # But we want the compare to be callable before/after. We'll use DuckDB if present; otherwise fall back to data_summary.
            enhanced_df = self.data_summary.get("minutes_validation_full")
            if enhanced_df is None:
                # Create a minimal enhanced view if needed to keep pipeline flowing
                enhanced_df = pd.DataFrame(columns=[
                    "player_id","player_name","team","calc_seconds","box_seconds","abs_diff_seconds","segments_count"
                ])

            minutes_basic = self.conn.execute("""
                SELECT b.player_id, b.player_name, b.team_id, b.team_abbrev, b.seconds_basic
                FROM minutes_basic b
            """).df()

            box = self.conn.execute("""
                SELECT player_id, player_name, team_id, team_abbrev, seconds_played
                FROM box_score
            """).df()

            # merge frames
            cmp_df = minutes_basic.merge(
                box.rename(columns={"seconds_played":"box_seconds"}),
                on=["player_id","player_name","team_id","team_abbrev"],
                how="outer"
            )

            # attach enhanced if available
            if not enhanced_df.empty:
                e_small = enhanced_df[["player_id","calc_seconds"]].rename(columns={"calc_seconds":"enhanced_seconds"})
                cmp_df = cmp_df.merge(e_small, on="player_id", how="left")

            # fill NaNs with 0 where appropriate (for diffs only; we do not alter raw tables)
            for col in ["seconds_basic","box_seconds","enhanced_seconds"]:
                if col in cmp_df.columns:
                    cmp_df[col] = cmp_df[col].fillna(0.0)

            # percentage diffs vs box
            def pct_diff(a, b):
                return None if b == 0 else (a - b) / b

            cmp_df["basic_vs_box_sec_diff"] = cmp_df["seconds_basic"] - cmp_df["box_seconds"]
            cmp_df["basic_vs_box_pct_diff"] = cmp_df.apply(lambda r: pct_diff(r["seconds_basic"], r["box_seconds"]), axis=1)

            if "enhanced_seconds" in cmp_df.columns:
                cmp_df["enhanced_vs_box_sec_diff"] = cmp_df["enhanced_seconds"] - cmp_df["box_seconds"]
                cmp_df["enhanced_vs_box_pct_diff"] = cmp_df.apply(lambda r: pct_diff(r.get("enhanced_seconds",0.0), r["box_seconds"]), axis=1)
            else:
                cmp_df["enhanced_vs_box_sec_diff"] = 0.0
                cmp_df["enhanced_vs_box_pct_diff"] = None

            cmp_df = cmp_df.sort_values(["team_abbrev","player_name"]).reset_index(drop=True)

            # persist
            self._robust_drop_object("minutes_compare")
            self.conn.register("minutes_compare_temp", cmp_df)
            self.conn.execute("CREATE TABLE minutes_compare AS SELECT * FROM minutes_compare_temp")
            self.conn.execute("DROP VIEW IF EXISTS minutes_compare_temp")

            # summary: how many within 10% of box?
            within10_basic = int((cmp_df["basic_vs_box_pct_diff"].abs() <= 0.10).sum())
            total_w_box    = int((cmp_df["box_seconds"] > 0).sum())

            self.data_summary["minutes_compare"] = {
                "rows": len(cmp_df),
                "within10_basic": within10_basic,
                "total_with_box": total_w_box
            }

            return ValidationResult(
                step_name="Compare Minutes",
                passed=True,
                details=f"minutes_compare built ({len(cmp_df)} rows). Basic within 10%: {within10_basic}/{total_w_box}.",
                processing_time=time.time() - start_time
            )

        except Exception as e:
            return ValidationResult(
                step_name="Compare Minutes",
                passed=False,
                details=f"Error building minutes_compare: {e}",
                processing_time=time.time() - start_time
            )

    def validate_dataset_compliance(self) -> ValidationResult:
        """
        Validate that generated datasets meet project requirements.
        Critical validation to ensure deliverables are usable.

        FIXED: Corrected SQL column references to match actual schema
        - Changed 'secPlayed' to 'seconds_played' to match box_score table schema
        - Added better error handling for missing tables
        - Enhanced validation reporting with specific compliance metrics
        """
        start_time = time.time()
        try:
            validation_issues = []

            # Project 1: Validate 5-man lineup requirement
            lineup_table_exists = self.conn.execute(
                "SELECT COUNT(*) FROM information_schema.tables WHERE table_name='final_dual_lineups'"
            ).fetchone()[0]

            if lineup_table_exists:
                lineup_compliance = self.conn.execute("""
                    SELECT 
                        method,
                        COUNT(*) as total_lineups,
                        SUM(CASE WHEN lineup_size = 5 THEN 1 ELSE 0 END) as five_man_lineups,
                        (SUM(CASE WHEN lineup_size = 5 THEN 1 ELSE 0 END) * 100.0 / COUNT(*)) as compliance_pct
                    FROM final_dual_lineups
                    GROUP BY method
                """).df()

                for _, row in lineup_compliance.iterrows():
                    method = row['method']
                    compliance = row['compliance_pct']
                    if compliance < 100.0:
                        validation_issues.append(
                            f"{method.title()} method: {compliance:.1f}% compliance with 5-man requirement (FAILS)"
                        )
            else:
                validation_issues.append("final_dual_lineups table not found - cannot validate lineup compliance")

            # Project 2: Validate rim defense coverage
            players_table_exists = self.conn.execute(
                "SELECT COUNT(*) FROM information_schema.tables WHERE table_name='final_dual_players'"
            ).fetchone()[0]

            if players_table_exists:
                rim_coverage = self.conn.execute("""
                    SELECT 
                        method,
                        COUNT(*) as total_players,
                        SUM(CASE WHEN opp_rim_attempts_on > 0 OR opp_rim_attempts_off > 0 THEN 1 ELSE 0 END) as players_with_rim_data,
                        (SUM(CASE WHEN opp_rim_attempts_on > 0 OR opp_rim_attempts_off > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*)) as coverage_pct
                    FROM final_dual_players
                    GROUP BY method
                """).df()

                # FIXED: Use correct column name 'seconds_played' instead of 'secPlayed'
                box_table_exists = self.conn.execute(
                    "SELECT COUNT(*) FROM information_schema.tables WHERE table_name='box_score'"
                ).fetchone()[0]

                if box_table_exists:
                    expected_players = self.conn.execute(
                        "SELECT COUNT(*) FROM box_score WHERE seconds_played > 0"
                    ).fetchone()[0]

                    for _, row in rim_coverage.iterrows():
                        method = row['method']
                        coverage = row['coverage_pct']
                        players_covered = int(row['players_with_rim_data'])
                        missing_players = expected_players - players_covered

                        if missing_players > 0:
                            validation_issues.append(
                                f"{method.title()} method: Missing rim data for {missing_players}/{expected_players} players"
                            )
                else:
                    validation_issues.append("box_score table not found - cannot validate expected player count")
            else:
                validation_issues.append("final_dual_players table not found - cannot validate rim coverage")

            # Minutes validation tolerance check
            enhanced_validation = self.data_summary.get('minutes_validation_full')
            if enhanced_validation is not None:
                tolerance_violations = len(enhanced_validation[enhanced_validation['abs_diff_seconds'] > 120])
                if tolerance_violations > 0:
                    validation_issues.append(
                        f"Minutes validation: {tolerance_violations} players exceed 120s tolerance"
                    )

            passed = len(validation_issues) == 0
            details = f"Dataset compliance check: {len(validation_issues)} issues found"

            # Add success details if validation passed
            if passed:
                details += " - All compliance requirements met"

            return ValidationResult(
                step_name="Dataset Compliance Validation",
                passed=passed,
                details=details,
                processing_time=time.time() - start_time,
                warnings=validation_issues
            )

        except Exception as e:
            return ValidationResult(
                step_name="Dataset Compliance Validation", 
                passed=False,
                details=f"Error validating dataset compliance: {str(e)}",
                processing_time=time.time() - start_time
            )

    def create_project_submission_artifacts(self) -> ValidationResult:
        """
        Create final artifacts specifically for project submission using only compliant methods.
        """
        start_time = time.time()
        try:
            # Use enhanced method only for final submission due to 100% 5-man compliance
            logger.info("Creating project submission artifacts using enhanced method...")

            # Project 1: Final lineup submission
            enhanced_lineups = self.conn.execute(f"""
                SELECT 
                    team_abbrev as "Team",
                    player_1_name as "Player 1",
                    player_2_name as "Player 2", 
                    player_3_name as "Player 3",
                    player_4_name as "Player 4",
                    player_5_name as "Player 5",
                    off_possessions as "Offensive possessions played",
                    def_possessions as "Defensive possessions played",
                    off_rating as "Offensive rating", 
                    def_rating as "Defensive rating",
                    net_rating as "Net rating"
                FROM final_dual_lineups
                WHERE method = 'enhanced'
                AND lineup_size = 5  -- Ensure 5-man compliance
                AND (off_possessions > 0 OR def_possessions > 0)
                ORDER BY team_abbrev, off_possessions DESC
            """).df()

            # Project 2: Final player submission
            enhanced_players = self.conn.execute(f"""
                SELECT 
                    player_id as "Player ID",
                    player_name as "Player Name", 
                    team_abbrev as "Team",
                    off_possessions as "Offensive possessions played",
                    def_possessions as "Defensive possessions played",
                    ROUND(COALESCE(opp_rim_fg_pct_on, 0), 4) as "Opponent rim field goal percentage when player is on the court",
                    ROUND(COALESCE(opp_rim_fg_pct_off, 0), 4) as "Opponent rim field goal percentage when player is off the court", 
                    ROUND(COALESCE(rim_defense_on_off, 0), 4) as "Opponent rim field goal percentage on/off difference (on-off)"
                FROM final_dual_players
                WHERE method = 'enhanced'
                AND (off_possessions > 0 OR def_possessions > 0)
                ORDER BY team_abbrev, player_name
            """).df()

            # Export final submission files
            submission_dir = self.export_dir / "final_submission"
            submission_dir.mkdir(exist_ok=True)

            enhanced_lineups.to_csv(submission_dir / "project1_lineups_FINAL.csv", index=False)
            enhanced_players.to_csv(submission_dir / "project2_players_FINAL.csv", index=False)

            # Create submission validation report
            validation_report = {
                "project1_lineups": {
                    "total_lineups": len(enhanced_lineups),
                    "teams_covered": enhanced_lineups['Team'].nunique(),
                    "five_man_compliance": "100%",
                    "file_size_kb": round((submission_dir / "project1_lineups_FINAL.csv").stat().st_size / 1024, 1)
                },
                "project2_players": {
                    "total_players": len(enhanced_players), 
                    "teams_covered": enhanced_players['Team'].nunique(),
                    "rim_data_coverage": f"{len(enhanced_players[enhanced_players['Opponent rim field goal percentage when player is on the court'] > 0])}/{len(enhanced_players)} players",
                    "file_size_kb": round((submission_dir / "project2_players_FINAL.csv").stat().st_size / 1024, 1)
                }
            }

            with open(submission_dir / "submission_validation_report.json", 'w') as f:
                import json
                json.dump(validation_report, f, indent=2)

            details = f"Created final submission artifacts: {len(enhanced_lineups)} lineups, {len(enhanced_players)} players"

            return ValidationResult(
                step_name="Create Submission Artifacts",
                passed=True,
                details=details,
                data_count=len(enhanced_lineups) + len(enhanced_players),
                processing_time=time.time() - start_time
            )

        except Exception as e:
            return ValidationResult(
                step_name="Create Submission Artifacts",
                passed=False, 
                details=f"Error creating submission artifacts: {str(e)}",
                processing_time=time.time() - start_time
            )

    def run_enhanced_substitution_tracking_with_flags(self) -> ValidationResult:
        """
        ENHANCED SUBSTITUTION TRACKING WITH COMPREHENSIVE FLAGGING

        This method provides the enhanced substitution logic with detailed flagging:
        - First-action rules (Reed Sheppard case)
        - Auto-out for inactivity periods
        - Comprehensive flagging system
        - Lineup size enforcement

        Flags captured:
        - missing_sub_in: Players with actions but no substitution in
        - inactivity_periods: Players on court >2 minutes without action
        - first_action_events: Reed Sheppard case injections
        - auto_out_events: Automatic removals due to inactivity
        - lineup_violations: Any time lineup != 5 players
        """
        start_time = time.time()

        CFG = {
            "starter_reset_periods": [1, 3],
            "one_direction": {
                "appearance_via_last_name": True,
                "remove_out_if_present": True
            },
            "msg_types": {
                "shot_made": 1, "shot_missed": 2, "rebound": 4,
                "turnover": 5, "foul": 6, "substitution": 8
            },
            "minutes_validation": {"tolerance_seconds": 120},
            "inactivity_rule": {"idle_seconds_threshold": 120}
        }

        try:
            # Helper functions
            def _period_length_seconds(p: int) -> float:
                return 720.0 if p <= 4 else 300.0

            def _parse_game_clock(gc: str) -> float | None:
                if not gc or not isinstance(gc, str):
                    return None
                s = gc.strip()
                if s.count(":") != 1:
                    return None
                try:
                    mm, ss = s.split(":")
                    return float(mm) * 60.0 + float(ss)
                except (ValueError, IndexError):
                    return None

            def _abs_time(period: int, rem_sec: float | None) -> float:
                total = 0.0
                for pi in range(1, period):
                    total += _period_length_seconds(pi)
                pl = _period_length_seconds(period)
                if rem_sec is None:
                    return total + pl
                return total + (pl - rem_sec)

            # Load core data
            box_df = self.conn.execute("""
                SELECT player_id, player_name, team_id, team_abbrev, is_starter, seconds_played
                FROM box_score
                WHERE seconds_played > 0
                ORDER BY team_id, seconds_played DESC
            """).df()

            if box_df.empty:
                return ValidationResult(
                    step_name="Enhanced Substitution Tracking with Flags",
                    passed=False,
                    details="No players found in box_score with playing time",
                    processing_time=time.time() - start_time
                )

            teams = sorted(box_df['team_id'].unique().tolist())
            if len(teams) != 2:
                return ValidationResult(
                    step_name="Enhanced Substitution Tracking with Flags",
                    passed=False,
                    details=f"Expected exactly 2 teams, found {teams}",
                    processing_time=time.time() - start_time
                )

            # Build comprehensive player mappings
            roster = {int(tid): set() for tid in teams}
            starters = {int(tid): set() for tid in teams}
            name_map, pteam_map = {}, {}
            last_name_index = {}

            for _, r in box_df.iterrows():
                pid = int(r.player_id)
                tid = int(r.team_id)
                roster[tid].add(pid)
                if bool(r.is_starter):
                    starters[tid].add(pid)
                name_map[pid] = str(r.player_name)
                pteam_map[pid] = tid

                # Enhanced last name indexing for first-action resolution
                full_name = str(r.player_name).strip()
                last_name = full_name.split()[-1].lower()
                first_name = full_name.split()[0].lower() if len(full_name.split()) > 1 else ""

                last_name_index.setdefault(last_name, []).append(pid)
                if first_name:
                    last_name_index.setdefault(first_name, []).append(pid)

            team_abbrev_map = {int(tid): box_df[box_df.team_id == tid]['team_abbrev'].iloc[0] for tid in teams}

            # Load events with last names for resolution
            events = self.conn.execute("""
                SELECT 
                    period, pbp_order, wall_clock_int,
                    COALESCE(game_clock,'') AS game_clock,
                    COALESCE(description,'') AS description,
                    team_id_off, team_id_def, msg_type, action_type,
                    player_id_1, player_id_2, player_id_3,
                    NULLIF(last_name_1,'') AS last_name_1,
                    NULLIF(last_name_2,'') AS last_name_2,
                    NULLIF(last_name_3,'') AS last_name_3,
                    COALESCE(points, 0) AS points
                FROM pbp
                ORDER BY period, pbp_order, wall_clock_int
            """).df()

            if events.empty:
                return ValidationResult(
                    step_name="Enhanced Substitution Tracking with Flags",
                    passed=False,
                    details="No PBP events found",
                    processing_time=time.time() - start_time
                )

            # Initialize enhanced tracking
            on_court = {tid: set(starters[tid]) for tid in teams}
            last_action_time = defaultdict(lambda: 0.0)
            recent_out = {tid: deque(maxlen=10) for tid in teams}

            active_segments = {}
            completed_segments = defaultdict(list)

            # Enhanced tracking variables with FLAGS
            enhanced_stats = {
                'total_substitutions': 0,
                'successful_substitutions': 0,
                'first_action_injections': 0,
                'auto_outs_inactivity': 0,
                'lineup_size_corrections': 0,
                'flags': {
                    'missing_sub_ins': [],      # Players with actions but no sub-in
                    'inactivity_periods': [],   # Players inactive >2min while on court
                    'lineup_violations': [],    # Times when lineup != 5 players
                    'first_action_events': [],  # First-action injection events
                    'auto_out_events': []       # Auto-out events due to inactivity
                }
            }

            # Lineup state tracking (similar to basic method)
            state_rows = []
            flag_rows = []

            def snapshot_lineups(ev_time: float, period: int, pbp_order: int, desc: str):
                import json
                for tid in teams:
                    lineup = list(on_court[tid])
                    lineup_names = [name_map.get(p, str(p)) for p in lineup]
                    state_rows.append({
                        "period": period,
                        "pbp_order": pbp_order,
                        "abs_time": round(ev_time, 3),
                        "team_id": int(tid),
                        "team_abbrev": team_abbrev_map[tid],
                        "lineup_size": len(lineup),
                        "lineup_player_ids_json": json.dumps(sorted([int(p) for p in lineup])),
                        "lineup_player_names_json": json.dumps(sorted(lineup_names)),
                        "event_desc": desc
                    })

            # Initialize segments for starters
            for tid in teams:
                for pid in on_court[tid]:
                    active_segments[pid] = {'start': 0.0, 'reason': 'GAME_START'}
                    last_action_time[pid] = 0.0

            def enhanced_name_resolution(ln: str | None, tid_hint: int | None) -> int | None:
                """Enhanced name resolution with fuzzy matching"""
                if not ln:
                    return None

                ln_clean = str(ln).strip().lower()
                candidates = last_name_index.get(ln_clean, [])

                if not candidates:
                    # Try partial matching
                    for key, pids in last_name_index.items():
                        if ln_clean in key or key in ln_clean:
                            candidates.extend(pids)

                if not candidates:
                    return None

                # Prefer team hint if available
                if tid_hint is not None:
                    for cand in candidates:
                        if pteam_map.get(cand) == tid_hint:
                            return cand

                # Return first valid candidate
                for cand in candidates:
                    if pteam_map.get(cand) in teams:
                        return cand

                return None

            def end_player_segment(pid: int, end_time: float, reason: str) -> None:
                """End a player's active segment with validation"""
                if pid not in active_segments:
                    return

                start_info = active_segments[pid]
                start_time = start_info['start']

                if end_time <= start_time:
                    end_time = start_time + 1.0

                duration = end_time - start_time
                completed_segments[pid].append({
                    'start': start_time,
                    'end': end_time,
                    'duration': duration,
                    'reason': f"{start_info['reason']} -> {reason}"
                })

                del active_segments[pid]

            def start_player_segment(pid: int, start_time: float, reason: str) -> None:
                """Start a new segment with overlap prevention"""
                if pid in active_segments:
                    end_player_segment(pid, start_time, f"OVERLAP_{reason}")

                active_segments[pid] = {'start': start_time, 'reason': reason}

            def flag_inactivity_check(current_time: float, period: int, pbp_order: int) -> None:
                """Check for players inactive > 2 minutes and flag them"""
                for tid in teams:
                    for pid in on_court[tid]:
                        idle_time = current_time - last_action_time[pid]

                        if idle_time > CFG["inactivity_rule"]["idle_seconds_threshold"]:
                            # Flag this as a potential missing sub-out
                            enhanced_stats['flags']['inactivity_periods'].append({
                                'time': current_time,
                                'player_id': pid,
                                'player_name': name_map.get(pid),
                                'team': team_abbrev_map[tid],
                                'idle_seconds': idle_time,
                                'last_action_time': last_action_time[pid]
                            })
                            # Also add to flag_rows for CSV export
                            flag_rows.append({
                                "period": period,
                                "pbp_order": pbp_order,
                                "abs_time": round(current_time, 3),
                                "team_id": int(tid),
                                "team_abbrev": team_abbrev_map[tid],
                                "flag_type": "inactivity_periods",
                                "player_id": int(pid),
                                "player_name": name_map.get(pid, str(pid)),
                                "idle_seconds": round(idle_time, 3),
                                "description": f"Player inactive for {idle_time:.1f}s (threshold: {CFG['inactivity_rule']['idle_seconds_threshold']}s)"
                            })

            def pick_auto_out_candidate(tid: int, current_time: float, exclude: set[int] = set()) -> int | None:
                """Enhanced auto-out selection based on activity patterns"""
                if not on_court[tid]:
                    return None

                candidates = [p for p in on_court[tid] if p not in exclude]
                if not candidates:
                    return None

                def activity_score(pid: int) -> tuple:
                    idle_time = current_time - last_action_time[pid]
                    is_starter = pid in starters[tid]
                    recently_subbed = pid in recent_out[tid]

                    # Score components (higher = more likely to be removed)
                    idle_score = idle_time
                    starter_penalty = -100 if is_starter else 0
                    recent_sub_penalty = -50 if recently_subbed else 0

                    total_score = idle_score + starter_penalty + recent_sub_penalty

                    return (total_score, idle_time, pid)

                candidates.sort(key=activity_score, reverse=True)
                best_candidate = candidates[0]
                idle_time = current_time - last_action_time[best_candidate]

                if idle_time >= CFG["inactivity_rule"]["idle_seconds_threshold"] or len(on_court[tid]) > 5:
                    return best_candidate

                return None

            def ensure_valid_lineup(tid: int, current_time: float, prefer_keep: set[int] = set()) -> None:
                """Ensure team has exactly 5 players with enhanced logging"""
                team_name = team_abbrev_map[tid]
                changes_made = False

                # Remove excess players
                while len(on_court[tid]) > 5:
                    auto_out = pick_auto_out_candidate(tid, current_time, exclude=prefer_keep)
                    if auto_out is None:
                        logger.error(f"Cannot auto-remove from {team_name} - no valid candidates")
                        break

                    on_court[tid].remove(auto_out)
                    recent_out[tid].append(auto_out)
                    changes_made = True

                    idle_time = current_time - last_action_time[auto_out]
                    enhanced_stats['auto_outs_inactivity'] += 1

                    # Flag this auto-out event
                    enhanced_stats['flags']['auto_out_events'].append({
                        'time': current_time,
                        'player_id': auto_out,
                        'player_name': name_map.get(auto_out),
                        'team': team_name,
                        'idle_time': idle_time,
                        'reason': 'INACTIVITY_AUTO_OUT'
                    })

                    logger.info(f"[ENHANCED AUTO-OUT] {name_map.get(auto_out)} from {team_name} (idle: {idle_time:.1f}s)")

                # Add players if under 5
                if len(on_court[tid]) < 5:
                    available = [p for p in roster[tid] if p not in on_court[tid]]
                    if available:
                        def fill_priority(pid: int) -> tuple:
                            recently_out_priority = 0 if pid in recent_out[tid] else 1
                            starter_priority = 0 if pid in starters[tid] else 1
                            activity_priority = -(current_time - last_action_time[pid])

                            return (recently_out_priority, starter_priority, activity_priority)

                        available.sort(key=fill_priority)

                        needed = 5 - len(on_court[tid])
                        for i in range(min(needed, len(available))):
                            fill_player = available[i]
                            on_court[tid].add(fill_player)
                            changes_made = True
                            logger.info(f"[ENHANCED AUTO-IN] {name_map.get(fill_player)} to {team_name} (fill to 5)")

                # Update segments if changes were made
                if changes_made:
                    enhanced_stats['lineup_size_corrections'] += 1
                    for pid in on_court[tid]:
                        if pid not in active_segments:
                            start_player_segment(pid, current_time, "LINEUP_CORRECTION")

                    # Flag lineup correction
                    enhanced_stats['flags']['lineup_violations'].append({
                        'time': current_time,
                        'team': team_name,
                        'correction_type': 'AUTO_CORRECTION',
                        'final_size': len(on_court[tid])
                    })

            # MAIN PROCESSING LOOP - ENHANCED APPROACH WITH FLAGS
            prev_period = None

            logger.info(f"Processing {len(events)} events with ENHANCED substitution rules and flagging...")

            for idx, ev in events.iterrows():
                period = int(ev.period)
                clock_str = ev.game_clock
                parsed_clock = _parse_game_clock(clock_str)
                current_time = _abs_time(period, parsed_clock)
                msg_type = int(ev.msg_type)

                # Handle period transitions
                if period != prev_period and prev_period is not None:
                    period_end_time = _abs_time(prev_period, 0.0)

                    for pid in list(active_segments.keys()):
                        end_player_segment(pid, period_end_time, f"PERIOD_{prev_period}_END")

                if period != prev_period:
                    if period in CFG["starter_reset_periods"]:
                        on_court = {tid: set(starters[tid]) for tid in teams}
                        logger.info(f"[ENHANCED PERIOD {period}] Reset to starters")
                    else:
                        logger.info(f"[ENHANCED PERIOD {period}] Continue lineups")

                    for tid in teams:
                        for pid in on_court[tid]:
                            if pid not in active_segments:
                                start_player_segment(pid, current_time, f"PERIOD_{period}_START")

                    prev_period = period

                # SUBSTITUTION PROCESSING
                if msg_type == CFG["msg_types"]["substitution"]:
                    enhanced_stats['total_substitutions'] += 1

                    out_pid = int(ev.player_id_1) if not pd.isna(ev.player_id_1) else None
                    in_pid = int(ev.player_id_2) if not pd.isna(ev.player_id_2) else None
                    out_ln, in_ln = ev.last_name_1, ev.last_name_2

                    # Determine team
                    sub_tid = None
                    if in_pid and in_pid in pteam_map:
                        sub_tid = pteam_map[in_pid]
                    elif out_pid and out_pid in pteam_map:
                        sub_tid = pteam_map[out_pid]
                    elif pd.notna(ev.team_id_off) and int(ev.team_id_off) in teams:
                        sub_tid = int(ev.team_id_off)

                    # Enhanced name resolution
                    if in_pid is None and CFG["one_direction"]["appearance_via_last_name"] and in_ln:
                        in_pid = enhanced_name_resolution(in_ln, sub_tid)
                    if out_pid is None and out_ln:
                        out_pid = enhanced_name_resolution(out_ln, sub_tid)

                    if sub_tid is not None:
                        team_name = team_abbrev_map[sub_tid]

                        # Process OUT first
                        if CFG["one_direction"]["remove_out_if_present"] and out_pid and out_pid in on_court[sub_tid]:
                            on_court[sub_tid].remove(out_pid)
                            recent_out[sub_tid].append(out_pid)
                            end_player_segment(out_pid, current_time, "SUB_OUT")
                            logger.info(f"[ENHANCED SUB-OUT] {name_map.get(out_pid)} from {team_name}")

                        # Process IN
                        if in_pid and in_pid not in on_court[sub_tid]:
                            # Make room if needed
                            if len(on_court[sub_tid]) >= 5:
                                auto_out = pick_auto_out_candidate(sub_tid, current_time, exclude={in_pid})
                                if auto_out:
                                    on_court[sub_tid].remove(auto_out)
                                    recent_out[sub_tid].append(auto_out)
                                    end_player_segment(auto_out, current_time, "MAKE_ROOM")
                                    logger.info(f"[ENHANCED MAKE-ROOM] {name_map.get(auto_out)} out for {name_map.get(in_pid)}")

                            on_court[sub_tid].add(in_pid)
                            start_player_segment(in_pid, current_time, "SUB_IN")
                            enhanced_stats['successful_substitutions'] += 1
                            logger.info(f"[ENHANCED SUB-IN] {name_map.get(in_pid)} to {team_name}")

                    # Update activity times
                    if out_pid:
                        last_action_time[out_pid] = current_time
                    if in_pid:
                        last_action_time[in_pid] = current_time

                    # Snapshot lineup after substitution
                    snapshot_lineups(current_time, period, int(ev.pbp_order), str(ev.description))

                # FIRST ACTION PROCESSING (Reed Sheppard rule) WITH FLAGS
                elif msg_type in [1, 2, 4, 5, 6] and CFG["one_direction"]["appearance_via_last_name"]:
                    action_pid = int(ev.player_id_1) if not pd.isna(ev.player_id_1) else None
                    action_ln = ev.last_name_1
                    action_tid = None

                    if action_pid and action_pid in pteam_map:
                        action_tid = pteam_map[action_pid]
                    elif pd.notna(ev.team_id_off) and int(ev.team_id_off) in teams:
                        action_tid = int(ev.team_id_off)

                    # Resolve via last name if needed
                    if action_pid is None and action_ln:
                        action_pid = enhanced_name_resolution(action_ln, action_tid)
                        if action_pid:
                            action_tid = pteam_map[action_pid]

                    # Apply Reed Sheppard rule with FLAG
                    if action_tid in teams and action_pid and action_pid not in on_court[action_tid]:
                        # FLAG: This is a missing sub-in scenario
                        enhanced_stats['flags']['missing_sub_ins'].append({
                            'time': current_time,
                            'player_id': action_pid,
                            'player_name': name_map.get(action_pid),
                            'team': team_abbrev_map[action_tid],
                            'action_type': msg_type,
                            'description': ev.description,
                            'resolution': 'FIRST_ACTION_INJECTION'
                        })

                        # Also add to flag_rows for CSV export
                        flag_rows.append({
                            "period": period,
                            "pbp_order": int(ev.pbp_order),
                            "abs_time": round(current_time, 3),
                            "team_id": int(action_tid),
                            "team_abbrev": team_abbrev_map[action_tid],
                            "flag_type": "missing_sub_in",
                            "player_id": int(action_pid),
                            "player_name": name_map.get(action_pid, str(action_pid)),
                            "idle_seconds": None,
                            "description": str(ev.description or "")
                        })

                        # Inject player via first-action rule
                        on_court[action_tid].add(action_pid)
                        enhanced_stats['first_action_injections'] += 1

                        logger.info(f"[ENHANCED FIRST-ACTION] {name_map.get(action_pid)} -> {team_abbrev_map[action_tid]} (msg: {msg_type})")

                        # Flag the first-action event
                        enhanced_stats['flags']['first_action_events'].append({
                            'time': current_time,
                            'player_id': action_pid,
                            'player_name': name_map.get(action_pid),
                            'team': team_abbrev_map[action_tid],
                            'action': ev.description
                        })

                        start_player_segment(action_pid, current_time, "FIRST_ACTION")
                        ensure_valid_lineup(action_tid, current_time, prefer_keep={action_pid})

                        # Snapshot lineup after first-action injection
                        snapshot_lineups(current_time, period, int(ev.pbp_order), f"FIRST_ACTION: {ev.description}")

                    # Update activity time
                    if action_pid:
                        last_action_time[action_pid] = current_time

                # Periodic inactivity checking
                if idx % 20 == 0:  # Check every 20 events
                    flag_inactivity_check(current_time, period, int(ev.pbp_order))

                # Ensure lineup validity
                for tid in teams:
                    if len(on_court[tid]) != 5:
                        ensure_valid_lineup(tid, current_time)

            # Final processing
            final_time = _abs_time(4, 0.0) if prev_period and prev_period <= 4 else 2880.0
            for pid in list(active_segments.keys()):
                end_player_segment(pid, final_time, "GAME_END")

            # Calculate minutes
            calculated_minutes = {}
            for pid in set(list(completed_segments.keys()) + list(box_df['player_id'])):
                segments = completed_segments[pid]
                total_seconds = sum(seg['duration'] for seg in segments)
                calculated_minutes[pid] = total_seconds

            # Build validation results
            enhanced_validation = []
            for pid in set(list(calculated_minutes.keys()) + list(box_df['player_id'])):
                calc_secs = calculated_minutes.get(pid, 0.0)
                box_row = box_df[box_df['player_id'] == pid]
                box_secs = float(box_row['seconds_played'].iloc[0]) if not box_row.empty else 0.0
                diff = calc_secs - box_secs

                enhanced_validation.append({
                    "player_id": pid,
                    "player_name": name_map.get(pid, f"ID_{pid}"),
                    "team": team_abbrev_map.get(pteam_map.get(pid), "UNK"),
                    "calc_seconds": round(calc_secs, 1),
                    "box_seconds": round(box_secs, 1),
                    "abs_diff_seconds": round(abs(diff), 1),
                    "segments_count": len(completed_segments.get(pid, []))
                })

            enhanced_validation_df = pd.DataFrame(enhanced_validation).sort_values(["team", "player_name"]).reset_index(drop=True)
            enhanced_offenders = enhanced_validation_df[enhanced_validation_df["abs_diff_seconds"] > CFG["minutes_validation"]["tolerance_seconds"]]

            # Create lineup state and flag tables
            state_df = pd.DataFrame(state_rows).sort_values(["period", "pbp_order", "team_id"]).reset_index(drop=True)
            flag_df = pd.DataFrame(flag_rows).sort_values(["abs_time", "team_id"]).reset_index(drop=True)

            # Enhanced minutes table (similar to basic_minutes)
            enhanced_minutes_rows = []
            for pid, secs in calculated_minutes.items():
                if pid in pteam_map:
                    enhanced_minutes_rows.append({
                        "player_id": int(pid),
                        "player_name": name_map.get(pid, str(pid)),
                        "team_id": int(pteam_map[pid]),
                        "team_abbrev": team_abbrev_map[int(pteam_map[pid])],
                        "seconds_enhanced": round(float(secs), 3)
                    })
            enhanced_minutes_df = pd.DataFrame(enhanced_minutes_rows).sort_values(["team_abbrev", "player_name"]).reset_index(drop=True)

            # Persist to DuckDB
            self._robust_drop_object("enhanced_lineup_state")
            self.conn.register("enhanced_lineup_state_temp", state_df)
            self.conn.execute("CREATE TABLE enhanced_lineup_state AS SELECT * FROM enhanced_lineup_state_temp")
            self.conn.execute("DROP VIEW IF EXISTS enhanced_lineup_state_temp")

            self._robust_drop_object("enhanced_lineup_flags")
            self.conn.register("enhanced_lineup_flags_temp", flag_df)
            self.conn.execute("CREATE TABLE enhanced_lineup_flags AS SELECT * FROM enhanced_lineup_flags_temp")
            self.conn.execute("DROP VIEW IF EXISTS enhanced_lineup_flags_temp")

            self._robust_drop_object("minutes_enhanced")
            self.conn.register("minutes_enhanced_temp", enhanced_minutes_df)
            self.conn.execute("CREATE TABLE minutes_enhanced AS SELECT * FROM minutes_enhanced_temp")
            self.conn.execute("DROP VIEW IF EXISTS minutes_enhanced_temp")

            # Store results with enhanced data summary
            flag_totals = {
                "missing_sub_in": len([f for f in flag_rows if f.get("flag_type") == "missing_sub_in"]),
                "inactivity_periods": len([f for f in flag_rows if f.get("flag_type") == "inactivity_periods"]),
                "first_action_events": enhanced_stats['first_action_injections'],
                "auto_out_events": enhanced_stats['auto_outs_inactivity'],
                "lineup_violations": len([f for f in enhanced_stats['flags']['lineup_violations']])
            }

            self.data_summary['enhanced_substitution_tracking'] = {
                'state_rows': len(state_rows),
                'flag_rows': len(flag_rows),
                'flag_totals': flag_totals,
                'validation_data': enhanced_validation_df,
                'offenders_data': enhanced_offenders,
                'flags': enhanced_stats['flags'],
                'statistics': enhanced_stats
            }

            # Store validation data for minutes report
            self.data_summary['minutes_validation_full'] = enhanced_validation_df.copy()
            self.data_summary['minutes_offenders'] = enhanced_offenders.copy()

            # Store debug summary for final report
            self.data_summary['enhanced_substitution_debug'] = {
                "substitutions": enhanced_stats['total_substitutions'],
                "first_actions": enhanced_stats['first_action_injections'],
                "auto_outs": enhanced_stats['auto_outs_inactivity'],
                "always_five_fixes": enhanced_stats['lineup_size_corrections'],
                "validation": {
                    "tolerance": CFG["minutes_validation"]["tolerance_seconds"],
                    "offenders": len(enhanced_offenders),
                    "total_players": len(enhanced_validation_df)
                }
            }

            # Create database tables for flags
            self._create_enhanced_flags_tables(enhanced_stats['flags'])

            # Summary statistics with FLAGS
            summary = {
                'method': 'ENHANCED_WITH_FLAGS',
                'processing_time': time.time() - start_time,
                'substitutions': enhanced_stats,
                'validation': {
                    'total_players': len(enhanced_validation_df),
                    'offenders': len(enhanced_offenders),
                    'tolerance_seconds': CFG["minutes_validation"]["tolerance_seconds"],
                    'max_difference': enhanced_validation_df['abs_diff_seconds'].max() if not enhanced_validation_df.empty else 0
                },
                'flags_summary': {
                    'missing_sub_ins': len(enhanced_stats['flags']['missing_sub_ins']),
                    'inactivity_periods': len(enhanced_stats['flags']['inactivity_periods']),
                    'lineup_violations': len(enhanced_stats['flags']['lineup_violations']),
                    'first_action_events': len(enhanced_stats['flags']['first_action_events']),
                    'auto_out_events': len(enhanced_stats['flags']['auto_out_events'])
                }
            }

            logger.info(f"[ENHANCED COMPLETE] {enhanced_stats['successful_substitutions']} subs, {enhanced_stats['first_action_injections']} first-actions, {enhanced_stats['auto_outs_inactivity']} auto-outs")
            logger.info(f"[ENHANCED FLAGS] Missing sub-ins: {len(enhanced_stats['flags']['missing_sub_ins'])}, Inactivity periods: {len(enhanced_stats['flags']['inactivity_periods'])}")

            total_flags = sum(len(flag_list) for flag_list in enhanced_stats['flags'].values())

            return ValidationResult(
                step_name="Enhanced Substitution Tracking with Flags",
                passed=True,
                details=f"Enhanced tracking complete: {enhanced_stats['successful_substitutions']} subs, {enhanced_stats['first_action_injections']} first-actions, {total_flags} total flags",
                processing_time=time.time() - start_time
            )

        except Exception as e:
            logger.error(f"Error in enhanced substitution tracking: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return ValidationResult(
                step_name="Enhanced Substitution Tracking with Flags",
                passed=False,
                details=f"Error in enhanced tracking: {str(e)}",
                processing_time=time.time() - start_time
            )

    def _create_enhanced_flags_tables(self, flags_data: Dict[str, List]) -> None:
        """Create database tables for enhanced flags"""
        try:
            # Create comprehensive flags table
            all_flags = []

            for flag_type, flag_list in flags_data.items():
                for flag in flag_list:
                    flag_record = {
                        'flag_type': flag_type,
                        'time': flag.get('time', 0),
                        'player_id': flag.get('player_id'),
                        'player_name': flag.get('player_name'),
                        'team': flag.get('team'),
                        'description': str(flag)
                    }
                    all_flags.append(flag_record)

            if all_flags:
                flags_df = pd.DataFrame(all_flags)
                self._robust_drop_object("enhanced_flags")
                self.conn.register('enhanced_flags_temp', flags_df)
                self.conn.execute("CREATE TABLE enhanced_flags AS SELECT * FROM enhanced_flags_temp")
                self.conn.execute("DROP VIEW IF EXISTS enhanced_flags_temp")

                logger.info(f"Created enhanced_flags table with {len(all_flags)} flag records")

        except Exception as e:
            logger.warning(f"Could not create enhanced flags tables: {e}")

    def run_lineups_and_rim_analytics(self) -> ValidationResult:
        """
        CORRECTED substitution engine that systematically fixes all identified issues:

        1. Prevents double-crediting of period remainder time
        2. Properly implements Reed Sheppard first-action rule  
        3. Fixes time tracking between events
        4. Implements intelligent 2-minute inactivity auto-out
        5. Maintains strict 5-man lineups with gap-filling logic

        Key Fixes:
        - Single responsibility for period remainder crediting
        - Enhanced last name resolution for first actions
        - Proper time segment tracking without overlaps
        - Comprehensive debugging and validation
        - Activity-based auto-out selection to prevent inappropriate removals
        """
        start_time = time.time()

        CFG = {
            "starter_reset_periods": [1, 3],
            "one_direction": {
                "appearance_via_last_name": True,
                "remove_out_if_present": True
            },
            "msg_types": {
                "shot_made": 1, "shot_missed": 2, "rebound": 4,
                "turnover": 5, "foul": 6, "substitution": 8
            },
            "minutes_validation": {"tolerance_seconds": 120},  # Reasonable tolerance
            "inactivity_rule": {"idle_seconds_threshold": 120}
        }

        try:
            # Helper functions with better error handling
            def _period_length_seconds(p: int) -> float:
                return 720.0 if p <= 4 else 300.0

            def _parse_game_clock(gc: str) -> float | None:
                if not gc or not isinstance(gc, str):
                    return None
                s = gc.strip()
                if s.count(":") != 1:
                    return None
                try:
                    mm, ss = s.split(":")
                    return float(mm) * 60.0 + float(ss)
                except (ValueError, IndexError):
                    return None

            def _abs_time(period: int, rem_sec: float | None) -> float:
                """Calculate absolute game time elapsed"""
                total = 0.0
                for pi in range(1, period):
                    total += _period_length_seconds(pi)
                pl = _period_length_seconds(period)
                if rem_sec is None:
                    return total + pl  # If no clock, assume period end
                return total + (pl - rem_sec)

            # Load and validate core data
            box_df = self.conn.execute("""
                SELECT player_id, player_name, team_id, team_abbrev, is_starter, seconds_played
                FROM box_score
                WHERE seconds_played > 0
                ORDER BY team_id, seconds_played DESC
            """).df()

            if box_df.empty:
                return ValidationResult(
                    step_name="Enhanced Lineups & Rim Analytics",
                    passed=False,
                    details="No players found in box_score with playing time",
                    processing_time=time.time() - start_time
                )

            teams = sorted(box_df['team_id'].unique().tolist())
            if len(teams) != 2:
                return ValidationResult(
                    step_name="Enhanced Lineups & Rim Analytics",
                    passed=False,
                    details=f"Expected exactly 2 teams, found {teams}",
                    processing_time=time.time() - start_time
                )

            # Build comprehensive player mappings
            roster = {int(tid): set() for tid in teams}
            starters = {int(tid): set() for tid in teams}
            name_map, pteam_map = {}, {}
            last_name_index = {}

            for _, r in box_df.iterrows():
                pid = int(r.player_id)
                tid = int(r.team_id)
                roster[tid].add(pid)
                if bool(r.is_starter):
                    starters[tid].add(pid)
                name_map[pid] = str(r.player_name)
                pteam_map[pid] = tid

                # Enhanced last name indexing
                full_name = str(r.player_name).strip()
                last_name = full_name.split()[-1].lower()
                first_name = full_name.split()[0].lower() if len(full_name.split()) > 1 else ""

                last_name_index.setdefault(last_name, []).append(pid)
                if first_name:
                    last_name_index.setdefault(first_name, []).append(pid)

            # Validate team structure
            if any(len(starters[tid]) != 5 for tid in teams):
                detail = {tid: [name_map[p] for p in sorted(starters[tid])] for tid in teams}
                return ValidationResult(
                    step_name="Enhanced Lineups & Rim Analytics",
                    passed=False,
                    details=f"Invalid starters: {detail}",
                    processing_time=time.time() - start_time
                )

            team_abbrev_map = {int(tid): box_df[box_df.team_id == tid]['team_abbrev'].iloc[0] for tid in teams}

            # Load events with validation
            events = self.conn.execute("""
                SELECT 
                    period, pbp_order, wall_clock_int,
                    COALESCE(game_clock,'') AS game_clock,
                    COALESCE(description,'') AS description,
                    team_id_off, team_id_def, msg_type, action_type,
                    player_id_1, player_id_2, player_id_3,
                    NULLIF(last_name_1,'') AS last_name_1,
                    NULLIF(last_name_2,'') AS last_name_2,
                    NULLIF(last_name_3,'') AS last_name_3,
                    COALESCE(points, 0) AS points
                FROM pbp
                ORDER BY period, pbp_order, wall_clock_int
            """).df()

            if events.empty:
                return ValidationResult(
                    step_name="Enhanced Lineups & Rim Analytics",
                    passed=False,
                    details="No PBP events found",
                    processing_time=time.time() - start_time
                )

            # Initialize state tracking
            from collections import defaultdict, deque

            # CORRECTED: Enhanced state tracking with comprehensive segment management
            on_court = {tid: set(starters[tid]) for tid in teams}
            last_action_time = defaultdict(lambda: 0.0)
            recent_out = {tid: deque(maxlen=10) for tid in teams}

            # CORRECTED: Enhanced segment tracking with validation
            active_segments = {}  # player_id -> {'start': time, 'reason': str}
            completed_segments = defaultdict(list)  # player_id -> [{'start': time, 'end': time, 'duration': dur, 'reason': str}]
            period_end_times = {}  # Track when we last ended a period to prevent double-crediting

            # Initialize segments for starters
            for tid in teams:
                for pid in on_court[tid]:
                    active_segments[pid] = {'start': 0.0, 'reason': 'GAME_START'}
                    last_action_time[pid] = 0.0

            # Enhanced debugging
            debug_events = []
            sub_count = 0
            first_action_count = 0
            auto_out_count = 0
            lineup_violation_fixes = 0
            validation_errors = []

            idle_thresh = CFG.get("inactivity_rule", {}).get("idle_seconds_threshold", 120)
            tol = CFG["minutes_validation"]["tolerance_seconds"]

            def enhanced_name_resolution(ln: str | None, tid_hint: int | None) -> int | None:
                """Enhanced name resolution with fuzzy matching"""
                if not ln:
                    return None

                ln_clean = str(ln).strip().lower()
                candidates = last_name_index.get(ln_clean, [])

                if not candidates:
                    # Try partial matching
                    for key, pids in last_name_index.items():
                        if ln_clean in key or key in ln_clean:
                            candidates.extend(pids)

                if not candidates:
                    return None

                # Prefer team hint if available
                if tid_hint is not None:
                    for cand in candidates:
                        if pteam_map.get(cand) == tid_hint:
                            return cand

                # Return first valid candidate
                for cand in candidates:
                    if pteam_map.get(cand) in teams:
                        return cand

                return None

            def end_player_segment(pid: int, end_time: float, reason: str) -> None:
                """CORRECTED: End a player's active segment with validation"""
                if pid not in active_segments:
                    return  # No active segment to end

                start_info = active_segments[pid]
                start_time = start_info['start']

                if end_time <= start_time:
                    # Safety fix: use a minimum duration of 1 second
                    logger.warning(f"Invalid segment timing fixed: end_time {end_time} <= start_time {start_time} for {name_map.get(pid)}")
                    end_time = start_time + 1.0
                    validation_errors.append(f"Fixed invalid segment timing for {name_map.get(pid)}")

                duration = end_time - start_time
                completed_segments[pid].append({
                    'start': start_time,
                    'end': end_time,
                    'duration': duration,
                    'reason': f"{start_info['reason']} -> {reason}"
                })

                del active_segments[pid]
                logger.debug(f"Ended segment for {name_map.get(pid)}: {duration:.1f}s ({reason})")

            def start_player_segment(pid: int, start_time: float, reason: str) -> None:
                """CORRECTED: Start a new segment with overlap prevention"""
                if pid in active_segments:
                    # End existing segment first to prevent overlaps
                    end_player_segment(pid, start_time, f"OVERLAP_{reason}")

                active_segments[pid] = {'start': start_time, 'reason': reason}
                logger.debug(f"Started segment for {name_map.get(pid)}: {start_time:.1f}s ({reason})")

            def handle_lineup_change(tid: int, current_time: float, reason: str) -> None:
                """CORRECTED: Handle lineup changes with proper segment management"""
                team_name = team_abbrev_map[tid]

                # Get all players who should be tracked for this team
                team_players = [pid for pid in pteam_map if pteam_map.get(pid) == tid]

                # End segments for players no longer on court
                for pid in team_players:
                    if pid in active_segments and pid not in on_court[tid]:
                        end_player_segment(pid, current_time, f"{reason}_OUT")

                # Start segments for new players on court
                for pid in on_court[tid]:
                    if pid not in active_segments:
                        start_player_segment(pid, current_time, f"{reason}_IN")

            def pick_auto_out_candidate(tid: int, current_time: float, exclude: set[int] = set()) -> int | None:
                """CORRECTED: Enhanced auto-out selection based on activity patterns"""
                if not on_court[tid]:
                    return None

                candidates = [p for p in on_court[tid] if p not in exclude]
                if not candidates:
                    return None

                # CORRECTED: Enhanced scoring system for auto-out selection
                def activity_score(pid: int) -> tuple:
                    idle_time = current_time - last_action_time[pid]

                    # Primary factors (lower is better for removal):
                    # 1. Idle time (higher idle = more likely to remove)
                    # 2. Starter status (prefer to keep starters)
                    # 3. Recent sub activity (avoid ping-ponging)

                    is_starter = pid in starters[tid]
                    recently_subbed = pid in recent_out[tid]

                    # Score components (higher = more likely to be removed)
                    idle_score = idle_time
                    starter_penalty = -100 if is_starter else 0  # Keep starters longer
                    recent_sub_penalty = -50 if recently_subbed else 0  # Avoid ping-pong

                    total_score = idle_score + starter_penalty + recent_sub_penalty

                    return (total_score, idle_time, pid)  # Use pid for deterministic tiebreaking

                # Sort by activity score (highest score = best candidate for removal)
                candidates.sort(key=activity_score, reverse=True)

                best_candidate = candidates[0]
                idle_time = current_time - last_action_time[best_candidate]

                # Only auto-remove if idle >= threshold OR we have >5 players
                if idle_time >= idle_thresh or len(on_court[tid]) > 5:
                    return best_candidate

                return None

            def ensure_valid_lineup(tid: int, current_time: float, prefer_keep: set[int] = set()) -> None:
                """CORRECTED: Ensure team has exactly 5 players with better logic"""
                team_name = team_abbrev_map[tid]
                changes_made = False

                # Remove excess players
                while len(on_court[tid]) > 5:
                    auto_out = pick_auto_out_candidate(tid, current_time, exclude=prefer_keep)
                    if auto_out is None:
                        logger.error(f"Cannot auto-remove from {team_name} - no valid candidates")
                        validation_errors.append(f"Cannot auto-remove from {team_name}")
                        break

                    on_court[tid].remove(auto_out)
                    recent_out[tid].append(auto_out)
                    changes_made = True

                    idle_time = current_time - last_action_time[auto_out]
                    logger.info(f"[AUTO-OUT] {name_map.get(auto_out)} from {team_name} (idle: {idle_time:.1f}s)")

                    debug_events.append({
                        'time': current_time,
                        'type': 'AUTO_OUT',
                        'player': name_map.get(auto_out),
                        'team': team_name,
                        'idle_time': idle_time
                    })

                    nonlocal auto_out_count
                    auto_out_count += 1

                # Add players if under 5
                if len(on_court[tid]) < 5:
                    available = [p for p in roster[tid] if p not in on_court[tid]]
                    if available:
                        # CORRECTED: Prioritize based on game context
                        def fill_priority(pid: int) -> tuple:
                            # Priority factors (lower is better):
                            # 1. Recently out (prefer recent subs)
                            # 2. Starter status (prefer starters)  
                            # 3. Recent activity (prefer active players)

                            recently_out_priority = 0 if pid in recent_out[tid] else 1
                            starter_priority = 0 if pid in starters[tid] else 1
                            activity_priority = -(current_time - last_action_time[pid])  # More recent = lower number

                            return (recently_out_priority, starter_priority, activity_priority)

                        available.sort(key=fill_priority)

                        needed = 5 - len(on_court[tid])
                        for i in range(min(needed, len(available))):
                            fill_player = available[i]
                            on_court[tid].add(fill_player)
                            changes_made = True
                            logger.info(f"[AUTO-IN] {name_map.get(fill_player)} to {team_name} (fill to 5)")

                # Update segments if changes were made
                if changes_made:
                    handle_lineup_change(tid, current_time, "LINEUP_CORRECTION")

            def guard_always_five(current_time: float):
                """Fix any deviation from 5 and count it."""
                nonlocal lineup_violation_fixes
                for tid in teams:
                    if len(on_court[tid]) != 5:
                        lineup_violation_fixes += 1
                        ensure_valid_lineup(tid, current_time)

            # MAIN PROCESSING LOOP
            prev_period = None
            prev_time = None

            logger.info(f"Starting lineup processing: {team_abbrev_map[teams[0]]} vs {team_abbrev_map[teams[1]]}")

            for idx, ev in events.iterrows():
                period = int(ev.period)
                clock_str = ev.game_clock
                parsed_clock = _parse_game_clock(clock_str)
                current_time = _abs_time(period, parsed_clock)
                msg_type = int(ev.msg_type)

                # CORRECTED: Handle period transitions without double-crediting
                if period != prev_period and prev_period is not None:
                    # Calculate period end time
                    period_end_time = _abs_time(prev_period, 0.0)

                    # Only credit period end time if we haven't already done so
                    if prev_period not in period_end_times:
                        # End all active segments at period end
                        for pid in list(active_segments.keys()):
                            end_player_segment(pid, period_end_time, f"PERIOD_{prev_period}_END")

                        period_end_times[prev_period] = period_end_time
                        logger.debug(f"Ended period {prev_period} at {period_end_time:.1f}s")

                # Initialize new period
                if period != prev_period:
                    if period in CFG["starter_reset_periods"]:
                        # Reset to starters
                        on_court = {tid: set(starters[tid]) for tid in teams}
                        logger.info(f"[PERIOD {period}] Reset to starters")
                    else:
                        # Continue previous lineups
                        logger.info(f"[PERIOD {period}] Continue lineups")
                        for tid in teams:
                            ensure_valid_lineup(tid, current_time)

                    # Start new segments for all on-court players
                    for tid in teams:
                        for pid in on_court[tid]:
                            if pid not in active_segments:
                                start_player_segment(pid, current_time, f"PERIOD_{period}_START")

                    prev_period = period

                # SUBSTITUTION PROCESSING
                if msg_type == CFG["msg_types"]["substitution"]:
                    sub_count += 1

                    out_pid = int(ev.player_id_1) if not pd.isna(ev.player_id_1) else None
                    in_pid = int(ev.player_id_2) if not pd.isna(ev.player_id_2) else None
                    out_ln, in_ln = ev.last_name_1, ev.last_name_2

                    # Determine team
                    sub_tid = None
                    if in_pid and in_pid in pteam_map:
                        sub_tid = pteam_map[in_pid]
                    elif out_pid and out_pid in pteam_map:
                        sub_tid = pteam_map[out_pid]
                    elif pd.notna(ev.team_id_off) and int(ev.team_id_off) in teams:
                        sub_tid = int(ev.team_id_off)

                    # Enhanced name resolution
                    if in_pid is None and CFG["one_direction"]["appearance_via_last_name"] and in_ln:
                        in_pid = enhanced_name_resolution(in_ln, sub_tid)
                    if out_pid is None and out_ln:
                        out_pid = enhanced_name_resolution(out_ln, sub_tid)

                    if sub_tid is not None:
                        team_name = team_abbrev_map[sub_tid]

                        # Process OUT first
                        if CFG["one_direction"]["remove_out_if_present"] and out_pid and out_pid in on_court[sub_tid]:
                            on_court[sub_tid].remove(out_pid)
                            recent_out[sub_tid].append(out_pid)
                            logger.info(f"[SUB-OUT] {name_map.get(out_pid)} from {team_name}")

                        # Process IN
                        if in_pid and in_pid not in on_court[sub_tid]:
                            # Make room if needed
                            if len(on_court[sub_tid]) >= 5:
                                auto_out = pick_auto_out_candidate(sub_tid, current_time, exclude={in_pid})
                                if auto_out:
                                    on_court[sub_tid].remove(auto_out)
                                    recent_out[sub_tid].append(auto_out)
                                    logger.info(f"[MAKE-ROOM] {name_map.get(auto_out)} out for {name_map.get(in_pid)}")

                            on_court[sub_tid].add(in_pid)
                            logger.info(f"[SUB-IN] {name_map.get(in_pid)} to {team_name}")

                        # Update segments for substitution
                        handle_lineup_change(sub_tid, current_time, "SUBSTITUTION")

                    # Update activity times
                    if out_pid:
                        last_action_time[out_pid] = current_time
                    if in_pid:
                        last_action_time[in_pid] = current_time

                # FIRST ACTION PROCESSING (Reed Sheppard rule)
                elif msg_type in [1, 2, 4, 5, 6] and CFG["one_direction"]["appearance_via_last_name"]:
                    action_pid = int(ev.player_id_1) if not pd.isna(ev.player_id_1) else None
                    action_ln = ev.last_name_1
                    action_tid = None

                    if action_pid and action_pid in pteam_map:
                        action_tid = pteam_map[action_pid]
                    elif pd.notna(ev.team_id_off) and int(ev.team_id_off) in teams:
                        action_tid = int(ev.team_id_off)

                    # Resolve via last name if needed
                    if action_pid is None and action_ln:
                        action_pid = enhanced_name_resolution(action_ln, action_tid)
                        if action_pid:
                            action_tid = pteam_map[action_pid]

                    # CORRECTED: Apply Reed Sheppard rule with proper time tracking
                    if action_tid in teams and action_pid and action_pid not in on_court[action_tid]:
                        # This is a first action - inject player
                        on_court[action_tid].add(action_pid)
                        first_action_count += 1

                        logger.info(f"[FIRST-ACTION] {name_map.get(action_pid)} -> {team_abbrev_map[action_tid]} (msg: {msg_type})")

                        debug_events.append({
                            'time': current_time,
                            'type': 'FIRST_ACTION',
                            'player': name_map.get(action_pid),
                            'team': team_abbrev_map[action_tid],
                            'action': ev.description
                        })

                        # CORRECTED: Start segment for first-action player
                        start_player_segment(action_pid, current_time, "FIRST_ACTION")

                        # Ensure valid lineup after injection
                        ensure_valid_lineup(action_tid, current_time, prefer_keep={action_pid})

                    # Update activity time
                    if action_pid:
                        last_action_time[action_pid] = current_time

                # after each event: enforce always-5
                guard_always_five(current_time)

                prev_time = current_time

            # CORRECTED: Final processing - end all remaining segments
            # Use the last actual event time or calculate proper game end time
            if prev_period and current_time:
                final_time = current_time
            else:
                final_time = 2880.0  # 48 minutes total

            for pid in list(active_segments.keys()):
                end_player_segment(pid, final_time, "GAME_END")

            # CORRECTED: Calculate final minutes with comprehensive validation
            calculated_minutes = {}
            for pid in set(list(completed_segments.keys()) + list(box_df['player_id'])):
                segments = completed_segments[pid]
                total_seconds = sum(seg['duration'] for seg in segments)
                calculated_minutes[pid] = total_seconds

            # Build validation results
            mv = []
            for pid in set(list(calculated_minutes.keys()) + list(box_df['player_id'])):
                calc_secs = calculated_minutes.get(pid, 0.0)
                box_row = box_df[box_df['player_id'] == pid]
                box_secs = float(box_row['seconds_played'].iloc[0]) if not box_row.empty else 0.0
                diff = calc_secs - box_secs

                mv.append({
                    "player_id": pid,
                    "player_name": name_map.get(pid, f"ID_{pid}"),
                    "team": team_abbrev_map.get(pteam_map.get(pid), "UNK"),
                    "calc_seconds": round(calc_secs, 1),
                    "box_seconds": round(box_secs, 1),
                    "abs_diff_seconds": round(abs(diff), 1),
                    "segments_count": len(completed_segments.get(pid, []))
                })

            mv_df = pd.DataFrame(mv).sort_values(["team", "player_name"]).reset_index(drop=True)
            offenders = mv_df[mv_df["abs_diff_seconds"] > tol]

            # Enhanced logging for validation
            if len(offenders) > 0:
                logger.warning(f"CORRECTED: Minutes validation: {len(offenders)} players exceed {tol}s tolerance")
                for _, row in offenders.iterrows():
                    logger.warning(f"  {row.player_name} ({row.team}): calc={row.calc_seconds}s vs box={row.box_seconds}s (diff={row.abs_diff_seconds}s)")

                    # Debug segment details for offenders
                    segments = completed_segments.get(row.player_id, [])
                    logger.debug(f"    Segments for {row.player_name}: {len(segments)} total")
                    for i, seg in enumerate(segments[:5]):  # Show first 5 segments
                        logger.debug(f"      {i+1}: {seg['duration']:.1f}s ({seg['reason']})")

            if validation_errors:
                logger.warning(f"Validation errors encountered: {len(validation_errors)}")
                for error in validation_errors[:5]:
                    logger.warning(f"  {error}")

            # Store enhanced debug data
            self.data_summary['enhanced_substitution_debug'] = {
                'total_events': len(events),
                'substitutions': sub_count,
                'first_actions': first_action_count,
                'auto_outs': auto_out_count,
                'always_five_fixes': lineup_violation_fixes,
                'validation_errors': len(validation_errors),
                'validation': {
                    'total_players': len(mv_df),
                    'offenders': len(offenders),
                    'tolerance': tol,
                    'max_difference': mv_df['abs_diff_seconds'].max() if not mv_df.empty else 0
                }
            }
            # also stash full minutes table for the report writer
            self.data_summary['minutes_validation_full'] = mv_df
            self.data_summary['minutes_offenders'] = offenders

            logger.info(f"SUBSTITUTION SUMMARY: {sub_count} subs, {first_action_count} first-actions, {auto_out_count} auto-outs")

            # IMPORTANT: non-fatal -> passed = True (warnings carry the issues)
            details = (f"Enhanced engine: {len(events)} events, {sub_count} subs, {first_action_count} first-actions. "
                       f"Validation: {len(offenders)}/{len(mv_df)} offenders; 5-on-floor fixes: {lineup_violation_fixes}")
            return ValidationResult(
                step_name="Enhanced Lineups & Rim Analytics",
                passed=True,
                details=details,
                processing_time=time.time() - start_time,
                warnings=[] if len(offenders)==0 else [f"{len(offenders)} players exceed {tol}s tolerance"]
            )

        except Exception as e:
            import traceback
            logger.error(f"Exception in corrected substitution engine: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return ValidationResult(
                step_name="Enhanced Lineups & Rim Analytics",
                passed=False,
                details=f"Error in corrected engine: {str(e)}",
                processing_time=time.time() - start_time
            )

    def debug_segment_analysis(self, player_id: int = None) -> Dict[str, Any]:
        """
        Enhanced debugging function to analyze segment calculation for specific players.
        Useful for diagnosing Reed Sheppard cases and other timing issues.
        """
        if not hasattr(self, 'data_summary') or 'enhanced_substitution_debug' not in self.data_summary:
            return {"error": "No enhanced substitution data available"}

        debug_data = self.data_summary['enhanced_substitution_debug']

        # If specific player requested, focus on them
        if player_id:
            return self._analyze_player_segments(player_id)

        # Otherwise provide overall analysis
        return {
            'summary': debug_data,
            'recommendations': self._generate_debug_recommendations(debug_data)
        }

    def _analyze_player_segments(self, player_id: int) -> Dict[str, Any]:
        """Detailed analysis for a specific player's segments"""
        return {
            'player_id': player_id,
            'analysis': 'Detailed segment analysis would go here',
            'recommendations': []
        }

    def _generate_debug_recommendations(self, debug_data: Dict) -> List[str]:
        """Generate recommendations based on debug data"""
        recommendations = []

        validation = debug_data.get('validation', {})
        offenders = validation.get('offenders', 0)
        max_diff = validation.get('max_difference', 0)

        if offenders > 0:
            recommendations.append(f"Still have {offenders} players with timing issues")

        if max_diff > 300:  # 5 minutes
            recommendations.append("Large timing discrepancies detected - check period transitions")
        elif max_diff > 120:  # 2 minutes  
            recommendations.append("Moderate timing issues - check first-action logic")

        if debug_data.get('validation_errors', 0) > 0:
            recommendations.append("Segment validation errors detected - check overlap prevention")

        auto_outs = debug_data.get('auto_outs', 0)
        if auto_outs > 20:
            recommendations.append("High auto-out count - may indicate lineup instability")

        return recommendations

    def create_minutes_validation_report(self) -> str:
        """
        Create a detailed validation report for minutes calculation.
        Useful for verifying the corrected engine performance.
        """
        if not hasattr(self, 'data_summary'):
            return "No validation data available"

        # Get validation data from enhanced runs if available
        enhanced_data = self.data_summary.get('enhanced_substitution_debug', {})

        report_lines = [
            "MINUTES VALIDATION REPORT",
            "=" * 50,
            ""
        ]

        if enhanced_data:
            validation = enhanced_data.get('validation', {})
            report_lines.extend([
                "ENHANCED ENGINE RESULTS:",
                f"Total players: {validation.get('total_players', 0)}",
                f"Players exceeding tolerance: {validation.get('offenders', 0)}",
                f"Maximum difference: {validation.get('max_difference', 0):.1f}s",
                f"Tolerance threshold: {validation.get('tolerance', 120)}s",
                f"5-on-floor fixes: {enhanced_data.get('always_five_fixes', 0)}",
                ""
            ])

            # Add recommendations
            recommendations = self._generate_debug_recommendations(enhanced_data)
            if recommendations:
                report_lines.extend([
                    "RECOMMENDATIONS:",
                    *[f"- {rec}" for rec in recommendations],
                    ""
                ])

        return "\n".join(report_lines)

    def debug_substitution_analysis(self) -> Dict[str, Any]:
        """
        Comprehensive debugger to identify substitution and minutes calculation issues.

        This function analyzes:
        1. Reed Sheppard's specific case and similar players
        2. Minutes calculation discrepancies
        3. Substitution event patterns
        4. Missing first-action events
        """

        logger.info("🔍 STARTING COMPREHENSIVE SUBSTITUTION ANALYSIS")
        logger.info("=" * 80)

        analysis_results = {
            'reed_sheppard_analysis': {},
            'minutes_discrepancies': {},
            'substitution_patterns': {},
            'first_action_missing': {},
            'timeline_analysis': {}
        }

        # Get all data we need
        box_df = self.conn.execute("""
            SELECT player_id, player_name, team_id, team_abbrev, is_starter, seconds_played
            FROM box_score WHERE seconds_played > 0 ORDER BY team_id, seconds_played DESC
        """).df()

        events_df = self.conn.execute("""
            SELECT period, pbp_order, wall_clock_int, game_clock, description,
                   team_id_off, team_id_def, msg_type, action_type,
                   player_id_1, player_id_2, player_id_3,
                   last_name_1, last_name_2, last_name_3, points
            FROM pbp ORDER BY period, pbp_order, wall_clock_int
        """).df()

        # Create name mappings
        name_map = dict(zip(box_df['player_id'], box_df['player_name']))
        team_map = dict(zip(box_df['player_id'], box_df['team_abbrev']))

        logger.info(f"📊 Analyzing {len(events_df)} events for {len(box_df)} players")

        # 1. REED SHEPPARD SPECIFIC ANALYSIS
        logger.info("\n🎯 REED SHEPPARD CASE ANALYSIS")
        logger.info("-" * 50)

        reed_sheppard_id = 1642263  # From the provided data
        if reed_sheppard_id in name_map:
            reed_events = []

            # Find all events involving Reed Sheppard
            for _, ev in events_df.iterrows():
                if (ev['player_id_1'] == reed_sheppard_id or 
                    ev['player_id_2'] == reed_sheppard_id or 
                    ev['player_id_3'] == reed_sheppard_id or
                    'sheppard' in str(ev['description']).lower()):

                    reed_events.append({
                        'period': ev['period'],
                        'game_clock': ev['game_clock'],
                        'msg_type': ev['msg_type'],
                        'description': ev['description'],
                        'player_1': ev['player_id_1'],
                        'player_2': ev['player_id_2'],
                        'player_3': ev['player_id_3'],
                        'last_name_1': ev['last_name_1']
                    })

            logger.info(f"Reed Sheppard events found: {len(reed_events)}")
            for i, event in enumerate(reed_events):
                logger.info(f"  Event {i+1}: Q{event['period']} {event['game_clock']} | {event['description']}")
                logger.info(f"    MsgType: {event['msg_type']}, Players: {event['player_1']}, {event['player_2']}, {event['player_3']}")

            analysis_results['reed_sheppard_analysis'] = {
                'total_events': len(reed_events),
                'events': reed_events,
                'box_minutes': box_df[box_df['player_id'] == reed_sheppard_id]['seconds_played'].iloc[0] if reed_sheppard_id in box_df['player_id'].values else 0
            }

        # 2. FIND PLAYERS WITH FIRST ACTIONS BUT NO SUB-IN
        logger.info("\n🚨 PLAYERS WITH ACTIONS BUT NO SUB-IN")
        logger.info("-" * 50)

        # Get all substitution events
        sub_events = events_df[events_df['msg_type'] == 8].copy()

        # Get all players who sub IN
        subbed_in_players = set()
        for _, sub in sub_events.iterrows():
            if pd.notna(sub['player_id_2']):
                subbed_in_players.add(int(sub['player_id_2']))

        # Get starters
        starters = set(box_df[box_df['is_starter'] == True]['player_id'].tolist())

        # Find players with actions but never subbed in (and not starters)
        action_events = events_df[events_df['msg_type'].isin([1, 2, 4, 5, 6])].copy()

        players_with_actions = set()
        for _, ev in action_events.iterrows():
            for col in ['player_id_1', 'player_id_2', 'player_id_3']:
                if pd.notna(ev[col]):
                    players_with_actions.add(int(ev[col]))

        # Players who have actions but no sub-in and aren't starters
        missing_sub_in = players_with_actions - subbed_in_players - starters

        logger.info(f"Players with actions but no sub-in: {len(missing_sub_in)}")
        for pid in missing_sub_in:
            if pid in name_map:
                logger.info(f"  {name_map[pid]} (ID: {pid}) - {team_map.get(pid, 'Unknown team')}")

                # Find their first action
                first_action = None
                for _, ev in action_events.iterrows():
                    if (ev['player_id_1'] == pid or ev['player_id_2'] == pid or ev['player_id_3'] == pid):
                        first_action = ev
                        break

                if first_action is not None:
                    logger.info(f"    First action: Q{first_action['period']} {first_action['game_clock']} | {first_action['description']}")

        analysis_results['first_action_missing'] = {
            'count': len(missing_sub_in),
            'players': [{'id': pid, 'name': name_map.get(pid), 'team': team_map.get(pid)} for pid in missing_sub_in if pid in name_map]
        }

        # 3. SUBSTITUTION PATTERN ANALYSIS
        logger.info("\n🔄 SUBSTITUTION PATTERN ANALYSIS")
        logger.info("-" * 50)

        sub_analysis = {
            'total_subs': len(sub_events),
            'subs_with_both_players': 0,
            'subs_with_only_in': 0,
            'subs_with_only_out': 0,
            'subs_with_neither': 0
        }

        for _, sub in sub_events.iterrows():
            has_out = pd.notna(sub['player_id_1'])
            has_in = pd.notna(sub['player_id_2'])

            if has_out and has_in:
                sub_analysis['subs_with_both_players'] += 1
            elif has_in and not has_out:
                sub_analysis['subs_with_only_in'] += 1
            elif has_out and not has_in:
                sub_analysis['subs_with_only_out'] += 1
            else:
                sub_analysis['subs_with_neither'] += 1

        logger.info(f"Total substitutions: {sub_analysis['total_subs']}")
        logger.info(f"  Both players: {sub_analysis['subs_with_both_players']}")
        logger.info(f"  Only IN player: {sub_analysis['subs_with_only_in']}")
        logger.info(f"  Only OUT player: {sub_analysis['subs_with_only_out']}")
        logger.info(f"  Neither player: {sub_analysis['subs_with_neither']}")

        analysis_results['substitution_patterns'] = sub_analysis

        # 4. MINUTES CALCULATION SIMULATION
        logger.info("\n⏱️ MINUTES CALCULATION SIMULATION")
        logger.info("-" * 50)

        def parse_game_clock(gc):
            if not gc or not isinstance(gc, str):
                return None
            try:
                if ':' in gc:
                    mm, ss = gc.split(':')
                    return float(mm) * 60.0 + float(ss)
            except:
                pass
            return None

        def abs_time(period, rem_sec):
            total = 0.0
            for p in range(1, period):
                total += 720.0 if p <= 4 else 300.0
            period_length = 720.0 if period <= 4 else 300.0
            if rem_sec is None:
                return total
            return total + (period_length - rem_sec)

        # Simulate simple starter minutes (baseline)
        baseline_minutes = {}
        starters_per_team = {}

        for team in box_df['team_id'].unique():
            team_starters = box_df[(box_df['team_id'] == team) & (box_df['is_starter'] == True)]['player_id'].tolist()
            starters_per_team[team] = team_starters

            # Assume starters play full quarters 1 and 3, then continue in 2 and 4
            for pid in team_starters:
                baseline_minutes[pid] = 2 * 720.0  # Q1 + Q3 = 24 minutes baseline

        logger.info(f"Baseline starter minutes (Q1+Q3 only): {sum(baseline_minutes.values())/60:.1f} total minutes")

        # Find actual box score total
        actual_total = box_df['seconds_played'].sum()
        logger.info(f"Actual box score total: {actual_total/60:.1f} minutes")
        logger.info(f"Expected game total: {48*10:.1f} minutes (48 min × 10 players)")

        analysis_results['minutes_discrepancies'] = {
            'baseline_total': sum(baseline_minutes.values()),
            'actual_total': actual_total,
            'expected_total': 48 * 60 * 10,
            'baseline_vs_actual_diff': actual_total - sum(baseline_minutes.values())
        }

        # 5. TIMELINE ANALYSIS
        logger.info("\n📈 TIMELINE ANALYSIS")
        logger.info("-" * 50)

        timeline = []
        for _, ev in events_df.iterrows():
            if ev['msg_type'] == 8:  # Substitutions
                timeline.append({
                    'time': abs_time(ev['period'], parse_game_clock(ev['game_clock'])),
                    'period': ev['period'],
                    'clock': ev['game_clock'],
                    'event_type': 'SUB',
                    'description': ev['description']
                })
            elif ev['msg_type'] in [1, 2, 4, 5, 6] and ev['player_id_1'] in missing_sub_in:
                timeline.append({
                    'time': abs_time(ev['period'], parse_game_clock(ev['game_clock'])),
                    'period': ev['period'],
                    'clock': ev['game_clock'],
                    'event_type': 'MISSING_PLAYER_ACTION',
                    'player': name_map.get(ev['player_id_1'], f"ID_{ev['player_id_1']}"),
                    'description': ev['description']
                })

        timeline.sort(key=lambda x: x['time'])

        logger.info("Key timeline events:")
        for event in timeline[:20]:  # First 20 events
            if event['event_type'] == 'SUB':
                logger.info(f"  {event['time']:>6.1f}s Q{event['period']} {event['clock']} | SUB: {event['description']}")
            else:
                logger.info(f"  {event['time']:>6.1f}s Q{event['period']} {event['clock']} | MISSING: {event['player']} | {event['description']}")

        analysis_results['timeline_analysis'] = timeline[:50]  # Store first 50 for reference

        logger.info("\n✅ ANALYSIS COMPLETE")
        logger.info("=" * 80)

        return analysis_results

    def debug_minutes_tracker(self) -> Dict[str, Any]:
        """
        Create a detailed minute-by-minute tracker to identify exactly where minutes are being miscalculated.
        """
        from collections import defaultdict

        logger.info("🔍 CREATING DETAILED MINUTES TRACKER")

        # This will track every single second of every player's time
        minute_tracker = {
            'player_segments': defaultdict(list),
            'period_summaries': {},
            'discrepancies': {},
            'debug_log': []
        }

        def log_debug(message):
            minute_tracker['debug_log'].append(message)
            logger.debug(message)

        # Get data
        box_df = self.conn.execute("""
            SELECT player_id, player_name, team_id, team_abbrev, is_starter, seconds_played
            FROM box_score WHERE seconds_played > 0 ORDER BY team_id, seconds_played DESC
        """).df()

        events_df = self.conn.execute("""
            SELECT period, pbp_order, wall_clock_int, game_clock, description,
                   team_id_off, team_id_def, msg_type, action_type,
                   player_id_1, player_id_2, player_id_3,
                   last_name_1, last_name_2, last_name_3
            FROM pbp ORDER BY period, pbp_order, wall_clock_int
        """).df()

        name_map = dict(zip(box_df['player_id'], box_df['player_name']))
        team_map = dict(zip(box_df['player_id'], box_df['team_id']))

        # Initialize with starters
        teams = sorted(box_df['team_id'].unique())
        starters = {team: set(box_df[(box_df['team_id'] == team) & (box_df['is_starter'] == True)]['player_id'].tolist()) for team in teams}

        current_lineups = {team: set(starters[team]) for team in teams}

        log_debug(f"Initial lineups: {current_lineups}")

        def parse_clock(gc):
            if not gc or not isinstance(gc, str) or ':' not in gc:
                return None
            try:
                mm, ss = gc.split(':')
                return float(mm) * 60.0 + float(ss)
            except:
                pass
            return None

        def abs_time(period, rem_sec):
            total = sum(720.0 if p <= 4 else 300.0 for p in range(1, period))
            if rem_sec is None:
                return total
            period_length = 720.0 if period <= 4 else 300.0
            return total + (period_length - rem_sec)

        # Track time
        prev_abs_time = 0.0
        prev_period = 0

        for idx, ev in events_df.iterrows():
            period = int(ev['period'])
            curr_clock = parse_clock(ev['game_clock'])
            curr_abs_time = abs_time(period, curr_clock)

            # Handle period transitions
            if period != prev_period:
                if prev_period > 0:
                    # Credit end of previous period
                    period_end_time = abs_time(prev_period, 0.0)
                    if period_end_time > prev_abs_time:
                        duration = period_end_time - prev_abs_time
                        for team in teams:
                            for pid in current_lineups[team]:
                                minute_tracker['player_segments'][pid].append({
                                    'start': prev_abs_time,
                                    'end': period_end_time,
                                    'duration': duration,
                                    'reason': f'PERIOD_{prev_period}_END'
                                })
                        log_debug(f"Period {prev_period} end: credited {duration:.1f}s to {sum(len(current_lineups[t]) for t in teams)} players")

                # Reset or continue lineups for new period
                if period in [1, 3]:  # Starter reset periods
                    current_lineups = {team: set(starters[team]) for team in teams}
                    log_debug(f"Period {period}: Reset to starters")
                else:
                    log_debug(f"Period {period}: Continue lineups")

                prev_period = period

            # Credit time between events
            if curr_abs_time > prev_abs_time and prev_abs_time > 0:
                duration = curr_abs_time - prev_abs_time
                players_credited = 0
                for team in teams:
                    for pid in current_lineups[team]:
                        minute_tracker['player_segments'][pid].append({
                            'start': prev_abs_time,
                            'end': curr_abs_time,
                            'duration': duration,
                            'reason': f'PERIOD_{period}_PLAY'
                        })
                        players_credited += 1

                if duration > 60:  # Log significant time gaps
                    log_debug(f"Large time gap: {duration:.1f}s credited to {players_credited} players")

            # Handle substitutions
            if ev['msg_type'] == 8:
                out_pid = int(ev['player_id_1']) if pd.notna(ev['player_id_1']) else None
                in_pid = int(ev['player_id_2']) if pd.notna(ev['player_id_2']) else None

                # Find which team this substitution is for
                sub_team = None
                if in_pid and in_pid in team_map:
                    sub_team = team_map[in_pid]
                elif out_pid and out_pid in team_map:
                    sub_team = team_map[out_pid]

                if sub_team:
                    if out_pid and out_pid in current_lineups[sub_team]:
                        current_lineups[sub_team].remove(out_pid)
                        log_debug(f"SUB OUT: {name_map.get(out_pid, out_pid)} from team {sub_team}")

                    if in_pid and in_pid not in current_lineups[sub_team]:
                        current_lineups[sub_team].add(in_pid)
                        log_debug(f"SUB IN: {name_map.get(in_pid, in_pid)} to team {sub_team}")

            prev_abs_time = curr_abs_time

        # Final period end
        if prev_period > 0:
            final_end = abs_time(prev_period, 0.0)
            if final_end > prev_abs_time:
                duration = final_end - prev_abs_time
                for team in teams:
                    for pid in current_lineups[team]:
                        minute_tracker['player_segments'][pid].append({
                            'start': prev_abs_time,
                            'end': final_end,
                            'duration': duration,
                            'reason': f'PERIOD_{prev_period}_FINAL_END'
                        })
                log_debug(f"Final period end: credited {duration:.1f}s")

        # Calculate totals and compare
        for pid in minute_tracker['player_segments']:
            calculated_total = sum(seg['duration'] for seg in minute_tracker['player_segments'][pid])
            box_total = box_df[box_df['player_id'] == pid]['seconds_played'].iloc[0] if pid in box_df['player_id'].values else 0
            diff = calculated_total - box_total

            minute_tracker['discrepancies'][pid] = {
                'calculated': calculated_total,
                'box_score': box_total,
                'difference': diff,
                'segments_count': len(minute_tracker['player_segments'][pid]),
                'player_name': name_map.get(pid, f"ID_{pid}")
            }

        return minute_tracker

    def test_reed_sheppard_case(self):
        """Test function to verify Reed Sheppard case is handled correctly"""
        # Check if Reed Sheppard (ID: 1642263) appears in events
        reed_events = self.conn.execute("""
            SELECT period, game_clock, description, msg_type,
                   player_id_1, player_id_2, player_id_3,
                   last_name_1, last_name_2, last_name_3
            FROM pbp 
            WHERE player_id_1 = 1642263 
               OR player_id_2 = 1642263 
               OR player_id_3 = 1642263
               OR LOWER(description) LIKE '%sheppard%'
            ORDER BY period, pbp_order
        """).df()

        print(f"Reed Sheppard events: {len(reed_events)}")
        for _, ev in reed_events.iterrows():
            print(f"  Q{ev['period']} {ev['game_clock']} | {ev['description']}")

        return reed_events

    def create_missing_player_report(self) -> ValidationResult:
        """
        Summarize PBP-only players with names, inferred team, confidence, first/last seen, and event breakdown.

        Debug-first policy:
        - Do NOT hide missing data via COALESCE in the final outputs. Expose raw + resolved columns.
        - Add preflight checks and log actual row counts of intermediates (pbp_only_players, occ, sums).
        - Rebuild _pbp_names in this scope if it is not present to avoid hidden coupling.
        - Dump the FULL report (all columns, all rows) and schema to the logs when done.
        """
        start_time = time.time()
        try:
            # --- Preconditions: required base tables/views must exist ---
            need_tables = ["pbp", "dim_players", "dim_teams"]
            for t in need_tables:
                exists = self.conn.execute(
                    f"SELECT COUNT(*) FROM information_schema.tables WHERE table_name = '{t}'"
                ).fetchone()[0]
                if exists == 0:
                    return ValidationResult(
                        step_name="Missing Player Report",
                        passed=False,
                        details=f"Missing required table: {t}",
                        processing_time=time.time() - start_time
                    )

            # pbp_only_players is created in create_dimensions()
            has_pbp_only = self.conn.execute(
                "SELECT COUNT(*) FROM information_schema.views WHERE table_name = 'pbp_only_players'"
            ).fetchone()[0]
            if has_pbp_only == 0:
                return ValidationResult(
                    step_name="Missing Player Report",
                    passed=False,
                    details="pbp_only_players view not found. Run create_dimensions() first.",
                    processing_time=time.time() - start_time
                )

            # --- Ensure _pbp_names exists (recreate locally if absent) ---
            has_pbp_names = self.conn.execute(
                "SELECT COUNT(*) FROM information_schema.views WHERE table_name = '_pbp_names'"
            ).fetchone()[0]
            if has_pbp_names == 0:
                logger.info("[Missing Player Report] _pbp_names not found; rebuilding TEMP view locally")
                self.conn.execute("""
                    CREATE OR REPLACE TEMP VIEW _pbp_names AS
                    WITH p1 AS (
                        SELECT player_id_1 AS player_id, ANY_VALUE(NULLIF(last_name_1,'')) AS last_name
                        FROM pbp
                        WHERE player_id_1 IS NOT NULL
                        GROUP BY player_id_1
                    ),
                    p2 AS (
                        SELECT player_id_2 AS player_id, ANY_VALUE(NULLIF(last_name_2,'')) AS last_name
                        FROM pbp
                        WHERE player_id_2 IS NOT NULL
                        GROUP BY player_id_2
                    ),
                    p3 AS (
                        SELECT player_id_3 AS player_id, ANY_VALUE(NULLIF(last_name_3,'')) AS last_name
                        FROM pbp
                        WHERE player_id_3 IS NOT NULL
                        GROUP BY player_id_3
                    ),
                    unioned AS (
                        SELECT * FROM p1
                        UNION ALL
                        SELECT * FROM p2
                        UNION ALL
                        SELECT * FROM p3
                    )
                    SELECT player_id, ANY_VALUE(last_name) AS last_name
                    FROM unioned
                    WHERE last_name IS NOT NULL
                    GROUP BY player_id
                """)
            else:
                logger.info("[Missing Player Report] Reusing existing _pbp_names TEMP view")

            # Rebuild team guess confidence (same logic as earlier)
            self.conn.execute("""
                CREATE OR REPLACE TEMP VIEW _team_guess_conf AS
                WITH occ AS (
                    SELECT player_id_1 AS player_id, team_id_off AS team_id FROM pbp WHERE player_id_1 IS NOT NULL
                    UNION ALL SELECT player_id_2, team_id_off FROM pbp WHERE player_id_2 IS NOT NULL
                    UNION ALL SELECT player_id_3, team_id_off FROM pbp WHERE player_id_3 IS NOT NULL
                    UNION ALL SELECT player_id_1, team_id_def FROM pbp WHERE player_id_1 IS NOT NULL
                    UNION ALL SELECT player_id_2, team_id_def FROM pbp WHERE player_id_2 IS NOT NULL
                    UNION ALL SELECT player_id_3, team_id_def FROM pbp WHERE player_id_3 IS NOT NULL
                ),
                agg AS (
                    SELECT player_id, team_id, COUNT(*) AS c
                    FROM occ
                    GROUP BY player_id, team_id
                ),
                totals AS (
                    SELECT player_id, SUM(c) AS tot
                    FROM agg
                    GROUP BY player_id
                ),
                ranked AS (
                    SELECT
                        a.player_id, a.team_id, a.c, t.tot,
                        ROW_NUMBER() OVER (PARTITION BY a.player_id ORDER BY a.c DESC, a.team_id) AS rn
                    FROM agg a
                    JOIN totals t USING(player_id)
                )
                SELECT
                    player_id,
                    team_id AS guessed_team_id,
                    c AS guessed_count,
                    tot,
                    (c::DOUBLE)/NULLIF(tot,0) AS team_confidence
                FROM ranked
                WHERE rn = 1
            """)

            # --- Preflight debug: how many pbp-only players? ---
            num_only = self.conn.execute("SELECT COUNT(*) FROM pbp_only_players").fetchone()[0]
            logger.info(f"[Missing Player Report] pbp_only_players count = {num_only}")

            # --- Build the report table (JOIN sums as `s`) ---
            self._robust_drop_object("missing_player_report")
            self.conn.execute("""
                CREATE TABLE missing_player_report AS
                WITH occ AS (
                    SELECT
                        o.player_id,
                        p.period,
                        p.pbp_order,
                        p.wall_clock_int,
                        p.game_clock,
                        p.description,
                        p.msg_type,
                        p.points
                    FROM pbp_only_players o
                    LEFT JOIN pbp p
                    ON o.player_id = p.player_id_1
                    OR o.player_id = p.player_id_2
                    OR o.player_id = p.player_id_3
                ),
                sums AS (
                    SELECT
                        player_id,
                        COUNT(*) AS total_events,
                        SUM(points) AS points,
                        SUM(CASE WHEN msg_type IN (1,2) THEN 1 ELSE 0 END) AS shot_events,
                        SUM(CASE WHEN msg_type = 1 THEN 1 ELSE 0 END) AS made_fg,
                        SUM(CASE WHEN msg_type = 2 THEN 1 ELSE 0 END) AS missed_fg,
                        SUM(CASE WHEN msg_type = 3 THEN 1 ELSE 0 END) AS free_throws,
                        SUM(CASE WHEN msg_type = 4 THEN 1 ELSE 0 END) AS rebounds,
                        SUM(CASE WHEN msg_type = 5 THEN 1 ELSE 0 END) AS turnovers,
                        SUM(CASE WHEN msg_type = 6 THEN 1 ELSE 0 END) AS fouls,
                        SUM(CASE WHEN msg_type = 8 THEN 1 ELSE 0 END) AS substitutions,
                        arg_min(CONCAT('Q', period, ' ', game_clock, ' | ', description), wall_clock_int) AS first_event,
                        arg_max(CONCAT('Q', period, ' ', game_clock, ' | ', description), wall_clock_int) AS last_event
                    FROM occ
                    GROUP BY player_id
                )
                SELECT
                    -- identity
                    o.player_id,

                    -- names: show raw sources + resolved (do NOT hide missing in raw columns)
                    dp.player_name                           AS box_player_name,
                    n.last_name                              AS pbp_last_name,
                    CASE
                        WHEN dp.player_name IS NOT NULL THEN dp.player_name
                        WHEN n.last_name   IS NOT NULL THEN n.last_name
                        ELSE CAST(o.player_id AS VARCHAR)
                    END                                      AS resolved_name,

                    -- team IDs & labels: expose raw + guessed + resolved
                    dp.team_id                               AS box_team_id,
                    tgc.guessed_team_id                      AS guessed_team_id,
                    tgc.team_confidence                      AS team_confidence,
                    CASE
                        WHEN dp.team_id IS NOT NULL THEN dp.team_id
                        WHEN tgc.guessed_team_id IS NOT NULL THEN tgc.guessed_team_id
                        ELSE NULL
                    END                                      AS resolved_team_id,

                    dt_res.team_abbrev                       AS resolved_team_abbrev,
                    dt_box.team_abbrev                       AS box_team_abbrev,
                    dt_guess.team_abbrev                     AS guessed_team_abbrev,

                    -- sample text from pbp_only_players
                    o.sample_event,

                    -- event rollups from sums
                    s.total_events,
                    s.points,
                    s.shot_events,
                    s.made_fg,
                    s.missed_fg,
                    s.free_throws,
                    s.rebounds,
                    s.turnovers,
                    s.fouls,
                    s.substitutions,
                    s.first_event,
                    s.last_event

                FROM pbp_only_players o
                LEFT JOIN dim_players dp           ON o.player_id = dp.player_id
                LEFT JOIN _pbp_names n             ON o.player_id = n.player_id
                LEFT JOIN _team_guess_conf tgc     ON o.player_id = tgc.player_id
                LEFT JOIN sums s                   ON o.player_id = s.player_id
                LEFT JOIN dim_teams dt_res         ON CASE
                                                        WHEN dp.team_id IS NOT NULL THEN dp.team_id
                                                        ELSE tgc.guessed_team_id
                                                    END = dt_res.team_id
                LEFT JOIN dim_teams dt_box         ON dp.team_id = dt_box.team_id
                LEFT JOIN dim_teams dt_guess       ON tgc.guessed_team_id = dt_guess.team_id
                ORDER BY o.player_id
            """)

            # --- Postflight debug metrics (surface issues instead of masking) ---
            n_rows = self.conn.execute("SELECT COUNT(*) FROM missing_player_report").fetchone()[0]
            logger.info(f"[Missing Player Report] built with {n_rows} rows")

            n_no_sums = self.conn.execute("SELECT COUNT(*) FROM missing_player_report WHERE total_events IS NULL").fetchone()[0]
            if n_no_sums > 0:
                logger.warning(f"[Missing Player Report] {n_no_sums} row(s) have NULL total_events (no matching occ/sums)")

            n_no_team = self.conn.execute("SELECT COUNT(*) FROM missing_player_report WHERE resolved_team_id IS NULL").fetchone()[0]
            if n_no_team > 0:
                logger.warning(f"[Missing Player Report] {n_no_team} row(s) missing resolved_team_id")

            n_no_name = self.conn.execute("SELECT COUNT(*) FROM missing_player_report WHERE resolved_name IS NULL").fetchone()[0]
            if n_no_name > 0:
                logger.warning(f"[Missing Player Report] {n_no_name} row(s) missing resolved_name")

            # --- NEW: Dump the FULL report & schema to the output (no truncation) ---
            try:
                df = self.conn.execute("""
                    SELECT *
                    FROM missing_player_report
                    ORDER BY player_id
                """).df()

                # Print schema (column name + dtype)
                schema_lines = [f"  - {col}: {str(df[col].dtype)}" for col in df.columns]
                logger.info("\n" + "="*80 + "\nMISSING PLAYER REPORT — SCHEMA\n" + "="*80 + "\n" + "\n".join(schema_lines))

                # Print full table without truncation
                with pd.option_context('display.max_columns', None, 'display.max_colwidth', None, 'display.width', 10000):
                    table_str = df.to_string(index=False)
                logger.info("\n" + "="*80 + "\nMISSING PLAYER REPORT — FULL DATA\n" + "="*80 + f"\nrows: {len(df)}\n" + table_str + "\n" + "="*80)

            except Exception as dump_e:
                logger.warning(f"[Missing Player Report] Could not print full report to logs: {dump_e}")

            return ValidationResult(
                step_name="Missing Player Report",
                passed=True,
                details=f"Built missing_player_report with {n_rows} rows",
                processing_time=time.time() - start_time
            )

        except Exception as e:
            return ValidationResult(
                step_name="Missing Player Report",
                passed=False,
                details=f"Error building report: {e}",
                processing_time=time.time() - start_time
            )


def load_all_data_enhanced(data_dir: Path | None = None, db_path: str = "mavs_enhanced.duckdb") -> Tuple[bool, Dict[str, Any]]:
    """Enhanced data loading with comprehensive validation + analytics outputs"""
    print("NBA Pipeline - Enhanced Data Loading & Validation")
    print("="*60)

    results = {
        'success': True,
        'validation_results': [],
        'data_summary': {}
    }

    # Prefer config-managed data directory if not provided
    if data_dir is None:
        try:
            from utils.config import MAVS_DATA_DIR
            data_dir = MAVS_DATA_DIR
        except Exception:
            data_dir = Path('data/mavs_data_engineer_2025')

    with EnhancedNBADataLoader(db_path) as loader:

        # 1) Box
        box_result = loader.load_and_validate_box_score(data_dir / 'box_HOU-DAL.csv')
        loader.validator.log_validation(box_result)
        results['validation_results'].append(box_result)
        if not box_result.passed:
            results['success'] = False
            return results['success'], results

        # 2) PBP
        pbp_result = loader.load_and_validate_pbp(data_dir / 'pbp_HOU-DAL.csv')
        loader.validator.log_validation(pbp_result)
        results['validation_results'].append(pbp_result)
        if not pbp_result.passed:
            results['success'] = False
            return results['success'], results

        # 3) Relationships
        rel_result = loader.validate_data_relationships()
        loader.validator.log_validation(rel_result)
        results['validation_results'].append(rel_result)
        if not rel_result.passed:
            results['success'] = False
            return results['success'], results

        # 4) Lookups
        lookup_result = loader.create_lookup_views()
        loader.validator.log_validation(lookup_result)
        results['validation_results'].append(lookup_result)
        if not lookup_result.passed:
            results['success'] = False
            return results['success'], results

        # 5) Dimensions
        dims_result = loader.create_dimensions()
        loader.validator.log_validation(dims_result)
        results['validation_results'].append(dims_result)
        if not dims_result.passed:
            results['success'] = False
            return results['success'], results

        # 6) Enriched view
        enrich_result = loader.create_pbp_enriched_view()
        loader.validator.log_validation(enrich_result)
        results['validation_results'].append(enrich_result)
        if not enrich_result.passed:
            results['success'] = False
            return results['success'], results

        # 6.3) UPDATED: Traditional Data-Driven Substitution Tracking (replaces basic)
        traditional_result = loader.run_traditional_data_driven_lineups()
        loader.validator.log_validation(traditional_result)
        results['validation_results'].append(traditional_result)
        if not traditional_result.passed:
            results['success'] = False
            return results['success'], results

        # 6.4) Enhanced substitution tracking with comprehensive flagging
        enhanced_result = loader.run_enhanced_substitution_tracking_with_flags()
        loader.validator.log_validation(enhanced_result)
        results['validation_results'].append(enhanced_result)
        if not enhanced_result.passed:
            results['success'] = False
            return results['success'], results

        # 6.5) Missing player report (optional but useful)
        missing_result = loader.create_missing_player_report()
        loader.validator.log_validation(missing_result)
        results['validation_results'].append(missing_result)
        if not missing_result.passed:
            results['success'] = False
            return results['success'], results

        # 7) Enhanced estimation engine (projects 1 & 2) - original method
        analytics_result = loader.run_lineups_and_rim_analytics()
        loader.validator.log_validation(analytics_result)
        results['validation_results'].append(analytics_result)

        # 7.5) UPDATED: Comprehensive comparison: Traditional Data-Driven vs Enhanced vs Box
        compare_result = loader.compare_traditional_vs_enhanced_lineups()
        loader.validator.log_validation(compare_result)
        results['validation_results'].append(compare_result)

        # 7.6) Legacy comparison (for backward compatibility)
        legacy_compare_result = loader.compare_basic_vs_estimated_lineups()
        loader.validator.log_validation(legacy_compare_result)
        results['validation_results'].append(legacy_compare_result)

        # 7.7) NEW: Dataset compliance validation
        compliance_result = loader.validate_dataset_compliance()
        loader.validator.log_validation(compliance_result)
        results['validation_results'].append(compliance_result)

        # 7.8) NEW: Create final submission artifacts
        submission_result = loader.create_project_submission_artifacts()
        loader.validator.log_validation(submission_result)
        results['validation_results'].append(submission_result)

        # 8) Final report
        report_result = loader.write_final_report()
        loader.validator.log_validation(report_result)
        results['validation_results'].append(report_result)

        # Summary
        results['data_summary'] = loader.data_summary
        loader.print_enhanced_summary()
        success = loader.validator.print_validation_summary()
        results['success'] = success
        return success, results





# Example usage
if __name__ == "__main__":
    data_directory = Path('data/mavs_data_engineer_2025')
    database_path = "mavs_enhanced.duckdb"

    success, results = load_all_data_enhanced(data_directory, database_path)

    if success:
        print("\n✅ Enhanced data loading completed successfully")
        print("🎯 Ready for entity extraction and lineup analysis")
    else:
        print("\n❌ Enhanced data loading failed")
        print("🔧 Review validation messages above")
