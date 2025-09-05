# Step 5: Dual-Method Possession Engine & Lineup Statistics Calculation
"""
NBA Pipeline - UPDATED Step 5: Dual-Method Possession Engine & Statistics
=========================================================================

UPDATED to integrate Step 2 findings:
- Processes possessions using BOTH traditional and enhanced lineup methods
- Generates separate statistics for each method
- Includes comprehensive violation and validation reporting
- Config-driven approach for automation settings
- Exports both result sets for comparison

Key Integration Points from Step 2:
1. Uses traditional_lineup_state (variable lineup sizes, raw data adherence)
2. Uses enhanced_lineup_state (5-man lineups, intelligent inference)  
3. Generates violation reports for traditional method
4. Comprehensive method comparison and validation
5. Config-driven automation paths

The possession engine determines when possessions change hands based on:
- Made field goals
- Turnovers
- Defensive rebounds (after missed shots)
- Free throw sequences
"""
import os
import sys
# Ensure we're in the right directory
cwd = os.getcwd()
if not cwd.endswith("airflow_project"):
    os.chdir('api/src/airflow_project')
sys.path.insert(0, os.getcwd())

import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import time
from typing import Dict, List, Tuple, Optional, Set, Any, NamedTuple
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import json

from eda.utils.nba_pipeline_analysis import NBADataValidator, ValidationResult
from eda.data.nba_entities_extractor import GameEntities
from eda.data.nba_pbp_processor import PBPProcessor, ProcessedEvent
import ast

# Load configuration
try:
    from utils.config import (
        NBA_SUBSTITUTION_CONFIG,
        RIM_DISTANCE_FEET,
        COORDINATE_SCALE,
        MINIMUM_SECONDS_PLAYED,
        DUCKDB_PATH,
        DUCKDB_DIR,
        EXPORTS_DIR
    )
    CONFIG = NBA_SUBSTITUTION_CONFIG
    RIM_THRESHOLD = RIM_DISTANCE_FEET
    COORD_SCALE = COORDINATE_SCALE
    MIN_SECONDS = MINIMUM_SECONDS_PLAYED
    DB_PATH = str(DUCKDB_PATH)
    EXPORT_DIR = EXPORTS_DIR
except ImportError:
    logger.warning("Config not available, using defaults")
    CONFIG = {"debug": {"log_all_substitutions": True}}
    RIM_THRESHOLD = 4.0
    COORD_SCALE = 10.0
    MIN_SECONDS = 30
    DB_PATH = "mavs_enhanced.duckdb"
    EXPORT_DIR = Path("exports")

logger = logging.getLogger(__name__)

class DualPossession(NamedTuple):
    """Represents a possession with both traditional and enhanced lineup contexts"""
    possession_id: int
    period: int
    start_pbp_order: int
    end_pbp_order: int
    off_team_id: int
    def_team_id: int

    # Traditional method lineups (may not be 5 players)
    traditional_off_lineup: Tuple[int, ...]
    traditional_def_lineup: Tuple[int, ...]

    # Enhanced method lineups (always 5 players)
    enhanced_off_lineup: Tuple[int, ...]
    enhanced_def_lineup: Tuple[int, ...]

    points_scored: int
    ended_by: str

@dataclass
class DualLineupStats:
    """Statistics for lineup with both method contexts"""
    team_id: int
    team_abbrev: str
    lineup_method: str  # 'traditional' or 'enhanced'
    player_ids: Tuple[int, ...]
    player_names: List[str]
    lineup_size: int  # Actual size (may vary for traditional)

    # Possession counts
    off_possessions: int = 0
    def_possessions: int = 0

    # Scoring
    points_for: int = 0
    points_against: int = 0

    # Ratings (per 100 possessions)
    off_rating: float = 0.0
    def_rating: float = 0.0
    net_rating: float = 0.0

    # Validation flags
    lineup_violations: List[str] = field(default_factory=list)

@dataclass  
class DualPlayerRimStats:
    """Player rim defense statistics with method context"""
    player_id: int
    player_name: str
    team_id: int
    team_abbrev: str
    method: str  # 'traditional' or 'enhanced'

    # Possession counts
    off_possessions: int = 0
    def_possessions: int = 0

    # Rim defense (when on court)
    opp_rim_attempts_on: int = 0
    opp_rim_makes_on: int = 0

    # Rim defense (when off court)  
    opp_rim_attempts_off: int = 0
    opp_rim_makes_off: int = 0

    # Calculated percentages
    opp_rim_fg_pct_on: float = None
    opp_rim_fg_pct_off: float = None
    rim_defense_on_off: float = None

class DualMethodPossessionEngine:
    """UPDATED: Possession engine that processes both traditional and enhanced methods"""

    def __init__(self, db_path: str = None, entities: GameEntities = None):
        self.db_path = db_path or DB_PATH
        self.conn = None
        self.entities = entities
        self.validator = NBADataValidator()

        # Dual possession tracking
        self.dual_possessions = []

        # Dual statistics containers
        self.traditional_lineup_stats = {}  # (team_id, lineup_tuple) -> DualLineupStats
        self.enhanced_lineup_stats = {}     # (team_id, lineup_tuple) -> DualLineupStats
        self.traditional_player_stats = {}  # player_id -> DualPlayerRimStats
        self.enhanced_player_stats = {}     # player_id -> DualPlayerRimStats

        # Violation tracking
        self.traditional_violations = []
        self.enhanced_violations = []

        # Method comparison metrics
        self.method_comparison = {}

        # Build entity mappings
        self.player_team = {}
        self.team_roster = {}
        self.team_abbrev = {}
        self.player_names = {}

        self._build_entity_mappings()


    def diagnose_pipeline_state(self) -> Dict[str, Any]:
        """Comprehensive diagnostic of pipeline state, with schema/NULL audits for step4 + contract stamp if present."""
        try:
            logger.info("=== PIPELINE DIAGNOSTIC ===")

            # All table names
            all_tables = self.conn.execute(
                "SELECT table_name FROM information_schema.tables ORDER BY table_name"
            ).fetchall()
            table_names = [t[0] for t in all_tables]

            diag = {
                "all_tables": table_names,
                "step_requirements": {
                    "step4_processed_events": "step4_processed_events" in table_names,
                    "traditional_lineup_state": "traditional_lineup_state" in table_names,
                    "enhanced_lineup_state": "enhanced_lineup_state" in table_names,
                    "traditional_lineup_flags": "traditional_lineup_flags" in table_names,
                    "enhanced_lineup_flags": "enhanced_lineup_flags" in table_names,
                },
                "alternative_tables": {
                    "step4_traditional_flags": "step4_traditional_flags" in table_names,
                    "step4_enhanced_flags": "step4_enhanced_flags" in table_names,
                    "processed_events": "processed_events" in table_names,
                    "traditional_violation_report": "traditional_violation_report" in table_names,
                    "enhanced_violation_report": "enhanced_violation_report" in table_names,
                },
                "table_counts": {},
                "sample_data": {},
                "step4_schema": {},
                "step4_null_audit": {},
                "contract": None,
            }

            # Counts & samples
            key_tables = [
                "step4_processed_events",
                "traditional_lineup_state",
                "enhanced_lineup_state",
                "traditional_lineup_flags",
                "enhanced_lineup_flags",
                "step4_traditional_flags",
                "step4_enhanced_flags",
                "traditional_violation_report",
                "enhanced_violation_report",
            ]
            for t in key_tables:
                if t in table_names:
                    try:
                        cnt = self.conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                        diag["table_counts"][t] = cnt
                        if t in ("step4_processed_events", "traditional_lineup_state", "enhanced_lineup_state"):
                            sample = self.conn.execute(f"SELECT * FROM {t} LIMIT 3").df()
                            diag["sample_data"][t] = sample.to_dict("records") if not sample.empty else []
                    except Exception as e:
                        diag["table_counts"][t] = f"Error: {e}"

            # Deep audit of step4_processed_events schema + null counts
            if "step4_processed_events" in table_names:
                cols = self.conn.execute("PRAGMA table_info('step4_processed_events')").fetchall()
                colnames = [c[1] for c in cols]
                diag["step4_schema"]["columns"] = colnames
                diag["step4_schema"]["has_legacy_lineups"] = ("off_lineup" in colnames and "def_lineup" in colnames)
                diag["step4_schema"]["has_traditional_lineups"] = (
                    "traditional_off_lineup" in colnames and "traditional_def_lineup" in colnames
                )
                diag["step4_schema"]["has_enhanced_lineups"] = (
                    "enhanced_off_lineup" in colnames and "enhanced_def_lineup" in colnames
                )
                diag["step4_schema"]["has_points"] = "points" in colnames
                diag["step4_schema"]["has_rim_flags"] = all(c in colnames for c in ("is_rim_attempt", "is_rim_make"))

                # Null audits (counts of non-null values)
                to_audit = [
                    "off_lineup", "def_lineup",
                    "traditional_off_lineup", "traditional_def_lineup",
                    "enhanced_off_lineup", "enhanced_def_lineup",
                    "points", "is_rim_attempt", "is_rim_make"
                ]
                for c in to_audit:
                    if c in colnames:
                        try:
                            n = self.conn.execute(
                                f"SELECT COUNT(*) FROM step4_processed_events WHERE {c} IS NOT NULL"
                            ).fetchone()[0]
                            diag["step4_null_audit"][c] = n
                        except Exception as e:
                            diag["step4_null_audit"][c] = f"Error: {e}"

            # Contract stamp if present
            if "pipeline_contract" in table_names:
                try:
                    dfc = self.conn.execute("""
                        SELECT *
                        FROM pipeline_contract
                        WHERE component='step4'
                        ORDER BY created_at DESC
                        LIMIT 1
                    """).df()
                    if not dfc.empty:
                        diag["contract"] = dfc.to_dict("records")[0]
                except Exception as e:
                    diag["contract"] = f"Error reading contract: {e}"

            logger.info(f"Diagnostic results: {diag}")
            return diag

        except Exception as e:
            logger.error(f"Error in pipeline diagnostic: {e}")
            return {"error": str(e)}



    def _build_entity_mappings(self):
        """Build entity mappings from provided entities"""
        if not self.entities or not hasattr(self.entities, 'unique_players'):
            return

        up = self.entities.unique_players
        if up is not None and not up.empty:
            # Player mappings
            for _, r in up.iterrows():
                pid = int(r["player_id"])
                tid = int(r["team_id"])
                self.player_team[pid] = tid
                self.player_names[pid] = str(r["player_name"])

                if tid not in self.team_roster:
                    self.team_roster[tid] = set()
                self.team_roster[tid].add(pid)

            # Team mappings
            for _, r in up.drop_duplicates("team_id").iterrows():
                tid = int(r["team_id"])
                self.team_abbrev[tid] = str(r["team_abbrev"])

    def _validate_lineup(self,
                        lineup: Optional[Tuple[int, ...]],
                        label: str,
                        poss_id: int,
                        method: str = "enhanced") -> None:
        """
        Validate lineup contents according to method rules.

        enhanced: exactly 5 unique players.
        traditional: non-empty, all unique (size may be != 5).
        """
        if lineup is None:
            raise AssertionError(f"[Possession {poss_id}] {label} lineup is None")

        if method == "enhanced":
            if len(lineup) != 5:
                raise AssertionError(
                    f"[Possession {poss_id}] {label} enhanced lineup len={len(lineup)} != 5 -> {lineup}"
                )
            if len(set(lineup)) != 5:
                raise AssertionError(
                    f"[Possession {poss_id}] {label} enhanced lineup has duplicates -> {lineup}"
                )
        else:
            if len(lineup) == 0:
                raise AssertionError(f"[Possession {poss_id}] {label} traditional lineup is empty")
            if len(set(lineup)) != len(lineup):
                raise AssertionError(
                    f"[Possession {poss_id}] {label} traditional lineup has duplicates -> {lineup}"
                )

    def _rebound_team(self, event) -> Optional[int]:
        """
        Infer rebounder team for msgType==4 using player_id_1 if available,
        falling back to def/off teams only if we cannot resolve player→team.
        """
        # Correct attribute name on ProcessedEvent is player_id_1
        pid = getattr(event, "player_id_1", None)
        # Be tolerant if a dict-like row sneaks in
        if pid is None and hasattr(event, "__getitem__"):
            try:
                pid = event["player_id_1"]
            except Exception:
                pid = None

        if pid is not None and pid in self.player_team:
            return self.player_team[pid]

        # Fallbacks (keep same order of preference)
        if getattr(event, "def_team_id", None) is not None:
            return event.def_team_id
        if getattr(event, "off_team_id", None) is not None:
            return event.off_team_id
        return None



    def __enter__(self):
        self.conn = duckdb.connect(self.db_path)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn:
            self.conn.close()



    def _load_events_from_db_as_processed(self) -> List[ProcessedEvent]:
        """
        Fallback loader: reconstruct ProcessedEvent objects from DuckDB table
        'processed_events' created by Step 4. Returns [] if not found/empty.
        """
        try:
            # Ensure we have a connection
            if self.conn is None:
                self.conn = duckdb.connect(self.db_path)

            # Check table existence
            tables = set(t[0].lower() for t in self.conn.execute(
                "SELECT table_name FROM information_schema.tables"
            ).fetchall())
            if "processed_events" not in tables:
                return []

            df = self.conn.execute("""
                SELECT
                    pbp_id, period, pbp_order, wall_clock_int, description,
                    msg_type, action_type, off_team_id, def_team_id,
                    player_id_1, player_id_2, player_id_3,
                    loc_x, loc_y, points,
                    is_shot, is_rim_attempt, is_rim_make, distance_ft,
                    is_substitution, sub_out_player, sub_in_player,
                    off_lineup, def_lineup
                FROM processed_events
                ORDER BY period, pbp_order, wall_clock_int
            """).df()
            if df.empty:
                return []

            events: List[ProcessedEvent] = []
            for _, r in df.iterrows():
                # Parse lineup strings back to tuples
                def _to_tuple(s):
                    if pd.isna(s) or s is None or s == "":
                        return None
                    try:
                        t = ast.literal_eval(str(s))
                        # normalize to sorted tuple of ints
                        if isinstance(t, (list, tuple)):
                            return tuple(int(x) for x in t)
                    except Exception:
                        pass
                    return None

                ev = ProcessedEvent(
                    pbp_id         = int(r["pbp_id"]),
                    period         = int(r["period"]),
                    pbp_order      = int(r["pbp_order"]),
                    wall_clock_int = int(r["wall_clock_int"]) if pd.notna(r["wall_clock_int"]) else 0,
                    msg_type       = int(r["msg_type"]) if pd.notna(r["msg_type"]) else None,
                    action_type    = int(r["action_type"]) if pd.notna(r["action_type"]) else None,
                    description    = str(r["description"]) if pd.notna(r["description"]) else "",
                    off_team_id    = int(r["off_team_id"]) if pd.notna(r["off_team_id"]) else None,
                    def_team_id    = int(r["def_team_id"]) if pd.notna(r["def_team_id"]) else None,
                    player_id_1    = int(r["player_id_1"]) if pd.notna(r["player_id_1"]) else None,
                    player_id_2    = int(r["player_id_2"]) if pd.notna(r["player_id_2"]) else None,
                    player_id_3    = int(r["player_id_3"]) if pd.notna(r["player_id_3"]) else None,
                    loc_x          = int(r["loc_x"]) if pd.notna(r["loc_x"]) and r["loc_x"] != 0 else None,
                    loc_y          = int(r["loc_y"]) if pd.notna(r["loc_y"]) and r["loc_y"] != 0 else None,
                    points         = int(r["points"]) if pd.notna(r["points"]) else 0,

                    is_shot        = bool(r["is_shot"]) if pd.notna(r["is_shot"]) else False,
                    is_rim_attempt = bool(r["is_rim_attempt"]) if pd.notna(r["is_rim_attempt"]) else False,
                    is_rim_make    = bool(r["is_rim_make"]) if pd.notna(r["is_rim_make"]) else False,
                    distance_ft    = float(r["distance_ft"]) if pd.notna(r["distance_ft"]) else None,
                    is_substitution= bool(r["is_substitution"]) if pd.notna(r["is_substitution"]) else False,
                    sub_out_player = int(r["sub_out_player"]) if pd.notna(r["sub_out_player"]) else None,
                    sub_in_player  = int(r["sub_in_player"]) if pd.notna(r["sub_in_player"]) else None,

                    off_lineup     = _to_tuple(r["off_lineup"]),
                    def_lineup     = _to_tuple(r["def_lineup"]),
                )
                events.append(ev)

            return events
        except Exception as e:
            logger.error(f"Failed to load processed_events from DB: {e}")
            return []

    def _assert_step4_schema_or_rebuild(self, autorun: bool = False) -> Dict[str, Any]:
        """
        Ensure 'step4_processed_events' contains the required dual-method lineup columns.
        - If missing and autorun=True, invoke Step 4 to rebuild once, then re-check.
        - Returns a dict: {"ok": bool, "details": str}
        """
        required = {
            "traditional_off_lineup", "traditional_def_lineup",
            "enhanced_off_lineup", "enhanced_def_lineup"
        }

        # existence
        tables = {t[0] for t in self.conn.execute("SELECT table_name FROM information_schema.tables").fetchall()}
        if "step4_processed_events" not in tables:
            return {"ok": False, "details": "Missing table 'step4_processed_events' (Step 4 not run)."}

        # schema
        cols = {r[1] for r in self.conn.execute("PRAGMA table_info('step4_processed_events')").fetchall()}
        if required.issubset(cols):
            return {"ok": True, "details": "Step 4 schema satisfied (dual-method columns present)."}

        missing = sorted(required - cols)
        msg = f"Step 4 schema missing required columns: {missing}. Present columns: {sorted(cols)}"

        if not autorun:
            # Do not fix; just report
            logger.error(f"[CONTRACT] {msg}")
            return {"ok": False, "details": msg + " | autorun=False"}

        # Try to rebuild by running the real Step 4
        logger.warning(f"[CONTRACT] {msg} -> autorun=True, invoking Step 4 to rebuild...")
        try:
            from eda.data.nba_pbp_processor import process_pbp_with_step2_integration
            ok, _ = process_pbp_with_step2_integration(db_path=self.db_path, entities=self.entities)
            if not ok:
                return {"ok": False, "details": "Invoked Step 4, but it reported failure."}

            # Re-check
            cols2 = {r[1] for r in self.conn.execute("PRAGMA table_info('step4_processed_events')").fetchall()}
            if required.issubset(cols2):
                return {"ok": True, "details": "Step 4 rebuilt and schema now satisfies contract."}
            return {"ok": False, "details": "Step 4 rebuilt but schema still missing required dual-method columns."}
        except Exception as e:
            return {"ok": False, "details": f"Failed to invoke Step 4: {e}"}


    def load_dual_method_data(self, autorun_rebuild: bool = False) -> ValidationResult:
        """
        Load upstream artifacts needed by Step 5.

        Hard requirement:
        - step4_processed_events with dual-method lineup columns (contract)

        Optional (diagnostics/enrichment; absence should NOT block Step 5):
        - traditional_lineup_state / enhanced_lineup_state
        - traditional_lineup_flags / enhanced_lineup_flags
        - step4_traditional_flags / step4_enhanced_flags (older naming)
        - traditional_violation_report / enhanced_violation_report (read-only, previous Step 5 outputs)

        We DO NOT fill in or synthesize missing data. We enforce the contract, report, and
        optionally rebuild via Step 4 when autorun_rebuild=True.
        """
        start_time = time.time()
        try:
            logger.info("Loading dual-method data from Step 2/4 integration...")

            # ---- Enforce Step 4 → Step 5 contract up front ----
            contract = self._assert_step4_schema_or_rebuild(autorun=autorun_rebuild)
            if not contract.get("ok"):
                return ValidationResult(
                    step_name="Load Dual Method Data",
                    passed=False,
                    details=("Contract failed: " + contract.get("details", "")),
                    processing_time=time.time() - start_time,
                )

            all_tables = self.conn.execute(
                "SELECT table_name FROM information_schema.tables ORDER BY table_name"
            ).fetchall()
            logger.info(f"DEBUG: Available tables in database: {[t[0] for t in all_tables]}")

            existing = {r[0] for r in all_tables}

            # ---- Optional sources (don't fail if missing) ----
            optional_expected = [
                "traditional_lineup_state",
                "enhanced_lineup_state",
                "traditional_lineup_flags",
                "enhanced_lineup_flags",
                "step4_traditional_flags",
                "step4_enhanced_flags",
                # read-only outputs from previous Step 5 runs (if present)
                "traditional_violation_report",
                "enhanced_violation_report",
            ]
            present_optional = [t for t in optional_expected if t in existing]
            missing_optional = [t for t in optional_expected if t not in existing]

            logger.info(f"DEBUG: Optional tables present: {present_optional}")
            logger.info(f"DEBUG: Optional tables missing (non-blocking): {missing_optional}")

            # ---- Load flags (non-blocking): prefer official names, then step4_*, then violation_report ----
            trad_flags = self._load_flags_with_fallback("traditional_lineup_flags", "step4_traditional_flags")
            enh_flags  = self._load_flags_with_fallback("enhanced_lineup_flags", "step4_enhanced_flags")

            self.traditional_violations = trad_flags.to_dict("records") if not trad_flags.empty else []
            self.enhanced_violations    = enh_flags.to_dict("records") if not enh_flags.empty else []

            details = (
                "Loaded Step 5 inputs. "
                f"Flags: traditional={len(self.traditional_violations)}, enhanced={len(self.enhanced_violations)}. "
                f"Optional sources missing (non-blocking): {missing_optional}."
            )
            return ValidationResult(
                step_name="Load Dual Method Data",
                passed=True,
                details=details,
                data_count=len(self.traditional_violations) + len(self.enhanced_violations),
                processing_time=time.time() - start_time,
            )

        except Exception as e:
            return ValidationResult(
                step_name="Load Dual Method Data",
                passed=False,
                details=f"Error loading dual-method data: {e}",
                processing_time=time.time() - start_time,
            )



    def _create_missing_tables_from_alternatives(self, missing: List[str], existing: set) -> bool:
        """Create missing tables from alternative sources (never synthesizes step4_processed_events)."""
        try:
            logger.info("DEBUG: Attempting to create missing tables from alternatives...")

            # Only allow flag tables to be backfilled from their step4_* equivalents.
            # DO NOT create step4_processed_events from legacy processed_events (schema would be wrong).
            table_mappings = {
                'step4_traditional_flags': 'traditional_lineup_flags',
                'step4_enhanced_flags': 'enhanced_lineup_flags',
                # 'processed_events': 'step4_processed_events'  # intentionally removed to avoid legacy schema propagation
            }

            created_count = 0
            for alt_name, expected_name in table_mappings.items():
                if expected_name.lower() in missing and alt_name.lower() in existing:
                    logger.info(f"DEBUG: Creating '{expected_name}' from '{alt_name}'")
                    try:
                        self.conn.execute(f"CREATE OR REPLACE TABLE {expected_name} AS SELECT * FROM {alt_name}")
                        created_count += 1
                        logger.info(f"DEBUG: Successfully created '{expected_name}'")
                    except Exception as e:
                        logger.warning(f"DEBUG: Failed to create '{expected_name}' from '{alt_name}': {e}")

            logger.info(f"DEBUG: Created {created_count} missing tables from alternatives")
            return created_count > 0

        except Exception as e:
            logger.error(f"DEBUG: Error creating missing tables: {e}")
            return False


    def _load_flags_with_fallback(self, primary_table: str, fallback_table: str) -> pd.DataFrame:
        """
        Load a flags-like table with sensible fallbacks and explicit debugs.

        Resolution order:
        1) primary_table (e.g., traditional_lineup_flags)
        2) fallback_table (e.g., step4_traditional_flags)
        3) read-only violation report (e.g., traditional_violation_report / enhanced_violation_report)

        We DO NOT synthesize records. If none are found, return an empty DataFrame.
        """
        try:
            tables = {
                t[0].lower()
                for t in self.conn.execute("SELECT table_name FROM information_schema.tables").fetchall()
            }

            # try primary
            target = None
            if primary_table.lower() in tables:
                target = primary_table
                logger.info(f"DEBUG: Using primary flags table '{primary_table}'")
            # else fallback
            elif fallback_table.lower() in tables:
                target = fallback_table
                logger.info(f"DEBUG: Using fallback flags table '{fallback_table}'")
            else:
                # consider read-only prior outputs if they exist
                alt_report = None
                if primary_table.lower().startswith("traditional"):
                    alt_report = "traditional_violation_report"
                elif primary_table.lower().startswith("enhanced"):
                    alt_report = "enhanced_violation_report"

                if alt_report and alt_report.lower() in tables:
                    target = alt_report
                    logger.info(
                        f"DEBUG: Using read-only prior output '{alt_report}' as violation source "
                        f"(no synthesis; purely for context)"
                    )

            if not target:
                logger.warning(f"DEBUG: No available tables among: '{primary_table}', '{fallback_table}', prior reports")
                return pd.DataFrame()

            # Determine a stable ordering column if present
            order_cols = [c[1] for c in self.conn.execute(f"PRAGMA table_info('{target}')").fetchall()]
            if "abs_time" in order_cols:
                order_expr = "abs_time"
            elif "wall_clock_int" in order_cols:
                order_expr = "wall_clock_int"
            elif "pbp_order" in order_cols:
                order_expr = "pbp_order"
            else:
                order_expr = None

            if order_expr:
                df = self.conn.execute(f"SELECT * FROM {target} ORDER BY {order_expr}").df()
            else:
                df = self.conn.execute(f"SELECT * FROM {target}").df()

            logger.info(f"DEBUG: Loaded {len(df)} rows from '{target}' for flags/violations")
            return df

        except Exception as e:
            logger.error(f"DEBUG: Error loading flags from '{primary_table}'/'{fallback_table}': {e}")
            return pd.DataFrame()

    ALLOW_LEGACY_FALLBACK = False

    def identify_dual_possessions(self) -> ValidationResult:
        """FIXED: Enhanced possession identification with proper team attribution"""
        start_time = time.time()
        try:
            logger.info("Identifying possessions with dual-method lineup contexts (FIXED MODE)...")

            event_count = self.conn.execute("SELECT COUNT(*) FROM step4_processed_events").fetchone()[0]
            logger.info(f"DEBUG: step4_processed_events has {event_count} rows")

            cols = {r[1] for r in self.conn.execute("PRAGMA table_info('step4_processed_events')").fetchall()}
            have_trad = {"traditional_off_lineup", "traditional_def_lineup"}.issubset(cols)
            have_enh  = {"enhanced_off_lineup", "enhanced_def_lineup"}.issubset(cols)

            if not (have_trad and have_enh):
                return ValidationResult(
                    step_name="Identify Dual Possessions",
                    passed=False,
                    details=("Required columns missing in 'step4_processed_events'. "
                            "Expected traditional_/enhanced_ lineups. Aborting without fallback."),
                    processing_time=time.time() - start_time
                )

            select_sql = """
                SELECT 
                    pbp_id, period, pbp_order, wall_clock_int, msg_type,
                    off_team_id, def_team_id, points,
                    traditional_off_lineup, traditional_def_lineup,
                    enhanced_off_lineup, enhanced_def_lineup,
                    player_id_1, description
                FROM step4_processed_events
                WHERE off_team_id IS NOT NULL AND def_team_id IS NOT NULL
                ORDER BY period, pbp_order, wall_clock_int
            """

            events_df = self.conn.execute(select_sql).df()
            logger.info(f"DEBUG: Retrieved {len(events_df)} events with valid team IDs")

            if events_df.empty:
                return ValidationResult(
                    step_name="Identify Dual Possessions",
                    passed=False,
                    details="No events with dual lineup context found",
                    processing_time=time.time() - start_time
                )

            def _parse_lineup(val) -> Optional[Tuple[int, ...]]:
                if val is None or (isinstance(val, float) and pd.isna(val)) or str(val).strip() == "":
                    return None
                try:
                    obj = json.loads(val) if isinstance(val, str) else val
                except Exception:
                    try:
                        obj = ast.literal_eval(str(val))
                    except Exception:
                        logger.debug(f"Lineup parse failed; raw={val!r}")
                        return None
                if isinstance(obj, (list, tuple, set)):
                    try:
                        tup = tuple(sorted(int(x) for x in obj))
                        return tup
                    except Exception:
                        logger.debug(f"Lineup normalization failed; raw={obj!r}")
                        return None
                logger.debug(f"Lineup had unexpected type; raw={obj!r}")
                return None

            self.dual_possessions = []
            possession_id = 0

            # FIXED: Initialize possession state properly
            current_possession = None

            # Enhanced debug tracking
            points_by_possession = []
            scoring_events_processed = []
            team_points_tracker = {"DAL": 0, "HOU": 0}  # Track cumulative points by team

            for idx, row in events_df.iterrows():
                msg_type = int(row["msg_type"]) if pd.notna(row["msg_type"]) else None
                event_team_id = int(row["off_team_id"])
                event_def_team = int(row["def_team_id"])
                event_points = int(row["points"]) if pd.notna(row["points"]) and row["points"] > 0 else 0

                # Track every scoring event with detailed context
                if event_points > 0:
                    team_abbrev = self.team_abbrev.get(event_team_id, f"Team{event_team_id}")
                    team_points_tracker[team_abbrev] += event_points

                    scoring_event = {
                        "pbp_id": int(row["pbp_id"]),
                        "period": int(row["period"]),
                        "pbp_order": int(row["pbp_order"]),
                        "points": event_points,
                        "off_team_id": event_team_id,
                        "description": str(row["description"]),
                        "msg_type": msg_type,
                        "current_possession_team": current_possession["off_team_id"] if current_possession else None,
                        "team_abbrev": team_abbrev,
                        "cumulative_team_points": team_points_tracker[team_abbrev]
                    }
                    scoring_events_processed.append(scoring_event)

                # FIXED: Determine if we need to end current possession
                should_end_possession = (
                    current_possession is None or  # No current possession
                    current_possession["off_team_id"] != event_team_id or  # Team changed
                    current_possession["def_team_id"] != event_def_team or  # Defense changed
                    msg_type in (1, 5, 12, 13) or  # Made shot, turnover, period boundaries
                    (msg_type == 4 and self._is_defensive_rebound(row))  # Defensive rebound
                )

                # End current possession if needed
                if should_end_possession and current_possession is not None:
                    # Close current possession
                    possession_info = {
                        "possession_id": possession_id,
                        "team_id": current_possession["off_team_id"],
                        "points": current_possession["points"],
                        "period": current_possession["period"],
                        "start_order": current_possession["start_order"],
                        "end_order": int(row["pbp_order"]) - 1,
                        "ended_by": self._determine_end_reason(row)
                    }
                    points_by_possession.append(possession_info)

                    self.dual_possessions.append(
                        DualPossession(
                            possession_id=possession_id,
                            period=current_possession["period"],
                            start_pbp_order=current_possession["start_order"],
                            end_pbp_order=int(row["pbp_order"]) - 1,
                            off_team_id=current_possession["off_team_id"],
                            def_team_id=current_possession["def_team_id"],
                            traditional_off_lineup=current_possession["trad_off"],
                            traditional_def_lineup=current_possession["trad_def"],
                            enhanced_off_lineup=current_possession["enh_off"],
                            enhanced_def_lineup=current_possession["enh_def"],
                            points_scored=current_possession["points"],
                            ended_by=possession_info["ended_by"],
                        )
                    )
                    possession_id += 1
                    current_possession = None

                # FIXED: Start new possession if needed (always when no current possession)
                if current_possession is None:
                    current_possession = {
                        "off_team_id": event_team_id,
                        "def_team_id": event_def_team,
                        "start_order": int(row["pbp_order"]),
                        "period": int(row["period"]),
                        "points": 0,  # Will accumulate points for this possession
                        "trad_off": _parse_lineup(row["traditional_off_lineup"]),
                        "trad_def": _parse_lineup(row["traditional_def_lineup"]),
                        "enh_off": _parse_lineup(row["enhanced_off_lineup"]),
                        "enh_def": _parse_lineup(row["enhanced_def_lineup"])
                    }

                    # Validate enhanced lineups
                    try:
                        if current_possession["enh_off"] is not None:
                            self._validate_lineup(current_possession["enh_off"], "off", possession_id, "enhanced")
                        if current_possession["enh_def"] is not None:
                            self._validate_lineup(current_possession["enh_def"], "def", possession_id, "enhanced")
                    except AssertionError as ae:
                        logger.warning(f"Enhanced lineup validation failed: {ae}")

                # FIXED: Add points to current possession only if teams match
                if event_points > 0 and current_possession["off_team_id"] == event_team_id:
                    current_possession["points"] += event_points
                elif event_points > 0:
                    # Points belong to different team - log this mismatch but don't add to possession
                    logger.warning(f"SCORING EVENT TEAM MISMATCH: pbp_id={row['pbp_id']}, "
                                 f"possession_team={current_possession['off_team_id']}, "
                                 f"event_team={event_team_id}, points={event_points}")

            # Close the final possession
            if current_possession is not None and not events_df.empty:
                final_possession = {
                    "possession_id": possession_id,
                    "team_id": current_possession["off_team_id"],
                    "points": current_possession["points"],
                    "period": current_possession["period"],
                    "start_order": current_possession["start_order"],
                    "end_order": int(events_df.iloc[-1]["pbp_order"])
                }
                points_by_possession.append(final_possession)

                self.dual_possessions.append(
                    DualPossession(
                        possession_id=possession_id,
                        period=current_possession["period"],
                        start_pbp_order=current_possession["start_order"],
                        end_pbp_order=int(events_df.iloc[-1]["pbp_order"]),
                        off_team_id=current_possession["off_team_id"],
                        def_team_id=current_possession["def_team_id"],
                        traditional_off_lineup=current_possession["trad_off"],
                        traditional_def_lineup=current_possession["trad_def"],
                        enhanced_off_lineup=current_possession["enh_off"],
                        enhanced_def_lineup=current_possession["enh_def"],
                        points_scored=current_possession["points"],
                        ended_by="game_end",
                    )
                )

            # FIXED: Enhanced debug output with corrected calculations
            logger.info("=== FIXED POSSESSION DEBUG ANALYSIS ===")
            total_possession_points = sum(p["points"] for p in points_by_possession)
            total_scoring_events = len(scoring_events_processed)
            total_event_points = sum(e["points"] for e in scoring_events_processed)

            logger.info(f"Total Possessions Created: {len(self.dual_possessions):,}")
            logger.info(f"Total Possession Points: {total_possession_points:,}")
            logger.info(f"Total Scoring Events Processed: {total_scoring_events:,}")
            logger.info(f"Total Event Points: {total_event_points:,}")

            # Team-specific analysis with corrected logic
            team_possession_points = {}
            team_event_points = {}

            for p in points_by_possession:
                team_abbrev = self.team_abbrev.get(p["team_id"], f"Team{p['team_id']}")
                team_possession_points[team_abbrev] = team_possession_points.get(team_abbrev, 0) + p["points"]

            for e in scoring_events_processed:
                team_abbrev = e["team_abbrev"]
                team_event_points[team_abbrev] = team_event_points.get(team_abbrev, 0) + e["points"]

            logger.info("FIXED TEAM-BY-TEAM ANALYSIS:")
            for team in ["DAL", "HOU"]:
                poss_pts = team_possession_points.get(team, 0)
                event_pts = team_event_points.get(team, 0)
                diff = poss_pts - event_pts  # Possession points should match event points

                logger.info(f"  {team}:")
                logger.info(f"    Possession Points: {poss_pts}")
                logger.info(f"    Event Points: {event_pts}")
                logger.info(f"    Difference: {diff:+}")

                if diff != 0:
                    logger.warning(f"    *** {team} POINTS DISCREPANCY: {diff:+} points ***")

            details = (f"FIXED: Identified {len(self.dual_possessions)} dual-method possessions, "
                      f"{total_possession_points} possession points vs {total_event_points} event points.")

            return ValidationResult(
                step_name="Identify Dual Possessions",
                passed=len(self.dual_possessions) > 0,
                details=details,
                data_count=len(self.dual_possessions),
                processing_time=time.time() - start_time
            )

        except Exception as e:
            return ValidationResult(
                step_name="Identify Dual Possessions",
                passed=False,
                details=f"Error identifying dual possessions: {e}",
                processing_time=time.time() - start_time
            )



    def _is_defensive_rebound(self, row) -> bool:
        """Determine if rebound is defensive based on player team"""
        if row['msg_type'] != 4:  # Not a rebound
            return False

        rebounder_id = row.get('player_id_1')
        if not rebounder_id or rebounder_id not in self.player_team:
            return False

        rebounder_team = self.player_team[rebounder_id]
        return rebounder_team == row['def_team_id']

    def _determine_end_reason(self, row) -> str:
        """Determine why possession ended (made FG, turnover, defensive rebound, or period boundary)."""
        mt = row.get('msg_type') if hasattr(row, 'get') else row['msg_type']
        if mt == 1:
            return "made_fg"
        elif mt == 5:
            return "turnover"
        elif mt == 4 and self._is_defensive_rebound(row):
            return "def_rebound"
        elif mt in (12, 13):  # 12: start period; 13: end period
            return "period_boundary"
        else:
            return "team_change"

    def debug_points_flow_comprehensive(self) -> Dict[str, Any]:
        """
        Comprehensive points flow analysis to identify where the 4-point HOU discrepancy originates.
        Traces from raw events -> processed events -> possessions -> lineups.
        """
        try:
            logger.info("=== COMPREHENSIVE POINTS FLOW DEBUG ===")

            debug_info = {
                "raw_pbp_points": {},
                "step4_processed_points": {},
                "possession_points": {},
                "lineup_points": {},
                "discrepancies": [],
                "detailed_scoring_events": []
            }

            # 1. Raw PBP points by team
            raw_points = self.conn.execute("""
                SELECT 
                    CASE 
                        WHEN team_id_off = 1610612742 THEN 'DAL'
                        WHEN team_id_off = 1610612745 THEN 'HOU'
                        ELSE CAST(team_id_off AS VARCHAR)
                    END as team,
                    SUM(COALESCE(points, 0)) as total_points,
                    COUNT(*) as scoring_events
                FROM pbp 
                WHERE points > 0 AND team_id_off IS NOT NULL
                GROUP BY team_id_off
                ORDER BY team
            """).df()

            for _, row in raw_points.iterrows():
                debug_info["raw_pbp_points"][row["team"]] = {
                    "points": int(row["total_points"]),
                    "events": int(row["scoring_events"])
                }

            # 2. Step4 processed points by team
            step4_points = self.conn.execute("""
                SELECT 
                    CASE 
                        WHEN off_team_id = 1610612742 THEN 'DAL'
                        WHEN off_team_id = 1610612745 THEN 'HOU'
                        ELSE CAST(off_team_id AS VARCHAR)
                    END as team,
                    SUM(COALESCE(points, 0)) as total_points,
                    COUNT(*) as scoring_events
                FROM step4_processed_events 
                WHERE points > 0 AND off_team_id IS NOT NULL
                GROUP BY off_team_id
                ORDER BY team
            """).df()

            for _, row in step4_points.iterrows():
                debug_info["step4_processed_points"][row["team"]] = {
                    "points": int(row["total_points"]),
                    "events": int(row["scoring_events"])
                }

            # 3. Possession-level points (from our dual possessions)
            possession_totals = {}
            for poss in self.dual_possessions:
                team_abbrev = self.team_abbrev.get(poss.off_team_id, f"Team{poss.off_team_id}")
                if team_abbrev not in possession_totals:
                    possession_totals[team_abbrev] = {"points": 0, "possessions": 0}
                possession_totals[team_abbrev]["points"] += poss.points_scored
                possession_totals[team_abbrev]["possessions"] += 1

            debug_info["possession_points"] = possession_totals

            # 4. Lineup-level points (both methods)
            debug_info["lineup_points"] = {}
            for method in ["traditional", "enhanced"]:
                lineup_stats = (self.traditional_lineup_stats if method == "traditional" 
                              else self.enhanced_lineup_stats)
                method_totals = {}
                for (team_id, lineup), stats in lineup_stats.items():
                    team_abbrev = self.team_abbrev.get(team_id, f"Team{team_id}")
                    if team_abbrev not in method_totals:
                        method_totals[team_abbrev] = {"points": 0, "lineups": 0}
                    method_totals[team_abbrev]["points"] += stats.points_for
                    method_totals[team_abbrev]["lineups"] += 1
                debug_info["lineup_points"][method] = method_totals

            # 5. Detailed scoring events for HOU (the problematic team)
            hou_events = self.conn.execute("""
                SELECT 
                    pbp_id, period, pbp_order, description, points,
                    off_team_id, msg_type, action_type,
                    traditional_off_lineup, enhanced_off_lineup
                FROM step4_processed_events 
                WHERE off_team_id = 1610612745 AND points > 0
                ORDER BY period, pbp_order
            """).df()

            debug_info["detailed_scoring_events"] = hou_events.to_dict('records')

            # 6. Calculate discrepancies at each stage
            teams = ['DAL', 'HOU']
            for team in teams:
                raw_pts = debug_info["raw_pbp_points"].get(team, {}).get("points", 0)
                step4_pts = debug_info["step4_processed_points"].get(team, {}).get("points", 0)
                poss_pts = debug_info["possession_points"].get(team, {}).get("points", 0)

                trad_lineup_pts = debug_info["lineup_points"].get("traditional", {}).get(team, {}).get("points", 0)
                enh_lineup_pts = debug_info["lineup_points"].get("enhanced", {}).get(team, {}).get("points", 0)

                discrepancy = {
                    "team": team,
                    "raw_pbp_points": raw_pts,
                    "step4_processed_points": step4_pts,
                    "possession_points": poss_pts,
                    "traditional_lineup_points": trad_lineup_pts,
                    "enhanced_lineup_points": enh_lineup_pts,
                    "raw_to_step4_diff": step4_pts - raw_pts,
                    "step4_to_possession_diff": poss_pts - step4_pts,
                    "possession_to_traditional_diff": trad_lineup_pts - poss_pts,
                    "possession_to_enhanced_diff": enh_lineup_pts - poss_pts
                }
                debug_info["discrepancies"].append(discrepancy)

            # Log findings
            logger.info("POINTS FLOW ANALYSIS:")
            for disc in debug_info["discrepancies"]:
                team = disc["team"]
                logger.info(f"  {team}:")
                logger.info(f"    Raw PBP: {disc['raw_pbp_points']}")
                logger.info(f"    Step4 Processed: {disc['step4_processed_points']} (diff: {disc['raw_to_step4_diff']:+})")
                logger.info(f"    Possessions: {disc['possession_points']} (diff: {disc['step4_to_possession_diff']:+})")
                logger.info(f"    Traditional Lineups: {disc['traditional_lineup_points']} (diff: {disc['possession_to_traditional_diff']:+})")
                logger.info(f"    Enhanced Lineups: {disc['enhanced_lineup_points']} (diff: {disc['possession_to_enhanced_diff']:+})")

            return debug_info

        except Exception as e:
            logger.error(f"Error in comprehensive points flow debug: {e}")
            return {"error": str(e)}

    def debug_hou_scoring_events_detailed(self) -> Dict[str, Any]:
        """
        Detailed analysis of every HOU scoring event to identify potential double-counting
        or attribution errors causing the 4-point discrepancy.
        """
        try:
            logger.info("=== DETAILED HOU SCORING EVENTS ANALYSIS ===")

            # Get all HOU scoring events with context
            hou_scoring = self.conn.execute("""
                SELECT 
                    se.pbp_id, se.period, se.pbp_order, se.wall_clock_int,
                    se.description, se.points, se.msg_type, se.action_type,
                    se.player_id_1, se.player_id_2, se.player_id_3,
                    se.traditional_off_lineup, se.enhanced_off_lineup,
                    -- Get original PBP data for comparison
                    pbp.points as original_points,
                    pbp.description as original_description
                FROM step4_processed_events se
                LEFT JOIN pbp ON se.pbp_id = pbp.pbp_id
                WHERE se.off_team_id = 1610612745 AND se.points > 0
                ORDER BY se.period, se.pbp_order, se.wall_clock_int
            """).df()

            analysis = {
                "total_hou_scoring_events": len(hou_scoring),
                "total_hou_points_calculated": int(hou_scoring['points'].sum()),
                "scoring_events_by_type": {},
                "potential_issues": [],
                "event_details": []
            }

            # Analyze by scoring event type
            if not hou_scoring.empty:
                by_type = hou_scoring.groupby('msg_type').agg({
                    'points': ['count', 'sum'],
                    'pbp_id': 'count'
                }).round(2)

                analysis["scoring_events_by_type"] = by_type.to_dict()

                # Check for potential issues
                for _, event in hou_scoring.iterrows():
                    event_detail = {
                        "pbp_id": int(event['pbp_id']),
                        "period": int(event['period']),
                        "pbp_order": int(event['pbp_order']),
                        "description": str(event['description']),
                        "points": int(event['points']),
                        "original_points": int(event['original_points']) if pd.notna(event['original_points']) else None,
                        "msg_type": int(event['msg_type']),
                        "has_traditional_lineup": bool(pd.notna(event['traditional_off_lineup']) and 
                                                     str(event['traditional_off_lineup']).strip() not in ['', '[]']),
                        "has_enhanced_lineup": bool(pd.notna(event['enhanced_off_lineup']) and 
                                                  str(event['enhanced_off_lineup']).strip() not in ['', '[]'])
                    }

                    # Flag potential issues
                    if event_detail["points"] != event_detail["original_points"]:
                        analysis["potential_issues"].append(f"Points mismatch for pbp_id {event_detail['pbp_id']}: processed={event_detail['points']}, original={event_detail['original_points']}")

                    if not event_detail["has_traditional_lineup"]:
                        analysis["potential_issues"].append(f"No traditional lineup for scoring event pbp_id {event_detail['pbp_id']}")

                    if not event_detail["has_enhanced_lineup"]:
                        analysis["potential_issues"].append(f"No enhanced lineup for scoring event pbp_id {event_detail['pbp_id']}")

                    analysis["event_details"].append(event_detail)

            # Check for possession attribution
            hou_possession_points = sum(p.points_scored for p in self.dual_possessions if p.off_team_id == 1610612745)
            analysis["possession_attribution"] = {
                "total_possession_points": hou_possession_points,
                "difference_from_events": hou_possession_points - analysis["total_hou_points_calculated"]
            }

            logger.info(f"HOU Scoring Analysis:")
            logger.info(f"  Total Scoring Events: {analysis['total_hou_scoring_events']}")
            logger.info(f"  Total Points from Events: {analysis['total_hou_points_calculated']}")
            logger.info(f"  Total Points from Possessions: {hou_possession_points}")
            logger.info(f"  Potential Issues Found: {len(analysis['potential_issues'])}")

            for issue in analysis["potential_issues"][:10]:  # Log first 10 issues
                logger.warning(f"    {issue}")

            return analysis

        except Exception as e:
            logger.error(f"Error in detailed HOU scoring analysis: {e}")
            return {"error": str(e)}




    def calculate_dual_lineup_stats(self) -> ValidationResult:
        """Calculate lineup statistics for both traditional and enhanced methods"""
        start_time = time.time()

        try:
            logger.info("Calculating dual-method lineup statistics...")

            if not self.dual_possessions:
                return ValidationResult(
                    step_name="Calculate Dual Lineup Stats",
                    passed=False,
                    details="No dual possessions available",
                    processing_time=time.time() - start_time
                )

            # Initialize stats containers
            self.traditional_lineup_stats = {}
            self.enhanced_lineup_stats = {}

            # Process each possession for both methods
            for poss in self.dual_possessions:
                # Traditional method stats
                if poss.traditional_off_lineup and poss.traditional_def_lineup:
                    self._update_lineup_stats(
                        poss, "traditional",
                        poss.traditional_off_lineup, poss.traditional_def_lineup
                    )

                # Enhanced method stats
                if poss.enhanced_off_lineup and poss.enhanced_def_lineup:
                    self._update_lineup_stats(
                        poss, "enhanced", 
                        poss.enhanced_off_lineup, poss.enhanced_def_lineup
                    )

            # Calculate ratings for both methods
            self._calculate_ratings(self.traditional_lineup_stats)
            self._calculate_ratings(self.enhanced_lineup_stats)

            # Add violation context to traditional lineups
            self._add_violation_context()

            trad_count = len(self.traditional_lineup_stats)
            enh_count = len(self.enhanced_lineup_stats)

            details = f"Calculated lineup stats: {trad_count} traditional, {enh_count} enhanced lineups"

            return ValidationResult(
                step_name="Calculate Dual Lineup Stats",
                passed=True,
                details=details,
                data_count=trad_count + enh_count,
                processing_time=time.time() - start_time
            )

        except Exception as e:
            return ValidationResult(
                step_name="Calculate Dual Lineup Stats",
                passed=False,
                details=f"Error calculating dual lineup stats: {str(e)}",
                processing_time=time.time() - start_time
            )

    def _update_lineup_stats(self, poss: DualPossession, method: str, 
                           off_lineup: Tuple[int, ...], def_lineup: Tuple[int, ...]):
        """Update lineup statistics for given method"""
        stats_container = (self.traditional_lineup_stats if method == "traditional" 
                          else self.enhanced_lineup_stats)

        # Offensive lineup
        off_key = (poss.off_team_id, off_lineup)
        if off_key not in stats_container:
            stats_container[off_key] = self._create_lineup_stats(
                poss.off_team_id, off_lineup, method
            )

        off_stats = stats_container[off_key]
        off_stats.off_possessions += 1
        off_stats.points_for += poss.points_scored

        # Defensive lineup
        def_key = (poss.def_team_id, def_lineup)
        if def_key not in stats_container:
            stats_container[def_key] = self._create_lineup_stats(
                poss.def_team_id, def_lineup, method
            )

        def_stats = stats_container[def_key]
        def_stats.def_possessions += 1
        def_stats.points_against += poss.points_scored

    def _create_lineup_stats(self, team_id: int, lineup: Tuple[int, ...], method: str) -> DualLineupStats:
        """Create lineup statistics object"""
        team_abbrev = self.team_abbrev.get(team_id, f"Team{team_id}")
        player_names = [self.player_names.get(pid, f"Player{pid}") for pid in lineup]

        return DualLineupStats(
            team_id=team_id,
            team_abbrev=team_abbrev,
            lineup_method=method,
            player_ids=lineup,
            player_names=player_names,
            lineup_size=len(lineup)
        )

    def _calculate_ratings(self, stats_container: Dict):
        """Calculate offensive/defensive/net ratings"""
        for stats in stats_container.values():
            if stats.off_possessions > 0:
                stats.off_rating = (100.0 * stats.points_for / stats.off_possessions)
            if stats.def_possessions > 0:
                stats.def_rating = (100.0 * stats.points_against / stats.def_possessions)
            stats.net_rating = stats.off_rating - stats.def_rating

    def _add_violation_context(self):
        """Add violation flags to traditional lineup stats"""
        # Group violations by lineup characteristics if possible
        violation_summary = defaultdict(list)

        for violation in self.traditional_violations:
            flag_type = violation.get('flag_type', 'unknown')
            team_id = violation.get('team_id')
            details = violation.get('description', '')

            violation_summary[f"team_{team_id}"].append(f"{flag_type}: {details}")

        # Add violations to lineup stats where applicable
        for (team_id, lineup), stats in self.traditional_lineup_stats.items():
            team_violations = violation_summary.get(f"team_{team_id}", [])
            stats.lineup_violations = team_violations[:5]  # Top 5 violations


    def calculate_dual_player_rim_stats(self) -> ValidationResult:
        """Calculate player rim defense statistics for both methods"""
        start_time = time.time()

        try:
            logger.info("Calculating dual-method player rim defense statistics...")

            if not self.dual_possessions:
                return ValidationResult(
                    step_name="Calculate Dual Player Rim Stats",
                    passed=False,
                    details="No dual possessions available",
                    processing_time=time.time() - start_time
                )

            # Initialize player stats for both methods
            self.traditional_player_stats = {}
            self.enhanced_player_stats = {}

            # Initialize all active players
            if hasattr(self.entities, 'unique_players') and self.entities.unique_players is not None:
                for _, r in self.entities.unique_players.iterrows():
                    pid = int(r["player_id"])

                    # Traditional method player
                    self.traditional_player_stats[pid] = DualPlayerRimStats(
                        player_id=pid,
                        player_name=str(r.get("player_name", pid)),
                        team_id=int(r.get("team_id")) if pd.notna(r.get("team_id")) else None,
                        team_abbrev=str(r.get("team_abbrev")) if pd.notna(r.get("team_abbrev")) else None,
                        method="traditional"
                    )

                    # Enhanced method player
                    self.enhanced_player_stats[pid] = DualPlayerRimStats(
                        player_id=pid,
                        player_name=str(r.get("player_name", pid)),
                        team_id=int(r.get("team_id")) if pd.notna(r.get("team_id")) else None,
                        team_abbrev=str(r.get("team_abbrev")) if pd.notna(r.get("team_abbrev")) else None,
                        method="enhanced"
                    )

            # Count possessions for both methods
            for poss in self.dual_possessions:
                # Traditional method
                if poss.traditional_off_lineup:
                    for pid in poss.traditional_off_lineup:
                        if pid in self.traditional_player_stats:
                            self.traditional_player_stats[pid].off_possessions += 1

                if poss.traditional_def_lineup:
                    for pid in poss.traditional_def_lineup:
                        if pid in self.traditional_player_stats:
                            self.traditional_player_stats[pid].def_possessions += 1

                # Enhanced method
                if poss.enhanced_off_lineup:
                    for pid in poss.enhanced_off_lineup:
                        if pid in self.enhanced_player_stats:
                            self.enhanced_player_stats[pid].off_possessions += 1

                if poss.enhanced_def_lineup:
                    for pid in poss.enhanced_def_lineup:
                        if pid in self.enhanced_player_stats:
                            self.enhanced_player_stats[pid].def_possessions += 1

            # Calculate rim defense stats
            self._calculate_dual_rim_defense()

            trad_with_rim = sum(1 for s in self.traditional_player_stats.values() 
                               if s.opp_rim_attempts_on > 0)
            enh_with_rim = sum(1 for s in self.enhanced_player_stats.values() 
                              if s.opp_rim_attempts_on > 0)

            details = (f"Calculated player rim stats: {trad_with_rim} traditional, "
                      f"{enh_with_rim} enhanced players with rim data")

            return ValidationResult(
                step_name="Calculate Dual Player Rim Stats",
                passed=True,
                details=details,
                data_count=len(self.traditional_player_stats) + len(self.enhanced_player_stats),
                processing_time=time.time() - start_time
            )

        except Exception as e:
            return ValidationResult(
                step_name="Calculate Dual Player Rim Stats",
                passed=False,
                details=f"Error calculating dual player rim stats: {str(e)}",
                processing_time=time.time() - start_time
            )

    def _calculate_dual_rim_defense(self):
        """Calculate rim defense stats using rim attempt events"""
        try:
            rim_events_df = self.conn.execute("""
                SELECT 
                    def_team_id, is_rim_make,
                    traditional_def_lineup, enhanced_def_lineup
                FROM step4_processed_events
                WHERE is_rim_attempt = TRUE AND def_team_id IS NOT NULL
            """).df()
            if rim_events_df.empty:
                logger.debug("No rim events found; skipping rim-defense aggregation.")
                return

            def _as_set(val) -> Set[int]:
                if val is None or (isinstance(val, float) and pd.isna(val)) or str(val).strip() == "":
                    return set()
                try:
                    obj = json.loads(val) if isinstance(val, str) else val
                except Exception:
                    try:
                        obj = ast.literal_eval(str(val))
                    except Exception:
                        logger.debug(f"Rim lineup parse failed; raw={val!r}")
                        return set()
                if not isinstance(obj, (list, tuple, set)):
                    logger.debug(f"Rim lineup had unexpected type; raw={obj!r}")
                    return set()
                out = set()
                for x in obj:
                    try:
                        out.add(int(x))
                    except Exception:
                        logger.debug(f"Rim lineup member non-int; raw={x!r}")
                return out

            for _, row in rim_events_df.iterrows():
                def_team_id = int(row["def_team_id"])
                is_make = bool(row["is_rim_make"])
                trad_def = _as_set(row["traditional_def_lineup"])
                enh_def = _as_set(row["enhanced_def_lineup"])
                roster = self.team_roster.get(def_team_id, set())

                # Traditional: on
                for pid in trad_def:
                    if pid in self.traditional_player_stats:
                        s = self.traditional_player_stats[pid]
                        s.opp_rim_attempts_on += 1
                        if is_make:
                            s.opp_rim_makes_on += 1
                # Traditional: off (rest of roster)
                for pid in (roster - trad_def):
                    if pid in self.traditional_player_stats:
                        s = self.traditional_player_stats[pid]
                        s.opp_rim_attempts_off += 1
                        if is_make:
                            s.opp_rim_makes_off += 1

                # Enhanced: on
                for pid in enh_def:
                    if pid in self.enhanced_player_stats:
                        s = self.enhanced_player_stats[pid]
                        s.opp_rim_attempts_on += 1
                        if is_make:
                            s.opp_rim_makes_on += 1
                # Enhanced: off
                for pid in (roster - enh_def):
                    if pid in self.enhanced_player_stats:
                        s = self.enhanced_player_stats[pid]
                        s.opp_rim_attempts_off += 1
                        if is_make:
                            s.opp_rim_makes_off += 1

            # Percentages (diagnostic only; no fill-ins)
            for s in list(self.traditional_player_stats.values()) + list(self.enhanced_player_stats.values()):
                if s.opp_rim_attempts_on > 0:
                    s.opp_rim_fg_pct_on = s.opp_rim_makes_on / s.opp_rim_attempts_on
                if s.opp_rim_attempts_off > 0:
                    s.opp_rim_fg_pct_off = s.opp_rim_makes_off / s.opp_rim_attempts_off
                if s.opp_rim_fg_pct_on is not None and s.opp_rim_fg_pct_off is not None:
                    s.rim_defense_on_off = s.opp_rim_fg_pct_on - s.opp_rim_fg_pct_off

        except Exception as e:
            logger.warning(f"Error calculating rim defense: {e}")

    def create_dual_method_tables(self) -> ValidationResult:
        """Create comprehensive tables for both traditional and enhanced methods"""
        start_time = time.time()

        try:
            logger.info("Creating dual-method output tables...")

            if self.conn is None:
                self.conn = duckdb.connect(self.db_path)

            # Create traditional lineups table
            trad_lineups_data = []
            for (team_id, lineup), stats in self.traditional_lineup_stats.items():
                # Pad lineup to 5 players if needed, or truncate if more
                padded_lineup = list(lineup) + [None] * (5 - len(lineup))
                padded_names = list(stats.player_names) + [""] * (5 - len(stats.player_names))

                row = {
                    "method": "traditional",
                    "team_id": team_id,
                    "team_abbrev": stats.team_abbrev,
                    "lineup_size": stats.lineup_size,
                    "player_1_id": padded_lineup[0],
                    "player_1_name": padded_names[0] if padded_names[0] else "",
                    "player_2_id": padded_lineup[1] if len(padded_lineup) > 1 else None,
                    "player_2_name": padded_names[1] if len(padded_names) > 1 and padded_names[1] else "",
                    "player_3_id": padded_lineup[2] if len(padded_lineup) > 2 else None,
                    "player_3_name": padded_names[2] if len(padded_names) > 2 and padded_names[2] else "",
                    "player_4_id": padded_lineup[3] if len(padded_lineup) > 3 else None,
                    "player_4_name": padded_names[3] if len(padded_names) > 3 and padded_names[3] else "",
                    "player_5_id": padded_lineup[4] if len(padded_lineup) > 4 else None,
                    "player_5_name": padded_names[4] if len(padded_names) > 4 and padded_names[4] else "",
                    "off_possessions": stats.off_possessions,
                    "def_possessions": stats.def_possessions,
                    "points_for": stats.points_for,
                    "points_against": stats.points_against,
                    "off_rating": round(stats.off_rating, 1),
                    "def_rating": round(stats.def_rating, 1),
                    "net_rating": round(stats.net_rating, 1),
                    "violation_count": len(stats.lineup_violations),
                    "violation_summary": "; ".join(stats.lineup_violations[:3])  # Top 3 violations
                }
                trad_lineups_data.append(row)

            # Create enhanced lineups table
            enh_lineups_data = []
            for (team_id, lineup), stats in self.enhanced_lineup_stats.items():
                row = {
                    "method": "enhanced",
                    "team_id": team_id,
                    "team_abbrev": stats.team_abbrev,
                    "lineup_size": stats.lineup_size,
                    "player_1_id": lineup[0],
                    "player_1_name": stats.player_names[0],
                    "player_2_id": lineup[1],
                    "player_2_name": stats.player_names[1],
                    "player_3_id": lineup[2],
                    "player_3_name": stats.player_names[2],
                    "player_4_id": lineup[3],
                    "player_4_name": stats.player_names[3],
                    "player_5_id": lineup[4],
                    "player_5_name": stats.player_names[4],
                    "off_possessions": stats.off_possessions,
                    "def_possessions": stats.def_possessions,
                    "points_for": stats.points_for,
                    "points_against": stats.points_against,
                    "off_rating": round(stats.off_rating, 1),
                    "def_rating": round(stats.def_rating, 1),
                    "net_rating": round(stats.net_rating, 1),
                    "violation_count": 0,  # Enhanced method maintains 5-man lineups
                    "violation_summary": ""
                }
                enh_lineups_data.append(row)

            # Combine and create unified lineups table
            all_lineups_data = trad_lineups_data + enh_lineups_data
            lineups_df = pd.DataFrame(all_lineups_data)

            if not lineups_df.empty:
                self.conn.register("dual_lineups_temp", lineups_df)
                try:
                    self.conn.execute("""
                        CREATE OR REPLACE TABLE final_dual_lineups AS
                        SELECT * FROM dual_lineups_temp
                        ORDER BY method, team_abbrev, off_possessions DESC
                    """)
                finally:
                    self.conn.unregister("dual_lineups_temp")

            # Create dual players table
            trad_players_data = []
            for pid, stats in self.traditional_player_stats.items():
                trad_players_data.append({
                    "method": "traditional",
                    "player_id": pid,
                    "player_name": stats.player_name,
                    "team_id": stats.team_id,
                    "team_abbrev": stats.team_abbrev,
                    "off_possessions": stats.off_possessions,
                    "def_possessions": stats.def_possessions,
                    "opp_rim_attempts_on": stats.opp_rim_attempts_on,
                    "opp_rim_makes_on": stats.opp_rim_makes_on,
                    "opp_rim_attempts_off": stats.opp_rim_attempts_off,
                    "opp_rim_makes_off": stats.opp_rim_makes_off,
                    "opp_rim_fg_pct_on": round(stats.opp_rim_fg_pct_on, 3) if stats.opp_rim_fg_pct_on is not None else None,
                    "opp_rim_fg_pct_off": round(stats.opp_rim_fg_pct_off, 3) if stats.opp_rim_fg_pct_off is not None else None,
                    "rim_defense_on_off": round(stats.rim_defense_on_off, 3) if stats.rim_defense_on_off is not None else None
                })

            enh_players_data = []
            for pid, stats in self.enhanced_player_stats.items():
                enh_players_data.append({
                    "method": "enhanced",
                    "player_id": pid,
                    "player_name": stats.player_name,
                    "team_id": stats.team_id,
                    "team_abbrev": stats.team_abbrev,
                    "off_possessions": stats.off_possessions,
                    "def_possessions": stats.def_possessions,
                    "opp_rim_attempts_on": stats.opp_rim_attempts_on,
                    "opp_rim_makes_on": stats.opp_rim_makes_on,
                    "opp_rim_attempts_off": stats.opp_rim_attempts_off,
                    "opp_rim_makes_off": stats.opp_rim_makes_off,
                    "opp_rim_fg_pct_on": round(stats.opp_rim_fg_pct_on, 3) if stats.opp_rim_fg_pct_on is not None else None,
                    "opp_rim_fg_pct_off": round(stats.opp_rim_fg_pct_off, 3) if stats.opp_rim_fg_pct_off is not None else None,
                    "rim_defense_on_off": round(stats.rim_defense_on_off, 3) if stats.rim_defense_on_off is not None else None
                })

            all_players_data = trad_players_data + enh_players_data
            players_df = pd.DataFrame(all_players_data)

            if not players_df.empty:
                self.conn.register("dual_players_temp", players_df)
                try:
                    self.conn.execute("""
                        CREATE OR REPLACE TABLE final_dual_players AS
                        SELECT * FROM dual_players_temp
                        ORDER BY method, team_abbrev, player_name
                    """)
                finally:
                    self.conn.unregister("dual_players_temp")


            # Create method comparison summary
            self._create_method_comparison_table()

            # Create violation reports
            self._create_violation_reports()

            details = (f"Created dual-method tables: lineups({len(lineups_df)}), "
                      f"players({len(players_df)}), comparisons, violations")

            return ValidationResult(
                step_name="Create Dual Method Tables",
                passed=True,
                details=details,
                data_count=len(lineups_df) + len(players_df),
                processing_time=time.time() - start_time
            )

        except Exception as e:
            return ValidationResult(
                step_name="Create Dual Method Tables",
                passed=False,
                details=f"Error creating dual-method tables: {str(e)}",
                processing_time=time.time() - start_time
            )

    def _create_method_comparison_table(self):
        """Create comprehensive method comparison table"""
        try:
            # Calculate comparison metrics
            trad_total_lineups = len(self.traditional_lineup_stats)
            enh_total_lineups = len(self.enhanced_lineup_stats)

            trad_5man_lineups = sum(1 for stats in self.traditional_lineup_stats.values() 
                                   if stats.lineup_size == 5)
            enh_5man_lineups = sum(1 for stats in self.enhanced_lineup_stats.values() 
                                  if stats.lineup_size == 5)

            trad_avg_possessions = np.mean([stats.off_possessions + stats.def_possessions 
                                          for stats in self.traditional_lineup_stats.values()])
            enh_avg_possessions = np.mean([stats.off_possessions + stats.def_possessions 
                                         for stats in self.enhanced_lineup_stats.values()])

            comparison_data = [
                {
                    "metric": "Total Lineups",
                    "traditional_value": trad_total_lineups,
                    "enhanced_value": enh_total_lineups,
                    "difference": enh_total_lineups - trad_total_lineups,
                    "better_method": "Enhanced" if enh_total_lineups < trad_total_lineups else "Traditional"
                },
                {
                    "metric": "5-Man Lineups",
                    "traditional_value": trad_5man_lineups,
                    "enhanced_value": enh_5man_lineups,
                    "difference": enh_5man_lineups - trad_5man_lineups,
                    "better_method": "Enhanced" if enh_5man_lineups > trad_5man_lineups else "Traditional"
                },
                {
                    "metric": "5-Man Percentage",
                    "traditional_value": round(100 * trad_5man_lineups / max(1, trad_total_lineups), 1),
                    "enhanced_value": round(100 * enh_5man_lineups / max(1, enh_total_lineups), 1),
                    "difference": round(100 * (enh_5man_lineups / max(1, enh_total_lineups) - 
                                             trad_5man_lineups / max(1, trad_total_lineups)), 1),
                    "better_method": "Enhanced"
                },
                {
                    "metric": "Avg Possessions per Lineup",
                    "traditional_value": round(trad_avg_possessions, 1),
                    "enhanced_value": round(enh_avg_possessions, 1),
                    "difference": round(enh_avg_possessions - trad_avg_possessions, 1),
                    "better_method": "Enhanced" if enh_avg_possessions > trad_avg_possessions else "Traditional"
                },
                {
                    "metric": "Violation Flags",
                    "traditional_value": len(self.traditional_violations),
                    "enhanced_value": len(self.enhanced_violations),
                    "difference": len(self.enhanced_violations) - len(self.traditional_violations),
                    "better_method": "Traditional" if len(self.traditional_violations) < len(self.enhanced_violations) else "Enhanced"
                }
            ]

            comparison_df = pd.DataFrame(comparison_data)
            self.conn.register("comparison_temp", comparison_df)
            self.conn.execute("""
                CREATE OR REPLACE TABLE method_comparison_summary AS
                SELECT * FROM comparison_temp
            """)
            self.conn.unregister("comparison_temp")

        except Exception as e:
            logger.warning(f"Error creating method comparison: {e}")

    def _create_violation_reports(self):
        """Create detailed violation reports for both methods with normalized schema and stable ordering.

        Normalization rules (no synthesis of unknown values):
        - Add 'order_ts' = first available of ['abs_time','wall_clock_int','pbp_order'] (as-is).
        - Ensure 'period' exists (if not, leave absent).
        - Ensure 'team_abbrev' exists; if only 'team_id' present, map using self.team_abbrev (deterministic).
        - Ensure 'player_name' exists; if only 'player_id' present, map using self.player_names (deterministic).
        - Pass through 'flag_type','description','flag_details' if present; omit if not.
        """
        try:
            def _normalize(df_in: pd.DataFrame) -> pd.DataFrame:
                if df_in is None or df_in.empty:
                    return pd.DataFrame()

                df = df_in.copy()

                cols = set(c.lower() for c in df.columns)

                # Standardize casing to avoid surprises later
                rename_map = {c: c for c in df.columns}  # identity
                # common alternates could be added here if you encounter them:
                # e.g., rename_map['team'] = 'team_abbrev' if 'team' in df

                df = df.rename(columns=rename_map)

                # Build order_ts without assuming existence
                order_candidates = [c for c in ["abs_time", "wall_clock_int", "pbp_order"] if c in df.columns]
                if order_candidates:
                    first = order_candidates[0]
                    df["order_ts"] = df[first]
                # else: no order column at all; exporter will handle lack of ordering gracefully

                # team_abbrev from team_id if needed
                if "team_abbrev" not in df.columns and "team_id" in df.columns and self.team_abbrev:
                    df["team_abbrev"] = df["team_id"].map(lambda tid: self.team_abbrev.get(int(tid)) if pd.notna(tid) else None)

                # player_name from player_id if needed
                if "player_name" not in df.columns and "player_id" in df.columns and self.player_names:
                    df["player_name"] = df["player_id"].map(lambda pid: self.player_names.get(int(pid)) if pd.notna(pid) else None)

                # Keep only a sensible set + anything extra that came in (don’t drop unknowns)
                preferred_order = [
                    "period", "order_ts", "abs_time", "wall_clock_int", "pbp_order",
                    "team_abbrev", "team_id", "flag_type", "player_name", "player_id",
                    "description", "flag_details"
                ]
                # Reorder columns: known first, then the rest in original order
                known = [c for c in preferred_order if c in df.columns]
                rest  = [c for c in df.columns if c not in known]
                df = df[known + rest]

                return df

            def _write_table(temp_name: str, final_name: str, df: pd.DataFrame):
                if df is None or df.empty:
                    return
                self.conn.register(temp_name, df)
                try:
                    cols = [c[1] for c in self.conn.execute(f"PRAGMA table_info('{temp_name}')").fetchall()]
                    order_expr = "order_ts" if "order_ts" in cols else None
                    if order_expr:
                        self.conn.execute(f"""
                            CREATE OR REPLACE TABLE {final_name} AS
                            SELECT * FROM {temp_name}
                            ORDER BY {order_expr}
                        """)
                    else:
                        self.conn.execute(f"""
                            CREATE OR REPLACE TABLE {final_name} AS
                            SELECT * FROM {temp_name}
                        """)
                finally:
                    self.conn.unregister(temp_name)

            # Traditional
            if self.traditional_violations:
                trad_df = _normalize(pd.DataFrame(self.traditional_violations))
                _write_table("trad_violations_temp", "traditional_violation_report", trad_df)

            # Enhanced
            if self.enhanced_violations:
                enh_df = _normalize(pd.DataFrame(self.enhanced_violations))
                _write_table("enh_violations_temp", "enhanced_violation_report", enh_df)

        except Exception as e:
            logger.warning(f"Error creating violation reports: {e}")



    def print_dual_method_summary(self):
        """Print comprehensive summary of both methods"""
        print("\n" + "="*80)
        print("NBA PIPELINE - STEP 5 DUAL-METHOD SUMMARY")
        print("="*80)

        print("POSSESSION ANALYSIS:")
        print(f"  Total Dual Possessions: {len(self.dual_possessions):,}")
        if self.dual_possessions:
            total_points = sum(p.points_scored for p in self.dual_possessions)
            periods = len(set(p.period for p in self.dual_possessions))
            print(f"  Total Points: {total_points:,}")
            print(f"  Periods: {periods}")

        print("\nTRADITIONAL METHOD RESULTS:")
        print(f"  Unique Lineups: {len(self.traditional_lineup_stats):,}")
        if self.traditional_lineup_stats:
            trad_5man = sum(1 for s in self.traditional_lineup_stats.values() if s.lineup_size == 5)
            print(f"  5-Man Lineups: {trad_5man:,} ({100*trad_5man/len(self.traditional_lineup_stats):.1f}%)")
            trad_violations = len(self.traditional_violations)
            print(f"  Violation Flags: {trad_violations:,}")

        print("\nENHANCED METHOD RESULTS:")
        print(f"  Unique Lineups: {len(self.enhanced_lineup_stats):,}")
        if self.enhanced_lineup_stats:
            enh_5man = sum(1 for s in self.enhanced_lineup_stats.values() if s.lineup_size == 5)
            print(f"  5-Man Lineups: {enh_5man:,} ({100*enh_5man/len(self.enhanced_lineup_stats):.1f}%)")
            enh_violations = len(self.enhanced_violations)
            print(f"  Violation Flags: {enh_violations:,}")

        print("\nPLAYER RIM DEFENSE:")
        trad_with_rim = sum(1 for s in self.traditional_player_stats.values() if s.opp_rim_attempts_on > 0)
        enh_with_rim = sum(1 for s in self.enhanced_player_stats.values() if s.opp_rim_attempts_on > 0)
        print(f"  Traditional Players with Rim Data: {trad_with_rim:,}")
        print(f"  Enhanced Players with Rim Data: {enh_with_rim:,}")

        print("\nMETHOD EFFECTIVENESS:")
        if self.traditional_lineup_stats and self.enhanced_lineup_stats:
            improvement = len(self.traditional_lineup_stats) - len(self.enhanced_lineup_stats)
            print(f"  Lineup Count Change: {improvement:+,} (Enhanced has fewer unique lineups)")

            trad_5_pct = 100 * sum(1 for s in self.traditional_lineup_stats.values() if s.lineup_size == 5) / len(self.traditional_lineup_stats)
            enh_5_pct = 100 * sum(1 for s in self.enhanced_lineup_stats.values() if s.lineup_size == 5) / len(self.enhanced_lineup_stats)
            print(f"  5-Man Accuracy: Traditional {trad_5_pct:.1f}% vs Enhanced {enh_5_pct:.1f}%")

        print("="*80)

    def create_project_output_tables(self) -> ValidationResult:
        """
        Project deliverables (no fallbacks):
        - project1_lineups: 5-man lineups per team (ENHANCED ONLY) with Off/Def/Net ratings.
        - project2_players: player-level rim defense on/off (ENHANCED ONLY).
        """
        start_time = time.time()
        try:
            # --- Project 1: 5-man lineups (enhanced only) ---
            p1_rows = []
            for (team_id, lineup), stats in self.enhanced_lineup_stats.items():
                # Enhanced is guaranteed 5
                if len(lineup) != 5:
                    continue
                row = {
                    "team_id": team_id,
                    "team_abbrev": stats.team_abbrev,
                    "player_1_id": lineup[0], "player_1_name": stats.player_names[0],
                    "player_2_id": lineup[1], "player_2_name": stats.player_names[1],
                    "player_3_id": lineup[2], "player_3_name": stats.player_names[2],
                    "player_4_id": lineup[3], "player_4_name": stats.player_names[3],
                    "player_5_id": lineup[4], "player_5_name": stats.player_names[4],
                    "off_possessions": stats.off_possessions,
                    "def_possessions": stats.def_possessions,
                    "off_rating": round(stats.off_rating, 1),
                    "def_rating": round(stats.def_rating, 1),
                    "net_rating": round(stats.net_rating, 1),
                }
                p1_rows.append(row)
            p1_df = pd.DataFrame(p1_rows).sort_values(
                ["team_abbrev", "off_possessions"], ascending=[True, False]
            )
            if not p1_df.empty:
                self.conn.register("project1_temp", p1_df)
                try:
                    self.conn.execute("""
                        CREATE OR REPLACE TABLE project1_lineups AS
                        SELECT * FROM project1_temp
                    """)
                finally:
                    self.conn.unregister("project1_temp")

            # --- Project 2: player rim defense on/off (enhanced only) ---
            p2_rows = []
            for pid, s in self.enhanced_player_stats.items():
                p2_rows.append({
                    "player_id": pid,
                    "player_name": s.player_name,
                    "team_id": s.team_id,
                    "team_abbrev": s.team_abbrev,
                    "off_possessions": s.off_possessions,
                    "def_possessions": s.def_possessions,
                    "opp_rim_attempts_on": s.opp_rim_attempts_on,
                    "opp_rim_makes_on": s.opp_rim_makes_on,
                    "opp_rim_attempts_off": s.opp_rim_attempts_off,
                    "opp_rim_makes_off": s.opp_rim_makes_off,
                    "opp_rim_fg_pct_on": round(s.opp_rim_fg_pct_on, 3) if s.opp_rim_fg_pct_on is not None else None,
                    "opp_rim_fg_pct_off": round(s.opp_rim_fg_pct_off, 3) if s.opp_rim_fg_pct_off is not None else None,
                    "rim_defense_on_off": round(s.rim_defense_on_off, 3) if s.rim_defense_on_off is not None else None,
                })
            p2_df = pd.DataFrame(p2_rows).sort_values(["team_abbrev", "player_name"])
            if not p2_df.empty:
                self.conn.register("project2_temp", p2_df)
                try:
                    self.conn.execute("""
                        CREATE OR REPLACE TABLE project2_players AS
                        SELECT * FROM project2_temp
                    """)
                finally:
                    self.conn.unregister("project2_temp")

            details = (f"Project outputs created: "
                    f"project1_lineups({len(p1_df)}), project2_players({len(p2_df)})")
            return ValidationResult(
                step_name="Create Project Output Tables",
                passed=True,
                details=details,
                data_count=len(p1_df) + len(p2_df),
                processing_time=time.time() - start_time
            )
        except Exception as e:
            return ValidationResult(
                step_name="Create Project Output Tables",
                passed=False,
                details=f"Error creating project outputs: {e}",
                processing_time=time.time() - start_time
            )


def run_dual_method_possession_engine(db_path: str = None,
                                    entities: GameEntities = None) -> Tuple[bool, DualMethodPossessionEngine]:
    """Run the complete dual-method possession engine pipeline"""

    print("NBA Pipeline - UPDATED Step 5: Dual-Method Possession Engine")
    print("="*70)

    if entities is None:
        logger.error("GameEntities required for dual-method possession engine")
        return False, None

    with DualMethodPossessionEngine(db_path, entities) as engine:

        # Run comprehensive diagnostic first
        logger.info("Step 5a: Running pipeline diagnostic...")
        diagnostic = engine.diagnose_pipeline_state()
        logger.info(f"Pipeline diagnostic completed: {diagnostic}")

        # Load dual-method data from Step 2/4 (autorun rebuild ON)
        logger.info("Step 5b: Loading dual-method data...")
        result = engine.load_dual_method_data(autorun_rebuild=True)
        engine.validator.log_validation(result)
        if not result.passed:
            return False, engine

        # Identify possessions with dual lineup contexts
        logger.info("Step 5c: Identifying dual-method possessions...")
        result = engine.identify_dual_possessions()
        engine.validator.log_validation(result)
        if not result.passed:
            return False, engine

        # Calculate lineup statistics for both methods
        logger.info("Step 5d: Calculating dual-method lineup statistics...")
        result = engine.calculate_dual_lineup_stats()
        engine.validator.log_validation(result)

        # Calculate player rim statistics for both methods
        logger.info("Step 5e: Calculating dual-method player rim statistics...")
        result = engine.calculate_dual_player_rim_stats()
        engine.validator.log_validation(result)

        # Run comprehensive debugging to identify points discrepancy
        logger.info("Step 5f: Running comprehensive points flow debugging...")
        debug_info = engine.debug_points_flow_comprehensive()
        logger.info(f"Points flow debug completed: {debug_info}")

        # Run detailed HOU analysis
        logger.info("Step 5g: Running detailed HOU scoring events analysis...")
        hou_analysis = engine.debug_hou_scoring_events_detailed()
        logger.info(f"HOU analysis completed: {hou_analysis}")

        # Create comprehensive output tables (dual-method)
        logger.info("Step 5f: Creating dual-method output tables...")
        result = engine.create_dual_method_tables()
        engine.validator.log_validation(result)

        # ---> NEW: Project deliverables (enhanced-only) <---
        logger.info("Step 5g: Creating project deliverable tables (enhanced only)...")
        result = engine.create_project_output_tables()
        engine.validator.log_validation(result)

        # Print summary
        engine.print_dual_method_summary()

        success = engine.validator.print_validation_summary()
        return success, engine





if __name__ == "__main__":
    from eda.data.nba_entities_extractor import extract_all_entities_robust

    database_path = "mavs_enhanced.duckdb"
    ok, entities = extract_all_entities_robust(database_path)

    if ok:
        success, engine = run_dual_method_possession_engine(database_path, entities)
        if success:
            print("\nUPDATED Step 5 Complete: Dual-method possession engine")
            print("Both Traditional and Enhanced statistics calculated")
            print("Ready for comprehensive dual-method export (Step 6)")
        else:
            print("\nUPDATED Step 5 Failed: Review validation messages")
    else:
        print("Failed to get entities - cannot proceed")
