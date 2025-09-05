# Step 4: Play-by-Play Processing & Lineup State Machine (Updated with Step 2 Integration)
"""
NBA Pipeline - Step 4: Process PBP Events & Track Lineups (UPDATED)
==================================================================

UPDATED to integrate Step 2 findings:
- Traditional Data-Driven Method: Follows raw substitution data strictly (3-6 man lineups)
- Enhanced Estimation Method: Uses intelligent inference to maintain 5-man lineups
- Both methods run in parallel to provide comparison and validation
- Comprehensive flagging system from Step 2 integrated
- Config-driven approach for easy switching between methods

Key Integration Points from Step 2:
1. Traditional method: msgType=8, playerId1=IN, playerId2=OUT, allows variable lineup sizes
2. Enhanced method: First-action rules, inactivity detection, always-5 enforcement
3. Comprehensive flagging and validation system
4. Both methods track the same events but with different lineup management strategies
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
import json
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque

from eda.utils.nba_pipeline_analysis import NBADataValidator, ValidationResult
from eda.data.nba_entities_extractor import GameEntities

# Load configuration
try:
    from utils.config import (
        NBA_SUBSTITUTION_CONFIG,
        RIM_DISTANCE_FEET,
        COORDINATE_SCALE,
        MINIMUM_SECONDS_PLAYED,
        DUCKDB_PATH
    )
    CONFIG = NBA_SUBSTITUTION_CONFIG
    RIM_THRESHOLD = RIM_DISTANCE_FEET
    COORD_SCALE = COORDINATE_SCALE
    MIN_SECONDS = MINIMUM_SECONDS_PLAYED
    DB_PATH = str(DUCKDB_PATH)
except ImportError:
    logger.warning("Config not available, using defaults")
    CONFIG = {
        "starter_reset_periods": [1, 3],
        "msg_types": {"substitution": 8, "shot_made": 1, "shot_missed": 2, "rebound": 4},
        "one_direction": {"enabled": True, "appearance_via_last_name": True},
        "validation": {"validate_team_membership": True, "min_lineup_size": 5},
        "debug": {"log_all_substitutions": True}
    }
    RIM_THRESHOLD = 4.0
    COORD_SCALE = 10.0
    MIN_SECONDS = 30
    DB_PATH = "mavs_enhanced.duckdb"

logger = logging.getLogger(__name__)


@dataclass
class TraditionalLineupState:
    """Traditional data-driven lineup state - follows raw data strictly"""
    team_lineups: Dict[int, Set[int]] = field(default_factory=dict)
    period: int = 0
    flags: List[Dict] = field(default_factory=list)
    substitution_log: List[Dict] = field(default_factory=list)

    def add_flag(self, flag_type: str, team_id: int, player_id: int = None, details: str = ""):
        """Add a flag for data quality issues"""
        self.flags.append({
            'type': flag_type,
            'team_id': team_id,
            'player_id': player_id,
            'details': details,
            'period': self.period
        })


@dataclass
class EnhancedLineupState:
    """Enhanced lineup state with intelligent inference - maintains 5-man lineups"""
    team_lineups: Dict[int, Set[int]] = field(default_factory=dict)
    period: int = 0
    last_action_time: Dict[int, float] = field(default_factory=dict)
    recent_out: Dict[int, deque] = field(default_factory=dict)
    flags: List[Dict] = field(default_factory=list)
    substitution_log: List[Dict] = field(default_factory=list)
    first_action_events: List[Dict] = field(default_factory=list)
    auto_out_events: List[Dict] = field(default_factory=list)

    def __post_init__(self):
        for team_id in self.team_lineups.keys():
            self.recent_out[team_id] = deque(maxlen=10)

    def add_flag(self, flag_type: str, team_id: int, player_id: int = None, details: str = ""):
        """Add a flag for tracking intelligent inference actions"""
        self.flags.append({
            'type': flag_type,
                    'team_id': team_id,
            'player_id': player_id,
            'details': details,
            'period': self.period
        })


@dataclass  
class ProcessedEvent:
    """Represents a processed play-by-play event with context from both methods"""
    pbp_id: int
    period: int
    pbp_order: int
    wall_clock_int: int
    msg_type: int
    action_type: int = None
    description: str = ""
    off_team_id: int = None
    def_team_id: int = None
    player_id_1: int = None
    player_id_2: int = None
    player_id_3: int = None
    loc_x: int = None
    loc_y: int = None
    points: int = 0

    # Computed fields
    is_shot: bool = False
    is_rim_attempt: bool = False
    is_rim_make: bool = False
    distance_ft: float = None
    is_substitution: bool = False
    sub_out_player: int = None
    sub_in_player: int = None

    # Lineup context from BOTH methods
    traditional_off_lineup: Tuple[int, ...] = None
    traditional_def_lineup: Tuple[int, ...] = None
    enhanced_off_lineup: Tuple[int, ...] = None
    enhanced_def_lineup: Tuple[int, ...] = None


class PBPProcessor:
    """UPDATED: Integrated processor using both Step 2 methods"""

    def __init__(self, db_path: str = None, entities: GameEntities = None):
        """Initialize with both traditional and enhanced tracking methods"""
        self.db_path = db_path or DB_PATH
        self.conn = None
        self.entities = entities
        self.validator = NBADataValidator()

        # DUAL STATE TRACKING - Both methods run in parallel
        self.traditional_state = TraditionalLineupState()
        self.enhanced_state = EnhancedLineupState()

        self.processed_events: List[ProcessedEvent] = []

        # Build team rosters and reference data
        self.team_rosters = self._build_team_rosters_from_entities()
        self.player_names = self._build_player_names()
        self.team_names = self._build_team_names()

        # Statistics tracking
        self.traditional_stats = {
            'substitutions': 0, 'flags': 0, 'lineup_size_deviations': 0
        }
        self.enhanced_stats = {
            'substitutions': 0, 'first_actions': 0, 'auto_outs': 0, 'flags': 0
        }

    def _build_team_rosters_from_entities(self) -> Dict[int, Set[int]]:
        """Build complete team rosters from entities"""
        rosters = {}
        if hasattr(self.entities, 'unique_players') and self.entities.unique_players is not None:
            for _, player in self.entities.unique_players.iterrows():
                team_id = int(player['team_id'])
                player_id = int(player['player_id'])
                if team_id not in rosters:
                    rosters[team_id] = set()
                rosters[team_id].add(player_id)
        return rosters

    def _build_player_names(self) -> Dict[int, str]:
        """Build player ID to name mapping"""
        names = {}
        if hasattr(self.entities, 'unique_players') and self.entities.unique_players is not None:
            for _, player in self.entities.unique_players.iterrows():
                names[int(player['player_id'])] = str(player['player_name'])
        return names

    def _build_team_names(self) -> Dict[int, str]:
        """Build team ID to abbreviation mapping"""
        if hasattr(self.entities, 'team_mapping'):
            return {int(k): str(v) for k, v in self.entities.team_mapping.items() if isinstance(k, int)}
        return {}

    def __enter__(self):
        self.conn = duckdb.connect(self.db_path)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn:
            self.conn.close()

    def initialize_lineups(self) -> ValidationResult:
        """Initialize starting lineups for BOTH tracking methods"""
        start_time = time.time()

        try:
            logger.info("Initializing lineups for both tracking methods...")

            if not self.entities.starters:
                return ValidationResult(
                    step_name="Initialize Lineups",
                    passed=False,
                    details="No starting lineups available in entities",
                    processing_time=time.time() - start_time
                )

            warnings = []

            # Initialize both states with same starting lineups
            for team_abbrev, starters_list in self.entities.starters.items():
                if isinstance(starters_list, list):
                    # Find team ID
                    team_id = None
                    for tid, tabbrev in self.entities.team_mapping.items():
                        if isinstance(tid, int) and tabbrev == team_abbrev:
                            team_id = tid
                            break

                    if team_id is None:
                        warnings.append(f"Could not find team ID for {team_abbrev}")
                        continue

                    starter_ids = {starter['player_id'] for starter in starters_list}

                    if len(starter_ids) != 5:
                        warnings.append(f"Team {team_abbrev} has {len(starter_ids)} starters (expected 5)")

                    # Set for BOTH methods
                    self.traditional_state.team_lineups[team_id] = starter_ids.copy()
                    self.enhanced_state.team_lineups[team_id] = starter_ids.copy()

                    # Initialize enhanced state tracking
                    self.enhanced_state.recent_out[team_id] = deque(maxlen=10)
                    for player_id in starter_ids:
                        self.enhanced_state.last_action_time[player_id] = 0.0

                    logger.info(f"Initialized {team_abbrev} starters: {sorted(starter_ids)}")

            details = f"Initialized lineups for both tracking methods: {len(self.traditional_state.team_lineups)} teams"

            return ValidationResult(
                step_name="Initialize Lineups",
                passed=True,
                details=details,
                processing_time=time.time() - start_time,
                warnings=warnings
            )

        except Exception as e:
            return ValidationResult(
                step_name="Initialize Lineups",
                passed=False,
                details=f"Error initializing lineups: {str(e)}",
                processing_time=time.time() - start_time
            )

    def load_pbp_events(self) -> ValidationResult:
        """Load PBP events with Step 2 integration"""
        start_time = time.time()

        try:
            logger.info("Loading PBP events with Step 2 classification...")

            # Use same canonical view approach from Step 2
            pbp_view = self._ensure_canonical_pbp_view()

            events_df = self.conn.execute(f"""
            SELECT 
                    pbp_id, period, pbp_order, wall_clock_int,
                    game_clock, description, msg_type, action_type,
                    off_team_id, def_team_id,
                    player_id_1, player_id_2, player_id_3,
                    loc_x, loc_y, points
            FROM {pbp_view}
            ORDER BY period, pbp_order, wall_clock_int
            """).df()

            if len(events_df) == 0:
                return ValidationResult(
                    step_name="Load PBP Events", 
                    passed=False,
                    details="No valid PBP events found",
                    processing_time=time.time() - start_time
                )

            # Convert to ProcessedEvent objects with Step 2 classification
            self.processed_events = []
            for _, row in events_df.iterrows():
                event = ProcessedEvent(
                    pbp_id=int(row['pbp_id']),
                    period=int(row['period']),
                    pbp_order=int(row['pbp_order']),
                    wall_clock_int=int(row['wall_clock_int']) if pd.notna(row['wall_clock_int']) else 0,
                    msg_type=int(row['msg_type']),
                    action_type=int(row['action_type']) if pd.notna(row['action_type']) else None,
                    description=str(row['description']) if pd.notna(row['description']) else "",
                    off_team_id=int(row['off_team_id']) if pd.notna(row['off_team_id']) else None,
                    def_team_id=int(row['def_team_id']) if pd.notna(row['def_team_id']) else None,
                    player_id_1=int(row['player_id_1']) if pd.notna(row['player_id_1']) else None,
                    player_id_2=int(row['player_id_2']) if pd.notna(row['player_id_2']) else None,
                    player_id_3=int(row['player_id_3']) if pd.notna(row['player_id_3']) else None,
                    loc_x=int(row['loc_x']) if pd.notna(row['loc_x']) and row['loc_x'] != 0 else None,
                    loc_y=int(row['loc_y']) if pd.notna(row['loc_y']) and row['loc_y'] != 0 else None,
                    points=int(row['points']) if pd.notna(row['points']) else 0
                )

                # Classify event using Step 2 logic
                self._classify_event_step2(event)
                self.processed_events.append(event)

            details = f"Loaded {len(self.processed_events)} events with Step 2 classification"

            return ValidationResult(
                step_name="Load PBP Events",
                passed=True,
                details=details,
                data_count=len(self.processed_events),
                processing_time=time.time() - start_time
            )

        except Exception as e:
            return ValidationResult(
                step_name="Load PBP Events",
                passed=False,
                details=f"Error loading events: {str(e)}",
                processing_time=time.time() - start_time
            )

    def _ensure_canonical_pbp_view(self) -> str:
        """Create canonical PBP view with robust column detection (no nulling)."""
        # Discover a pbp-like table (prefer enriched if present)
        tables = [r[0] for r in self.conn.execute("""
            SELECT table_name FROM information_schema.tables
            WHERE table_schema NOT IN ('information_schema','pg_catalog')
            ORDER BY table_name
        """).fetchall()]

        pbp_candidates = [t for t in tables if t.lower() in ("pbp_enriched", "pbp")]
        if pbp_candidates:
            pbp_table = pbp_candidates[0]
        else:
            # last resort heuristic
            pbp_table = next((t for t in tables if "pbp" in t.lower()), None) or "pbp"

        # Inspect columns to build a robust SELECT
        cols = {r[1].lower(): r[1] for r in self.conn.execute(f"PRAGMA table_info('{pbp_table}')").fetchall()}

        # Map required fields with safe coalesces
        period_col        = cols.get("period",        "period")
        pbp_order_col     = cols.get("pbp_order",     "pbp_order")
        wall_clock_col    = cols.get("wall_clock_int","wall_clock_int")
        game_clock_col    = cols.get("game_clock",    "game_clock")
        desc_col          = cols.get("description",   "description")
        msg_type_col      = cols.get("msg_type",      "msg_type")
        action_type_col   = cols.get("action_type",   "action_type")
        off_team_col      = cols.get("team_id_off",   cols.get("off_team_id", "team_id_off"))
        def_team_col      = cols.get("team_id_def",   cols.get("def_team_id", "team_id_def"))
        p1_col            = cols.get("player_id_1",   "player_id_1")
        p2_col            = cols.get("player_id_2",   "player_id_2")
        p3_col            = cols.get("player_id_3",   "player_id_3")
        loc_x_col         = cols.get("loc_x",         cols.get("x", "loc_x"))
        # Common alternates for loc_y
        loc_y_source      = cols.get("loc_y", cols.get("y", cols.get("pos_y")))
        # Common alternates for points
        points_source     = cols.get("points", cols.get("points_scored", cols.get("score_change")))

        # Build COALESCE expressions where necessary
        loc_y_expr   = (loc_y_source or "NULL")
        points_expr  = (points_source or "0")

        # pbp_id mapping (fallback to row_number if missing)
        pbp_id_expr = "pbp_id" if "pbp_id" in cols else f"row_number() over (order by {period_col}, {pbp_order_col}) as pbp_id"

        self.conn.execute("DROP VIEW IF EXISTS canonical_pbp")
        self.conn.execute(f"""
            CREATE VIEW canonical_pbp AS
            SELECT 
                {pbp_id_expr},
                {period_col}       AS period,
                {pbp_order_col}    AS pbp_order,
                {wall_clock_col}   AS wall_clock_int,
                {game_clock_col}   AS game_clock,
                {desc_col}         AS description,
                {msg_type_col}     AS msg_type,
                {action_type_col}  AS action_type,
                {off_team_col}     AS off_team_id,
                {def_team_col}     AS def_team_id,
                {p1_col}           AS player_id_1,
                {p2_col}           AS player_id_2,
                {p3_col}           AS player_id_3,
                {loc_x_col}        AS loc_x,
                {loc_y_expr}       AS loc_y,
                {points_expr}      AS points
            FROM {pbp_table}
            WHERE {off_team_col} IS NOT NULL AND {def_team_col} IS NOT NULL
        """)
        return "canonical_pbp"


    def _classify_event_step2(self, event: ProcessedEvent):
        """Classify events using Step 2 methodology"""
        # Shot classification
        if event.msg_type in [1, 2]:  # Made/Missed shots
            event.is_shot = True

            if event.loc_x is not None and event.loc_y is not None:
                event.distance_ft = np.sqrt(event.loc_x**2 + event.loc_y**2) / COORD_SCALE
                event.is_rim_attempt = event.distance_ft <= RIM_THRESHOLD
                event.is_rim_make = event.is_rim_attempt and event.msg_type == 1

        # Substitution classification - Step 2 methodology
        if event.msg_type == CONFIG["msg_types"]["substitution"]:
            event.is_substitution = True
            # Step 2 finding: playerId1 = IN, playerId2 = OUT for traditional
            event.sub_in_player = event.player_id_1
            event.sub_out_player = event.player_id_2

    def process_traditional_substitution(self, event: ProcessedEvent) -> bool:
        """Process substitution using TRADITIONAL DATA-DRIVEN method from Step 2"""
        if not event.is_substitution:
            return False

        in_player = event.sub_in_player
        out_player = event.sub_out_player

        # Traditional method: follow data strictly, allow variable lineup sizes
        try:
            # Determine team (prefer out_player's current team)
            team_id = None
            for tid, lineup in self.traditional_state.team_lineups.items():
                if out_player and out_player in lineup:
                    team_id = tid
                    break

            if not team_id:
                # Fallback to roster check
                for tid, roster in self.team_rosters.items():
                    if (out_player and out_player in roster) or (in_player and in_player in roster):
                        team_id = tid
                        break

            if not team_id:
                self.traditional_state.add_flag(
                    "unknown_team_substitution", 0, in_player,
                    f"Cannot determine team for sub: {out_player} -> {in_player}"
                )
                return False

            lineup = self.traditional_state.team_lineups[team_id]

            # Traditional method flags (from Step 2)
            if out_player and out_player not in lineup:
                self.traditional_state.add_flag(
                    "sub_out_player_not_in_lineup", team_id, out_player,
                    f"OUT player {out_player} not in current lineup"
                )

            if in_player and in_player in lineup:
                self.traditional_state.add_flag(
                    "sub_in_player_already_in_lineup", team_id, in_player,
                    f"IN player {in_player} already in lineup"
                )

            # Execute substitution strictly as recorded
            if out_player and out_player in lineup:
                lineup.remove(out_player)
            if in_player:
                lineup.add(in_player)

            # Flag lineup size deviations (Step 2 finding: 3-6 man lineups)
            if len(lineup) != 5:
                self.traditional_state.add_flag(
                    "lineup_size_deviation", team_id, None,
                    f"Lineup size {len(lineup)}/5 after substitution"
                )
                self.traditional_stats['lineup_size_deviations'] += 1

            self.traditional_stats['substitutions'] += 1
            self.traditional_state.substitution_log.append({
                'period': event.period,
                'team_id': team_id,
                'in_player': in_player,
                'out_player': out_player,
                'lineup_size_after': len(lineup)
            })

            return True

        except Exception as e:
            self.traditional_state.add_flag(
                "substitution_error", team_id or 0, None, f"Error: {str(e)}"
            )
            return False

    def process_enhanced_substitution(self, event: ProcessedEvent, current_time: float) -> bool:
        """Process substitution using ENHANCED method from Step 2"""
        if not event.is_substitution:
            return False

        in_player = event.sub_in_player
        out_player = event.sub_out_player

        # Enhanced method: intelligent inference to maintain 5-man lineups
        try:
            # Determine team with fallbacks
            team_id = self._determine_team_enhanced(in_player, out_player, event)
            if not team_id:
                return False

            lineup = self.enhanced_state.team_lineups[team_id]

            # Enhanced method: prepare lineup for substitution
            if out_player and out_player not in lineup:
                # Try to find and move player
                for other_tid, other_lineup in self.enhanced_state.team_lineups.items():
                    if other_tid != team_id and out_player in other_lineup:
                        other_lineup.remove(out_player)
                        lineup.add(out_player)
                        self.enhanced_state.add_flag(
                            "moved_player_between_teams", team_id, out_player,
                            f"Moved OUT player from team {other_tid} to {team_id}"
                        )
                        break

            if in_player and in_player in lineup:
                # Remove duplicate
                lineup.remove(in_player)
                self.enhanced_state.add_flag(
                    "removed_duplicate_in_player", team_id, in_player,
                    "Removed duplicate IN player"
                )

            # Remove from other teams
            for other_tid, other_lineup in self.enhanced_state.team_lineups.items():
                if other_tid != team_id and in_player and in_player in other_lineup:
                    other_lineup.remove(in_player)

            # Execute substitution
            if out_player and out_player in lineup:
                lineup.remove(out_player)
                self.enhanced_state.recent_out[team_id].append(out_player)

            if in_player:
                lineup.add(in_player)

            # Enhanced method: ensure exactly 5 players
            self._ensure_five_players_enhanced(team_id, current_time)

            self.enhanced_stats['substitutions'] += 1
            self.enhanced_state.substitution_log.append({
                'period': event.period,
                'team_id': team_id,
                'in_player': in_player,
                'out_player': out_player,
                'lineup_size_after': len(self.enhanced_state.team_lineups[team_id])
            })

            return True

        except Exception as e:
            self.enhanced_state.add_flag(
                "substitution_error", team_id or 0, None, f"Error: {str(e)}"
            )
            return False

    def _determine_team_enhanced(self, in_player: int, out_player: int, event) -> Optional[int]:
        """Enhanced team determination with multiple fallbacks"""
        # Check current lineups
        for team_id, lineup in self.enhanced_state.team_lineups.items():
            if out_player and out_player in lineup:
                return team_id

        # Check rosters
        for team_id, roster in self.team_rosters.items():
            if (out_player and out_player in roster) or (in_player and in_player in roster):
                    return team_id

        return None

    def _ensure_five_players_enhanced(self, team_id: int, current_time: float):
        """Ensure exactly 5 players using Enhanced method logic from Step 2"""
        lineup = self.enhanced_state.team_lineups[team_id]

        # Remove excess players (auto-out logic)
        while len(lineup) > 5:
            # Find least active player for auto-out
            candidate = None
            max_idle = -1

            for player_id in lineup:
                idle_time = current_time - self.enhanced_state.last_action_time.get(player_id, 0)
                if idle_time > max_idle:
                    max_idle = idle_time
                    candidate = player_id

            if candidate:
                lineup.remove(candidate)
                self.enhanced_state.recent_out[team_id].append(candidate)
                self.enhanced_state.add_flag(
                    "auto_out_excess_player", team_id, candidate,
                    f"Auto-out due to excess players (idle: {max_idle:.1f}s)"
                )
                self.enhanced_stats['auto_outs'] += 1

        # Add players if under 5
        if len(lineup) < 5:
            available = self.team_rosters.get(team_id, set()) - lineup
            # Prefer recently out players
            recent = [p for p in self.enhanced_state.recent_out[team_id] if p in available]

            for player_id in (recent + list(available))[:5-len(lineup)]:
                lineup.add(player_id)
                self.enhanced_state.add_flag(
                    "auto_in_fill_lineup", team_id, player_id,
                    "Auto-in to fill lineup to 5 players"
                )

    def handle_first_action_events(self, event: ProcessedEvent, current_time: float):
        """Handle first-action events (Reed Sheppard case) from Step 2"""
        if event.msg_type not in [1, 2, 4, 5, 6]:  # Only for action events
            return

        action_player = event.player_id_1
        if not action_player:
            return

        # Check if player is in any lineup
        player_team = None
        player_in_lineup = False

        for team_id, lineup in self.enhanced_state.team_lineups.items():
            if action_player in lineup:
                player_in_lineup = True
                break
            elif action_player in self.team_rosters.get(team_id, set()):
                player_team = team_id

        # First-action injection (Reed Sheppard case)
        if not player_in_lineup and player_team:
            self.enhanced_state.team_lineups[player_team].add(action_player)
            self.enhanced_state.add_flag(
                "first_action_injection", player_team, action_player,
                f"First-action injection: {event.description}"
            )
            self.enhanced_stats['first_actions'] += 1
            self.enhanced_state.first_action_events.append({
                'period': event.period,
                'player_id': action_player,
                'team_id': player_team,
                'event_type': event.msg_type,
                'description': event.description
            })

            # Ensure 5 players after injection
            self._ensure_five_players_enhanced(player_team, current_time)

        # Update activity time
        if action_player:
            self.enhanced_state.last_action_time[action_player] = current_time

    def process_all_events(self) -> ValidationResult:
        """Process all events with BOTH tracking methods from Step 2"""
        start_time = time.time()

        try:
            logger.info(f"Processing {len(self.processed_events)} events with both methods...")

            periods_seen = set()
            current_time = 0.0

            for i, event in enumerate(self.processed_events):
                current_time = float(event.wall_clock_int)

                # Handle period transitions
                if event.period not in periods_seen:
                    periods_seen.add(event.period)
                    logger.info(f"Processing period {event.period}")

                    # Reset periods for enhanced method (Step 2 logic)
                    if event.period in CONFIG["starter_reset_periods"]:
                        self._reset_to_starters_enhanced()

                    # Update period in states
                    self.traditional_state.period = event.period
                    self.enhanced_state.period = event.period

                # Capture lineup context BEFORE processing
                event.traditional_off_lineup = tuple(sorted(self.traditional_state.team_lineups.get(event.off_team_id, set())))
                event.traditional_def_lineup = tuple(sorted(self.traditional_state.team_lineups.get(event.def_team_id, set())))
                event.enhanced_off_lineup = tuple(sorted(self.enhanced_state.team_lineups.get(event.off_team_id, set())))
                event.enhanced_def_lineup = tuple(sorted(self.enhanced_state.team_lineups.get(event.def_team_id, set())))

                # Process substitutions with BOTH methods
                if event.is_substitution:
                    traditional_success = self.process_traditional_substitution(event)
                    enhanced_success = self.process_enhanced_substitution(event, current_time)

                    if CONFIG["debug"]["log_all_substitutions"]:
                        logger.info(f"Substitution P{event.period}: Traditional={traditional_success}, Enhanced={enhanced_success}")

                # Handle first-action events (Enhanced method only)
                self.handle_first_action_events(event, current_time)

            # Collect final statistics
            self.traditional_stats['flags'] = len(self.traditional_state.flags)
            self.enhanced_stats['flags'] = len(self.enhanced_state.flags)

            # Calculate lineup size distribution for traditional method
            traditional_sizes = defaultdict(int)
            for team_lineup in self.traditional_state.team_lineups.values():
                traditional_sizes[len(team_lineup)] += 1

            details = f"Processed {len(self.processed_events)} events. "
            details += f"Traditional: {self.traditional_stats['substitutions']} subs, {self.traditional_stats['flags']} flags, {self.traditional_stats['lineup_size_deviations']} size deviations. "
            details += f"Enhanced: {self.enhanced_stats['substitutions']} subs, {self.enhanced_stats['first_actions']} first-actions, {self.enhanced_stats['auto_outs']} auto-outs."

            return ValidationResult(
                step_name="Process All Events",
                passed=True,
                details=details,
                data_count=len(self.processed_events),
                processing_time=time.time() - start_time
            )

        except Exception as e:
            return ValidationResult(
                step_name="Process All Events",
                passed=False,
                details=f"Error processing events: {str(e)}",
                processing_time=time.time() - start_time
            )

    def _reset_to_starters_enhanced(self):
        """Reset to starters for enhanced method (Q1, Q3)"""
        if hasattr(self.entities, 'starters'):
            for team_abbrev, starters_list in self.entities.starters.items():
                if isinstance(starters_list, list):
                    team_id = None
                    for tid, tabbrev in self.entities.team_mapping.items():
                        if isinstance(tid, int) and tabbrev == team_abbrev:
                            team_id = tid
                            break

                    if team_id:
                        starter_ids = {starter['player_id'] for starter in starters_list}
                        self.enhanced_state.team_lineups[team_id] = starter_ids.copy()



    def _step4_required_columns(self) -> Set[str]:
        """
        The required dual-method lineup columns that Step 5 relies on.
        """
        return {
            "traditional_off_lineup", "traditional_def_lineup",
            "enhanced_off_lineup", "enhanced_def_lineup"
        }

    def _write_step4_contract_stamp(self, table_name: str) -> None:
        """
        Write a versioned contract stamp indicating Step 4 produced the expected schema.
        This does not alter data; it records meta only.
        """
        try:
            # Discover columns & row count of the output table
            cols = [r[1] for r in self.conn.execute(f"PRAGMA table_info('{table_name}')").fetchall()]
            row_count = self.conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]

            # Create contract table if missing
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS pipeline_contract (
                    component    VARCHAR,
                    version      VARCHAR,
                    table_name   VARCHAR,
                    columns_json VARCHAR,
                    row_count    BIGINT,
                    created_at   TIMESTAMP
                )
            """)

            # Insert a new stamp
            self.conn.execute("""
                INSERT INTO pipeline_contract(component, version, table_name, columns_json, row_count, created_at)
                VALUES (?, ?, ?, ?, ?, now())
            """, ("step4", "dual_lineups_v1", table_name, json.dumps(cols, ensure_ascii=True), row_count))

            logger.info(f"[CONTRACT] Step 4 stamped version 'dual_lineups_v1' for table '{table_name}' ({row_count} rows)")
        except Exception as e:
            logger.warning(f"[CONTRACT] Failed to write Step 4 contract stamp: {e}")

    def validate_step4_schema(self) -> bool:
        """
        Validate that step4_processed_events contains the expected dual-method columns.
        Logs full column list and returns True/False.
        """
        try:
            cols = [r[1] for r in self.conn.execute("PRAGMA table_info('step4_processed_events')").fetchall()]
            required = self._step4_required_columns()
            ok = required.issubset(set(cols))
            logger.info(f"[VALIDATE] Step 4 schema columns: {cols}")
            if not ok:
                missing = sorted(required - set(cols))
                logger.error(f"[VALIDATE] Step 4 schema missing required columns: {missing}")
            return ok
        except Exception as e:
            logger.error(f"[VALIDATE] Error validating Step 4 schema: {e}")
            return False

    def create_step4_output_tables(self) -> ValidationResult:
        """Create Step 4 output tables integrating both methods (schema-safe)"""
        start_time = time.time()

        try:
            logger.info("Creating Step 4 output tables with both tracking methods...")

            # --- Hard drop both VIEW and TABLE (handles stale views/tables) ---
            for obj in ("step4_processed_events", "step4_traditional_flags", "step4_enhanced_flags"):
                try:
                    self.conn.execute(f"DROP VIEW IF EXISTS {obj}")
                except Exception:
                    pass
                try:
                    self.conn.execute(f"DROP TABLE IF EXISTS {obj}")
                except Exception:
                    pass

            # Build events dataframe
            events_data = []
            for event in self.processed_events:
                events_data.append({
                    'pbp_id': event.pbp_id,
                    'period': event.period,
                    'pbp_order': event.pbp_order,
                    'wall_clock_int': event.wall_clock_int,
                    'description': event.description,
                    'msg_type': event.msg_type,
                    'action_type': event.action_type,
                    'off_team_id': event.off_team_id,
                    'def_team_id': event.def_team_id,
                    'player_id_1': event.player_id_1,
                    'player_id_2': event.player_id_2,
                    'player_id_3': event.player_id_3,
                    'is_shot': bool(event.is_shot),
                    'is_rim_attempt': bool(event.is_rim_attempt),
                    'is_rim_make': bool(event.is_rim_make),
                    'distance_ft': float(event.distance_ft) if event.distance_ft is not None else None,
                    'is_substitution': bool(event.is_substitution),
                    'points': int(event.points) if event.points is not None else 0,
                    # Store lineups as ASCII JSON arrays for portability
                    'traditional_off_lineup': json.dumps(list(event.traditional_off_lineup), ensure_ascii=True) if event.traditional_off_lineup else None,
                    'traditional_def_lineup': json.dumps(list(event.traditional_def_lineup), ensure_ascii=True) if event.traditional_def_lineup else None,
                    'enhanced_off_lineup': json.dumps(list(event.enhanced_off_lineup), ensure_ascii=True) if event.enhanced_off_lineup else None,
                    'enhanced_def_lineup': json.dumps(list(event.enhanced_def_lineup), ensure_ascii=True) if event.enhanced_def_lineup else None
                })

            events_df = pd.DataFrame(events_data)

            # Persist processed events
            self.conn.register("events_temp", events_df)
            try:
                self.conn.execute("""
                    CREATE TABLE step4_processed_events AS
                    SELECT * FROM events_temp
                    ORDER BY period, pbp_order, wall_clock_int
                """)
            finally:
                self.conn.unregister("events_temp")

            # Traditional flags
            traditional_flags_df = pd.DataFrame(self.traditional_state.flags)
            if not traditional_flags_df.empty:
                self.conn.register("trad_flags_temp", traditional_flags_df)
                try:
                    self.conn.execute("CREATE TABLE step4_traditional_flags AS SELECT * FROM trad_flags_temp")
                finally:
                    self.conn.unregister("trad_flags_temp")

            # Enhanced flags
            enhanced_flags_df = pd.DataFrame(self.enhanced_state.flags)
            if not enhanced_flags_df.empty:
                self.conn.register("enh_flags_temp", enhanced_flags_df)
                try:
                    self.conn.execute("CREATE TABLE step4_enhanced_flags AS SELECT * FROM enh_flags_temp")
                finally:
                    self.conn.unregister("enh_flags_temp")

            # Method comparison (quick summary table)
            comparison_data = [{
                'method': 'Traditional',
                'substitutions_processed': self.traditional_stats['substitutions'],
                'flags_generated': self.traditional_stats['flags'],
                'lineup_size_deviations': self.traditional_stats['lineup_size_deviations'],
                'maintains_5_man_lineups': False
            }, {
                'method': 'Enhanced',
                'substitutions_processed': self.enhanced_stats['substitutions'],
                'flags_generated': self.enhanced_stats['flags'],
                'first_action_injections': self.enhanced_stats['first_actions'],
                'auto_out_corrections': self.enhanced_stats['auto_outs'],
                'maintains_5_man_lineups': True
            }]
            comparison_df = pd.DataFrame(comparison_data)
            self.conn.register("comp_temp", comparison_df)
            try:
                self.conn.execute("CREATE OR REPLACE TABLE step4_method_comparison AS SELECT * FROM comp_temp")
            finally:
                self.conn.unregister("comp_temp")

            # --- Post-create schema validation & contract stamp ---
            ok = self.validate_step4_schema()
            self._write_step4_contract_stamp("step4_processed_events")

            details = (
                f"Created Step 4 output tables: processed_events ({len(events_data)} rows), "
                f"traditional_flags ({len(traditional_flags_df)} rows), "
                f"enhanced_flags ({len(enhanced_flags_df)} rows), method_comparison"
            )
            return ValidationResult(
                step_name="Create Step 4 Output Tables",
                passed=ok,
                details=details if ok else details + " [SCHEMA INVALID]",
                data_count=len(events_data),
                processing_time=time.time() - start_time
            )

        except Exception as e:
            return ValidationResult(
                step_name="Create Step 4 Output Tables",
                passed=False,
                details=f"Error creating output tables: {str(e)}",
                processing_time=time.time() - start_time
            )


    def print_step4_summary(self):
        """Print Step 4 summary with both methods (ASCII only)"""
        print("\n" + "="*80)
        print("NBA PIPELINE - STEP 4 SUMMARY (INTEGRATED WITH STEP 2)")
        print("="*80)

        print("TRADITIONAL DATA-DRIVEN METHOD:")
        print(f"  Substitutions Processed: {self.traditional_stats['substitutions']}")
        print(f"  Flags Generated: {self.traditional_stats['flags']}")
        print(f"  Lineup Size Deviations: {self.traditional_stats['lineup_size_deviations']}")
        print("  Current Lineup Sizes:")
        for team_id, lineup in self.traditional_state.team_lineups.items():
            team_name = self.team_names.get(team_id, f"Team_{team_id}")
            print(f"    {team_name}: {len(lineup)} players")

        print("\nENHANCED ESTIMATION METHOD:")
        print(f"  Substitutions Processed: {self.enhanced_stats['substitutions']}")
        print(f"  First-Action Injections: {self.enhanced_stats['first_actions']}")
        print(f"  Auto-Out Corrections: {self.enhanced_stats['auto_outs']}")
        print(f"  Flags Generated: {self.enhanced_stats['flags']}")
        print("  Current Lineup Sizes:")
        for team_id, lineup in self.enhanced_state.team_lineups.items():
            team_name = self.team_names.get(team_id, f"Team_{team_id}")
            print(f"    {team_name}: {len(lineup)} players")

        print(f"\nTOTAL EVENTS PROCESSED: {len(self.processed_events)}")

        trad_correct = sum(1 for lineup in self.traditional_state.team_lineups.values() if len(lineup) == 5)
        enh_correct = sum(1 for lineup in self.enhanced_state.team_lineups.values() if len(lineup) == 5)
        total_lineups = max(1, len(self.traditional_state.team_lineups))

        print("LINEUP SIZE ACCURACY:")
        print(f"  Traditional: {trad_correct}/{total_lineups} teams have 5-man lineups "
            f"({trad_correct/total_lineups*100:.1f}%)")
        print(f"  Enhanced: {enh_correct}/{total_lineups} teams have 5-man lineups "
            f"({enh_correct/total_lineups*100:.1f}%)")
        print("="*80)



def process_pbp_with_step2_integration(db_path: str = None, 
                                      entities: GameEntities = None) -> Tuple[bool, PBPProcessor]:
    """Process PBP events using integrated Step 2 methods"""

    print("NBA Pipeline - Step 4: Integrated PBP Processing (Updated with Step 2)")
    print("="*75)

    if entities is None:
        logger.error("GameEntities required for PBP processing")
        return False, None

    with PBPProcessor(db_path, entities) as processor:

        # Initialize lineups for both methods
        logger.info("Step 4a: Initializing lineups for both tracking methods...")
        result = processor.initialize_lineups()
        processor.validator.log_validation(result)
        if not result.passed:
            logger.error("Failed to initialize lineups")
            return False, processor

        # Load PBP events with Step 2 classification
        logger.info("Step 4b: Loading PBP events with Step 2 classification...")
        result = processor.load_pbp_events()
        processor.validator.log_validation(result)
        if not result.passed:
            logger.error("Failed to load PBP events")
            return False, processor

        # Process all events with both methods
        logger.info("Step 4c: Processing events with both Traditional and Enhanced methods...")
        result = processor.process_all_events()
        processor.validator.log_validation(result)

        # Create output tables
        logger.info("Step 4d: Creating Step 4 output tables...")
        result = processor.create_step4_output_tables()
        print('output tables results===============', result)
        processor.validator.log_validation(result)

        # Print summary
        processor.print_step4_summary()

        success = processor.validator.print_validation_summary()
        return success, processor


# Example usage
if __name__ == "__main__":
    from eda.data.nba_entities_extractor import extract_all_entities_robust

    # Extract entities first
    entities_success, entities = extract_all_entities_robust()

    if entities_success:
        success, processor = process_pbp_with_step2_integration(entities=entities)

        if success:
            print("\n✅ Step 4 Complete: Integrated processing with Step 2 methods")
            print("🎯 Both Traditional and Enhanced methods available for comparison")
        else:
            print("\n❌ Step 4 Failed: Review validation messages")
    else:
        print("❌ Failed to get entities - cannot proceed")
