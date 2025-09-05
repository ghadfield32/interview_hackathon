# api/src/airflow_project/eda/utils/nba_pipeline_analysis.py
# Step 1: NBA Pipeline Analysis & Data Structure Validation
"""
NBA Play-by-Play Data Pipeline - Step 1: Analysis & Setup
---------------------------------------------------------

This step analyzes the required inputs and sets up basic validations for:

1. Box score data (player info, starters, team mapping)
2. Play-by-play data (events, shots, substitutions, possessions)
3. Lookup tables (event types, action types, option types)

Key requirements:
- Track 5-man lineups
- Count offensive/defensive possessions per lineup
- Track rim attempts (≤ 4 feet from basket)
- Compute on/off rim defense stats
- Handle substitutions and lineup changes
- Validate data integrity at each step
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import logging
import sys
import time

import duckdb  # kept intentionally; may be imported transitively elsewhere
import numpy as np
import pandas as pd


def configure_logging(level: int = logging.INFO) -> None:
    """
    Configure plain logging to stderr with timestamps.
    No custom formatters; avoids encoding issues by keeping ASCII-only output.
    """
    logging.basicConfig(
        stream=sys.stderr,
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,
    )


# Activate plain logging for the module execution context
configure_logging(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Structure to track validation results at each step."""
    step_name: str
    passed: bool
    details: str
    data_count: int = 0
    processing_time: float = 0.0
    warnings: List[str] | None = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


class NBADataValidator:
    """Data validation routines for the NBA pipeline."""

    def __init__(self):
        self.validation_results: List[ValidationResult] = []
        self.rim_distance_threshold: float = 4.0  # feet
        self.coordinate_scale: float = 10.0       # coordinates are in tenths of feet

    def log_validation(self, result: ValidationResult) -> None:
        """Log and store validation results."""
        self.validation_results.append(result)
        status = "[PASS]" if result.passed else "[FAIL]"
        logger.info(f"{status} {result.step_name}: {result.details}")
        for warning in result.warnings:
            logger.warning(f"[WARN] {result.step_name}: {warning}")

    def validate_file_structure(self, file_paths: Dict[str, Path]) -> ValidationResult:
        """Validate that all required files exist and are non-empty."""
        start_time = time.time()
        missing_files: List[str] = []
        empty_files: List[str] = []
        total_files = len(file_paths)

        for name, path in file_paths.items():
            if not path.exists():
                missing_files.append(f"{name}: {path}")
            elif path.stat().st_size == 0:
                empty_files.append(f"{name}: {path}")

        warnings: List[str] = []
        if empty_files:
            warnings.extend([f"Empty file: {f}" for f in empty_files])

        passed = len(missing_files) == 0
        details = f"Checked {total_files} files. Missing: {len(missing_files)}, Empty: {len(empty_files)}"

        return ValidationResult(
            step_name="File Structure Validation",
            passed=passed,
            details=details,
            processing_time=time.time() - start_time,
            warnings=warnings,
        )

    def validate_box_score_structure(self, df: pd.DataFrame) -> ValidationResult:
        """Validate box score has expected structure and data quality."""
        start_time = time.time()

        required_columns = ['nbaId', 'name', 'nbaTeamId', 'team', 'isHome', 'gs', 'status']
        missing_cols = [col for col in required_columns if col not in df.columns]

        if missing_cols:
            return ValidationResult(
                step_name="Box Score Structure",
                passed=False,
                details=f"Missing columns: {missing_cols}",
                data_count=len(df),
                processing_time=time.time() - start_time,
            )

        active_players = df[df['status'] == 'ACTIVE']
        teams = active_players['team'].unique()
        starters_per_team: Dict[str, int] = {}
        warnings: List[str] = []

        for team in teams:
            team_players = active_players[active_players['team'] == team]
            starters = team_players[team_players['gs'] == 1]
            starters_per_team[team] = len(starters)
            if len(starters) != 5:
                warnings.append(f"Team {team} has {len(starters)} starters (expected 5)")

        details = (
            f"Active players: {len(active_players)}, "
            f"Teams: {len(teams)}, "
            f"Starters per team: {starters_per_team}"
        )

        return ValidationResult(
            step_name="Box Score Structure",
            passed=True,
            details=details,
            data_count=len(df),
            processing_time=time.time() - start_time,
            warnings=warnings,
        )

    def validate_pbp_structure(self, df: pd.DataFrame) -> ValidationResult:
        """Validate play-by-play structure and event types."""
        start_time = time.time()

        required_columns = [
            'period', 'pbpOrder', 'msgType', 'offTeamId', 'defTeamId',
            'playerId1', 'locX', 'locY', 'pts'
        ]
        missing_cols = [col for col in required_columns if col not in df.columns]

        if missing_cols:
            return ValidationResult(
                step_name="PBP Structure",
                passed=False,
                details=f"Missing columns: {missing_cols}",
                data_count=len(df),
                processing_time=time.time() - start_time,
            )

        valid_events = df[df['offTeamId'].notna() & df['defTeamId'].notna()]
        event_types = df['msgType'].value_counts()

        shots = df[df['msgType'].isin([1, 2])]
        shots_with_coords = shots[(shots['locX'].notna()) & (shots['locY'].notna())]

        warnings: List[str] = []
        if len(shots) > 0 and len(shots_with_coords) < len(shots) * 0.8:
            warnings.append(f"Only {len(shots_with_coords)}/{len(shots)} shots have coordinates")

        details = (
            f"Total events: {len(df)}, "
            f"Valid events: {len(valid_events)}, "
            f"Event types: {len(event_types)}, "
            f"Shots with coords: {len(shots_with_coords)}"
        )

        return ValidationResult(
            step_name="PBP Structure",
            passed=True,
            details=details,
            data_count=len(df),
            processing_time=time.time() - start_time,
            warnings=warnings,
        )

    def validate_coordinate_system(self, df: pd.DataFrame) -> ValidationResult:
        """Validate coordinate scaling and rim detection logic."""
        start_time = time.time()

        shots = df[
            (df['msgType'].isin([1, 2])) &
            (df['locX'].notna()) &
            (df['locY'].notna())
        ].copy()

        if len(shots) == 0:
            return ValidationResult(
                step_name="Coordinate System",
                passed=False,
                details="No shots with coordinates found",
                processing_time=time.time() - start_time,
            )

        # distance in feet (coords are tenths of feet)
        shots['distance_ft'] = np.sqrt(shots['locX'] ** 2 + shots['locY'] ** 2) / self.coordinate_scale
        shots['is_rim_attempt'] = shots['distance_ft'] <= self.rim_distance_threshold

        rim_attempts = shots[shots['is_rim_attempt']]
        rim_makes = rim_attempts[rim_attempts['msgType'] == 1]

        warnings: List[str] = []
        max_distance = float(shots['distance_ft'].max())
        if max_distance > 35:
            warnings.append(f"Suspiciously long shot distance: {max_distance:.1f} feet")

        details = (
            f"Total shots: {len(shots)}, "
            f"Rim attempts: {len(rim_attempts)}, "
            f"Rim makes: {len(rim_makes)}, "
            f"Max distance: {max_distance:.1f}ft"
        )

        return ValidationResult(
            step_name="Coordinate System",
            passed=True,
            details=details,
            data_count=len(shots),
            processing_time=time.time() - start_time,
            warnings=warnings,
        )

    def print_validation_summary(self) -> bool:
        """Print a concise validation summary. Returns True if all tests passed."""
        print("\n" + "=" * 80)
        print("NBA PIPELINE VALIDATION SUMMARY")
        print("=" * 80)

        total_passed = sum(1 for r in self.validation_results if r.passed)
        total_tests = len(self.validation_results)
        total_time = sum(r.processing_time for r in self.validation_results)
        total_warnings = sum(len(r.warnings) for r in self.validation_results)

        print(f"OVERALL STATUS: {total_passed}/{total_tests} tests passed")
        print(f"TOTAL VALIDATION TIME: {total_time:.2f} seconds")
        print(f"TOTAL WARNINGS: {total_warnings}\n")

        for result in self.validation_results:
            status = "[PASS]" if result.passed else "[FAIL]"
            print(f"{status} {result.step_name}")
            print(f"   Details: {result.details}")
            print(f"   Data Count: {result.data_count:,}")
            print(f"   Time: {result.processing_time:.3f}s")
            for warning in result.warnings:
                print(f"   [WARN] {warning}")
            print()

        print("=" * 80)
        return total_passed == total_tests


if __name__ == "__main__":
    print("NBA Pipeline - Step 1: Analysis & Validation Setup")
    print("=" * 60)

    validator = NBADataValidator()

    expected_files = {
        'box_score': Path('api/src/airflow_project/data/mavs_data_engineer_2025/box_HOU-DAL.csv'),
        'pbp': Path('api/src/airflow_project/data/mavs_data_engineer_2025/pbp_HOU-DAL.csv'),
        'event_types': Path('api/src/airflow_project/data/mavs_data_engineer_2025/pbp_event_msg_types.csv'),
        'action_types': Path('api/src/airflow_project/data/mavs_data_engineer_2025/pbp_action_types.csv'),
        'option_types': Path('api/src/airflow_project/data/mavs_data_engineer_2025/pbp_option_types.csv'),
    }

    # Step 1A: Validate file structure
    file_validation = validator.validate_file_structure(expected_files)
    validator.log_validation(file_validation)

    print("\nStep 1 Complete: Foundation analysis ready")
    print("Next Step: Load and validate data content")
    print("The pipeline will process data with validations at each stage")

    print("\nDATA REQUIREMENTS SUMMARY:")
    print("- Box Score: Player info, starters (gs=1), team mapping, active status")
    print("- Play-by-Play: Events in chronological order, coordinates for shots")
    print("- Rim Attempts: Shots ≤ 4 feet from basket (coordinates/10 = feet)")
    print("- Lineup Tracking: 5 players per team; handle substitutions")
    print("- Possession Counting: Offensive/defensive possessions per lineup")
    print("- Outputs: 5-man lineups and individual player rim defense stats")
