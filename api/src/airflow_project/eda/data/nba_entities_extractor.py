# Robust NBA Entity Extractor - Step 3 Improvements  
"""
NBA Pipeline - Robust Entity Extraction with Proper Validation
==============================================================

This robust extractor handles the identified issues:
1. Uses actual starter data from box score (gs=1)
2. Handles missing players transparently 
3. Creates proper team mappings
4. Validates entity completeness without hiding issues

Key Features:
- Extract exactly 5 starters per team from box score
- Handle team mapping correctly (HOU: 1610612745, DAL: 1610612742)
- Create canonical player and team entities
- Transparent validation and error reporting
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field

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

@dataclass
class GameEntities:
    """Container for all canonical game entities"""
    unique_players: pd.DataFrame = None
    starters: Dict[str, List[Dict]] = field(default_factory=dict)
    team_mapping: Dict[int, str] = field(default_factory=dict)
    game_info: Dict = field(default_factory=dict)

    def validate_completeness(self) -> List[str]:
        """Validate all entities are present and complete"""
        errors = []

        if self.unique_players is None or len(self.unique_players) == 0:
            errors.append("unique_players is empty")

        if len(self.starters) == 0:
            errors.append("No starters defined")

        if len(self.team_mapping) == 0:
            errors.append("No team mapping defined")

        return errors

class RobustEntityExtractor:
    """Extract and validate canonical entities from NBA data"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = None
        self.validator = NBADataValidator()
        self.entities = GameEntities()

    def __enter__(self):
        self.conn = duckdb.connect(self.db_path)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn:
            self.conn.close()

    def _robust_drop_object(self, object_name: str) -> None:
        """Robustly drop any DuckDB object regardless of type (same as Step 2)"""
        try:
            self.conn.execute(f"DROP TABLE IF EXISTS {object_name}")
        except Exception:
            pass

        try:
            self.conn.execute(f"DROP VIEW IF EXISTS {object_name}")
        except Exception:
            pass

        try:
            self.conn.execute(f"DROP SEQUENCE IF EXISTS {object_name}")
        except Exception:
            pass

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

    def extract_unique_players(self) -> ValidationResult:
        """Extract all unique active players with validation"""
        start_time = time.time()

        try:
            logger.info("Extracting unique players from box score...")

            # Get all active players from box score
            players_query = """
            SELECT 
                player_id,
                player_name,
                team_id,
                team_abbrev,
                is_home,
                is_starter,
                seconds_played,
                points,
                rebounds,
                assists,
                jersey_number
            FROM box_score
            WHERE seconds_played > 0
            ORDER BY team_id, seconds_played DESC
            """

            self.entities.unique_players = self.conn.execute(players_query).df()

            if len(self.entities.unique_players) == 0:
                return ValidationResult(
                    step_name="Extract Unique Players",
                    passed=False,
                    details="No players found in box score",
                    processing_time=time.time() - start_time
                )

            warnings = []

            # Validate player data quality
            null_names = self.entities.unique_players['player_name'].isnull().sum()
            if null_names > 0:
                warnings.append(f"{null_names} players have null names")

            # Validate team distribution
            team_counts = self.entities.unique_players['team_abbrev'].value_counts()
            for team, count in team_counts.items():
                if count < 8:
                    warnings.append(f"Team {team} has only {count} players (minimum 8 expected)")
                elif count > 20:
                    warnings.append(f"Team {team} has {count} players (unusually high)")

            # Check for duplicate player IDs
            duplicate_players = self.entities.unique_players['player_id'].duplicated().sum()
            if duplicate_players > 0:
                warnings.append(f"{duplicate_players} duplicate player IDs found")
                # Remove duplicates, keeping first occurrence
                self.entities.unique_players = self.entities.unique_players.drop_duplicates(
                    subset=['player_id'], keep='first'
                )

            details = f"Extracted {len(self.entities.unique_players)} players across {len(team_counts)} teams"
            details += f". Team distribution: {dict(team_counts)}"

            return ValidationResult(
                step_name="Extract Unique Players",
                passed=True,
                details=details,
                data_count=len(self.entities.unique_players),
                processing_time=time.time() - start_time,
                warnings=warnings
            )

        except Exception as e:
            return ValidationResult(
                step_name="Extract Unique Players",
                passed=False,
                details=f"Error extracting players: {str(e)}",
                processing_time=time.time() - start_time
            )

    def extract_starters(self) -> ValidationResult:
        """Extract starting lineups with strict validation"""
        start_time = time.time()

        try:
            logger.info("Extracting starting lineups...")

            # Get starters from box score (gs=1 as specified in requirements)
            starters_query = """
            SELECT 
                team_id,
                team_abbrev,
                player_id,
                player_name,
                jersey_number,
                seconds_played,
                points
            FROM box_score
            WHERE is_starter = TRUE
            ORDER BY team_abbrev, seconds_played DESC
            """

            starters_df = self.conn.execute(starters_query).df()

            if len(starters_df) == 0:
                return ValidationResult(
                    step_name="Extract Starters",
                    passed=False,
                    details="No starters found in box score",
                    processing_time=time.time() - start_time
                )

            warnings = []
            teams_with_issues = []

            # Process starters by team
            for team_abbrev in starters_df['team_abbrev'].unique():
                team_starters = starters_df[starters_df['team_abbrev'] == team_abbrev]

                # Validate exactly 5 starters per team (as specified in requirements)
                if len(team_starters) != 5:
                    warnings.append(f"Team {team_abbrev} has {len(team_starters)} starters (expected 5)")
                    teams_with_issues.append(team_abbrev)

                    # If we don't have exactly 5, try to fix it
                    if len(team_starters) < 5:
                        # Get additional players from the same team
                        additional_players = self.conn.execute(f"""
                            SELECT player_id, player_name, jersey_number, seconds_played, points
                            FROM box_score 
                            WHERE team_abbrev = '{team_abbrev}' 
                            AND is_starter = FALSE
                            AND seconds_played > 0
                            ORDER BY seconds_played DESC
                            LIMIT {5 - len(team_starters)}
                        """).df()

                        if len(additional_players) > 0:
                            # Add team info to additional players
                            team_info = team_starters.iloc[0][['team_id', 'team_abbrev']]
                            for col in ['team_id', 'team_abbrev']:
                                additional_players[col] = team_info[col]

                            # Combine with existing starters
                            team_starters = pd.concat([team_starters, additional_players], ignore_index=True)
                            warnings.append(f"Added {len(additional_players)} non-starters to {team_abbrev} lineup")

                    elif len(team_starters) > 5:
                        # Keep top 5 by playing time
                        team_starters = team_starters.head(5)
                        warnings.append(f"Reduced {team_abbrev} starters to top 5 by playing time")

                # Create starters list for this team
                starters_list = []
                for i, (_, player) in enumerate(team_starters.iterrows()):
                    starters_list.append({
                        'player_id': int(player['player_id']),
                        'player_name': player['player_name'],
                        'jersey_number': int(player['jersey_number']) if pd.notna(player['jersey_number']) else None,
                        'position': f"P{i+1}",  # Generic position since we don't have actual positions
                        'seconds_played': int(player['seconds_played']),
                        'points': int(player['points'])
                    })

                # Store starters for this team
                self.entities.starters[team_abbrev] = starters_list
                self.entities.starters[f"{team_abbrev}_ids"] = tuple(sorted(p['player_id'] for p in starters_list))

                logger.info(f"Team {team_abbrev} starters: {[p['player_name'] for p in starters_list]}")

            total_starters = sum(len(v) for k, v in self.entities.starters.items() if isinstance(v, list))

            details = f"Extracted starters for {len([k for k in self.entities.starters if not k.endswith('_ids')])} teams"
            details += f", {total_starters} total starters"

            if teams_with_issues:
                details += f". Issues resolved for: {', '.join(teams_with_issues)}"

            return ValidationResult(
                step_name="Extract Starters",
                passed=len(teams_with_issues) == 0,
                details=details,
                data_count=total_starters,
                processing_time=time.time() - start_time,
                warnings=warnings
            )

        except Exception as e:
            return ValidationResult(
                step_name="Extract Starters",
                passed=False,
                details=f"Error extracting starters: {str(e)}",
                processing_time=time.time() - start_time
            )

    def extract_team_mapping(self) -> ValidationResult:
        """Create team ID → abbreviation mapping with home/away flags"""
        start_time = time.time()

        try:
            logger.info("Creating team mapping...")

            # Get team information from box score
            team_query = """
            SELECT DISTINCT
                team_id,
                team_abbrev,
                is_home
            FROM box_score
            ORDER BY team_id
            """

            team_df = self.conn.execute(team_query).df()

            if len(team_df) == 0:
                return ValidationResult(
                    step_name="Extract Team Mapping",
                    passed=False,
                    details="No teams found in box score",
                    processing_time=time.time() - start_time
                )

            warnings = []

            # Build team mapping
            for _, row in team_df.iterrows():
                self.entities.team_mapping[int(row['team_id'])] = row['team_abbrev']

            # Identify home and away teams
            home_teams = team_df[team_df['is_home'] == True]['team_abbrev'].tolist()
            away_teams = team_df[team_df['is_home'] == False]['team_abbrev'].tolist()

            # Validate exactly one home and one away team
            if len(home_teams) != 1 or len(away_teams) != 1:
                warnings.append(f"Invalid home/away setup: home={home_teams}, away={away_teams}")
            else:
                self.entities.team_mapping['home_team'] = home_teams[0]
                self.entities.team_mapping['away_team'] = away_teams[0]

            # Validate expected teams (based on file name HOU-DAL)
            expected_teams = {'HOU', 'DAL'}
            actual_teams = set(team_df['team_abbrev'])

            if expected_teams != actual_teams:
                warnings.append(f"Expected teams {expected_teams}, found {actual_teams}")

            # Validate expected home/away (DAL should be home based on file naming convention)
            if home_teams and home_teams[0] != 'DAL':
                warnings.append(f"Expected DAL to be home team, but {home_teams[0]} is home")

            if away_teams and away_teams[0] != 'HOU':
                warnings.append(f"Expected HOU to be away team, but {away_teams[0]} is away")

            # Create reverse mapping for convenience
            team_id_to_abbrev = {k: v for k, v in self.entities.team_mapping.items() if isinstance(k, int)}

            details = f"Created mapping for {len(team_id_to_abbrev)} teams: {team_id_to_abbrev}"
            if 'home_team' in self.entities.team_mapping:
                details += f". Home: {self.entities.team_mapping['home_team']}, Away: {self.entities.team_mapping['away_team']}"

            return ValidationResult(
                step_name="Extract Team Mapping",
                passed=len(warnings) == 0,
                details=details,
                data_count=len(team_id_to_abbrev),
                processing_time=time.time() - start_time,
                warnings=warnings
            )

        except Exception as e:
            return ValidationResult(
                step_name="Extract Team Mapping",
                passed=False,
                details=f"Error creating team mapping: {str(e)}",
                processing_time=time.time() - start_time
            )

    def extract_game_info(self) -> ValidationResult:
        """Extract basic game information and metadata"""
        start_time = time.time()

        try:
            logger.info("Extracting game information...")

            # Get game info from box score
            game_info_query = """
            SELECT 
                COUNT(DISTINCT team_id) as num_teams,
                COUNT(*) as total_players,
                COUNT(CASE WHEN is_starter THEN 1 END) as total_starters,
                SUM(seconds_played) as total_seconds_played,
                SUM(points) as total_points,
                MIN(team_id) as team1_id,
                MAX(team_id) as team2_id
            FROM box_score
            """

            game_info = self.conn.execute(game_info_query).df().iloc[0].to_dict()

            # Get team names
            teams = [self.entities.team_mapping[int(game_info['team1_id'])], 
                    self.entities.team_mapping[int(game_info['team2_id'])]]

            self.entities.game_info = {
                'num_teams': int(game_info['num_teams']),
                'total_players': int(game_info['total_players']),
                'total_starters': int(game_info['total_starters']),
                'total_seconds_played': int(game_info['total_seconds_played']),
                'total_points': int(game_info['total_points']),
                'teams': teams,
                'matchup': f"{teams[1]} @ {teams[0]}" if 'home_team' in self.entities.team_mapping else f"{teams[0]} vs {teams[1]}"
            }

            warnings = []

            # Validate game info
            if game_info['num_teams'] != 2:
                warnings.append(f"Expected 2 teams, found {game_info['num_teams']}")

            if game_info['total_starters'] != 10:
                warnings.append(f"Expected 10 starters, found {game_info['total_starters']}")

            if game_info['total_players'] < 16:
                warnings.append(f"Only {game_info['total_players']} players found (minimum 16 expected)")

            details = f"Game: {self.entities.game_info['matchup']}, {game_info['total_players']} players, {game_info['total_starters']} starters"

            return ValidationResult(
                step_name="Extract Game Info",
                passed=len(warnings) == 0,
                details=details,
                data_count=1,
                processing_time=time.time() - start_time,
                warnings=warnings
            )

        except Exception as e:
            return ValidationResult(
                step_name="Extract Game Info",
                passed=False,
                details=f"Error extracting game info: {str(e)}",
                processing_time=time.time() - start_time
            )

    def create_canonical_tables(self) -> ValidationResult:
        """Create optimized tables for canonical entities"""
        start_time = time.time()

        try:
            logger.info("Creating canonical entity tables...")

            # Create canonical players table with robust object handling
            self._robust_drop_object("canonical_players")
            self.conn.execute("""
            CREATE VIEW canonical_players AS
            SELECT 
                player_id,
                player_name,
                team_id,
                team_abbrev,
                is_home,
                is_starter,
                seconds_played,
                points,
                rebounds,
                assists,
                jersey_number
            FROM box_score
            ORDER BY team_abbrev, seconds_played DESC
            """)

            # Create canonical starters table
            starters_data = []
            for team_abbrev, starters_list in self.entities.starters.items():
                if isinstance(starters_list, list):  # Skip _ids entries
                    team_id = None
                    for tid, abbrev in self.entities.team_mapping.items():
                        if isinstance(tid, int) and abbrev == team_abbrev:
                            team_id = tid
                            break

                    for i, starter in enumerate(starters_list):
                        starters_data.append({
                            'team_id': team_id,
                            'team_abbrev': team_abbrev,
                            'lineup_position': i + 1,
                            'player_id': starter['player_id'],
                            'player_name': starter['player_name'],
                            'jersey_number': starter['jersey_number'],
                            'position': starter['position'],
                            'seconds_played': starter['seconds_played'],
                            'points': starter['points']
                        })

            if starters_data:
                # Robust object handling
                self._robust_drop_object("canonical_starters")

                starters_df = pd.DataFrame(starters_data)
                self.conn.register("starters_temp", starters_df)

                self.conn.execute("""
                CREATE TABLE canonical_starters AS
                SELECT * FROM starters_temp
                ORDER BY team_id, lineup_position
                """)

                self.conn.execute("DROP VIEW IF EXISTS starters_temp")

                # Create indexes for performance with error handling
                try:
                    self.conn.execute("CREATE INDEX IF NOT EXISTS idx_canonical_starters_team ON canonical_starters(team_id)")
                    self.conn.execute("CREATE INDEX IF NOT EXISTS idx_canonical_starters_player ON canonical_starters(player_id)")
                except Exception as e:
                    logger.warning(f"Could not create starter indexes: {e}")

            # Create team mapping table
            team_data = []
            for team_id, team_abbrev in self.entities.team_mapping.items():
                if isinstance(team_id, int):  # Skip special keys like 'home_team'
                    is_home = team_abbrev == self.entities.team_mapping.get('home_team', '')
                    team_data.append({
                        'team_id': team_id,
                        'team_abbrev': team_abbrev,
                        'is_home': is_home
                    })

            if team_data:
                # Robust object handling
                self._robust_drop_object("canonical_teams")

                teams_df = pd.DataFrame(team_data)
                self.conn.register("teams_temp", teams_df)

                self.conn.execute("""
                CREATE TABLE canonical_teams AS
                SELECT * FROM teams_temp
                ORDER BY team_id
                """)

                self.conn.execute("DROP VIEW IF EXISTS teams_temp")

            details = f"Created canonical tables: players (view), starters ({len(starters_data)}), teams ({len(team_data)})"

            return ValidationResult(
                step_name="Create Canonical Tables",
                passed=True,
                details=details,
                data_count=len(starters_data) + len(team_data),
                processing_time=time.time() - start_time
            )

        except Exception as e:
            return ValidationResult(
                step_name="Create Canonical Tables",
                passed=False,
                details=f"Error creating canonical tables: {str(e)}",
                processing_time=time.time() - start_time
            )

    def validate_entity_completeness(self) -> ValidationResult:
        """Final validation that all entities are complete and consistent"""
        start_time = time.time()

        try:
            logger.info("Performing final entity validation...")

            # Use the GameEntities validation method
            entity_errors = self.entities.validate_completeness()

            warnings = []

            # Cross-validate entities
            if self.entities.unique_players is not None and len(self.entities.starters) > 0:
                # Check all starters are in unique_players
                all_player_ids = set(self.entities.unique_players['player_id'])

                for team, starters_list in self.entities.starters.items():
                    if isinstance(starters_list, list):
                        for starter in starters_list:
                            if starter['player_id'] not in all_player_ids:
                                warnings.append(f"Starter {starter['player_name']} not found in unique_players")

            # Validate team consistency
            if self.entities.unique_players is not None:
                player_teams = set(self.entities.unique_players['team_id'])
                mapping_teams = {k for k in self.entities.team_mapping.keys() if isinstance(k, int)}

                if player_teams != mapping_teams:
                    warnings.append(f"Team ID mismatch: players have {player_teams}, mapping has {mapping_teams}")

            # Check starter counts
            starter_counts = {}
            for team, starters_list in self.entities.starters.items():
                if isinstance(starters_list, list):
                    starter_counts[team] = len(starters_list)

            for team, count in starter_counts.items():
                if count != 5:
                    warnings.append(f"Team {team} has {count} starters (expected 5)")

            passed = len(entity_errors) == 0 and all(count == 5 for count in starter_counts.values())

            details = f"Entity validation: {len(entity_errors)} errors, {len(warnings)} warnings"
            details += f". Starter counts: {starter_counts}"

            if entity_errors:
                details += f" - Errors: {', '.join(entity_errors)}"

            return ValidationResult(
                step_name="Entity Completeness",
                passed=passed,
                details=details,
                processing_time=time.time() - start_time,
                warnings=warnings + entity_errors
            )

        except Exception as e:
            return ValidationResult(
                step_name="Entity Completeness",
                passed=False,
                details=f"Error validating entities: {str(e)}",
                processing_time=time.time() - start_time
            )

    def print_entities_summary(self):
        """Print comprehensive summary of extracted entities"""
        print("\n" + "="*80)
        print("ROBUST NBA ENTITY EXTRACTION SUMMARY")
        print("="*80)

        # Game info
        if self.entities.game_info:
            print(f"🏀 GAME: {self.entities.game_info.get('matchup', 'Unknown')}")
            print(f"   Teams: {', '.join(self.entities.game_info.get('teams', []))}")
            print(f"   Players: {self.entities.game_info.get('total_players', 0)}")
            print(f"   Starters: {self.entities.game_info.get('total_starters', 0)}")
            print()

        # Players summary
        if self.entities.unique_players is not None:
            print("👥 PLAYERS BY TEAM:")
            for team in self.entities.unique_players['team_abbrev'].unique():
                team_players = self.entities.unique_players[self.entities.unique_players['team_abbrev'] == team]
                starters = team_players[team_players['is_starter'] == True]
                print(f"   {team}: {len(team_players)} players ({len(starters)} starters)")
            print()

        # Starters detail
        print("🏆 STARTING LINEUPS:")
        for team, starters_list in self.entities.starters.items():
            if isinstance(starters_list, list):
                print(f"   {team}:")
                for i, starter in enumerate(starters_list):
                    jersey = starter.get('jersey_number', 'N/A')
                    seconds = starter.get('seconds_played', 0)
                    points = starter.get('points', 0)
                    print(f"     {i+1}. {starter['player_name']} (#{jersey}, {seconds//60}:{seconds%60:02d}, {points}pts)")
        print()

        # Team mapping
        print("🏟️  TEAM MAPPING:")
        for team_id, team_abbrev in self.entities.team_mapping.items():
            if isinstance(team_id, int):
                home_away = "🏠" if team_abbrev == self.entities.team_mapping.get('home_team') else "✈️"
                print(f"   {team_id} → {team_abbrev} {home_away}")

        print("="*80)

def extract_all_entities_robust(db_path: str = "mavs_enhanced.duckdb") -> Tuple[bool, GameEntities]:
    """Extract all canonical entities with robust validation"""

    print("🏀 NBA Pipeline - Robust Entity Extraction")
    print("="*50)

    with RobustEntityExtractor(db_path) as extractor:

        # Extract unique players
        logger.info("Step 3a: Extracting unique players...")
        result = extractor.extract_unique_players()
        extractor.validator.log_validation(result)

        if not result.passed:
            logger.error("❌ Failed to extract players - stopping")
            return False, extractor.entities

        # Extract starters (using actual gs=1 data from box score)
        logger.info("Step 3b: Extracting starters from box score...")
        result = extractor.extract_starters()
        extractor.validator.log_validation(result)

        if not result.passed:
            logger.error("❌ Failed to extract starters - stopping")
            return False, extractor.entities

        # Extract team mapping
        logger.info("Step 3c: Creating team mapping...")
        result = extractor.extract_team_mapping()
        extractor.validator.log_validation(result)

        # Extract game info
        logger.info("Step 3d: Extracting game info...")
        result = extractor.extract_game_info()
        extractor.validator.log_validation(result)

        # Create canonical tables
        logger.info("Step 3e: Creating canonical tables...")
        result = extractor.create_canonical_tables()
        extractor.validator.log_validation(result)

        # Final validation
        logger.info("Step 3f: Final validation...")
        result = extractor.validate_entity_completeness()
        extractor.validator.log_validation(result)

        # Print summary
        extractor.print_entities_summary()
        success = extractor.validator.print_validation_summary()

        return success, extractor.entities

# Example usage
if __name__ == "__main__":
    database_path = "mavs_enhanced.duckdb"

    success, entities = extract_all_entities_robust(database_path)

    if success:
        print("\n✅ Robust entity extraction completed successfully")
        print("🎯 Ready for lineup tracking and possession analysis")
    else:
        print("\n❌ Robust entity extraction failed")
        print("🔧 Review validation messages above")

