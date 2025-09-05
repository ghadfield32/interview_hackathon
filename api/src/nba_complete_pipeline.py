# Step 7: Complete NBA Pipeline Integration
"""
NBA Pipeline - Step 7: Complete Integration
===========================================

This is the master pipeline that integrates all steps into a single, cohesive workflow.
It provides easy-to-use functions that execute the entire NBA data processing pipeline
from raw CSV files to final validated results.

Key Features:
- Single function to run entire pipeline
- Comprehensive error handling and rollback
- Detailed progress reporting
- Configurable validation thresholds
- Automatic cleanup on failure
- Performance monitoring

Usage:
    success = run_complete_nba_pipeline(data_directory)
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
import traceback
import sys
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

# Import all pipeline components
from eda.utils.nba_pipeline_analysis import NBADataValidator, ValidationResult
from eda.data.nba_data_loader import EnhancedNBADataLoader, load_all_data_enhanced
from eda.data.nba_entities_extractor import RobustEntityExtractor, extract_all_entities_robust, GameEntities
from eda.data.nba_pbp_processor import PBPProcessor, process_pbp_events
from eda.data.nba_possession_engine import PossessionEngine, calculate_all_statistics
from eda.data.nba_final_export import FinalValidator, validate_and_export_results

# Configure logging for pipeline
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('nba_pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class PipelineConfig:
    """Configuration for NBA pipeline execution"""
    # File paths
    data_directory: Path = Path('api/src/airflow_project/data/mavs_data_engineer_2025')
    database_path: str = "mavs_enhanced.duckdb"
    export_directory: Path = Path('exports')

    # Processing options
    skip_validation: bool = False
    export_detailed: bool = True
    cleanup_on_failure: bool = True

    # Performance settings
    max_runtime_minutes: int = 10
    memory_limit_gb: int = 4
    threads: int = 4

    # Validation thresholds
    min_possessions_per_lineup: int = 1
    max_possession_difference_pct: float = 0.10
    min_players_with_rim_data: int = 5

class CompletePipeline:
    """Complete NBA data processing pipeline"""

    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()
        self.validator = NBADataValidator()
        self.start_time = None
        self.components = {}

        # Ensure directories exist
        self.config.export_directory.mkdir(parents=True, exist_ok=True)

        # Setup logging
        log_file = self.config.export_directory / "pipeline_execution.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    def run_complete_pipeline(self) -> Tuple[bool, Dict[str, Any]]:
        """Execute the complete NBA data processing pipeline"""

        self.start_time = time.time()

        try:
            logger.info("🏀 Starting Complete NBA Data Processing Pipeline")
            logger.info("="*80)
            self._log_pipeline_config()

            # Step 1: Validate input files
            logger.info("📋 STEP 1: Validating Input Files")
            step1_success = self._validate_input_files()
            if not step1_success:
                return False, {"error": "Input file validation failed"}

            # Step 2: Load and normalize data
            logger.info("💾 STEP 2: Loading and Normalizing Data")
            step2_success = self._load_and_normalize_data()
            if not step2_success:
                return False, {"error": "Data loading failed"}

            # Step 3: Extract canonical entities
            logger.info("👥 STEP 3: Extracting Canonical Entities")
            step3_success = self._extract_canonical_entities()
            if not step3_success:
                return False, {"error": "Entity extraction failed"}

            # Step 4: Process play-by-play events
            logger.info("🏀 STEP 4: Processing Play-by-Play Events")
            step4_success = self._process_pbp_events()
            if not step4_success:
                return False, {"error": "PBP processing failed"}

            # Step 5: Calculate possessions and statistics
            logger.info("📊 STEP 5: Calculating Possessions and Statistics")
            step5_success = self._calculate_statistics()
            if not step5_success:
                return False, {"error": "Statistics calculation failed"}

            # Step 6: Final validation and export
            logger.info("✅ STEP 6: Final Validation and Export")
            step6_success = self._validate_and_export()
            if not step6_success:
                return False, {"error": "Final validation/export failed"}

            # Pipeline completed successfully
            total_time = time.time() - self.start_time

            logger.info("🎉 PIPELINE COMPLETED SUCCESSFULLY")
            logger.info(f"⏱️  Total Runtime: {total_time:.2f} seconds")

            return True, {
                "success": True,
                "runtime_seconds": total_time,
                "components": self.components,
                "export_directory": str(self.config.export_directory),
                "database_path": self.config.database_path
            }

        except Exception as e:
            logger.error(f"❌ PIPELINE FAILED: {str(e)}")
            logger.error(f"Stack trace: {traceback.format_exc()}")

            if self.config.cleanup_on_failure:
                self._cleanup_on_failure()

            return False, {
                "error": str(e),
                "traceback": traceback.format_exc(),
                "runtime_seconds": time.time() - self.start_time if self.start_time else 0
            }

    def _validate_input_files(self) -> bool:
        """Validate all required input files exist and are readable"""
        try:
            required_files = {
                'box_score': self.config.data_directory / 'box_HOU-DAL.csv',
                'pbp': self.config.data_directory / 'pbp_HOU-DAL.csv',
                'event_types': self.config.data_directory / 'pbp_event_msg_types.csv',
                'action_types': self.config.data_directory / 'pbp_action_types.csv',
                'option_types': self.config.data_directory / 'pbp_option_types.csv'
            }

            validation_result = self.validator.validate_file_structure(required_files)
            self.validator.log_validation(validation_result)

            if not validation_result.passed:
                logger.error("Required files are missing or inaccessible")
                return False

            # Check file sizes for reasonableness
            for name, path in required_files.items():
                if path.exists():
                    size_mb = path.stat().st_size / (1024*1024)
                    logger.info(f"   📄 {name}: {size_mb:.1f} MB")

                    if size_mb == 0:
                        logger.error(f"File {name} is empty")
                        return False

            return True

        except Exception as e:
            logger.error(f"File validation error: {str(e)}")
            return False

    def _load_and_normalize_data(self) -> bool:
        """Load all data into DuckDB with validation"""
        try:
            success = load_all_data_enhanced(self.config.data_directory, self.config.database_path)

            if success:
                logger.info("✅ Data loading completed successfully")
                return True
            else:
                logger.error("❌ Data loading failed validation")
                return False

        except Exception as e:
            logger.error(f"Data loading error: {str(e)}")
            return False

    def _extract_canonical_entities(self) -> bool:
        """Extract players, starters, and team information"""
        try:
            success, entities = extract_all_entities_robust(self.config.database_path)

            if success and entities:
                self.components['entities'] = entities
                logger.info("✅ Entity extraction completed successfully")

                # Log key entity counts
                if entities.unique_players is not None:
                    logger.info(f"   Players extracted: {len(entities.unique_players)}")

                logger.info(f"   Teams: {len([k for k in entities.starters.keys() if not k.endswith('_ids')])}")

                return True
            else:
                logger.error("❌ Entity extraction failed")
                return False

        except Exception as e:
            logger.error(f"Entity extraction error: {str(e)}")
            return False

    def _process_pbp_events(self) -> bool:
        """Process play-by-play events with lineup tracking"""
        try:
            entities = self.components.get('entities')
            if not entities:
                logger.error("Entities not available for PBP processing")
                return False

            success, processor = process_pbp_events(self.config.database_path, entities)

            if success and processor:
                self.components['processor'] = processor
                logger.info("✅ PBP event processing completed successfully")

                # Log key processing stats
                if processor.processed_events:
                    events_count = len(processor.processed_events)
                    shots = sum(1 for e in processor.processed_events if e.is_shot)
                    rim_attempts = sum(1 for e in processor.processed_events if e.is_rim_attempt)
                    subs = len(processor.substitution_log) if hasattr(processor, 'substitution_log') else 0

                    logger.info(f"   Events processed: {events_count:,}")
                    logger.info(f"   Shots: {shots}, Rim attempts: {rim_attempts}, Substitutions: {subs}")

                return True
            else:
                logger.error("❌ PBP event processing failed")
                return False

        except Exception as e:
            logger.error(f"PBP processing error: {str(e)}")
            return False

    def _calculate_statistics(self) -> bool:
        """Calculate possessions and lineup/player statistics"""
        try:
            entities = self.components.get('entities')
            processor = self.components.get('processor')

            if not entities or not processor:
                logger.error("Required components not available for statistics calculation")
                return False

            success, engine = calculate_all_statistics(self.config.database_path, entities, processor)

            if success and engine:
                self.components['engine'] = engine
                logger.info("✅ Statistics calculation completed successfully")

                # Log key statistics
                if engine.possessions:
                    logger.info(f"   Possessions identified: {len(engine.possessions):,}")

                if engine.lineup_stats:
                    logger.info(f"   Unique lineups: {len(engine.lineup_stats):,}")

                if engine.player_rim_stats:
                    players_with_rim = sum(1 for p in engine.player_rim_stats.values() if p.opp_rim_attempts_on > 0)
                    logger.info(f"   Players with rim data: {players_with_rim}")

                return True
            else:
                logger.error("❌ Statistics calculation failed")
                return False

        except Exception as e:
            logger.error(f"Statistics calculation error: {str(e)}")
            return False

    def _validate_and_export(self) -> bool:
        """Perform final validation and export results"""
        try:
            entities = self.components.get('entities')
            processor = self.components.get('processor')
            engine = self.components.get('engine')

            if not all([entities, processor, engine]):
                logger.error("Required components not available for final validation")
                return False

            success, validator = validate_and_export_results(
                self.config.database_path, entities, processor, engine
            )

            if success and validator:
                self.components['validator'] = validator
                logger.info("✅ Final validation and export completed successfully")

                # Log export details
                if validator.export_dir.exists():
                    export_files = list(validator.export_dir.glob("*.csv"))
                    logger.info(f"   Files exported: {len(export_files)}")
                    for file in export_files:
                        size_kb = file.stat().st_size / 1024
                        logger.info(f"   📄 {file.name} ({size_kb:.1f} KB)")

                return True
            else:
                logger.error("❌ Final validation and export failed")
                return False

        except Exception as e:
            logger.error(f"Final validation error: {str(e)}")
            return False

    def _cleanup_on_failure(self):
        """Clean up resources on pipeline failure"""
        try:
            logger.info("🧹 Cleaning up resources after failure...")

            # Remove incomplete database
            db_path = Path(self.config.database_path)
            if db_path.exists():
                db_path.unlink()
                logger.info(f"   Removed incomplete database: {db_path}")

            # Remove incomplete exports
            if self.config.export_directory.exists():
                for file in self.config.export_directory.glob("*.csv"):
                    file.unlink()
                    logger.info(f"   Removed incomplete export: {file.name}")

        except Exception as e:
            logger.warning(f"Cleanup error (non-critical): {str(e)}")

    def _log_pipeline_config(self):
        """Log pipeline configuration details"""
        logger.info("PIPELINE CONFIGURATION:")
        logger.info(f"   Data Directory: {self.config.data_directory}")
        logger.info(f"   Database Path: {self.config.database_path}")
        logger.info(f"   Export Directory: {self.config.export_directory}")
        logger.info(f"   Max Runtime: {self.config.max_runtime_minutes} minutes")
        logger.info(f"   Memory Limit: {self.config.memory_limit_gb} GB")
        logger.info(f"   Threads: {self.config.threads}")
        logger.info("")

    def get_results_summary(self) -> Dict[str, Any]:
        """Get summary of pipeline results"""
        if 'validator' not in self.components:
            return {"error": "Pipeline not completed successfully"}

        try:
            with duckdb.connect(self.config.database_path) as conn:
                # Get lineup results
                lineups = conn.execute("""
                SELECT 
                    team_abbrev,
                    COUNT(*) as lineup_count,
                    AVG(off_rating) as avg_off_rating,
                    AVG(def_rating) as avg_def_rating
                FROM final_lineups 
                WHERE off_possessions > 0
                GROUP BY team_abbrev
                ORDER BY team_abbrev
                """).df()

                # Get player results
                players = conn.execute("""
                SELECT 
                    team_abbrev,
                    COUNT(*) as player_count,
                    COUNT(CASE WHEN opp_rim_attempts_on > 0 THEN 1 END) as players_with_rim_data
                FROM final_players
                GROUP BY team_abbrev
                ORDER BY team_abbrev
                """).df()

                return {
                    "pipeline_success": True,
                    "runtime_seconds": time.time() - self.start_time if self.start_time else 0,
                    "lineup_summary": lineups.to_dict('records'),
                    "player_summary": players.to_dict('records'),
                    "export_files": [f.name for f in self.config.export_directory.glob("*.csv")],
                    "database_path": self.config.database_path
                }

        except Exception as e:
            return {"error": f"Could not generate summary: {str(e)}"}

# Main pipeline execution functions
def run_complete_nba_pipeline(data_directory: str = None, 
                             database_path: str = "mavs_enhanced.duckdb",
                             export_directory: str = "exports") -> Tuple[bool, Dict[str, Any]]:
    """
    Run the complete NBA data processing pipeline

    Args:
        data_directory: Path to directory containing CSV files
        database_path: Path for DuckDB database file
        export_directory: Path for exported results

    Returns:
        Tuple of (success: bool, results: dict)
    """

    # Create configuration
    config = PipelineConfig()

    if data_directory:
        config.data_directory = Path(data_directory)

    config.database_path = database_path
    config.export_directory = Path(export_directory)

    # Run pipeline
    pipeline = CompletePipeline(config)
    return pipeline.run_complete_pipeline()

def get_pipeline_results(database_path: str = "mavs_enhanced.duckdb") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Get the final results from a completed pipeline

    Args:
        database_path: Path to the pipeline database

    Returns:
        Tuple of (lineups_df, players_df)
    """

    try:
        with duckdb.connect(database_path) as conn:
            # Get formatted lineup results
            lineups = conn.execute("""
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
            FROM final_lineups
            WHERE off_possessions > 0 OR def_possessions > 0
            ORDER BY team_abbrev, off_possessions DESC
            """).df()

            # Get formatted player results
            players = conn.execute("""
            SELECT 
                player_id as "Player ID",
                player_name as "Player Name",
                team_abbrev as "Team",
                off_possessions as "Offensive possessions played",
                def_possessions as "Defensive possessions played",
                COALESCE(opp_rim_fg_pct_on, 0) as "Opponent rim field goal percentage when player is on the court",
                COALESCE(opp_rim_fg_pct_off, 0) as "Opponent rim field goal percentage when player is off the court", 
                COALESCE(rim_defense_on_off, 0) as "Opponent rim field goal percentage on/off difference (on-off)"
            FROM final_players
            WHERE off_possessions > 0 OR def_possessions > 0
            ORDER BY team_abbrev, player_name
            """).df()

            return lineups, players

    except Exception as e:
        logger.error(f"Error retrieving results: {str(e)}")
        return pd.DataFrame(), pd.DataFrame()

# Example usage and testing
if __name__ == "__main__":
    print("🏀 NBA Complete Pipeline - Testing")
    print("="*50)

    # Example 1: Run complete pipeline with default settings
    print("\n📋 Running complete pipeline with default settings...")

    success, results = run_complete_nba_pipeline()

    if success:
        print("✅ Pipeline completed successfully!")
        print(f"⏱️  Runtime: {results['runtime_seconds']:.2f} seconds")
        print(f"📁 Results exported to: {results['export_directory']}")

        # Get and display results
        lineups_df, players_df = get_pipeline_results(results['database_path'])

        print(f"\n📊 FINAL RESULTS:")
        print(f"   Unique Lineups: {len(lineups_df):,}")
        print(f"   Active Players: {len(players_df):,}")

        # Show sample results
        if len(lineups_df) > 0:
            print(f"\n👥 Sample Lineups (Top 3):")
            sample_lineups = lineups_df.head(3)
            for i, row in sample_lineups.iterrows():
                print(f"   {row['Team']}: {row['Offensive rating']:.1f} ORtg, {row['Defensive rating']:.1f} DRtg")

        if len(players_df) > 0:
            print(f"\n🏀 Sample Players (Top 3):")
            sample_players = players_df.head(3)
            for i, row in sample_players.iterrows():
                rim_on = row['Opponent rim field goal percentage when player is on the court'] * 100
                rim_diff = row['Opponent rim field goal percentage on/off difference (on-off)'] * 100
                print(f"   {row['Player Name']} ({row['Team']}): {rim_on:.1f}% rim FG allowed, {rim_diff:+.1f}% on/off")

    else:
        print("❌ Pipeline failed!")
        print(f"Error: {results.get('error', 'Unknown error')}")
        if 'traceback' in results:
            print("Traceback:")
            print(results['traceback'])

    print("\n🎯 Pipeline testing complete")
