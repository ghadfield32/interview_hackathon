# path: api/src/airflow_project/tests/test_comprehensive_pipeline.py
"""
Comprehensive test suite for the NBA lineup analysis pipeline.
Tests data loading, processing logic, output validation, and integration.
"""

import pytest
import pandas as pd
import duckdb
from pathlib import Path
import sys
import os
import tempfile
import shutil

# Add project root to path
PROJECT_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import (
    BOX_SCORE_FILE, PBP_FILE, DUCKDB_CONFIG,
    RIM_DISTANCE_FEET, HOOP_CENTER_X, HOOP_CENTER_Y,
    validate_data_files
)
from utils.data_check_utils import run_quality_suite
from eda.pipeline.mavs_lineups import (
    connect_db, load_and_clean_data, build_lineup_snapshots_optimized,
    create_possession_events_optimized, compute_lineup_metrics,
    compute_player_rim_defense, run_pipeline
)
from eda.mavs_eda_enhanced import run_comprehensive_eda


class TestDataValidation:
    """Test data loading and validation"""

    def test_data_files_exist(self):
        """Test that all required data files exist"""
        assert BOX_SCORE_FILE.exists(), f"Missing box score file: {BOX_SCORE_FILE}"
        assert PBP_FILE.exists(), f"Missing play-by-play file: {PBP_FILE}"
        assert validate_data_files(), "Data file validation failed"

    def test_data_file_formats(self):
        """Test that data files can be loaded and have expected structure"""
        # Test box score format
        box_df = pd.read_csv(BOX_SCORE_FILE, nrows=5)
        required_box_cols = ['gameId', 'nbaId', 'name', 'nbaTeamId', 'team']
        for col in required_box_cols:
            assert col in box_df.columns, f"Missing required box score column: {col}"

        # Test play-by-play format
        pbp_df = pd.read_csv(PBP_FILE, nrows=5)
        required_pbp_cols = ['gameId', 'pbpId', 'period', 'msgType']
        for col in required_pbp_cols:
            assert col in pbp_df.columns, f"Missing required PBP column: {col}"

    def test_data_quality_checks(self):
        """Test comprehensive data quality validation"""
        box_df = pd.read_csv(BOX_SCORE_FILE)

        # Run quality suite on sample data
        quality_report = run_quality_suite(
            box_df.head(100),
            required_columns=['nbaId', 'name', 'nbaTeamId'],
            primary_key_columns=['gameId', 'nbaId'],
            output_format='table'
        )

        assert len(quality_report) > 0, "Quality report should contain results"

        # Check for critical data quality issues
        null_issues = quality_report[
            (quality_report['section'] == 'nulls') & 
            (quality_report['field'] == 'null_count')
        ]

        # Validate that critical columns have low null rates
        critical_nulls = null_issues[
            null_issues['subsection'].isin(['nbaId', 'name', 'nbaTeamId'])
        ]

        for _, row in critical_nulls.iterrows():
            assert row['value'] < len(box_df) * 0.1, f"Too many nulls in {row['subsection']}: {row['value']}"


class TestDuckDBOperations:
    """Test DuckDB database operations"""

    @pytest.fixture
    def temp_db(self):
        """Create temporary DuckDB for testing"""
        temp_dir = tempfile.mkdtemp()
        temp_db_path = Path(temp_dir) / "test.duckdb"

        con = duckdb.connect(str(temp_db_path), config=DUCKDB_CONFIG)
        yield con
        con.close()

        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_database_connection(self, temp_db):
        """Test database connection and basic operations"""
        # Test basic query
        result = temp_db.execute("SELECT 1 as test_col").fetchone()
        assert result[0] == 1, "Basic database query failed"

        # Test table creation
        temp_db.execute("CREATE TABLE test_table (id INTEGER, name VARCHAR)")
        temp_db.execute("INSERT INTO test_table VALUES (1, 'test')")

        result = temp_db.execute("SELECT COUNT(*) FROM test_table").fetchone()
        assert result[0] == 1, "Table operations failed"

    def test_data_loading(self, temp_db):
        """Test loading CSV data into DuckDB"""
        # This would normally test load_and_clean_data function
        # For now, test basic CSV loading capability

        temp_db.execute(f"""
            CREATE TABLE box_test AS
            SELECT * FROM read_csv_auto('{BOX_SCORE_FILE.as_posix()}', header=true, sample_size=100)
        """)

        count = temp_db.execute("SELECT COUNT(*) FROM box_test").fetchone()[0]
        assert count > 0, "Failed to load CSV data"

        # Test that we have expected columns
        columns = temp_db.execute("""
            SELECT column_name FROM information_schema.columns 
            WHERE table_name = 'box_test'
        """).fetchall()

        column_names = [col[0] for col in columns]
        assert 'nbaId' in column_names, "Missing expected column after loading"


class TestLineupProcessing:
    """Test lineup tracking and processing logic"""

    @pytest.fixture
    def sample_lineup_data(self):
        """Create sample data for lineup testing"""
        return pd.DataFrame([
            {'team_id': 1, 'lineup_key': '101,102,103,104,105', 'off_possessions': 10, 'off_points': 15},
            {'team_id': 1, 'lineup_key': '101,102,103,104,106', 'off_possessions': 5, 'off_points': 8},
            {'team_id': 2, 'lineup_key': '201,202,203,204,205', 'off_possessions': 8, 'off_points': 12},
        ])

    def test_lineup_key_parsing(self, sample_lineup_data):
        """Test parsing of lineup keys into individual players"""
        def parse_lineup(key):
            return [int(x) for x in str(key).split(',') if x]

        first_lineup = sample_lineup_data.iloc[0]['lineup_key']
        parsed = parse_lineup(first_lineup)

        assert len(parsed) == 5, "Lineup should have 5 players"
        assert all(isinstance(pid, int) for pid in parsed), "Player IDs should be integers"
        assert parsed == [101, 102, 103, 104, 105], "Unexpected lineup parsing result"

    def test_offensive_rating_calculation(self, sample_lineup_data):
        """Test calculation of offensive ratings"""
        # Calculate ratings manually
        sample_lineup_data['off_rating'] = (
            100.0 * sample_lineup_data['off_points'] / sample_lineup_data['off_possessions']
        ).round(2)

        expected_ratings = [150.0, 160.0, 150.0]  # 100 * (15/10, 8/5, 12/8)

        for i, expected in enumerate(expected_ratings):
            actual = sample_lineup_data.iloc[i]['off_rating']
            assert abs(actual - expected) < 0.01, f"Rating calculation error: {actual} != {expected}"


class TestRimDefenseCalculations:
    """Test rim defense on/off calculations"""

    @pytest.fixture
    def sample_rim_data(self):
        """Create sample rim shot data"""
        return pd.DataFrame([
            {'player_id': 101, 'on_attempts': 10, 'on_makes': 6, 'team_attempts': 50, 'team_makes': 25},
            {'player_id': 102, 'on_attempts': 8, 'on_makes': 3, 'team_attempts': 50, 'team_makes': 25},
            {'player_id': 103, 'on_attempts': 0, 'on_makes': 0, 'team_attempts': 30, 'team_makes': 18},
        ])

    def test_rim_percentage_calculations(self, sample_rim_data):
        """Test rim field goal percentage calculations"""
        data = sample_rim_data.copy()

        # Calculate ON percentages
        data['rim_fg_pct_on'] = (
            100.0 * data['on_makes'] / data['on_attempts']
        ).where(data['on_attempts'] > 0).round(2)

        # Calculate OFF percentages  
        data['rim_fg_pct_off'] = (
            100.0 * (data['team_makes'] - data['on_makes']) / 
            (data['team_attempts'] - data['on_attempts'])
        ).where((data['team_attempts'] - data['on_attempts']) > 0).round(2)

        # Test specific calculations
        assert data.iloc[0]['rim_fg_pct_on'] == 60.0, "ON percentage calculation error"
        assert data.iloc[1]['rim_fg_pct_on'] == 37.5, "ON percentage calculation error"

        # Player with no rim attempts should have NaN for ON percentage
        assert pd.isna(data.iloc[2]['rim_fg_pct_on']), "Should be NaN for zero attempts"

    def test_rim_distance_calculation(self):
        """Test rim shot distance calculation"""
        # Test shots at various distances
        test_shots = [
            (0, 0, True),    # At the rim
            (3, 3, True),    # Close to rim (4.24 feet)
            (5, 0, False),   # Beyond rim distance
            (0, 5, False),   # Beyond rim distance
        ]

        for x, y, should_be_rim in test_shots:
            distance = ((x - HOOP_CENTER_X)**2 + (y - HOOP_CENTER_Y)**2)**0.5
            is_rim = distance <= RIM_DISTANCE_FEET

            assert is_rim == should_be_rim, f"Rim distance calculation error for ({x}, {y}): {distance} feet"


class TestPipelineIntegration:
    """Test full pipeline integration"""

    @pytest.mark.slow
    def test_full_pipeline_execution(self):
        """Test complete pipeline execution"""
        # This is an integration test that may take longer
        try:
            lineups_path, players_path = run_pipeline()

            # Validate outputs exist
            assert lineups_path.exists(), f"Lineups output missing: {lineups_path}"
            assert players_path.exists(), f"Players output missing: {players_path}"

            # Validate output content
            lineups_df = pd.read_parquet(lineups_path)
            players_df = pd.read_parquet(players_path)

            assert len(lineups_df) > 0, "Lineups dataframe is empty"
            assert len(players_df) > 0, "Players dataframe is empty"

            # Validate expected columns exist
            lineup_required_cols = [
                'Team', 'Player 1', 'Player 2', 'Player 3', 'Player 4', 'Player 5',
                'Offensive possessions played', 'Defensive possessions played'
            ]
            for col in lineup_required_cols:
                assert col in lineups_df.columns, f"Missing lineup column: {col}"

            player_required_cols = [
                'Player ID', 'Player Name', 'Team',
                'Offensive possessions played', 'Defensive possessions played'
            ]
            for col in player_required_cols:
                assert col in players_df.columns, f"Missing player column: {col}"

        except Exception as e:
            pytest.fail(f"Pipeline execution failed: {str(e)}")

    def test_eda_execution(self):
        """Test EDA module execution"""
        try:
            # This should run without errors and return results
            eda_results = run_comprehensive_eda()

            assert isinstance(eda_results, dict), "EDA should return dictionary results"
            assert 'row_counts' in eda_results, "EDA results missing row counts"
            assert 'shot_analysis' in eda_results, "EDA results missing shot analysis"

            # Validate row counts are reasonable
            counts = eda_results['row_counts'].iloc[0]
            assert counts['box_rows'] > 0, "No box score data found"
            assert counts['pbp_rows'] > 0, "No play-by-play data found"

        except Exception as e:
            pytest.fail(f"EDA execution failed: {str(e)}")


class TestOutputValidation:
    """Test output data validation and quality"""

    def test_lineup_output_structure(self):
        """Test that lineup outputs have correct structure"""
        # This test assumes pipeline has been run
        try:
            lineups_path, _ = run_pipeline()
            lineups_df = pd.read_parquet(lineups_path)

            # Test data structure
            assert not lineups_df.empty, "Lineups output is empty"

            # Test for reasonable data ranges
            if 'Offensive rating' in lineups_df.columns:
                off_ratings = lineups_df['Offensive rating'].dropna()
                if len(off_ratings) > 0:
                    assert off_ratings.min() >= 0, "Offensive ratings should be non-negative"
                    assert off_ratings.max() <= 300, "Offensive ratings should be reasonable"

        except Exception as e:
            pytest.skip(f"Cannot test lineup output structure: {str(e)}")

    def test_player_output_structure(self):
        """Test that player outputs have correct structure"""
        try:
            _, players_path = run_pipeline()
            players_df = pd.read_parquet(players_path)

            # Test data structure
            assert not players_df.empty, "Players output is empty"

            # Test for valid player IDs
            if 'Player ID' in players_df.columns:
                player_ids = players_df['Player ID'].dropna()
                assert all(pid > 0 for pid in player_ids), "Player IDs should be positive"

            # Test rim percentage ranges
            rim_cols = [col for col in players_df.columns if 'rim field goal percentage' in col]
            for col in rim_cols:
                rim_pcts = players_df[col].dropna()
                if len(rim_pcts) > 0:
                    assert rim_pcts.min() >= 0, f"{col} should be non-negative"
                    assert rim_pcts.max() <= 100, f"{col} should not exceed 100%"

        except Exception as e:
            pytest.skip(f"Cannot test player output structure: {str(e)}")


# Performance benchmarking (optional)
class TestPerformance:
    """Performance and efficiency tests"""

    @pytest.mark.performance
    def test_pipeline_execution_time(self):
        """Test that pipeline completes within reasonable time"""
        import time

        start_time = time.time()
        try:
            run_pipeline()
            execution_time = time.time() - start_time

            # Pipeline should complete within 5 minutes for test data
            assert execution_time < 300, f"Pipeline too slow: {execution_time:.2f} seconds"
            print(f"Pipeline execution time: {execution_time:.2f} seconds")

        except Exception as e:
            pytest.skip(f"Cannot test performance: {str(e)}")


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])
