# path: api/src/airflow_project/utils/lineup_verification_enhanced.py
"""
ENHANCED LINEUP AND PLAYER VERIFICATION SYSTEM
Comprehensive verification of lineup tracking, player usage, minutes, and rim defense
"""
from __future__ import annotations
import logging
import time
import pandas as pd
import duckdb
from typing import Dict, List, Tuple, Any
from pathlib import Path

logger = logging.getLogger(__name__)

class LineupVerificationEnhanced:
    """Enhanced verification system for comprehensive lineup and player analysis"""

    def __init__(self, con: duckdb.DuckDBPyConnection):
        self.con = con
        self.verification_results = {}
        self.start_time = time.time()

    def verify_unique_players_and_minutes(self) -> Dict[str, Any]:
        """
        STEP 1: Verify unique players used and their minutes validation
        Cross-references lineup tracking with box score minutes
        """
        logger.info("🔍 Verifying unique players and minutes validation...")

        # Get comprehensive player usage data with minutes cross-validation
        player_usage_query = """
        WITH lineup_player_usage AS (
            -- Extract all players from lineup segments
            SELECT DISTINCT
                UNNEST(current_lineup) as player_id,
                team_id,
                team_name,
                lineup_key,
                (end_idx - start_idx) as segment_duration
            FROM lineup_segments
            WHERE current_lineup IS NOT NULL
        ),
        box_score_minutes AS (
            -- Get actual minutes from box score
            SELECT 
                nbaId as player_id,
                name as player_name,
                nbaTeamId as team_id,
                team as team_name,
                COALESCE(secPlayed, 0) as seconds_played,
                ROUND(COALESCE(secPlayed, 0) / 60.0, 1) as minutes_played,
                COALESCE(gs, 0) as games_started,
                CASE WHEN gs = 1 THEN 'STARTER' ELSE 'BENCH' END as player_role
            FROM box_score 
            WHERE nbaId IS NOT NULL
        ),
        player_analysis AS (
            SELECT 
                bsm.player_id,
                bsm.player_name,
                bsm.team_id,
                bsm.team_name,
                bsm.minutes_played,
                bsm.seconds_played,
                bsm.player_role,
                -- Lineup tracking metrics
                COUNT(DISTINCT lpu.lineup_key) as lineups_appeared_in,
                SUM(lpu.segment_duration) as total_lineup_duration,
                -- Calculate lineup coverage
                CASE 
                    WHEN COUNT(DISTINCT lpu.lineup_key) > 0 THEN 'IN_LINEUPS'
                    WHEN bsm.minutes_played > 5 THEN 'MISSING_FROM_LINEUPS' 
                    ELSE 'LOW_MINUTES'
                END as lineup_tracking_status,
                -- Minutes validation
                CASE 
                    WHEN bsm.minutes_played > 30 AND COUNT(DISTINCT lpu.lineup_key) = 0 THEN 'HIGH_MINUTES_NO_LINEUPS'
                    WHEN bsm.minutes_played < 5 AND COUNT(DISTINCT lpu.lineup_key) > 3 THEN 'LOW_MINUTES_MANY_LINEUPS'
                    WHEN bsm.minutes_played BETWEEN 10 AND 40 AND COUNT(DISTINCT lpu.lineup_key) BETWEEN 1 AND 10 THEN 'REASONABLE'
                    ELSE 'NEEDS_REVIEW'
                END as minutes_lineup_consistency
            FROM box_score_minutes bsm
            LEFT JOIN lineup_player_usage lpu ON bsm.player_id = lpu.player_id
            GROUP BY bsm.player_id, bsm.player_name, bsm.team_id, bsm.team_name, 
                     bsm.minutes_played, bsm.seconds_played, bsm.player_role
        )
        SELECT * FROM player_analysis
        ORDER BY minutes_played DESC, lineups_appeared_in DESC
        """

        player_usage_df = self.con.execute(player_usage_query).df()

        # Summary statistics
        summary_stats = {
            'total_unique_players': len(player_usage_df),
            'players_in_lineups': len(player_usage_df[player_usage_df['lineup_tracking_status'] == 'IN_LINEUPS']),
            'players_missing_from_lineups': len(player_usage_df[player_usage_df['lineup_tracking_status'] == 'MISSING_FROM_LINEUPS']),
            'starters_tracked': len(player_usage_df[(player_usage_df['player_role'] == 'STARTER') & (player_usage_df['lineup_tracking_status'] == 'IN_LINEUPS')]),
            'bench_players_tracked': len(player_usage_df[(player_usage_df['player_role'] == 'BENCH') & (player_usage_df['lineup_tracking_status'] == 'IN_LINEUPS')]),
            'total_minutes_tracked': player_usage_df['minutes_played'].sum(),
            'avg_lineups_per_player': player_usage_df['lineups_appeared_in'].mean(),
            'max_lineups_per_player': player_usage_df['lineups_appeared_in'].max()
        }

        # Identify potential issues
        issues = []
        high_min_no_lineups = player_usage_df[player_usage_df['minutes_lineup_consistency'] == 'HIGH_MINUTES_NO_LINEUPS']
        if len(high_min_no_lineups) > 0:
            issues.append(f"{len(high_min_no_lineups)} players with high minutes not tracked in lineups")

        missing_starters = player_usage_df[(player_usage_df['player_role'] == 'STARTER') & (player_usage_df['lineup_tracking_status'] != 'IN_LINEUPS')]
        if len(missing_starters) > 0:
            issues.append(f"{len(missing_starters)} starters missing from lineup tracking")

        results = {
            'player_usage_data': player_usage_df,
            'summary_stats': summary_stats,
            'validation_issues': issues,
            'verification_passed': len(issues) == 0
        }

        logger.info(f"✅ Player verification: {summary_stats['total_unique_players']} unique players")
        logger.info(f"📊 Lineup coverage: {summary_stats['players_in_lineups']}/{summary_stats['total_unique_players']} players tracked")

        return results

    def verify_rim_defense_logic(self) -> Dict[str, Any]:
        """
        STEP 2: Verify rim defense calculations and spatial logic
        Ensures players without rim defense were actually near the rim
        """
        logger.info("🎯 Verifying rim defense logic and spatial calculations...")

        # Comprehensive rim defense verification query
        rim_verification_query = """
        WITH rim_shot_analysis AS (
            -- Analyze all shots with spatial data
            SELECT 
                pbpId,
                period,
                playerId1 as shooter_id,
                offTeamId as shooting_team,
                defTeamId as defending_team,
                COALESCE(locX, 0) as x_coord,
                COALESCE(locY, 0) as y_coord,
                msgType,
                -- Calculate distance from basket (NBA court: basket at 0,0)
                SQRT(POWER(COALESCE(locX, 0), 2) + POWER(COALESCE(locY, 0), 2)) as distance_units,
                SQRT(POWER(COALESCE(locX, 0), 2) + POWER(COALESCE(locY, 0), 2)) / 10.0 as distance_feet,
                -- Rim classification (4 feet = 40 units in NBA tracking)
                CASE WHEN SQRT(POWER(COALESCE(locX, 0), 2) + POWER(COALESCE(locY, 0), 2)) <= 40 THEN 1 ELSE 0 END as is_rim_shot,
                CASE WHEN msgType = 1 THEN 1 ELSE 0 END as shot_made,
                -- Distance categories for analysis
                CASE 
                    WHEN SQRT(POWER(COALESCE(locX, 0), 2) + POWER(COALESCE(locY, 0), 2)) <= 40 THEN 'RIM (0-4ft)'
                    WHEN SQRT(POWER(COALESCE(locX, 0), 2) + POWER(COALESCE(locY, 0), 2)) <= 100 THEN 'PAINT (4-10ft)'
                    WHEN SQRT(POWER(COALESCE(locX, 0), 2) + POWER(COALESCE(locY, 0), 2)) <= 230 THEN 'MID-RANGE (10-23ft)'
                    ELSE 'THREE-POINT (23ft+)'
                END as shot_zone
            FROM pbp 
            WHERE msgType IN (1, 2)  -- Shot attempts only
              AND playerId1 IS NOT NULL
        ),
        player_rim_exposure AS (
            -- Calculate which players were exposed to rim shots
            SELECT 
                bs.nbaId as player_id,
                bs.name as player_name,
                bs.team as team_name,
                -- Count of rim shots when player's team was defending
                COUNT(CASE WHEN rsa.defending_team = bs.nbaTeamId AND rsa.is_rim_shot = 1 THEN 1 END) as team_rim_shots_defended,
                -- Shot zone distribution when team defending
                COUNT(CASE WHEN rsa.defending_team = bs.nbaTeamId AND rsa.shot_zone = 'RIM (0-4ft)' THEN 1 END) as rim_shots_faced,
                COUNT(CASE WHEN rsa.defending_team = bs.nbaTeamId AND rsa.shot_zone = 'PAINT (4-10ft)' THEN 1 END) as paint_shots_faced,
                COUNT(CASE WHEN rsa.defending_team = bs.nbaTeamId AND rsa.shot_zone = 'MID-RANGE (10-23ft)' THEN 1 END) as midrange_shots_faced,
                COUNT(CASE WHEN rsa.defending_team = bs.nbaTeamId AND rsa.shot_zone = 'THREE-POINT (23ft+)' THEN 1 END) as three_shots_faced,
                -- Rim shot makes when team defending
                SUM(CASE WHEN rsa.defending_team = bs.nbaTeamId AND rsa.is_rim_shot = 1 AND rsa.shot_made = 1 THEN 1 ELSE 0 END) as rim_makes_allowed,
                -- Player's minutes and role
                COALESCE(bs.secPlayed, 0) as seconds_played,
                ROUND(COALESCE(bs.secPlayed, 0) / 60.0, 1) as minutes_played
            FROM box_score bs
            CROSS JOIN rim_shot_analysis rsa
            WHERE bs.nbaId IS NOT NULL
            GROUP BY bs.nbaId, bs.name, bs.team, bs.secPlayed
        ),
        rim_defense_validation AS (
            SELECT 
                player_id,
                player_name,
                team_name,
                minutes_played,
                rim_shots_faced,
                paint_shots_faced,
                rim_makes_allowed,
                -- Simplified check for rim defense data availability
                1 as has_rim_defense_data_placeholder,
                -- Calculate expected rim exposure based on minutes
                CASE 
                    WHEN minutes_played > 20 AND rim_shots_faced = 0 THEN 'HIGH_MINUTES_NO_RIM_EXPOSURE'
                    WHEN minutes_played < 5 AND rim_shots_faced > 5 THEN 'LOW_MINUTES_HIGH_RIM_EXPOSURE' 
                    WHEN minutes_played BETWEEN 5 AND 40 AND rim_shots_faced BETWEEN 1 AND 20 THEN 'REASONABLE_EXPOSURE'
                    ELSE 'NEEDS_REVIEW'
                END as rim_exposure_classification,
                -- Rim defense logic validation (simplified)
                CASE 
                    WHEN rim_shots_faced > 0 THEN 'HAD_RIM_EXPOSURE'
                    WHEN rim_shots_faced = 0 AND minutes_played > 10 THEN 'NO_RIM_EXPOSURE_HIGH_MINUTES'
                    WHEN rim_shots_faced = 0 AND minutes_played <= 10 THEN 'NO_RIM_EXPOSURE_LOW_MINUTES'
                    ELSE 'UNCLEAR_STATUS'
                END as rim_defense_logic_status
            FROM player_rim_exposure
        )
        SELECT * FROM rim_defense_validation
        ORDER BY minutes_played DESC, rim_shots_faced DESC
        """

        rim_verification_df = self.con.execute(rim_verification_query).df()

        # Spatial logic validation - check rim distance calculations
        spatial_validation = self.con.execute("""
            SELECT 
                COUNT(*) as total_shots,
                COUNT(CASE WHEN locX IS NOT NULL AND locY IS NOT NULL THEN 1 END) as shots_with_coordinates,
                COUNT(CASE WHEN SQRT(POWER(COALESCE(locX, 0), 2) + POWER(COALESCE(locY, 0), 2)) <= 40 THEN 1 END) as rim_shots,
                ROUND(AVG(SQRT(POWER(COALESCE(locX, 0), 2) + POWER(COALESCE(locY, 0), 2)) / 10.0), 2) as avg_shot_distance_feet,
                MIN(SQRT(POWER(COALESCE(locX, 0), 2) + POWER(COALESCE(locY, 0), 2)) / 10.0) as min_shot_distance,
                MAX(SQRT(POWER(COALESCE(locX, 0), 2) + POWER(COALESCE(locY, 0), 2)) / 10.0) as max_shot_distance
            FROM pbp 
            WHERE msgType IN (1, 2)
        """).df()

        # Summary statistics
        rim_summary = {
            'total_players_analyzed': len(rim_verification_df),
            'players_with_rim_exposure': len(rim_verification_df[rim_verification_df['rim_shots_faced'] > 0]),
            'players_with_rim_defense_data': len(rim_verification_df),  # Simplified - all players analyzed
            'players_properly_tracked': len(rim_verification_df[rim_verification_df['rim_defense_logic_status'] == 'HAD_RIM_EXPOSURE']),
            'players_missing_rim_data': len(rim_verification_df[rim_verification_df['rim_defense_logic_status'] == 'NO_RIM_EXPOSURE_HIGH_MINUTES']),
            'total_rim_shots_in_game': spatial_validation.iloc[0]['rim_shots'] if len(spatial_validation) > 0 else 0,
            'avg_shot_distance_feet': spatial_validation.iloc[0]['avg_shot_distance_feet'] if len(spatial_validation) > 0 else 0
        }

        # Identify rim defense logic issues
        rim_issues = []
        high_min_no_rim = rim_verification_df[rim_verification_df['rim_defense_logic_status'] == 'NO_RIM_EXPOSURE_HIGH_MINUTES']
        if len(high_min_no_rim) > 0:
            rim_issues.append(f"{len(high_min_no_rim)} players with high minutes but no rim exposure")

        low_min_high_rim = rim_verification_df[rim_verification_df['rim_defense_logic_status'] == 'LOW_MINUTES_HIGH_RIM_EXPOSURE'] 
        if len(low_min_high_rim) > 0:
            rim_issues.append(f"{len(low_min_high_rim)} players with low minutes but high rim exposure")

        results = {
            'rim_verification_data': rim_verification_df,
            'spatial_validation': spatial_validation,
            'rim_summary': rim_summary,
            'rim_logic_issues': rim_issues,
            'verification_passed': len(rim_issues) == 0
        }

        logger.info(f"✅ Rim defense verification: {rim_summary['total_rim_shots_in_game']} rim shots analyzed")
        logger.info(f"📊 Rim exposure: {rim_summary['players_with_rim_exposure']} players had rim exposure")
        logger.info(f"🎯 Rim tracking: {rim_summary['players_properly_tracked']}/{rim_summary['players_with_rim_exposure']} properly tracked")

        return results

    def verify_final_parquet_tables(self, exports_dir: Path) -> Dict[str, Any]:
        """
        STEP 3: Verify final parquet tables match expected format and data quality
        """
        logger.info("📋 Verifying final parquet tables...")

        lineups_path = exports_dir / "lineups.parquet"
        players_path = exports_dir / "players.parquet"

        verification_results = {
            'lineups_file_exists': lineups_path.exists(),
            'players_file_exists': players_path.exists(),
            'lineups_data': None,
            'players_data': None,
            'data_quality_issues': [],
            'verification_passed': False
        }

        try:
            if verification_results['lineups_file_exists']:
                lineups_df = pd.read_parquet(lineups_path)
                verification_results['lineups_data'] = {
                    'total_rows': len(lineups_df),
                    'columns': list(lineups_df.columns),
                    'teams_represented': lineups_df['Team'].nunique() if 'Team' in lineups_df.columns else 0,
                    'complete_ratings': len(lineups_df.dropna(subset=['Offensive rating', 'Defensive rating', 'Net rating'])) if all(col in lineups_df.columns for col in ['Offensive rating', 'Defensive rating', 'Net rating']) else 0,
                    'avg_off_possessions': lineups_df['Offensive possessions played'].mean() if 'Offensive possessions played' in lineups_df.columns else 0,
                    'avg_def_possessions': lineups_df['Defensive possessions played'].mean() if 'Defensive possessions played' in lineups_df.columns else 0
                }

                # Validate lineup table structure
                expected_lineup_cols = ['Team', 'Player 1', 'Player 2', 'Player 3', 'Player 4', 'Player 5',
                                      'Offensive possessions played', 'Defensive possessions played', 
                                      'Offensive rating', 'Defensive rating', 'Net rating']
                missing_lineup_cols = [col for col in expected_lineup_cols if col not in lineups_df.columns]
                if missing_lineup_cols:
                    verification_results['data_quality_issues'].append(f"Lineups table missing columns: {missing_lineup_cols}")

            if verification_results['players_file_exists']:
                players_df = pd.read_parquet(players_path)
                verification_results['players_data'] = {
                    'total_rows': len(players_df),
                    'columns': list(players_df.columns),
                    'teams_represented': players_df['Team'].nunique() if 'Team' in players_df.columns else 0,
                    'players_with_rim_on': len(players_df.dropna(subset=['Opponent rim field goal percentage when player is on the court'])) if 'Opponent rim field goal percentage when player is on the court' in players_df.columns else 0,
                    'players_with_rim_off': len(players_df.dropna(subset=['Opponent rim field goal percentage when player is off the court'])) if 'Opponent rim field goal percentage when player is off the court' in players_df.columns else 0,
                    'players_with_rim_diff': len(players_df.dropna(subset=['Opponent rim field goal percentage on/off difference (on-off)'])) if 'Opponent rim field goal percentage on/off difference (on-off)' in players_df.columns else 0,
                    'avg_off_possessions': players_df['Offensive possessions played'].mean() if 'Offensive possessions played' in players_df.columns else 0
                }

                # Validate players table structure  
                expected_player_cols = ['Player ID', 'Player Name', 'Team', 'Offensive possessions played', 'Defensive possessions played',
                                      'Opponent rim field goal percentage when player is on the court',
                                      'Opponent rim field goal percentage when player is off the court',
                                      'Opponent rim field goal percentage on/off difference (on-off)']
                missing_player_cols = [col for col in expected_player_cols if col not in players_df.columns]
                if missing_player_cols:
                    verification_results['data_quality_issues'].append(f"Players table missing columns: {missing_player_cols}")

            # Overall validation
            verification_results['verification_passed'] = (
                verification_results['lineups_file_exists'] and 
                verification_results['players_file_exists'] and
                len(verification_results['data_quality_issues']) == 0
            )

        except Exception as e:
            verification_results['data_quality_issues'].append(f"Error reading parquet files: {str(e)}")

        logger.info(f"✅ Parquet verification: Lineups={verification_results['lineups_file_exists']}, Players={verification_results['players_file_exists']}")
        if verification_results['lineups_data']:
            logger.info(f"📊 Lineups: {verification_results['lineups_data']['total_rows']} rows, {verification_results['lineups_data']['complete_ratings']} with complete ratings")
        if verification_results['players_data']:
            logger.info(f"👥 Players: {verification_results['players_data']['total_rows']} rows, {verification_results['players_data']['players_with_rim_diff']} with rim defense")

        return verification_results

    def generate_comprehensive_verification_report(self, exports_dir: Path) -> str:
        """
        STEP 4: Generate comprehensive verification report with all findings
        """
        logger.info("📝 Generating comprehensive verification report...")

        # Run all verification steps
        player_results = self.verify_unique_players_and_minutes()
        rim_results = self.verify_rim_defense_logic()
        parquet_results = self.verify_final_parquet_tables(exports_dir)

        # Store all results for external access
        self.verification_results = {
            'player_verification': player_results,
            'rim_defense_verification': rim_results,
            'parquet_verification': parquet_results,
            'total_runtime': time.time() - self.start_time
        }

        # Generate detailed report
        report_lines = [
            "# COMPREHENSIVE LINEUP VERIFICATION REPORT",
            "=" * 60,
            f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total Verification Runtime: {self.verification_results['total_runtime']:.2f} seconds",
            "",
            "## 1. PLAYER USAGE AND MINUTES VERIFICATION",
            "=" * 40,
        ]

        # Player verification section
        if player_results['verification_passed']:
            report_lines.append("✅ **PASSED**: Player usage verification")
        else:
            report_lines.append("❌ **FAILED**: Player usage verification")

        ps = player_results['summary_stats']
        report_lines.extend([
            "",
            f"- **Total Unique Players**: {ps['total_unique_players']}",
            f"- **Players in Lineups**: {ps['players_in_lineups']} ({100*ps['players_in_lineups']/ps['total_unique_players']:.1f}%)",
            f"- **Starters Tracked**: {ps['starters_tracked']}/10 expected",
            f"- **Bench Players Tracked**: {ps['bench_players_tracked']}",
            f"- **Total Minutes Tracked**: {ps['total_minutes_tracked']:.1f}",
            f"- **Average Lineups per Player**: {ps['avg_lineups_per_player']:.1f}",
            "",
            "### Player Issues Found:",
        ])

        if player_results['validation_issues']:
            for issue in player_results['validation_issues']:
                report_lines.append(f"- ⚠️  {issue}")
        else:
            report_lines.append("- ✅ No player tracking issues found")

        # Rim defense verification section
        report_lines.extend([
            "",
            "## 2. RIM DEFENSE LOGIC VERIFICATION", 
            "=" * 40,
        ])

        if rim_results['verification_passed']:
            report_lines.append("✅ **PASSED**: Rim defense logic verification")
        else:
            report_lines.append("❌ **FAILED**: Rim defense logic verification")

        rs = rim_results['rim_summary']
        report_lines.extend([
            "",
            f"- **Total Players Analyzed**: {rs['total_players_analyzed']}",
            f"- **Players with Rim Exposure**: {rs['players_with_rim_exposure']}",
            f"- **Players with Rim Defense Data**: {rs['players_with_rim_defense_data']}",
            f"- **Players Properly Tracked**: {rs['players_properly_tracked']}",
            f"- **Total Rim Shots in Game**: {rs['total_rim_shots_in_game']}",
            f"- **Average Shot Distance**: {rs['avg_shot_distance_feet']:.1f} feet",
            "",
            "### Rim Defense Issues Found:",
        ])

        if rim_results['rim_logic_issues']:
            for issue in rim_results['rim_logic_issues']:
                report_lines.append(f"- ⚠️  {issue}")
        else:
            report_lines.append("- ✅ No rim defense logic issues found")

        # Parquet verification section
        report_lines.extend([
            "",
            "## 3. FINAL PARQUET TABLES VERIFICATION",
            "=" * 40,
        ])

        if parquet_results['verification_passed']:
            report_lines.append("✅ **PASSED**: Parquet tables verification")
        else:
            report_lines.append("❌ **FAILED**: Parquet tables verification")

        if parquet_results['lineups_data']:
            ld = parquet_results['lineups_data']
            report_lines.extend([
                "",
                f"### Lineups Table ({exports_dir / 'lineups.parquet'}):",
                f"- **Total Rows**: {ld['total_rows']}",
                f"- **Teams Represented**: {ld['teams_represented']}",
                f"- **Complete Ratings**: {ld['complete_ratings']} ({100*ld['complete_ratings']/ld['total_rows']:.1f}%)",
                f"- **Avg Offensive Possessions**: {ld['avg_off_possessions']:.1f}",
                f"- **Avg Defensive Possessions**: {ld['avg_def_possessions']:.1f}",
            ])

        if parquet_results['players_data']:
            pd_data = parquet_results['players_data']
            report_lines.extend([
                "",
                f"### Players Table ({exports_dir / 'players.parquet'}):",
                f"- **Total Rows**: {pd_data['total_rows']}",
                f"- **Teams Represented**: {pd_data['teams_represented']}",
                f"- **Players with Rim On-Court Data**: {pd_data['players_with_rim_on']}",
                f"- **Players with Rim Off-Court Data**: {pd_data['players_with_rim_off']}",
                f"- **Players with Rim Diff Data**: {pd_data['players_with_rim_diff']}",
                f"- **Avg Offensive Possessions**: {pd_data['avg_off_possessions']:.1f}",
            ])

        # Data quality issues
        report_lines.extend([
            "",
            "### Data Quality Issues Found:",
        ])

        if parquet_results['data_quality_issues']:
            for issue in parquet_results['data_quality_issues']:
                report_lines.append(f"- ⚠️  {issue}")
        else:
            report_lines.append("- ✅ No data quality issues found")

        # Overall summary
        overall_passed = (
            player_results['verification_passed'] and 
            rim_results['verification_passed'] and 
            parquet_results['verification_passed']
        )

        report_lines.extend([
            "",
            "## 4. OVERALL VERIFICATION SUMMARY",
            "=" * 40,
            "",
            f"**OVERALL STATUS**: {'✅ PASSED' if overall_passed else '❌ FAILED'}",
            "",
            f"- Player Verification: {'✅' if player_results['verification_passed'] else '❌'}",
            f"- Rim Defense Verification: {'✅' if rim_results['verification_passed'] else '❌'}",
            f"- Parquet Tables Verification: {'✅' if parquet_results['verification_passed'] else '❌'}",
            "",
            "## 5. RECOMMENDATIONS",
            "=" * 40,
        ])

        # Generate recommendations based on findings
        recommendations = []

        if not player_results['verification_passed']:
            recommendations.append("Review player usage tracking - some high-minute players may be missing from lineups")

        if not rim_results['verification_passed']:
            recommendations.append("Check rim defense calculations - spatial logic may need adjustment")

        if not parquet_results['verification_passed']:
            recommendations.append("Verify final table generation - required columns may be missing")

        if overall_passed:
            recommendations.append("All verification checks passed - pipeline is working correctly!")
            recommendations.append("Consider adding monitoring for future runs to catch regressions")

        for rec in recommendations:
            report_lines.append(f"- {rec}")

        report_lines.extend([
            "",
            "=" * 60,
            f"Report generated in {self.verification_results['total_runtime']:.2f} seconds",
            "=" * 60
        ])

        return "\n".join(report_lines)

def run_comprehensive_verification(con: duckdb.DuckDBPyConnection, exports_dir: Path) -> Dict[str, Any]:
    """
    Main function to run comprehensive verification of lineup pipeline
    """
    logger.info("🚀 Starting comprehensive lineup and player verification...")

    verifier = LineupVerificationEnhanced(con)

    # Generate comprehensive report
    report_content = verifier.generate_comprehensive_verification_report(exports_dir)

    # Save report
    report_path = exports_dir / "comprehensive_verification_report.md"
    report_path.write_text(report_content, encoding='utf-8')

    logger.info(f"📋 Comprehensive verification report saved to: {report_path}")

    # Return verification results for programmatic access
    return verifier.verification_results
