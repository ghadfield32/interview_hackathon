# path: api/src/airflow_project/utils/player_analysis_utility.py
"""
PLAYER ANALYSIS AND VERIFICATION UTILITY
Provides detailed analysis of unique players, minutes, rim defense status, and spatial verification
"""
from __future__ import annotations
import logging
import pandas as pd
import duckdb
from typing import Dict, List, Tuple
import numpy as np

logger = logging.getLogger(__name__)

def analyze_unique_players_and_minutes(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    Generate comprehensive analysis of all unique players with minutes and lineup usage

    Returns:
        DataFrame with complete player analysis including lineup tracking validation
    """
    logger.info("🔍 Analyzing unique players, minutes, and lineup usage...")

    query = """
    WITH box_score_base AS (
        SELECT 
            nbaId as player_id,
            name as player_name,
            nbaTeamId as team_id,
            team as team_name,
            COALESCE(secPlayed, 0) as seconds_played,
            ROUND(COALESCE(secPlayed, 0) / 60.0, 2) as minutes_played,
            COALESCE(gs, 0) as is_starter,
            CASE WHEN gs = 1 THEN 'STARTER' ELSE 'BENCH' END as player_role,
            COALESCE(pts, 0) as points,
            COALESCE(reb, 0) as rebounds,
            COALESCE(ast, 0) as assists,
            COALESCE(fgm, 0) as field_goals_made,
            COALESCE(fga, 0) as field_goals_attempted
        FROM box_score 
        WHERE nbaId IS NOT NULL
    ),
    lineup_tracking AS (
        SELECT 
            UNNEST(current_lineup) as player_id,
            lineup_key,
            team_id,
            (end_idx - start_idx) as segment_duration,
            segment_type
        FROM lineup_segments
        WHERE current_lineup IS NOT NULL
    ),
    player_lineup_stats AS (
        SELECT 
            player_id,
            COUNT(DISTINCT lineup_key) as unique_lineups,
            COUNT(*) as total_lineup_segments,
            SUM(segment_duration) as total_lineup_duration,
            COUNT(CASE WHEN segment_type = 'PERIOD_START' THEN 1 END) as period_start_segments,
            COUNT(CASE WHEN segment_type = 'POST_SUBSTITUTION' THEN 1 END) as post_sub_segments
        FROM lineup_tracking
        GROUP BY player_id
    ),
    rim_defense_tracking AS (
        SELECT 
            bs.nbaId as player_id,
            -- Check if player appears in any rim defense calculations
            COUNT(CASE WHEN rs.defending_team = bs.nbaTeamId AND rs.is_rim_shot = 1 THEN 1 END) as potential_rim_opportunities,
            -- Check if player has actual rim defense data in final calculations
            0 as has_rim_defense_data  -- Placeholder - will be updated by actual rim defense query
        FROM box_score bs
        LEFT JOIN (
            SELECT 
                defTeamId as defending_team,
                CASE WHEN SQRT(POWER(COALESCE(locX, 0), 2) + POWER(COALESCE(locY, 0), 2)) <= 40 THEN 1 ELSE 0 END as is_rim_shot
            FROM pbp 
            WHERE msgType IN (1, 2)
        ) rs ON 1=1
        WHERE bs.nbaId IS NOT NULL
        GROUP BY bs.nbaId
    )
    SELECT 
        bsb.player_id,
        bsb.player_name,
        bsb.team_id,
        bsb.team_name,
        bsb.minutes_played,
        bsb.seconds_played,
        bsb.player_role,
        bsb.is_starter,
        bsb.points,
        bsb.rebounds,
        bsb.assists,
        bsb.field_goals_made,
        bsb.field_goals_attempted,
        -- Lineup tracking metrics
        COALESCE(pls.unique_lineups, 0) as unique_lineups_appeared_in,
        COALESCE(pls.total_lineup_segments, 0) as total_lineup_segments,
        COALESCE(pls.total_lineup_duration, 0) as total_lineup_duration,
        COALESCE(pls.period_start_segments, 0) as period_start_segments,
        COALESCE(pls.post_sub_segments, 0) as post_sub_segments,
        -- Rim defense opportunity tracking  
        COALESCE(rdt.potential_rim_opportunities, 0) as potential_rim_opportunities,
        -- Analysis flags
        CASE 
            WHEN pls.unique_lineups > 0 THEN 'TRACKED_IN_LINEUPS'
            WHEN bsb.minutes_played > 10 THEN 'HIGH_MINUTES_NOT_TRACKED'
            WHEN bsb.minutes_played BETWEEN 3 AND 10 THEN 'MEDIUM_MINUTES_NOT_TRACKED'
            ELSE 'LOW_MINUTES_NOT_TRACKED'
        END as lineup_tracking_status,
        -- Minutes validation
        CASE 
            WHEN bsb.minutes_played > 25 AND COALESCE(pls.unique_lineups, 0) = 0 THEN 'CRITICAL_MISSING'
            WHEN bsb.minutes_played BETWEEN 15 AND 25 AND COALESCE(pls.unique_lineups, 0) <= 1 THEN 'LIKELY_MISSING'
            WHEN bsb.minutes_played BETWEEN 5 AND 15 AND COALESCE(pls.unique_lineups, 0) = 0 THEN 'POSSIBLY_MISSING'
            WHEN bsb.minutes_played < 5 AND COALESCE(pls.unique_lineups, 0) > 0 THEN 'UNEXPECTEDLY_TRACKED'
            ELSE 'REASONABLE'
        END as minutes_validation_status
    FROM box_score_base bsb
    LEFT JOIN player_lineup_stats pls ON bsb.player_id = pls.player_id
    LEFT JOIN rim_defense_tracking rdt ON bsb.player_id = rdt.player_id
    ORDER BY bsb.minutes_played DESC, bsb.team_id
    """

    return con.execute(query).df()

def analyze_rim_defense_players(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    Analyze which players have rim defense data vs those who don't, and verify spatial logic
    """
    logger.info("🎯 Analyzing rim defense coverage and spatial verification...")

    query = """
    WITH player_rim_analysis AS (
        SELECT 
            bs.nbaId as player_id,
            bs.name as player_name,
            bs.team as team_name,
            ROUND(COALESCE(bs.secPlayed, 0) / 60.0, 2) as minutes_played,
            -- Check if player appears in lineup segments (potential for rim defense tracking)
            EXISTS (
                SELECT 1 FROM lineup_segments ls 
                WHERE bs.nbaId = ANY(ls.current_lineup)
            ) as appears_in_lineups,
            -- Count rim shots when player's team was defending
            COUNT(CASE WHEN pbp.defTeamId = bs.nbaTeamId AND pbp.msgType IN (1,2) AND 
                              SQRT(POWER(COALESCE(pbp.locX, 0), 2) + POWER(COALESCE(pbp.locY, 0), 2)) <= 40 
                       THEN 1 END) as team_rim_shots_defended,
            -- Count all shots near rim (within 6 feet for broader analysis)
            COUNT(CASE WHEN pbp.defTeamId = bs.nbaTeamId AND pbp.msgType IN (1,2) AND 
                              SQRT(POWER(COALESCE(pbp.locX, 0), 2) + POWER(COALESCE(pbp.locY, 0), 2)) <= 60 
                       THEN 1 END) as team_near_rim_shots_defended,
            -- Shot distance analysis when team defending
            AVG(CASE WHEN pbp.defTeamId = bs.nbaTeamId AND pbp.msgType IN (1,2) AND pbp.locX IS NOT NULL 
                     THEN SQRT(POWER(COALESCE(pbp.locX, 0), 2) + POWER(COALESCE(pbp.locY, 0), 2)) / 10.0 
                END) as avg_shot_distance_defended_feet,
            MIN(CASE WHEN pbp.defTeamId = bs.nbaTeamId AND pbp.msgType IN (1,2) AND pbp.locX IS NOT NULL 
                     THEN SQRT(POWER(COALESCE(pbp.locX, 0), 2) + POWER(COALESCE(pbp.locY, 0), 2)) / 10.0 
                END) as closest_shot_defended_feet,
            -- Count shots in various zones when team defending
            COUNT(CASE WHEN pbp.defTeamId = bs.nbaTeamId AND pbp.msgType IN (1,2) AND 
                              SQRT(POWER(COALESCE(pbp.locX, 0), 2) + POWER(COALESCE(pbp.locY, 0), 2)) BETWEEN 40 AND 100 
                       THEN 1 END) as paint_shots_defended,
            COUNT(CASE WHEN pbp.defTeamId = bs.nbaTeamId AND pbp.msgType IN (1,2) AND 
                              SQRT(POWER(COALESCE(pbp.locX, 0), 2) + POWER(COALESCE(pbp.locY, 0), 2)) > 230 
                       THEN 1 END) as three_point_shots_defended
        FROM box_score bs
        CROSS JOIN pbp
        WHERE bs.nbaId IS NOT NULL
        GROUP BY bs.nbaId, bs.name, bs.team, bs.secPlayed
    ),
    rim_defense_status AS (
        SELECT 
            *,
            -- Determine rim defense eligibility
            CASE 
                WHEN NOT appears_in_lineups THEN 'NOT_IN_LINEUPS'
                WHEN team_rim_shots_defended = 0 THEN 'NO_RIM_EXPOSURE'
                WHEN team_rim_shots_defended > 0 AND minutes_played > 5 THEN 'ELIGIBLE_FOR_RIM_DEFENSE'
                WHEN team_rim_shots_defended > 0 AND minutes_played <= 5 THEN 'LIMITED_MINUTES_RIM_EXPOSURE'
                ELSE 'NEEDS_REVIEW'
            END as rim_defense_eligibility,
            -- Spatial verification for players without rim defense data
            CASE 
                WHEN team_rim_shots_defended = 0 AND closest_shot_defended_feet IS NOT NULL AND closest_shot_defended_feet <= 6.0 
                THEN 'WAS_NEAR_RIM_BUT_NO_RIM_SHOTS'
                WHEN team_rim_shots_defended = 0 AND closest_shot_defended_feet IS NULL 
                THEN 'NO_DEFENSIVE_SHOT_DATA'
                WHEN team_rim_shots_defended = 0 AND closest_shot_defended_feet > 6.0 
                THEN 'NOT_NEAR_RIM_APPROPRIATELY_NO_DATA'
                WHEN team_rim_shots_defended > 0 
                THEN 'HAD_RIM_EXPOSURE_SHOULD_HAVE_DATA'
                ELSE 'UNCLEAR_SPATIAL_STATUS'
            END as spatial_verification_status,
            -- Overall assessment
            CASE 
                WHEN appears_in_lineups AND team_rim_shots_defended > 0 THEN 'SHOULD_HAVE_RIM_DEFENSE_DATA'
                WHEN appears_in_lineups AND team_rim_shots_defended = 0 AND closest_shot_defended_feet <= 6.0 
                THEN 'REASONABLE_NO_RIM_DATA_BUT_WAS_NEAR_RIM'
                WHEN appears_in_lineups AND team_rim_shots_defended = 0 AND (closest_shot_defended_feet > 6.0 OR closest_shot_defended_feet IS NULL)
                THEN 'REASONABLE_NO_RIM_DATA_NOT_NEAR_RIM'
                WHEN NOT appears_in_lineups 
                THEN 'REASONABLE_NO_DATA_NOT_TRACKED'
                ELSE 'NEEDS_MANUAL_REVIEW'
            END as overall_rim_defense_assessment
        FROM player_rim_analysis
    )
    SELECT * FROM rim_defense_status
    ORDER BY minutes_played DESC, team_rim_shots_defended DESC
    """

    return con.execute(query).df()

def print_comprehensive_player_analysis(con: duckdb.DuckDBPyConnection) -> Dict[str, pd.DataFrame]:
    """
    Print comprehensive analysis as requested: unique players, minutes, rim defense status, spatial verification
    """
    logger.info("📊 COMPREHENSIVE PLAYER ANALYSIS - As Requested")
    logger.info("=" * 80)

    # Get comprehensive player data
    players_df = analyze_unique_players_and_minutes(con)
    rim_defense_df = analyze_rim_defense_players(con)

    # ===== UNIQUE PLAYERS AND MINUTES ANALYSIS =====
    logger.info("👥 UNIQUE PLAYERS AND MINUTES:")
    logger.info("-" * 40)
    logger.info(f"Total Unique Players: {len(players_df)}")

    # Print summary by team
    team_summary = players_df.groupby('team_name').agg({
        'player_name': 'count',
        'minutes_played': ['sum', 'mean'],
        'unique_lineups_appeared_in': 'sum',
        'is_starter': 'sum'
    }).round(2)
    team_summary.columns = ['Player_Count', 'Total_Minutes', 'Avg_Minutes', 'Total_Lineup_Appearances', 'Starters']

    logger.info("Per-Team Summary:")
    for team, row in team_summary.iterrows():
        logger.info(f"  {team}: {int(row['Player_Count'])} players, "
                   f"{row['Total_Minutes']:.1f} total mins, "
                   f"{row['Avg_Minutes']:.1f} avg mins, "
                   f"{int(row['Starters'])} starters")

    # ===== PLAYERS WITH/WITHOUT RIM DEFENSE =====
    logger.info("\n🎯 PLAYERS WITH/WITHOUT RIM DEFENSE:")
    logger.info("-" * 40)

    rim_summary = rim_defense_df['rim_defense_eligibility'].value_counts()
    logger.info("Rim Defense Eligibility Breakdown:")
    for status, count in rim_summary.items():
        logger.info(f"  {status}: {count} players")

    # ===== SPATIAL VERIFICATION FOR PLAYERS WITHOUT RIM DEFENSE =====
    logger.info("\n🌍 SPATIAL VERIFICATION - Players WITHOUT Rim Defense:")
    logger.info("-" * 40)

    no_rim_defense = rim_defense_df[rim_defense_df['rim_defense_eligibility'].isin(['NO_RIM_EXPOSURE', 'NOT_IN_LINEUPS'])]
    spatial_summary = no_rim_defense['spatial_verification_status'].value_counts()

    logger.info("Spatial Verification for Players Without Rim Defense:")
    for status, count in spatial_summary.items():
        logger.info(f"  {status}: {count} players")

    # Show specific players who were near rim but have no rim defense data
    near_rim_no_data = no_rim_defense[no_rim_defense['spatial_verification_status'] == 'WAS_NEAR_RIM_BUT_NO_RIM_SHOTS']
    if len(near_rim_no_data) > 0:
        logger.info(f"\n⚠️  Players who were near rim but have no rim defense data ({len(near_rim_no_data)}):")
        for _, player in near_rim_no_data.head(10).iterrows():
            logger.info(f"  - {player['player_name']} ({player['team_name']}): "
                       f"{player['minutes_played']:.1f} mins, "
                       f"closest shot: {player['closest_shot_defended_feet']:.1f} ft")

    # ===== HIGH-LEVEL VALIDATION SUMMARY =====
    logger.info("\n✅ LINEUP TRACKING VALIDATION SUMMARY:")
    logger.info("-" * 40)

    tracking_summary = players_df['lineup_tracking_status'].value_counts()
    for status, count in tracking_summary.items():
        logger.info(f"  {status}: {count} players")

    minutes_validation = players_df['minutes_validation_status'].value_counts()
    logger.info("\nMinutes Validation Summary:")
    for status, count in minutes_validation.items():
        logger.info(f"  {status}: {count} players")

    # ===== DETAILED PLAYER TABLES =====
    logger.info("\n📋 DETAILED PLAYER BREAKDOWN:")
    logger.info("-" * 40)

    # High-minute players breakdown
    high_minute_players = players_df[players_df['minutes_played'] > 15].sort_values('minutes_played', ascending=False)
    logger.info(f"\nHigh-Minute Players (>15 mins): {len(high_minute_players)}")
    logger.info("Player Name | Team | Minutes | Lineups | Tracking Status")
    logger.info("-" * 70)
    for _, player in high_minute_players.head(15).iterrows():
        logger.info(f"{player['player_name']:<20} | {player['team_name']:<3} | "
                   f"{player['minutes_played']:>6.1f} | {player['unique_lineups_appeared_in']:>7} | "
                   f"{player['lineup_tracking_status']}")

    # Rim defense breakdown for tracked players
    rim_eligible = rim_defense_df[rim_defense_df['rim_defense_eligibility'] == 'ELIGIBLE_FOR_RIM_DEFENSE'].sort_values('team_rim_shots_defended', ascending=False)
    logger.info(f"\nPlayers Eligible for Rim Defense ({len(rim_eligible)}):")
    logger.info("Player Name | Team | Minutes | Rim Shots | Spatial Status")
    logger.info("-" * 70)
    for _, player in rim_eligible.head(15).iterrows():
        logger.info(f"{player['player_name']:<20} | {player['team_name']:<3} | "
                   f"{player['minutes_played']:>6.1f} | {player['team_rim_shots_defended']:>9} | "
                   f"{player['overall_rim_defense_assessment']}")

    logger.info("=" * 80)
    logger.info("📊 COMPREHENSIVE PLAYER ANALYSIS COMPLETE")
    logger.info("=" * 80)

    return {
        'players_minutes': players_df,
        'rim_defense_analysis': rim_defense_df
    }

def print_final_parquet_table_summaries(exports_dir) -> Dict[str, pd.DataFrame]:
    """
    Print final parquet table summaries as requested
    """
    from pathlib import Path

    logger.info("📋 FINAL PARQUET TABLE SUMMARIES:")
    logger.info("=" * 60)

    results = {}

    try:
        # Load and analyze lineups table
        lineups_path = Path(exports_dir) / "lineups.parquet"
        if lineups_path.exists():
            lineups_df = pd.read_parquet(lineups_path)
            results['lineups'] = lineups_df

            logger.info("🏀 LINEUPS TABLE (lineups.parquet):")
            logger.info(f"  - Shape: {lineups_df.shape[0]} rows × {lineups_df.shape[1]} columns")
            logger.info(f"  - Columns: {list(lineups_df.columns)}")
            logger.info(f"  - Teams: {lineups_df['Team'].nunique()} ({lineups_df['Team'].unique()})")
            logger.info(f"  - Complete Ratings: {len(lineups_df.dropna(subset=['Offensive rating', 'Defensive rating']))} lineups")
            logger.info(f"  - Avg Off Rating: {lineups_df['Offensive rating'].mean():.1f}")
            logger.info(f"  - Avg Def Rating: {lineups_df['Defensive rating'].mean():.1f}")
            logger.info(f"  - Avg Net Rating: {lineups_df['Net rating'].mean():.1f}")

            logger.info("  Sample Lineups:")
            for i, row in lineups_df.head(3).iterrows():
                lineup_str = f"{row['Player 1']} | {row['Player 2']} | {row['Player 3']} | {row['Player 4']} | {row['Player 5']}"
                logger.info(f"    {row['Team']}: {lineup_str}")
                logger.info(f"      Off: {row['Offensive rating']:.1f}, Def: {row['Defensive rating']:.1f}, Net: {row['Net rating']:.1f}")

        # Load and analyze players table  
        players_path = Path(exports_dir) / "players.parquet"
        if players_path.exists():
            players_df = pd.read_parquet(players_path)
            results['players'] = players_df

            logger.info("\n👥 PLAYERS TABLE (players.parquet):")
            logger.info(f"  - Shape: {players_df.shape[0]} rows × {players_df.shape[1]} columns")
            logger.info(f"  - Columns: {list(players_df.columns)}")
            logger.info(f"  - Teams: {players_df['Team'].nunique()} ({players_df['Team'].unique()})")

            rim_on_col = 'Opponent rim field goal percentage when player is on the court'
            rim_off_col = 'Opponent rim field goal percentage when player is off the court' 
            rim_diff_col = 'Opponent rim field goal percentage on/off difference (on-off)'

            if rim_on_col in players_df.columns:
                logger.info(f"  - Players with Rim On-Court Data: {len(players_df.dropna(subset=[rim_on_col]))}")
                logger.info(f"  - Avg Rim FG% (On Court): {players_df[rim_on_col].mean():.1f}%")

            if rim_off_col in players_df.columns:
                logger.info(f"  - Players with Rim Off-Court Data: {len(players_df.dropna(subset=[rim_off_col]))}")
                logger.info(f"  - Avg Rim FG% (Off Court): {players_df[rim_off_col].mean():.1f}%")

            if rim_diff_col in players_df.columns:
                logger.info(f"  - Players with Rim Diff Data: {len(players_df.dropna(subset=[rim_diff_col]))}")
                logger.info(f"  - Avg Rim FG% Difference: {players_df[rim_diff_col].mean():.1f}%")

            logger.info("  Sample Players:")
            for i, row in players_df.head(5).iterrows():
                rim_on = row.get(rim_on_col, 'N/A')
                rim_off = row.get(rim_off_col, 'N/A')
                rim_diff = row.get(rim_diff_col, 'N/A')
                logger.info(f"    {row['Player Name']} ({row['Team']}): "
                           f"On={rim_on}, Off={rim_off}, Diff={rim_diff}")

    except Exception as e:
        logger.error(f"Error reading parquet files: {e}")

    logger.info("=" * 60)
    return results
