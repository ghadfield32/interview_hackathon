# path: api/src/airflow_project/utils/data_analysis_validator.py
"""
NBA Data Analysis & Validation Tool
Comprehensive analysis of pipeline outputs to verify correctness and completeness
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Tuple, Any
import logging

logger = logging.getLogger(__name__)

class NBADataAnalyzer:
    """Comprehensive NBA data analysis and validation"""

    def __init__(self, lineups_path: str = None, players_path: str = None):
        self.lineups_path = Path(lineups_path) if lineups_path else None
        self.players_path = Path(players_path) if players_path else None
        self.lineups_df = None
        self.players_df = None

    def load_data(self) -> Dict[str, Any]:
        """Load and validate parquet files"""
        results = {"load_status": {}, "data_info": {}}

        try:
            if self.lineups_path and self.lineups_path.exists():
                self.lineups_df = pd.read_parquet(self.lineups_path)
                results["load_status"]["lineups"] = "SUCCESS"
                results["data_info"]["lineups_shape"] = self.lineups_df.shape
                results["data_info"]["lineups_columns"] = list(self.lineups_df.columns)
            else:
                results["load_status"]["lineups"] = "MISSING_FILE"

            if self.players_path and self.players_path.exists():
                self.players_df = pd.read_parquet(self.players_path)
                results["load_status"]["players"] = "SUCCESS"
                results["data_info"]["players_shape"] = self.players_df.shape
                results["data_info"]["players_columns"] = list(self.players_df.columns)
            else:
                results["load_status"]["players"] = "MISSING_FILE"

            return results

        except Exception as e:
            results["load_status"]["error"] = str(e)
            return results

    def analyze_lineup_data(self) -> Dict[str, Any]:
        """Comprehensive analysis of lineup data"""
        if self.lineups_df is None:
            return {"error": "Lineups data not loaded"}

        df = self.lineups_df
        analysis = {
            "basic_stats": {
                "total_lineups": len(df),
                "teams": list(df['Team'].unique()) if 'Team' in df.columns else [],
                "team_count": df['Team'].nunique() if 'Team' in df.columns else 0,
                "lineups_per_team": df['Team'].value_counts().to_dict() if 'Team' in df.columns else {}
            }
        }

        # Analyze possessions data
        if 'Offensive possessions played' in df.columns:
            off_poss = df['Offensive possessions played'].dropna()
            analysis["possessions"] = {
                "offensive": {
                    "total": off_poss.sum(),
                    "average_per_lineup": off_poss.mean(),
                    "min": off_poss.min(),
                    "max": off_poss.max(),
                    "distribution": off_poss.describe().to_dict()
                }
            }

        if 'Defensive possessions played' in df.columns:
            def_poss = df['Defensive possessions played'].dropna()
            analysis["possessions"]["defensive"] = {
                "total": def_poss.sum(),
                "average_per_lineup": def_poss.mean(),
                "min": def_poss.min(),
                    "max": def_poss.max(),
                    "distribution": def_poss.describe().to_dict()
                }

        # Analyze ratings
        rating_cols = ['Offensive rating', 'Defensive rating', 'Net rating']
        analysis["ratings"] = {}

        for col in rating_cols:
            if col in df.columns:
                ratings = df[col].dropna()
                analysis["ratings"][col.lower().replace(' ', '_')] = {
                    "count_with_data": len(ratings),
                    "percentage_complete": len(ratings) / len(df) * 100,
                    "average": ratings.mean() if len(ratings) > 0 else None,
                    "min": ratings.min() if len(ratings) > 0 else None,
                    "max": ratings.max() if len(ratings) > 0 else None,
                    "std": ratings.std() if len(ratings) > 0 else None
                }

        # Unique player analysis
        player_cols = ['Player 1', 'Player 2', 'Player 3', 'Player 4', 'Player 5']
        all_lineup_players = []

        for col in player_cols:
            if col in df.columns:
                all_lineup_players.extend(df[col].dropna().tolist())
        lineup_players = set(all_lineup_players)

        analysis["players"] = {
            "total_player_instances": len(all_lineup_players),
            "unique_players": len(lineup_players),
            "unique_player_list": sorted(list(lineup_players)),
            "most_frequent_players": pd.Series(all_lineup_players).value_counts().head(10).to_dict()
        }

        # Data quality checks
        analysis["data_quality"] = {
            "null_counts": df.isnull().sum().to_dict(),
            "null_percentages": (df.isnull().sum() / len(df) * 100).to_dict(),
            "duplicate_lineups": df.duplicated(subset=player_cols if all(col in df.columns for col in player_cols) else None).sum()
        }

        return analysis

    def analyze_player_data(self) -> Dict[str, Any]:
        """Comprehensive analysis of player data"""
        if self.players_df is None:
            return {"error": "Players data not loaded"}

        df = self.players_df
        analysis = {
            "basic_stats": {
                "total_players": len(df),
                "teams": list(df['Team'].unique()) if 'Team' in df.columns else [],
                "team_count": df['Team'].nunique() if 'Team' in df.columns else 0,
                "players_per_team": df['Team'].value_counts().to_dict() if 'Team' in df.columns else {}
            }
        }

        # Analyze possessions
        poss_cols = ['Offensive possessions played', 'Defensive possessions played']
        analysis["possessions"] = {}

        for col in poss_cols:
            if col in df.columns:
                poss = df[col].dropna()
                col_key = col.lower().replace(' ', '_').replace('possessions_played', 'possessions')
                analysis["possessions"][col_key] = {
                    "total": poss.sum(),
                    "average_per_player": poss.mean(),
                    "min": poss.min(),
                    "max": poss.max(),
                    "players_with_data": len(poss)
                }

        # Analyze rim defense
        rim_cols = [
            'Opponent rim field goal percentage when player is on the court',
            'Opponent rim field goal percentage when player is off the court',
            'Opponent rim field goal percentage on/off difference (on-off)'
        ]

        analysis["rim_defense"] = {}

        for col in rim_cols:
            if col in df.columns:
                rim_data = df[col].dropna()
                col_key = 'on_court' if 'on the court' in col else 'off_court' if 'off the court' in col else 'difference'
                analysis["rim_defense"][col_key] = {
                    "players_with_data": len(rim_data),
                    "percentage_complete": len(rim_data) / len(df) * 100,
                    "average": rim_data.mean() if len(rim_data) > 0 else None,
                    "min": rim_data.min() if len(rim_data) > 0 else None,
                    "max": rim_data.max() if len(rim_data) > 0 else None,
                    "std": rim_data.std() if len(rim_data) > 0 else None
                }

        # Players with no rim defense data
        if 'Opponent rim field goal percentage when player is on the court' in df.columns:
            players_no_rim = df[df['Opponent rim field goal percentage when player is on the court'].isna()]
            analysis["rim_defense"]["players_without_rim_data"] = {
                "count": len(players_no_rim),
                "percentage": len(players_no_rim) / len(df) * 100,
                "player_list": players_no_rim['Player Name'].tolist() if 'Player Name' in players_no_rim.columns else []
            }

        # Best and worst rim defenders
        if 'Opponent rim field goal percentage on/off difference (on-off)' in df.columns:
            rim_diff = df['Opponent rim field goal percentage on/off difference (on-off)'].dropna()
            if len(rim_diff) > 0:
                best_defenders = df.loc[rim_diff.nsmallest(5).index][['Player Name', 'Team', 'Opponent rim field goal percentage on/off difference (on-off)']]
                worst_defenders = df.loc[rim_diff.nlargest(5).index][['Player Name', 'Team', 'Opponent rim field goal percentage on/off difference (on-off)']]

                analysis["rim_defense"]["best_defenders"] = best_defenders.to_dict('records')
                analysis["rim_defense"]["worst_defenders"] = worst_defenders.to_dict('records')

        # Data quality
        analysis["data_quality"] = {
            "null_counts": df.isnull().sum().to_dict(),
            "null_percentages": (df.isnull().sum() / len(df) * 100).to_dict(),
            "duplicate_players": df.duplicated(subset=['Player ID'] if 'Player ID' in df.columns else None).sum()
        }

        return analysis

    def cross_validate_data(self) -> Dict[str, Any]:
        """Cross-validate lineup and player data for consistency"""
        if self.lineups_df is None or self.players_df is None:
            return {"error": "Both datasets required for cross-validation"}

        validation = {}

        # Team consistency
        lineup_teams = set(self.lineups_df['Team'].unique()) if 'Team' in self.lineups_df.columns else set()
        player_teams = set(self.players_df['Team'].unique()) if 'Team' in self.players_df.columns else set()

        validation["team_consistency"] = {
            "lineup_teams": list(lineup_teams),
            "player_teams": list(player_teams),
            "teams_match": lineup_teams == player_teams,
            "teams_in_lineups_only": list(lineup_teams - player_teams),
            "teams_in_players_only": list(player_teams - lineup_teams)
        }

        # Player consistency - players in lineups should be in player data
        player_cols = ['Player 1', 'Player 2', 'Player 3', 'Player 4', 'Player 5']
        all_lineup_players = []

        if all(col in self.lineups_df.columns for col in player_cols):
            for col in player_cols:
                all_lineup_players.extend(self.lineups_df[col].dropna().tolist())
        lineup_players = set(all_lineup_players)

        data_players = set(self.players_df['Player Name'].dropna().tolist()) if 'Player Name' in self.players_df.columns else set()

        validation["player_consistency"] = {
            "players_in_lineups": len(lineup_players),
            "players_in_data": len(data_players),
            "players_in_lineups_list": sorted(list(lineup_players)),
            "players_missing_from_data": list(lineup_players - data_players),
            "players_not_in_lineups": list(data_players - lineup_players)
        }

        return validation

    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive analysis report"""
        logger.info("Generating comprehensive NBA data analysis report...")

        report = {
            "analysis_timestamp": pd.Timestamp.now().isoformat(),
            "data_loading": self.load_data(),
            "lineup_analysis": self.analyze_lineup_data(),
            "player_analysis": self.analyze_player_data(),
            "cross_validation": self.cross_validate_data()
        }

        # Summary insights
        report["summary_insights"] = self._generate_insights(report)

        return report

    def _generate_insights(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """Generate actionable insights from analysis"""
        insights = {
            "data_quality_score": 0,
            "key_findings": [],
            "recommendations": [],
            "concerns": []
        }

        # Calculate overall data quality score
        quality_factors = []

        # Lineup data quality
        if "lineup_analysis" in report and "data_quality" in report["lineup_analysis"]:
            lineup_null_pct = np.mean(list(report["lineup_analysis"]["data_quality"]["null_percentages"].values()))
            quality_factors.append(100 - lineup_null_pct)

        # Player data quality
        if "player_analysis" in report and "data_quality" in report["player_analysis"]:
            player_null_pct = np.mean(list(report["player_analysis"]["data_quality"]["null_percentages"].values()))
            quality_factors.append(100 - player_null_pct)

        # Cross-validation
        if "cross_validation" in report:
            cv = report["cross_validation"]
            if cv.get("team_consistency", {}).get("teams_match", False):
                quality_factors.append(100)
            else:
                quality_factors.append(50)

        insights["data_quality_score"] = np.mean(quality_factors) if quality_factors else 0

        # Generate findings
        if "lineup_analysis" in report:
            la = report["lineup_analysis"]
            if la.get("basic_stats", {}).get("team_count", 0) == 2:
                insights["key_findings"].append("✅ Confirmed 2-team game data as expected")

            if la.get("basic_stats", {}).get("total_lineups", 0) > 0:
                insights["key_findings"].append(f"📊 Generated {la['basic_stats']['total_lineups']} unique lineup combinations")

            if "players" in la:
                insights["key_findings"].append(f"👥 {la['players']['unique_players']} unique players across all lineups")

        if "player_analysis" in report:
            pa = report["player_analysis"]
            if "rim_defense" in pa and "players_without_rim_data" in pa["rim_defense"]:
                no_rim_pct = pa["rim_defense"]["players_without_rim_data"]["percentage"]
                if no_rim_pct > 50:
                    insights["concerns"].append(f"⚠️ {no_rim_pct:.1f}% of players missing rim defense data")
                else:
                    insights["key_findings"].append(f"✅ {100-no_rim_pct:.1f}% of players have rim defense data")

        # Recommendations based on findings
        if insights["data_quality_score"] < 80:
            insights["recommendations"].append("🔧 Improve data quality by addressing null values and validation issues")

        if "cross_validation" in report and report["cross_validation"].get("player_consistency", {}).get("players_missing_from_data"):
            insights["recommendations"].append("🔧 Investigate players appearing in lineups but missing from player data")

        return insights

    def save_report(self, report: Dict[str, Any], output_path: str = None) -> Path:
        """Save comprehensive report to JSON"""
        if output_path is None:
            output_path = "nba_data_analysis_report.json"

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Analysis report saved to: {output_file}")
        return output_file


def analyze_nba_data(lineups_path: str = None, players_path: str = None, 
                     save_report: bool = True, output_dir: str = None) -> Dict[str, Any]:
    """Main function to analyze NBA pipeline output data"""

    # Default paths if not provided
    if lineups_path is None:
        lineups_path = "lineups.parquet" 
    if players_path is None:
        players_path = "players.parquet"

    analyzer = NBADataAnalyzer(lineups_path, players_path)
    report = analyzer.generate_comprehensive_report()

    if save_report:
        report_path = Path(output_dir or ".") / "nba_comprehensive_analysis.json"
        analyzer.save_report(report, report_path)

    return report


def print_analysis_summary(report: Dict[str, Any]) -> None:
    """Print a formatted summary of the analysis"""
    print("=" * 60)
    print("NBA DATA PIPELINE ANALYSIS SUMMARY")
    print("=" * 60)

    # Data loading status
    if "data_loading" in report:
        dl = report["data_loading"]
        print(f"📁 Data Loading Status:")
        print(f"   Lineups: {dl['load_status'].get('lineups', 'UNKNOWN')}")
        print(f"   Players: {dl['load_status'].get('players', 'UNKNOWN')}")

    # Basic stats
    if "lineup_analysis" in report and "basic_stats" in report["lineup_analysis"]:
        la = report["lineup_analysis"]["basic_stats"]
        print(f"\n📊 Lineup Data:")
        print(f"   Total Lineups: {la.get('total_lineups', 0)}")
        print(f"   Teams: {', '.join(la.get('teams', []))}")
        print(f"   Lineups per Team: {la.get('lineups_per_team', {})}")
    elif "lineup_analysis" in report and "error" in report["lineup_analysis"]:
        print(f"\n📊 Lineup Data: {report['lineup_analysis']['error']}")

    if "player_analysis" in report and "basic_stats" in report["player_analysis"]:
        pa = report["player_analysis"]["basic_stats"]
        print(f"\n👥 Player Data:")
        print(f"   Total Players: {pa.get('total_players', 0)}")
        print(f"   Teams: {', '.join(pa.get('teams', []))}")
        print(f"   Players per Team: {pa.get('players_per_team', {})}")
    elif "player_analysis" in report and "error" in report["player_analysis"]:
        print(f"\n👥 Player Data: {report['player_analysis']['error']}")

    # Summary insights
    if "summary_insights" in report:
        si = report["summary_insights"]
        print(f"\n🎯 Analysis Summary:")
        print(f"   Data Quality Score: {si.get('data_quality_score', 0):.1f}/100")

        if si.get("key_findings"):
            print(f"\n🔍 Key Findings:")
            for finding in si["key_findings"]:
                print(f"   {finding}")

        if si.get("concerns"):
            print(f"\n⚠️  Concerns:")
            for concern in si["concerns"]:
                print(f"   {concern}")

        if si.get("recommendations"):
            print(f"\n💡 Recommendations:")
            for rec in si["recommendations"]:
                print(f"   {rec}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    import sys

    # Handle command line arguments
    if len(sys.argv) >= 3:
        lineups_path = sys.argv[1]
        players_path = sys.argv[2]
    else:
        # Default paths
        lineups_path = "lineups.parquet"
        players_path = "players.parquet"

    print(f"Analyzing NBA data with:")
    print(f"  Lineups: {lineups_path}")
    print(f"  Players: {players_path}")
    print()

    # Example usage
    report = analyze_nba_data(
        lineups_path=lineups_path,
        players_path=players_path,
        save_report=True
    )

    print_analysis_summary(report)
