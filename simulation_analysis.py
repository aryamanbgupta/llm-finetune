#!/usr/bin/env python3
"""
Cricket Simulation Analysis
Analyzes results from parallel simulations
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
from typing import Dict, List
import pandas as pd

def load_simulation_results(results_dir: Path) -> Dict:
    """Load all results from simulation directory"""
    all_results = []
    
    # Find all checkpoint directories
    checkpoints = sorted([d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint_")])
    
    if not checkpoints:
        raise ValueError(f"No checkpoint directories found in {results_dir}")
    
    # Load from latest checkpoint
    latest_checkpoint = checkpoints[-1]
    print(f"Loading from {latest_checkpoint}")
    
    # Load statistics
    with open(latest_checkpoint / "statistics.json", "r") as f:
        stats = json.load(f)
    
    # Load sample results for detailed analysis
    with open(latest_checkpoint / "results.json", "r") as f:
        sample_results = json.load(f)
    
    return {"stats": stats, "sample_results": sample_results}

def analyze_match_results(results: List[Dict]) -> Dict:
    """Detailed analysis of match results"""
    analysis = {
        "score_distribution": defaultdict(list),
        "wicket_distribution": defaultdict(list),
        "run_rates": defaultdict(list),
        "batting_performances": defaultdict(list),
        "bowling_performances": defaultdict(list),
        "margins": {"runs": [], "wickets": []},
        "high_scores": {"team": [], "individual": []},
        "best_bowling": []
    }
    
    for match in results:
        # Score distributions
        for team, innings in match["innings_scores"].items():
            analysis["score_distribution"][team].append(innings["runs"])
            analysis["wicket_distribution"][team].append(innings["wickets"])
            
            # Run rate
            if innings["overs"] > 0:
                balls = int(innings["overs"]) * 6 + int((innings["overs"] * 10) % 10)
                run_rate = innings["runs"] / (balls / 6)
                analysis["run_rates"][team].append(run_rate)
        
        # Individual performances
        for team, batsmen in match["batting_stats"].items():
            for player_name, stats in batsmen.items():
                if stats["balls"] > 0:
                    analysis["batting_performances"][team].append({
                        "player": player_name,
                        "runs": stats["runs"],
                        "balls": stats["balls"],
                        "sr": stats["strike_rate"],
                        "fours": stats["fours"],
                        "sixes": stats["sixes"]
                    })
                    
                    # Track high scores
                    if stats["runs"] >= 50:
                        analysis["high_scores"]["individual"].append({
                            "player": player_name,
                            "team": team,
                            "runs": stats["runs"],
                            "balls": stats["balls"]
                        })
        
        # Bowling performances
        for team, bowlers in match["bowling_stats"].items():
            for player_name, stats in bowlers.items():
                if stats["balls"] > 0:
                    analysis["bowling_performances"][team].append({
                        "player": player_name,
                        "wickets": stats["wickets"],
                        "runs": stats["runs"],
                        "overs": stats["overs"],
                        "economy": stats["economy"]
                    })
                    
                    # Track best bowling
                    if stats["wickets"] >= 3:
                        analysis["best_bowling"].append({
                            "player": player_name,
                            "team": team,
                            "figures": f"{stats['wickets']}/{stats['runs']}"
                        })
        
        # Match margins
        if "runs" in match["margin"]:
            margin_value = int(match["margin"].split(" ")[1])
            analysis["margins"]["runs"].append(margin_value)
        else:
            margin_value = int(match["margin"].split(" ")[1])
            analysis["margins"]["wickets"].append(margin_value)
    
    return analysis

def create_visualizations(analysis: Dict, stats: Dict, output_dir: Path):
    """Create comprehensive visualizations"""
    output_dir.mkdir(exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # 1. Score Distribution Histogram
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    for team, scores in analysis["score_distribution"].items():
        ax1.hist(scores, bins=20, alpha=0.6, label=f"{team} (μ={np.mean(scores):.1f})")
    
    ax1.set_xlabel("Team Score")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Team Score Distribution")
    ax1.legend()
    ax1.axvline(x=160, color='red', linestyle='--', alpha=0.5, label='Typical T20 Score')
    
    # 2. Wickets Distribution
    for team, wickets in analysis["wicket_distribution"].items():
        wicket_counts = np.bincount(wickets, minlength=11)[:11]
        ax2.bar(range(11), wicket_counts, alpha=0.6, label=team)
    
    ax2.set_xlabel("Wickets Lost")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Wickets Lost Distribution")
    ax2.legend()
    ax2.set_xticks(range(11))
    
    plt.tight_layout()
    plt.savefig(output_dir / "score_wicket_distribution.png", dpi=300)
    plt.close()
    
    # 3. Run Rate Analysis
    fig, ax = plt.subplots(figsize=(10, 6))
    
    run_rate_data = []
    for team, rates in analysis["run_rates"].items():
        run_rate_data.extend([(team, rate) for rate in rates])
    
    df_rr = pd.DataFrame(run_rate_data, columns=["Team", "Run Rate"])
    df_rr.boxplot(by="Team", column="Run Rate", ax=ax)
    ax.set_ylabel("Run Rate")
    ax.set_title("Run Rate Distribution by Team")
    plt.suptitle("")  # Remove default title
    
    plt.tight_layout()
    plt.savefig(output_dir / "run_rate_distribution.png", dpi=300)
    plt.close()
    
    # 4. Win Probability Over Time (if we had ball-by-ball data)
    # This would require storing more detailed match progression data
    
    # 5. Individual Performance Scatter
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Batting performances
    all_batting = []
    for team, perfs in analysis["batting_performances"].items():
        all_batting.extend(perfs)
    
    if all_batting:
        df_bat = pd.DataFrame(all_batting)
        df_bat = df_bat[df_bat["balls"] >= 10]  # Filter significant innings
        
        ax1.scatter(df_bat["balls"], df_bat["runs"], alpha=0.5, s=30)
        ax1.set_xlabel("Balls Faced")
        ax1.set_ylabel("Runs Scored")
        ax1.set_title("Individual Batting Performances")
        
        # Add strike rate lines
        for sr in [100, 150, 200]:
            balls = np.linspace(0, 60, 100)
            runs = balls * sr / 100
            ax1.plot(balls, runs, '--', alpha=0.3, label=f"SR={sr}")
        ax1.legend()
    
    # Bowling performances
    all_bowling = []
    for team, perfs in analysis["bowling_performances"].items():
        all_bowling.extend(perfs)
    
    if all_bowling:
        df_bowl = pd.DataFrame(all_bowling)
        
        ax2.scatter(df_bowl["overs"], df_bowl["wickets"], alpha=0.5, s=30)
        ax2.set_xlabel("Overs Bowled")
        ax2.set_ylabel("Wickets Taken")
        ax2.set_title("Individual Bowling Performances")
    
    plt.tight_layout()
    plt.savefig(output_dir / "individual_performances.png", dpi=300)
    plt.close()
    
    # 6. Match Margins
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    if analysis["margins"]["runs"]:
        ax1.hist(analysis["margins"]["runs"], bins=20, color='blue', alpha=0.7)
        ax1.set_xlabel("Victory Margin (runs)")
        ax1.set_ylabel("Frequency")
        ax1.set_title("Victory Margins - Batting First Wins")
        ax1.axvline(x=np.median(analysis["margins"]["runs"]), color='red', 
                   linestyle='--', label=f'Median: {np.median(analysis["margins"]["runs"]):.0f}')
        ax1.legend()
    
    if analysis["margins"]["wickets"]:
        ax2.hist(analysis["margins"]["wickets"], bins=10, color='green', alpha=0.7)
        ax2.set_xlabel("Victory Margin (wickets)")
        ax2.set_ylabel("Frequency")
        ax2.set_title("Victory Margins - Chasing Wins")
        ax2.set_xticks(range(1, 11))
    
    plt.tight_layout()
    plt.savefig(output_dir / "victory_margins.png", dpi=300)
    plt.close()

def generate_report(analysis: Dict, stats: Dict, output_path: Path):
    """Generate comprehensive text report"""
    report = []
    
    report.append("CRICKET SIMULATION ANALYSIS REPORT")
    report.append("=" * 50)
    report.append(f"\nTotal Matches Simulated: {stats['total_matches']}\n")
    
    # Win/Loss Summary
    report.append("WIN/LOSS SUMMARY")
    report.append("-" * 30)
    for team, wins in stats["winners"].items():
        win_pct = (wins / stats['total_matches']) * 100
        report.append(f"{team}: {wins} wins ({win_pct:.1f}%)")
    
    # Score Statistics
    report.append("\n\nSCORE STATISTICS")
    report.append("-" * 30)
    for team in stats["avg_scores"]:
        avg = stats["avg_scores"][team]
        std = stats["score_std"][team]
        report.append(f"{team}: {avg:.1f} ± {std:.1f}")
        
        if team in analysis["score_distribution"]:
            scores = analysis["score_distribution"][team]
            report.append(f"  Range: {min(scores)} - {max(scores)}")
            report.append(f"  Median: {np.median(scores):.0f}")
    
    # Notable Performances
    report.append("\n\nNOTABLE PERFORMANCES")
    report.append("-" * 30)
    
    # High scores
    if analysis["high_scores"]["individual"]:
        report.append("\nTop Individual Scores (50+):")
        high_scores = sorted(analysis["high_scores"]["individual"], 
                           key=lambda x: x["runs"], reverse=True)[:10]
        for perf in high_scores:
            report.append(f"  {perf['player']} ({perf['team']}): {perf['runs']} ({perf['balls']}b)")
    
    # Best bowling
    if analysis["best_bowling"]:
        report.append("\nBest Bowling Figures (3+ wickets):")
        best_bowling = sorted(analysis["best_bowling"], 
                            key=lambda x: int(x["figures"].split("/")[0]), reverse=True)[:10]
        for perf in best_bowling:
            report.append(f"  {perf['player']} ({perf['team']}): {perf['figures']}")
    
    # Statistical Validation
    report.append("\n\nSTATISTICAL VALIDATION")
    report.append("-" * 30)
    
    # Check if scores are realistic
    avg_score = np.mean([stats["avg_scores"][team] for team in stats["avg_scores"]])
    report.append(f"Average match score: {avg_score:.1f}")
    
    if 140 <= avg_score <= 180:
        report.append("✓ Scores are within typical T20 range (140-180)")
    else:
        report.append("⚠ Scores are outside typical T20 range (140-180)")
    
    # Check run rates
    all_rr = []
    for rates in analysis["run_rates"].values():
        all_rr.extend(rates)
    avg_rr = np.mean(all_rr)
    report.append(f"\nAverage run rate: {avg_rr:.2f}")
    
    if 7.5 <= avg_rr <= 9.0:
        report.append("✓ Run rate is within typical T20 range (7.5-9.0)")
    else:
        report.append("⚠ Run rate is outside typical T20 range (7.5-9.0)")
    
    # Save report
    with open(output_path / "analysis_report.txt", "w") as f:
        f.write("\n".join(report))
    
    print("\n".join(report))

def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", required=True, help="Directory containing simulation results")
    parser.add_argument("--output_dir", default="analysis_output", help="Directory for analysis output")
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f"Loading results from {results_dir}")
    data = load_simulation_results(results_dir)
    
    print("Analyzing match results...")
    analysis = analyze_match_results(data["sample_results"])
    
    print("Creating visualizations...")
    create_visualizations(analysis, data["stats"], output_dir)
    
    print("Generating report...")
    generate_report(analysis, data["stats"], output_dir)
    
    print(f"\nAnalysis complete! Results saved to {output_dir}")

if __name__ == "__main__":
    main()