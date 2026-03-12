#!/usr/bin/env python3
"""Backtest: individual team score projection accuracy.

For each historical game, projects each team's final score independently
at halftime (end of Q2) and Q4 start (end of Q3). Each game produces TWO
predictions (one per team).

Model: team_final = team_score_so_far + team_remaining_avg (from rolling profile)
Variance: sum of team's remaining quarter stds (squared), inflated by sigma factor.

Usage:
    PYTHONPATH=src python -m nba.backtest_team_score
    PYTHONPATH=src python -m nba.backtest_team_score --q4
    PYTHONPATH=src python -m nba.backtest_team_score --q4 --high-conf --confidence 0.90
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _src) if _src not in sys.path else None
from paths import project_path

from nba.backtest_total_score import build_rolling_profiles

CSV_PATH = project_path("data", "nba", "team_scores.csv")
CHART_DIR = project_path("charts", "nba")


# ---------------------------------------------------------------------------
# Halftime backtest (per-team score)
# ---------------------------------------------------------------------------

def run_backtest_halftime(
    last_n: int = 20,
    min_games: int = 10,
    sigma_inflation: float = 1.45,
) -> pd.DataFrame:
    """Backtest individual team score projection at halftime.

    For each team in each game:
      - team_h1 = actual Q1 + Q2
      - projected_final = team_h1 + profile Q3 avg + profile Q4 avg
      - calibrated_std = sqrt(q3_std^2 + q4_std^2) * sigma_inflation

    Returns DataFrame with one row per team per game.
    """
    df = pd.read_csv(CSV_PATH)
    df = df[(df["q1"] > 0) & (df["q2"] > 0) & (df["q3"] > 0) & (df["q4"] > 0)]

    print(f"Building rolling profiles (last_n={last_n}, min_games={min_games})...")
    profiles = build_rolling_profiles(df, last_n=last_n, min_games=min_games)
    print(f"  Built {len(profiles)} team-game profiles")

    home = df[df["home_away"] == "home"].set_index("game_id")
    away = df[df["home_away"] == "away"].set_index("game_id")
    common = home.index.intersection(away.index)
    home = home.loc[common]
    away = away.loc[common]

    results = []
    for game_id in common:
        h = home.loc[game_id]
        a = away.loc[game_id]

        for row, opp_row, ha_label in [(h, a, "home"), (a, h, "away")]:
            team = row["team"]
            prof = profiles.get((game_id, team))
            if prof is None:
                continue

            h1 = row["q1"] + row["q2"]
            actual = row["total"]

            expected_h2 = prof["q3_avg"] + prof["q4_avg"]
            projected = h1 + expected_h2

            remaining_var = prof["q3_std"] ** 2 + prof["q4_std"] ** 2
            raw_std = math.sqrt(remaining_var)
            calibrated_std = raw_std * sigma_inflation

            error = projected - actual

            results.append({
                "game_id": game_id,
                "game_date": row["game_date"],
                "team": team,
                "opponent": opp_row["team"],
                "home_away": ha_label,
                "h1_score": h1,
                "projected": round(projected, 1),
                "actual": actual,
                "error": round(error, 1),
                "abs_error": round(abs(error), 1),
                "calibrated_std": round(calibrated_std, 1),
                "games_used": prof["games_used"],
            })

    results_df = pd.DataFrame(results)
    print(f"  Backtested {len(results_df)} team-games ({results_df['game_id'].nunique()} games)")
    return results_df


# ---------------------------------------------------------------------------
# Q4 start backtest (per-team score)
# ---------------------------------------------------------------------------

def run_backtest_q4(
    last_n: int = 20,
    min_games: int = 10,
    sigma_inflation: float = 1.45,
) -> pd.DataFrame:
    """Backtest individual team score projection at Q4 start.

    For each team in each game:
      - team_q3 = actual Q1 + Q2 + Q3
      - projected_final = team_q3 + profile Q4 avg
      - calibrated_std = q4_std * sigma_inflation
    """
    df = pd.read_csv(CSV_PATH)
    df = df[(df["q1"] > 0) & (df["q2"] > 0) & (df["q3"] > 0) & (df["q4"] > 0)]

    print(f"Building rolling profiles (last_n={last_n}, min_games={min_games})...")
    profiles = build_rolling_profiles(df, last_n=last_n, min_games=min_games)
    print(f"  Built {len(profiles)} team-game profiles")

    home = df[df["home_away"] == "home"].set_index("game_id")
    away = df[df["home_away"] == "away"].set_index("game_id")
    common = home.index.intersection(away.index)
    home = home.loc[common]
    away = away.loc[common]

    results = []
    for game_id in common:
        h = home.loc[game_id]
        a = away.loc[game_id]

        for row, opp_row, ha_label in [(h, a, "home"), (a, h, "away")]:
            team = row["team"]
            prof = profiles.get((game_id, team))
            if prof is None:
                continue

            q3_score = row["q1"] + row["q2"] + row["q3"]
            actual = row["total"]
            actual_q4 = row["q4"]

            projected = q3_score + prof["q4_avg"]

            raw_std = prof["q4_std"]
            calibrated_std = raw_std * sigma_inflation

            error = projected - actual

            results.append({
                "game_id": game_id,
                "game_date": row["game_date"],
                "team": team,
                "opponent": opp_row["team"],
                "home_away": ha_label,
                "q3_score": q3_score,
                "actual_q4": actual_q4,
                "projected": round(projected, 1),
                "actual": actual,
                "error": round(error, 1),
                "abs_error": round(abs(error), 1),
                "calibrated_std": round(calibrated_std, 1),
                "games_used": prof["games_used"],
            })

    results_df = pd.DataFrame(results)
    print(f"  Backtested {len(results_df)} team-games ({results_df['game_id'].nunique()} games)")
    return results_df


# ---------------------------------------------------------------------------
# Print results
# ---------------------------------------------------------------------------

def print_results(results: pd.DataFrame, checkpoint: str = "Halftime") -> None:
    if results.empty:
        print("No results.")
        return

    n = len(results)
    mae = results["abs_error"].mean()
    bias = results["error"].mean()
    rmse = math.sqrt((results["error"] ** 2).mean())
    median_ae = results["abs_error"].median()

    print(f"\n{'=' * 65}")
    print(f"TEAM SCORE BACKTEST — {checkpoint.upper()}")
    print(f"{'=' * 65}")
    print(f"  Predictions:  {n} ({results['game_id'].nunique()} games × 2 teams)")
    print(f"  Date range:   {results['game_date'].min()} to {results['game_date'].max()}")

    print(f"\n  --- Model Performance ---")
    print(f"  MAE:          {mae:.1f} pts")
    print(f"  Median AE:    {median_ae:.1f} pts")
    print(f"  RMSE:         {rmse:.1f} pts")
    print(f"  Bias:         {bias:+.1f} pts")

    # Naive baseline: 2x halftime (or Q3 + avg Q4)
    if "q3_score" in results.columns:
        avg_q4 = results["actual_q4"].mean()
        naive = results["q3_score"] + avg_q4
        naive_err = (naive - results["actual"]).abs().mean()
        naive_bias = (naive - results["actual"]).mean()
        print(f"\n  --- Naive Baseline (Q3 + avg Q4 = {avg_q4:.1f}) ---")
    else:
        naive = results["h1_score"] * 2
        naive_err = (naive - results["actual"]).abs().mean()
        naive_bias = (naive - results["actual"]).mean()
        print(f"\n  --- Naive Baseline (2× halftime) ---")
    print(f"  MAE:          {naive_err:.1f} pts")
    print(f"  Bias:         {naive_bias:+.1f} pts")
    print(f"  Improvement:  {naive_err - mae:+.1f} pts MAE")

    # Coverage
    print(f"\n  --- Coverage ---")
    for mult, label in [(1.0, "1.0σ (~68%)"), (1.5, "1.5σ (~87%)"), (2.0, "2.0σ (~95%)")]:
        lo = results["projected"] - mult * results["calibrated_std"]
        hi = results["projected"] + mult * results["calibrated_std"]
        within = ((results["actual"] >= lo) & (results["actual"] <= hi)).mean()
        print(f"  {label}:  {within * 100:.1f}%")

    # Home vs away
    print(f"\n  --- Home vs Away ---")
    for ha in ["home", "away"]:
        sub = results[results["home_away"] == ha]
        if len(sub) > 0:
            print(f"  {ha.upper():5s}: n={len(sub):4d}  MAE={sub['abs_error'].mean():.1f}  "
                  f"bias={sub['error'].mean():+.1f}")

    # Per-team breakdown (sorted by MAE)
    print(f"\n  --- Per-Team MAE (sorted) ---")
    team_stats = results.groupby("team").agg(
        n=("abs_error", "count"),
        mae=("abs_error", "mean"),
        bias=("error", "mean"),
    ).sort_values("mae")
    print(f"  {'Team':>5s} | {'n':>3s} | {'MAE':>5s} | {'Bias':>6s}")
    print(f"  {'-' * 30}")
    for team, s in team_stats.iterrows():
        print(f"  {team:>5s} | {s.n:3.0f} | {s.mae:5.1f} | {s.bias:+5.1f}")

    # Error percentiles
    print(f"\n  --- Error Percentiles ---")
    for p in [10, 25, 50, 75, 90]:
        val = np.percentile(results["abs_error"], p)
        print(f"  P{p:2d}:  {val:.1f} pts")


# ---------------------------------------------------------------------------
# High-confidence simulation
# ---------------------------------------------------------------------------

def simulate_high_confidence(
    results: pd.DataFrame,
    confidence: float = 0.90,
    no_plot: bool = False,
) -> None:
    """High-confidence OVER/UNDER lines for individual team scores.

    OVER line  = projected - z * calibrated_std  (team scores above this w/ conf%)
    UNDER line = projected + z * calibrated_std  (team scores below this w/ conf%)
    """
    from statistics import NormalDist

    if results.empty:
        print("No results for high-confidence simulation.")
        return

    z = NormalDist().inv_cdf(confidence)
    n = len(results)

    over_wins = 0
    under_wins = 0
    over_lines = []
    under_lines = []

    for _, row in results.iterrows():
        projected = row["projected"]
        std = row["calibrated_std"]
        actual = row["actual"]
        if std <= 0:
            continue

        over_line = projected - z * std
        under_line = projected + z * std
        over_lines.append(over_line)
        under_lines.append(under_line)

        if actual > over_line:
            over_wins += 1
        if actual < under_line:
            under_wins += 1

    total = len(over_lines)
    if total == 0:
        print("No valid predictions.")
        return

    over_pct = over_wins / total * 100
    under_pct = under_wins / total * 100
    avg_over = np.mean(over_lines)
    avg_under = np.mean(under_lines)
    avg_cushion = z * results["calibrated_std"].mean()

    print(f"\n{'=' * 65}")
    print(f"HIGH-CONFIDENCE TEAM SCORE SIM (z={z:.4f}, conf={confidence:.0%})")
    print(f"{'=' * 65}")
    print(f"  Predictions: {total}")
    print(f"  Avg cushion: {avg_cushion:.1f} pts")

    prices = [75, 78, 80, 82, 85, 88, 90, 92]

    for label, wins, pct, avg_line in [
        ("OVER", over_wins, over_pct, avg_over),
        ("UNDER", under_wins, under_pct, avg_under),
    ]:
        print(f"\n  --- {label} (win={wins}/{total} = {pct:.1f}%) avg line={avg_line:.1f} ---")
        print(f"  {'price':>6s} | {'EV/bet':>8s} | {'total P&L':>10s} | {'ROI':>7s}")
        print(f"  {'-' * 6}-+-{'-' * 8}-+-{'-' * 10}-+-{'-' * 7}")
        lose_pct = 100.0 - pct
        for price in prices:
            ev = (pct / 100) * (100 - price) - (lose_pct / 100) * price
            total_pnl = wins * (100 - price) - (total - wins) * price
            risked = total * price
            roi = total_pnl / risked * 100 if risked > 0 else 0
            print(f"  {price:5d}¢ | {ev:+7.1f}¢ | {total_pnl:+9d}¢ | {roi:+6.1f}%")

    # By home/away
    print(f"\n  By venue:")
    for ha in ["home", "away"]:
        sub = results[results["home_away"] == ha]
        if sub.empty:
            continue
        o_w = sum(1 for _, r in sub.iterrows() if r["actual"] > r["projected"] - z * r["calibrated_std"])
        u_w = sum(1 for _, r in sub.iterrows() if r["actual"] < r["projected"] + z * r["calibrated_std"])
        print(f"    {ha.upper():5s}: n={len(sub):4d}  over={o_w/len(sub)*100:.1f}%  under={u_w/len(sub)*100:.1f}%")

    # Confidence sweep
    print(f"\n  {'=' * 55}")
    print(f"  CONFIDENCE LEVEL SWEEP (at 85¢)")
    print(f"  {'=' * 55}")
    print(f"  {'conf':>5s} | {'dir':>5s} | {'win%':>6s} | {'EV/bet':>8s} | {'ROI':>7s}")
    print(f"  {'-' * 5}-+-{'-' * 5}-+-{'-' * 6}-+-{'-' * 8}-+-{'-' * 7}")
    ref_price = 85
    for conf_level in [0.85, 0.90, 0.95]:
        z_l = NormalDist().inv_cdf(conf_level)
        for direction in ["over", "under"]:
            w = 0
            t = 0
            for _, row in results.iterrows():
                std = row["calibrated_std"]
                if std <= 0:
                    continue
                t += 1
                if direction == "over" and row["actual"] > row["projected"] - z_l * std:
                    w += 1
                elif direction == "under" and row["actual"] < row["projected"] + z_l * std:
                    w += 1
            if t == 0:
                continue
            wp = w / t * 100
            ev = (wp / 100) * (100 - ref_price) - ((100 - wp) / 100) * ref_price
            roi = (w * (100 - ref_price) - (t - w) * ref_price) / (t * ref_price) * 100
            print(f"  {conf_level:4.0%}  | {direction:>5s} | {wp:5.1f}% | {ev:+7.1f}¢ | {roi:+6.1f}%")

    # Chart
    if not no_plot:
        os.makedirs(CHART_DIR, exist_ok=True)
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        # Error distribution
        ax = axes[0]
        ax.hist(results["error"], bins=np.arange(-30, 32, 2), color="#5C6BC0",
                alpha=0.7, edgecolor="white")
        ax.axvline(0, color="black", linestyle="--", alpha=0.5)
        ax.set_xlabel("Error (projected - actual)")
        ax.set_ylabel("Count")
        ax.set_title(f"Team Score Error (MAE={results['abs_error'].mean():.1f})")
        ax.grid(alpha=0.3)

        # Per-team MAE
        ax = axes[1]
        team_stats = results.groupby("team")["abs_error"].mean().sort_values()
        ax.barh(range(len(team_stats)), team_stats.values, color="#4CAF50", alpha=0.7)
        ax.set_yticks(range(len(team_stats)))
        ax.set_yticklabels(team_stats.index, fontsize=7)
        ax.set_xlabel("MAE (pts)")
        ax.set_title("Per-Team MAE")
        ax.axvline(results["abs_error"].mean(), color="red", linestyle="--",
                    label=f"avg={results['abs_error'].mean():.1f}")
        ax.legend(fontsize=8)
        ax.grid(axis="x", alpha=0.3)

        # Projected vs actual scatter
        ax = axes[2]
        ax.scatter(results["actual"], results["projected"], alpha=0.2, s=8, color="#FF9800")
        mn = min(results["actual"].min(), results["projected"].min()) - 5
        mx = max(results["actual"].max(), results["projected"].max()) + 5
        ax.plot([mn, mx], [mn, mx], "k--", alpha=0.5)
        ax.set_xlabel("Actual Score")
        ax.set_ylabel("Projected Score")
        ax.set_title("Projected vs Actual")
        ax.grid(alpha=0.3)

        checkpoint = "q4" if "q3_score" in results.columns else "halftime"
        fig.suptitle(f"Team Score Backtest — {checkpoint}", fontsize=13, fontweight="bold")
        fig.tight_layout()
        out = os.path.join(CHART_DIR, f"backtest_team_score_{checkpoint}.png")
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"\n  Saved {out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Team score backtest")
    parser.add_argument("--last-n", type=int, default=20,
                        help="Rolling window size (default: 20)")
    parser.add_argument("--min-games", type=int, default=10,
                        help="Min prior games before including (default: 10)")
    parser.add_argument("--sigma", type=float, default=1.45,
                        help="Sigma inflation factor (default: 1.45)")
    parser.add_argument("--q4", action="store_true",
                        help="Backtest at Q4 start instead of halftime")
    parser.add_argument("--high-conf", action="store_true",
                        help="Run high-confidence betting simulation")
    parser.add_argument("--confidence", type=float, default=0.90,
                        help="Confidence threshold (default: 0.90)")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip chart generation")
    args = parser.parse_args()

    if args.q4:
        results = run_backtest_q4(
            last_n=args.last_n, min_games=args.min_games,
            sigma_inflation=args.sigma,
        )
        print_results(results, checkpoint="Q4 Start")
    else:
        results = run_backtest_halftime(
            last_n=args.last_n, min_games=args.min_games,
            sigma_inflation=args.sigma,
        )
        print_results(results, checkpoint="Halftime")

    if args.high_conf:
        simulate_high_confidence(
            results, confidence=args.confidence, no_plot=args.no_plot,
        )
    elif not args.no_plot:
        # Generate charts in high_conf too, so only do standalone if not high_conf
        simulate_high_confidence(
            results, confidence=args.confidence, no_plot=True,
        )
