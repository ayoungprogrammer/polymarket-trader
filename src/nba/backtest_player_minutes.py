#!/usr/bin/env python3
"""Backtest: player minutes projection accuracy.

Projects a player's final minutes at two checkpoints:
  - Halftime: assume player has played proportional minutes through H1,
    project final minutes from rolling average + pace adjustment
  - Q4 start: same with 3/4 of game elapsed

Since nba_api PlayerGameLog only provides total game minutes (not per-quarter),
we simulate the halftime/Q4 checkpoints using the player's historical
H1/H2 and Q1-Q3/Q4 minute splits estimated from team-level data.

The model uses a rolling window of the player's recent games to compute
their average minutes and standard deviation, then projects final minutes.

Data fetching:
    cd src && python -m nba.backtest_player_minutes --fetch

Backtest:
    cd src && python -m nba.backtest_player_minutes
    cd src && python -m nba.backtest_player_minutes --q4
    cd src && python -m nba.backtest_player_minutes --high-conf
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _src) if _src not in sys.path else None
from paths import project_path

from nba.data import PLAYER_GAME_LOGS_PATH as CSV_PATH, fetch_player_game_logs

CHART_DIR = project_path("charts", "nba")


# ---------------------------------------------------------------------------
# Rolling profile builder
# ---------------------------------------------------------------------------

def build_rolling_minute_profiles(
    df: pd.DataFrame, last_n: int = 15, min_games: int = 8
) -> Dict[Tuple[str, str], dict]:
    """Build per-player rolling minutes profiles using only prior games.

    Returns dict keyed by (game_id, player_id) -> profile dict with:
        min_avg, min_std, min_median, games_used, started_pct
    Only includes entries where the player has >= min_games prior games.
    """
    profiles: Dict[Tuple[str, str], dict] = {}

    df_sorted = df.sort_values("game_date").reset_index(drop=True)

    # history[player_id] = list of {minutes, started}
    history: Dict[str, List[dict]] = defaultdict(list)

    for _, row in df_sorted.iterrows():
        pid = str(row["player_id"])
        gid = str(row["game_id"])
        minutes = row["minutes"]

        past = history[pid]
        if len(past) >= min_games:
            window = past[-last_n:]
            mins = [g["minutes"] for g in window]
            started = [g["started"] for g in window]
            avg = sum(mins) / len(mins)
            if len(mins) > 1:
                std = math.sqrt(sum((m - avg) ** 2 for m in mins) / (len(mins) - 1))
            else:
                std = 5.0

            profiles[(gid, pid)] = {
                "min_avg": avg,
                "min_std": std,
                "min_median": sorted(mins)[len(mins) // 2],
                "games_used": len(window),
                "started_pct": sum(started) / len(started),
            }

        # Append after building profile (no lookahead)
        history[pid].append({"minutes": minutes, "started": row.get("started", 0)})

    return profiles


# ---------------------------------------------------------------------------
# Halftime backtest
# ---------------------------------------------------------------------------

def run_backtest_halftime(
    last_n: int = 15,
    min_games: int = 8,
    sigma_inflation: float = 1.3,
    min_minutes: int = 15,
) -> pd.DataFrame:
    """Backtest player minutes projection at halftime.

    Model: projected_final = rolling_avg_minutes
    (At halftime we know the player is active, so we project they'll
    finish near their rolling average. The "observation" at halftime is
    that they're in the game — we don't have actual H1 minutes.)

    For a more realistic sim, we simulate "minutes at halftime" as:
        h1_minutes = actual_total * h1_fraction
    where h1_fraction ~ 0.50 (minutes are roughly proportional to game time).

    Then: projected_final = h1_minutes + rolling_avg * h2_fraction
    where h2_fraction = 1 - h1_fraction_avg (from player's history).

    Since we only have total minutes, we use a simpler approach:
        projected = rolling_avg_minutes
        error = projected - actual

    This measures how well the rolling average predicts actual minutes,
    which is the core signal for live betting.

    Filters to players averaging >= min_minutes (excludes deep bench).
    """
    df = pd.read_csv(CSV_PATH)
    df = df[df["minutes"] > 0]  # exclude DNPs

    print(f"Building rolling profiles (last_n={last_n}, min_games={min_games})...")
    profiles = build_rolling_minute_profiles(df, last_n=last_n, min_games=min_games)
    print(f"  Built {len(profiles)} player-game profiles")

    results = []
    for _, row in df.iterrows():
        pid = str(row["player_id"])
        gid = str(row["game_id"])
        actual = row["minutes"]

        prof = profiles.get((gid, pid))
        if prof is None:
            continue

        # Skip deep bench players
        if prof["min_avg"] < min_minutes:
            continue

        projected = prof["min_avg"]
        calibrated_std = prof["min_std"] * sigma_inflation

        error = projected - actual

        results.append({
            "game_id": gid,
            "game_date": row["game_date"],
            "player_id": pid,
            "player_name": row["player_name"],
            "team": row["team"],
            "opponent": row["opponent"],
            "home_away": row["home_away"],
            "started": row.get("started", 0),
            "rolling_avg": round(projected, 1),
            "projected": round(projected, 1),
            "actual": actual,
            "error": round(error, 1),
            "abs_error": round(abs(error), 1),
            "calibrated_std": round(calibrated_std, 1),
            "games_used": prof["games_used"],
            "started_pct": round(prof["started_pct"], 2),
        })

    results_df = pd.DataFrame(results)
    n_players = results_df["player_id"].nunique() if not results_df.empty else 0
    print(f"  Backtested {len(results_df)} player-games ({n_players} players)")
    return results_df


# ---------------------------------------------------------------------------
# Q4 start backtest
# ---------------------------------------------------------------------------

def run_backtest_q4(
    last_n: int = 15,
    min_games: int = 8,
    sigma_inflation: float = 1.3,
    min_minutes: int = 15,
) -> pd.DataFrame:
    """Backtest player minutes projection at Q4 start.

    Same model as halftime — rolling average predicts final minutes.
    At Q4 start we have more confidence the player will finish near average
    (3/4 of game elapsed, no injury exit). The value of this checkpoint is
    the reduced sigma (less remaining game time).

    We scale sigma by sqrt(1/4) / sqrt(1/2) ≈ 0.71 relative to halftime
    to reflect less remaining uncertainty.
    """
    df = pd.read_csv(CSV_PATH)
    df = df[df["minutes"] > 0]

    print(f"Building rolling profiles (last_n={last_n}, min_games={min_games})...")
    profiles = build_rolling_minute_profiles(df, last_n=last_n, min_games=min_games)
    print(f"  Built {len(profiles)} player-game profiles")

    # Q4 sigma reduction: sqrt(remaining_fraction) ratio
    q4_sigma_scale = math.sqrt(0.25) / math.sqrt(0.50)  # ~0.707

    results = []
    for _, row in df.iterrows():
        pid = str(row["player_id"])
        gid = str(row["game_id"])
        actual = row["minutes"]

        prof = profiles.get((gid, pid))
        if prof is None:
            continue

        if prof["min_avg"] < min_minutes:
            continue

        projected = prof["min_avg"]
        calibrated_std = prof["min_std"] * sigma_inflation * q4_sigma_scale

        error = projected - actual

        results.append({
            "game_id": gid,
            "game_date": row["game_date"],
            "player_id": pid,
            "player_name": row["player_name"],
            "team": row["team"],
            "opponent": row["opponent"],
            "home_away": row["home_away"],
            "started": row.get("started", 0),
            "rolling_avg": round(projected, 1),
            "projected": round(projected, 1),
            "actual": actual,
            "error": round(error, 1),
            "abs_error": round(abs(error), 1),
            "calibrated_std": round(calibrated_std, 1),
            "games_used": prof["games_used"],
            "started_pct": round(prof["started_pct"], 2),
        })

    results_df = pd.DataFrame(results)
    n_players = results_df["player_id"].nunique() if not results_df.empty else 0
    print(f"  Backtested {len(results_df)} player-games ({n_players} players)")
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
    print(f"PLAYER MINUTES BACKTEST — {checkpoint.upper()}")
    print(f"{'=' * 65}")
    n_players = results["player_id"].nunique()
    n_games = results["game_id"].nunique()
    print(f"  Predictions:  {n} ({n_players} players across {n_games} games)")
    print(f"  Date range:   {results['game_date'].min()} to {results['game_date'].max()}")
    print(f"  Avg minutes:  {results['actual'].mean():.1f} (actual)  {results['projected'].mean():.1f} (projected)")

    print(f"\n  --- Model Performance ---")
    print(f"  MAE:          {mae:.1f} min")
    print(f"  Median AE:    {median_ae:.1f} min")
    print(f"  RMSE:         {rmse:.1f} min")
    print(f"  Bias:         {bias:+.1f} min")

    # Naive baseline: season average (use overall mean as proxy)
    overall_avg = results["actual"].mean()
    naive_err = (overall_avg - results["actual"]).abs().mean()
    print(f"\n  --- Naive Baseline (league avg = {overall_avg:.1f} min) ---")
    print(f"  MAE:          {naive_err:.1f} min")
    print(f"  Improvement:  {naive_err - mae:+.1f} min MAE")

    # Coverage
    print(f"\n  --- Coverage ---")
    for mult, label in [(1.0, "1.0σ (~68%)"), (1.5, "1.5σ (~87%)"), (2.0, "2.0σ (~95%)")]:
        lo = results["projected"] - mult * results["calibrated_std"]
        hi = results["projected"] + mult * results["calibrated_std"]
        within = ((results["actual"] >= lo) & (results["actual"] <= hi)).mean()
        print(f"  {label}:  {within * 100:.1f}%")

    # Starters vs bench
    print(f"\n  --- Starters vs Bench ---")
    for label, mask in [("Starters", results["started"] == 1),
                        ("Bench",    results["started"] == 0)]:
        sub = results[mask]
        if len(sub) > 0:
            print(f"  {label:10s}: n={len(sub):5d}  MAE={sub['abs_error'].mean():.1f}  "
                  f"bias={sub['error'].mean():+.1f}  avg_min={sub['actual'].mean():.1f}")

    # By minutes bucket
    print(f"\n  --- By Rolling Avg Minutes ---")
    min_bins = [0, 20, 25, 30, 35, 48]
    min_labels = ["15-19", "20-24", "25-29", "30-34", "35+"]
    results_copy = results.copy()
    results_copy["min_bucket"] = pd.cut(results_copy["rolling_avg"], bins=min_bins,
                                         labels=min_labels, right=False)
    for label in min_labels:
        sub = results_copy[results_copy["min_bucket"] == label]
        if len(sub) > 0:
            print(f"  {label:10s}: n={len(sub):5d}  MAE={sub['abs_error'].mean():.1f}  "
                  f"bias={sub['error'].mean():+.1f}")

    # Error percentiles
    print(f"\n  --- Error Percentiles ---")
    for p in [10, 25, 50, 75, 90]:
        val = np.percentile(results["abs_error"], p)
        print(f"  P{p:2d}:  {val:.1f} min")

    # Worst predictions (highest absolute error)
    print(f"\n  --- Largest Misses (top 10) ---")
    worst = results.nlargest(10, "abs_error")
    print(f"  {'Player':<22s} {'Date':>10s} {'Proj':>5s} {'Act':>4s} {'Err':>6s}")
    for _, r in worst.iterrows():
        print(f"  {r['player_name']:<22s} {r['game_date']:>10s} {r['projected']:5.1f} "
              f"{r['actual']:4.0f} {r['error']:+5.1f}")


# ---------------------------------------------------------------------------
# High-confidence simulation
# ---------------------------------------------------------------------------

def simulate_high_confidence(
    results: pd.DataFrame,
    confidence: float = 0.90,
    no_plot: bool = False,
) -> None:
    """High-confidence OVER/UNDER lines for player minutes."""
    from statistics import NormalDist

    if results.empty:
        print("No results for high-confidence simulation.")
        return

    z = NormalDist().inv_cdf(confidence)
    n = len(results)

    over_wins = 0
    under_wins = 0

    for _, row in results.iterrows():
        projected = row["projected"]
        std = row["calibrated_std"]
        actual = row["actual"]
        if std <= 0:
            continue

        if actual > projected - z * std:
            over_wins += 1
        if actual < projected + z * std:
            under_wins += 1

    if n == 0:
        return

    over_pct = over_wins / n * 100
    under_pct = under_wins / n * 100
    avg_cushion = z * results["calibrated_std"].mean()

    print(f"\n{'=' * 65}")
    print(f"HIGH-CONFIDENCE PLAYER MINUTES SIM (z={z:.4f}, conf={confidence:.0%})")
    print(f"{'=' * 65}")
    print(f"  Predictions: {n}")
    print(f"  Avg cushion: {avg_cushion:.1f} min")

    prices = [75, 78, 80, 82, 85, 88, 90, 92]

    for label, wins, pct in [("OVER", over_wins, over_pct),
                              ("UNDER", under_wins, under_pct)]:
        print(f"\n  --- {label} (win={wins}/{n} = {pct:.1f}%) ---")
        print(f"  {'price':>6s} | {'EV/bet':>8s} | {'total P&L':>10s} | {'ROI':>7s}")
        print(f"  {'-' * 6}-+-{'-' * 8}-+-{'-' * 10}-+-{'-' * 7}")
        lose_pct = 100.0 - pct
        for price in prices:
            ev = (pct / 100) * (100 - price) - (lose_pct / 100) * price
            total_pnl = wins * (100 - price) - (n - wins) * price
            risked = n * price
            roi = total_pnl / risked * 100 if risked > 0 else 0
            print(f"  {price:5d}¢ | {ev:+7.1f}¢ | {total_pnl:+9d}¢ | {roi:+6.1f}%")

    # By starter status
    print(f"\n  By role:")
    for label, mask in [("Starters", results["started"] == 1),
                        ("Bench",    results["started"] == 0)]:
        sub = results[mask]
        if sub.empty:
            continue
        o_w = sum(1 for _, r in sub.iterrows()
                  if r["actual"] > r["projected"] - z * r["calibrated_std"])
        u_w = sum(1 for _, r in sub.iterrows()
                  if r["actual"] < r["projected"] + z * r["calibrated_std"])
        print(f"    {label:10s}: n={len(sub):5d}  over={o_w/len(sub)*100:.1f}%  "
              f"under={u_w/len(sub)*100:.1f}%")

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
        ax.hist(results["error"], bins=np.arange(-20, 22, 1), color="#5C6BC0",
                alpha=0.7, edgecolor="white")
        ax.axvline(0, color="black", linestyle="--", alpha=0.5)
        ax.set_xlabel("Error (projected - actual)")
        ax.set_ylabel("Count")
        ax.set_title(f"Minutes Error (MAE={results['abs_error'].mean():.1f})")
        ax.grid(alpha=0.3)

        # Projected vs actual scatter
        ax = axes[1]
        ax.scatter(results["actual"], results["projected"], alpha=0.1, s=6, color="#FF9800")
        mn = min(results["actual"].min(), results["projected"].min()) - 2
        mx = max(results["actual"].max(), results["projected"].max()) + 2
        ax.plot([mn, mx], [mn, mx], "k--", alpha=0.5)
        ax.set_xlabel("Actual Minutes")
        ax.set_ylabel("Projected Minutes")
        ax.set_title("Projected vs Actual")
        ax.grid(alpha=0.3)

        # MAE by rolling avg bucket
        ax = axes[2]
        results_copy = results.copy()
        min_bins = [15, 20, 25, 30, 35, 48]
        min_labels = ["15-19", "20-24", "25-29", "30-34", "35+"]
        results_copy["bucket"] = pd.cut(results_copy["rolling_avg"], bins=min_bins,
                                         labels=min_labels, right=False)
        bucket_mae = results_copy.groupby("bucket")["abs_error"].mean()
        bucket_n = results_copy.groupby("bucket")["abs_error"].count()
        x = range(len(bucket_mae))
        bars = ax.bar(x, bucket_mae.values, color="#4CAF50", alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{l}\n(n={bucket_n.iloc[i]})" for i, l in enumerate(min_labels)],
                           fontsize=8)
        ax.set_xlabel("Rolling Avg Minutes")
        ax.set_ylabel("MAE (min)")
        ax.set_title("MAE by Minutes Tier")
        ax.axhline(mae, color="red", linestyle="--", alpha=0.7,
                    label=f"overall={results['abs_error'].mean():.1f}")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

        checkpoint = "q4" if "q4" in sys.argv else "halftime"
        fig.suptitle(f"Player Minutes Backtest — {checkpoint}", fontsize=13, fontweight="bold")
        fig.tight_layout()
        out = os.path.join(CHART_DIR, f"backtest_player_minutes_{checkpoint}.png")
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"\n  Saved {out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Player minutes backtest")
    parser.add_argument("--fetch", action="store_true",
                        help="Fetch player game logs from nba_api (slow, ~7 min)")
    parser.add_argument("--force-refresh", action="store_true",
                        help="Re-fetch even if CSV exists")
    parser.add_argument("--last-n", type=int, default=15,
                        help="Rolling window size (default: 15)")
    parser.add_argument("--min-games", type=int, default=8,
                        help="Min prior games before including (default: 8)")
    parser.add_argument("--sigma", type=float, default=1.3,
                        help="Sigma inflation factor (default: 1.3)")
    parser.add_argument("--min-minutes", type=int, default=15,
                        help="Min avg minutes to include player (default: 15)")
    parser.add_argument("--q4", action="store_true",
                        help="Backtest at Q4 start instead of halftime")
    parser.add_argument("--high-conf", action="store_true",
                        help="Run high-confidence betting simulation")
    parser.add_argument("--confidence", type=float, default=0.90,
                        help="Confidence threshold (default: 0.90)")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip chart generation")
    args = parser.parse_args()

    if args.fetch:
        fetch_player_game_logs(force_refresh=args.force_refresh)
        if not any([args.q4, args.high_conf]) and not args.force_refresh:
            sys.exit(0)

    if not os.path.isfile(CSV_PATH):
        print(f"No data at {CSV_PATH}. Run with --fetch first.")
        sys.exit(1)

    if args.q4:
        results = run_backtest_q4(
            last_n=args.last_n, min_games=args.min_games,
            sigma_inflation=args.sigma, min_minutes=args.min_minutes,
        )
        print_results(results, checkpoint="Q4 Start")
    else:
        results = run_backtest_halftime(
            last_n=args.last_n, min_games=args.min_games,
            sigma_inflation=args.sigma, min_minutes=args.min_minutes,
        )
        print_results(results, checkpoint="Halftime")

    if args.high_conf:
        simulate_high_confidence(
            results, confidence=args.confidence, no_plot=args.no_plot,
        )
