#!/usr/bin/env python3
"""EDA: Points per quarter analysis from NBA team_scores.csv."""

from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

_src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _src) if _src not in sys.path else None
from paths import project_path

CSV_PATH = project_path("data", "nba", "team_scores.csv")
CHART_DIR = project_path("charts", "nba")


def load() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    df = df[(df["q1"] > 0) & (df["q2"] > 0) & (df["q3"] > 0) & (df["q4"] > 0)]
    return df


def _paired(df: pd.DataFrame):
    """Return home/away DataFrames indexed by game_id on their common games."""
    home = df[df["home_away"] == "home"].set_index("game_id")
    away = df[df["home_away"] == "away"].set_index("game_id")
    common = home.index.intersection(away.index)
    return home.loc[common], away.loc[common]


# ---------------------------------------------------------------------------
# Text summaries
# ---------------------------------------------------------------------------

def league_quarter_summary(df: pd.DataFrame) -> None:
    print("=" * 60)
    print("LEAGUE-WIDE QUARTER SCORING")
    print("=" * 60)
    quarters = ["q1", "q2", "q3", "q4"]
    for q in quarters:
        vals = df[q]
        print(f"  {q.upper()}: mean={vals.mean():.1f}  std={vals.std():.1f}  "
              f"min={vals.min()}  max={vals.max()}  median={vals.median():.0f}")
    print(f"  Total: mean={df['total'].mean():.1f}  std={df['total'].std():.1f}")

    print("\nQuarter share of total points:")
    for q in quarters:
        pct = (df[q] / df["total"]).mean() * 100
        print(f"  {q.upper()}: {pct:.1f}%")

    ot_games = df[df["ot1"] > 0]
    print(f"\nOvertime games: {len(ot_games)} / {len(df)} ({len(ot_games)/len(df)*100:.1f}%)")


def home_away_split(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("HOME vs AWAY QUARTER SCORING")
    print("=" * 60)
    quarters = ["q1", "q2", "q3", "q4", "total"]
    for loc in ["home", "away"]:
        sub = df[df["home_away"] == loc]
        vals = {q: sub[q].mean() for q in quarters}
        print(f"  {loc.upper():5s}: Q1={vals['q1']:.1f}  Q2={vals['q2']:.1f}  "
              f"Q3={vals['q3']:.1f}  Q4={vals['q4']:.1f}  Total={vals['total']:.1f}")


def quarter_correlations(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("QUARTER CORRELATIONS")
    print("=" * 60)
    quarters = ["q1", "q2", "q3", "q4", "total"]
    corr = df[quarters].corr()
    print(corr.round(2).to_string())


def highest_lowest_quarter_teams(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("TEAM QUARTER AVERAGES (sorted by total)")
    print("=" * 60)
    team_avgs = df.groupby("team")[["q1", "q2", "q3", "q4", "total"]].mean()
    team_avgs = team_avgs.sort_values("total", ascending=False)
    print(team_avgs.round(1).to_string())

    print("\n" + "-" * 60)
    print("BEST & WORST QUARTER PER TEAM (relative to team mean)")
    print("-" * 60)
    quarters = ["q1", "q2", "q3", "q4"]
    for team in team_avgs.index:
        row = team_avgs.loc[team, quarters]
        best = row.idxmax().upper()
        worst = row.idxmin().upper()
        spread = row.max() - row.min()
        print(f"  {team}: best={best} ({row.max():.1f})  "
              f"worst={worst} ({row.min():.1f})  spread={spread:.1f}")


def quarter_distribution(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("QUARTER SCORE DISTRIBUTION")
    print("=" * 60)
    quarters = ["q1", "q2", "q3", "q4"]
    bins = [0, 20, 25, 30, 35, 40, 100]
    labels = ["<20", "20-24", "25-29", "30-34", "35-39", "40+"]
    for q in quarters:
        cuts = pd.cut(df[q], bins=bins, labels=labels, right=False)
        dist = cuts.value_counts().sort_index()
        pcts = (dist / len(df) * 100).round(1)
        parts = [f"{l}:{p}%" for l, p in zip(labels, pcts)]
        print(f"  {q.upper()}: {' | '.join(parts)}")


def combined_quarter_totals(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("COMBINED QUARTER TOTALS (both teams)")
    print("=" * 60)
    home, away = _paired(df)
    quarters = ["q1", "q2", "q3", "q4"]
    for q in quarters:
        combined = home[q] + away[q]
        print(f"  {q.upper()}: mean={combined.mean():.1f}  std={combined.std():.1f}  "
              f"min={combined.min()}  max={combined.max()}")
    total_combined = home["total"] + away["total"]
    print(f"  Game total: mean={total_combined.mean():.1f}  "
          f"std={total_combined.std():.1f}")


def q4_pace_correlation(df: pd.DataFrame) -> None:
    """How well does Q1-3 pace predict Q4 pace?

    Pace = combined points (both teams) / league average for those quarters.
    If pace_q1q3 and pace_q4 are correlated, games that run hot/cold early
    continue that trend in Q4 — which validates pace-based projection.
    """
    print("\n" + "=" * 60)
    print("Q4 PACE vs Q1-Q3 PACE CORRELATION")
    print("=" * 60)

    home, away = _paired(df)

    # Combined scoring per quarter
    cq1 = home["q1"] + away["q1"]
    cq2 = home["q2"] + away["q2"]
    cq3 = home["q3"] + away["q3"]
    cq4 = home["q4"] + away["q4"]

    q1q3_total = cq1 + cq2 + cq3
    game_total = q1q3_total + cq4

    # Pace = actual / league average
    avg_q1q3 = q1q3_total.mean()
    avg_q4 = cq4.mean()

    pace_q1q3 = q1q3_total / avg_q1q3
    pace_q4 = cq4 / avg_q4

    r = pace_q1q3.corr(pace_q4)
    print(f"  League avg Q1-Q3 combined: {avg_q1q3:.1f}")
    print(f"  League avg Q4 combined:    {avg_q4:.1f}")
    print(f"  Pearson r(pace_Q1Q3, pace_Q4): {r:.3f}")

    # Also show raw points correlation
    r_raw = q1q3_total.corr(cq4)
    print(f"  Pearson r(pts_Q1Q3, pts_Q4):   {r_raw:.3f}")

    # Bucket analysis: games by Q1-3 pace quintile, show Q4 outcome
    pace_q1q3_arr = pace_q1q3.values.astype(float)
    pace_q4_arr = pace_q4.values.astype(float)
    q1q3_arr = q1q3_total.values.astype(float)
    q4_arr = cq4.values.astype(float)

    print("\n  Q1-3 pace bucket  |  n  | avg Q4 pts | avg Q4 pace | Q4 std")
    print("  " + "-" * 64)
    pct_cuts = [0, 20, 40, 60, 80, 100]
    thresholds = np.percentile(pace_q1q3_arr, pct_cuts)
    for i in range(len(pct_cuts) - 1):
        lo, hi = thresholds[i], thresholds[i + 1]
        if i == len(pct_cuts) - 2:
            mask = (pace_q1q3_arr >= lo) & (pace_q1q3_arr <= hi)
        else:
            mask = (pace_q1q3_arr >= lo) & (pace_q1q3_arr < hi)
        n = mask.sum()
        if n > 0:
            q4_sub = q4_arr[mask]
            pace_sub = pace_q4_arr[mask]
            lo_pct, hi_pct = pct_cuts[i], pct_cuts[i + 1]
            print(f"  P{lo_pct:2d}-P{hi_pct:3d} ({lo:.2f}-{hi:.2f}) | {n:3d} | "
                  f"   {q4_sub.mean():5.1f}    |    {pace_sub.mean():.3f}    | {q4_sub.std():.1f}")

    # Half-by-half: Q1Q2 pace vs Q3Q4 pace (halftime projection relevance)
    cq3q4 = cq3 + cq4
    cq1q2 = cq1 + cq2
    avg_q1q2 = cq1q2.mean()
    avg_q3q4 = cq3q4.mean()
    pace_q1q2 = cq1q2 / avg_q1q2
    pace_q3q4 = cq3q4 / avg_q3q4
    r_half = pace_q1q2.corr(pace_q3q4)
    r_half_raw = cq1q2.corr(cq3q4)
    print(f"\n  Halftime pace correlation:")
    print(f"    Pearson r(pace_H1, pace_H2): {r_half:.3f}")
    print(f"    Pearson r(pts_H1, pts_H2):   {r_half_raw:.3f}")

    # Q1-Q3 each individually vs Q4
    for q_label, cq in [("Q1", cq1), ("Q2", cq2), ("Q3", cq3)]:
        r_q = cq.corr(cq4)
        print(f"    r({q_label}, Q4): {r_q:.3f}")


def plot_pace_correlation(df: pd.DataFrame) -> None:
    """Scatter + regression: Q1-3 pace vs Q4 pace."""
    home, away = _paired(df)

    q1q3 = (home["q1"] + away["q1"] + home["q2"] + away["q2"] +
            home["q3"] + away["q3"])
    cq4 = home["q4"] + away["q4"]

    pace_q1q3 = (q1q3 / q1q3.mean()).values.astype(float)
    pace_q4 = (cq4 / cq4.mean()).values.astype(float)

    r = np.corrcoef(pace_q1q3, pace_q4)[0, 1]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: pace scatter
    ax = axes[0]
    ax.scatter(pace_q1q3, pace_q4, alpha=0.35, s=15, color="#5C6BC0")
    # Regression line
    m, b = np.polyfit(pace_q1q3, pace_q4, 1)
    x_line = np.linspace(pace_q1q3.min(), pace_q1q3.max(), 100)
    ax.plot(x_line, m * x_line + b, color="#F44336", linewidth=2,
            label=f"y={m:.2f}x+{b:.2f}  r={r:.3f}")
    ax.axhline(1.0, color="gray", linestyle=":", alpha=0.5)
    ax.axvline(1.0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Q1-Q3 Pace (actual / avg)")
    ax.set_ylabel("Q4 Pace (actual / avg)")
    ax.set_title(f"Q1-Q3 Pace vs Q4 Pace (r={r:.3f})")
    ax.legend()
    ax.grid(alpha=0.3)

    # Right: halftime pace
    h1 = (home["q1"] + away["q1"] + home["q2"] + away["q2"])
    h2 = (home["q3"] + away["q3"] + home["q4"] + away["q4"])
    pace_h1 = (h1 / h1.mean()).values.astype(float)
    pace_h2 = (h2 / h2.mean()).values.astype(float)
    r2 = np.corrcoef(pace_h1, pace_h2)[0, 1]

    ax = axes[1]
    ax.scatter(pace_h1, pace_h2, alpha=0.35, s=15, color="#4CAF50")
    m2, b2 = np.polyfit(pace_h1, pace_h2, 1)
    x_line2 = np.linspace(pace_h1.min(), pace_h1.max(), 100)
    ax.plot(x_line2, m2 * x_line2 + b2, color="#F44336", linewidth=2,
            label=f"y={m2:.2f}x+{b2:.2f}  r={r2:.3f}")
    ax.axhline(1.0, color="gray", linestyle=":", alpha=0.5)
    ax.axvline(1.0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("1st Half Pace")
    ax.set_ylabel("2nd Half Pace")
    ax.set_title(f"1st Half vs 2nd Half Pace (r={r2:.3f})")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.suptitle("Pace Persistence Analysis", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(CHART_DIR, "pace_correlation.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved pace_correlation.png")


def q4_blowout_analysis(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("Q4 SCORING: BLOWOUTS vs CLOSE GAMES")
    print("=" * 60)
    home, away = _paired(df)
    q3_margin = abs(
        (home["q1"] + home["q2"] + home["q3"]) -
        (away["q1"] + away["q2"] + away["q3"])
    )
    q4_combined = home["q4"] + away["q4"]
    for label, lo, hi in [("Close (<10)", 0, 10), ("Medium (10-20)", 10, 20), ("Blowout (20+)", 20, 100)]:
        mask = (q3_margin >= lo) & (q3_margin < hi)
        subset = q4_combined[mask]
        n = mask.sum()
        if n > 0:
            print(f"  {label:20s}: n={n:4d}  Q4 combined mean={subset.mean():.1f}  std={subset.std():.1f}")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_quarter_distributions(df: pd.DataFrame) -> None:
    """Overlaid histograms of single-team quarter scores."""
    fig, ax = plt.subplots(figsize=(10, 5))
    quarters = ["q1", "q2", "q3", "q4"]
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336"]
    bins = np.arange(5, 55, 2)
    for q, c in zip(quarters, colors):
        ax.hist(df[q], bins=bins, alpha=0.45, label=q.upper(), color=c, edgecolor="white")
    ax.set_xlabel("Points")
    ax.set_ylabel("Frequency")
    ax.set_title("Single-Team Quarter Score Distributions")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(CHART_DIR, "quarter_distributions.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved quarter_distributions.png")


def plot_combined_quarter_distributions(df: pd.DataFrame) -> None:
    """Histograms of combined (both teams) quarter scores."""
    home, away = _paired(df)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    quarters = ["q1", "q2", "q3", "q4"]
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336"]
    bins = np.arange(20, 100, 3)

    for ax, q, c in zip(axes.flat, quarters, colors):
        combined = home[q] + away[q]
        ax.hist(combined, bins=bins, color=c, alpha=0.7, edgecolor="white")
        ax.axvline(combined.mean(), color="black", linestyle="--", linewidth=1.5,
                   label=f"mean={combined.mean():.1f}")
        ax.set_title(f"{q.upper()} Combined", fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    axes[1, 0].set_xlabel("Combined Points")
    axes[1, 1].set_xlabel("Combined Points")
    axes[0, 0].set_ylabel("Frequency")
    axes[1, 0].set_ylabel("Frequency")
    fig.suptitle("Combined Quarter Totals (Both Teams)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(CHART_DIR, "combined_quarter_distributions.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved combined_quarter_distributions.png")


def plot_team_quarter_heatmap(df: pd.DataFrame) -> None:
    """Heatmap of per-team quarter averages."""
    quarters = ["q1", "q2", "q3", "q4"]
    team_avgs = df.groupby("team")[quarters].mean()
    team_avgs = team_avgs.sort_values("q1", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 10))
    im = ax.imshow(team_avgs.values, aspect="auto", cmap="RdYlGn")
    ax.set_xticks(range(len(quarters)))
    ax.set_xticklabels([q.upper() for q in quarters], fontsize=11)
    ax.set_yticks(range(len(team_avgs)))
    ax.set_yticklabels(team_avgs.index, fontsize=9)

    # Annotate cells
    for i in range(len(team_avgs)):
        for j in range(len(quarters)):
            val = team_avgs.values[i, j]
            ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=8,
                    color="white" if val < 26 or val > 32 else "black")

    fig.colorbar(im, ax=ax, label="Avg Points", shrink=0.6)
    ax.set_title("Team Quarter Scoring Averages", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(CHART_DIR, "team_quarter_heatmap.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved team_quarter_heatmap.png")


def plot_quarter_boxplots(df: pd.DataFrame) -> None:
    """Side-by-side boxplots for each quarter."""
    fig, ax = plt.subplots(figsize=(8, 5))
    quarters = ["q1", "q2", "q3", "q4"]
    data = [np.array(df[q].values, dtype=float) for q in quarters]
    bp = ax.boxplot(data, labels=["Q1", "Q2", "Q3", "Q4"], patch_artist=True,
                    showmeans=True, meanline=True)
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336"]
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.5)
    ax.set_ylabel("Points")
    ax.set_title("Quarter Score Distributions (Single Team)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(CHART_DIR, "quarter_boxplots.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved quarter_boxplots.png")


def plot_q4_vs_margin(df: pd.DataFrame) -> None:
    """Scatter: Q4 combined points vs margin entering Q4."""
    home, away = _paired(df)
    margin_q3 = abs(
        (home["q1"] + home["q2"] + home["q3"]) -
        (away["q1"] + away["q2"] + away["q3"])
    )
    q4_combined = home["q4"] + away["q4"]

    margin_arr = margin_q3.values.astype(float)
    q4_arr = q4_combined.values.astype(float)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(margin_arr, q4_arr, alpha=0.3, s=15, color="#5C6BC0")

    # Rolling mean trend
    order = np.argsort(margin_arr)
    margin_sorted = margin_arr[order]
    q4_sorted = q4_arr[order]
    window = max(30, len(margin_sorted) // 20)
    trend = pd.Series(q4_sorted).rolling(window, center=True).mean().values
    ax.plot(margin_sorted, trend, color="#F44336", linewidth=2.5, label=f"Rolling mean (n={window})")

    ax.set_xlabel("Absolute Margin Entering Q4")
    ax.set_ylabel("Q4 Combined Points")
    ax.set_title("Q4 Combined Scoring vs Game Closeness")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(CHART_DIR, "q4_vs_margin.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved q4_vs_margin.png")


def plot_home_away_quarters(df: pd.DataFrame) -> None:
    """Grouped bar chart: home vs away scoring by quarter."""
    quarters = ["q1", "q2", "q3", "q4"]
    home_means = [df[df["home_away"] == "home"][q].mean() for q in quarters]
    away_means = [df[df["home_away"] == "away"][q].mean() for q in quarters]

    x = np.arange(len(quarters))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, home_means, width, label="Home", color="#2196F3", alpha=0.8)
    ax.bar(x + width / 2, away_means, width, label="Away", color="#FF9800", alpha=0.8)

    # Value labels
    for i, (h, a) in enumerate(zip(home_means, away_means)):
        ax.text(i - width / 2, h + 0.2, f"{h:.1f}", ha="center", fontsize=9)
        ax.text(i + width / 2, a + 0.2, f"{a:.1f}", ha="center", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(["Q1", "Q2", "Q3", "Q4"])
    ax.set_ylabel("Avg Points")
    ax.set_title("Home vs Away Scoring by Quarter")
    ax.legend()
    ax.set_ylim(bottom=25)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(CHART_DIR, "home_away_quarters.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved home_away_quarters.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    os.makedirs(CHART_DIR, exist_ok=True)

    df = load()
    print(f"Loaded {len(df)} rows ({df['game_id'].nunique()} games, "
          f"{df['team'].nunique()} teams)")
    print(f"Date range: {df['game_date'].min()} to {df['game_date'].max()}\n")

    # Text summaries
    league_quarter_summary(df)
    home_away_split(df)
    quarter_distribution(df)
    combined_quarter_totals(df)
    q4_pace_correlation(df)
    q4_blowout_analysis(df)
    quarter_correlations(df)
    highest_lowest_quarter_teams(df)

    # Charts
    print("\n" + "=" * 60)
    print("GENERATING CHARTS -> charts/nba/")
    print("=" * 60)
    plot_quarter_distributions(df)
    plot_combined_quarter_distributions(df)
    plot_quarter_boxplots(df)
    plot_team_quarter_heatmap(df)
    plot_q4_vs_margin(df)
    plot_home_away_quarters(df)
    plot_pace_correlation(df)
    print("\nDone.")
