#!/usr/bin/env python3
"""Backtest: game-total projection accuracy at halftime / Q4 start.

For each historical game, simulates being at a checkpoint:
  - Builds team profiles using ONLY games before the current game (no lookahead)
  - Projects remaining quarters using per-team averages (home/away split)
  - Measures projected vs actual game total

Usage:
    PYTHONPATH=src python -m nba.backtest_total_score
    PYTHONPATH=src python -m nba.backtest_total_score --improved --alpha 0.0 --high-conf
    PYTHONPATH=src python -m nba.backtest_total_score --q4 --alpha 0.0 --high-conf
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _src) if _src not in sys.path else None
from paths import project_path

CSV_PATH = project_path("data", "nba", "team_scores.csv")
CHART_DIR = project_path("charts", "nba")


# ---------------------------------------------------------------------------
# Rolling profile builder (no lookahead)
# ---------------------------------------------------------------------------

def build_rolling_profiles(
    df: pd.DataFrame, last_n: int = 20, min_games: int = 10
) -> Dict[Tuple[int, str], dict]:
    """Build per-team rolling profiles using only games before each row.

    For each (game_id, team) pair, computes Q1-Q4 averages and stds from
    the team's prior games (up to last_n), split by home/away.

    Returns dict keyed by (game_id, team) -> profile dict with:
        q1_avg, q2_avg, q3_avg, q4_avg, q1_std, q2_std, q3_std, q4_std,
        games_used, home_away
    Only includes entries where the team has >= min_games prior games
    in the same home/away split.
    """
    profiles: Dict[Tuple[int, str], dict] = {}

    # Sort by date for chronological processing
    df_sorted = df.sort_values("game_date").reset_index(drop=True)

    # Group rows by (team, home_away) for efficient lookback
    # We'll iterate chronologically and maintain a running list of past games
    history: Dict[Tuple[str, str], List[dict]] = defaultdict(list)

    quarters = ["q1", "q2", "q3", "q4"]

    for _, row in df_sorted.iterrows():
        team = row["team"]
        ha = row["home_away"]
        game_id = row["game_id"]
        key = (team, ha)

        past = history[key]
        if len(past) >= min_games:
            window = past[-last_n:]  # most recent last_n games
            prof = {}
            for q in quarters:
                vals = [g[q] for g in window]
                prof[f"{q}_avg"] = sum(vals) / len(vals)
                if len(vals) > 1:
                    mean = prof[f"{q}_avg"]
                    prof[f"{q}_std"] = math.sqrt(
                        sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
                    )
                else:
                    prof[f"{q}_std"] = 5.0
            prof["games_used"] = len(window)
            prof["home_away"] = ha
            profiles[(game_id, team)] = prof

        # Append current game to history (after building profile, so no lookahead)
        history[key].append({q: row[q] for q in quarters})

    return profiles


# ---------------------------------------------------------------------------
# Defensive profile builder (no lookahead)
# ---------------------------------------------------------------------------

def build_defensive_profiles(
    df: pd.DataFrame, last_n: int = 20, min_games: int = 10
) -> Tuple[Dict[Tuple[int, str], dict], Dict[int, dict]]:
    """Build per-team rolling defensive profiles using only prior games.

    Each row (team=X, opponent=Y, q1-q4) means X scored q1-q4 against Y,
    so those scores are Y's defensive data (points allowed).

    Also computes running league-average points per quarter (all prior games).

    Returns:
        def_profiles: Dict[(game_id, team) -> {def_q1_avg..def_q4_avg, def_games_used}]
            Defensive averages for team at the time of game_id.
        league_avgs: Dict[game_id -> {q1..q4}]
            League-wide average scoring per quarter at the time of game_id.
    """
    def_profiles: Dict[Tuple[int, str], dict] = {}
    league_avgs: Dict[int, dict] = {}

    df_sorted = df.sort_values("game_date").reset_index(drop=True)

    # defense_history[team] = list of dicts {q1..q4} — points ALLOWED by team
    defense_history: Dict[str, List[dict]] = defaultdict(list)

    # Running league totals for league avg
    league_totals = {"q1": 0.0, "q2": 0.0, "q3": 0.0, "q4": 0.0}
    league_count = 0

    quarters = ["q1", "q2", "q3", "q4"]

    seen_game_ids: set = set()

    for _, row in df_sorted.iterrows():
        team = row["team"]
        opponent = row["opponent"]
        game_id = row["game_id"]

        # Build league avg snapshot before processing this game
        if game_id not in seen_game_ids:
            seen_game_ids.add(game_id)
            if league_count > 0:
                league_avgs[game_id] = {
                    q: league_totals[q] / league_count for q in quarters
                }

        # Build defensive profile for the opponent (this row's scores are
        # points allowed BY the opponent)
        past = defense_history[opponent]
        if len(past) >= min_games:
            window = past[-last_n:]
            prof = {}
            for q in quarters:
                vals = [g[q] for g in window]
                prof[f"def_{q}_avg"] = sum(vals) / len(vals)
            prof["def_games_used"] = len(window)
            def_profiles[(game_id, opponent)] = prof

        # Also build profile for the team itself (so we have it when needed)
        past_team = defense_history[team]
        # We don't overwrite if opponent already set it for same game_id
        if (game_id, team) not in def_profiles and len(past_team) >= min_games:
            window = past_team[-last_n:]
            prof = {}
            for q in quarters:
                vals = [g[q] for g in window]
                prof[f"def_{q}_avg"] = sum(vals) / len(vals)
            prof["def_games_used"] = len(window)
            def_profiles[(game_id, team)] = prof

        # Append this row's scores as defensive data for the opponent
        # (team scored these points, so opponent allowed them)
        defense_history[opponent].append({q: row[q] for q in quarters})

        # Update league totals
        league_totals_update = {q: row[q] for q in quarters}
        for q in quarters:
            league_totals[q] += league_totals_update[q]
        league_count += 1

    return def_profiles, league_avgs


# ---------------------------------------------------------------------------
# Back-to-back detection
# ---------------------------------------------------------------------------

def build_b2b_flags(df: pd.DataFrame) -> Dict[Tuple[int, str], bool]:
    """Flag games where a team played the previous day (back-to-back).

    Returns Dict[(game_id, team) -> True] for games on a back-to-back.
    """
    b2b: Dict[Tuple[int, str], bool] = {}

    for team, group in df.groupby("team"):
        group_sorted = group.sort_values("game_date").reset_index(drop=True)
        dates = pd.to_datetime(group_sorted["game_date"])
        game_ids = group_sorted["game_id"].values

        for i in range(len(group_sorted)):
            is_b2b = False
            if i > 0:
                delta = (dates.iloc[i] - dates.iloc[i - 1]).days
                if delta == 1:
                    is_b2b = True
            if is_b2b:
                b2b[(game_ids[i], team)] = True

    return b2b


# ---------------------------------------------------------------------------
# Backtest runner
# ---------------------------------------------------------------------------

def run_backtest(
    last_n: int = 20, min_games: int = 10
) -> pd.DataFrame:
    """Run halftime projection backtest over all historical games.

    Returns DataFrame with columns:
        game_id, game_date, home, away, h1_score, projected_total,
        actual_total, error, abs_error, pace, margin_at_half,
        remaining_std, lo_1sig, hi_1sig
    """
    df = pd.read_csv(CSV_PATH)
    # Filter to games with valid quarter scores
    df = df[(df["q1"] > 0) & (df["q2"] > 0) & (df["q3"] > 0) & (df["q4"] > 0)]

    print(f"Building rolling profiles (last_n={last_n}, min_games={min_games})...")
    profiles = build_rolling_profiles(df, last_n=last_n, min_games=min_games)
    print(f"  Built {len(profiles)} team-game profiles")

    # Pair home/away rows by game_id
    home = df[df["home_away"] == "home"].set_index("game_id")
    away = df[df["home_away"] == "away"].set_index("game_id")
    common = home.index.intersection(away.index)
    home = home.loc[common]
    away = away.loc[common]

    results = []
    for game_id in common:
        h = home.loc[game_id]
        a = away.loc[game_id]

        home_team = h["team"]
        away_team = a["team"]

        # Need profiles for both teams in their respective home/away splits
        h_prof = profiles.get((game_id, home_team))
        a_prof = profiles.get((game_id, away_team))
        if h_prof is None or a_prof is None:
            continue

        # Actual scores
        h1_home = h["q1"] + h["q2"]
        h1_away = a["q1"] + a["q2"]
        h1_score = h1_home + h1_away
        actual_total = h["total"] + a["total"]

        # Expected H1 from profiles
        expected_h1 = (
            h_prof["q1_avg"] + h_prof["q2_avg"] +
            a_prof["q1_avg"] + a_prof["q2_avg"]
        )
        if expected_h1 <= 0:
            continue

        pace = h1_score / expected_h1

        # Project H2
        expected_h2 = (
            h_prof["q3_avg"] + h_prof["q4_avg"] +
            a_prof["q3_avg"] + a_prof["q4_avg"]
        )
        projected_total = h1_score + expected_h2 * pace

        # Confidence band
        remaining_var = (
            h_prof["q3_std"] ** 2 + h_prof["q4_std"] ** 2 +
            a_prof["q3_std"] ** 2 + a_prof["q4_std"] ** 2
        )
        remaining_std = math.sqrt(remaining_var) * pace

        error = projected_total - actual_total
        margin_at_half = abs(h1_home - h1_away)

        results.append({
            "game_id": game_id,
            "game_date": h["game_date"],
            "home": home_team,
            "away": away_team,
            "h1_score": h1_score,
            "projected_total": round(projected_total, 1),
            "actual_total": actual_total,
            "error": round(error, 1),
            "abs_error": round(abs(error), 1),
            "pace": round(pace, 3),
            "margin_at_half": margin_at_half,
            "remaining_std": round(remaining_std, 1),
            "lo_1sig": round(projected_total - remaining_std, 1),
            "hi_1sig": round(projected_total + remaining_std, 1),
        })

    results_df = pd.DataFrame(results)
    print(f"  Backtested {len(results_df)} games")
    return results_df


# ---------------------------------------------------------------------------
# Improved backtest with pace dampening + sigma inflation
# ---------------------------------------------------------------------------

def run_backtest_improved(
    last_n: int = 20,
    min_games: int = 10,
    pace_alpha: float = 0.25,
    sigma_inflation: float = 1.45,
    opp_adjust: bool = False,
    b2b_adjust: bool = False,
    b2b_penalty: float = 1.5,
    blowout_adjust: bool = False,
    blowout_threshold: float = 15.0,
    blowout_rate: float = 0.1,
) -> pd.DataFrame:
    """Run halftime backtest with pace dampening and calibrated sigma.

    Two fixes over the raw model:
      1. Pace dampening: dampened_pace = 1.0 + alpha * (raw_pace - 1.0)
         At alpha=0.25, a slow game (pace=0.85) becomes 0.9625, reducing
         H2 over-adjustment from -16.5 pts to -4.1 pts.
      2. Sigma inflation: calibrated_std = raw_std * sigma_inflation
         Raw 1σ covers only ~50%, so inflate by 1.45x to reach ~68%.

    Optional adjustments:
      - opp_adjust: scale expected H2 by opponent defensive strength
      - b2b_adjust: subtract b2b_penalty per remaining quarter for b2b teams
      - blowout_adjust: reduce expected scoring when margin exceeds threshold
        reduction = max(0, margin - threshold) * rate * remaining_qs

    Returns same columns as run_backtest() plus dampened_pace, calibrated_std.
    """
    df = pd.read_csv(CSV_PATH)
    df = df[(df["q1"] > 0) & (df["q2"] > 0) & (df["q3"] > 0) & (df["q4"] > 0)]

    print(f"Building rolling profiles (last_n={last_n}, min_games={min_games})...")
    profiles = build_rolling_profiles(df, last_n=last_n, min_games=min_games)
    print(f"  Built {len(profiles)} team-game profiles")

    def_profiles: Optional[Dict] = None
    league_avgs: Optional[Dict] = None
    b2b_flags: Optional[Dict] = None

    if opp_adjust:
        def_profiles, league_avgs = build_defensive_profiles(df, last_n=last_n, min_games=min_games)
        print(f"  Built {len(def_profiles)} defensive profiles")
    if b2b_adjust:
        b2b_flags = build_b2b_flags(df)
        print(f"  Found {len(b2b_flags)} back-to-back games")

    home = df[df["home_away"] == "home"].set_index("game_id")
    away = df[df["home_away"] == "away"].set_index("game_id")
    common = home.index.intersection(away.index)
    home = home.loc[common]
    away = away.loc[common]

    results = []
    for game_id in common:
        h = home.loc[game_id]
        a = away.loc[game_id]

        home_team = h["team"]
        away_team = a["team"]

        h_prof = profiles.get((game_id, home_team))
        a_prof = profiles.get((game_id, away_team))
        if h_prof is None or a_prof is None:
            continue

        h1_home = h["q1"] + h["q2"]
        h1_away = a["q1"] + a["q2"]
        h1_score = h1_home + h1_away
        actual_total = h["total"] + a["total"]

        expected_h1 = (
            h_prof["q1_avg"] + h_prof["q2_avg"] +
            a_prof["q1_avg"] + a_prof["q2_avg"]
        )
        if expected_h1 <= 0:
            continue

        raw_pace = h1_score / expected_h1
        dampened_pace = 1.0 + pace_alpha * (raw_pace - 1.0)

        expected_h2 = (
            h_prof["q3_avg"] + h_prof["q4_avg"] +
            a_prof["q3_avg"] + a_prof["q4_avg"]
        )

        # Opponent defensive adjustment
        if opp_adjust and def_profiles and league_avgs and game_id in league_avgs:
            lg = league_avgs[game_id]
            lg_q34 = lg["q3"] + lg["q4"]
            if lg_q34 > 0:
                # Home team faces away team's defense
                a_def = def_profiles.get((game_id, away_team))
                if a_def is not None:
                    opp_def_q34 = a_def["def_q3_avg"] + a_def["def_q4_avg"]
                    mult_home = max(0.85, min(1.15, opp_def_q34 / lg_q34))
                else:
                    mult_home = 1.0
                # Away team faces home team's defense
                h_def = def_profiles.get((game_id, home_team))
                if h_def is not None:
                    opp_def_q34 = h_def["def_q3_avg"] + h_def["def_q4_avg"]
                    mult_away = max(0.85, min(1.15, opp_def_q34 / lg_q34))
                else:
                    mult_away = 1.0
                expected_h2 = (
                    (h_prof["q3_avg"] + h_prof["q4_avg"]) * mult_home +
                    (a_prof["q3_avg"] + a_prof["q4_avg"]) * mult_away
                )

        projected_total = h1_score + expected_h2 * dampened_pace

        # Back-to-back adjustment
        if b2b_adjust and b2b_flags:
            remaining_qs = 2  # Q3 + Q4
            if b2b_flags.get((game_id, home_team)):
                projected_total -= b2b_penalty * remaining_qs
            if b2b_flags.get((game_id, away_team)):
                projected_total -= b2b_penalty * remaining_qs

        # Blowout adjustment
        margin_at_half = abs(h1_home - h1_away)
        if blowout_adjust and margin_at_half > blowout_threshold:
            remaining_qs = 2  # Q3 + Q4
            blowout_reduction = (margin_at_half - blowout_threshold) * blowout_rate * remaining_qs
            projected_total -= blowout_reduction

        remaining_var = (
            h_prof["q3_std"] ** 2 + h_prof["q4_std"] ** 2 +
            a_prof["q3_std"] ** 2 + a_prof["q4_std"] ** 2
        )
        raw_std = math.sqrt(remaining_var) * dampened_pace
        calibrated_std = raw_std * sigma_inflation

        error = projected_total - actual_total

        results.append({
            "game_id": game_id,
            "game_date": h["game_date"],
            "home": home_team,
            "away": away_team,
            "h1_score": h1_score,
            "projected_total": round(projected_total, 1),
            "actual_total": actual_total,
            "error": round(error, 1),
            "abs_error": round(abs(error), 1),
            "pace": round(raw_pace, 3),
            "dampened_pace": round(dampened_pace, 3),
            "margin_at_half": margin_at_half,
            "remaining_std": round(calibrated_std, 1),
            "calibrated_std": round(calibrated_std, 1),
            "lo_1sig": round(projected_total - calibrated_std, 1),
            "hi_1sig": round(projected_total + calibrated_std, 1),
        })

    results_df = pd.DataFrame(results)
    flags = []
    if opp_adjust:
        flags.append("opp")
    if b2b_adjust:
        flags.append(f"b2b({b2b_penalty})")
    if blowout_adjust:
        flags.append(f"blowout(>{blowout_threshold:.0f},{blowout_rate})")
    flag_str = f", {'+'.join(flags)}" if flags else ""
    print(f"  Backtested {len(results_df)} games (improved: alpha={pace_alpha}, sigma_inflation={sigma_inflation}{flag_str})")
    return results_df


# ---------------------------------------------------------------------------
# Q4 start backtest (3 quarters completed)
# ---------------------------------------------------------------------------

def run_backtest_q4(
    last_n: int = 20,
    min_games: int = 10,
    pace_alpha: float = 0.25,
    sigma_inflation: float = 1.45,
    opp_adjust: bool = False,
    b2b_adjust: bool = False,
    b2b_penalty: float = 1.5,
    blowout_adjust: bool = False,
    blowout_threshold: float = 20.0,
    blowout_rate: float = 0.1,
) -> pd.DataFrame:
    """Backtest projection accuracy at start of Q4 (3 quarters completed).

    Same pace dampening + sigma inflation as the improved halftime model,
    but with only Q4 remaining → lower uncertainty, better accuracy.

    Optional adjustments:
      - opp_adjust: scale expected Q4 by opponent defensive strength
      - b2b_adjust: subtract b2b_penalty for each b2b team (1 remaining quarter)
      - blowout_adjust: reduce expected Q4 scoring when Q3 margin exceeds threshold

    Pace is computed from Q1-Q3 actual vs expected.
    """
    df = pd.read_csv(CSV_PATH)
    df = df[(df["q1"] > 0) & (df["q2"] > 0) & (df["q3"] > 0) & (df["q4"] > 0)]

    print(f"Building rolling profiles (last_n={last_n}, min_games={min_games})...")
    profiles = build_rolling_profiles(df, last_n=last_n, min_games=min_games)
    print(f"  Built {len(profiles)} team-game profiles")

    def_profiles: Optional[Dict] = None
    league_avgs: Optional[Dict] = None
    b2b_flags: Optional[Dict] = None

    if opp_adjust:
        def_profiles, league_avgs = build_defensive_profiles(df, last_n=last_n, min_games=min_games)
        print(f"  Built {len(def_profiles)} defensive profiles")
    if b2b_adjust:
        b2b_flags = build_b2b_flags(df)
        print(f"  Found {len(b2b_flags)} back-to-back games")

    home = df[df["home_away"] == "home"].set_index("game_id")
    away = df[df["home_away"] == "away"].set_index("game_id")
    common = home.index.intersection(away.index)
    home = home.loc[common]
    away = away.loc[common]

    results = []
    for game_id in common:
        h = home.loc[game_id]
        a = away.loc[game_id]

        home_team = h["team"]
        away_team = a["team"]

        h_prof = profiles.get((game_id, home_team))
        a_prof = profiles.get((game_id, away_team))
        if h_prof is None or a_prof is None:
            continue

        # Score through 3 quarters
        q3_home = h["q1"] + h["q2"] + h["q3"]
        q3_away = a["q1"] + a["q2"] + a["q3"]
        q3_score = q3_home + q3_away
        actual_total = h["total"] + a["total"]
        actual_q4 = actual_total - q3_score

        # Expected through Q1-Q3
        expected_q3 = (
            h_prof["q1_avg"] + h_prof["q2_avg"] + h_prof["q3_avg"] +
            a_prof["q1_avg"] + a_prof["q2_avg"] + a_prof["q3_avg"]
        )
        if expected_q3 <= 0:
            continue

        raw_pace = q3_score / expected_q3
        dampened_pace = 1.0 + pace_alpha * (raw_pace - 1.0)

        # Only Q4 remaining
        expected_q4 = h_prof["q4_avg"] + a_prof["q4_avg"]

        # Opponent defensive adjustment (Q4 only)
        if opp_adjust and def_profiles and league_avgs and game_id in league_avgs:
            lg = league_avgs[game_id]
            lg_q4 = lg["q4"]
            if lg_q4 > 0:
                # Home team faces away team's defense
                a_def = def_profiles.get((game_id, away_team))
                if a_def is not None:
                    mult_home = max(0.85, min(1.15, a_def["def_q4_avg"] / lg_q4))
                else:
                    mult_home = 1.0
                # Away team faces home team's defense
                h_def = def_profiles.get((game_id, home_team))
                if h_def is not None:
                    mult_away = max(0.85, min(1.15, h_def["def_q4_avg"] / lg_q4))
                else:
                    mult_away = 1.0
                expected_q4 = (
                    h_prof["q4_avg"] * mult_home +
                    a_prof["q4_avg"] * mult_away
                )

        projected_total = q3_score + expected_q4 * dampened_pace

        # Back-to-back adjustment
        if b2b_adjust and b2b_flags:
            remaining_qs = 1  # Q4 only
            if b2b_flags.get((game_id, home_team)):
                projected_total -= b2b_penalty * remaining_qs
            if b2b_flags.get((game_id, away_team)):
                projected_total -= b2b_penalty * remaining_qs

        # Blowout adjustment
        margin_at_q3 = abs(q3_home - q3_away)
        if blowout_adjust and margin_at_q3 > blowout_threshold:
            remaining_qs = 1  # Q4 only
            blowout_reduction = (margin_at_q3 - blowout_threshold) * blowout_rate * remaining_qs
            projected_total -= blowout_reduction

        # Variance from Q4 only
        remaining_var = h_prof["q4_std"] ** 2 + a_prof["q4_std"] ** 2
        raw_std = math.sqrt(remaining_var) * dampened_pace
        calibrated_std = raw_std * sigma_inflation

        error = projected_total - actual_total

        results.append({
            "game_id": game_id,
            "game_date": h["game_date"],
            "home": home_team,
            "away": away_team,
            "h1_score": q3_score,  # reuse column name for print_results compat
            "q3_score": q3_score,
            "actual_q4": actual_q4,
            "projected_total": round(projected_total, 1),
            "actual_total": actual_total,
            "error": round(error, 1),
            "abs_error": round(abs(error), 1),
            "pace": round(raw_pace, 3),
            "dampened_pace": round(dampened_pace, 3),
            "margin_at_half": margin_at_q3,  # reuse for print_results compat
            "margin_at_q3": margin_at_q3,
            "remaining_std": round(calibrated_std, 1),
            "calibrated_std": round(calibrated_std, 1),
            "lo_1sig": round(projected_total - calibrated_std, 1),
            "hi_1sig": round(projected_total + calibrated_std, 1),
        })

    results_df = pd.DataFrame(results)
    flags = []
    if opp_adjust:
        flags.append("opp")
    if b2b_adjust:
        flags.append(f"b2b({b2b_penalty})")
    if blowout_adjust:
        flags.append(f"blowout(>{blowout_threshold:.0f},{blowout_rate})")
    flag_str = f", {'+'.join(flags)}" if flags else ""
    print(f"  Backtested {len(results_df)} games (Q4 start: alpha={pace_alpha}, sigma={sigma_inflation}{flag_str})")
    return results_df


def print_results_q4(results: pd.DataFrame) -> None:
    """Print Q4-start backtest summary."""
    if results.empty:
        print("No results to display.")
        return

    n = len(results)
    mae = results["abs_error"].mean()
    bias = results["error"].mean()
    rmse = math.sqrt((results["error"] ** 2).mean())
    median_ae = results["abs_error"].median()

    # Naive baseline: Q3 score + average Q4 (~55 pts combined)
    results = results.copy()
    avg_q4 = results["actual_q4"].mean()

    print("\n" + "=" * 60)
    print("Q4 START PROJECTION BACKTEST RESULTS")
    print("=" * 60)
    print(f"  Games backtested:  {n}")
    print(f"  Date range:        {results['game_date'].min()} to {results['game_date'].max()}")
    print(f"  Avg actual Q4:     {avg_q4:.1f} pts combined")

    print(f"\n  --- Model Performance ---")
    print(f"  MAE:               {mae:.1f} pts")
    print(f"  Median AE:         {median_ae:.1f} pts")
    print(f"  RMSE:              {rmse:.1f} pts")
    print(f"  Bias:              {bias:+.1f} pts")

    # Naive: assume Q4 = league-average Q4
    results["naive_projected"] = results["q3_score"] + avg_q4
    results["naive_error"] = results["naive_projected"] - results["actual_total"]
    naive_mae = results["naive_error"].abs().mean()
    naive_bias = results["naive_error"].mean()
    print(f"\n  --- Naive Baseline (Q3 + avg Q4 = {avg_q4:.1f}) ---")
    print(f"  MAE:               {naive_mae:.1f} pts")
    print(f"  Bias:              {naive_bias:+.1f} pts")
    print(f"  Improvement:       {naive_mae - mae:+.1f} pts MAE")

    # Coverage
    print(f"\n  --- Coverage (% of actuals within band) ---")
    for mult, label in [(1.0, "1.0σ (expect ~68%)"),
                        (1.5, "1.5σ (expect ~87%)"),
                        (2.0, "2.0σ (expect ~95%)")]:
        lo = results["projected_total"] - mult * results["remaining_std"]
        hi = results["projected_total"] + mult * results["remaining_std"]
        within = ((results["actual_total"] >= lo) & (results["actual_total"] <= hi)).mean()
        print(f"  {label}:  {within * 100:.1f}%")

    # Accuracy by pace bucket
    print(f"\n  --- Accuracy by Pace Factor (Q1-Q3) ---")
    pace_bins = [0, 0.90, 0.95, 1.05, 1.10, 999]
    pace_labels = ["<0.90 (slow)", "0.90-0.95", "0.95-1.05 (normal)", "1.05-1.10", ">1.10 (fast)"]
    results["pace_bucket"] = pd.cut(results["pace"], bins=pace_bins, labels=pace_labels)
    for label in pace_labels:
        subset = results[results["pace_bucket"] == label]
        if len(subset) > 0:
            print(f"  {label:25s}: n={len(subset):4d}  MAE={subset['abs_error'].mean():.1f}  "
                  f"bias={subset['error'].mean():+.1f}")

    # Accuracy by Q3 margin
    print(f"\n  --- Accuracy by Q3 Margin ---")
    margin_bins = [0, 5, 10, 15, 20, 999]
    margin_labels = ["0-4 (tight)", "5-9", "10-14", "15-19", "20+ (blowout)"]
    results["margin_bucket"] = pd.cut(results["margin_at_q3"], bins=margin_bins,
                                       labels=margin_labels, right=False)
    for label in margin_labels:
        subset = results[results["margin_bucket"] == label]
        if len(subset) > 0:
            print(f"  {label:20s}: n={len(subset):4d}  MAE={subset['abs_error'].mean():.1f}  "
                  f"bias={subset['error'].mean():+.1f}")

    # Error percentiles
    print(f"\n  --- Error Percentiles ---")
    for p in [10, 25, 50, 75, 90]:
        val = np.percentile(results["abs_error"], p)
        print(f"  P{p:2d}:  {val:.1f} pts")


# ---------------------------------------------------------------------------
# Grid search for optimal pace alpha
# ---------------------------------------------------------------------------

def grid_search_alpha(last_n: int = 20, min_games: int = 10) -> None:
    """Test alpha values to validate pace dampening parameter.

    For each alpha in [0.0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5]:
      - Run improved backtest
      - Report overall MAE, bias, and per-pace-bucket bias
    """
    alphas = [0.0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5]

    pace_bins = [0, 0.90, 0.95, 1.05, 1.10, 999]
    pace_labels = ["<0.90", "0.90-0.95", "0.95-1.05", "1.05-1.10", ">1.10"]

    print("\n" + "=" * 80)
    print("GRID SEARCH: Pace Dampening Alpha")
    print("=" * 80)

    header = f"{'alpha':>6s} | {'MAE':>6s} {'Bias':>7s} | "
    header += " | ".join(f"{l:>10s}" for l in pace_labels)
    print(header)
    print("-" * len(header))

    for alpha in alphas:
        # Suppress per-run output
        import io
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            results = run_backtest_improved(
                last_n=last_n, min_games=min_games,
                pace_alpha=alpha, sigma_inflation=1.45,
            )

        if results.empty:
            continue

        mae = results["abs_error"].mean()
        bias = results["error"].mean()

        results["pace_bucket"] = pd.cut(
            results["pace"], bins=pace_bins, labels=pace_labels
        )

        bucket_biases = []
        for label in pace_labels:
            subset = results[results["pace_bucket"] == label]
            if len(subset) > 0:
                bucket_biases.append(f"{subset['error'].mean():+7.1f}")
            else:
                bucket_biases.append(f"{'N/A':>7s}")

        row = f"{alpha:6.2f} | {mae:6.1f} {bias:+7.1f} | "
        row += " | ".join(f"{b:>10s}" for b in bucket_biases)
        print(row)

    print()
    print("  alpha=0.0 is the raw model (no dampening).")
    print("  alpha=1.0 would be full pace pass-through (same as raw model).")
    print("  Lower alpha → more dampening → less pace bias at extremes.")


# ---------------------------------------------------------------------------
# P&L simulation
# ---------------------------------------------------------------------------

def simulate_pnl(
    results: pd.DataFrame,
    edge_threshold: float = 0.10,
    max_pace_deviation: float = 0.15,
    bet_size_cents: int = 1000,
) -> pd.DataFrame:
    """Simulate betting P&L against synthetic market lines.

    For each game:
      1. Synthetic line = pre-game expected total (sum of Q1-Q4 avgs for both teams)
         proxied by projected_total at pace=1.0: h1_score + expected_h2
      2. P(over) from dampened projection + calibrated σ
      3. Edge = |P(over) - 0.50|; side = over if P > 0.50
      4. Bet if edge > threshold AND |raw_pace - 1.0| ≤ max_pace_deviation
      5. Resolution: actual_total vs synthetic_line
      6. P&L: win pays (100 - price_cents), lose costs price_cents

    Returns DataFrame of bets with P&L columns.
    """
    if results.empty:
        print("No results for P&L simulation.")
        return pd.DataFrame()

    price_cents = 50  # assume market at 50¢

    bets = []
    for _, row in results.iterrows():
        raw_pace = row["pace"]
        # Filter: skip extreme pace games
        if abs(raw_pace - 1.0) > max_pace_deviation:
            continue

        projected = row["projected_total"]
        std = row.get("calibrated_std", row.get("remaining_std", 10.0))
        actual = row["actual_total"]

        # Synthetic line: what a pre-game total would be
        # h1_score + expected_h2 at pace=1.0
        # We can derive it: projected_total = h1_score + expected_h2 * dampened_pace
        # At dampened_pace=1: line = h1_score + expected_h2
        dampened_pace = row.get("dampened_pace", 1.0)
        if dampened_pace > 0:
            expected_h2 = (projected - row["h1_score"]) / dampened_pace
        else:
            expected_h2 = projected - row["h1_score"]
        synthetic_line = row["h1_score"] + expected_h2

        # P(over line)
        if std <= 0:
            continue
        z = (projected - synthetic_line) / std
        # Normal CDF approximation
        from statistics import NormalDist
        p_over = NormalDist(0, 1).cdf(z)

        edge = abs(p_over - 0.50)
        if edge < edge_threshold:
            continue

        side = "over" if p_over > 0.50 else "under"
        # Resolution
        if side == "over":
            won = actual > synthetic_line
        else:
            won = actual < synthetic_line

        pnl = (100 - price_cents) if won else -price_cents

        bets.append({
            "game_id": row["game_id"],
            "game_date": row["game_date"],
            "home": row["home"],
            "away": row["away"],
            "side": side,
            "edge_pct": round(edge * 100, 1),
            "p_over": round(p_over, 3),
            "projected": projected,
            "synthetic_line": round(synthetic_line, 1),
            "actual": actual,
            "pace": raw_pace,
            "won": won,
            "pnl_cents": pnl,
        })

    bets_df = pd.DataFrame(bets)

    if bets_df.empty:
        print("No bets triggered at edge_threshold=%.0f%%" % (edge_threshold * 100))
        return bets_df

    total_bets = len(bets_df)
    wins = bets_df["won"].sum()
    total_pnl = bets_df["pnl_cents"].sum()
    total_risked = total_bets * price_cents
    roi = total_pnl / total_risked * 100 if total_risked > 0 else 0

    print("\n" + "=" * 60)
    print("P&L SIMULATION RESULTS")
    print("=" * 60)
    print(f"  Edge threshold:    {edge_threshold * 100:.0f}%")
    print(f"  Max pace deviation: {max_pace_deviation:.2f}")
    print(f"  Price assumed:     {price_cents}¢")
    print(f"  Total bets:        {total_bets}")
    print(f"  Wins:              {wins} ({wins/total_bets*100:.1f}%)")
    print(f"  Losses:            {total_bets - wins}")
    print(f"  Total P&L:         {total_pnl:+d}¢ (${total_pnl/100:+.2f})")
    print(f"  ROI:               {roi:+.1f}%")

    # Breakdown by side
    for side in ["over", "under"]:
        subset = bets_df[bets_df["side"] == side]
        if len(subset) > 0:
            s_wins = subset["won"].sum()
            s_pnl = subset["pnl_cents"].sum()
            print(f"\n  {side.upper()}:")
            print(f"    Bets: {len(subset)}, Wins: {s_wins} ({s_wins/len(subset)*100:.1f}%), "
                  f"P&L: {s_pnl:+d}¢")

    # Breakdown by edge bucket
    print(f"\n  --- By Edge Size ---")
    edge_bins = [0, 15, 20, 30, 100]
    edge_labels = ["10-15%", "15-20%", "20-30%", "30%+"]
    bets_df["edge_bucket"] = pd.cut(bets_df["edge_pct"], bins=edge_bins, labels=edge_labels)
    for label in edge_labels:
        subset = bets_df[bets_df["edge_bucket"] == label]
        if len(subset) > 0:
            s_wins = subset["won"].sum()
            s_pnl = subset["pnl_cents"].sum()
            print(f"    {label:10s}: n={len(subset):3d}  win={s_wins/len(subset)*100:.0f}%  "
                  f"P&L={s_pnl:+d}¢")

    return bets_df


# ---------------------------------------------------------------------------
# Q4 P&L simulation
# ---------------------------------------------------------------------------

def simulate_pnl_q4(
    results: pd.DataFrame,
    edge_threshold: float = 0.06,
    max_pace_deviation: float = 0.15,
    bet_size_cents: int = 1000,
    no_plot: bool = False,
) -> pd.DataFrame:
    """Simulate betting P&L for Q4-start projections.

    Q4 has 3 quarters of pace data and only 1 quarter of uncertainty,
    so uses a lower default edge threshold (6% vs 10% for halftime).

    For each game:
      1. Synthetic line = q3_score + expected_q4 at pace=1.0
      2. P(over) from dampened projection + calibrated sigma
      3. Edge = |P(over) - 0.50|; side = over if P > 0.50
      4. Bet if edge > threshold AND |raw_pace - 1.0| <= max_pace_deviation
      5. Resolution: actual_total vs synthetic_line
      6. P&L: win pays (100 - price_cents), lose costs price_cents
    """
    from statistics import NormalDist

    if results.empty:
        print("No results for Q4 P&L simulation.")
        return pd.DataFrame()

    price_cents = 50  # assume market at 50c

    bets = []
    for _, row in results.iterrows():
        raw_pace = row["pace"]
        if abs(raw_pace - 1.0) > max_pace_deviation:
            continue

        projected = row["projected_total"]
        std = row.get("calibrated_std", row.get("remaining_std", 7.0))
        actual = row["actual_total"]
        q3_score = row["q3_score"]

        # Synthetic line: q3_score + expected_q4 at pace=1.0
        dampened_pace = row.get("dampened_pace", 1.0)
        if dampened_pace > 0:
            expected_q4 = (projected - q3_score) / dampened_pace
        else:
            expected_q4 = projected - q3_score
        synthetic_line = q3_score + expected_q4

        if std <= 0:
            continue
        z = (projected - synthetic_line) / std
        p_over = NormalDist(0, 1).cdf(z)

        edge = abs(p_over - 0.50)
        if edge < edge_threshold:
            continue

        side = "over" if p_over > 0.50 else "under"
        if side == "over":
            won = actual > synthetic_line
        else:
            won = actual < synthetic_line

        pnl = (100 - price_cents) if won else -price_cents

        bets.append({
            "game_id": row["game_id"],
            "game_date": row["game_date"],
            "home": row["home"],
            "away": row["away"],
            "side": side,
            "edge_pct": round(edge * 100, 1),
            "p_over": round(p_over, 3),
            "projected": projected,
            "synthetic_line": round(synthetic_line, 1),
            "actual": actual,
            "pace": raw_pace,
            "margin_at_q3": row.get("margin_at_q3", 0),
            "won": won,
            "pnl_cents": pnl,
        })

    bets_df = pd.DataFrame(bets)

    if bets_df.empty:
        print("No bets triggered at edge_threshold=%.0f%%" % (edge_threshold * 100))
        return bets_df

    # --- Summary ---
    total_bets = len(bets_df)
    wins = int(bets_df["won"].sum())
    total_pnl = int(bets_df["pnl_cents"].sum())
    total_risked = total_bets * price_cents
    roi = total_pnl / total_risked * 100 if total_risked > 0 else 0

    print("\n" + "=" * 60)
    print("Q4 P&L SIMULATION RESULTS")
    print("=" * 60)
    print(f"  Edge threshold:     {edge_threshold * 100:.0f}%")
    print(f"  Max pace deviation: {max_pace_deviation:.2f}")
    print(f"  Price assumed:      {price_cents}¢")
    print(f"  Total bets:         {total_bets}")
    print(f"  Wins:               {wins} ({wins/total_bets*100:.1f}%)")
    print(f"  Losses:             {total_bets - wins}")
    print(f"  Total P&L:          {total_pnl:+d}¢ (${total_pnl/100:+.2f})")
    print(f"  ROI:                {roi:+.1f}%")

    # --- Breakdown by side ---
    for side in ["over", "under"]:
        subset = bets_df[bets_df["side"] == side]
        if len(subset) > 0:
            s_wins = int(subset["won"].sum())
            s_pnl = int(subset["pnl_cents"].sum())
            print(f"\n  {side.upper()}:")
            print(f"    Bets: {len(subset)}, Wins: {s_wins} ({s_wins/len(subset)*100:.1f}%), "
                  f"P&L: {s_pnl:+d}¢")

    # --- Breakdown by edge size ---
    print(f"\n  --- By Edge Size ---")
    edge_bins = [0, 10, 15, 20, 100]
    edge_labels = ["6-10%", "10-15%", "15-20%", "20%+"]
    bets_df["edge_bucket"] = pd.cut(bets_df["edge_pct"], bins=edge_bins, labels=edge_labels)
    for label in edge_labels:
        subset = bets_df[bets_df["edge_bucket"] == label]
        if len(subset) > 0:
            s_wins = int(subset["won"].sum())
            s_pnl = int(subset["pnl_cents"].sum())
            print(f"    {label:10s}: n={len(subset):3d}  win={s_wins/len(subset)*100:.0f}%  "
                  f"P&L={s_pnl:+d}¢")

    # --- Breakdown by Q3 margin ---
    print(f"\n  --- By Q3 Margin ---")
    margin_bins = [0, 5, 10, 15, 20, 999]
    margin_labels = ["0-4", "5-9", "10-14", "15-19", "20+"]
    bets_df["margin_bucket"] = pd.cut(bets_df["margin_at_q3"], bins=margin_bins,
                                       labels=margin_labels, right=False)
    for label in margin_labels:
        subset = bets_df[bets_df["margin_bucket"] == label]
        if len(subset) > 0:
            s_wins = int(subset["won"].sum())
            s_pnl = int(subset["pnl_cents"].sum())
            print(f"    {label:10s}: n={len(subset):3d}  win={s_wins/len(subset)*100:.0f}%  "
                  f"P&L={s_pnl:+d}¢")

    # --- Cumulative P&L ---
    bets_df = bets_df.sort_values("game_date").reset_index(drop=True)
    bets_df["cumulative_pnl"] = bets_df["pnl_cents"].cumsum()
    print(f"\n  --- Cumulative P&L ---")
    print(f"    Peak:  {bets_df['cumulative_pnl'].max():+d}¢")
    print(f"    Trough: {bets_df['cumulative_pnl'].min():+d}¢")
    print(f"    Final: {bets_df['cumulative_pnl'].iloc[-1]:+d}¢")

    # --- Chart ---
    if not no_plot:
        os.makedirs(CHART_DIR, exist_ok=True)
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(range(len(bets_df)), bets_df["cumulative_pnl"] / 100,
                color="#2196F3", linewidth=1.5)
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.fill_between(range(len(bets_df)),
                        bets_df["cumulative_pnl"] / 100, 0,
                        alpha=0.15, color="#2196F3")
        ax.set_xlabel("Bet #")
        ax.set_ylabel("Cumulative P&L ($)")
        ax.set_title(f"Q4 P&L — edge≥{edge_threshold*100:.0f}%, pace±{max_pace_deviation:.0%} "
                     f"| {total_bets} bets, {wins/total_bets*100:.0f}% win, ROI {roi:+.1f}%")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        out = os.path.join(CHART_DIR, "pnl_q4.png")
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"\n  Saved {out}")

    return bets_df


# ---------------------------------------------------------------------------
# Q4 parameter grid search
# ---------------------------------------------------------------------------

def grid_search_q4(
    last_n: int = 20,
    min_games: int = 10,
    pace_alpha: float = 0.25,
    sigma_inflation: float = 1.45,
) -> None:
    """Grid search over Q4 P&L parameters: edge_threshold and max_pace_deviation.

    Tests all combinations and reports #bets, win rate, total P&L, ROI.
    """
    import contextlib
    import io

    edge_thresholds = [0.04, 0.06, 0.08, 0.10, 0.12, 0.15]
    pace_deviations = [0.10, 0.12, 0.15, 0.20]

    # Build Q4 results once
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        results = run_backtest_q4(
            last_n=last_n, min_games=min_games,
            pace_alpha=pace_alpha, sigma_inflation=sigma_inflation,
        )

    if results.empty:
        print("No Q4 backtest results for grid search.")
        return

    print("\n" + "=" * 80)
    print("GRID SEARCH: Q4 P&L Parameters")
    print("=" * 80)

    header = f"{'edge%':>6s} {'pace±':>6s} | {'#bets':>6s} {'win%':>6s} {'P&L¢':>8s} {'ROI%':>7s}"
    print(header)
    print("-" * len(header))

    best_roi = -999.0
    best_combo = ("", "")
    best_line = ""

    for edge_t in edge_thresholds:
        for pace_d in pace_deviations:
            buf2 = io.StringIO()
            with contextlib.redirect_stdout(buf2):
                bets = simulate_pnl_q4(
                    results,
                    edge_threshold=edge_t,
                    max_pace_deviation=pace_d,
                    no_plot=True,
                )

            if bets.empty:
                line = f"{edge_t*100:5.0f}% {pace_d:5.0%} |      0    N/A       N/A     N/A"
                print(line)
                continue

            n = len(bets)
            wins = int(bets["won"].sum())
            win_pct = wins / n * 100
            total_pnl = int(bets["pnl_cents"].sum())
            roi = total_pnl / (n * 50) * 100

            line = f"{edge_t*100:5.0f}% {pace_d:5.0%} | {n:6d} {win_pct:5.1f}% {total_pnl:+8d} {roi:+6.1f}%"
            print(line)

            if roi > best_roi:
                best_roi = roi
                best_combo = (edge_t, pace_d)
                best_line = line

    print()
    if best_combo[0]:
        print(f"  Best: edge={best_combo[0]*100:.0f}%, pace±{best_combo[1]:.0%}")
        print(f"  {best_line}")


# ---------------------------------------------------------------------------
# Margin sweep: win rate at projected - margin
# ---------------------------------------------------------------------------

def sweep_margin_winrate(
    results: pd.DataFrame,
    max_pace_deviation: float = 0.15,
) -> None:
    """For each margin, bet OVER on line = projected_total - margin.

    Shows win rate and implied fair price for each margin value.
    At margin=0 you're betting the projected total is too low (~50%).
    As margin grows, the line drops and win rate rises, but the contract
    gets more expensive (higher fair price).
    """
    if results.empty:
        print("No results for margin sweep.")
        return

    # Filter by pace
    filtered = results[results["pace"].apply(lambda p: abs(p - 1.0) <= max_pace_deviation)].copy()
    n = len(filtered)
    if n == 0:
        print("No games within pace filter.")
        return

    margins = list(range(0, 22, 2))

    print("\n" + "=" * 60)
    print("MARGIN SWEEP: Win Rate at Projected - Margin (OVER)")
    print("=" * 60)
    print(f"  Games: {n} (pace within ±{max_pace_deviation:.0%})")
    print()

    header = f"{'margin':>7s} | {'line':>8s} | {'wins':>5s}/{n:<5d} | {'win%':>6s} | {'fair¢':>6s} | {'P&L @50¢':>9s}"
    print(header)
    print("-" * len(header))

    for margin in margins:
        lines = filtered["projected_total"] - margin
        wins = (filtered["actual_total"] > lines).sum()
        win_pct = wins / n * 100
        fair_cents = round(win_pct)  # fair price in cents
        # P&L if buying at 50c: win pays +50c, lose costs -50c
        pnl = wins * 50 - (n - wins) * 50

        avg_line = lines.mean()
        print(f"  {margin:5d}  | {avg_line:7.1f}  | {wins:5d}/{n:<5d} | {win_pct:5.1f}% | {fair_cents:5d}¢ | {pnl:+8d}¢")

    # Also show UNDER direction
    print()
    print("=" * 60)
    print("MARGIN SWEEP: Win Rate at Projected + Margin (UNDER)")
    print("=" * 60)
    print(f"  Games: {n} (pace within ±{max_pace_deviation:.0%})")
    print()

    header = f"{'margin':>7s} | {'line':>8s} | {'wins':>5s}/{n:<5d} | {'win%':>6s} | {'fair¢':>6s} | {'P&L @50¢':>9s}"
    print(header)
    print("-" * len(header))

    for margin in margins:
        lines = filtered["projected_total"] + margin
        wins = (filtered["actual_total"] < lines).sum()
        win_pct = wins / n * 100
        fair_cents = round(win_pct)
        pnl = wins * 50 - (n - wins) * 50

        avg_line = lines.mean()
        print(f"  {margin:5d}  | {avg_line:7.1f}  | {wins:5d}/{n:<5d} | {win_pct:5.1f}% | {fair_cents:5d}¢ | {pnl:+8d}¢")


# ---------------------------------------------------------------------------
# High-confidence betting simulation
# ---------------------------------------------------------------------------

def simulate_high_confidence(
    results: pd.DataFrame,
    confidence: float = 0.90,
    max_pace_deviation: float = 0.15,
    no_plot: bool = False,
) -> pd.DataFrame:
    """Simulate high-confidence betting using per-game sigma.

    Instead of a fixed margin, computes the exact line where the model has
    >= `confidence` probability of winning:
      OVER line  = projected_total - z * calibrated_std
      UNDER line = projected_total + z * calibrated_std

    Reports true win rate and P&L at realistic market prices (75-95¢).
    """
    from statistics import NormalDist

    if results.empty:
        print("No results for high-confidence simulation.")
        return pd.DataFrame()

    z = NormalDist().inv_cdf(confidence)

    # Filter by pace
    filtered = results[
        results["pace"].apply(lambda p: abs(p - 1.0) <= max_pace_deviation)
    ].copy()
    n = len(filtered)
    if n == 0:
        print("No games within pace filter.")
        return pd.DataFrame()

    # --- Build per-game bets for OVER and UNDER ---
    over_bets = []
    under_bets = []
    for _, row in filtered.iterrows():
        projected = row["projected_total"]
        std = row.get("calibrated_std", row.get("remaining_std", 10.0))
        actual = row["actual_total"]
        if std <= 0:
            continue

        base = {
            "game_id": row["game_id"],
            "game_date": row["game_date"],
            "home": row["home"],
            "away": row["away"],
            "projected": projected,
            "actual": actual,
            "calibrated_std": std,
            "pace": row["pace"],
        }

        # OVER: line = projected - z * std
        over_line = projected - z * std
        over_won = actual > over_line
        over_bets.append({
            **base,
            "side": "over",
            "line": round(over_line, 1),
            "distance": round(z * std, 1),
            "won": over_won,
        })

        # UNDER: line = projected + z * std
        under_line = projected + z * std
        under_won = actual < under_line
        under_bets.append({
            **base,
            "side": "under",
            "line": round(under_line, 1),
            "distance": round(z * std, 1),
            "won": under_won,
        })

    over_df = pd.DataFrame(over_bets)
    under_df = pd.DataFrame(under_bets)

    prices = [75, 78, 80, 82, 85, 88, 90, 92]

    def _print_direction(label: str, bets_df: pd.DataFrame) -> None:
        if bets_df.empty:
            return
        total = len(bets_df)
        wins = int(bets_df["won"].sum())
        win_pct = wins / total * 100
        avg_line = bets_df["line"].mean()
        avg_dist = bets_df["distance"].mean()

        print(f"\n  --- {label} (line = projected {'−' if label == 'OVER' else '+'} z×σ) ---")
        print(f"  Total bets:      {total}")
        print(f"  Wins:            {wins} ({win_pct:.1f}%)")
        print(f"  Avg line:        {avg_line:.1f}")
        print(f"  Avg z×σ cushion: {avg_dist:.1f} pts")

        # P&L at various prices
        print(f"\n  {'price':>6s} | {'EV/bet':>8s} | {'total P&L':>10s} | {'ROI':>7s}")
        print(f"  {'-'*6}-+-{'-'*8}-+-{'-'*10}-+-{'-'*7}")
        lose_pct = 100.0 - win_pct
        for price in prices:
            ev = (win_pct / 100) * (100 - price) - (lose_pct / 100) * price
            total_pnl = wins * (100 - price) - (total - wins) * price
            risked = total * price
            roi = total_pnl / risked * 100 if risked > 0 else 0
            print(f"  {price:5d}¢ | {ev:+7.1f}¢ | {total_pnl:+9d}¢ | {roi:+6.1f}%")

        # Breakdown by Q3/halftime margin
        margin_col = "margin_at_q3" if "margin_at_q3" in bets_df.columns else None
        if margin_col is None and "margin_at_half" in results.columns:
            # Merge margin from results
            margin_col = "margin_at_half"
        if margin_col:
            # Join margin from results onto bets_df
            margin_map = results.set_index("game_id")["margin_at_half"] if "margin_at_half" in results.columns else None
            if margin_map is not None and "margin_at_half" not in bets_df.columns:
                bets_df = bets_df.copy()
                bets_df["margin_at_half"] = bets_df["game_id"].map(margin_map)
                margin_col = "margin_at_half"

        if margin_col and margin_col in bets_df.columns:
            print(f"\n  By margin at checkpoint:")
            margin_bins = [0, 5, 10, 15, 20, 999]
            margin_labels = ["0-4", "5-9", "10-14", "15-19", "20+"]
            bets_df = bets_df.copy()
            bets_df["margin_bucket"] = pd.cut(
                bets_df[margin_col], bins=margin_bins,
                labels=margin_labels, right=False,
            )
            for ml in margin_labels:
                subset = bets_df[bets_df["margin_bucket"] == ml]
                if len(subset) > 0:
                    sw = int(subset["won"].sum())
                    print(f"    {ml:10s}: n={len(subset):4d}  win={sw/len(subset)*100:.1f}%")

        # Breakdown by calibrated_std bucket
        print(f"\n  By uncertainty (calibrated_std):")
        std_vals = bets_df["calibrated_std"]
        lo_thresh = std_vals.quantile(0.33)
        hi_thresh = std_vals.quantile(0.67)
        for bucket_label, mask in [
            (f"low  (<{lo_thresh:.1f})", std_vals < lo_thresh),
            (f"med  ({lo_thresh:.1f}-{hi_thresh:.1f})", (std_vals >= lo_thresh) & (std_vals < hi_thresh)),
            (f"high (>{hi_thresh:.1f})", std_vals >= hi_thresh),
        ]:
            subset = bets_df[mask]
            if len(subset) > 0:
                sw = int(subset["won"].sum())
                print(f"    {bucket_label:22s}: n={len(subset):4d}  win={sw/len(subset)*100:.1f}%")

    # --- Print results ---
    print("\n" + "=" * 65)
    print(f"HIGH-CONFIDENCE BETTING SIMULATION (z={z:.4f}, conf={confidence:.0%})")
    print("=" * 65)
    print(f"  Games (pace ±{max_pace_deviation:.0%}): {n}")

    _print_direction("OVER", over_df)
    _print_direction("UNDER", under_df)

    # --- Confidence level sweep ---
    print(f"\n  {'='*65}")
    print(f"  CONFIDENCE LEVEL COMPARISON (at 85¢ contract price)")
    print(f"  {'='*65}")
    sweep_levels = [0.85, 0.90, 0.95]
    print(f"  {'conf':>5s} | {'z':>6s} | {'dir':>5s} | {'win%':>6s} | {'EV/bet':>8s} | {'ROI':>7s}")
    print(f"  {'-'*5}-+-{'-'*6}-+-{'-'*5}-+-{'-'*6}-+-{'-'*8}-+-{'-'*7}")
    ref_price = 85
    for conf_level in sweep_levels:
        z_l = NormalDist().inv_cdf(conf_level)
        for direction in ["over", "under"]:
            wins_l = 0
            total_l = 0
            for _, row in filtered.iterrows():
                projected = row["projected_total"]
                std = row.get("calibrated_std", row.get("remaining_std", 10.0))
                actual = row["actual_total"]
                if std <= 0:
                    continue
                total_l += 1
                if direction == "over":
                    line_l = projected - z_l * std
                    if actual > line_l:
                        wins_l += 1
                else:
                    line_l = projected + z_l * std
                    if actual < line_l:
                        wins_l += 1
            if total_l == 0:
                continue
            wp = wins_l / total_l * 100
            ev = (wp / 100) * (100 - ref_price) - ((100 - wp) / 100) * ref_price
            roi = ((wins_l * (100 - ref_price) - (total_l - wins_l) * ref_price)
                   / (total_l * ref_price) * 100)
            print(f"  {conf_level:4.0%}  | {z_l:6.4f} | {direction:>5s} | {wp:5.1f}% | {ev:+7.1f}¢ | {roi:+6.1f}%")

    # --- Chart: cumulative P&L at 85¢ for OVER ---
    if not no_plot and not over_df.empty:
        chart_price = 85
        over_sorted = over_df.sort_values("game_date").reset_index(drop=True)
        over_sorted["pnl"] = over_sorted["won"].apply(
            lambda w: (100 - chart_price) if w else -chart_price
        )
        over_sorted["cum_pnl"] = over_sorted["pnl"].cumsum()

        os.makedirs(CHART_DIR, exist_ok=True)
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(range(len(over_sorted)), over_sorted["cum_pnl"] / 100,
                color="#2196F3", linewidth=1.5)
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.fill_between(range(len(over_sorted)),
                        over_sorted["cum_pnl"] / 100, 0,
                        alpha=0.15, color="#2196F3")
        wins_o = int(over_sorted["won"].sum())
        total_o = len(over_sorted)
        roi_o = over_sorted["cum_pnl"].iloc[-1] / (total_o * chart_price) * 100
        ax.set_xlabel("Bet #")
        ax.set_ylabel("Cumulative P&L ($)")
        ax.set_title(
            f"High-Conf OVER @ {chart_price}¢ — conf={confidence:.0%}, "
            f"{total_o} bets, {wins_o/total_o*100:.0f}% win, ROI {roi_o:+.1f}%"
        )
        ax.grid(alpha=0.3)
        fig.tight_layout()
        out = os.path.join(CHART_DIR, "pnl_high_conf.png")
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"\n  Saved {out}")

    return over_df


# ---------------------------------------------------------------------------
# Results summary
# ---------------------------------------------------------------------------

def print_results(results: pd.DataFrame) -> None:
    """Print backtest summary statistics."""
    if results.empty:
        print("No results to display.")
        return

    n = len(results)
    mae = results["abs_error"].mean()
    bias = results["error"].mean()
    rmse = math.sqrt((results["error"] ** 2).mean())
    median_ae = results["abs_error"].median()

    # Naive baseline: double the halftime score
    results = results.copy()
    results["naive_projected"] = results["h1_score"] * 2
    results["naive_error"] = results["naive_projected"] - results["actual_total"]
    naive_mae = results["naive_error"].abs().mean()
    naive_bias = results["naive_error"].mean()

    print("\n" + "=" * 60)
    print("HALFTIME PROJECTION BACKTEST RESULTS")
    print("=" * 60)
    print(f"  Games backtested:  {n}")
    print(f"  Date range:        {results['game_date'].min()} to {results['game_date'].max()}")

    print(f"\n  --- Model Performance ---")
    print(f"  MAE:               {mae:.1f} pts")
    print(f"  Median AE:         {median_ae:.1f} pts")
    print(f"  RMSE:              {rmse:.1f} pts")
    print(f"  Bias:              {bias:+.1f} pts")

    print(f"\n  --- Naive Baseline (2x halftime) ---")
    print(f"  MAE:               {naive_mae:.1f} pts")
    print(f"  Bias:              {naive_bias:+.1f} pts")
    print(f"  Improvement:       {naive_mae - mae:+.1f} pts MAE")

    # Coverage at 1σ, 1.5σ, 2σ
    print(f"\n  --- Coverage (% of actuals within band) ---")
    for mult, label in [(1.0, "1.0σ (expect ~68%)"),
                        (1.5, "1.5σ (expect ~87%)"),
                        (2.0, "2.0σ (expect ~95%)")]:
        lo = results["projected_total"] - mult * results["remaining_std"]
        hi = results["projected_total"] + mult * results["remaining_std"]
        within = ((results["actual_total"] >= lo) & (results["actual_total"] <= hi)).mean()
        print(f"  {label}:  {within * 100:.1f}%")

    # Accuracy by pace bucket
    print(f"\n  --- Accuracy by Pace Factor ---")
    pace_bins = [0, 0.90, 0.95, 1.05, 1.10, 999]
    pace_labels = ["<0.90 (slow)", "0.90-0.95", "0.95-1.05 (normal)", "1.05-1.10", ">1.10 (fast)"]
    results["pace_bucket"] = pd.cut(results["pace"], bins=pace_bins, labels=pace_labels)
    for label in pace_labels:
        subset = results[results["pace_bucket"] == label]
        if len(subset) > 0:
            print(f"  {label:25s}: n={len(subset):4d}  MAE={subset['abs_error'].mean():.1f}  "
                  f"bias={subset['error'].mean():+.1f}")

    # Accuracy by halftime margin
    print(f"\n  --- Accuracy by Halftime Margin ---")
    margin_bins = [0, 5, 10, 15, 20, 999]
    margin_labels = ["0-4 (tight)", "5-9", "10-14", "15-19", "20+ (blowout)"]
    results["margin_bucket"] = pd.cut(results["margin_at_half"], bins=margin_bins,
                                       labels=margin_labels, right=False)
    for label in margin_labels:
        subset = results[results["margin_bucket"] == label]
        if len(subset) > 0:
            print(f"  {label:20s}: n={len(subset):4d}  MAE={subset['abs_error'].mean():.1f}  "
                  f"bias={subset['error'].mean():+.1f}")

    # Error distribution percentiles
    print(f"\n  --- Error Percentiles ---")
    for p in [10, 25, 50, 75, 90]:
        val = np.percentile(results["abs_error"], p)
        print(f"  P{p:2d}:  {val:.1f} pts")


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def plot_results(results: pd.DataFrame) -> None:
    """Generate backtest diagnostic charts."""
    if results.empty:
        print("No results to plot.")
        return

    os.makedirs(CHART_DIR, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Projected vs Actual scatter
    ax = axes[0, 0]
    ax.scatter(results["actual_total"], results["projected_total"],
               alpha=0.3, s=12, color="#5C6BC0")
    lo = min(results["actual_total"].min(), results["projected_total"].min()) - 5
    hi = max(results["actual_total"].max(), results["projected_total"].max()) + 5
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.5, label="Perfect")
    ax.set_xlabel("Actual Game Total")
    ax.set_ylabel("Projected Game Total")
    ax.set_title("Projected vs Actual")
    ax.legend()
    ax.grid(alpha=0.3)

    # 2. Error distribution histogram
    ax = axes[0, 1]
    ax.hist(results["error"], bins=40, color="#4CAF50", alpha=0.7, edgecolor="white")
    ax.axvline(0, color="red", linestyle="--", linewidth=1.5)
    ax.axvline(results["error"].mean(), color="orange", linestyle="-", linewidth=1.5,
               label=f"bias={results['error'].mean():+.1f}")
    ax.set_xlabel("Error (projected - actual)")
    ax.set_ylabel("Frequency")
    ax.set_title("Error Distribution")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # 3. Error vs pace factor
    ax = axes[1, 0]
    ax.scatter(results["pace"], results["error"], alpha=0.3, s=12, color="#FF9800")
    ax.axhline(0, color="red", linestyle="--", linewidth=1)
    # Trend line
    z = np.polyfit(results["pace"], results["error"], 1)
    x_range = np.linspace(results["pace"].min(), results["pace"].max(), 50)
    ax.plot(x_range, np.polyval(z, x_range), color="black", linewidth=2,
            label=f"slope={z[0]:.1f}")
    ax.set_xlabel("Pace Factor")
    ax.set_ylabel("Error (projected - actual)")
    ax.set_title("Error vs Pace Factor")
    ax.legend()
    ax.grid(alpha=0.3)

    # 4. Coverage calibration
    ax = axes[1, 1]
    sigmas = np.arange(0.5, 3.1, 0.25)
    expected = []
    actual_coverage = []
    for s in sigmas:
        # Expected coverage for normal distribution
        from statistics import NormalDist
        exp = NormalDist(0, 1).cdf(s) - NormalDist(0, 1).cdf(-s)
        expected.append(exp * 100)

        lo = results["projected_total"] - s * results["remaining_std"]
        hi = results["projected_total"] + s * results["remaining_std"]
        cov = ((results["actual_total"] >= lo) & (results["actual_total"] <= hi)).mean()
        actual_coverage.append(cov * 100)

    ax.plot(expected, actual_coverage, "o-", color="#2196F3", linewidth=2, label="Model")
    ax.plot([0, 100], [0, 100], "r--", linewidth=1.5, label="Perfect calibration")
    ax.set_xlabel("Expected Coverage (%)")
    ax.set_ylabel("Actual Coverage (%)")
    ax.set_title("Coverage Calibration")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.suptitle("Halftime Game Total Backtest", fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = os.path.join(CHART_DIR, "backtest_halftime.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\n  Saved {out}")


# ---------------------------------------------------------------------------
# Comparison: baseline vs opp/b2b adjustments
# ---------------------------------------------------------------------------

def run_comparison(
    last_n: int = 20,
    min_games: int = 10,
    pace_alpha: float = 0.25,
    sigma_inflation: float = 1.45,
    b2b_penalty: float = 1.5,
    blowout_threshold: float = 15.0,
    blowout_rate: float = 0.1,
) -> None:
    """Run configs side-by-side (baseline, +opp, +b2b, +blowout, +all) for halftime and Q4."""
    import contextlib
    import io

    # (name, opp, b2b, blowout)
    configs = [
        ("Baseline",       False, False, False),
        ("+ Opp-Adjusted", True,  False, False),
        ("+ B2B",          False, True,  False),
        ("+ Blowout",      False, False, True),
        ("+ Opp+B2B",      True,  True,  False),
        ("+ Opp+Blowout",  True,  False, True),
        ("+ All",          True,  True,  True),
    ]

    # Q4 uses higher blowout threshold (more margin data available)
    q4_blowout_threshold = max(blowout_threshold, 20.0)

    for checkpoint, runner_fn, bo_thresh in [
        ("HALFTIME", run_backtest_improved, blowout_threshold),
        ("Q4 START", run_backtest_q4, q4_blowout_threshold),
    ]:
        print(f"\n{'=' * 75}")
        print(f"  {checkpoint} COMPARISON")
        print(f"{'=' * 75}")
        header = f"  {'Configuration':<22s} {'MAE':>6s} {'Bias':>7s} {'RMSE':>7s} {'1σ Cov':>7s} {'Games':>6s}"
        print(header)
        print("  " + "-" * (len(header) - 2))

        for config_name, opp, b2b, blowout in configs:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                results = runner_fn(
                    last_n=last_n, min_games=min_games,
                    pace_alpha=pace_alpha, sigma_inflation=sigma_inflation,
                    opp_adjust=opp, b2b_adjust=b2b, b2b_penalty=b2b_penalty,
                    blowout_adjust=blowout, blowout_threshold=bo_thresh,
                    blowout_rate=blowout_rate,
                )

            if results.empty:
                print(f"  {config_name:<22s}   (no results)")
                continue

            n = len(results)
            mae = results["abs_error"].mean()
            bias = results["error"].mean()
            rmse = math.sqrt((results["error"] ** 2).mean())

            # 1-sigma coverage
            lo = results["projected_total"] - results["remaining_std"]
            hi = results["projected_total"] + results["remaining_std"]
            cov = ((results["actual_total"] >= lo) & (results["actual_total"] <= hi)).mean() * 100

            print(f"  {config_name:<22s} {mae:6.1f} {bias:+7.1f} {rmse:7.1f} {cov:6.1f}% {n:6d}")

    # Blowout parameter sweep (halftime only)
    print(f"\n{'=' * 75}")
    print(f"  BLOWOUT PARAMETER SWEEP (Halftime)")
    print(f"{'=' * 75}")
    header = f"  {'threshold':>9s} {'rate':>6s} | {'MAE':>6s} {'Bias':>7s} {'RMSE':>7s} {'Blowout MAE':>11s} {'n_blow':>6s}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for thresh in [10, 15, 20, 25]:
        for rate in [0.05, 0.10, 0.15, 0.20, 0.30]:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                results = run_backtest_improved(
                    last_n=last_n, min_games=min_games,
                    pace_alpha=pace_alpha, sigma_inflation=sigma_inflation,
                    blowout_adjust=True, blowout_threshold=float(thresh),
                    blowout_rate=rate,
                )
            if results.empty:
                continue
            mae = results["abs_error"].mean()
            bias = results["error"].mean()
            rmse = math.sqrt((results["error"] ** 2).mean())
            # MAE for blowout subset only
            blow_mask = results["margin_at_half"] > thresh
            n_blow = int(blow_mask.sum())
            blow_mae = results.loc[blow_mask, "abs_error"].mean() if n_blow > 0 else float("nan")
            print(f"  {thresh:>8d}° {rate:5.2f}  | {mae:6.1f} {bias:+7.1f} {rmse:7.1f} {blow_mae:11.1f} {n_blow:6d}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Per-team totals model: project each team independently, then sum
# ---------------------------------------------------------------------------

def run_backtest_perteam(
    last_n: int = 20,
    min_games: int = 10,
    pace_alpha: float = 0.0,
    sigma_inflation: float = 1.45,
) -> pd.DataFrame:
    """Halftime backtest projecting each team's total independently.

    Instead of computing a single combined pace for both teams, this model:
      1. Computes per-team pace: home_pace = home_h1 / expected_home_h1
      2. Projects each team's H2 using its own pace + profile
      3. Sums for game total
      4. Variance = sum of both teams' independent H2 variances

    This captures asymmetric pace (one team hot, other cold) that combined
    pace averages away.
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

        home_team = h["team"]
        away_team = a["team"]

        h_prof = profiles.get((game_id, home_team))
        a_prof = profiles.get((game_id, away_team))
        if h_prof is None or a_prof is None:
            continue

        h1_home = h["q1"] + h["q2"]
        h1_away = a["q1"] + a["q2"]
        actual_total = h["total"] + a["total"]

        # Per-team expected H1
        exp_h1_home = h_prof["q1_avg"] + h_prof["q2_avg"]
        exp_h1_away = a_prof["q1_avg"] + a_prof["q2_avg"]
        if exp_h1_home <= 0 or exp_h1_away <= 0:
            continue

        # Per-team pace + dampening
        home_raw_pace = h1_home / exp_h1_home
        away_raw_pace = h1_away / exp_h1_away
        home_pace = 1.0 + pace_alpha * (home_raw_pace - 1.0)
        away_pace = 1.0 + pace_alpha * (away_raw_pace - 1.0)

        # Per-team H2 projection
        exp_h2_home = h_prof["q3_avg"] + h_prof["q4_avg"]
        exp_h2_away = a_prof["q3_avg"] + a_prof["q4_avg"]
        proj_home = h1_home + exp_h2_home * home_pace
        proj_away = h1_away + exp_h2_away * away_pace
        projected_total = proj_home + proj_away

        # Per-team H2 variance (independent)
        home_var = (h_prof["q3_std"] ** 2 + h_prof["q4_std"] ** 2) * home_pace ** 2
        away_var = (a_prof["q3_std"] ** 2 + a_prof["q4_std"] ** 2) * away_pace ** 2
        raw_std = math.sqrt(home_var + away_var)
        calibrated_std = raw_std * sigma_inflation

        # Combined pace for logging/filtering compatibility
        exp_h1_combined = exp_h1_home + exp_h1_away
        combined_pace = (h1_home + h1_away) / exp_h1_combined if exp_h1_combined > 0 else 1.0

        error = projected_total - actual_total
        margin_at_half = abs(h1_home - h1_away)

        results.append({
            "game_id": game_id,
            "game_date": h["game_date"],
            "home": home_team,
            "away": away_team,
            "h1_score": h1_home + h1_away,
            "projected_total": round(projected_total, 1),
            "proj_home": round(proj_home, 1),
            "proj_away": round(proj_away, 1),
            "actual_total": actual_total,
            "error": round(error, 1),
            "abs_error": round(abs(error), 1),
            "pace": round(combined_pace, 3),
            "home_pace": round(home_raw_pace, 3),
            "away_pace": round(away_raw_pace, 3),
            "dampened_pace": round((home_pace + away_pace) / 2, 3),
            "margin_at_half": margin_at_half,
            "remaining_std": round(calibrated_std, 1),
            "calibrated_std": round(calibrated_std, 1),
            "lo_1sig": round(projected_total - calibrated_std, 1),
            "hi_1sig": round(projected_total + calibrated_std, 1),
        })

    results_df = pd.DataFrame(results)
    print(f"  Backtested {len(results_df)} games (per-team: alpha={pace_alpha}, sigma={sigma_inflation})")
    return results_df


def run_backtest_perteam_q4(
    last_n: int = 20,
    min_games: int = 10,
    pace_alpha: float = 0.0,
    sigma_inflation: float = 1.45,
) -> pd.DataFrame:
    """Q4-start backtest projecting each team's Q4 independently."""
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

        home_team = h["team"]
        away_team = a["team"]

        h_prof = profiles.get((game_id, home_team))
        a_prof = profiles.get((game_id, away_team))
        if h_prof is None or a_prof is None:
            continue

        q3_home = h["q1"] + h["q2"] + h["q3"]
        q3_away = a["q1"] + a["q2"] + a["q3"]
        q3_score = q3_home + q3_away
        actual_total = h["total"] + a["total"]
        actual_q4 = actual_total - q3_score

        # Per-team expected Q1-Q3
        exp_q3_home = h_prof["q1_avg"] + h_prof["q2_avg"] + h_prof["q3_avg"]
        exp_q3_away = a_prof["q1_avg"] + a_prof["q2_avg"] + a_prof["q3_avg"]
        if exp_q3_home <= 0 or exp_q3_away <= 0:
            continue

        # Per-team pace
        home_raw_pace = q3_home / exp_q3_home
        away_raw_pace = q3_away / exp_q3_away
        home_pace = 1.0 + pace_alpha * (home_raw_pace - 1.0)
        away_pace = 1.0 + pace_alpha * (away_raw_pace - 1.0)

        # Per-team Q4 projection
        proj_home = q3_home + h_prof["q4_avg"] * home_pace
        proj_away = q3_away + a_prof["q4_avg"] * away_pace
        projected_total = proj_home + proj_away

        # Per-team Q4 variance
        home_var = (h_prof["q4_std"] ** 2) * home_pace ** 2
        away_var = (a_prof["q4_std"] ** 2) * away_pace ** 2
        raw_std = math.sqrt(home_var + away_var)
        calibrated_std = raw_std * sigma_inflation

        # Combined pace for logging
        exp_q3_combined = exp_q3_home + exp_q3_away
        combined_pace = q3_score / exp_q3_combined if exp_q3_combined > 0 else 1.0

        error = projected_total - actual_total
        margin_at_q3 = abs(q3_home - q3_away)

        results.append({
            "game_id": game_id,
            "game_date": h["game_date"],
            "home": home_team,
            "away": away_team,
            "h1_score": q3_score,
            "q3_score": q3_score,
            "actual_q4": actual_q4,
            "projected_total": round(projected_total, 1),
            "proj_home": round(proj_home, 1),
            "proj_away": round(proj_away, 1),
            "actual_total": actual_total,
            "error": round(error, 1),
            "abs_error": round(abs(error), 1),
            "pace": round(combined_pace, 3),
            "home_pace": round(home_raw_pace, 3),
            "away_pace": round(away_raw_pace, 3),
            "dampened_pace": round((home_pace + away_pace) / 2, 3),
            "margin_at_half": margin_at_q3,
            "margin_at_q3": margin_at_q3,
            "remaining_std": round(calibrated_std, 1),
            "calibrated_std": round(calibrated_std, 1),
            "lo_1sig": round(projected_total - calibrated_std, 1),
            "hi_1sig": round(projected_total + calibrated_std, 1),
        })

    results_df = pd.DataFrame(results)
    print(f"  Backtested {len(results_df)} games (per-team Q4: alpha={pace_alpha}, sigma={sigma_inflation})")
    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Halftime game total backtest")
    parser.add_argument("--last-n", type=int, default=20,
                        help="Rolling window size (default: 20)")
    parser.add_argument("--min-games", type=int, default=10,
                        help="Min prior games before including in backtest (default: 10)")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip chart generation")
    parser.add_argument("--improved", action="store_true",
                        help="Run improved model with pace dampening + sigma inflation")
    parser.add_argument("--alpha", type=float, default=0.25,
                        help="Pace dampening alpha (default: 0.25)")
    parser.add_argument("--sigma", type=float, default=1.45,
                        help="Sigma inflation factor (default: 1.45)")
    parser.add_argument("--pnl", action="store_true",
                        help="Run P&L simulation (requires --improved)")
    parser.add_argument("--edge", type=float, default=None,
                        help="Edge threshold for P&L sim (default: 0.10 halftime, 0.06 Q4)")
    parser.add_argument("--q4", action="store_true",
                        help="Backtest at start of Q4 (3 quarters completed)")
    parser.add_argument("--q4-grid", action="store_true",
                        help="Grid search over Q4 P&L parameters")
    parser.add_argument("--margin-sweep", action="store_true",
                        help="Sweep margin values and show win rates (use with --q4)")
    parser.add_argument("--grid-search", action="store_true",
                        help="Grid search over alpha values")
    parser.add_argument("--high-conf", action="store_true",
                        help="Run high-confidence betting simulation (use with --q4 or --improved)")
    parser.add_argument("--confidence", type=float, default=0.90,
                        help="Confidence threshold for --high-conf (default: 0.90)")
    parser.add_argument("--perteam", action="store_true",
                        help="Per-team totals model (project each team independently)")
    parser.add_argument("--opp-adjust", action="store_true",
                        help="Apply opponent defensive strength adjustment")
    parser.add_argument("--b2b-adjust", action="store_true",
                        help="Apply back-to-back fatigue penalty")
    parser.add_argument("--b2b-penalty", type=float, default=1.5,
                        help="Points penalty per remaining quarter per b2b team (default: 1.5)")
    parser.add_argument("--blowout-adjust", action="store_true",
                        help="Apply blowout (large margin) scoring reduction")
    parser.add_argument("--blowout-threshold", type=float, default=15.0,
                        help="Margin above which blowout reduction kicks in (default: 15)")
    parser.add_argument("--blowout-rate", type=float, default=0.1,
                        help="Points reduction per margin point above threshold per remaining Q (default: 0.1)")
    parser.add_argument("--compare", action="store_true",
                        help="Run side-by-side comparison of baseline vs opp/b2b/blowout adjustments")
    args = parser.parse_args()

    if args.compare:
        run_comparison(
            last_n=args.last_n, min_games=args.min_games,
            pace_alpha=args.alpha, sigma_inflation=args.sigma,
            b2b_penalty=args.b2b_penalty,
            blowout_threshold=args.blowout_threshold,
            blowout_rate=args.blowout_rate,
        )
    elif args.perteam:
        runner = run_backtest_perteam_q4 if args.q4 else run_backtest_perteam
        results = runner(
            last_n=args.last_n, min_games=args.min_games,
            pace_alpha=args.alpha, sigma_inflation=args.sigma,
        )
        if args.q4:
            print_results_q4(results)
        else:
            print_results(results)
        if args.high_conf:
            simulate_high_confidence(
                results, confidence=args.confidence, no_plot=args.no_plot,
            )
        if not args.no_plot and not args.high_conf:
            plot_results(results)
    elif args.q4_grid:
        grid_search_q4(
            last_n=args.last_n, min_games=args.min_games,
            pace_alpha=args.alpha, sigma_inflation=args.sigma,
        )
    elif args.grid_search:
        grid_search_alpha(last_n=args.last_n, min_games=args.min_games)
    elif args.q4:
        results = run_backtest_q4(
            last_n=args.last_n, min_games=args.min_games,
            pace_alpha=args.alpha, sigma_inflation=args.sigma,
            opp_adjust=args.opp_adjust, b2b_adjust=args.b2b_adjust,
            b2b_penalty=args.b2b_penalty,
            blowout_adjust=args.blowout_adjust,
            blowout_threshold=args.blowout_threshold,
            blowout_rate=args.blowout_rate,
        )
        print_results_q4(results)
        if args.margin_sweep:
            sweep_margin_winrate(results)
        if args.high_conf:
            simulate_high_confidence(
                results, confidence=args.confidence, no_plot=args.no_plot,
            )
        if args.pnl:
            edge = args.edge if args.edge is not None else 0.06
            simulate_pnl_q4(results, edge_threshold=edge, no_plot=args.no_plot)
        if not args.no_plot and not args.high_conf:
            plot_results(results)
    elif args.improved or args.margin_sweep:
        results = run_backtest_improved(
            last_n=args.last_n, min_games=args.min_games,
            pace_alpha=args.alpha, sigma_inflation=args.sigma,
            opp_adjust=args.opp_adjust, b2b_adjust=args.b2b_adjust,
            b2b_penalty=args.b2b_penalty,
            blowout_adjust=args.blowout_adjust,
            blowout_threshold=args.blowout_threshold,
            blowout_rate=args.blowout_rate,
        )
        print_results(results)
        if args.margin_sweep:
            sweep_margin_winrate(results)
        if args.high_conf:
            simulate_high_confidence(
                results, confidence=args.confidence, no_plot=args.no_plot,
            )
        if args.pnl:
            edge = args.edge if args.edge is not None else 0.10
            simulate_pnl(results, edge_threshold=edge)
        if not args.no_plot and not args.high_conf:
            plot_results(results)
    else:
        results = run_backtest(last_n=args.last_n, min_games=args.min_games)
        print_results(results)
        if not args.no_plot:
            plot_results(results)
