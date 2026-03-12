#!/usr/bin/env python3
"""Halftime game total prediction for live NBA games.

Usage:
    PYTHONPATH=src python -m nba.predict                    # show all live games
    PYTHONPATH=src python -m nba.predict LAL BOS            # specific matchup
    PYTHONPATH=src python -m nba.predict --all              # include scheduled/final
    PYTHONPATH=src python -m nba.predict --sim 55 53 LAL BOS  # simulate
"""

from __future__ import annotations

import argparse
import sys
import os

_src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _src) if _src not in sys.path else None

from nba.data import get_live_scoreboard, build_team_quarter_profiles, build_team_quarter_profiles_ha
from nba.strategy import (
    project_game_total, project_halftime_total, project_spread,
    filter_halftime_bet, _total_to_probability,
    PACE_DAMPEN_ALPHA, SIGMA_INFLATION, MAX_PACE_DEVIATION,
    MIN_PROFILE_GAMES, HALFTIME_EDGE_THRESHOLD,
)


def print_prediction(game: dict, profiles: dict, profiles_ha: dict) -> None:
    """Print full prediction for a single game."""
    home = game["home_team"]
    away = game["away_team"]
    period = game.get("period", 0)
    status = game["status"]

    # Header
    print(f"\n{'=' * 64}")
    print(f"  {away} {game['away_score']}  @  {home} {game['home_score']}"
          f"  (P{period} {game.get('clock', '')}  {status})")
    print(f"  Q scores:  {away} {game['away_q1']}/{game['away_q2']}/{game['away_q3']}/{game['away_q4']}"
          f"   {home} {game['home_q1']}/{game['home_q2']}/{game['home_q3']}/{game['home_q4']}")
    print(f"{'=' * 64}")

    current_total = game["home_score"] + game["away_score"]

    if status != "live" or period < 1:
        if status == "final":
            print(f"  Final score: {current_total} pts")
        else:
            # Pre-game: show expected totals from profiles
            hp = profiles_ha.get(f"{home}|home")
            ap = profiles_ha.get(f"{away}|away")
            if hp and ap:
                expected = (hp["q1_avg"] + hp["q2_avg"] + hp["q3_avg"] + hp["q4_avg"] +
                            ap["q1_avg"] + ap["q2_avg"] + ap["q3_avg"] + ap["q4_avg"])
                print(f"  Pre-game expected total: {expected:.1f}")
                print(f"    {home} (home): {hp['total_avg']:.1f} avg ({hp['games_used']} games)")
                print(f"    {away} (away): {ap['total_avg']:.1f} avg ({ap['games_used']} games)")
            else:
                print("  No profile data available.")
        return

    completed = min(period, 4)

    # --- Standard projection (Q1+) ---
    std_proj = project_game_total(game, profiles)
    if std_proj:
        print(f"\n  Standard model (Q{completed}):")
        print(f"    Projected total: {std_proj['projected_total']:.1f}"
              f"  [{std_proj['lo_band']:.0f} - {std_proj['hi_band']:.0f}]")
        print(f"    Pace factor:     {std_proj['pace_factor']:.3f}")
        print(f"    Remaining σ:     {std_proj['remaining_std']:.1f}")

    # --- Halftime model (Q2+) ---
    if completed >= 2:
        ht_proj = project_halftime_total(game, profiles_ha)
        if ht_proj:
            print(f"\n  Halftime model (improved):")
            print(f"    Projected total: {ht_proj['projected_total']:.1f}"
                  f"  [{ht_proj['lo_band']:.0f} - {ht_proj['hi_band']:.0f}]")
            print(f"    Raw pace:        {ht_proj['raw_pace']:.3f}")
            print(f"    Dampened pace:   {ht_proj['dampened_pace']:.3f}"
                  f"  (α={PACE_DAMPEN_ALPHA})")
            print(f"    Calibrated σ:    {ht_proj['calibrated_std']:.1f}"
                  f"  (×{SIGMA_INFLATION})")
            print(f"    Profile depth:   {home}={ht_proj['home_games']}g"
                  f"  {away}={ht_proj['away_games']}g")

            # Naive baseline (only meaningful at halftime)
            if completed == 2:
                h1_home = game["home_q1"] + game["home_q2"]
                h1_away = game["away_q1"] + game["away_q2"]
                naive = (h1_home + h1_away) * 2
                print(f"\n  Baselines:")
                print(f"    Naive (2×H1):    {naive}")
                print(f"    Model delta:     {ht_proj['projected_total'] - naive:+.1f} from naive")

            # Betting filter check
            passed, reason = filter_halftime_bet(ht_proj, edge_pct=15.0, ask_price_cents=50)
            pace_ok = abs(ht_proj["raw_pace"] - 1.0) <= MAX_PACE_DEVIATION
            games_ok = (ht_proj["home_games"] >= MIN_PROFILE_GAMES and
                        ht_proj["away_games"] >= MIN_PROFILE_GAMES)

            print(f"\n  Filter status:")
            print(f"    Pace in range:   {'YES' if pace_ok else 'NO'}"
                  f"  (|{ht_proj['raw_pace']:.3f} - 1.0| = {abs(ht_proj['raw_pace']-1.0):.3f},"
                  f" max {MAX_PACE_DEVIATION})")
            print(f"    Profile depth:   {'YES' if games_ok else 'NO'}"
                  f"  (min {MIN_PROFILE_GAMES})")
            if period > 4:
                print(f"    Overtime:        SKIP")

            # 90% confidence lines
            from statistics import NormalDist
            z90 = NormalDist().inv_cdf(0.90)
            std = ht_proj["calibrated_std"]
            proj = ht_proj["projected_total"]
            over_line = proj - z90 * std
            under_line = proj + z90 * std

            print(f"\n  90% confidence lines:")
            print(f"    OVER  {over_line:.1f}  (profitable up to ~90¢)")
            print(f"    UNDER {under_line:.1f}  (profitable up to ~90¢)")

            # Per-team projections
            hp = profiles_ha.get(f"{home}|home")
            ap = profiles_ha.get(f"{away}|away")
            if hp and ap:
                home_h1 = game["home_q1"] + game["home_q2"]
                away_h1 = game["away_q1"] + game["away_q2"]
                if completed >= 3:
                    home_so_far = home_h1 + game["home_q3"]
                    away_so_far = away_h1 + game["away_q3"]
                    home_remaining = hp["q4_avg"] if completed == 3 else 0
                    away_remaining = ap["q4_avg"] if completed == 3 else 0
                    home_std = hp["q4_std"] * SIGMA_INFLATION if completed == 3 else 0
                    away_std = ap["q4_std"] * SIGMA_INFLATION if completed == 3 else 0
                else:
                    home_so_far = home_h1
                    away_so_far = away_h1
                    home_remaining = hp["q3_avg"] + hp["q4_avg"]
                    away_remaining = ap["q3_avg"] + ap["q4_avg"]
                    home_std = (hp["q3_std"] ** 2 + hp["q4_std"] ** 2) ** 0.5 * SIGMA_INFLATION
                    away_std = (ap["q3_std"] ** 2 + ap["q4_std"] ** 2) ** 0.5 * SIGMA_INFLATION

                home_proj = home_so_far + home_remaining
                away_proj = away_so_far + away_remaining
                home_over = home_proj - z90 * home_std
                home_under = home_proj + z90 * home_std
                away_over = away_proj - z90 * away_std
                away_under = away_proj + z90 * away_std

                print(f"\n  Team score projections (90% conf):")
                print(f"    {home}: {home_proj:.1f}  [OVER {home_over:.1f} / UNDER {home_under:.1f}]")
                print(f"    {away}: {away_proj:.1f}  [OVER {away_over:.1f} / UNDER {away_under:.1f}]")

            # Over/under probabilities at various lines
            print(f"\n  Over/under probabilities:")
            for offset in [-10, -5, -2, 0, 2, 5, 10]:
                line = round(proj + offset)
                p_over = _total_to_probability(proj, line, std)
                edge = abs(p_over - 0.5)
                marker = " <-- projected" if offset == 0 else ""
                edge_flag = f"  EDGE {edge*100:.0f}%" if edge >= HALFTIME_EDGE_THRESHOLD else ""
                print(f"    O {line:5.1f}: {p_over*100:5.1f}%{edge_flag}{marker}")
        else:
            print("\n  Halftime model: unavailable (missing profile data)")
    else:
        print(f"\n  Halftime model: waiting for Q2 (currently Q{completed})")

    # --- Spread ---
    sp = project_spread(game, profiles)
    if sp:
        sign = "+" if sp["projected_margin"] >= 0 else ""
        print(f"\n  Spread:")
        print(f"    {home} {sign}{sp['projected_margin']:.1f}  (±{sp['confidence_band']})")
        print(f"    Projected: {home}={sp['home_projected']:.0f}"
              f"  {away}={sp['away_projected']:.0f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="NBA halftime game total predictions"
    )
    parser.add_argument("teams", nargs="*",
                        help="Filter by team codes (e.g. LAL BOS)")
    parser.add_argument("--all", action="store_true",
                        help="Show all games including scheduled/final")
    parser.add_argument("--sim", nargs=2, type=int, metavar=("HOME_SCORE", "AWAY_SCORE"),
                        help="Simulate with custom scores (requires 2 team args)")
    parser.add_argument("--period", type=int, default=2,
                        help="Period for simulation (default: 2)")
    args = parser.parse_args()

    # Load profiles
    print("Loading profiles...")
    profiles = build_team_quarter_profiles()
    profiles_ha = build_team_quarter_profiles_ha()
    print(f"  {len(profiles)} team profiles, {len(profiles_ha)} home/away entries\n")

    # Simulation mode
    if args.sim:
        if len(args.teams) < 2:
            print("Error: --sim requires two team codes (home away)")
            print("Usage: python predict.py --sim 55 53 LAL BOS")
            sys.exit(1)
        home_team = args.teams[0].upper()
        away_team = args.teams[1].upper()
        home_score, away_score = args.sim

        # Build synthetic quarter scores (split evenly)
        completed = min(args.period, 4)
        per_q = home_score // completed if completed > 0 else 0
        per_q_a = away_score // completed if completed > 0 else 0

        game = {
            "game_id": "SIM",
            "status": "live",
            "period": args.period,
            "clock": "0:00",
            "home_team": home_team,
            "away_team": away_team,
            "home_score": home_score,
            "away_score": away_score,
            "home_q1": per_q, "home_q2": home_score - per_q * (completed - 1) if completed == 2 else per_q,
            "home_q3": 0, "home_q4": 0,
            "away_q1": per_q_a, "away_q2": away_score - per_q_a * (completed - 1) if completed == 2 else per_q_a,
            "away_q3": 0, "away_q4": 0,
        }
        print_prediction(game, profiles, profiles_ha)
        return

    # Live mode
    print("Fetching live scoreboard...")
    games = get_live_scoreboard()

    if not games:
        print("No games today.")
        return

    # Filter by team if specified
    team_filter = {t.upper() for t in args.teams} if args.teams else None

    shown = 0
    for game in games:
        if not args.all and game["status"] not in ("live",):
            continue
        if team_filter:
            if game["home_team"] not in team_filter and game["away_team"] not in team_filter:
                continue

        print_prediction(game, profiles, profiles_ha)
        shown += 1

    if shown == 0:
        statuses = [g["status"] for g in games]
        if "live" not in statuses:
            print("No live games right now.")
            if any(s == "scheduled" for s in statuses):
                print("Use --all to see scheduled games.")
        elif team_filter:
            print(f"No games found for {', '.join(team_filter)}.")
            print(f"Teams playing: {', '.join(set(g['home_team'] for g in games) | set(g['away_team'] for g in games))}")


if __name__ == "__main__":
    main()
