#!/usr/bin/env python3
"""NBA quarter-by-quarter projection and edge detection.

Core strategy: use historical quarter scoring patterns to project game
outcomes mid-game, compare projections against current Kalshi market
prices, and identify edges where markets are mispriced.

Projection methods:
  - Game totals: pace-adjusted remaining-quarter extrapolation
  - Spreads:     per-team pace scaling with mean-reversion
  - Player props: per-minute rate projection with blowout adjustment
"""

from __future__ import annotations

import logging
import math
import os
import statistics
from typing import Dict, List, Optional

log = logging.getLogger(__name__)

# Edge thresholds by quarter — require larger edge early when uncertainty is high
EDGE_THRESHOLDS = {
    1: 0.15,  # Q1: need 15% edge
    2: 0.10,  # Q2 / halftime: 10%
    3: 0.08,  # Q3: 8%
    4: 0.06,  # Q4: 6% (most certain)
}

# Typical minutes per game for starters / key players
EXPECTED_TOTAL_MINUTES = 34.0

# Blowout threshold: if margin exceeds this, reduce projected minutes
BLOWOUT_MARGIN = 20
BLOWOUT_MINUTES_REDUCTION = 0.80  # 80% of expected minutes

# Halftime total projection constants (from backtest validation)
PACE_DAMPEN_ALPHA = 0.0
SIGMA_INFLATION = 1.45
HALFTIME_EDGE_THRESHOLD = 0.10
MAX_PACE_DEVIATION = 0.15
MIN_PROFILE_GAMES = 12


# ---------------------------------------------------------------------------
# Game Total Projection
# ---------------------------------------------------------------------------

def project_game_total(live_game: dict, quarter_profiles: dict) -> Optional[dict]:
    """Project final combined score from current quarter data.

    Approach:
      - Compute how many points we'd expect through the current period
        based on historical averages for both teams
      - Pace factor = actual_so_far / expected_so_far
      - Project remaining quarters using historical averages * pace factor
      - Confidence band from historical standard deviations

    Args:
        live_game: dict from get_live_scoreboard() with quarter scores
        quarter_profiles: dict from build_team_quarter_profiles()

    Returns:
        {projected_total, lo_band, hi_band, pace_factor, period,
         current_total, home_team, away_team}
        or None if game isn't live or teams not in profiles.
    """
    period = live_game.get("period", 0)
    if period < 1 or live_game.get("status") != "live":
        return None

    home = live_game["home_team"]
    away = live_game["away_team"]

    home_prof = quarter_profiles.get(home)
    away_prof = quarter_profiles.get(away)
    if not home_prof or not away_prof:
        log.warning("Missing quarter profile for %s or %s", home, away)
        return None

    # Actual scores through completed quarters
    current_total = live_game["home_score"] + live_game["away_score"]
    completed = min(period, 4)  # treat OT as Q4 for projection

    # Expected points through completed quarters (both teams combined)
    q_keys = ["q1_avg", "q2_avg", "q3_avg", "q4_avg"]
    expected_through = sum(
        home_prof[q_keys[q]] + away_prof[q_keys[q]]
        for q in range(completed)
    )

    if expected_through <= 0:
        return None

    pace_factor = current_total / expected_through

    # Project remaining quarters
    remaining_expected = sum(
        home_prof[q_keys[q]] + away_prof[q_keys[q]]
        for q in range(completed, 4)
    )

    projected_total = current_total + remaining_expected * pace_factor

    # Confidence band: combine remaining-quarter std devs
    q_std_keys = ["q1_std", "q2_std", "q3_std", "q4_std"]
    remaining_var = sum(
        home_prof[q_std_keys[q]] ** 2 + away_prof[q_std_keys[q]] ** 2
        for q in range(completed, 4)
    )
    remaining_std = math.sqrt(remaining_var) * pace_factor

    return {
        "projected_total": round(projected_total, 1),
        "lo_band": round(projected_total - 1.5 * remaining_std, 1),
        "hi_band": round(projected_total + 1.5 * remaining_std, 1),
        "pace_factor": round(pace_factor, 3),
        "period": period,
        "current_total": current_total,
        "home_team": home,
        "away_team": away,
        "remaining_std": round(remaining_std, 1),
    }


# ---------------------------------------------------------------------------
# Game clock parsing
# ---------------------------------------------------------------------------

def _parse_quarter_remaining(clock: str) -> float:
    """Parse NBA game clock to fraction of quarter remaining (0.0 to 1.0).

    Handles formats:
      - "PT03M47.00S" (ISO 8601 duration from live API)
      - "3:47" (simple minutes:seconds)
      - "" or None → 0.0 (assume quarter just ended)

    NBA quarter = 12 minutes.
    """
    if not clock:
        return 0.0

    import re

    # ISO 8601: PT03M47.00S
    m = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:([\d.]+)S)?", clock)
    if m:
        hours = int(m.group(1) or 0)
        mins = int(m.group(2) or 0)
        secs = float(m.group(3) or 0)
        total_sec = hours * 3600 + mins * 60 + secs
        return min(total_sec / 720.0, 1.0)  # 12 min = 720 sec

    # Simple: "3:47"
    m = re.match(r"(\d+):(\d+)", clock)
    if m:
        mins = int(m.group(1))
        secs = int(m.group(2))
        total_sec = mins * 60 + secs
        return min(total_sec / 720.0, 1.0)

    return 0.0


# ---------------------------------------------------------------------------
# Halftime Total Projection (improved model)
# ---------------------------------------------------------------------------

def project_halftime_total(
    live_game: dict,
    quarter_profiles_ha: dict,
    pace_alpha: float = PACE_DAMPEN_ALPHA,
    sigma_inflation: float = SIGMA_INFLATION,
) -> Optional[dict]:
    """Project final game total at halftime using pace-dampened model.

    Requires period >= 2 (at least halftime).  Uses home/away-split profiles
    for more accurate per-team estimates.

    Improvements over project_game_total():
      - Pace dampening: dampened = 1.0 + alpha * (raw - 1.0)
      - Sigma inflation: calibrated_std = raw_std * sigma_inflation

    Args:
        live_game: dict from get_live_scoreboard() with quarter scores
        quarter_profiles_ha: dict keyed by "TEAM|home" or "TEAM|away"
        pace_alpha: dampening factor (0=full dampen, 1=raw pace)
        sigma_inflation: multiplier on raw std for calibration

    Returns:
        {projected_total, lo_band, hi_band, pace_factor, dampened_pace,
         raw_pace, period, current_total, home_team, away_team,
         remaining_std, calibrated_std, home_games, away_games}
        or None if conditions not met.
    """
    period = live_game.get("period", 0)
    if period < 2 or live_game.get("status") != "live":
        return None

    home = live_game["home_team"]
    away = live_game["away_team"]

    home_prof = quarter_profiles_ha.get(f"{home}|home")
    away_prof = quarter_profiles_ha.get(f"{away}|away")
    if not home_prof or not away_prof:
        log.warning("Missing home/away profile for %s or %s", home, away)
        return None

    current_total = live_game["home_score"] + live_game["away_score"]
    completed = min(period, 4)

    # Parse game clock to estimate fraction of current quarter remaining
    q4_frac_remaining = _parse_quarter_remaining(live_game.get("clock", ""))

    q_keys = ["q1_avg", "q2_avg", "q3_avg", "q4_avg"]

    # For pace calculation, use completed quarters only (not partial current)
    pace_quarters = completed if period > 4 else max(completed - 1, 1)
    expected_through_pace = sum(
        home_prof[q_keys[q]] + away_prof[q_keys[q]]
        for q in range(pace_quarters)
    )
    # For mid-quarter: add the played fraction of current quarter's expected
    if completed >= 2 and completed <= 4 and q4_frac_remaining < 1.0:
        cur_q_idx = completed - 1
        cur_q_expected = home_prof[q_keys[cur_q_idx]] + away_prof[q_keys[cur_q_idx]]
        played_frac = 1.0 - q4_frac_remaining
        expected_through_pace += cur_q_expected * played_frac

    if expected_through_pace <= 0:
        return None

    raw_pace = current_total / expected_through_pace
    dampened_pace = 1.0 + pace_alpha * (raw_pace - 1.0)

    # Remaining expected: full future quarters + remaining fraction of current quarter
    remaining_expected = sum(
        home_prof[q_keys[q]] + away_prof[q_keys[q]]
        for q in range(completed, 4)
    )
    if completed <= 4 and q4_frac_remaining > 0 and q4_frac_remaining < 1.0:
        cur_q_idx = completed - 1
        cur_q_expected = home_prof[q_keys[cur_q_idx]] + away_prof[q_keys[cur_q_idx]]
        remaining_expected += cur_q_expected * q4_frac_remaining

    projected_total = current_total + remaining_expected * dampened_pace

    q_std_keys = ["q1_std", "q2_std", "q3_std", "q4_std"]
    remaining_var = sum(
        home_prof[q_std_keys[q]] ** 2 + away_prof[q_std_keys[q]] ** 2
        for q in range(completed, 4)
    )
    # Add partial current-quarter variance
    if completed <= 4 and q4_frac_remaining > 0 and q4_frac_remaining < 1.0:
        cur_q_idx = completed - 1
        cur_q_var = home_prof[q_std_keys[cur_q_idx]] ** 2 + away_prof[q_std_keys[cur_q_idx]] ** 2
        remaining_var += cur_q_var * q4_frac_remaining

    raw_std = math.sqrt(remaining_var) * dampened_pace if remaining_var > 0 else 0.0
    calibrated_std = raw_std * sigma_inflation

    return {
        "projected_total": round(projected_total, 1),
        "lo_band": round(projected_total - 1.5 * calibrated_std, 1),
        "hi_band": round(projected_total + 1.5 * calibrated_std, 1),
        "pace_factor": round(dampened_pace, 3),
        "raw_pace": round(raw_pace, 3),
        "dampened_pace": round(dampened_pace, 3),
        "period": period,
        "current_total": current_total,
        "home_team": home,
        "away_team": away,
        "remaining_std": round(calibrated_std, 1),
        "calibrated_std": round(calibrated_std, 1),
        "home_games": home_prof.get("games_used", 0),
        "away_games": away_prof.get("games_used", 0),
    }


def filter_halftime_bet(
    projection: dict,
    edge_pct: float,
    ask_price_cents: float,
) -> tuple:
    """Check if a halftime total bet passes all filters.

    Returns (passed: bool, reason: str).
    """
    raw_pace = projection.get("raw_pace", 1.0)
    if abs(raw_pace - 1.0) > MAX_PACE_DEVIATION:
        return False, f"pace too extreme ({raw_pace:.3f}, max dev {MAX_PACE_DEVIATION})"

    home_games = projection.get("home_games", 0)
    away_games = projection.get("away_games", 0)
    if home_games < MIN_PROFILE_GAMES or away_games < MIN_PROFILE_GAMES:
        return False, f"insufficient profile games (home={home_games}, away={away_games}, min={MIN_PROFILE_GAMES})"

    period = projection.get("period", 0)
    if period > 4:
        return False, "overtime — skipping"

    if ask_price_cents > 90:
        return False, f"ask too high ({ask_price_cents}¢)"
    if ask_price_cents < 10:
        return False, f"ask too low ({ask_price_cents}¢)"

    if edge_pct < HALFTIME_EDGE_THRESHOLD * 100:
        return False, f"edge too small ({edge_pct:.1f}% < {HALFTIME_EDGE_THRESHOLD*100:.0f}%)"

    return True, "ok"


# ---------------------------------------------------------------------------
# Spread Projection
# ---------------------------------------------------------------------------

def project_spread(live_game: dict, quarter_profiles: dict) -> Optional[dict]:
    """Project final margin from current per-team quarter scoring.

    Uses mean reversion: large early leads tend to shrink as the game
    progresses.  We blend the current margin with the historical
    expected margin.

    Returns:
        {projected_margin (home-away), home_projected, away_projected,
         confidence_band, period, current_margin}
    """
    period = live_game.get("period", 0)
    if period < 1 or live_game.get("status") != "live":
        return None

    home = live_game["home_team"]
    away = live_game["away_team"]
    home_prof = quarter_profiles.get(home)
    away_prof = quarter_profiles.get(away)
    if not home_prof or not away_prof:
        return None

    completed = min(period, 4)
    home_score = live_game["home_score"]
    away_score = live_game["away_score"]
    current_margin = home_score - away_score

    q_keys = ["q1_avg", "q2_avg", "q3_avg", "q4_avg"]

    # Historical expected points through completed quarters per team
    home_expected = sum(home_prof[q_keys[q]] for q in range(completed))
    away_expected = sum(away_prof[q_keys[q]] for q in range(completed))

    # Per-team pace factors
    home_pace = home_score / home_expected if home_expected > 0 else 1.0
    away_pace = away_score / away_expected if away_expected > 0 else 1.0

    # Project remaining
    home_remaining = sum(home_prof[q_keys[q]] for q in range(completed, 4))
    away_remaining = sum(away_prof[q_keys[q]] for q in range(completed, 4))

    home_projected = home_score + home_remaining * home_pace
    away_projected = away_score + away_remaining * away_pace
    raw_margin = home_projected - away_projected

    # Mean reversion: blend toward 0 based on how early we are
    # After Q1: 50% reversion; Q2: 30%; Q3: 15%; Q4: 5%
    reversion_factor = {1: 0.50, 2: 0.30, 3: 0.15, 4: 0.05}.get(completed, 0.05)
    projected_margin = raw_margin * (1 - reversion_factor)

    # Confidence band
    q_std_keys = ["q1_std", "q2_std", "q3_std", "q4_std"]
    remaining_var = sum(
        home_prof[q_std_keys[q]] ** 2 + away_prof[q_std_keys[q]] ** 2
        for q in range(completed, 4)
    )
    confidence_band = round(1.5 * math.sqrt(remaining_var), 1)

    return {
        "projected_margin": round(projected_margin, 1),
        "home_projected": round(home_projected, 1),
        "away_projected": round(away_projected, 1),
        "confidence_band": confidence_band,
        "period": period,
        "current_margin": current_margin,
        "home_team": home,
        "away_team": away,
    }


# ---------------------------------------------------------------------------
# Player Prop Projection
# ---------------------------------------------------------------------------

def project_player_stat(player_name: str, current_stats: dict,
                        minutes_played: float, season_avg: dict,
                        game_margin: int = 0) -> Optional[dict]:
    """Project final stat line from current per-minute rates.

    Approach: (current_stat / minutes_played) * expected_total_minutes
    Adjusted for blowout (reduce expected minutes if margin > 20).

    Args:
        player_name: for logging
        current_stats: {pts, reb, ast, fg3m} — current in-game stats
        minutes_played: how many minutes played so far
        season_avg: from get_player_recent_stats() — {avg_pts, avg_reb, ...}
        game_margin: absolute margin; if > BLOWOUT_MARGIN, reduce projected minutes

    Returns:
        {projected_pts, projected_reb, projected_ast, projected_3pm,
         confidence_band_pts, minutes_projected}
    """
    if minutes_played < 5:
        return None  # too early for reliable per-minute rates

    expected_minutes = season_avg.get("avg_min", EXPECTED_TOTAL_MINUTES)
    if abs(game_margin) > BLOWOUT_MARGIN:
        expected_minutes *= BLOWOUT_MINUTES_REDUCTION

    remaining_minutes = max(0, expected_minutes - minutes_played)
    rate_factor = remaining_minutes / minutes_played if minutes_played > 0 else 0

    pts = current_stats.get("pts", 0)
    reb = current_stats.get("reb", 0)
    ast = current_stats.get("ast", 0)
    fg3m = current_stats.get("fg3m", 0)

    projected_pts = pts + pts * rate_factor
    projected_reb = reb + reb * rate_factor
    projected_ast = ast + ast * rate_factor
    projected_3pm = fg3m + fg3m * rate_factor

    # Blend with season average (weight current game more as it progresses)
    game_weight = min(minutes_played / expected_minutes, 1.0)
    season_weight = 1.0 - game_weight

    projected_pts = projected_pts * game_weight + season_avg.get("avg_pts", 0) * season_weight
    projected_reb = projected_reb * game_weight + season_avg.get("avg_reb", 0) * season_weight
    projected_ast = projected_ast * game_weight + season_avg.get("avg_ast", 0) * season_weight
    projected_3pm = projected_3pm * game_weight + season_avg.get("avg_fg3m", 0) * season_weight

    # Rough confidence band (narrows as game progresses)
    confidence_band_pts = round(8.0 * (1 - game_weight) + 3.0, 1)

    return {
        "player_name": player_name,
        "projected_pts": round(projected_pts, 1),
        "projected_reb": round(projected_reb, 1),
        "projected_ast": round(projected_ast, 1),
        "projected_3pm": round(projected_3pm, 1),
        "confidence_band_pts": confidence_band_pts,
        "minutes_projected": round(expected_minutes, 1),
        "minutes_played": round(minutes_played, 1),
    }


# ---------------------------------------------------------------------------
# Probability conversion
# ---------------------------------------------------------------------------

def _total_to_probability(projected: float, line: float, std: float) -> float:
    """Convert a projected total + std dev to P(over line).

    Uses normal CDF approximation.
    """
    if std <= 0:
        return 1.0 if projected > line else 0.0

    z = (projected - line) / std
    # Approximation of normal CDF
    return _norm_cdf(z)


def _norm_cdf(z: float) -> float:
    """Standard normal CDF approximation (Abramowitz & Stegun)."""
    if z > 6:
        return 1.0
    if z < -6:
        return 0.0
    a1, a2, a3, a4, a5 = (0.254829592, -0.284496736, 1.421413741,
                           -1.453152027, 1.061405429)
    p = 0.3275911
    sign = 1 if z >= 0 else -1
    z_abs = abs(z)
    t = 1.0 / (1.0 + p * z_abs)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-z_abs * z_abs / 2)
    return 0.5 * (1.0 + sign * y)


# ---------------------------------------------------------------------------
# Edge Scanner
# ---------------------------------------------------------------------------

def find_live_edges(kalshi_client, quarter_profiles: dict,
                    edge_threshold: float = 0.08,
                    quarter_profiles_ha: Optional[dict] = None) -> List[dict]:
    """Scan all live games for mispriced Kalshi markets.

    1. Get live scoreboard from nba_api
    2. Get Kalshi NBA markets + orderbook prices
    3. For each game: project total, spread
    4. Convert projections to probability
    5. Compare vs Kalshi price → edge = projected_prob - kalshi_implied
    6. Filter by quarter-appropriate threshold

    Returns list of edge dicts sorted by absolute edge descending.
    """
    from nba.data import get_live_scoreboard
    from nba.markets import discover_nba_series, get_game_markets, parse_market_line

    edges: List[dict] = []

    # Get live games
    try:
        games = get_live_scoreboard()
    except Exception as e:
        log.error("Failed to fetch live scoreboard: %s", e)
        return []

    live_games = [g for g in games if g["status"] == "live"]
    if not live_games:
        log.info("No live games right now")
        return []

    log.info("Found %d live games", len(live_games))

    # Discover Kalshi NBA series
    try:
        series = discover_nba_series(kalshi_client)
    except Exception as e:
        log.error("Failed to discover NBA series: %s", e)
        return []

    if not series:
        log.warning("No NBA series tickers found on Kalshi")
        return []

    # Get all open markets
    all_markets = []
    for ticker in series:
        try:
            markets = get_game_markets(kalshi_client, ticker, status="open")
            all_markets.extend(markets)
        except Exception as e:
            log.debug("Failed to get markets for %s: %s", ticker, e)

    if not all_markets:
        log.info("No open NBA markets found")
        return []

    log.info("Found %d open NBA markets", len(all_markets))

    # Scan each live game
    for game in live_games:
        period = game.get("period", 0)
        if period < 1:
            continue

        completed = min(period, 4)
        threshold = EDGE_THRESHOLDS.get(completed, edge_threshold)

        # --- Game total projection ---
        # Use halftime model (pace-dampened) when at halftime+ and HA profiles available
        if completed >= 2 and quarter_profiles_ha:
            total_proj = project_halftime_total(game, quarter_profiles_ha)
        else:
            total_proj = project_game_total(game, quarter_profiles)
        if total_proj:
            # Find matching Kalshi total markets
            for mkt in all_markets:
                parsed = parse_market_line(mkt)
                if not parsed:
                    continue

                # Match game total over/under markets
                if parsed.get("side") in ("over", "under") and not parsed.get("player"):
                    line = parsed["line"]
                    # Check if this market plausibly relates to this game
                    # (title should mention teams or be in a relevant series)
                    title = mkt.get("title", "") + " " + mkt.get("subtitle", "")
                    home = game["home_team"]
                    away = game["away_team"]
                    if home not in title and away not in title:
                        continue

                    std = total_proj.get("remaining_std", 10.0)
                    proj_prob_over = _total_to_probability(
                        total_proj["projected_total"], line, std
                    )

                    # Get Kalshi implied probability
                    try:
                        book = kalshi_client.get_orderbook(mkt["ticker"])
                        no_bids = book.get("no", [])
                        yes_bids = book.get("yes", [])
                        # Best ask for YES = 100 - best NO bid
                        if no_bids:
                            yes_ask = (100 - no_bids[-1][0]) / 100.0
                        elif yes_bids:
                            yes_ask = yes_bids[-1][0] / 100.0
                        else:
                            continue
                    except Exception:
                        continue

                    if parsed["side"] == "over":
                        edge = proj_prob_over - yes_ask
                    else:  # under
                        edge = (1.0 - proj_prob_over) - yes_ask

                    if abs(edge) >= threshold:
                        buy_side = "yes" if edge > 0 else "no"
                        edges.append({
                            "game": f"{away}@{home}",
                            "market_type": "total",
                            "ticker": mkt["ticker"],
                            "side": f"{parsed['side']} {line}",
                            "buy_side": buy_side,
                            "kalshi_price": round(yes_ask * 100, 1),
                            "projected_prob": round(
                                (proj_prob_over if parsed["side"] == "over"
                                 else 1.0 - proj_prob_over) * 100, 1
                            ),
                            "edge_pct": round(abs(edge) * 100, 1),
                            "detail": (f"proj={total_proj['projected_total']:.1f} "
                                       f"pace={total_proj['pace_factor']:.2f} "
                                       f"band=[{total_proj['lo_band']:.0f}-{total_proj['hi_band']:.0f}]"),
                            "period": period,
                        })

        # --- Spread projection ---
        spread_proj = project_spread(game, quarter_profiles)
        if spread_proj:
            for mkt in all_markets:
                parsed = parse_market_line(mkt)
                if not parsed or parsed.get("side") != "spread":
                    continue

                title = mkt.get("title", "") + " " + mkt.get("subtitle", "")
                home = game["home_team"]
                away = game["away_team"]
                if home not in title and away not in title:
                    continue

                line = parsed["line"]
                band = spread_proj["confidence_band"]
                if band <= 0:
                    band = 8.0
                proj_margin = spread_proj["projected_margin"]

                # P(home covers spread line)
                prob_cover = _norm_cdf((proj_margin - line) / band)

                try:
                    book = kalshi_client.get_orderbook(mkt["ticker"])
                    no_bids = book.get("no", [])
                    if no_bids:
                        yes_ask = (100 - no_bids[-1][0]) / 100.0
                    else:
                        continue
                except Exception:
                    continue

                edge = prob_cover - yes_ask
                if abs(edge) >= threshold:
                    edges.append({
                        "game": f"{away}@{home}",
                        "market_type": "spread",
                        "ticker": mkt["ticker"],
                        "side": f"{parsed.get('team', '')} {line:+.1f}",
                        "buy_side": "yes" if edge > 0 else "no",
                        "kalshi_price": round(yes_ask * 100, 1),
                        "projected_prob": round(prob_cover * 100, 1),
                        "edge_pct": round(abs(edge) * 100, 1),
                        "detail": (f"margin={proj_margin:+.1f} "
                                   f"band=±{band:.1f}"),
                        "period": period,
                    })

    # Sort by edge descending
    edges.sort(key=lambda e: e["edge_pct"], reverse=True)
    return edges


# ---------------------------------------------------------------------------
# Watch loop
# ---------------------------------------------------------------------------

def nba_watch_loop(config, client, stop_event, dry_run: bool = True) -> None:
    """Poll for NBA edges during live games.

    1. Build/load quarter profiles at startup
    2. Poll every 60s while games are live
    3. Place orders (or dry-run log) for edges above threshold
    4. Per-market lock: don't bet same ticker twice
    """
    import time
    from nba.data import build_team_quarter_profiles, build_team_quarter_profiles_ha, get_live_scoreboard

    log.info("[NBA] Building quarter profiles...")
    profiles = build_team_quarter_profiles()
    log.info("[NBA] Loaded profiles for %d teams", len(profiles))

    log.info("[NBA] Building home/away-split profiles...")
    profiles_ha = build_team_quarter_profiles_ha()
    log.info("[NBA] Loaded %d home/away profile entries", len(profiles_ha))

    bet_tickers: set = set()  # already-bet tickers

    while not stop_event.is_set():
        try:
            games = get_live_scoreboard()
            live_games = [g for g in games if g["status"] == "live"]

            if not live_games:
                # Check if any games are scheduled
                scheduled = [g for g in games if g["status"] == "scheduled"]
                if scheduled:
                    log.info("[NBA] %d games scheduled, waiting...", len(scheduled))
                    stop_event.wait(120)
                    continue
                else:
                    # All done or no games today — sleep and retry
                    final = [g for g in games if g["status"] == "final"]
                    if final:
                        log.info("[NBA] All %d games are final. Sleeping 30 min.", len(final))
                    else:
                        log.info("[NBA] No games today. Sleeping 30 min.")
                    bet_tickers.clear()
                    stop_event.wait(1800)
                    continue

            log.info("[NBA] %d live games — scanning for edges...", len(live_games))
            edges = find_live_edges(client, profiles, quarter_profiles_ha=profiles_ha)

            for edge in edges:
                ticker = edge["ticker"]
                if ticker in bet_tickers:
                    log.debug("[NBA] Already bet %s — skipping", ticker)
                    continue

                log.info(
                    "[NBA] EDGE: %s %s — %s %s | "
                    "Kalshi=%s¢ Proj=%.1f%% Edge=%.1f%% | %s",
                    edge["game"], edge["market_type"], edge["side"],
                    edge["buy_side"].upper(),
                    edge["kalshi_price"], edge["projected_prob"],
                    edge["edge_pct"], edge["detail"],
                )

                if not dry_run:
                    # TODO: place order via client.place_order()
                    log.info("[NBA] Would place order on %s (live trading not yet enabled)",
                             ticker)

                bet_tickers.add(ticker)

        except Exception as e:
            log.error("[NBA] Watch loop error: %s", e, exc_info=True)

        stop_event.wait(60)

    log.info("[NBA] Watch loop exited.")


# ---------------------------------------------------------------------------
# Live checkpoint alerts (Telegram-only)
# ---------------------------------------------------------------------------

def _setup_file_logger(name: str, filename: str) -> logging.Logger:
    """Create a dedicated file logger at log/<filename>."""
    from paths import project_path

    log_dir = project_path("log")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, filename)

    flog = logging.getLogger(name)
    flog.setLevel(logging.DEBUG)
    if not flog.handlers:
        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s"
        ))
        flog.addHandler(fh)
    return flog


def watch_checkpoints(
    confidence: float = 0.90,
    poll_interval: int = 30,
    stop_event: Optional[threading.Event] = None,
) -> None:
    """Poll live games and send Telegram alerts at halftime and Q4 start.

    At each checkpoint, runs project_halftime_total() to get projected total
    and calibrated_std, then computes high-confidence OVER/UNDER lines.

    Args:
        confidence: confidence level for betting lines (default 0.90)
        poll_interval: seconds between polls (default 30)
        stop_event: optional threading.Event to signal shutdown
    """
    import time
    from statistics import NormalDist

    from bot.app import _send_telegram
    from nba.data import build_team_quarter_profiles_ha, get_live_scoreboard

    glog = _setup_file_logger("nba.game", "nba.log")

    z = NormalDist().inv_cdf(confidence)
    conf_pct = int(confidence * 100)

    log.info("[NBA-ALERTS] Building home/away profiles...")
    glog.info("=== NBA Watch Session Started (confidence=%d%%, poll=%ds) ===",
              conf_pct, poll_interval)
    profiles_ha = build_team_quarter_profiles_ha()
    log.info("[NBA-ALERTS] Loaded %d profile entries", len(profiles_ha))
    log.info("[NBA-ALERTS] Watching for checkpoints (confidence=%d%%, z=%.4f)", conf_pct, z)

    # game_id -> set of checkpoints already notified
    notified: Dict[str, set] = {}
    # game_id -> last known state for transition detection
    game_state: Dict[str, dict] = {}

    def _game_label(g: dict) -> str:
        return f"{g['away_team']}@{g['home_team']}"

    if stop_event is None:
        stop_event = threading.Event()

    while not stop_event.is_set():
        try:
            games = get_live_scoreboard()
        except Exception as e:
            log.error("[NBA-ALERTS] Scoreboard error: %s", e)
            glog.error("Scoreboard fetch failed: %s", e)
            stop_event.wait(poll_interval)
            continue

        live_games = [g for g in games if g["status"] == "live"]
        scheduled = [g for g in games if g["status"] == "scheduled"]
        final = [g for g in games if g["status"] == "final"]

        if not live_games and not scheduled:
            if final:
                log.info("[NBA-ALERTS] All %d games are final. Sleeping until next check.", len(final))
                glog.info("All %d games final. Session ended.", len(final))
            else:
                log.info("[NBA-ALERTS] No games today. Sleeping until next check.")
                glog.info("No games today. Sleeping.")
            # Sleep 30 min then re-poll (games may start later or tomorrow)
            # Reset state for next batch of games
            notified.clear()
            game_state.clear()
            stop_event.wait(1800)
            continue

        # Detect game state transitions and log to nba.log
        for game in games:
            gid = game["game_id"]
            label = _game_label(game)
            period = game.get("period", 0)
            clock = game.get("clock", "")
            status = game["status"]
            score = f"{game['away_score']}-{game['home_score']}"

            prev = game_state.get(gid)

            if prev is None:
                # First time seeing this game
                if status == "live":
                    glog.info("[%s] Game in progress — P%d %s (%s)", label, period, clock, score)
                elif status == "scheduled":
                    glog.info("[%s] Scheduled", label)
                game_state[gid] = {"status": status, "period": period, "clock": clock}
                continue

            prev_status = prev["status"]
            prev_period = prev["period"]
            prev_clock = prev["clock"]

            # Status transitions
            if prev_status == "scheduled" and status == "live":
                glog.info("[%s] TIPOFF — game started", label)

            if prev_status == "live" and status == "final":
                glog.info("[%s] FINAL — %s", label, score)

            # Period transitions (while live)
            if status == "live" and prev_period != period:
                if period == 2 and prev_period == 1:
                    glog.info("[%s] Q1 ended — Q2 started (%s)", label, score)
                elif period == 3 and prev_period == 2:
                    glog.info("[%s] HALFTIME — Q2 ended (%s)", label, score)
                elif period == 3 and prev_period != 3:
                    glog.info("[%s] Q3 started (%s)", label, score)
                elif period == 4 and prev_period == 3:
                    glog.info("[%s] Q3 ended — Q4 started (%s)", label, score)
                elif period > 4:
                    ot_num = period - 4
                    glog.info("[%s] OT%d started (%s)", label, ot_num, score)

            # Clock transition: quarter ended (clock went from non-empty to empty)
            if status == "live" and prev_period == period and prev_clock and not clock:
                if period == 1:
                    glog.info("[%s] Q1 ended (%s)", label, score)
                elif period == 2:
                    glog.info("[%s] HALFTIME — Q2 ended (%s)", label, score)
                elif period == 3:
                    glog.info("[%s] Q3 ended (%s)", label, score)
                elif period == 4:
                    glog.info("[%s] Q4 ended (%s)", label, score)
                elif period > 4:
                    glog.info("[%s] OT%d ended (%s)", label, period - 4, score)

            game_state[gid] = {"status": status, "period": period, "clock": clock}

        # Checkpoint alerts (only for live games)
        for game in live_games:
            gid = game["game_id"]
            period = game.get("period", 0)
            clock = game.get("clock", "")
            home = game["home_team"]
            away = game["away_team"]
            label = _game_label(game)

            if gid not in notified:
                notified[gid] = set()

            # Detect halftime: period 2 ended (clock empty) or already in Q3+
            if "halftime" not in notified[gid]:
                is_halftime = (period == 2 and not clock) or period >= 3
                if is_halftime:
                    glog.info("[%s] >>> Sending HALFTIME alert", label)
                    _send_checkpoint_alert(
                        game, profiles_ha, "HALFTIME", z, conf_pct,
                    )
                    notified[gid].add("halftime")

            # Detect Q4 start: period 3 ended (clock empty) or already in Q4+
            if "q4" not in notified[gid]:
                is_q4 = (period == 3 and not clock) or period >= 4
                if is_q4:
                    glog.info("[%s] >>> Sending Q4 START alert", label)
                    _send_checkpoint_alert(
                        game, profiles_ha, "Q4 START", z, conf_pct,
                    )
                    notified[gid].add("q4")

        stop_event.wait(poll_interval)


def _send_checkpoint_alert(
    game: dict,
    profiles_ha: dict,
    label: str,
    z: float,
    conf_pct: int,
) -> None:
    """Build projection and send a Telegram alert for a checkpoint."""
    from bot.app import _send_telegram

    home = game["home_team"]
    away = game["away_team"]
    hs = game["home_score"]
    as_ = game["away_score"]
    total = hs + as_

    proj = project_halftime_total(game, profiles_ha)
    if not proj:
        log.warning("[NBA-ALERTS] No projection for %s@%s — skipping "
                     "(period=%s status=%s home_prof=%s away_prof=%s)",
                     away, home, game.get("period"), game.get("status"),
                     f"{home}|home" in profiles_ha, f"{away}|away" in profiles_ha)
        return

    projected = proj["projected_total"]
    cal_std = proj["calibrated_std"]
    pace = proj["dampened_pace"]

    over_line = round(projected - z * cal_std, 1)
    under_line = round(projected + z * cal_std, 1)

    msg = (
        f"<b>{label}: {away} {as_} - {home} {hs}</b>\n"
        f"Total: {total} | Pace: {pace:.2f} | Projected: {projected} ±{cal_std}\n\n"
        f"<b>Game Total:</b>\n"
        f"  <b>OVER {over_line}</b> — profitable up to ~{conf_pct}¢\n"
        f"  <b>UNDER {under_line}</b> — profitable up to ~{conf_pct}¢\n"
    )

    # Per-team score projections
    hp = profiles_ha.get(f"{home}|home")
    ap = profiles_ha.get(f"{away}|away")
    if hp and ap:
        completed = min(game.get("period", 0), 4)
        q_avg = ["q1_avg", "q2_avg", "q3_avg", "q4_avg"]
        q_std = ["q1_std", "q2_std", "q3_std", "q4_std"]

        home_so_far = sum(game.get(f"home_q{q}", 0) for q in range(1, completed + 1))
        away_so_far = sum(game.get(f"away_q{q}", 0) for q in range(1, completed + 1))

        home_remaining = sum(hp[q_avg[q]] for q in range(completed, 4))
        away_remaining = sum(ap[q_avg[q]] for q in range(completed, 4))

        home_var = sum(hp[q_std[q]] ** 2 for q in range(completed, 4))
        away_var = sum(ap[q_std[q]] ** 2 for q in range(completed, 4))
        home_std = home_var ** 0.5 * SIGMA_INFLATION
        away_std = away_var ** 0.5 * SIGMA_INFLATION

        home_proj = round(home_so_far + home_remaining, 1)
        away_proj = round(away_so_far + away_remaining, 1)

        home_over = round(home_proj - z * home_std, 1)
        home_under = round(home_proj + z * home_std, 1)
        away_over = round(away_proj - z * away_std, 1)
        away_under = round(away_proj + z * away_std, 1)

        msg += (
            f"\n<b>{home}</b> (home): {home_proj} "
            f"[OVER {home_over} / UNDER {home_under}]\n"
            f"<b>{away}</b> (away): {away_proj} "
            f"[OVER {away_over} / UNDER {away_under}]\n"
        )

    msg += f"({conf_pct}% confidence lines)"

    log.info("[NBA-ALERTS] %s: %s %d - %s %d | proj=%.1f ±%.1f | OVER %.1f / UNDER %.1f",
             label, away, as_, home, hs, projected, cal_std, over_line, under_line)
    if hp and ap:
        log.info("[NBA-ALERTS] %s: %s(home) proj=%.1f [OVER %.1f / UNDER %.1f] | "
                 "%s(away) proj=%.1f [OVER %.1f / UNDER %.1f]",
                 label, home, home_proj, home_over, home_under,
                 away, away_proj, away_over, away_under)

    _send_telegram(msg)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import os

    from dotenv import load_dotenv
    from paths import project_path
    load_dotenv(project_path(".env.demo"))

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="NBA edge scanner")
    parser.add_argument("--scan", action="store_true", help="Scan for live edges")
    parser.add_argument("--watch", action="store_true", help="Continuous watch mode")
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--projections", action="store_true",
                        help="Show projections for all live games")
    parser.add_argument("--alerts", action="store_true",
                        help="Watch live games and send Telegram alerts at halftime/Q4")
    parser.add_argument("--confidence", type=float, default=0.90,
                        help="Confidence level for alert betting lines (default: 0.90)")
    args = parser.parse_args()

    from nba.data import get_live_scoreboard, build_team_quarter_profiles

    if args.alerts:
        watch_checkpoints(confidence=args.confidence)

    elif args.projections:
        profiles = build_team_quarter_profiles()
        games = get_live_scoreboard()
        live = [g for g in games if g["status"] == "live"]
        if not live:
            print("No live games right now.")
            sys.exit(0)

        for g in live:
            print(f"\n{'='*60}")
            print(f"{g['away_team']} {g['away_score']}  @  "
                  f"{g['home_team']} {g['home_score']}  (P{g['period']} {g['clock']})")

            tp = project_game_total(g, profiles)
            if tp:
                print(f"  Total: projected={tp['projected_total']:.1f} "
                      f"[{tp['lo_band']:.0f}-{tp['hi_band']:.0f}] "
                      f"pace={tp['pace_factor']:.2f}")

            sp = project_spread(g, profiles)
            if sp:
                sign = "+" if sp["projected_margin"] >= 0 else ""
                print(f"  Spread: {g['home_team']} {sign}{sp['projected_margin']:.1f} "
                      f"(±{sp['confidence_band']}) "
                      f"[{g['home_team']}={sp['home_projected']:.0f} "
                      f"{g['away_team']}={sp['away_projected']:.0f}]")

    elif args.scan:
        from bot.app import _get_client
        client = _get_client()
        profiles = build_team_quarter_profiles()
        edges = find_live_edges(client, profiles)
        if not edges:
            print("No edges found (no live games or no matching Kalshi markets).")
        else:
            print(f"\nFound {len(edges)} edges:\n")
            for e in edges:
                print(f"  {e['game']:12s} {e['market_type']:8s} {e['side']:20s} "
                      f"BUY {e['buy_side']:3s} @ {e['kalshi_price']}¢  "
                      f"proj={e['projected_prob']:.1f}%  edge={e['edge_pct']:.1f}%  "
                      f"{e['detail']}")

    elif args.watch:
        import threading
        from bot.app import _get_client
        client = _get_client()
        stop = threading.Event()
        try:
            nba_watch_loop(None, client, stop, dry_run=args.dry_run)
        except KeyboardInterrupt:
            stop.set()
            print("\nStopped.")
