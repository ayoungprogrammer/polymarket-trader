#!/usr/bin/env python3
"""Weather betting strategy.

Claude decides **when** to bet (has the peak passed?).
The bracket model decides **what** bracket to bet on (target temp).

Usage:
    cd src && python -m weather.strategy --site KLAX
    cd src && python -m weather.strategy --site KMIA --market low
"""

from __future__ import annotations

import os

import json
import logging
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import pandas as pd

from paths import project_path
from weather.forecast import ForecastIngestion
from weather.prediction import (
    compute_momentum,
    extrapolate_momentum,
)
from weather.observations import SynopticIngestion, fetch_solar_noon, fetch_sun_times
from weather.bracket_model import load_model as load_bracket_model, get_probability
from weather.backtest_rounding import extract_regression_features

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Momentum params (used by both strategy and watch loop in app.py)
# ---------------------------------------------------------------------------

# MA crossover momentum params: (cross_threshold, margin_below_peak)
# Either set triggers LOCKED — strong crossover OR weak crossover with large margin
MOMENTUM_PARAMS_FAST = (-0.5, 2.0)   # strong crossover, modest margin
MOMENTUM_PARAMS_WIDE = (-0.2, 3.0)   # weak crossover, large margin (requires 3-reading confirmation)
WIDE_CONFIRM_COUNT = 3               # consecutive readings ma_cross must stay below threshold

# Per-site optimal params from backtest (cross_threshold, margin, confirm_count)
SITE_MOMENTUM_PARAMS = {
    "KLAX": (-1.0, 1.5, 1),
    "KMIA": (-0.5, 1.5, 1),
    "KSFO": (-1.0, 1.5, 1),
    "KMDW": (-1.0, 1.5, 1),
    "KDEN": (-1.0, 1.5, 3),
    "KPHX": (-1.0, 1.5, 1),
    "KOKC": (-0.5, 1.5, 1),
    "KATL": (-1.0, 1.5, 1),
    "KDFW": (-0.5, 1.5, 1),
    "KSAT": (-1.0, 1.5, 3),
    "KHOU": (-0.5, 1.5, 1),
    "KMSP": (-0.5, 1.5, 3),
    "KDCA": (-1.0, 1.5, 1),
    "KAUS": (-0.5, 1.5, 1),
    "KBOS": (-0.5, 1.5, 1),
    "KPHL": (-0.5, 1.5, 1),
}

# ---------------------------------------------------------------------------
# Bracket model cache
# ---------------------------------------------------------------------------

_bracket_model_cache = None


def _get_bracket_model():
    global _bracket_model_cache
    if _bracket_model_cache is None:
        try:
            _bracket_model_cache = load_bracket_model()
        except FileNotFoundError:
            log.warning("No trained bracket model found")
    return _bracket_model_cache


# ---------------------------------------------------------------------------
# Claude peak-detection prompt
# ---------------------------------------------------------------------------

PEAK_DECISION_TOOL = {
    "name": "make_bet_decision",
    "description": "Decide whether to bet and which bracket to select.",
    "input_schema": {
        "type": "object",
        "properties": {
            "reasoning": {"type": "string", "description": "Step-by-step analysis of whether the peak has passed and which bracket to select"},
            "peak_locked": {"type": "boolean", "description": "True if you believe the daily peak has been reached"},
            "bracket": {"type": "array", "items": {"type": "integer"}, "description": "The [low, high] °F bracket to bet on, e.g. [69, 70]. Required if peak_locked=true. Copy from bracket model output."},
            "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        },
        "required": ["reasoning", "peak_locked", "confidence"],
    },
}

PEAK_SYSTEM_PROMPT = """\
You are an expert weather analyst betting on daily high/low temperature markets.

You have two jobs:
1. Decide if the daily temperature peak (for highs) or trough (for lows) has been reached.
2. If yes, select the bracket (target_temp) to bet on using the bracket model probabilities.

## When to say peak_locked=true (HIGH confidence)
- Temps are clearly declining (negative momentum) and well below the observed peak
- It's late afternoon/evening with no forecast risk of temps rebounding
- Remaining forecast hours are all well below the observed max (>3°F margin)

## When to say peak_locked=true (MEDIUM confidence)
- Temps are declining but margin is tight (<3°F)
- Momentum is negative but slow — peak likely passed but not certain

## When to say peak_locked=false
- Temps are still rising or flat — peak hasn't passed yet
- Forecast shows remaining hours could exceed observed max
- Time of day is too early (before 2 PM local for highs, before 10 AM local for lows)

## Selecting target_temp
When peak_locked=true, set target_temp to the bracket the market will settle on. \
Use the bracket model probabilities as your primary guide — set bracket to the \
[low, high] pair with the highest probability unless you have strong reason to \
override (e.g. forecast shows a later peak that the model hasn't seen yet).

## Key Data Points
- **Momentum**: MA cross < 0 means short-term avg is below long-term avg (declining). \
The more negative, the stronger the decline.
- **Forecast remaining max**: if remaining hours forecast temps above observed max, peak is NOT locked.
- **6h METAR max**: most precise observation of the max so far (0.1°C precision).
- **Bracket model**: ML model predicting settlement bracket from live observations. \
~82% accuracy on 3-class offset prediction.
"""


def _call_claude_for_peak_decision(user_message: str) -> dict:
    """Call Claude API to decide if peak has been reached and which bracket to bet.

    Returns dict with {peak_locked, target_temp, confidence, reasoning}.
    On any error, returns peak_locked=False — never bet on failure.
    """
    try:
        import anthropic
    except ImportError:
        log.error("anthropic package not installed — run: pip install anthropic")
        return {"peak_locked": False, "confidence": "low", "reasoning": "anthropic package not installed"}

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        log.error("ANTHROPIC_API_KEY not set in environment")
        return {"peak_locked": False, "confidence": "low", "reasoning": "API key not configured"}

    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=1024,
            system=PEAK_SYSTEM_PROMPT,
            tools=[PEAK_DECISION_TOOL],
            tool_choice={"type": "tool", "name": "make_bet_decision"},
            messages=[{"role": "user", "content": user_message}],
        )
    except Exception as e:
        log.error(f"Anthropic API error: {e}")
        return {"peak_locked": False, "confidence": "low", "reasoning": f"API error: {e}"}

    log.debug(f"Claude stop_reason: {response.stop_reason}")
    for block in response.content:
        if block.type == "tool_use" and block.name == "make_bet_decision":
            decision = block.input
            log.info(f"Claude decision: {json.dumps(decision, indent=2)}")
            return decision

    log.warning(f"Claude response contained no tool_use block — treating as not locked")
    return {"peak_locked": False, "confidence": "low", "reasoning": "No tool_use in response"}


# ---------------------------------------------------------------------------
# Bracket parsing helper
# ---------------------------------------------------------------------------

def _parse_all_brackets(markets: List[Dict]) -> List[Tuple[str, int, int]]:
    """Extract (ticker, lo, hi) bounds from all markets.

    Reuses the same regex patterns as find_matching_bracket.
    Returns list of (ticker, lo_f, hi_f) for each parseable market.
    """
    results: List[Tuple[str, int, int]] = []
    for market in markets:
        title = market.get("title", "") + " " + market.get("subtitle", "")
        ticker = market.get("ticker", "")

        range_match = re.search(r"(\d+)°?\s*(?:to|-)\s*(\d+)°?", title, re.IGNORECASE)
        if range_match:
            results.append((ticker, int(range_match.group(1)), int(range_match.group(2))))
            continue

        # Skip open-ended "or above" / "or below" brackets — not useful for model
        above_match = re.search(r"(\d+)°?\s*or\s*(above|higher|more)", title, re.IGNORECASE)
        if above_match:
            continue

        below_match = re.search(r"(\d+)°?\s*or\s*(below|lower|less)", title, re.IGNORECASE)
        if below_match:
            continue

        floor_strike = market.get("floor_strike")
        cap_strike = market.get("cap_strike")
        if floor_strike is not None and cap_strike is not None:
            results.append((ticker, int(float(floor_strike)), int(float(cap_strike))))

    return results


# ---------------------------------------------------------------------------
# Main strategy function
# ---------------------------------------------------------------------------

def run_strategy(
    market_type: str,
    label: str,
    series_ticker: str,
    site: str = "KLAX",
    client: Optional[object] = None,
    sim_now: Optional[datetime] = None,
) -> Optional[dict]:
    """Weather betting strategy: Claude for timing, bracket model for target.

    1. Fetches observations, runs bracket model for probabilities
    2. Computes momentum indicators
    3. Sends weather data to Claude to decide if peak has been reached
    4. If peak locked, uses bracket model's top prediction as target_temp

    Returns dict with {action, target_temp, confidence, reasoning}.
    If *client* is None, bracket model market fetch is skipped.
    """
    from weather.forecast import STATIONS as FORECAST_STATIONS
    station_tz = ZoneInfo(FORECAST_STATIONS[site][2]) if site in FORECAST_STATIONS else ZoneInfo("America/Los_Angeles")
    now = sim_now or datetime.now(station_tz)
    if sim_now:
        log.info(f"=== Strategy [{site}] (simulated time: {now.isoformat()}) ===")
    else:
        log.info(f"=== Strategy [{site}] ===")

    mom_df = pd.DataFrame()
    forecast_df = pd.DataFrame()
    bracket_target_temp = None
    bracket_model_lines = ""
    momentum_locked = False
    current_cross = 0.0
    current_rate = 0.0
    observed_max = 0.0
    readings_after_peak = 0
    sun_times = None
    metar_6h_max = None
    top_bracket = None
    bracket_probs = None
    reasons = []

    # 1. Observations (fetch full day so we don't miss the peak)
    try:
        si = SynopticIngestion(site)
        obs_df = si.fetch_live_weather(hours=24)
        # Filter to simulated time if set
        if sim_now and not obs_df.empty and "timestamp" in obs_df.columns:
            sim_iso = now.isoformat()
            obs_df = obs_df[obs_df["timestamp"] <= sim_iso].reset_index(drop=True)
            log.info(f"Filtered observations to before {sim_iso}: {len(obs_df)} rows")
        if not obs_df.empty and "temperature_f" in obs_df.columns:
            current_temp = obs_df["temperature_f"].dropna().iloc[-1]
            observed_max = obs_df["temperature_f"].max()
            observed_min = obs_df["temperature_f"].min()
            peak_idx = obs_df["temperature_f"].idxmax()
            readings_after_peak = len(obs_df) - 1 - peak_idx
            log.info(f"Current: {current_temp}°F | Max: {observed_max}°F | Min: {observed_min}°F")

            # 6h METAR max/min (0.1°C precision — most accurate for CLI prediction)
            if "max_temp_6h_f" in obs_df.columns:
                metar6h = obs_df[["timestamp", "max_temp_6h_f", "min_temp_6h_f"]].dropna(subset=["max_temp_6h_f"])
                if not metar6h.empty:
                    metar_6h_max = metar6h["max_temp_6h_f"].max()
                    log.info(f"6h METAR max: {metar_6h_max}°F")

            # Bracket model probabilities
            try:
                bmodel = _get_bracket_model()
                if bmodel is not None and not obs_df.empty:
                    # Filter obs to today (station local time)
                    today_str = now.strftime("%Y-%m-%d")
                    today_obs = obs_df.copy()
                    if "timestamp" in today_obs.columns:
                        today_obs["_dt"] = pd.to_datetime(today_obs["timestamp"], utc=True)
                        today_obs["_date"] = today_obs["_dt"].dt.tz_convert(station_tz).dt.date.astype(str)
                        today_obs = today_obs[today_obs["_date"] == today_str].reset_index(drop=True)
                    if len(today_obs) >= 20:
                        # Build day_df in same format as load_site_history
                        day_df = today_obs.drop(columns=["_dt", "_date"], errors="ignore")
                        day_df["temperature_c"] = pd.to_numeric(day_df.get("temperature_c"), errors="coerce")
                        day_df["temperature_f"] = pd.to_numeric(day_df.get("temperature_f"), errors="coerce")
                        day_df["ts"] = pd.to_datetime(day_df["timestamp"].str[:19])
                        day_df["date"] = day_df["ts"].dt.date
                        for col in ["max_temp_24h_f", "wind_speed_mph", "dewpoint_f",
                                    "relative_humidity_pct", "sea_level_pressure",
                                    "pressure_tendency"]:
                            if col in day_df.columns:
                                day_df[col] = pd.to_numeric(day_df[col], errors="coerce")

                        # Fetch sun times for today (sunrise, solar noon, sunset)
                        solar_noon_hour = None
                        sun_times = None
                        coords = FORECAST_STATIONS.get(site)
                        if coords:
                            sun_times = fetch_sun_times(coords[0], coords[1], today_str)
                            if sun_times:
                                solar_noon_hour = sun_times.get("solar_noon")

                        feats = extract_regression_features(day_df, solar_noon_hour=solar_noon_hour)
                        if feats is not None:
                            # Build brackets from markets if client available,
                            # otherwise synthesize from naive rounding
                            brackets_parsed = []
                            if client is not None:
                                today_suffix = now.strftime("%y%b%d").upper()
                                resp = client.get_markets(series_ticker)
                                today_markets = [m for m in resp.get("markets", [])
                                                 if today_suffix in m.get("ticker", "")]
                                all_brackets = _parse_all_brackets(today_markets)
                                # Filter to brackets within 3°F of naive temp
                                naive_f = round(float(feats.get("max_c", 0)) * 9.0 / 5.0 + 32.0)
                                brackets_parsed = [
                                    (tk, lo, hi) for tk, lo, hi in all_brackets
                                    if abs((lo + hi) / 2.0 - naive_f) <= 3
                                ]
                            if not brackets_parsed:
                                # Synthesize Kalshi-style 2°F brackets from naive rounding
                                max_c = feats.get("max_c")
                                if max_c is not None:
                                    naive_f = round(float(max_c) * 9.0 / 5.0 + 32.0)
                                    # Kalshi brackets are even-odd pairs: [64,65], [66,67], [68,69], ...
                                    # Find the bracket containing naive_f
                                    if naive_f % 2 == 0:
                                        center_lo = naive_f
                                    else:
                                        center_lo = naive_f - 1
                                    for lo in [center_lo - 2, center_lo, center_lo + 2]:
                                        brackets_parsed.append((f"[{lo},{lo+1}]", lo, lo + 1))
                            if brackets_parsed:
                                bracket_tuples = [(lo, hi) for _, lo, hi in brackets_parsed]
                                ticker_map = {(lo, hi): tk for tk, lo, hi in brackets_parsed}
                                bprobs = get_probability(bmodel, feats, bracket_tuples,
                                                         metar_6h_f=metar_6h_max)
                                bprobs = [bp for bp in bprobs if bp["prob"] >= 0.005]
                                bprobs.sort(key=lambda x: -x["prob"])
                                if bprobs:
                                    # Use top bracket prediction as target temp
                                    top = bprobs[0]
                                    lo, hi = top["bracket"]
                                    bracket_target_temp = (lo + hi) / 2.0 if lo != hi else float(lo)
                                    top_bracket = [lo, hi]
                                    bracket_probs = bprobs
                                    lines = []
                                    for bp in bprobs:
                                        blo, bhi = bp["bracket"]
                                        s1 = bp.get("stage1_prob", 0.0)
                                        log.info(f"  Bracket model: [{blo}, {bhi}]°F: "
                                                 f"stage1={s1:.0%} → final={bp['prob']:.0%}")
                                        lines.append(f"  [{blo}, {bhi}]°F: {bp['prob']:.0%}")
                                    bracket_model_lines = (
                                        "Bracket model (2-stage ML, ~82% accuracy, brackets are inclusive):\n"
                                        + "\n".join(lines)
                                    )
            except Exception as e:
                log.debug(f"Bracket model unavailable: {e}")
        else:
            log.warning("No observation data available")
            obs_df = pd.DataFrame()
    except Exception as e:
        log.warning(f"Could not fetch observations: {e}")
        obs_df = pd.DataFrame()

    # 2. Momentum (MA crossover)
    try:
        if not obs_df.empty and "temperature_f" in obs_df.columns:
            mom_df = compute_momentum(obs_df)
            crosses = mom_df["ma_cross"].dropna()
            rates = mom_df["rate_f_per_hr"].dropna()
            if not crosses.empty:
                current_cross = float(crosses.iloc[-1])
                current_rate = float(rates.iloc[-1]) if not rates.empty else 0.0

                site_cross, site_margin, site_confirm = SITE_MOMENTUM_PARAMS.get(
                    site, (MOMENTUM_PARAMS_FAST[0], MOMENTUM_PARAMS_FAST[1], 1))
                confirm_count = 0
                for val in reversed(crosses.tolist()):
                    if val < site_cross:
                        confirm_count += 1
                    else:
                        break

                momentum_locked = confirm_count >= site_confirm
                log.info(f"Momentum: cross={current_cross:+.2f}°F rate={current_rate:+.2f}°F/hr "
                         f"confirm={confirm_count}/{site_confirm} "
                         f"{'LOCKED' if momentum_locked else 'NOT LOCKED'}")
    except Exception as e:
        log.warning(f"Could not compute momentum: {e}")

    # 3. Forecast
    try:
        fi = ForecastIngestion(site)
        forecast_df = fi.fetch_forecast()
        if not forecast_df.empty and "temperature_f" in forecast_df.columns:
            forecast_peak = forecast_df["temperature_f"].max()
            peak_idx = forecast_df["temperature_f"].idxmax()
            peak_time = forecast_df.loc[peak_idx, "timestamp"]
            log.info(f"Forecast peak: {forecast_peak}°F at {peak_time}")
    except Exception as e:
        log.warning(f"Could not fetch forecast: {e}")

    # Generate momentum chart
    if not mom_df.empty:
        try:
            from weather.prediction import plot_momentum
            chart_path = project_path("charts", "weather", f"momentum_{site}.png")
            os.makedirs(os.path.dirname(chart_path), exist_ok=True)
            plot_momentum(mom_df, site, output=chart_path,
                          forecast_df=forecast_df if not forecast_df.empty else None,
                          locked_rate=MOMENTUM_PARAMS_FAST[0],
                          likely_rate=MOMENTUM_PARAMS_WIDE[0],
                          margin_threshold=MOMENTUM_PARAMS_FAST[1],
                          sun_times=sun_times,
                          metar_6h_f=metar_6h_max,
                          bracket=top_bracket,
                          bracket_probs=bracket_probs)
            print(f"Saved momentum chart to {chart_path}")
        except Exception as e:
            log.warning(f"Could not generate momentum chart: {e}")

    # --- Build context for Claude peak decision ---
    peak_sections = []
    peak_sections.append(f"Market: {series_ticker} (daily {label})")
    peak_sections.append(f"Current time: {now.isoformat()}")

    if not obs_df.empty and "temperature_f" in obs_df.columns:
        last_readings = obs_df[["timestamp", "temperature_f"]].tail(20).to_string(index=False)
        obs_summary = f"Current: {obs_df['temperature_f'].dropna().iloc[-1]}°F | Observed max: {observed_max}°F"

        # Most recent 6h METAR reported high (0.1°C precision)
        metar6h_line = ""
        if "max_temp_6h_f" in obs_df.columns:
            metar6h = obs_df[["timestamp", "max_temp_6h_f"]].dropna(subset=["max_temp_6h_f"])
            if not metar6h.empty:
                latest_metar_row = metar6h.iloc[-1]
                metar6h_max = metar6h["max_temp_6h_f"].max()
                obs_summary += f" | 6h METAR max: {metar6h_max}°F"
                metar6h_line = (
                    f"\n6h METAR reported high: {latest_metar_row['max_temp_6h_f']}°F "
                    f"(at {latest_metar_row['timestamp']})"
                )

        peak_sections.append(
            f"Observations:\n{obs_summary}{metar6h_line}"
            f"\nLast 20 readings:\n{last_readings}"
        )

    if not mom_df.empty:
        from weather.prediction import MA_SHORT_MIN, MA_LONG_MIN
        ma_short_val = mom_df["ma_short"].dropna().iloc[-1] if "ma_short" in mom_df.columns else None
        ma_long_val = mom_df["ma_long"].dropna().iloc[-1] if "ma_long" in mom_df.columns else None
        ma_vals_line = ""
        if ma_short_val is not None and ma_long_val is not None:
            ma_vals_line = f"  {MA_SHORT_MIN}min MA: {ma_short_val:.1f}°F | {MA_LONG_MIN}min MA: {ma_long_val:.1f}°F\n"
        peak_sections.append(
            f"Momentum ({MA_SHORT_MIN}min MA − {MA_LONG_MIN}min MA):\n"
            f"{ma_vals_line}"
            f"  MA cross: {current_cross:+.2f}°F (negative = declining temps)\n"
            f"  Rate of change: {current_rate:+.2f}°F/hr\n"
            f"  Readings since peak: {readings_after_peak}"
        )

    if not forecast_df.empty and "temperature_f" in forecast_df.columns:
        # Show remaining forecast hours
        remaining = []
        for _, row in forecast_df.iterrows():
            try:
                ts = datetime.fromisoformat(row["timestamp"])
                if ts > now:
                    remaining.append(row)
            except (ValueError, TypeError):
                continue
        if remaining:
            rem_df = pd.DataFrame(remaining)
            rem_max = rem_df["temperature_f"].max()
            peak_sections.append(
                f"Forecast: remaining hours max={rem_max}°F ({len(remaining)} hours left), "
                f"today's forecast peak={forecast_df['temperature_f'].max()}°F")
        else:
            peak_sections.append("Forecast: no remaining hours today")

    if bracket_model_lines:
        peak_sections.append(bracket_model_lines)

    user_message = "\n\n".join(peak_sections)
    print(f"\n{'='*60}\nCLAUDE PROMPT\n{'='*60}\n{user_message}\n{'='*60}\n")

    # --- Ask Claude: has peak been reached? Which bracket? ---
    decision = _call_claude_for_peak_decision(user_message)
    peak_locked = decision.get("peak_locked", False)
    confidence = decision.get("confidence", "low")
    reasoning = decision.get("reasoning", "")

    log.info(f"Peak decision: locked={peak_locked} | confidence={confidence}")
    log.info(f"Reasoning: {reasoning}")

    if not peak_locked:
        return {"action": "no_bet", "confidence": confidence, "reasoning": reasoning}

    # Use Claude's bracket selection, fall back to bracket model top, then observed max
    bracket = decision.get("bracket")
    if bracket and len(bracket) == 2:
        target_temp = (bracket[0] + bracket[1]) / 2.0 if bracket[0] != bracket[1] else float(bracket[0])
        log.info(f"BET: [{bracket[0]}, {bracket[1]}]°F → target {target_temp}°F (Claude selection) | confidence: {confidence}")
    elif bracket_target_temp is not None:
        target_temp = bracket_target_temp
        bracket = [int(target_temp), int(target_temp)]
        log.info(f"BET: [{bracket[0]}, {bracket[1]}]°F → target {target_temp}°F (bracket model fallback) | confidence: {confidence}")
    else:
        target_temp = round(observed_max)
        bracket = [int(target_temp), int(target_temp)]
        log.info(f"BET: [{bracket[0]}, {bracket[1]}]°F → target {target_temp}°F (observed max fallback) | confidence: {confidence}")

    return {"action": "bet", "bracket": bracket, "target_temp": target_temp,
            "confidence": confidence, "reasoning": reasoning}


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    import argparse
    from bot.app import STATIONS, KalshiClient
    from dotenv import load_dotenv
    from paths import project_path

    load_dotenv(project_path(".env.demo"))

    parser = argparse.ArgumentParser(description="Run weather betting strategy standalone")
    parser.add_argument("--site", default="KLAX", help="ICAO station code (default: KLAX)")
    parser.add_argument("--market", default="high", choices=["high", "low"],
                        help="Market type (default: high)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    site = args.site.upper()
    market_type = args.market

    # Derive series ticker from STATIONS
    station_info = STATIONS.get(site)
    if not station_info:
        log.error(f"Unknown station: {site}")
        return
    city_name, kalshi_suffix = station_info
    prefix = f"KX{market_type.upper()}"
    series_ticker = f"{prefix}{kalshi_suffix}"
    label = "high temperature" if market_type == "high" else "low temperature"

    # Create Kalshi client for bracket fetching
    client = None
    api_key = os.getenv("KALSHI_API_KEY_ID")
    pk_path = os.getenv("KALSHI_PRIVATE_KEY_PATH")
    if api_key and pk_path:
        try:
            client = KalshiClient(api_key, pk_path, env="demo")
            log.info("Kalshi client initialized (demo)")
        except Exception as e:
            log.warning(f"Could not init Kalshi client: {e}")

    log.info(f"Running strategy for {city_name} ({site}) — {series_ticker}")

    decision = run_strategy(
        market_type=market_type,
        label=label,
        series_ticker=series_ticker,
        site=site,
        client=client,
    )

    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
