#!/usr/bin/env python3
"""Kalshi Weather Bot — LA High Temp.

Fetches the NWS observation for LAX, and if the 6-hour report timestamp
is past 3 PM PT, places a $10 bet on the Kalshi bracket matching the
observed max temp. One-shot, cron-friendly.
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import re
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from dotenv import load_dotenv

from forecast import ForecastIngestion
from prediction import compute_momentum, extrapolate_momentum
from weather import (
    SynopticIngestion,
    fetch_nws_observation,
    is_past_3pm_pacific,
    parse_6hr_section,
)

load_dotenv(".env.demo")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

KALSHI_BASE_URLS = {
    "prod": "https://api.elections.kalshi.com",
    "demo": "https://demo-api.kalshi.co",
}
API_PREFIX = "/trade-api/v2"

SERIES_TICKERS = {
    "high": "KXHIGHLAX",
    "low": "KXLOWLAX",
}
BET_AMOUNT_CENTS = 1000  # $10 in cents

# ---------------------------------------------------------------------------
# Claude strategy constants
# ---------------------------------------------------------------------------

DECISION_TOOL = {
    "name": "make_bet_decision",
    "description": "Submit your betting decision with reasoning.",
    "input_schema": {
        "type": "object",
        "properties": {
            "reasoning": {"type": "string", "description": "Step-by-step analysis: 1) predict CLI temp from data, 2) identify which bracket it falls into, 3) evaluate price vs your confidence, 4) decide bet or no_bet"},
            "action": {"type": "string", "enum": ["bet", "no_bet"]},
            "ticker": {"type": "string", "description": "Kalshi ticker to bet on (required if action=bet)"},
            "side": {"type": "string", "enum": ["yes", "no"], "description": "Which side to buy (required if action=bet)"},
            "target_temp": {"type": "number", "description": "Whole °F you predict the CLI will report"},
            "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        },
        "required": ["reasoning", "action", "confidence"],
    },
}

CLAUDE_SYSTEM_PROMPT = """\
You are an expert weather betting analyst for Kalshi KXHIGHLAX/KXLOWLAX markets \
(LA daily high/low temperature).

## How These Markets Work
- Kalshi settles on the NWS CLI (Climatological Report) for LAX
- The CLI reports whole-degree Fahrenheit (rounded from precise °C observations)
- The final CLI publishes the morning AFTER the measurement day
- You are betting BEFORE the market closes, so you are always predicting

## Temperature Data Precision
- Synoptic 5-min observations: whole-degree C (~2°F uncertainty), but covers ALL hours
- METAR hourly (:53) observations: 0.1°C precision via T-group (most accurate)
- The 6h METAR max/min (1snTTT/2snTTT remarks): 0.1°C, continuously tracked by ASOS sensor
- CLI rounds the precise observation to whole °F

## IMPORTANT: Understanding 6h METAR Windows
- 6h METAR reports drop at :53 and cover the PRECEDING 6 hours only
- Reports: 03:53 (covers 22-04), 09:53 (covers 04-10), 15:53 (covers 10-16), 21:53 (covers 16-22)
- If the 5-min observed max is HIGHER than the latest 6h METAR max, it means the peak \
happened AFTER the latest METAR window — this is normal, not a discrepancy
- The daily high for CLI comes from the best reading across the entire day, NOT just METAR reports
- Use the 5-min observed max as the likely daily high; use 6h METAR max only when it \
covers the peak window (e.g., the 15:53 report for an afternoon peak)

## When to Bet (HIGH confidence)
- The 6h METAR max has dropped AND remaining forecast hours are all well below (>3°F margin)
- Momentum is strongly negative (rate ≤ -2°F/hr) and current temp is well below the peak
- It's late afternoon/evening and temps are clearly declining with no forecast risk

## When to Bet (MEDIUM confidence)
- Temps are declining but margin is tight (<3°F) — still reasonable if price is good
- Momentum is negative but slow (-1 to -0.5°F/hr) — peak likely passed but not certain

## When NOT to Bet
- Temps are still rising or flat — peak hasn't passed yet
- Forecast shows remaining hours could exceed observed max/go below observed min
- Time of day is too early (before 2 PM for highs, before 10 AM for lows)
- Rounding is ambiguous (temp is X.5°F — could round either way)
- No bracket has positive expected value at current prices

## Bracket Selection & Pricing
- You are given the Kalshi brackets with YES bid/ask or last trade prices
- Bracket types: "between" (e.g. 81-82°), "greater than" (e.g. >82°), "less than" (e.g. <75°)
- Pick the bracket that your predicted temp falls into
- Evaluate the price: your estimated probability vs the market price
- A YES at 95c means you need >95% confidence to have edge
- A YES at 70c with 90% confidence = good value (20c edge)
- Do NOT bet if the bracket YES price > 95c — no profit margin
- Do NOT bet if the bracket YES price < 30c — likely a bad read
- You can bet YES or NO on any bracket. Betting NO on an unlikely bracket can also be +EV
- When betting, specify the exact ticker and side (yes/no)

## Your Task
Analyze all provided data. Predict the CLI daily high/low temperature. \
Evaluate each bracket's probability vs its market price. If you find a +EV bet, \
specify the ticker and side. Always explain your reasoning including the edge.
"""


# ---------------------------------------------------------------------------
# Kalshi auth helpers
# ---------------------------------------------------------------------------

def load_private_key(path: str):
    with open(path, "rb") as f:
        return serialization.load_pem_private_key(f.read(), password=None)


def sign_request(private_key, timestamp_ms: str, method: str, path: str) -> str:
    """RSA-PSS sign  timestamp_ms + method + path  (no query params)."""
    message = f"{timestamp_ms}{method}{path}".encode("utf-8")
    signature = private_key.sign(
        message,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.MAX_LENGTH,
        ),
        hashes.SHA256(),
    )
    return base64.b64encode(signature).decode("utf-8")


class KalshiClient:
    def __init__(self, api_key_id: str, private_key_path: str, env: str = "demo"):
        self.api_key_id = api_key_id
        self.private_key = load_private_key(private_key_path)
        self.base_url = KALSHI_BASE_URLS[env] + API_PREFIX
        self.session = requests.Session()

    def _headers(self, method: str, path: str) -> dict:
        ts = str(int(time.time() * 1000))
        sig = sign_request(self.private_key, ts, method, path)
        return {
            "KALSHI-ACCESS-KEY": self.api_key_id,
            "KALSHI-ACCESS-TIMESTAMP": ts,
            "KALSHI-ACCESS-SIGNATURE": sig,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _request(self, method: str, path: str, params=None, json_body=None):
        url = self.base_url + path
        headers = self._headers(method.upper(), API_PREFIX + path)
        resp = self.session.request(
            method, url, headers=headers, params=params, json=json_body
        )
        resp.raise_for_status()
        return resp.json()

    def get_markets(self, series_ticker: str, status: str = "open"):
        return self._request("GET", "/markets", params={
            "series_ticker": series_ticker,
            "status": status,
        })

    def get_orderbook(self, ticker: str):
        resp = self._request("GET", f"/markets/{ticker}/orderbook")
        return resp.get("orderbook", resp)

    def get_trades(self, ticker: str, limit: int = 100, cursor: str = ""):
        params = {"ticker": ticker, "limit": limit}
        if cursor:
            params["cursor"] = cursor
        return self._request("GET", "/markets/trades", params=params)

    def place_order(self, ticker: str, side: str, price: int, count: int):
        body = {
            "ticker": ticker,
            "action": "buy",
            "side": side,
            "type": "limit",
            "yes_price": price,
            "count": count,
        }
        return self._request("POST", f"/markets/{ticker}/orders", json_body=body)


# ---------------------------------------------------------------------------
# Forecast-based confidence assessment
# ---------------------------------------------------------------------------

def _fetch_remaining_forecast(
    site: str = "KLAX",
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Fetch forecast and split into (full_df, remaining_df).

    Returns (None, None) if forecast is unavailable.
    """
    try:
        fi = ForecastIngestion(site)
        df = fi.fetch_forecast()
    except Exception as e:
        log.warning(f"Could not fetch forecast: {e}")
        return (None, None)

    if df.empty or "temperature_f" not in df.columns:
        log.warning("Forecast returned no temperature data")
        return (None, None)

    now = datetime.now(ZoneInfo("America/Los_Angeles"))
    remaining = []
    for _, row in df.iterrows():
        try:
            ts = datetime.fromisoformat(row["timestamp"])
            if ts > now:
                remaining.append(row)
        except (ValueError, TypeError):
            continue

    remaining_df = pd.DataFrame(remaining) if remaining else pd.DataFrame()
    return (df, remaining_df)


def assess_high_confidence(
    observed_max_f: float,
    site: str = "KLAX",
) -> Tuple[str, float, Optional[pd.DataFrame]]:
    """Check the NWS hourly forecast to see if the daily high is locked in.

    Compares remaining forecast hours' temperatures against the observed
    6-hour max. Returns (confidence, forecast_extreme, forecast_df) where
    confidence is "HIGH", "MEDIUM", or "LOW".

    HIGH   — all remaining forecast hours are >3°F below observed max
    MEDIUM — remaining hours are below but within 3°F of observed max
    LOW    — at least one remaining hour is forecast at or above observed max
    """
    df, remaining_df = _fetch_remaining_forecast(site)
    if df is None:
        return ("UNKNOWN", 0.0, None)

    if remaining_df.empty:
        log.info("No remaining forecast hours today — high is locked in by end of day")
        return ("HIGH", df["temperature_f"].max(), df)

    remaining_max = remaining_df["temperature_f"].max()
    forecast_high = df["temperature_f"].max()
    margin = observed_max_f - remaining_max

    log.info(f"Forecast high for today: {forecast_high}°F")
    log.info(f"Remaining hours forecast max: {remaining_max}°F ({len(remaining_df)} hours left)")
    log.info(f"Observed 6h max: {observed_max_f}°F — margin: {margin:+.1f}°F")

    if remaining_max >= observed_max_f:
        confidence = "LOW"
        log.warning(f"Forecast shows temps could reach {remaining_max}°F — high may NOT be locked in")
    elif margin < 3.0:
        confidence = "MEDIUM"
        log.info(f"Forecast is close but below observed max (margin {margin:.1f}°F)")
    else:
        confidence = "HIGH"
        log.info(f"All remaining hours well below observed max (margin {margin:.1f}°F)")

    return (confidence, forecast_high, df)


def assess_low_confidence(
    observed_min_f: float,
    site: str = "KLAX",
) -> Tuple[str, float, Optional[pd.DataFrame]]:
    """Check the NWS hourly forecast to see if the daily low is locked in.

    Compares remaining forecast hours' temperatures against the observed
    6-hour min. Returns (confidence, forecast_low, forecast_df) where
    confidence is "HIGH", "MEDIUM", or "LOW".

    HIGH   — all remaining forecast hours are >3°F above observed min
    MEDIUM — remaining hours are above but within 3°F of observed min
    LOW    — at least one remaining hour is forecast at or below observed min
    """
    df, remaining_df = _fetch_remaining_forecast(site)
    if df is None:
        return ("UNKNOWN", 0.0, None)

    if remaining_df.empty:
        log.info("No remaining forecast hours today — low is locked in by end of day")
        return ("HIGH", df["temperature_f"].min(), df)

    remaining_min = remaining_df["temperature_f"].min()
    forecast_low = df["temperature_f"].min()
    margin = remaining_min - observed_min_f

    log.info(f"Forecast low for today: {forecast_low}°F")
    log.info(f"Remaining hours forecast min: {remaining_min}°F ({len(remaining_df)} hours left)")
    log.info(f"Observed 6h min: {observed_min_f}°F — margin: {margin:+.1f}°F above")

    if remaining_min <= observed_min_f:
        confidence = "LOW"
        log.warning(f"Forecast shows temps could drop to {remaining_min}°F — low may NOT be locked in")
    elif margin < 3.0:
        confidence = "MEDIUM"
        log.info(f"Forecast is close but above observed min (margin {margin:.1f}°F)")
    else:
        confidence = "HIGH"
        log.info(f"All remaining hours well above observed min (margin {margin:.1f}°F)")

    return (confidence, forecast_low, df)


# ---------------------------------------------------------------------------
# Peak-tracking strategy
# ---------------------------------------------------------------------------

def peak_track_strategy(
    site: str = "KLAX",
) -> Tuple[str, float, float, Optional[pd.DataFrame]]:
    """Peak-tracking strategy: use forecast peak + live obs declining trend.

    1. Fetch forecast → find forecast peak temp for today
    2. Fetch live Synoptic observations (last 6h)
    3. Find today's observed max from the live data
    4. Check if the last 3 readings are monotonically decreasing AND below
       the forecast peak → the high has passed

    Returns (status, observed_max_f, forecast_peak_f, obs_df) where status is:
      "LOCKED"    — last 3 readings declining and below forecast peak
      "NEAR_PEAK" — temps are near or at the forecast peak, not yet declining
      "TOO_EARLY" — not enough data or temps still rising
      "ERROR"     — could not fetch data
    """
    # Step 1: Forecast peak
    try:
        fi = ForecastIngestion(site)
        forecast_df = fi.fetch_forecast()
    except Exception as e:
        log.warning(f"Could not fetch forecast: {e}")
        return ("ERROR", 0.0, 0.0, None)

    if forecast_df.empty or "temperature_f" not in forecast_df.columns:
        log.warning("Forecast returned no temperature data")
        return ("ERROR", 0.0, 0.0, None)

    forecast_peak = forecast_df["temperature_f"].max()
    peak_idx = forecast_df["temperature_f"].idxmax()
    peak_time = forecast_df.loc[peak_idx, "timestamp"]
    log.info(f"Forecast peak: {forecast_peak}°F at {peak_time}")

    # Step 2: Live observations
    try:
        si = SynopticIngestion(site)
        obs_df = si.fetch_live_weather(hours=6)
    except Exception as e:
        log.warning(f"Could not fetch live observations: {e}")
        return ("ERROR", forecast_peak, forecast_peak, None)

    if obs_df.empty or "temperature_f" not in obs_df.columns:
        log.warning("No live observation data available")
        return ("ERROR", forecast_peak, forecast_peak, None)

    # Step 3: Today's observed max
    observed_max = obs_df["temperature_f"].max()
    log.info(f"Today's observed max (last 6h): {observed_max}°F")

    # Step 4: Check last 3 readings for declining trend
    temps = obs_df["temperature_f"].dropna()
    if len(temps) < 3:
        log.info(f"Only {len(temps)} readings available, need at least 3")
        return ("TOO_EARLY", observed_max, forecast_peak, obs_df)

    last3 = temps.iloc[-3:].tolist()
    last3_times = obs_df.loc[temps.index[-3:], "timestamp"].tolist()
    is_declining = last3[0] > last3[1] > last3[2]
    all_below_peak = all(t < forecast_peak for t in last3)
    current_temp = last3[-1]

    log.info(f"Last 3 readings: {last3[0]:.1f} → {last3[1]:.1f} → {last3[2]:.1f}°F")
    log.info(f"  Times: {last3_times[0]} → {last3_times[1]} → {last3_times[2]}")
    log.info(f"  Declining: {is_declining}, All below forecast peak: {all_below_peak}")

    if is_declining and all_below_peak:
        margin = forecast_peak - current_temp
        log.info(f"Peak LOCKED — temps declining, {margin:.1f}°F below forecast peak")
        return ("LOCKED", observed_max, forecast_peak, obs_df)
    elif all_below_peak:
        log.info("Temps below forecast peak but not yet in a clear decline")
        return ("NEAR_PEAK", observed_max, forecast_peak, obs_df)
    else:
        log.info("Temps still near or above forecast peak — too early to call")
        return ("TOO_EARLY", observed_max, forecast_peak, obs_df)


# ---------------------------------------------------------------------------
# Momentum strategy
# ---------------------------------------------------------------------------

def _compute_rate_of_change(
    obs_df: pd.DataFrame,
    window_minutes: int = 30,
) -> Tuple[Optional[float], int]:
    """Compute temperature rate of change over the last *window_minutes*.

    Returns (rate_f_per_hour, n_points).  rate is negative when cooling.
    Returns (None, 0) if insufficient data.
    """
    temps = obs_df[["timestamp", "temperature_f"]].dropna(subset=["temperature_f"])
    if len(temps) < 2:
        return (None, 0)

    # Parse timestamps and find readings within the window
    parsed = []
    for _, row in temps.iterrows():
        try:
            ts = datetime.fromisoformat(row["timestamp"])
            parsed.append((ts, row["temperature_f"]))
        except (ValueError, TypeError):
            continue

    if len(parsed) < 2:
        return (None, 0)

    latest_ts = parsed[-1][0]
    cutoff = latest_ts - pd.Timedelta(minutes=window_minutes)
    window = [(ts, t) for ts, t in parsed if ts >= cutoff]

    if len(window) < 2:
        return (None, len(window))

    first_ts, first_temp = window[0]
    last_ts, last_temp = window[-1]
    hours = (last_ts - first_ts).total_seconds() / 3600.0

    if hours < 0.05:  # less than 3 minutes — not meaningful
        return (None, len(window))

    rate = (last_temp - first_temp) / hours
    return (rate, len(window))


def momentum_strategy(
    site: str = "KLAX",
    window_minutes: int = 30,
) -> Tuple[str, float, float, float, Optional[pd.DataFrame]]:
    """Momentum strategy: rate of temperature change + forecast peak.

    Computes °F/hour over the last *window_minutes* of live data and
    compares against the forecast peak.

    Returns (status, observed_max_f, forecast_peak_f, rate_f_per_hr, obs_df)
    where status is:
      "LOCKED"    — rate ≤ -2°F/hr and current temp below forecast peak
      "LIKELY"    — rate ≤ -1°F/hr and below peak
      "POSSIBLE"  — rate ≤ -0.5°F/hr and below peak (or rate < 0 but slow)
      "TOO_EARLY" — rate > -0.5°F/hr, or rising, or at/above peak
      "ERROR"     — could not fetch data
    """
    # Step 1: Forecast peak
    try:
        fi = ForecastIngestion(site)
        forecast_df = fi.fetch_forecast()
    except Exception as e:
        log.warning(f"Could not fetch forecast: {e}")
        return ("ERROR", 0.0, 0.0, 0.0, None)

    if forecast_df.empty or "temperature_f" not in forecast_df.columns:
        log.warning("Forecast returned no temperature data")
        return ("ERROR", 0.0, 0.0, 0.0, None)

    forecast_peak = forecast_df["temperature_f"].max()
    peak_idx = forecast_df["temperature_f"].idxmax()
    peak_time = forecast_df.loc[peak_idx, "timestamp"]
    log.info(f"Forecast peak: {forecast_peak}°F at {peak_time}")

    # Step 2: Live observations
    try:
        si = SynopticIngestion(site)
        obs_df = si.fetch_live_weather(hours=6)
    except Exception as e:
        log.warning(f"Could not fetch live observations: {e}")
        return ("ERROR", forecast_peak, forecast_peak, 0.0, None)

    if obs_df.empty or "temperature_f" not in obs_df.columns:
        log.warning("No live observation data available")
        return ("ERROR", forecast_peak, forecast_peak, 0.0, None)

    # Step 3: Today's observed max
    observed_max = obs_df["temperature_f"].max()
    current_temp = obs_df["temperature_f"].dropna().iloc[-1]
    log.info(f"Today's observed max (last 6h): {observed_max}°F")
    log.info(f"Current temp: {current_temp}°F")

    # Step 4: Rate of change
    rate, n_points = _compute_rate_of_change(obs_df, window_minutes)

    if rate is None:
        log.info(f"Insufficient data for rate calculation ({n_points} points in {window_minutes}min window)")
        return ("TOO_EARLY", observed_max, forecast_peak, 0.0, obs_df)

    log.info(f"Rate of change: {rate:+.2f}°F/hr over last {window_minutes}min ({n_points} points)")

    below_peak = current_temp < forecast_peak

    if not below_peak:
        log.info(f"Current temp {current_temp}°F is at/above forecast peak {forecast_peak}°F")
        return ("TOO_EARLY", observed_max, forecast_peak, rate, obs_df)

    margin = forecast_peak - current_temp

    if rate <= -2.0:
        status = "LOCKED"
        log.info(f"LOCKED — rapid cooling at {rate:+.1f}°F/hr, {margin:.1f}°F below peak")
    elif rate <= -1.0:
        status = "LIKELY"
        log.info(f"LIKELY — steady decline at {rate:+.1f}°F/hr, {margin:.1f}°F below peak")
    elif rate <= -0.5:
        status = "POSSIBLE"
        log.info(f"POSSIBLE — slow decline at {rate:+.1f}°F/hr, {margin:.1f}°F below peak")
    else:
        status = "TOO_EARLY"
        log.info(f"TOO_EARLY — rate {rate:+.1f}°F/hr too flat, could plateau or rebound")

    return (status, observed_max, forecast_peak, rate, obs_df)


# ---------------------------------------------------------------------------
# Claude strategy
# ---------------------------------------------------------------------------

def _call_claude_for_decision(system_prompt: str, user_message: str) -> dict:
    """Call Claude API with the decision tool. Returns the tool input dict.

    On any error, returns {"action": "no_bet"} — never bet on failure.
    """
    try:
        import anthropic
    except ImportError:
        log.error("anthropic package not installed — run: pip install anthropic")
        return {"action": "no_bet", "confidence": "low", "reasoning": "anthropic package not installed"}

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        log.error("ANTHROPIC_API_KEY not set in environment")
        return {"action": "no_bet", "confidence": "low", "reasoning": "API key not configured"}

    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=2048,
            system=system_prompt,
            tools=[DECISION_TOOL],
            tool_choice={"type": "tool", "name": "make_bet_decision"},
            messages=[{"role": "user", "content": user_message}],
        )
    except Exception as e:
        log.error(f"Anthropic API error: {e}")
        return {"action": "no_bet", "confidence": "low", "reasoning": f"API error: {e}"}

    # Extract tool_use block
    log.debug(f"Claude stop_reason: {response.stop_reason}")
    for block in response.content:
        if block.type == "tool_use" and block.name == "make_bet_decision":
            decision = block.input
            log.info(f"Claude decision: {json.dumps(decision, indent=2)}")
            return decision

    log.warning(f"Claude response contained no tool_use block (stop_reason={response.stop_reason}) — treating as no-bet")
    return {"action": "no_bet", "confidence": "low", "reasoning": "No tool_use in response"}


def _get_orderbook_price(client: "KalshiClient", ticker: str) -> str:
    """Format current orderbook bid/ask for a bracket.

    Kalshi orderbook arrays are [price, quantity] sorted ascending by price.
    ``yes`` = resting YES bids;  ``no`` = resting NO bids.
    Best YES bid  = yes[-1][0]  (highest YES buy order)
    Best YES ask  = 100 - no[-1][0]  (selling NO = buying YES)
    """
    ob = client.get_orderbook(ticker)
    yes_bids = ob.get("yes", [])
    no_bids = ob.get("no", [])
    if yes_bids:
        best_yes_bid = yes_bids[-1][0]
    else:
        best_yes_bid = "none"
    if no_bids:
        best_yes_ask = 100 - no_bids[-1][0]
    else:
        best_yes_ask = "none"
    return f"YES bid/ask: {best_yes_bid}/{best_yes_ask}c"


def _parse_trade_time(ts: str) -> datetime:
    """Parse a Kalshi trade timestamp into a timezone-aware datetime.

    Handles variable fractional-second precision (Python 3.9 fromisoformat
    only accepts 0, 3, or 6 fractional digits).
    """
    ts = ts.replace("Z", "+00:00")
    # Normalize fractional seconds to 6 digits
    m = re.match(r"^(.*\.\d+)(\+.*|-.*)$", ts)
    if m:
        base, tz_part = m.group(1), m.group(2)
        # Split at the dot
        dt_part, frac = base.rsplit(".", 1)
        frac = frac[:6].ljust(6, "0")
        ts = f"{dt_part}.{frac}{tz_part}"
    return datetime.fromisoformat(ts)


def _get_last_trade_price(client: "KalshiClient", ticker: str, before_dt: datetime) -> str:
    """Find the last trade price before a given datetime. Pages up to 5x."""
    cursor = ""
    for _ in range(5):
        resp = client.get_trades(ticker, limit=100, cursor=cursor)
        trades = resp.get("trades", [])
        for t in trades:
            trade_dt = _parse_trade_time(t["created_time"])
            if trade_dt < before_dt:
                return f"last trade: YES {t['yes_price']}c NO {t['no_price']}c at {t['created_time'][:19]}Z"
        cursor = resp.get("cursor", "")
        if not cursor or not trades:
            break
    return "no trades before sim time"


def _run_claude_strategy(
    args: argparse.Namespace,
    market_type: str,
    label: str,
    client: "KalshiClient",
    series_ticker: str,
    sim_now: Optional[datetime] = None,
) -> Optional[dict]:
    """Claude strategy: send all weather data + market prices to Claude for a decision.

    Returns dict with {ticker, side, target_temp, confidence} if bet, or None for no-bet.
    """
    now = sim_now or datetime.now(ZoneInfo("America/Los_Angeles"))
    if sim_now:
        log.info(f"=== Claude strategy (simulated time: {now.isoformat()}) ===")
    else:
        log.info("=== Claude strategy ===")
    sections = []

    # 1. Observations (fetch full day so we don't miss the peak)
    try:
        si = SynopticIngestion("KLAX")
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
            last5 = obs_df[["timestamp", "temperature_f"]].tail(20).to_string(index=False)

            # 6h METAR max/min (0.1°C precision — most accurate for CLI prediction)
            # METAR reports drop at :53 and cover the PRECEDING 6 hours:
            #   03:53 covers 22:00-04:00, 09:53 covers 04:00-10:00,
            #   15:53 covers 10:00-16:00, 21:53 covers 16:00-22:00
            metar6h_lines = ""
            if "max_temp_6h_f" in obs_df.columns:
                metar6h = obs_df[["timestamp", "max_temp_6h_f", "min_temp_6h_f"]].dropna(subset=["max_temp_6h_f"])
                if not metar6h.empty:
                    metar6h_max = metar6h["max_temp_6h_f"].max()
                    latest_metar_ts = metar6h["timestamp"].iloc[-1]
                    metar6h_lines = (
                        f"6h METAR max (0.1°C precision): {metar6h_max}°F\n"
                        f"NOTE: Each METAR covers the 6h BEFORE its timestamp. "
                        f"If observed max > METAR max, the peak occurred after the "
                        f"latest METAR window ({latest_metar_ts}) and will be captured "
                        f"in the next report.\n"
                        f"6h METAR reports:\n{metar6h.to_string(index=False)}\n"
                    )

            sections.append(
                f"## Live Observations (Synoptic, today)\n"
                f"Current temp: {current_temp}°F\n"
                f"Observed max (5-min obs): {observed_max}°F\n"
                f"Observed min (5-min obs): {observed_min}°F\n"
                f"{metar6h_lines}"
                f"Last 20 readings:\n{last5}"
            )
        else:
            sections.append("## Live Observations\nUnavailable (empty data)")
    except Exception as e:
        log.warning(f"Could not fetch observations: {e}")
        sections.append(f"## Live Observations\nUnavailable: {e}")
        obs_df = pd.DataFrame()

    # 2. Momentum
    try:
        if not obs_df.empty and "temperature_f" in obs_df.columns:
            mom_df = compute_momentum(obs_df, window_minutes=30)
            rates = mom_df["rate_f_per_hr"].dropna()
            if not rates.empty:
                current_rate = float(rates.iloc[-1])
                # METAR rates (last 6)
                metar_mask = mom_df.get("is_metar", pd.Series(False, index=mom_df.index))
                metar_rates = mom_df.loc[metar_mask, "rate_f_per_hr"].dropna().tail(6)
                metar_rates_str = ", ".join(f"{r:+.2f}" for r in metar_rates)

                extrap = extrapolate_momentum(mom_df)
                extrap_str = "N/A"
                if not extrap.empty:
                    extrap_str = (
                        f"{extrap['temperature_f'].iloc[-1]:.1f}°F "
                        f"(range {extrap['temp_lo'].iloc[-1]:.1f}–{extrap['temp_hi'].iloc[-1]:.1f}°F)"
                    )

                sections.append(
                    f"## Momentum (30-min weighted WLS)\n"
                    f"Current rate: {current_rate:+.2f}°F/hr\n"
                    f"METAR rates (last 6): [{metar_rates_str}]\n"
                    f"+60min extrapolation: {extrap_str}"
                )
            else:
                sections.append("## Momentum\nInsufficient data for rate calculation")
        else:
            sections.append("## Momentum\nNo observations available for momentum calc")
    except Exception as e:
        log.warning(f"Could not compute momentum: {e}")
        sections.append(f"## Momentum\nUnavailable: {e}")

    # 3. Forecast
    try:
        fi = ForecastIngestion("KLAX")
        forecast_df = fi.fetch_forecast()
        if not forecast_df.empty and "temperature_f" in forecast_df.columns:
            forecast_high = forecast_df["temperature_f"].max()
            forecast_low = forecast_df["temperature_f"].min()
            peak_idx = forecast_df["temperature_f"].idxmax()
            peak_time = forecast_df.loc[peak_idx, "timestamp"]

            # Remaining hours (use sim_now if set)
            remaining = []
            for _, row in forecast_df.iterrows():
                try:
                    ts = datetime.fromisoformat(row["timestamp"])
                    if ts > now:
                        remaining.append(row["temperature_f"])
                except (ValueError, TypeError):
                    continue

            remaining_max = max(remaining) if remaining else None
            remaining_min = min(remaining) if remaining else None

            obs_max_val = observed_max if not obs_df.empty and "temperature_f" in obs_df.columns else None
            obs_min_val = observed_min if not obs_df.empty and "temperature_f" in obs_df.columns else None

            delta_high = f"{obs_max_val - remaining_max:+.1f}°F" if obs_max_val and remaining_max else "N/A"
            delta_low = f"{remaining_min - obs_min_val:+.1f}°F" if obs_min_val and remaining_min else "N/A"

            # Hourly forecast table
            fc_table = forecast_df[["timestamp", "temperature_f"]].to_string(index=False)

            sections.append(
                f"## NWS Forecast (today)\n"
                f"Forecast high: {forecast_high}°F\n"
                f"Forecast low: {forecast_low}°F\n"
                f"Forecast peak time: {peak_time}\n"
                f"Remaining hours: {len(remaining)} left\n"
                f"Remaining max: {remaining_max}°F\n"
                f"Remaining min: {remaining_min}°F\n"
                f"Delta (observed max - remaining max): {delta_high}\n"
                f"Delta (remaining min - observed min): {delta_low}\n"
                f"Hourly forecast:\n{fc_table}"
            )
        else:
            sections.append("## NWS Forecast\nUnavailable (empty data)")
    except Exception as e:
        log.warning(f"Could not fetch forecast: {e}")
        sections.append(f"## NWS Forecast\nUnavailable: {e}")

    # 4. Kalshi brackets + prices (today only)
    try:
        resp = client.get_markets(series_ticker)
        today_suffix = now.strftime("%y%b%d").upper()
        markets = [m for m in resp.get("markets", []) if today_suffix in m.get("ticker", "")]
        if markets:
            bracket_lines = []
            sim_utc_dt = now.astimezone(ZoneInfo("UTC")) if sim_now else None
            for m in markets:
                ticker = m.get("ticker", "")
                title = m.get("title", "") + " " + m.get("subtitle", "")
                try:
                    if sim_utc_dt:
                        # Historical: find last trade price before simulated time
                        price_str = _get_last_trade_price(client, ticker, sim_utc_dt)
                    else:
                        # Live: use current orderbook bid/ask
                        price_str = _get_orderbook_price(client, ticker)
                    bracket_lines.append(f"  {ticker}: {title.strip()} — {price_str}")
                except Exception as e:
                    log.debug(f"Price error for {ticker}: {e}")
                    bracket_lines.append(f"  {ticker}: {title.strip()} — price unavailable")
            header = f"## Kalshi Brackets ({series_ticker})"
            if sim_utc_dt:
                header += " [historical last-trade prices]"
            sections.append(header + "\n" + "\n".join(bracket_lines))
        else:
            sections.append(f"## Kalshi Brackets\nNo open {series_ticker} markets found")
    except Exception as e:
        log.warning(f"Could not fetch Kalshi markets: {e}")
        sections.append(f"## Kalshi Brackets\nUnavailable: {e}")

    # Build user message
    now_str = now.isoformat()
    user_message = (
        f"# Betting Decision Request\n"
        f"Market: {series_ticker} (daily {label})\n"
        f"Current time: {now_str}\n\n"
        + "\n\n".join(sections)
    )

    log.info(f"Sending {len(user_message)} chars to Claude for analysis...")
    log.info(f"User message:\n{user_message}")

    # Call Claude
    decision = _call_claude_for_decision(CLAUDE_SYSTEM_PROMPT, user_message)

    action = decision.get("action", "no_bet")
    confidence = decision.get("confidence", "unknown")
    reasoning = decision.get("reasoning", "")
    target_temp = decision.get("target_temp")
    ticker = decision.get("ticker")
    side = decision.get("side", "yes")

    log.info(f"Decision: {action} | Confidence: {confidence}")
    log.info(f"Reasoning: {reasoning}")

    if action != "bet":
        log.info("Claude says no bet — exiting.")
        return None

    if not ticker:
        log.warning("Claude said bet but provided no ticker — treating as no-bet.")
        return None

    log.info(f"Claude says bet {side.upper()} on {ticker} (target: {target_temp}°F, confidence: {confidence})")
    return {"ticker": ticker, "side": side, "target_temp": target_temp, "confidence": confidence}


# ---------------------------------------------------------------------------
# Bracket matching
# ---------------------------------------------------------------------------

def find_matching_bracket(markets: List[Dict], temp_f: float) -> Optional[Dict]:
    """Find the market bracket that contains the observed temperature.

    Kalshi KXHIGHLAX tickers encode temperature bounds, e.g.:
      KXHIGHLAX-26FEB7-T68  means "68°F or above" or a bracket like 68-70.
    The subtitle/title usually spells out the range.
    """
    temp_int = int(round(temp_f))
    log.info(f"Looking for bracket containing {temp_int}°F (raw: {temp_f}°F)")

    for market in markets:
        title = market.get("title", "") + " " + market.get("subtitle", "")
        ticker = market.get("ticker", "")

        # Try to extract range from title like "68° to 69°" or "68 to 70"
        range_match = re.search(r"(\d+)°?\s*(?:to|-)\s*(\d+)°?", title, re.IGNORECASE)
        if range_match:
            lo = int(range_match.group(1))
            hi = int(range_match.group(2))
            if lo <= temp_int <= hi:
                log.info(f"Matched bracket: {ticker} ({lo}–{hi}°F)")
                return market
            continue

        # Try "X° or above" / "X° or below" patterns
        above_match = re.search(r"(\d+)°?\s*or\s*(above|higher|more)", title, re.IGNORECASE)
        if above_match:
            threshold = int(above_match.group(1))
            if temp_int >= threshold:
                log.info(f"Matched bracket: {ticker} ({threshold}°F or above)")
                return market
            continue

        below_match = re.search(r"(\d+)°?\s*or\s*(below|lower|less)", title, re.IGNORECASE)
        if below_match:
            threshold = int(below_match.group(1))
            if temp_int <= threshold:
                log.info(f"Matched bracket: {ticker} ({threshold}°F or below)")
                return market
            continue

        # Try extracting floor/ceiling from floor_strike/cap_strike if available
        floor_strike = market.get("floor_strike")
        cap_strike = market.get("cap_strike")
        if floor_strike is not None and cap_strike is not None:
            lo = int(float(floor_strike))
            hi = int(float(cap_strike))
            if lo <= temp_int <= hi:
                log.info(f"Matched bracket via strikes: {ticker} ({lo}–{hi}°F)")
                return market
            continue

    return None


# ---------------------------------------------------------------------------
# Strategy runners (return target_temp or exit)
# ---------------------------------------------------------------------------

def _run_metar6h_strategy(args: argparse.Namespace, market_type: str, label: str) -> float:
    """Original strategy: wait for 6h METAR report, then confirm with forecast."""
    # --- Fetch NWS observation ---
    log.info("Fetching NWS observation for KLAX...")
    html = fetch_nws_observation()

    ts, max_temp, min_temp = parse_6hr_section(html)
    log.info(f"Observation timestamp: {ts}")
    log.info(f"6-hour max temp: {max_temp}°F, min temp: {min_temp}°F")

    target_temp = max_temp if market_type == "high" else min_temp

    # --- Check timing ---
    if market_type == "high":
        if not is_past_3pm_pacific(ts):
            pacific = ZoneInfo("America/Los_Angeles")
            ts_pt = ts.astimezone(pacific)
            log.info(f"Observation time ({ts_pt.strftime('%I:%M %p %Z')}) is before 3:00 PM Pacific. Exiting.")
            sys.exit(0)
        log.info("Observation is past 3 PM Pacific — proceeding with bet.")
    else:
        pacific = ZoneInfo("America/Los_Angeles")
        ts_pt = ts.astimezone(pacific)
        if ts_pt.hour < 9:
            log.info(f"Observation time ({ts_pt.strftime('%I:%M %p %Z')}) is before 9:00 AM Pacific. Exiting.")
            sys.exit(0)
        log.info(f"Observation is past 9 AM Pacific ({ts_pt.strftime('%I:%M %p %Z')}) — proceeding.")

    # --- Forecast confidence check ---
    if market_type == "high":
        log.info("Checking NWS forecast to assess if daily high is locked in...")
        confidence, forecast_extreme, forecast_df = assess_high_confidence(target_temp)
    else:
        log.info("Checking NWS forecast to assess if daily low is locked in...")
        confidence, forecast_extreme, forecast_df = assess_low_confidence(target_temp)

    if confidence == "LOW" and not args.force:
        log.warning(f"Forecast confidence is LOW — {label} may not be locked in yet.")
        log.warning(f"Observed 6h {label}: {target_temp}°F, forecast extreme: {forecast_extreme}°F")
        log.warning("Use --force to override this safety check.")
        sys.exit(0)
    elif confidence == "LOW" and args.force:
        log.warning(f"Forecast confidence is LOW but --force specified. Proceeding anyway.")
    elif confidence == "UNKNOWN":
        log.warning("Could not fetch forecast — proceeding with observed data only.")
    else:
        log.info(f"Forecast confidence: {confidence} — daily {label} appears locked in at {target_temp}°F")

    return target_temp


def _run_peak_track_strategy(args: argparse.Namespace, market_type: str, label: str) -> float:
    """Peak-tracking strategy: forecast peak + live obs declining trend.

    Fetches the forecast peak, pulls live Synoptic observations, and checks
    if the last 3 readings are declining and below the forecast peak.
    """
    if market_type != "high":
        log.error("peak-track strategy currently only supports --market high")
        sys.exit(1)

    log.info("=== Peak-tracking strategy ===")
    status, observed_max, forecast_peak, obs_df = peak_track_strategy()

    if status == "ERROR":
        log.error("Could not run peak-track strategy — data unavailable.")
        sys.exit(1)

    if status == "TOO_EARLY" and not args.force:
        log.warning("Temps still rising or near forecast peak — too early to bet.")
        log.warning("Use --force to override.")
        sys.exit(0)
    elif status == "NEAR_PEAK" and not args.force:
        log.warning("Temps below forecast peak but not in a clear decline yet.")
        log.warning("Use --force to override.")
        sys.exit(0)
    elif status in ("TOO_EARLY", "NEAR_PEAK") and args.force:
        log.warning(f"Status is {status} but --force specified. Proceeding with observed max.")
    else:
        log.info(f"Peak LOCKED — observed max: {observed_max}°F, forecast peak: {forecast_peak}°F")

    log.info(f"Betting on observed max: {observed_max}°F")
    return observed_max


def _run_momentum_strategy(args: argparse.Namespace, market_type: str, label: str) -> float:
    """Momentum strategy: rate of change over sliding window + forecast peak."""
    if market_type != "high":
        log.error("momentum strategy currently only supports --market high")
        sys.exit(1)

    log.info("=== Momentum strategy ===")
    status, observed_max, forecast_peak, rate, obs_df = momentum_strategy()

    log.info(f"Status: {status} | Rate: {rate:+.2f}°F/hr | Observed max: {observed_max}°F | Forecast peak: {forecast_peak}°F")

    if status == "ERROR":
        log.error("Could not run momentum strategy — data unavailable.")
        sys.exit(1)

    if status == "LOCKED":
        log.info(f"Peak LOCKED — rapid cooling confirms high is in. Betting on {observed_max}°F")
    elif status == "LIKELY" and not args.force:
        log.info(f"Peak LIKELY passed (rate {rate:+.1f}°F/hr) but not rapid enough for auto-bet.")
        log.info("Use --force to bet, or wait for faster decline.")
        sys.exit(0)
    elif status == "LIKELY" and args.force:
        log.warning(f"Status LIKELY (rate {rate:+.1f}°F/hr) — --force specified, proceeding.")
    elif status in ("POSSIBLE", "TOO_EARLY") and not args.force:
        log.warning(f"Status {status} (rate {rate:+.1f}°F/hr) — not confident enough to bet.")
        log.warning("Use --force to override.")
        sys.exit(0)
    elif status in ("POSSIBLE", "TOO_EARLY") and args.force:
        log.warning(f"Status {status} but --force specified. Proceeding with observed max.")

    log.info(f"Betting on observed max: {observed_max}°F")
    return observed_max


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Kalshi LA Temp Weather Bot")
    parser.add_argument("--market", choices=["high", "low"], default="high",
                        help="Which market to bet on: daily high or daily low (default: high)")
    parser.add_argument("--strategy", choices=["metar6h", "peak-track", "momentum", "claude"], default="metar6h",
                        help="Strategy: metar6h | peak-track | momentum | claude")
    parser.add_argument("--dry-run", action="store_true",
                        help="Do everything except place the order")
    parser.add_argument("--force", action="store_true",
                        help="Place bet even if forecast confidence is LOW")
    parser.add_argument("--simulate-time", type=str, default=None,
                        help="Override current time for claude strategy context (e.g. 2026-02-08T14:00:00-0800)")
    args = parser.parse_args()

    market_type = args.market
    series_ticker = SERIES_TICKERS[market_type]
    label = "high" if market_type == "high" else "low"

    # --- Set up Kalshi client (needed by claude strategy and order placement) ---
    api_key_id = os.getenv("KALSHI_API_KEY_ID")
    private_key_path = os.getenv("KALSHI_PRIVATE_KEY_PATH")
    kalshi_env = os.getenv("KALSHI_ENV", "demo")

    if not api_key_id or not private_key_path:
        log.error("KALSHI_API_KEY_ID and KALSHI_PRIVATE_KEY_PATH must be set in .env")
        sys.exit(1)

    client = KalshiClient(api_key_id, private_key_path, env=kalshi_env)
    log.info(f"Using Kalshi {kalshi_env} API")

    if args.strategy == "claude":
        sim_now = None
        if args.simulate_time:
            # Normalize -0800 to -08:00 for Python 3.9 fromisoformat
            ts_str = args.simulate_time
            if re.search(r"[+-]\d{4}$", ts_str):
                ts_str = ts_str[:-2] + ":" + ts_str[-2:]
            sim_now = datetime.fromisoformat(ts_str)
            if sim_now.tzinfo is None:
                sim_now = sim_now.replace(tzinfo=ZoneInfo("America/Los_Angeles"))
            log.info(f"Simulating time: {sim_now.isoformat()}")
        decision = _run_claude_strategy(args, market_type, label, client, series_ticker, sim_now=sim_now)
        if decision is None:
            sys.exit(0)

        # Claude picked the bracket — go straight to order
        ticker = decision["ticker"]
        side = decision["side"]
        log.info(f"Claude picked: {side.upper()} on {ticker} (target: {decision.get('target_temp')}°F)")

        # Orderbook: yes/no arrays are BIDS sorted ascending.
        # To buy YES: ask = 100 - best NO bid (no[-1][0])
        # To buy NO:  ask = 100 - best YES bid (yes[-1][0])
        orderbook = client.get_orderbook(ticker)
        if side == "yes":
            opposite_bids = orderbook.get("no", [])
        else:
            opposite_bids = orderbook.get("yes", [])
        if not opposite_bids:
            log.warning(f"No opposite-side bids on orderbook for {side.upper()}. Falling back to 99c limit.")
            ask_price = 99
        else:
            ask_price = 100 - opposite_bids[-1][0]

        # Sanity checks on price
        if ask_price > 95:
            log.error(f"Ask price {ask_price}c is too high (>95c) — no profit margin. Aborting.")
            sys.exit(0)
        if ask_price < 30:
            log.error(f"Ask price {ask_price}c is suspiciously low (<30c) — likely a bad match. Aborting.")
            sys.exit(0)

        count = BET_AMOUNT_CENTS // ask_price
        if count < 1:
            count = 1

        log.info(f"Orderbook best {side.upper()} ask: {ask_price}c — buying {count} contract(s) (${count * ask_price / 100:.2f})")

        if args.dry_run:
            log.info(f"[DRY RUN] Would place order: "
                     f"ticker={ticker}, side={side}, price={ask_price}c, count={count}")
            log.info("Dry run complete — no order placed.")
            return

        result = client.place_order(ticker, side=side, price=ask_price, count=count)
        log.info(f"Order placed! Response: {json.dumps(result, indent=2)}")
        return

    elif args.strategy == "peak-track":
        target_temp = _run_peak_track_strategy(args, market_type, label)
    elif args.strategy == "momentum":
        target_temp = _run_momentum_strategy(args, market_type, label)
    else:
        target_temp = _run_metar6h_strategy(args, market_type, label)

    # --- Step 6: Query markets (today only) ---
    log.info(f"Querying open {series_ticker} markets...")
    resp = client.get_markets(series_ticker)
    today_suffix = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%y%b%d").upper()
    markets = [m for m in resp.get("markets", []) if today_suffix in m.get("ticker", "")]
    if not markets:
        log.error(f"No open {series_ticker} markets found for today ({today_suffix}).")
        sys.exit(1)
    log.info(f"Found {len(markets)} open market(s) for today")

    # --- Step 7: Find matching bracket ---
    bracket = find_matching_bracket(markets, target_temp)
    if not bracket:
        log.error(f"No bracket found for {target_temp}°F. Available markets:")
        for m in markets:
            log.error(f"  {m.get('ticker')}: {m.get('title', '')} {m.get('subtitle', '')}")
        sys.exit(1)

    ticker = bracket["ticker"]
    log.info(f"Target bracket: {ticker}")

    # --- Step 8: Get orderbook and place order ---
    # Orderbook: yes/no arrays are BIDS sorted ascending.
    # To buy YES: ask = 100 - best NO bid (no[-1][0])
    orderbook = client.get_orderbook(ticker)
    no_bids = orderbook.get("no", [])
    if not no_bids:
        log.warning("No NO bids on orderbook. Falling back to 99c limit order.")
        ask_price = 99  # cents
    else:
        ask_price = 100 - no_bids[-1][0]

    # Sanity checks on price
    if ask_price > 95:
        log.error(f"Ask price {ask_price}c is too high (>95c) — no profit margin. Aborting.")
        sys.exit(0)
    if ask_price < 30:
        log.error(f"Ask price {ask_price}c is suspiciously low (<30c) — likely a bad match. Aborting.")
        sys.exit(0)

    # Calculate contract count: $10 / ask_price_in_cents, rounded down
    count = BET_AMOUNT_CENTS // ask_price
    if count < 1:
        count = 1

    log.info(f"Orderbook best ask: {ask_price}c — buying {count} contract(s) at {ask_price}c (${count * ask_price / 100:.2f})")

    if args.dry_run:
        log.info("[DRY RUN] Would place order: "
                 f"ticker={ticker}, side=yes, price={ask_price}c, count={count}")
        log.info("Dry run complete — no order placed.")
        return

    result = client.place_order(ticker, side="yes", price=ask_price, count=count)
    log.info(f"Order placed! Response: {json.dumps(result, indent=2)}")


if __name__ == "__main__":
    main()
