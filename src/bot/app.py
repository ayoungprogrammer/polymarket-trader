#!/usr/bin/env python3
"""Kalshi Weather Bot — Flask Dashboard.

Multi-city bot that bets on Kalshi daily high/low temperature markets
using NWS weather data. Flask web dashboard for monitoring, control,
and bet history.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import signal
import sys
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from dotenv import load_dotenv
from flask import Flask, jsonify, request as flask_request, send_file, Response

from paths import project_path
from weather.forecast import ForecastIngestion
from weather.prediction import (
    compute_momentum,
    extrapolate_momentum,
    predict_settlement_from_obs,
)
from weather.observations import (
    SynopticIngestion,
    fetch_nws_observation,
    is_past_3pm_pacific,
    parse_6hr_section,
)
from weather.strategy import (
    run_strategy,
    MOMENTUM_PARAMS_FAST, MOMENTUM_PARAMS_WIDE, WIDE_CONFIRM_COUNT,
    SITE_MOMENTUM_PARAMS,
)
from weather.market import parse_bracket, parse_all_brackets, find_matching_bracket, get_today_brackets

load_dotenv(project_path(".env.demo"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

KALSHI_BASE_URLS = {
    "prod": "https://api.elections.kalshi.com",
    "demo": "https://demo-api.kalshi.co",
}
API_PREFIX = "/trade-api/v2"

from weather.sites import KALSHI_STATIONS as STATIONS
from db import init_db, log_bet, is_locked_today, get_recent_bets

BET_AMOUNT_CENTS = 1000  # $10 in cents


# ---------------------------------------------------------------------------
# BotConfig (replaces argparse.Namespace)
# ---------------------------------------------------------------------------

@dataclass
class BotConfig:
    strategy: str = "claude"
    market: str = "high"
    dry_run: bool = True
    force: bool = False
    watch_interval: int = 300
    simulate_time: Optional[str] = None


# ---------------------------------------------------------------------------
# SiteStatus + WatchStateManager (thread-safe shared state for dashboard)
# ---------------------------------------------------------------------------

@dataclass
class SiteStatus:
    site: str = ""
    city: str = ""
    series_ticker: str = ""
    state: str = "idle"           # idle | waiting | polling | locked | stopped | error
    strategy: str = ""
    market_type: str = "high"
    iteration: int = 0
    strategy_status: str = ""     # TOO_EARLY | LOCKED | LIKELY | etc.
    observed_max_f: float = 0.0
    settlement_f: int = 0
    forecast_peak_f: float = 0.0
    forecast_peak_time: str = ""
    current_temp_f: float = 0.0
    momentum_rate: float = 0.0
    minutes_to_close: float = 0.0
    last_poll_time: str = ""
    next_poll_time: str = ""
    message: str = ""
    chart_path: str = ""
    locked_today: bool = False
    bracket_ticker: str = ""
    best_yes_bid: int = 0
    error: str = ""
    updated_at: str = ""


class WatchStateManager:
    """Thread-safe manager for per-site watch status."""

    def __init__(self):
        self._lock = threading.Lock()
        self._statuses: Dict[str, SiteStatus] = {}
        self._stop_events: Dict[str, threading.Event] = {}
        self._threads: Dict[str, threading.Thread] = {}

    def init_site(self, site: str, city: str, series_ticker: str = "",
                  strategy: str = "", market_type: str = "high") -> None:
        with self._lock:
            self._statuses[site] = SiteStatus(
                site=site, city=city, series_ticker=series_ticker,
                strategy=strategy, market_type=market_type,
                updated_at=datetime.now(ZoneInfo("America/Los_Angeles")).isoformat(),
            )
            self._stop_events[site] = threading.Event()

    def update(self, site: str, **kwargs) -> None:
        with self._lock:
            status = self._statuses.get(site)
            if status:
                for k, v in kwargs.items():
                    if hasattr(status, k):
                        setattr(status, k, v)
                status.updated_at = datetime.now(ZoneInfo("America/Los_Angeles")).isoformat()

    def get_all(self) -> Dict[str, dict]:
        with self._lock:
            return {site: asdict(s) for site, s in self._statuses.items()}

    def get(self, site: str) -> Optional[dict]:
        with self._lock:
            s = self._statuses.get(site)
            return asdict(s) if s else None

    def get_stop_event(self, site: str) -> Optional[threading.Event]:
        with self._lock:
            return self._stop_events.get(site)

    def stop_site(self, site: str) -> None:
        with self._lock:
            ev = self._stop_events.get(site)
            if ev:
                ev.set()

    def stop_all(self) -> None:
        with self._lock:
            for ev in self._stop_events.values():
                ev.set()

    def set_thread(self, site: str, thread: threading.Thread) -> None:
        with self._lock:
            self._threads[site] = thread

    def is_running(self, site: str) -> bool:
        with self._lock:
            t = self._threads.get(site)
            return t is not None and t.is_alive()

    def remove_site(self, site: str) -> None:
        with self._lock:
            self._statuses.pop(site, None)
            self._stop_events.pop(site, None)
            self._threads.pop(site, None)


# Module-level singleton
watch_state = WatchStateManager()

# NBA checkpoint alerts thread state
_nba_alerts_thread: Optional[threading.Thread] = None
_nba_alerts_stop: Optional[threading.Event] = None
_nba_alerts_lock = threading.Lock()


def _discover_series_ticker(
    client: "KalshiClient", site: str, market_type: str,
) -> Optional[str]:
    """Try candidate series tickers for a station and return one that exists.

    Uses the STATIONS suffix first, then falls back to ICAO[1:] derivation.
    For low markets, also tries a 'T'-prefixed variant (e.g. DEN→TDEN)
    since some cities use different suffixes for high vs low.

    Checks for any markets (open, closed, or settled) to confirm the series
    exists — today's market may not be open yet.
    """
    prefix = f"KX{market_type.upper()}"
    default_suffix = site[1:]  # KLAX -> LAX

    station_info = STATIONS.get(site)
    configured_suffix = station_info[1] if station_info else default_suffix

    # Build ordered list of unique candidates
    seen = set()
    candidates = []
    for suffix in [configured_suffix, default_suffix, f"T{default_suffix}"]:
        ticker = f"{prefix}{suffix}"
        if ticker not in seen:
            seen.add(ticker)
            candidates.append(ticker)

    for candidate in candidates:
        try:
            # Check open markets first
            resp = client.get_markets(candidate, status="open")
            if resp.get("markets"):
                return candidate
            # Fall back: check if series exists at all (any status)
            resp = client.get_markets(candidate, status=None)
            if resp.get("markets"):
                return candidate
        except Exception as e:
            log.debug(f"Ticker {candidate} lookup failed: {e}")
    return None


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
        self._lock = threading.Lock()

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
        with self._lock:
            url = self.base_url + path
            headers = self._headers(method.upper(), API_PREFIX + path)
            resp = self.session.request(
                method, url, headers=headers, params=params, json=json_body
            )
            resp.raise_for_status()
            return resp.json()

    def get_markets(self, series_ticker: str, status: Optional[str] = "open"):
        params: Dict[str, str] = {"series_ticker": series_ticker}
        if status:
            params["status"] = status
        return self._request("GET", "/markets", params=params)

    def get_orderbook(self, ticker: str):
        """Fetch orderbook and normalize to {yes: [[cents,qty]], no: [[cents,qty]]}."""
        resp = self._request("GET", f"/markets/{ticker}/orderbook")
        # New API format: orderbook_fp with yes_dollars/no_dollars as string pairs
        ob_fp = resp.get("orderbook_fp")
        if ob_fp:
            yes = [
                [int(round(float(p) * 100)), int(round(float(q) * 100))]
                for p, q in ob_fp.get("yes_dollars", [])
            ]
            no = [
                [int(round(float(p) * 100)), int(round(float(q) * 100))]
                for p, q in ob_fp.get("no_dollars", [])
            ]
            return {"yes": yes, "no": no}
        # Legacy format
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
# Lazy KalshiClient singleton
# ---------------------------------------------------------------------------

_kalshi_client: Optional[KalshiClient] = None
_kalshi_client_lock = threading.Lock()


def _get_client() -> KalshiClient:
    """Lazily create and return a singleton KalshiClient."""
    global _kalshi_client
    with _kalshi_client_lock:
        if _kalshi_client is None:
            api_key_id = os.getenv("KALSHI_API_KEY_ID")
            private_key_path = os.getenv("KALSHI_PRIVATE_KEY_PATH")
            kalshi_env = os.getenv("KALSHI_ENV", "demo")
            if not api_key_id or not private_key_path:
                raise RuntimeError("KALSHI_API_KEY_ID and KALSHI_PRIVATE_KEY_PATH must be set in .env")
            _kalshi_client = KalshiClient(api_key_id, private_key_path, env=kalshi_env)
            log.info(f"Initialized Kalshi {kalshi_env} client")
        return _kalshi_client


# ---------------------------------------------------------------------------
# Forecast-based confidence assessment
# ---------------------------------------------------------------------------

def _fetch_remaining_forecast(
    site: str = "KLAX",
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Fetch forecast and split into (full_df, remaining_df).

    Returns (None, None) if forecast is unavailable.
    """
    from weather.forecast import STATIONS as FORECAST_STATIONS

    try:
        fi = ForecastIngestion(site)
        df = fi.fetch_forecast()
    except Exception as e:
        log.warning(f"Could not fetch forecast: {e}")
        return (None, None)

    if df.empty or "temperature_f" not in df.columns:
        log.warning("Forecast returned no temperature data")
        return (None, None)

    tz = ZoneInfo(FORECAST_STATIONS[site][2]) if site in FORECAST_STATIONS else ZoneInfo("America/Los_Angeles")
    now = datetime.now(tz)
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
    """Peak-tracking strategy: use observed peak + live obs declining trend.

    1. Fetch forecast → find forecast peak temp for today
    2. Fetch live Synoptic observations (last 24h, filtered to today)
    3. Find today's observed max from the live data
    4. Check if the last 5 readings are all below the observed max → the
       high has passed and temps are falling away from the peak

    Returns (status, observed_max_f, forecast_peak_f, obs_df) where status is:
      "LOCKED"    — last 5 readings all below observed max
      "NEAR_PEAK" — some recent readings still at/near the peak
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
        obs_df = si.fetch_live_weather(hours=24)
    except Exception as e:
        log.warning(f"Could not fetch live observations: {e}")
        return ("ERROR", forecast_peak, forecast_peak, None)

    if obs_df.empty or "temperature_f" not in obs_df.columns:
        log.warning("No live observation data available")
        return ("ERROR", forecast_peak, forecast_peak, None)

    # Step 3: Today's observed max
    observed_max = obs_df["temperature_f"].max()
    log.info(f"Today's observed max (last 6h): {observed_max}°F")

    # Step 4: Check last 5 readings are all below the observed max
    temps = obs_df["temperature_f"].dropna()
    if len(temps) < 5:
        log.info(f"Only {len(temps)} readings available, need at least 5")
        return ("TOO_EARLY", observed_max, forecast_peak, obs_df)

    last5 = temps.iloc[-5:].tolist()
    last5_times = obs_df.loc[temps.index[-5:], "timestamp"].tolist()
    all_below_peak = all(t < observed_max for t in last5)
    current_temp = last5[-1]

    readings_str = " → ".join(f"{t:.1f}" for t in last5)
    times_str = " → ".join(str(t) for t in last5_times)
    log.info(f"Last 5 readings: {readings_str}°F")
    log.info(f"  Times: {times_str}")
    log.info(f"  All below observed peak ({observed_max:.1f}°F): {all_below_peak}")

    if all_below_peak:
        margin = observed_max - current_temp
        log.info(f"Peak LOCKED — last 5 readings below observed peak, {margin:.1f}°F off peak")
        settlement = predict_settlement_from_obs(obs_df)
        if settlement:
            top_prob = max(settlement.probabilities.values())
            log.info(f"Settlement prediction: {settlement.center_f}°F "
                     f"(possible: {settlement.possible_f}, "
                     f"confidence: {top_prob:.0%})")
            return ("LOCKED", settlement.center_f, forecast_peak, obs_df)
        return ("LOCKED", observed_max, forecast_peak, obs_df)
    else:
        log.info("Temps still at or near observed peak — too early to call")
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
    window_minutes: Optional[int] = None,
) -> Tuple[str, float, float, float, Optional[pd.DataFrame]]:
    """MA crossover momentum strategy.

    Uses dual moving-average crossover (ma_short vs ma_long) from
    ``compute_momentum`` plus a margin check against the observed peak.

    Returns (status, observed_max_f, forecast_peak_f, ma_cross, mom_df, forecast_df)
    where status is:
      "LOCKED"    — crossover + margin triggered (fast or wide)
      "POSSIBLE"  — close to triggering but not quite
      "TOO_EARLY" — no crossover or margin not met
      "ERROR"     — could not fetch data
    mom_df has MA columns from compute_momentum; forecast_df is the NWS forecast.
    """
    # Use per-site optimal params if available, otherwise fall back to defaults
    if site in SITE_MOMENTUM_PARAMS:
        cross_thresh, margin_thresh, confirm_needed = SITE_MOMENTUM_PARAMS[site]
    else:
        cross_thresh, margin_thresh = MOMENTUM_PARAMS_FAST[:2]
        confirm_needed = 1

    # Step 1: Forecast peak
    try:
        fi = ForecastIngestion(site)
        forecast_df = fi.fetch_forecast()
    except Exception as e:
        log.warning(f"Could not fetch forecast: {e}")
        return ("ERROR", 0.0, 0.0, 0.0, None, None)

    if forecast_df.empty or "temperature_f" not in forecast_df.columns:
        log.warning("Forecast returned no temperature data")
        return ("ERROR", 0.0, 0.0, 0.0, None, None)

    forecast_peak = forecast_df["temperature_f"].max()
    peak_idx = forecast_df["temperature_f"].idxmax()
    peak_time = forecast_df.loc[peak_idx, "timestamp"]
    log.info(f"Forecast peak: {forecast_peak}°F at {peak_time}")

    # Time-of-day and peak proximity checks
    from weather.forecast import STATIONS as FORECAST_STATIONS
    _, _, station_tz_name = FORECAST_STATIONS.get(site, (0, 0, "America/Los_Angeles"))
    station_tz = ZoneInfo(station_tz_name)
    peak_dt = datetime.fromisoformat(peak_time)
    if peak_dt.tzinfo is None:
        peak_dt = peak_dt.replace(tzinfo=station_tz)
    else:
        peak_dt = peak_dt.astimezone(station_tz)
    now = datetime.now(station_tz)

    # Gate: don't trigger before forecast peak - 1h (floor: 10 AM local)
    start_dt = max(
        peak_dt - timedelta(hours=1),
        now.replace(hour=10, minute=0, second=0, microsecond=0),
    )
    if now < start_dt:
        log.info(f"Too early — forecast peak at {peak_dt.strftime('%I:%M %p %Z')}, "
                 f"watching from {start_dt.strftime('%I:%M %p %Z')}")
        # Still fetch obs + compute momentum so charts can be generated
        try:
            si = SynopticIngestion(site)
            early_obs = si.fetch_live_weather(hours=24)
            early_mom = compute_momentum(early_obs) if not early_obs.empty and "temperature_f" in early_obs.columns else None
        except Exception:
            early_mom = None
        return ("TOO_EARLY", 0.0, forecast_peak, 0.0, early_mom, forecast_df)

    # Too late: forecast peak + 3h OR 8 PM local, whichever comes first
    cutoff_peak = peak_dt + timedelta(hours=3)
    cutoff_evening = now.replace(hour=20, minute=0, second=0, microsecond=0)
    cutoff = min(cutoff_peak, cutoff_evening)
    if now > cutoff:
        reason = ("past 8 PM local" if cutoff == cutoff_evening
                  else f"3h past forecast peak ({peak_dt.strftime('%I:%M %p %Z')})")
        log.info(f"Too late — {reason}")
        # Still fetch obs + compute momentum so chart can be generated
        try:
            si = SynopticIngestion(site)
            late_obs = si.fetch_live_weather(hours=24)
            late_mom = compute_momentum(late_obs) if not late_obs.empty and "temperature_f" in late_obs.columns else None
        except Exception:
            late_mom = None
        return ("TOO_LATE", 0.0, forecast_peak, 0.0, late_mom, forecast_df)

    # Step 2: Live observations
    try:
        si = SynopticIngestion(site)
        obs_df = si.fetch_live_weather(hours=24)
    except Exception as e:
        log.warning(f"Could not fetch live observations: {e}")
        return ("ERROR", forecast_peak, forecast_peak, 0.0, None, forecast_df)

    if obs_df.empty or "temperature_f" not in obs_df.columns:
        log.warning("No live observation data available")
        return ("ERROR", forecast_peak, forecast_peak, 0.0, None, forecast_df)

    # Step 3: Today's observed max
    observed_max = obs_df["temperature_f"].max()
    log.info(f"Today's observed max (last 6h): {observed_max}°F")

    # Step 4: MA crossover
    mom_df = compute_momentum(obs_df)
    crosses = mom_df["ma_cross"].dropna()

    if crosses.empty:
        log.info("Insufficient data for MA crossover")
        return ("TOO_EARLY", observed_max, forecast_peak, 0.0, None, forecast_df)

    ma_cross = float(crosses.iloc[-1])
    ma_short_now = float(mom_df["ma_short"].dropna().iloc[-1])
    ma_long_peak = float(mom_df["ma_long"].dropna().max())
    margin = ma_long_peak - ma_short_now

    # Count consecutive readings where ma_cross < threshold
    confirm_count = 0
    for val in reversed(crosses.tolist()):
        if val < cross_thresh:
            confirm_count += 1
        else:
            break

    log.info(f"MA cross: {ma_cross:+.2f}°F, margin: {margin:.1f}°F, "
             f"confirmed: {confirm_count}/{confirm_needed} readings "
             f"(site params: cross ≤{cross_thresh}, margin ≥{margin_thresh})")

    # Trigger logic: per-site params
    triggered = (ma_cross <= cross_thresh
                 and margin >= margin_thresh
                 and confirm_count >= confirm_needed)

    if triggered:
        status = "LOCKED"
        log.info(f"LOCKED — MA cross {ma_cross:+.2f}°F, {margin:.1f}°F margin, "
                 f"{confirm_count} confirmed")
        settlement = predict_settlement_from_obs(obs_df)
        if settlement:
            top_prob = max(settlement.probabilities.values())
            log.info(f"Settlement prediction: {settlement.center_f}°F "
                     f"(possible: {settlement.possible_f}, "
                     f"confidence: {top_prob:.0%})")
            observed_max = settlement.center_f
    elif margin < margin_thresh:
        status = "TOO_EARLY"
        log.info(f"TOO_EARLY — margin {margin:.1f}°F too small (need ≥{margin_thresh})")
    elif ma_cross > 0:
        status = "TOO_EARLY"
        log.info(f"TOO_EARLY — no crossover (ma_cross {ma_cross:+.2f}°F)")
    else:
        status = "POSSIBLE"
        log.info(f"POSSIBLE — MA cross {ma_cross:+.2f}°F, {margin:.1f}°F margin (close to trigger)")

    return (status, observed_max, forecast_peak, ma_cross, mom_df, forecast_df)


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


def _kalshi_url(series_ticker: str) -> str:
    """Build a Kalshi market URL from a series ticker."""
    return f"https://kalshi.com/markets/{series_ticker.lower()}"


def _forecast_peak_time_iso(forecast_df: Optional[pd.DataFrame], station_tz_name: str = "America/Los_Angeles") -> str:
    """Extract forecast peak time as ISO string from a forecast DataFrame."""
    if forecast_df is None or forecast_df.empty or "temperature_f" not in forecast_df.columns:
        return ""
    try:
        peak_idx = forecast_df["temperature_f"].idxmax()
        peak_ts = forecast_df.loc[peak_idx, "timestamp"]
        peak_dt = datetime.fromisoformat(peak_ts)
        tz = ZoneInfo(station_tz_name)
        if peak_dt.tzinfo is None:
            peak_dt = peak_dt.replace(tzinfo=tz)
        else:
            peak_dt = peak_dt.astimezone(tz)
        return peak_dt.isoformat()
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Chart updates
# ---------------------------------------------------------------------------

def _update_momentum_chart(
    site: str,
    mom_df: Optional[pd.DataFrame],
    forecast_df: Optional[pd.DataFrame] = None,
    series_ticker: Optional[str] = None,
) -> None:
    """Regenerate the momentum chart for a site after each poll.

    Accepts already-computed *mom_df* and *forecast_df* to avoid
    redundant API calls and computation.  Also fetches sun times
    and bracket model probabilities for the chart overlay.

    Fetches real Kalshi market brackets via weather.market module.
    """
    if mom_df is None or mom_df.empty:
        return
    try:
        from weather.prediction import plot_momentum
        from weather.forecast import STATIONS as FORECAST_STATIONS
        from weather.observations import fetch_sun_times
        import gc

        site_cross = SITE_MOMENTUM_PARAMS.get(site, (MOMENTUM_PARAMS_FAST[0],))[0]
        chart_path = project_path("charts", "weather", f"momentum_{site}.png")
        os.makedirs(os.path.dirname(chart_path), exist_ok=True)

        # Sun times
        sun_times = None
        coords = FORECAST_STATIONS.get(site)
        if coords:
            station_tz = ZoneInfo(coords[2])
            today_str = datetime.now(station_tz).strftime("%Y-%m-%d")
            sun_times = fetch_sun_times(coords[0], coords[1], today_str)

        # 6h METAR max — only use records from today (station timezone)
        metar_6h_f = None
        if "max_temp_6h_f" in mom_df.columns:
            _today_df = mom_df
            if coords and "timestamp" in mom_df.columns:
                _stz = ZoneInfo(coords[2])
                _today_local = datetime.now(_stz).date()
                _ts_parsed = pd.to_datetime(mom_df["timestamp"].str[:19])
                _today_df = mom_df[_ts_parsed.dt.tz_localize("UTC").dt.tz_convert(_stz).dt.date == _today_local]
            m6h = _today_df["max_temp_6h_f"].dropna()
            if not m6h.empty:
                metar_6h_f = float(m6h.max())

        # Bracket model
        bracket = None
        bracket_probs = None
        bracket_error = None
        try:
            from weather.bracket_model import load_model, get_probability
            from weather.backtest_rounding import extract_regression_features
            bmodel = load_model()
            solar_noon_hour = sun_times.get("solar_noon") if sun_times else None
            feats = extract_regression_features(mom_df, solar_noon_hour=solar_noon_hour)
            if feats is None:
                bracket_error = f"Not enough observations ({len(mom_df)} rows, need 20+ auto-obs)"
                log.info(f"[{site}] Bracket model: {bracket_error}")
            if feats is not None:
                max_c = feats.get("max_c")
                if max_c is not None:
                    naive_f = round(float(max_c) * 9.0 / 5.0 + 32.0)

                    # Fetch real Kalshi market brackets via market.py
                    brackets_parsed = []
                    try:
                        client = _get_client()
                        market_type = "high"  # TODO: support low
                        today_brackets = get_today_brackets(client, site, market_type)
                        if today_brackets:
                            brackets_parsed = [(b["lo"], b["hi"]) for b in today_brackets]
                            log.info(f"[{site}] Kalshi brackets: {brackets_parsed}, naive_f={naive_f}")
                    except Exception as e:
                        log.warning(f"[{site}] Could not fetch market brackets: {e}", exc_info=True)

                    # Fallback: synthesize even-odd brackets
                    if not brackets_parsed:
                        if naive_f % 2 == 0:
                            center_lo = naive_f
                        else:
                            center_lo = naive_f - 1
                        brackets_parsed = [(lo, lo + 1) for lo in [center_lo - 2, center_lo, center_lo + 2]]

                    bracket_probs = get_probability(bmodel, feats, brackets_parsed,
                                                    metar_6h_f=metar_6h_f)
                    bracket_probs.sort(key=lambda x: -x["prob"])
                    if bracket_probs:
                        top = bracket_probs[0]
                        bracket = list(top["bracket"])
        except Exception as e:
            bracket_error = str(e)
            log.warning(f"[{site}] Bracket model for chart: {e}", exc_info=True)

        # Peak model: will settlement °F increase by +1?
        peak_result = None
        try:
            from weather.peak_model import load_model as load_peak_model, predict as peak_predict
            from weather.forecast import STATIONS as _FS
            peak_bundle = load_peak_model()
            _fcst_high = forecast_df["temperature_f"].max() if forecast_df is not None and not forecast_df.empty else 70.0
            _sn = sun_times.get("solar_noon", 12.0) if sun_times else 12.0
            peak_result = peak_predict(peak_bundle, mom_df, forecast_high_f=float(_fcst_high),
                                       solar_noon_hour=_sn)
        except Exception as e:
            log.warning(f"[{site}] Peak model for chart: {e}", exc_info=True)

        # METAR 6h report time — vertical line on chart
        metar_6h_local_dt = None
        try:
            from weather.sites import get_site_config
            site_cfg = get_site_config(site)
            metar_utc_str = site_cfg.get("metar_6h_utc")
            if metar_utc_str and coords:
                station_tz = ZoneInfo(coords[2])
                _today_local = datetime.now(station_tz).date()
                hh, mm = int(metar_utc_str.split(":")[0]), int(metar_utc_str.split(":")[1])
                metar_utc_dt = datetime(
                    _today_local.year, _today_local.month, _today_local.day,
                    hh, mm, tzinfo=ZoneInfo("UTC"),
                )
                metar_6h_local_dt = metar_utc_dt.astimezone(station_tz)
        except Exception as e:
            log.debug(f"[{site}] Could not compute METAR 6h time: {e}")

        plot_momentum(mom_df, site, output=chart_path,
                      forecast_df=forecast_df if forecast_df is not None and not forecast_df.empty else None,
                      locked_rate=site_cross,
                      likely_rate=MOMENTUM_PARAMS_WIDE[0],
                      margin_threshold=SITE_MOMENTUM_PARAMS.get(site, MOMENTUM_PARAMS_FAST)[1],
                      sun_times=sun_times,
                      metar_6h_f=metar_6h_f,
                      bracket=bracket,
                      bracket_probs=bracket_probs,
                      bracket_error=bracket_error,
                      peak_result=peak_result,
                      metar_6h_local_dt=metar_6h_local_dt)
        watch_state.update(site, chart_path=chart_path)
        gc.collect()
    except Exception as e:
        log.warning(f"[{site}] Could not update momentum chart: {e}")


# ---------------------------------------------------------------------------
# Bracket lock detection + Telegram alert
# ---------------------------------------------------------------------------

# Track which (site, date) combos have already fired a bracket-lock alert
# to avoid spamming on every poll.
_bracket_lock_notified: Dict[Tuple[str, str], bool] = {}
_bracket_lock_notified_lock = threading.Lock()


def _check_bracket_lock_notify(
    site: str,
    series_ticker: str,
    station_tz: "ZoneInfo",
) -> None:
    """Check if bracket model locks in at >95% and send Telegram if so.

    Queries the bracket model for the current observations. If the top
    bracket has >95% probability, fetches Kalshi orderbooks for brackets
    below it and notifies if any have NO asks > 10c (i.e. opportunities
    to sell YES / buy NO on brackets the model says won't hit).
    """
    today_str = datetime.now(station_tz).strftime("%Y-%m-%d")
    with _bracket_lock_notified_lock:
        if _bracket_lock_notified.get((site, today_str)):
            return

    tag = f"[{site}]"
    try:
        from weather.bracket_model import load_model as load_bracket_model, get_probability
        from weather.backtest_rounding import extract_regression_features
        from weather.observations import SynopticIngestion, fetch_sun_times
        from weather.forecast import STATIONS as FORECAST_STATIONS

        # Fetch live obs
        si = SynopticIngestion(site)
        obs_df = si.fetch_live_weather(hours=24)
        if obs_df.empty or len(obs_df) < 20:
            return

        # Sun times for solar noon
        coords = FORECAST_STATIONS.get(site)
        solar_noon_hour = None
        if coords:
            sun = fetch_sun_times(coords[0], coords[1], today_str)
            if sun:
                solar_noon_hour = sun.get("solar_noon")

        # Extract features + run bracket model
        bmodel = load_bracket_model()
        feats = extract_regression_features(obs_df, solar_noon_hour=solar_noon_hour)
        if feats is None:
            return

        max_c = feats.get("max_c")
        if max_c is None:
            return
        naive_f = round(float(max_c) * 9.0 / 5.0 + 32.0)

        # 6h METAR lock-in — only use records from today
        metar_6h_f = None
        if "max_temp_6h_f" in obs_df.columns:
            _ts_p = pd.to_datetime(obs_df["timestamp"].str[:19])
            _today_obs = obs_df[_ts_p.dt.tz_localize("UTC").dt.tz_convert(station_tz).dt.date == datetime.now(station_tz).date()]
            m6h = _today_obs["max_temp_6h_f"].dropna()
            if not m6h.empty:
                metar_6h_f = float(m6h.max())

        # Fetch all today's Kalshi brackets
        client = _get_client()
        today_suffix = datetime.now(station_tz).strftime("%y%b%d").upper()
        resp = client.get_markets(series_ticker)
        today_markets = [m for m in resp.get("markets", [])
                         if today_suffix in m.get("ticker", "")]
        if not today_markets:
            return

        all_brackets = parse_all_brackets(today_markets)
        bracket_tuples = [(lo, hi) for _, lo, hi in all_brackets]

        bracket_probs = get_probability(bmodel, feats, bracket_tuples,
                                        metar_6h_f=metar_6h_f)
        if not bracket_probs:
            return

        bracket_probs.sort(key=lambda x: -x["prob"])
        top = bracket_probs[0]
        top_prob = top["prob"]
        top_lo, top_hi = top["bracket"]

        if top_prob < 0.95:
            return

        # Model locked in at >95% — check brackets below for NO opportunities
        # Build ticker lookup: (lo, hi) -> ticker
        ticker_by_bracket = {(lo, hi): tk for tk, lo, hi in all_brackets}

        opportunities = []
        for tk, lo, hi in all_brackets:
            if hi <= top_lo:  # bracket is strictly below predicted
                try:
                    ob = client.get_orderbook(tk)
                    yes_bids = ob.get("yes", [])
                    no_bids = ob.get("no", [])
                    # NO ask = price to buy NO = 100 - best_yes_bid
                    # YES ask = price to buy YES = 100 - best_no_bid
                    best_yes_bid = yes_bids[-1][0] if yes_bids else 0
                    yes_ask = (100 - no_bids[-1][0]) if no_bids else None
                    no_ask = (100 - best_yes_bid) if best_yes_bid > 0 else None
                    # We want brackets where YES still has asks > 10c
                    # (market hasn't fully priced out this bracket yet)
                    if yes_ask is not None and yes_ask > 10:
                        opportunities.append({
                            "ticker": tk,
                            "bracket": (lo, hi),
                            "yes_ask": yes_ask,
                            "yes_bid": best_yes_bid,
                            "no_ask": no_ask,
                        })
                except Exception:
                    pass

        if not opportunities:
            return

        # Build notification
        reason = top.get("reason", "")
        lines = [
            f"<b>{tag} BRACKET LOCK: [{top_lo}, {top_hi}]°F @ {top_prob:.0%}</b>",
            f"Reason: {reason}" if reason else "",
            f"Naive: {naive_f}°F | Max °C: {max_c}",
            "",
            "<b>Brackets below with YES ask &gt; 10c:</b>",
        ]
        for opp in sorted(opportunities, key=lambda x: -x["yes_ask"]):
            lo, hi = opp["bracket"]
            lines.append(
                f"  [{lo}, {hi}]°F — YES ask: {opp['yes_ask']}c"
                f" | bid: {opp['yes_bid']}c"
                f"  <code>{opp['ticker']}</code>"
            )
        lines.append("")
        lines.append(f'<a href="{_kalshi_url(series_ticker)}">View on Kalshi</a>')

        caption = "\n".join(l for l in lines if l is not None)
        chart_path = project_path("charts", "weather", f"momentum_{site}.png")
        _send_telegram(caption, image_path=chart_path)
        log.info(f"{tag} Bracket lock notification sent: [{top_lo}, {top_hi}]°F @ {top_prob:.0%}")

        with _bracket_lock_notified_lock:
            _bracket_lock_notified[(site, today_str)] = True

    except Exception as e:
        log.debug(f"{tag} Bracket lock check failed: {e}")


# ---------------------------------------------------------------------------
# Telegram notifications
# ---------------------------------------------------------------------------

def _get_notif_log() -> logging.Logger:
    """Lazy-init file logger for log/notifications.log."""
    nlog = logging.getLogger("notifications")
    if not nlog.handlers:
        from paths import project_path
        log_dir = project_path("log")
        os.makedirs(log_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(log_dir, "notifications.log"))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        nlog.addHandler(fh)
        nlog.setLevel(logging.DEBUG)
    return nlog


def _send_telegram(
    caption: str,
    image_path: str = "momentum.png",
) -> None:
    """Send a Telegram message with an optional chart image.

    Uses sendPhoto if *image_path* exists, otherwise falls back to
    sendMessage.  Silently logs on failure — never blocks betting.
    All attempts are logged to log/notifications.log.
    """
    import re
    nlog = _get_notif_log()

    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        log.debug("Telegram not configured — skipping notification")
        nlog.warning("Telegram not configured (missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID)")
        return

    preview = re.sub(r"<[^>]+>", "", caption)[:200]
    nlog.info("Sending: %s", preview)

    base = f"https://api.telegram.org/bot{token}"

    try:
        if os.path.isfile(image_path):
            with open(image_path, "rb") as img:
                resp = requests.post(
                    f"{base}/sendPhoto",
                    data={"chat_id": chat_id, "caption": caption, "parse_mode": "HTML"},
                    files={"photo": img},
                    timeout=15,
                )
        else:
            resp = requests.post(
                f"{base}/sendMessage",
                json={"chat_id": chat_id, "text": caption, "parse_mode": "HTML"},
                timeout=15,
            )
        if not resp.ok:
            log.warning(f"Telegram API error {resp.status_code}: {resp.text[:200]}")
            nlog.error("Telegram API error %d: %s", resp.status_code, resp.text[:300])
        else:
            log.info("Telegram notification sent")
            nlog.info("Sent OK")
    except Exception as e:
        log.warning(f"Telegram send failed: {e}")
        nlog.error("Telegram send failed: %s", e, exc_info=True)


# ---------------------------------------------------------------------------
# Telegram command listener (runs in background daemon thread)
# ---------------------------------------------------------------------------

def _handle_predict(chat_id: str, site: str, token: str,
                    config: BotConfig, client: KalshiClient,
                    market_type: str, site_tickers: Dict[str, str],
                    label: str) -> None:
    """Handle a /predict command: run Claude strategy, send chart + decision."""
    base = f"https://api.telegram.org/bot{token}"

    # Look up series ticker for the requested site
    series_ticker = site_tickers.get(site)
    if not series_ticker:
        # Try discovering it on the fly
        series_ticker = _discover_series_ticker(client, site, market_type)
    if not series_ticker:
        try:
            requests.post(
                f"{base}/sendMessage",
                json={"chat_id": chat_id, "text": f"No {market_type} market found for {site}"},
                timeout=15,
            )
        except Exception:
            pass
        return

    try:
        decision = run_strategy(
            market_type, label, series_ticker, site=site, client=client)

        kalshi_link = f'<a href="{_kalshi_url(series_ticker)}">View on Kalshi</a>'
        reasoning = decision.get('reasoning', '')[:500] if decision else ''
        if decision and decision.get('action') != 'no_bet':
            b = decision.get('bracket', [])
            b_str = f"[{b[0]}, {b[1]}]" if len(b) == 2 else "?"
            caption = (
                f"<b>[{site}] {series_ticker} Predict</b>\n"
                f"Bracket: {b_str}°F\n"
                f"Confidence: {decision.get('confidence')}\n"
                f"Reasoning: {reasoning}\n"
                f"{kalshi_link}"
            )
        else:
            caption = (
                f"<b>[{site}] {series_ticker} Predict</b>\n"
                f"No bet recommended.\n"
                f"Reasoning: {reasoning}\n"
                f"{kalshi_link}"
            )

        chart_path = project_path("charts", "weather", f"momentum_{site}.png")
        if os.path.isfile(chart_path):
            with open(chart_path, "rb") as img:
                requests.post(
                    f"{base}/sendPhoto",
                    data={"chat_id": chat_id, "caption": caption, "parse_mode": "HTML"},
                    files={"photo": img},
                    timeout=15,
                )
        else:
            requests.post(
                f"{base}/sendMessage",
                json={"chat_id": chat_id, "text": caption, "parse_mode": "HTML"},
                timeout=15,
            )

    except Exception as e:
        log.warning(f"[telegram] /predict {site} error: {e}")
        try:
            requests.post(
                f"{base}/sendMessage",
                json={"chat_id": chat_id, "text": f"Error running /predict {site}: {e}"},
                timeout=15,
            )
        except Exception:
            pass


def _telegram_listener(config: BotConfig, client: KalshiClient,
                       market_type: str, site_tickers: Dict[str, str],
                       label: str) -> None:
    """Long-poll Telegram getUpdates in a loop. Handles /predict commands."""
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        log.debug("[telegram] TELEGRAM_BOT_TOKEN not set — listener not started")
        return

    log.info("[telegram] Listener started")
    base = f"https://api.telegram.org/bot{token}"
    offset = 0

    while True:
        try:
            resp = requests.get(
                f"{base}/getUpdates",
                params={"offset": offset, "timeout": 30},
                timeout=35,
            )
            if not resp.ok:
                log.warning(f"[telegram] getUpdates HTTP {resp.status_code}")
                time.sleep(5)
                continue

            data = resp.json()
            for update in data.get("result", []):
                offset = update["update_id"] + 1
                msg = update.get("message", {})
                text = msg.get("text", "").strip()
                chat_id = str(msg.get("chat", {}).get("id", ""))
                if not chat_id or not text:
                    continue

                # Parse /predict [SITE]
                if text.startswith("/predict"):
                    parts = text.split()
                    if len(parts) >= 2:
                        site = parts[1].upper()
                        if not site.startswith("K"):
                            site = "K" + site
                    else:
                        site = "KLAX"
                    log.info(f"[telegram] /predict {site} from chat {chat_id}")
                    _handle_predict(chat_id, site, token,
                                    config, client, market_type, site_tickers, label)

        except Exception as e:
            log.warning(f"[telegram] Listener error: {e}")
            time.sleep(5)


def _fetch_bracket_bid(
    client: "KalshiClient",
    series_ticker: str,
    obs_max_f: float,
    station_tz: "ZoneInfo",
) -> tuple:
    """Return (bracket_ticker, best_yes_bid) for the bracket matching obs_max_f.

    Returns ("", 0) on any error — never breaks the poll loop.
    """
    try:
        today_suffix = datetime.now(station_tz).strftime("%y%b%d").upper()
        resp = client.get_markets(series_ticker)
        markets = [m for m in resp.get("markets", []) if today_suffix in m.get("ticker", "")]
        if not markets:
            return ("", 0)
        matched = find_matching_bracket(markets, obs_max_f)
        if not matched:
            return ("", 0)
        ticker = matched["ticker"]
        orderbook = client.get_orderbook(ticker)
        yes_bids = orderbook.get("yes", [])
        best_bid = yes_bids[-1][0] if yes_bids else 0
        return (ticker, best_bid)
    except Exception:
        return ("", 0)


# ---------------------------------------------------------------------------
# Strategy runners (return target_temp or None)
# ---------------------------------------------------------------------------

def _run_metar6h_strategy(config: BotConfig, market_type: str, label: str) -> Optional[float]:
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
            log.info(f"Observation time ({ts_pt.strftime('%I:%M %p %Z')}) is before 3:00 PM Pacific.")
            return None
        log.info("Observation is past 3 PM Pacific — proceeding with bet.")
    else:
        pacific = ZoneInfo("America/Los_Angeles")
        ts_pt = ts.astimezone(pacific)
        if ts_pt.hour < 9:
            log.info(f"Observation time ({ts_pt.strftime('%I:%M %p %Z')}) is before 9:00 AM Pacific.")
            return None
        log.info(f"Observation is past 9 AM Pacific ({ts_pt.strftime('%I:%M %p %Z')}) — proceeding.")

    # --- Forecast confidence check ---
    if market_type == "high":
        log.info("Checking NWS forecast to assess if daily high is locked in...")
        confidence, forecast_extreme, forecast_df = assess_high_confidence(target_temp)
    else:
        log.info("Checking NWS forecast to assess if daily low is locked in...")
        confidence, forecast_extreme, forecast_df = assess_low_confidence(target_temp)

    if confidence == "LOW" and not config.force:
        log.warning(f"Forecast confidence is LOW — {label} may not be locked in yet.")
        log.warning(f"Observed 6h {label}: {target_temp}°F, forecast extreme: {forecast_extreme}°F")
        return None
    elif confidence == "LOW" and config.force:
        log.warning(f"Forecast confidence is LOW but force specified. Proceeding anyway.")
    elif confidence == "UNKNOWN":
        log.warning("Could not fetch forecast — proceeding with observed data only.")
    else:
        log.info(f"Forecast confidence: {confidence} — daily {label} appears locked in at {target_temp}°F")

    return target_temp


def _run_peak_track_strategy(config: BotConfig, market_type: str, label: str) -> Optional[float]:
    """Peak-tracking strategy: forecast peak + live obs declining trend."""
    if market_type != "high":
        log.error("peak-track strategy currently only supports high market")
        return None

    log.info("=== Peak-tracking strategy ===")
    status, observed_max, forecast_peak, obs_df = peak_track_strategy()

    if status == "ERROR":
        log.error("Could not run peak-track strategy — data unavailable.")
        return None

    if status == "TOO_EARLY" and not config.force:
        log.warning("Temps still rising or near forecast peak — too early to bet.")
        return None
    elif status == "NEAR_PEAK" and not config.force:
        log.warning("Temps below forecast peak but not in a clear decline yet.")
        return None
    elif status in ("TOO_EARLY", "NEAR_PEAK") and config.force:
        log.warning(f"Status is {status} but force specified. Proceeding with observed max.")
    else:
        log.info(f"Peak LOCKED — observed max: {observed_max}°F, forecast peak: {forecast_peak}°F")

    log.info(f"Betting on observed max: {observed_max}°F")
    return observed_max


def _run_momentum_strategy(config: BotConfig, market_type: str, label: str) -> Optional[float]:
    """Momentum strategy: rate of change over sliding window + forecast peak."""
    if market_type != "high":
        log.error("momentum strategy currently only supports high market")
        return None

    log.info("=== Momentum strategy ===")
    status, observed_max, forecast_peak, rate, _, _ = momentum_strategy()

    log.info(f"Status: {status} | MA cross: {rate:+.2f}°F | Observed max: {observed_max}°F | Forecast peak: {forecast_peak}°F")

    if status == "ERROR":
        log.error("Could not run momentum strategy — data unavailable.")
        return None

    if status == "LOCKED":
        log.info(f"Peak LOCKED — MA crossover confirms high is in. Betting on {observed_max}°F")
    elif status in ("POSSIBLE", "TOO_EARLY") and not config.force:
        log.warning(f"Status {status} (MA cross {rate:+.2f}°F) — not confident enough to bet.")
        return None
    elif status in ("POSSIBLE", "TOO_EARLY") and config.force:
        log.warning(f"Status {status} but force specified. Proceeding with observed max.")

    log.info(f"Betting on observed max: {observed_max}°F")
    return observed_max


# ---------------------------------------------------------------------------
# Watch mode
# ---------------------------------------------------------------------------

def _sleep_until_next_day(site: str, station_tz: ZoneInfo,
                          stop_event: Optional[threading.Event],
                          reason: str) -> bool:
    """Sleep until 6 AM next day in station timezone.

    Returns True if stop was requested, False if sleep completed.
    """
    tag = f"[{site}]"
    now_local = datetime.now(station_tz)
    tomorrow_6am = (now_local + pd.Timedelta(days=1)).replace(
        hour=6, minute=0, second=0, microsecond=0)
    wait_secs = (tomorrow_6am - now_local).total_seconds()

    log.info(f"{tag} {reason} — sleeping until {tomorrow_6am.strftime('%I:%M %p %Z %b %d')} "
             f"({wait_secs / 3600:.1f}h)")
    watch_state.update(site, state="too_late",
                       message=f"{reason} — next day {tomorrow_6am.strftime('%I:%M %p %Z')}",
                       next_poll_time=tomorrow_6am.isoformat())

    if stop_event and stop_event.wait(timeout=wait_secs):
        log.info(f"{tag} Stop requested during overnight sleep.")
        watch_state.update(site, state="stopped", message="Stopped by user")
        return True
    elif not stop_event:
        time.sleep(wait_secs)
    return False


def _watch_loop(
    config: BotConfig,
    market_type: str,
    label: str,
    client: "KalshiClient",
    series_ticker: str,
    site: str = "KLAX",
    stop_event: Optional[threading.Event] = None,
):
    """Poll strategy repeatedly until conditions trigger a bet.

    Runs continuously across days: when today's market closes, the current
    day is done (already bet or too late), sleeps until 6 AM next day and
    restarts.

    Returns target_temp (float) for non-claude strategies, decision dict for
    claude strategy, or None only if stop is requested.
    """
    from weather.forecast import STATIONS as FORECAST_STATIONS
    tag = f"[{site}]"
    strategy = config.strategy
    eastern = ZoneInfo("America/New_York")
    station_tz = ZoneInfo(FORECAST_STATIONS[site][2]) if site in FORECAST_STATIONS else ZoneInfo("America/Los_Angeles")

    while True:  # daily loop
        # Check stop event
        if stop_event and stop_event.is_set():
            watch_state.update(site, state="stopped", message="Stopped by user")
            return None

        interval = config.watch_interval
        today_et = datetime.now(eastern)
        market_close = today_et.replace(hour=23, minute=59, second=0, microsecond=0)

        # Check DB for existing bet today
        if is_locked_today(strategy, series_ticker):
            log.info(f"{tag} Already bet today ({strategy}/{series_ticker}).")
            if _sleep_until_next_day(site, station_tz, stop_event, "Already bet today"):
                return None
            continue  # next day

        # Forecast-based watch window: sleep until ~1h before forecast peak
        try:
            fi = ForecastIngestion(site)
            forecast_df = fi.fetch_forecast()
            if not forecast_df.empty and "temperature_f" in forecast_df.columns:
                peak_idx = forecast_df["temperature_f"].idxmax()
                peak_time_str = forecast_df.loc[peak_idx, "timestamp"]
                peak_dt = datetime.fromisoformat(peak_time_str)
                if peak_dt.tzinfo is None:
                    peak_dt = peak_dt.replace(tzinfo=station_tz)
                else:
                    peak_dt = peak_dt.astimezone(station_tz)
                start_time = peak_dt - timedelta(hours=1)
                now = datetime.now(station_tz)
                if now < start_time:
                    wait_secs = (start_time - now).total_seconds()
                    log.info(f"{tag} Forecast peak at {peak_dt.strftime('%I:%M %p %Z')} — "
                             f"waiting until {start_time.strftime('%I:%M %p %Z')} "
                             f"({wait_secs / 60:.0f} min)")
                    watch_state.update(site, state="waiting",
                                       forecast_peak_f=float(forecast_df["temperature_f"].max()),
                                       forecast_peak_time=peak_dt.isoformat(),
                                       message=f"Waiting until {start_time.strftime('%I:%M %p %Z')}",
                                       next_poll_time=start_time.isoformat())
                    if stop_event and stop_event.wait(timeout=wait_secs):
                        log.info(f"{tag} Stop requested during wait.")
                        watch_state.update(site, state="stopped", message="Stopped by user")
                        return None
                else:
                    log.info(f"{tag} Forecast peak at {peak_dt.strftime('%I:%M %p %Z')} — "
                             f"already within 1h window, starting immediately")
                # Override interval to 5 min once in the watch window
                interval = 300
            else:
                log.warning(f"{tag} Forecast empty — starting immediately with default interval")
        except Exception as e:
            log.warning(f"{tag} Could not fetch forecast for watch window: {e} — "
                        f"starting immediately with default interval")

        # Intra-day polling loop
        iteration = 0
        too_late = False
        while True:
            # Check stop event
            if stop_event and stop_event.is_set():
                log.info(f"{tag} Stop requested. Stopping.")
                watch_state.update(site, state="stopped", message="Stopped by user")
                return None

            iteration += 1
            now_et = datetime.now(eastern)
            now_local = datetime.now(station_tz)

            if now_et >= market_close:
                log.info(f"{tag} Market close reached.")
                too_late = True
                break

            mins_left = (market_close - now_et).total_seconds() / 60
            log.info(f"{tag} === Poll #{iteration} | "
                     f"{now_local.strftime('%I:%M %p %Z')} | "
                     f"{mins_left:.0f} min to close ===")

            watch_state.update(site, state="polling", iteration=iteration,
                               last_poll_time=now_local.isoformat(),
                               minutes_to_close=mins_left)

            obs_max = 0.0
            try:
                if strategy == "peak-track":
                    status, obs_max, fcst_peak, _ = peak_track_strategy(site)
                    log.info(f"{tag} peak-track: {status} | "
                             f"obs_max={obs_max:.1f}°F | fcst_peak={fcst_peak:.1f}°F")
                    watch_state.update(site, strategy_status=status,
                                       observed_max_f=obs_max, forecast_peak_f=fcst_peak)
                    if status == "LOCKED":
                        log.info(f"{tag} LOCKED — betting on {obs_max:.1f}°F")
                        watch_state.update(site, state="locked",
                                           message=f"LOCKED at {obs_max:.1f}°F")
                        _send_telegram(
                            f"<b>{tag} LOCKED — peak-track</b>\n"
                            f"Observed max: {obs_max:.1f}°F\n"
                            f"Forecast peak: {fcst_peak:.1f}°F\n"
                            f"Poll #{iteration} | {mins_left:.0f} min to close\n"
                            f"<a href=\"{_kalshi_url(series_ticker)}\">View on Kalshi</a>",
                            image_path=project_path("charts", "weather", f"momentum_{site}.png"),
                        )
                        return obs_max

                elif strategy == "momentum":
                    status, obs_max, fcst_peak, ma_cross, mom_df, fcst_df = momentum_strategy(site)
                    log.info(f"{tag} momentum: {status} | "
                             f"ma_cross={ma_cross:+.2f}°F | obs_max={obs_max:.1f}°F | "
                             f"fcst_peak={fcst_peak:.1f}°F")
                    watch_state.update(site, strategy_status=status,
                                       observed_max_f=obs_max, forecast_peak_f=fcst_peak,
                                       forecast_peak_time=_forecast_peak_time_iso(fcst_df, station_tz.key),
                                       momentum_rate=ma_cross)
                    _update_momentum_chart(site, mom_df, fcst_df, series_ticker=series_ticker)
                    if series_ticker:
                        _check_bracket_lock_notify(site, series_ticker, station_tz)
                    if status == "TOO_LATE":
                        log.info(f"{tag} Past peak window — done for today")
                        too_late = True
                        break
                    if status == "LOCKED":
                        log.info(f"{tag} LOCKED — betting on {obs_max:.1f}°F")
                        watch_state.update(site, state="locked",
                                           message=f"LOCKED at {obs_max:.1f}°F")
                        _send_telegram(
                            f"<b>{tag} LOCKED — momentum (MA crossover)</b>\n"
                            f"Observed max: {obs_max:.1f}°F\n"
                            f"Forecast peak: {fcst_peak:.1f}°F\n"
                            f"MA cross: {ma_cross:+.2f}°F\n"
                            f"Poll #{iteration} | {mins_left:.0f} min to close\n"
                            f"<a href=\"{_kalshi_url(series_ticker)}\">View on Kalshi</a>",
                            image_path=project_path("charts", "weather", f"momentum_{site}.png"),
                        )
                        return obs_max

                elif strategy == "claude":
                    # Gate: run momentum first (free) — only call Claude when
                    # conditions look ready, to avoid wasting API credits.
                    mom_status, obs_max, fcst_peak, ma_cross, mom_df, fcst_df = momentum_strategy(site)
                    log.info(f"{tag} momentum gate: {mom_status} | "
                             f"ma_cross={ma_cross:+.2f}°F | obs_max={obs_max:.1f}°F | "
                             f"fcst_peak={fcst_peak:.1f}°F")
                    watch_state.update(site, strategy_status=mom_status,
                                       observed_max_f=obs_max, forecast_peak_f=fcst_peak,
                                       forecast_peak_time=_forecast_peak_time_iso(fcst_df, station_tz.key),
                                       momentum_rate=ma_cross)
                    _update_momentum_chart(site, mom_df, fcst_df, series_ticker=series_ticker)
                    if series_ticker:
                        _check_bracket_lock_notify(site, series_ticker, station_tz)

                    if mom_status == "TOO_LATE":
                        log.info(f"{tag} Past peak window — done for today")
                        too_late = True
                        break
                    elif mom_status in ("TOO_EARLY", "ERROR"):
                        log.info(f"{tag} Momentum not ready — skipping Claude call")
                    else:
                        log.info(f"{tag} Momentum {mom_status} — calling Claude for decision")
                        decision = run_strategy(
                            market_type, label, series_ticker, site=site, client=client)
                        if decision and decision.get('action') != 'no_bet':
                            b = decision.get('bracket', [])
                            b_str = f"[{b[0]}, {b[1]}]" if len(b) == 2 else "?"
                            log.info(f"{tag} Claude says BET: "
                                     f"bracket {b_str}°F, "
                                     f"confidence {decision.get('confidence')}")
                            watch_state.update(site, state="locked",
                                               message=f"Claude BET: {b_str}°F")
                            _send_telegram(
                                f"<b>{tag} Claude says BET</b>\n"
                                f"Bracket: {b_str}°F | Confidence: {decision.get('confidence')}\n"
                                f"Reasoning: {decision.get('reasoning', '')[:300]}\n"
                                f"Poll #{iteration} | {mins_left:.0f} min to close\n"
                                f"<a href=\"{_kalshi_url(series_ticker)}\">View on Kalshi</a>",
                                image_path=project_path("charts", "weather", f"momentum_{site}.png"),
                            )
                            return decision
                        log.info(f"{tag} Claude says no_bet")

                elif strategy == "metar6h":
                    try:
                        html = fetch_nws_observation(site)
                        ts, max_temp, min_temp = parse_6hr_section(html)
                        target = max_temp if market_type == "high" else min_temp
                        obs_max = target
                        ts_local = ts.astimezone(station_tz)
                        ready = (is_past_3pm_pacific(ts) if market_type == "high"
                                 else ts_local.hour >= 9)
                        log.info(f"{tag} metar6h: "
                                 f"obs_time={ts_local.strftime('%I:%M %p %Z')} | "
                                 f"target={target:.1f}°F | ready={ready}")
                        watch_state.update(site, strategy_status="READY" if ready else "TOO_EARLY",
                                           observed_max_f=target)
                        if ready:
                            watch_state.update(site, state="locked",
                                               message=f"LOCKED at {target:.1f}°F")
                            _send_telegram(
                                f"<b>{tag} LOCKED — metar6h</b>\n"
                                f"Target {label}: {target:.1f}°F\n"
                                f"Obs time: {ts_local.strftime('%I:%M %p %Z')}\n"
                                f"Poll #{iteration} | {mins_left:.0f} min to close\n"
                                f"<a href=\"{_kalshi_url(series_ticker)}\">View on Kalshi</a>",
                                image_path=project_path("charts", "weather", f"momentum_{site}.png"),
                            )
                            return target
                    except Exception as e:
                        log.warning(f"{tag} metar6h fetch error: {e}")

            except Exception as e:
                log.error(f"{tag} Poll error: {e}")
                watch_state.update(site, state="error", error=str(e))

            # Update momentum chart for dashboard (all strategies)
            if strategy not in ("momentum", "claude"):
                try:
                    _, _, _, _, _mom_df, _fcst_df = momentum_strategy(site)
                    _update_momentum_chart(site, _mom_df, _fcst_df, series_ticker=series_ticker)
                except Exception as e:
                    log.debug(f"{tag} Could not update momentum chart: {e}")
                if series_ticker:
                    _check_bracket_lock_notify(site, series_ticker, station_tz)

            # Update settlement prediction for dashboard
            if obs_max > 0:
                try:
                    si = SynopticIngestion(site)
                    _obs = si.fetch_live_weather(hours=24)
                    _settle = predict_settlement_from_obs(_obs)
                    if _settle:
                        watch_state.update(site, settlement_f=_settle.center_f)
                except Exception:
                    watch_state.update(site, settlement_f=int(round(obs_max)))

            if obs_max > 0:
                bticker, ybid = _fetch_bracket_bid(client, series_ticker, obs_max, station_tz)
                watch_state.update(site, bracket_ticker=bticker, best_yes_bid=ybid)

            remaining = (market_close - datetime.now(eastern)).total_seconds()
            sleep_secs = min(interval, max(0, remaining - 60))
            if sleep_secs <= 0:
                log.info(f"{tag} No time left for today.")
                too_late = True
                break

            next_poll = datetime.now(station_tz) + pd.Timedelta(seconds=sleep_secs)
            log.info(f"{tag} Sleeping {sleep_secs:.0f}s...")
            watch_state.update(site, state="waiting",
                               next_poll_time=next_poll.isoformat(),
                               message=f"Next poll ~{next_poll.strftime('%I:%M %p %Z')}")

            if stop_event and stop_event.wait(timeout=sleep_secs):
                log.info(f"{tag} Stop requested during sleep.")
                watch_state.update(site, state="stopped", message="Stopped by user")
                return None
            elif not stop_event:
                time.sleep(sleep_secs)

        # End of intra-day loop — sleep until next day
        if too_late:
            if _sleep_until_next_day(site, station_tz, stop_event, "Done for today"):
                return None
            # Continue outer daily loop


# ---------------------------------------------------------------------------
# Order placement helpers
# ---------------------------------------------------------------------------

def _place_claude_order(
    config: BotConfig,
    client: "KalshiClient",
    decision: dict,
    site: str,
    series_ticker: str = "",
) -> None:
    """Place (or log) a bracket-matched YES order from a Claude strategy decision."""
    tag = f"[{site}]"
    bracket = decision.get("bracket")
    if not bracket or len(bracket) != 2:
        log.warning(f"{tag} Claude decision has no bracket — cannot place order.")
        return

    bracket_lo, bracket_hi = bracket
    log.info(f"{tag} Claude bracket: [{bracket_lo}, {bracket_hi}]°F (confidence: {decision.get('confidence')})")

    # Find today's markets and match bracket by bounds
    from weather.forecast import STATIONS as FORECAST_STATIONS
    station_tz = ZoneInfo(FORECAST_STATIONS[site][2]) if site in FORECAST_STATIONS else ZoneInfo("America/New_York")
    today_suffix = datetime.now(station_tz).strftime("%y%b%d").upper()

    resp = client.get_markets(series_ticker)
    markets = [m for m in resp.get("markets", []) if today_suffix in m.get("ticker", "")]
    if not markets:
        log.error(f"{tag} No open {series_ticker} markets for today ({today_suffix}). Aborting.")
        return

    # Match by bracket midpoint
    target_temp = (bracket_lo + bracket_hi) / 2.0 if bracket_lo != bracket_hi else float(bracket_lo)
    matched = find_matching_bracket(markets, target_temp)
    if not matched:
        log.error(f"{tag} No bracket matches [{bracket_lo}, {bracket_hi}]°F. Aborting.")
        return

    ticker = matched["ticker"]
    side = "yes"
    log.info(f"{tag} Matched bracket: {ticker}")

    # Orderbook: yes/no arrays are BIDS sorted ascending.
    orderbook = client.get_orderbook(ticker)
    opposite_bids = orderbook.get("no", [])
    if not opposite_bids:
        log.warning(f"{tag} No NO bids on orderbook. Falling back to 99c limit.")
        ask_price = 99
    else:
        ask_price = 100 - opposite_bids[-1][0]

    if ask_price > 95:
        log.error(f"{tag} Ask price {ask_price}c is too high (>95c) — no profit margin. Aborting.")
        return
    if ask_price < 30:
        log.error(f"{tag} Ask price {ask_price}c is suspiciously low (<30c) — likely a bad match. Aborting.")
        return

    count = BET_AMOUNT_CENTS // ask_price
    if count < 1:
        count = 1

    log.info(f"{tag} Orderbook best YES ask: {ask_price}c — buying {count} contract(s) (${count * ask_price / 100:.2f})")

    log_bet(
        strategy=config.strategy,
        market=f"{ticker}:{side}",
        price_cents=ask_price,
        count=count,
        dry_run=config.dry_run,
        metadata={
            "bracket": bracket,
            "confidence": decision.get("confidence"),
            "reasoning": decision.get("reasoning", ""),
        },
    )

    market_link = f'\n<a href="{_kalshi_url(series_ticker)}">View on Kalshi</a>' if series_ticker else ""
    _send_telegram(
        f"<b>{tag} {ticker}</b> — YES @ {ask_price}c x{count}\n"
        f"Bracket: [{bracket_lo}, {bracket_hi}]°F | Confidence: {decision.get('confidence')}\n"
        f"Reasoning: {decision.get('reasoning', '')[:300]}\n"
        f"{'[DRY RUN]' if config.dry_run else '[LIVE]'}{market_link}",
        image_path=project_path("charts", "weather", f"momentum_{site}.png"),
    )

    if config.dry_run:
        log.info(f"{tag} [DRY RUN] Would place order: "
                 f"ticker={ticker}, side=yes, price={ask_price}c, count={count}")
        return

    # TODO: re-enable when ready to go live
    log.info(f"{tag} Order placement DISABLED — would have placed: ticker={ticker}, side=yes, price={ask_price}c, count={count}")


def _place_bracket_order(
    config: BotConfig,
    client: "KalshiClient",
    series_ticker: str,
    target_temp: float,
    site: str,
) -> None:
    """Place (or log) a bracket-matched YES order for non-claude strategies."""
    tag = f"[{site}]"

    log.info(f"{tag} Querying open {series_ticker} markets...")
    resp = client.get_markets(series_ticker)
    from weather.forecast import STATIONS as FORECAST_STATIONS
    station_tz = ZoneInfo(FORECAST_STATIONS[site][2]) if site in FORECAST_STATIONS else ZoneInfo("America/New_York")
    today_suffix = datetime.now(station_tz).strftime("%y%b%d").upper()
    markets = [m for m in resp.get("markets", []) if today_suffix in m.get("ticker", "")]
    if not markets:
        log.error(f"{tag} No open {series_ticker} markets found for today ({today_suffix}).")
        return
    log.info(f"{tag} Found {len(markets)} open market(s) for today")

    bracket = find_matching_bracket(markets, target_temp)
    if not bracket:
        log.error(f"{tag} No bracket found for {target_temp}°F. Available markets:")
        for m in markets:
            log.error(f"  {m.get('ticker')}: {m.get('title', '')} {m.get('subtitle', '')}")
        return

    ticker = bracket["ticker"]
    log.info(f"{tag} Target bracket: {ticker}")

    orderbook = client.get_orderbook(ticker)
    no_bids = orderbook.get("no", [])
    if not no_bids:
        log.warning(f"{tag} No NO bids on orderbook. Falling back to 99c limit order.")
        ask_price = 99
    else:
        ask_price = 100 - no_bids[-1][0]

    if ask_price > 95:
        log.error(f"{tag} Ask price {ask_price}c is too high (>95c) — no profit margin. Aborting.")
        return
    if ask_price < 30:
        log.error(f"{tag} Ask price {ask_price}c is suspiciously low (<30c) — likely a bad match. Aborting.")
        return

    count = BET_AMOUNT_CENTS // ask_price
    if count < 1:
        count = 1

    log.info(f"{tag} Orderbook best ask: {ask_price}c — buying {count} contract(s) at {ask_price}c (${count * ask_price / 100:.2f})")

    log_bet(
        strategy=config.strategy,
        market=f"{ticker}:yes",
        price_cents=ask_price,
        count=count,
        dry_run=config.dry_run,
        metadata={
            "target_temp": int(round(target_temp)),
            "observed_max": target_temp,
        },
    )

    _send_telegram(
        f"<b>{tag} {ticker}</b> — YES @ {ask_price}c x{count}\n"
        f"Target: {int(round(target_temp))}°F | Strategy: {config.strategy}\n"
        f"{'[DRY RUN]' if config.dry_run else '[LIVE]'}\n"
        f"<a href=\"{_kalshi_url(series_ticker)}\">View on Kalshi</a>",
        image_path=project_path("charts", "weather", f"momentum_{site}.png"),
    )

    if config.dry_run:
        log.info(f"{tag} [DRY RUN] Would place order: "
                 f"ticker={ticker}, side=yes, price={ask_price}c, count={count}")
        return

    # TODO: re-enable when ready to go live
    log.info(f"{tag} Order placement DISABLED — would have placed: ticker={ticker}, side=yes, price={ask_price}c, count={count}")


def _run_single_market_watch(
    config: BotConfig,
    client: "KalshiClient",
    market_type: str,
    site: str,
    series_ticker: str,
    label: str,
) -> None:
    """Watch one market end-to-end: poll until trigger, then place order."""
    tag = f"[{site}]"
    city = STATIONS[site][0] if site in STATIONS else site
    log.info(f"{tag} Starting watch for {series_ticker} ({city})")

    stop_event = watch_state.get_stop_event(site)
    result = _watch_loop(config, market_type, label, client, series_ticker,
                         site=site, stop_event=stop_event)
    if result is None:
        log.info(f"{tag} No bet triggered.")
        return

    if config.strategy == "claude":
        if isinstance(result, dict) and result.get('action') == 'no_bet':
            log.info(f"{tag} No bet triggered.")
            return
        _place_claude_order(config, client, result, site, series_ticker)
    else:
        _place_bracket_order(config, client, series_ticker, result, site)


# ---------------------------------------------------------------------------
# Flask app + routes
# ---------------------------------------------------------------------------

HTML_DIR = project_path("html")

app = Flask(__name__, static_folder=HTML_DIR, static_url_path="/static")
app.secret_key = os.environ.get("FLASK_SECRET_KEY", os.urandom(32).hex())

# Simple password auth — set WEB_PASSWORD in .env
_WEB_PASSWORD = os.environ.get("WEB_PASSWORD", "")


@app.before_request
def _check_auth():
    if not _WEB_PASSWORD:
        return  # no password set, skip auth
    if flask_request.cookies.get("auth") == _WEB_PASSWORD:
        return
    if flask_request.endpoint == "login" and flask_request.method == "POST":
        return
    if flask_request.endpoint == "login_page":
        return
    return Response(
        '<form method="POST" action="/login" style="font-family:monospace;margin:2em">'
        '<label>Password: <input type="password" name="pw" autofocus></label> '
        '<button type="submit">Login</button></form>',
        status=401,
        content_type="text/html",
    )


@app.route("/login", methods=["POST"])
def login():
    pw = flask_request.form.get("pw", "")
    if pw == _WEB_PASSWORD:
        resp = Response(status=302, headers={"Location": "/"})
        resp.set_cookie("auth", pw, httponly=True, samesite="Strict", max_age=86400 * 30)
        return resp
    return Response("Wrong password", status=401)


@app.route("/login")
def login_page():
    return Response(
        '<form method="POST" action="/login" style="font-family:monospace;margin:2em">'
        '<label>Password: <input type="password" name="pw" autofocus></label> '
        '<button type="submit">Login</button></form>',
        content_type="text/html",
    )


# Store site_tickers discovered at watch start (shared between routes)
_site_tickers: Dict[str, str] = {}
_site_tickers_lock = threading.Lock()


def _load_template(name: str) -> str:
    with open(os.path.join(HTML_DIR, name)) as f:
        return f.read()


@app.route("/")
def dashboard():
    html = _load_template("dashboard.html")
    stations_json = json.dumps({k: v[0] for k, v in STATIONS.items()})
    html = html.replace("{{STATIONS_JSON}}", stations_json)
    return Response(html, content_type="text/html")


@app.route("/api/status")
def api_status():
    return jsonify(watch_state.get_all())


@app.route("/api/bets")
def api_bets():
    limit = flask_request.args.get("limit", 50, type=int)
    return jsonify(get_recent_bets(limit))


@app.route("/api/watch/start", methods=["POST"])
def api_watch_start():
    data = flask_request.get_json(force=True)
    sites = data.get("sites", list(STATIONS.keys()))
    strategy = data.get("strategy", "claude")
    market = data.get("market", "high")
    dry_run = data.get("dry_run", True)
    watch_interval = data.get("watch_interval", 300)

    config = BotConfig(
        strategy=strategy, market=market, dry_run=dry_run,
        watch_interval=watch_interval,
    )
    market_type = config.market
    label = "high" if market_type == "high" else "low"

    try:
        client = _get_client()
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500

    started = []
    for site in sites:
        if site not in STATIONS:
            continue
        if watch_state.is_running(site):
            continue

        # Discover ticker
        with _site_tickers_lock:
            ticker = _site_tickers.get(site)
        if not ticker:
            ticker = _discover_series_ticker(client, site, market_type)
            if ticker:
                with _site_tickers_lock:
                    _site_tickers[site] = ticker

        if not ticker:
            log.warning(f"[{site}] No {market_type} market found — skipping")
            continue

        city = STATIONS[site][0]
        watch_state.init_site(site, city, series_ticker=ticker,
                              strategy=strategy, market_type=market_type)

        t = threading.Thread(
            target=_run_single_market_watch,
            args=(config, client, market_type, site, ticker, label),
            daemon=True,
        )
        watch_state.set_thread(site, t)
        t.start()
        started.append(site)

    return jsonify({"started": started})


@app.route("/api/watch/stop", methods=["POST"])
def api_watch_stop():
    data = flask_request.get_json(force=True)
    sites = data.get("sites", [])
    for site in sites:
        watch_state.stop_site(site)
    return jsonify({"stopped": sites})


@app.route("/api/watch/stop-all", methods=["POST"])
def api_watch_stop_all():
    watch_state.stop_all()
    return jsonify({"status": "stopping all"})


@app.route("/site/<site>")
def site_page(site):
    site = site.upper()
    if not site.startswith("K"):
        site = "K" + site
    if site not in STATIONS:
        return Response(f"Unknown site {site}", status=404)
    city, suffix = STATIONS[site]
    kalshi_url = f"https://kalshi.com/markets/kxhigh{suffix.lower()}"

    # Look up series ticker for real market brackets
    series_ticker = ""
    status = watch_state.get_all().get(site, {})
    series_ticker = status.get("series_ticker", "")
    if not series_ticker:
        with _site_tickers_lock:
            series_ticker = _site_tickers.get(site, "")

    # Always regenerate momentum chart on page load for fresh data
    try:
        _, _, _, _, mom_df, fcst_df = momentum_strategy(site)
        _update_momentum_chart(site, mom_df, fcst_df, series_ticker=series_ticker or None)
    except Exception as e:
        log.debug(f"[{site}] Could not generate chart: {e}")

    html = (_load_template("site.html")
            .replace("{{SITE}}", site)
            .replace("{{CITY}}", city)
            .replace("{{KALSHI_URL}}", kalshi_url))
    return Response(html, content_type="text/html")


@app.route("/api/markets/<site>")
def api_markets(site):
    """Return today's market brackets with orderbook prices for a site."""
    site = site.upper()
    if not site.startswith("K"):
        site = "K" + site
    if site not in STATIONS:
        return jsonify({"error": f"Unknown site {site}"}), 404

    from weather.forecast import STATIONS as FORECAST_STATIONS
    station_tz = ZoneInfo(FORECAST_STATIONS[site][2]) if site in FORECAST_STATIONS else ZoneInfo("America/Los_Angeles")
    today_suffix = datetime.now(station_tz).strftime("%y%b%d").upper()

    # Get series ticker from watch state or discover
    status = watch_state.get_all().get(site, {})
    series_ticker = status.get("series_ticker", "")
    if not series_ticker:
        with _site_tickers_lock:
            series_ticker = _site_tickers.get(site, "")

    if not series_ticker:
        return jsonify({"error": "No series ticker discovered", "markets": []})

    try:
        client = _get_client()
        resp = client.get_markets(series_ticker)
        today_markets = [m for m in resp.get("markets", [])
                         if today_suffix in m.get("ticker", "")]

        brackets = []
        for market in today_markets:
            label_str, lo, hi, ticker = parse_bracket(market)

            # Fetch orderbook prices
            yes_bid, yes_ask = 0, 0
            try:
                ob = client.get_orderbook(ticker)
                yes_bids = ob.get("yes", [])
                no_bids = ob.get("no", [])
                if yes_bids:
                    yes_bid = yes_bids[-1][0]
                if no_bids:
                    yes_ask = 100 - no_bids[-1][0]
            except Exception:
                pass

            brackets.append({
                "ticker": ticker,
                "label": label_str,
                "lo": lo,
                "hi": hi,
                "yes_bid": yes_bid,
                "yes_ask": yes_ask,
            })

        # Sort by lo bound
        brackets.sort(key=lambda b: (b["lo"], b["hi"]))

        kalshi_url = _kalshi_url(series_ticker)
        return jsonify({
            "series_ticker": series_ticker,
            "kalshi_url": kalshi_url,
            "nws_url": f"https://www.weather.gov/wrh/timeseries?site={site}",
            "markets": brackets,
        })
    except Exception as e:
        return jsonify({"error": str(e), "markets": []}), 500


@app.route("/api/chart/<site>")
def api_chart(site):
    chart_path = project_path("charts", "weather", f"momentum_{site.upper()}.png")
    if os.path.isfile(chart_path):
        return send_file(chart_path, mimetype="image/png")
    return Response("No chart", status=404)


@app.route("/api/predict/<site>", methods=["POST"])
def api_predict(site):
    site = site.upper()
    if not site.startswith("K"):
        site = "K" + site
    if site not in STATIONS:
        return jsonify({"error": f"Unknown site {site}"}), 400

    data = flask_request.get_json(force=True) if flask_request.is_json else {}
    market_type = data.get("market", "high")
    label = "high" if market_type == "high" else "low"

    try:
        client = _get_client()
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500

    # Discover ticker
    with _site_tickers_lock:
        ticker = _site_tickers.get(site)
    if not ticker:
        ticker = _discover_series_ticker(client, site, market_type)
        if ticker:
            with _site_tickers_lock:
                _site_tickers[site] = ticker

    if not ticker:
        return jsonify({"error": f"No {market_type} market found for {site}"}), 404

    config = BotConfig(strategy="claude", market=market_type, dry_run=True)
    decision = run_strategy(market_type, label, ticker, site=site, client=client)
    return jsonify(decision or {"action": "no_bet"})


# ---------------------------------------------------------------------------
# Analysis pages
# ---------------------------------------------------------------------------

@app.route("/analysis")
def analysis_page():
    return Response(_load_template("analysis.html"), content_type="text/html")


@app.route("/analysis/<site>")
def analysis_site_page(site):
    site = site.upper()
    if not site.startswith("K"):
        site = "K" + site
    html = _load_template("analysis_site.html").replace("{{SITE}}", site)
    return Response(html, content_type="text/html")


@app.route("/api/analysis")
def api_analysis():
    results_path = project_path("data", "analysis_results.json")
    if not os.path.isfile(results_path):
        return jsonify([]), 404
    with open(results_path) as f:
        return jsonify(json.load(f))


@app.route("/api/analysis/chart/<name>")
def api_analysis_chart(name):
    # Sanitize: only allow alphanumeric, underscore, hyphen
    import re as _re
    if not _re.match(r'^[A-Za-z0-9_-]+$', name):
        return Response("Invalid chart name", status=400)
    charts_dir = project_path("charts")
    chart_path = os.path.join(charts_dir, f"{name}.png")
    if os.path.isfile(chart_path):
        return send_file(chart_path, mimetype="image/png")
    return Response("No chart", status=404)


# ---------------------------------------------------------------------------
# NBA routes
# ---------------------------------------------------------------------------

@app.route("/nba")
def nba_dashboard():
    return Response(_load_template("nba.html"), content_type="text/html")


@app.route("/api/nba/live")
def api_nba_live():
    """Live games with quarter scores + projections."""
    try:
        from nba.data import get_live_scoreboard, build_team_quarter_profiles, build_team_quarter_profiles_ha
        from nba.strategy import (project_game_total, project_spread,
                                  project_halftime_total, filter_halftime_bet,
                                  _total_to_probability, MAX_PACE_DEVIATION, MIN_PROFILE_GAMES)
    except ImportError as e:
        return jsonify({"error": f"nba modules not available: {e}"}), 500

    try:
        games = get_live_scoreboard()
    except Exception as e:
        return jsonify({"error": str(e), "games": []}), 200

    # Load profiles for projections (cached, fast after first call)
    try:
        profiles = build_team_quarter_profiles()
    except Exception:
        profiles = {}

    try:
        profiles_ha = build_team_quarter_profiles_ha()
    except Exception:
        profiles_ha = {}

    enriched = []
    live_count = 0
    for g in games:
        if g["status"] == "live":
            live_count += 1
            g["projection"] = project_game_total(g, profiles)
            g["spread_projection"] = project_spread(g, profiles)

            # Halftime model for P2+ games
            if g.get("period", 0) >= 2 and profiles_ha:
                ht = project_halftime_total(g, profiles_ha)
                if ht:
                    g["halftime_projection"] = ht

                    # O/U probability table: 5 lines centered on projected total
                    proj = ht["projected_total"]
                    std = ht["calibrated_std"]
                    center = round(proj / 5.0) * 5  # round to nearest 5
                    ou_table = []
                    for offset in [-10, -5, 0, 5, 10]:
                        line = center + offset
                        p_over = round(_total_to_probability(proj, line, std) * 100, 1)
                        ou_table.append({"line": line, "over": p_over, "under": round(100 - p_over, 1)})
                    g["ou_table"] = ou_table

                    # Filter status
                    raw_pace = ht.get("raw_pace", 1.0)
                    home_games = ht.get("home_games", 0)
                    away_games = ht.get("away_games", 0)
                    period = g.get("period", 0)
                    pace_ok = abs(raw_pace - 1.0) <= MAX_PACE_DEVIATION
                    depth_ok = home_games >= MIN_PROFILE_GAMES and away_games >= MIN_PROFILE_GAMES
                    not_ot = period <= 4

                    reasons = []
                    if not pace_ok:
                        reasons.append(f"pace extreme ({raw_pace:.3f})")
                    if not depth_ok:
                        reasons.append(f"low depth (H={home_games},A={away_games})")
                    if not not_ot:
                        reasons.append("overtime")

                    g["filter_status"] = {
                        "pace_ok": pace_ok,
                        "profile_depth_ok": depth_ok,
                        "not_overtime": not_ot,
                        "reason": "; ".join(reasons) if reasons else "all filters pass",
                    }

        enriched.append(g)

    return jsonify({"games": enriched, "live_count": live_count})


@app.route("/api/nba/edges")
def api_nba_edges():
    """Scan live games for edges vs Kalshi prices."""
    try:
        from nba.data import build_team_quarter_profiles
        from nba.strategy import find_live_edges
    except ImportError as e:
        return jsonify({"error": f"nba modules not available: {e}"}), 500

    try:
        client = _get_client()
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500

    try:
        profiles = build_team_quarter_profiles()
        edges = find_live_edges(client, profiles)
    except Exception as e:
        log.error("NBA edge scan failed: %s", e, exc_info=True)
        return jsonify({"error": str(e), "edges": []}), 200

    return jsonify({"edges": edges})


@app.route("/api/nba/games")
def api_nba_games():
    """Today's schedule + Kalshi market availability."""
    try:
        from nba.data import get_live_scoreboard
        from nba.markets import discover_nba_series
    except ImportError as e:
        return jsonify({"error": f"nba modules not available: {e}"}), 500

    try:
        games = get_live_scoreboard()
    except Exception as e:
        return jsonify({"error": str(e), "games": []}), 200

    # Try to count Kalshi NBA markets
    kalshi_count = 0
    try:
        client = _get_client()
        series = discover_nba_series(client)
        for ticker in series:
            from nba.markets import get_game_markets
            markets = get_game_markets(client, ticker, status="open")
            kalshi_count += len(markets)
    except Exception:
        pass

    for g in games:
        g["kalshi_markets"] = kalshi_count

    return jsonify({"games": games})


# ---------------------------------------------------------------------------
# NBA checkpoint alerts
# ---------------------------------------------------------------------------

def _start_nba_alerts(confidence: float = 0.90, poll_interval: int = 30) -> bool:
    """Start the NBA checkpoint alerts thread. Returns True if started."""
    global _nba_alerts_thread, _nba_alerts_stop
    with _nba_alerts_lock:
        if _nba_alerts_thread and _nba_alerts_thread.is_alive():
            return False
        try:
            from nba.strategy import watch_checkpoints
        except ImportError:
            log.warning("[NBA-ALERTS] nba.strategy not available — skipping")
            return False
        _nba_alerts_stop = threading.Event()
        _nba_alerts_thread = threading.Thread(
            target=watch_checkpoints,
            kwargs={"confidence": confidence, "poll_interval": poll_interval,
                    "stop_event": _nba_alerts_stop},
            daemon=True,
        )
        _nba_alerts_thread.start()
        log.info("[NBA-ALERTS] Started checkpoint alerts thread")
        return True


def _stop_nba_alerts() -> bool:
    """Stop the NBA checkpoint alerts thread. Returns True if stopped."""
    global _nba_alerts_thread, _nba_alerts_stop
    with _nba_alerts_lock:
        if not _nba_alerts_stop:
            return False
        _nba_alerts_stop.set()
        log.info("[NBA-ALERTS] Stop signal sent")
        return True


@app.route("/api/nba/alerts/start", methods=["POST"])
def api_nba_alerts_start():
    data = flask_request.get_json(force=True) if flask_request.data else {}
    confidence = data.get("confidence", 0.90)
    poll_interval = data.get("poll_interval", 30)
    started = _start_nba_alerts(confidence=confidence, poll_interval=poll_interval)
    if started:
        return jsonify({"status": "started"})
    return jsonify({"status": "already running"})


@app.route("/api/nba/alerts/stop", methods=["POST"])
def api_nba_alerts_stop():
    stopped = _stop_nba_alerts()
    if stopped:
        return jsonify({"status": "stopping"})
    return jsonify({"status": "not running"})


@app.route("/api/nba/alerts/status")
def api_nba_alerts_status():
    with _nba_alerts_lock:
        alive = _nba_alerts_thread is not None and _nba_alerts_thread.is_alive()
    return jsonify({"running": alive})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    init_db()
    port = int(os.getenv("FLASK_PORT", "5000"))
    host = os.getenv("FLASK_HOST", "0.0.0.0")

    # Start Telegram listener if configured
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if token:
        try:
            client = _get_client()
            config = BotConfig()
            site_tickers: Dict[str, str] = {}
            threading.Thread(
                target=_telegram_listener,
                args=(config, client, "high", site_tickers, "high"),
                daemon=True,
            ).start()
        except RuntimeError:
            log.warning("Kalshi credentials not configured — Telegram listener disabled")

    # Start NBA checkpoint alerts
    _start_nba_alerts()

    log.info(f"Starting Flask dashboard on {host}:{port}")
    app.run(host=host, port=port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
