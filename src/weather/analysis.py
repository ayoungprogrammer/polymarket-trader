#!/usr/bin/env python3
"""Historical peak consistency analysis for temperature betting sites.

Fetches ~180 days of 5-min weather observations from the Synoptic API,
extracts daily highs, and computes per-site "peak consistency" metrics
that indicate how predictable each station's daily peak timing and
post-peak decline are.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from paths import project_path
from weather.sites import FORECAST_STATIONS, TRAINING_STATIONS, ALL_SITES, ALL_SITES_WITH_TRAINING
from weather.observations import SYNOPTIC_API_BASE, SYNOPTIC_TOKEN, _synoptic_obs_to_df

log = logging.getLogger(__name__)

DATA_DIR = project_path("data", "weather")

# Combined station coords: Kalshi sites + training-only sites
_ALL_STATION_COORDS = {**TRAINING_STATIONS, **FORECAST_STATIONS}
CHARTS_DIR = project_path("charts")
SOLAR_NOON_CSV = os.path.join(DATA_DIR, "solar_noon.csv")


# ---------------------------------------------------------------------------
# Solar noon (sunrisesunset.io API)
# ---------------------------------------------------------------------------

def fetch_solar_noon(
    site: str,
    dates: list,
) -> pd.DataFrame:
    """Fetch solar noon times from sunrisesunset.io for a site and date list.

    Returns DataFrame with columns: site, date, solar_noon_hour (decimal hours).
    Batches into date ranges to minimize API calls. Sleeps 1s between requests.
    """
    import requests

    coords = _ALL_STATION_COORDS.get(site)
    if not coords:
        log.warning(f"[{site}] No coordinates for site")
        return pd.DataFrame()

    lat, lon, tz = coords

    # Group dates into contiguous ranges for batch fetching
    sorted_dates = sorted(set(dates))
    if not sorted_dates:
        return pd.DataFrame()

    rows = []
    # Batch into 30-day chunks (API supports date ranges)
    chunk_size = 30
    for i in range(0, len(sorted_dates), chunk_size):
        chunk = sorted_dates[i:i + chunk_size]
        date_start = str(chunk[0])
        date_end = str(chunk[-1])

        try:
            resp = requests.get("https://api.sunrisesunset.io/json", params={
                "lat": lat,
                "lng": lon,
                "date_start": date_start,
                "date_end": date_end,
                "time_format": "24",
            }, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            log.warning(f"[{site}] Solar noon API error ({date_start} to {date_end}): {e}")
            time.sleep(1)
            continue

        if data.get("status") != "OK":
            log.warning(f"[{site}] Solar noon API status: {data.get('status')}")
            time.sleep(1)
            continue

        results = data.get("results", [])
        # Single-day response wraps results as a dict, multi-day as a list
        if isinstance(results, dict):
            results = [results]

        for j, entry in enumerate(results):
            noon_str = entry.get("solar_noon", "")
            # Parse HH:MM:SS (24h format) to decimal hours
            if noon_str:
                parts = noon_str.split(":")
                if len(parts) >= 2:
                    hour = int(parts[0]) + int(parts[1]) / 60.0
                    if len(parts) >= 3:
                        hour += int(parts[2]) / 3600.0
                    # Match to the correct date in the chunk
                    entry_date = entry.get("date")
                    if not entry_date and j < len(chunk):
                        entry_date = str(chunk[j])
                    rows.append({
                        "site": site,
                        "date": entry_date,
                        "solar_noon_hour": round(hour, 4),
                    })

        time.sleep(1)

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def load_or_fetch_solar_noon(
    sites: list,
    dates_by_site: dict,
    force: bool = False,
) -> pd.DataFrame:
    """Load cached solar noon CSV, fetching missing site/date combos.

    Args:
        sites: list of ICAO site codes
        dates_by_site: dict mapping site -> list of date strings (YYYY-MM-DD)
        force: if True, re-fetch everything

    Returns DataFrame with columns: site, date, solar_noon_hour
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    existing = pd.DataFrame()
    if not force and os.path.isfile(SOLAR_NOON_CSV):
        existing = pd.read_csv(SOLAR_NOON_CSV)
        log.info(f"Loaded {len(existing)} cached solar noon rows")

    frames = [existing] if not existing.empty else []

    for site in sites:
        dates = dates_by_site.get(site, [])
        if not dates:
            continue

        # Find dates not already cached
        if not existing.empty:
            cached = existing[existing["site"] == site]["date"].astype(str).tolist()
            missing = [d for d in dates if str(d) not in cached]
        else:
            missing = list(dates)

        if not missing:
            continue

        log.info(f"[{site}] Fetching solar noon for {len(missing)} dates")
        new = fetch_solar_noon(site, missing)
        if not new.empty:
            frames.append(new)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["site", "date"]).sort_values(
        ["site", "date"]).reset_index(drop=True)

    # Save cache
    combined.to_csv(SOLAR_NOON_CSV, index=False)
    log.info(f"Cached {len(combined)} solar noon rows to {SOLAR_NOON_CSV}")
    return combined


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def fetch_historical_obs(
    site: str,
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    """Fetch historical Synoptic observations in 7-day chunks.

    Calls the Synoptic timeseries API with explicit start/end params
    (format YYYYmmddHHMM UTC) instead of ``recent``.  Sleeps 1 s
    between requests to avoid rate limits.
    """
    import requests

    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0",
        "Referer": f"https://www.weather.gov/wrh/timeseries?site={site}",
        "Origin": "https://www.weather.gov",
    })

    url = f"{SYNOPTIC_API_BASE}/stations/timeseries"
    chunk_days = 7
    frames = []
    cursor = start

    while cursor < end:
        chunk_end = min(cursor + timedelta(days=chunk_days), end)
        params = {
            "STID": site,
            "start": cursor.strftime("%Y%m%d%H%M"),
            "end": chunk_end.strftime("%Y%m%d%H%M"),
            "vars": "air_temp,air_temp_high_6_hour,air_temp_low_6_hour,"
                    "air_temp_high_24_hour,air_temp_low_24_hour,"
                    "dew_point_temperature,relative_humidity,wind_speed,"
                    "cloud_layer_1_code,sea_level_pressure,pressure_tendency",
            "showemptystations": "1",
            "complete": "1",
            "units": "metric",
            "token": SYNOPTIC_TOKEN,
            "obtimezone": "local",
        }

        log.info(f"[{site}] Fetching {cursor.date()} to {chunk_end.date()}")
        try:
            resp = session.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            summary = data.get("SUMMARY", {})
            if summary.get("RESPONSE_CODE") != 1:
                log.warning(f"[{site}] API error: {summary.get('RESPONSE_MESSAGE')}")
            else:
                stations = data.get("STATION", [])
                if stations:
                    obs = stations[0].get("OBSERVATIONS", {})
                    df = _synoptic_obs_to_df(obs)
                    if not df.empty:
                        frames.append(df)
        except Exception as e:
            log.warning(f"[{site}] Fetch error for {cursor.date()}: {e}")

        cursor = chunk_end
        time.sleep(1)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return combined


def load_or_fetch(
    site: str,
    days: int = 180,
    force: bool = False,
) -> pd.DataFrame:
    """Load cached history CSV, fetching missing date ranges as needed."""
    os.makedirs(DATA_DIR, exist_ok=True)
    csv_path = os.path.join(DATA_DIR, f"history_{site}.csv")

    from zoneinfo import ZoneInfo
    tz_name = _ALL_STATION_COORDS.get(site, (0, 0, "America/New_York"))[2]
    tz = ZoneInfo(tz_name)
    now_local = datetime.now(tz)
    target_start = (now_local - timedelta(days=days)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    target_end = now_local.replace(hour=23, minute=59, second=59, microsecond=0)

    existing = pd.DataFrame()
    if not force and os.path.isfile(csv_path):
        existing = pd.read_csv(csv_path)
        if not existing.empty and "timestamp" in existing.columns:
            # Force re-fetch if cached CSV lacks required fields
            if "max_temp_6h_f" not in existing.columns:
                log.info(f"[{site}] Cache missing 6h max fields — forcing re-fetch")
                existing = pd.DataFrame()
            elif "max_temp_24h_f" not in existing.columns:
                log.info(f"[{site}] Cache missing 24h max fields — forcing re-fetch")
                existing = pd.DataFrame()
            elif "wind_speed_mph" not in existing.columns:
                log.info(f"[{site}] Cache missing met fields (wind/dew/rh) — forcing re-fetch")
                existing = pd.DataFrame()
            else:
                log.info(f"[{site}] Loaded {len(existing)} cached rows from {csv_path}")

    # Determine what we already have
    if not existing.empty and "timestamp" in existing.columns:
        existing_ts = pd.to_datetime(existing["timestamp"].str[:19])
        cached_start = existing_ts.min().to_pydatetime().replace(tzinfo=None)
        cached_end = existing_ts.max().to_pydatetime().replace(tzinfo=None)
    else:
        cached_start = None
        cached_end = None

    target_start_naive = target_start.replace(tzinfo=None)
    target_end_naive = target_end.replace(tzinfo=None)

    frames = [existing] if not existing.empty else []

    # Fetch before cached range
    if cached_start is None or target_start_naive < cached_start - timedelta(hours=1):
        fetch_end = cached_start if cached_start else target_end_naive
        new = fetch_historical_obs(
            site,
            target_start_naive,
            fetch_end,
        )
        if not new.empty:
            frames.append(new)

    # Fetch after cached range
    if cached_end is None or target_end_naive > cached_end + timedelta(hours=1):
        fetch_start = cached_end if cached_end else target_start_naive
        new = fetch_historical_obs(
            site,
            fetch_start,
            target_end_naive,
        )
        if not new.empty:
            frames.append(new)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    # Save cache
    combined.to_csv(csv_path, index=False)
    log.info(f"[{site}] Cached {len(combined)} rows to {csv_path}")
    return combined


# ---------------------------------------------------------------------------
# Daily high extraction
# ---------------------------------------------------------------------------

def extract_daily_highs(obs_df: pd.DataFrame, site: str) -> pd.DataFrame:
    """Extract daily high temperature stats from raw observations.

    Groups by calendar day in the station's local timezone.  Drops days
    with fewer than 200 readings (incomplete data).

    Computes ``true_high_f`` / ``true_low_f`` from the ASOS 24-hour
    max/min fields (``max_temp_24h_f`` / ``min_temp_24h_f``) when
    available, falling back to the 5-min observation extremes.
    """
    if obs_df.empty or "temperature_f" not in obs_df.columns:
        return pd.DataFrame()

    df = obs_df.copy()
    df["ts"] = pd.to_datetime(df["timestamp"].str[:19])
    df["temperature_f"] = pd.to_numeric(df["temperature_f"], errors="coerce")
    df = df.dropna(subset=["temperature_f"])

    if df.empty:
        return pd.DataFrame()

    has_24h_max = "max_temp_24h_f" in df.columns
    has_24h_min = "min_temp_24h_f" in df.columns

    if has_24h_max:
        df["max_temp_24h_f"] = pd.to_numeric(df["max_temp_24h_f"], errors="coerce")
    if has_24h_min:
        df["min_temp_24h_f"] = pd.to_numeric(df["min_temp_24h_f"], errors="coerce")

    df["date"] = df["ts"].dt.date

    rows = []
    for date, grp in df.groupby("date"):
        n = len(grp)
        if n < 200:
            continue

        obs_high = float(grp["temperature_f"].max())
        obs_low = float(grp["temperature_f"].min())

        max_idx = grp["temperature_f"].idxmax()
        peak_row = grp.loc[max_idx]
        peak_ts = peak_row["ts"]
        peak_hour = peak_ts.hour + peak_ts.minute / 60.0

        # True high from 6h METARs (skip before 01:00 — previous evening window)
        if "max_temp_6h_f" not in grp.columns:
            raise ValueError("max_temp_6h_f column missing — re-fetch history data")
        _m6h_grp = grp[grp["max_temp_6h_f"].notna() & (grp["ts"].dt.hour >= 1)]
        if _m6h_grp.empty:
            continue  # skip day with no daytime 6h METAR
        true_high = float(pd.to_numeric(_m6h_grp["max_temp_6h_f"], errors="coerce").max())

        # True low: 24h ASOS min, fallback to obs min
        true_low = obs_low
        if has_24h_min:
            val = grp["min_temp_24h_f"].min()
            if not np.isnan(val):
                true_low = float(val)

        rows.append({
            "site": site,
            "date": str(date),
            "daily_high_f": obs_high,
            "daily_low_f": obs_low,
            "true_high_f": round(true_high, 1),
            "true_low_f": round(true_low, 1),
            "daily_range_f": round(obs_high - obs_low, 1),
            "peak_time": peak_ts.isoformat(),
            "peak_hour": round(peak_hour, 2),
            "n_readings": n,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Peak metrics computation
# ---------------------------------------------------------------------------

def _compute_per_day_metrics(
    daily_row: dict,
    obs_df: pd.DataFrame,
) -> dict:
    """Compute post-peak metrics for a single day from raw observations."""
    peak_ts = pd.Timestamp(daily_row["peak_time"])
    daily_high = daily_row["daily_high_f"]
    date_str = daily_row["date"]

    df = obs_df.copy()
    df["ts"] = pd.to_datetime(df["timestamp"].str[:19])
    df["temperature_f"] = pd.to_numeric(df["temperature_f"], errors="coerce")

    # Filter to same day
    day_mask = df["ts"].dt.date.astype(str) == date_str
    day_df = df[day_mask].sort_values("ts").reset_index(drop=True)

    if day_df.empty:
        return {}

    # After-peak observations
    after_peak = day_df[day_df["ts"] >= peak_ts].reset_index(drop=True)

    # Duration near peak (within 1 deg F)
    near_peak = day_df[day_df["temperature_f"] >= daily_high - 1.0]
    if len(near_peak) >= 2:
        near_peak_ts = near_peak["ts"]
        duration_min = (near_peak_ts.max() - near_peak_ts.min()).total_seconds() / 60.0
    else:
        duration_min = 0.0

    # Time to 3 deg decline
    time_to_3deg = None
    if not after_peak.empty:
        below_3 = after_peak[after_peak["temperature_f"] <= daily_high - 3.0]
        if not below_3.empty:
            first_below = below_3["ts"].iloc[0]
            time_to_3deg = (first_below - peak_ts).total_seconds() / 60.0

    # Rate 30 min after peak
    rate_30min = None
    if not after_peak.empty:
        window_end = peak_ts + pd.Timedelta(minutes=30)
        window = after_peak[(after_peak["ts"] >= peak_ts) & (after_peak["ts"] <= window_end)]
        if len(window) >= 2:
            t_start = window["temperature_f"].iloc[0]
            t_end = window["temperature_f"].iloc[-1]
            elapsed_hr = (window["ts"].iloc[-1] - window["ts"].iloc[0]).total_seconds() / 3600.0
            if elapsed_hr > 0.05:
                rate_30min = (t_end - t_start) / elapsed_hr

    return {
        "duration_near_peak_min": round(duration_min, 1),
        "time_to_3deg_decline_min": round(time_to_3deg, 1) if time_to_3deg is not None else None,
        "rate_30min_after_peak": round(rate_30min, 2) if rate_30min is not None else None,
    }


def compute_peak_metrics(daily_df: pd.DataFrame, obs_df: pd.DataFrame, site: str) -> dict:
    """Compute aggregated peak consistency metrics for a site."""
    if daily_df.empty:
        return {"site": site, "n_days": 0, "score": 0}

    site_daily = daily_df[daily_df["site"] == site].copy()
    if site_daily.empty:
        return {"site": site, "n_days": 0, "score": 0}

    # Per-day metrics
    per_day = []
    for _, row in site_daily.iterrows():
        metrics = _compute_per_day_metrics(row.to_dict(), obs_df)
        if metrics:
            per_day.append(metrics)

    per_day_df = pd.DataFrame(per_day) if per_day else pd.DataFrame()

    # Aggregated stats
    peak_hours = site_daily["peak_hour"].dropna()
    peak_hour_mean = float(peak_hours.mean()) if not peak_hours.empty else 0
    peak_hour_std = float(peak_hours.std()) if len(peak_hours) > 1 else 3.0

    near_peak_durations = per_day_df["duration_near_peak_min"].dropna() if not per_day_df.empty and "duration_near_peak_min" in per_day_df.columns else pd.Series(dtype=float)
    near_peak_mean = float(near_peak_durations.mean()) if not near_peak_durations.empty else 0
    near_peak_std = float(near_peak_durations.std()) if len(near_peak_durations) > 1 else 60.0

    decline_rates = per_day_df["rate_30min_after_peak"].dropna() if not per_day_df.empty and "rate_30min_after_peak" in per_day_df.columns else pd.Series(dtype=float)
    decline_rate_mean = float(decline_rates.mean()) if not decline_rates.empty else 0
    decline_rate_std = float(decline_rates.std()) if len(decline_rates) > 1 else 3.0

    time_to_3deg_vals = per_day_df["time_to_3deg_decline_min"].dropna() if not per_day_df.empty and "time_to_3deg_decline_min" in per_day_df.columns else pd.Series(dtype=float)
    time_to_3deg_mean = float(time_to_3deg_vals.mean()) if not time_to_3deg_vals.empty else 0
    time_to_3deg_std = float(time_to_3deg_vals.std()) if len(time_to_3deg_vals) > 1 else 120.0

    # Composite predictability score (0-100)
    score = (
        25 * max(0, 1 - peak_hour_std / 3)
        + 25 * max(0, 1 - decline_rate_std / 3)
        + 25 * min(near_peak_mean / 120, 1)
        + 25 * min(abs(decline_rate_mean) / 4, 1)
    )

    # Daily high stats
    daily_high_mean = float(site_daily["daily_high_f"].mean())
    daily_high_std = float(site_daily["daily_high_f"].std()) if len(site_daily) > 1 else 0

    tz_name = _ALL_STATION_COORDS.get(site, (0, 0, "America/New_York"))[2]

    return {
        "site": site,
        "timezone": tz_name,
        "n_days": len(site_daily),
        "peak_hour_mean": round(peak_hour_mean, 2),
        "peak_hour_std": round(peak_hour_std, 2),
        "near_peak_duration_mean": round(near_peak_mean, 1),
        "near_peak_duration_std": round(near_peak_std, 1),
        "decline_rate_mean": round(decline_rate_mean, 2),
        "decline_rate_std": round(decline_rate_std, 2),
        "time_to_3deg_mean": round(time_to_3deg_mean, 1),
        "time_to_3deg_std": round(time_to_3deg_std, 1),
        "daily_high_mean": round(daily_high_mean, 1),
        "daily_high_std": round(daily_high_std, 1),
        "score": round(score, 1),
    }


# ---------------------------------------------------------------------------
# Chart generation
# ---------------------------------------------------------------------------

def _setup_dark_style():
    """Configure matplotlib for dark theme charts."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": "#0f1117",
        "axes.facecolor": "#161b22",
        "axes.edgecolor": "#30363d",
        "axes.labelcolor": "#e1e4e8",
        "text.color": "#e1e4e8",
        "xtick.color": "#8b949e",
        "ytick.color": "#8b949e",
        "grid.color": "#21262d",
        "legend.facecolor": "#161b22",
        "legend.edgecolor": "#30363d",
    })
    return plt


def generate_peak_consistency_chart(daily_df: pd.DataFrame, site: str) -> str:
    """Generate per-site peak consistency chart.

    Top: histogram of peak hour-of-day.
    Bottom: daily high scatter + 7-day rolling mean.

    Returns the output file path.
    """
    plt = _setup_dark_style()
    os.makedirs(CHARTS_DIR, exist_ok=True)

    site_df = daily_df[daily_df["site"] == site].copy()
    if site_df.empty:
        return ""

    site_df["date_dt"] = pd.to_datetime(site_df["date"])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Top: peak hour histogram
    peak_hours = site_df["peak_hour"].dropna()
    ax1.hist(peak_hours, bins=24, range=(6, 22), color="#58a6ff", edgecolor="#30363d", alpha=0.8)
    ax1.set_xlabel("Peak Hour (local time)")
    ax1.set_ylabel("Count (days)")
    ax1.set_title(f"{site} — Peak Hour Distribution")
    ax1.grid(True, alpha=0.3)
    if not peak_hours.empty:
        mean_hr = peak_hours.mean()
        ax1.axvline(mean_hr, color="#f0883e", linestyle="--", linewidth=2,
                     label=f"Mean: {int(mean_hr)}:{int((mean_hr % 1) * 60):02d}")
        ax1.legend()

    # Bottom: daily high scatter + rolling mean
    ax2.scatter(site_df["date_dt"], site_df["daily_high_f"],
                color="#58a6ff", s=20, alpha=0.6, label="Daily High")
    if len(site_df) >= 7:
        rolling = site_df.set_index("date_dt")["daily_high_f"].rolling(7, min_periods=3).mean()
        ax2.plot(rolling.index, rolling.values, color="#f0883e", linewidth=2,
                 label="7-day Rolling Mean")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Daily High (°F)")
    ax2.set_title(f"{site} — Daily High Temperature")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    fig.autofmt_xdate(rotation=30)

    plt.tight_layout()
    out = os.path.join(CHARTS_DIR, f"peak_consistency_{site}.png")
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def generate_decline_profile_chart(
    daily_df: pd.DataFrame, obs_df: pd.DataFrame, site: str,
) -> str:
    """Generate per-site decline profile chart.

    Mean temp decline curve (minutes after peak vs deg F below peak)
    with +/- 1 std dev shaded band and reference threshold lines.

    Returns the output file path.
    """
    plt = _setup_dark_style()
    os.makedirs(CHARTS_DIR, exist_ok=True)

    site_daily = daily_df[daily_df["site"] == site].copy()
    if site_daily.empty:
        return ""

    df = obs_df.copy()
    df["ts"] = pd.to_datetime(df["timestamp"].str[:19])
    df["temperature_f"] = pd.to_numeric(df["temperature_f"], errors="coerce")

    # Collect decline curves: minutes_after_peak -> deg below peak
    max_minutes = 180
    all_curves = []

    for _, row in site_daily.iterrows():
        peak_ts = pd.Timestamp(row["peak_time"])
        daily_high = row["daily_high_f"]
        date_str = row["date"]

        day_mask = df["ts"].dt.date.astype(str) == date_str
        day_df = df[day_mask].sort_values("ts")
        after = day_df[day_df["ts"] >= peak_ts]

        if after.empty:
            continue

        minutes_after = (after["ts"] - peak_ts).dt.total_seconds() / 60.0
        deg_below = daily_high - after["temperature_f"].values

        curve = pd.DataFrame({
            "minutes": minutes_after.values,
            "deg_below": deg_below,
        })
        curve = curve[curve["minutes"] <= max_minutes]
        if not curve.empty:
            all_curves.append(curve)

    if not all_curves:
        return ""

    # Bin into 5-min intervals and compute mean/std
    bin_edges = np.arange(0, max_minutes + 5, 5)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    means = []
    stds = []

    for i in range(len(bin_edges) - 1):
        vals = []
        for curve in all_curves:
            mask = (curve["minutes"] >= bin_edges[i]) & (curve["minutes"] < bin_edges[i + 1])
            if mask.any():
                vals.append(curve.loc[mask, "deg_below"].mean())
        if vals:
            means.append(np.mean(vals))
            stds.append(np.std(vals))
        else:
            means.append(np.nan)
            stds.append(np.nan)

    means = np.array(means)
    stds = np.array(stds)

    fig, ax = plt.subplots(figsize=(12, 6))

    valid = ~np.isnan(means)
    ax.plot(bin_centers[valid], means[valid], color="#58a6ff", linewidth=2, label="Mean decline")
    ax.fill_between(
        bin_centers[valid],
        (means - stds)[valid],
        (means + stds)[valid],
        color="#58a6ff", alpha=0.2, label="±1 std dev",
    )

    # Reference lines for momentum thresholds
    # -2 deg/hr = -2/60 deg/min: at t minutes, decline = 2*t/60
    t = np.arange(0, max_minutes + 1)
    ax.plot(t, 2 * t / 60, color="#3fb950", linestyle="--", alpha=0.5,
            label="LOCKED (-2°F/hr)")
    ax.plot(t, 1 * t / 60, color="#d29922", linestyle="--", alpha=0.5,
            label="LIKELY (-1°F/hr)")

    ax.set_xlabel("Minutes After Peak")
    ax.set_ylabel("°F Below Peak")
    ax.set_title(f"{site} — Post-Peak Decline Profile")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.invert_yaxis()

    plt.tight_layout()
    out = os.path.join(CHARTS_DIR, f"decline_profile_{site}.png")
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def generate_comparison_chart(results: list) -> str:
    """Generate cross-site comparison bar chart ranked by composite score.

    Returns the output file path.
    """
    plt = _setup_dark_style()
    os.makedirs(CHARTS_DIR, exist_ok=True)

    if not results:
        return ""

    # Sort by score descending
    results = sorted(results, key=lambda r: r.get("score", 0), reverse=True)

    sites = [r["site"] for r in results]
    scores = [r.get("score", 0) for r in results]

    # Component breakdown
    comp_timing = []
    comp_decline_consistency = []
    comp_near_peak = []
    comp_decline_magnitude = []

    for r in results:
        peak_std = r.get("peak_hour_std", 3.0)
        decline_std = r.get("decline_rate_std", 3.0)
        near_peak_mean = r.get("near_peak_duration_mean", 0)
        decline_mean = r.get("decline_rate_mean", 0)

        comp_timing.append(25 * max(0, 1 - peak_std / 3))
        comp_decline_consistency.append(25 * max(0, 1 - decline_std / 3))
        comp_near_peak.append(25 * min(near_peak_mean / 120, 1))
        comp_decline_magnitude.append(25 * min(abs(decline_mean) / 4, 1))

    y = np.arange(len(sites))
    bar_height = 0.6

    fig, ax = plt.subplots(figsize=(12, max(6, len(sites) * 0.5)))

    # Stacked horizontal bars
    ax.barh(y, comp_timing, bar_height, color="#58a6ff", label="Timing consistency")
    ax.barh(y, comp_decline_consistency, bar_height, left=comp_timing,
            color="#3fb950", label="Decline consistency")
    left2 = [a + b for a, b in zip(comp_timing, comp_decline_consistency)]
    ax.barh(y, comp_near_peak, bar_height, left=left2,
            color="#d29922", label="Near-peak duration")
    left3 = [a + b for a, b in zip(left2, comp_near_peak)]
    ax.barh(y, comp_decline_magnitude, bar_height, left=left3,
            color="#f0883e", label="Decline magnitude")

    # Score labels
    for i, s in enumerate(scores):
        ax.text(s + 1, i, f"{s:.0f}", va="center", fontsize=9, color="#e1e4e8")

    ax.set_yticks(y)
    ax.set_yticklabels(sites)
    ax.set_xlabel("Predictability Score (0–100)")
    ax.set_title("Peak Predictability — Site Comparison")
    ax.set_xlim(0, 105)
    ax.grid(True, alpha=0.3, axis="x")
    ax.legend(loc="lower right", fontsize=8)
    ax.invert_yaxis()

    plt.tight_layout()
    out = os.path.join(CHARTS_DIR, "comparison_summary.png")
    plt.savefig(out, dpi=150)
    plt.close()
    return out


# ---------------------------------------------------------------------------
# Forecast accuracy (Open-Meteo Historical Forecast API)
# ---------------------------------------------------------------------------

OPEN_METEO_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"


def compute_forecast_accuracy(
    site: str,
    obs_df: pd.DataFrame,
) -> Optional[dict]:
    """Compare Open-Meteo historical forecast highs vs actual METAR highs.

    Returns dict with MAE, bias, within_1f/2f/3f percentages, or None on error.
    """
    import requests

    coords = _ALL_STATION_COORDS.get(site)
    if not coords:
        return None

    lat, lon, tz = coords

    # Get actual daily highs from METAR data
    df = obs_df.copy()
    df["ts"] = pd.to_datetime(df["timestamp"].str[:19])
    df["temperature_f"] = pd.to_numeric(df["temperature_f"], errors="coerce")
    df["date"] = df["ts"].dt.date.astype(str)
    actual = df.groupby("date")["temperature_f"].max()
    if actual.empty:
        return None

    start_date = actual.index.min()
    end_date = actual.index.max()

    try:
        resp = requests.get(OPEN_METEO_URL, params={
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "daily": "temperature_2m_max",
            "temperature_unit": "fahrenheit",
            "timezone": tz,
        }, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        log.warning(f"Open-Meteo request failed for {site}: {e}")
        return None

    forecast = dict(zip(data["daily"]["time"], data["daily"]["temperature_2m_max"]))

    # Compare overlapping dates
    errors = []
    for date in sorted(set(actual.index) & set(forecast.keys())):
        f_val = forecast[date]
        a_val = actual[date]
        if f_val is not None and not np.isnan(a_val):
            errors.append(f_val - a_val)

    if len(errors) < 7:
        return None

    errors = np.array(errors)
    return {
        "n_days": len(errors),
        "bias": round(float(errors.mean()), 2),
        "mae": round(float(np.abs(errors).mean()), 2),
        "rmse": round(float(np.sqrt((errors**2).mean())), 2),
        "within_1f_pct": round(float((np.abs(errors) <= 1).mean() * 100), 1),
        "within_2f_pct": round(float((np.abs(errors) <= 2).mean() * 100), 1),
        "within_3f_pct": round(float((np.abs(errors) <= 3).mean() * 100), 1),
        "max_error": round(float(np.max(np.abs(errors))), 1),
    }


FORECAST_HIGHS_CSV = os.path.join(DATA_DIR, "forecast_highs.csv")
FORECAST_HOURLY_CSV = os.path.join(DATA_DIR, "forecast_hourly.csv")


def fetch_historical_forecasts(
    sites: Optional[list] = None,
) -> pd.DataFrame:
    """Fetch per-day historical forecast highs from Open-Meteo for all sites.

    Uses the Open-Meteo Historical Forecast API to get the forecast high
    (temperature_2m_max) for each date that has observation data in
    data/history_{SITE}.csv.

    Saves to data/forecast_highs.csv with columns: site, date, forecast_high_f.
    Merges with any existing CSV to avoid re-fetching.
    """
    import requests
    import time as _time

    if sites is None:
        sites = ALL_SITES_WITH_TRAINING

    # Load existing data to avoid re-fetching
    existing = pd.DataFrame()
    if os.path.isfile(FORECAST_HIGHS_CSV):
        existing = pd.read_csv(FORECAST_HIGHS_CSV)
        print(f"  Loaded {len(existing)} existing forecast entries")

    existing_keys = set()
    if not existing.empty:
        existing_keys = set(zip(existing["site"], existing["date"]))

    new_rows: list = []

    for site in sites:
        coords = _ALL_STATION_COORDS.get(site)
        if not coords:
            print(f"  {site}: no coordinates, skipping")
            continue

        lat, lon, tz = coords

        # Get date range from history CSV
        csv_path = os.path.join(DATA_DIR, f"history_{site}.csv")
        if not os.path.isfile(csv_path):
            print(f"  {site}: no history CSV, skipping")
            continue

        hist = pd.read_csv(csv_path)
        ts = pd.to_datetime(hist["timestamp"].str[:19])
        dates = sorted(ts.dt.date.astype(str).unique())
        if not dates:
            continue

        # Filter to dates we don't already have
        needed = [d for d in dates if (site, d) not in existing_keys]
        if not needed:
            print(f"  {site}: all {len(dates)} dates already cached")
            continue

        start_date = needed[0]
        end_date = needed[-1]

        print(f"  {site}: fetching {len(needed)} dates ({start_date} to {end_date})...")

        try:
            resp = requests.get(OPEN_METEO_URL, params={
                "latitude": lat,
                "longitude": lon,
                "start_date": start_date,
                "end_date": end_date,
                "daily": "temperature_2m_max",
                "temperature_unit": "fahrenheit",
                "timezone": tz,
            }, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"  {site}: Open-Meteo request failed: {e}")
            continue

        api_dates = data["daily"]["time"]
        api_highs = data["daily"]["temperature_2m_max"]

        count = 0
        for d, h in zip(api_dates, api_highs):
            if (site, d) not in existing_keys and h is not None:
                new_rows.append({"site": site, "date": d, "forecast_high_f": round(h, 1)})
                count += 1

        print(f"  {site}: got {count} new entries")
        _time.sleep(0.5)  # rate limit courtesy

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = existing

    if not combined.empty:
        combined = combined.sort_values(["site", "date"]).reset_index(drop=True)
        combined.to_csv(FORECAST_HIGHS_CSV, index=False)
        print(f"\n  Saved {len(combined)} entries to {FORECAST_HIGHS_CSV}")

    return combined


def fetch_historical_hourly_forecasts(
    sites: Optional[list] = None,
) -> pd.DataFrame:
    """Fetch hourly Open-Meteo historical forecasts for all sites.

    Saves to data/forecast_hourly.csv with columns:
        site, date, hour, temperature_f

    Each row is one forecast hour (0-23) for one site-day.
    Merges with existing CSV to avoid re-fetching.
    """
    import requests
    import time as _time

    if sites is None:
        sites = ALL_SITES_WITH_TRAINING

    existing = pd.DataFrame()
    if os.path.isfile(FORECAST_HOURLY_CSV):
        existing = pd.read_csv(FORECAST_HOURLY_CSV)
        print(f"  Loaded {len(existing)} existing hourly forecast rows")

    existing_keys = set()
    if not existing.empty:
        existing_keys = set(zip(existing["site"], existing["date"]))

    new_rows: list = []

    for site in sites:
        coords = _ALL_STATION_COORDS.get(site)
        if not coords:
            print(f"  {site}: no coordinates, skipping")
            continue

        lat, lon, tz = coords

        csv_path = os.path.join(DATA_DIR, f"history_{site}.csv")
        if not os.path.isfile(csv_path):
            print(f"  {site}: no history CSV, skipping")
            continue

        hist = pd.read_csv(csv_path)
        ts = pd.to_datetime(hist["timestamp"].str[:19])
        dates = sorted(ts.dt.date.astype(str).unique())
        if not dates:
            continue

        needed = [d for d in dates if (site, d) not in existing_keys]
        if not needed:
            print(f"  {site}: all {len(dates)} dates already cached")
            continue

        # Batch into 30-day chunks to avoid huge API responses
        chunk_size = 30
        count = 0
        for i in range(0, len(needed), chunk_size):
            chunk = needed[i:i + chunk_size]
            start_date = chunk[0]
            end_date = chunk[-1]

            try:
                resp = requests.get(OPEN_METEO_URL, params={
                    "latitude": lat,
                    "longitude": lon,
                    "start_date": start_date,
                    "end_date": end_date,
                    "hourly": "temperature_2m",
                    "temperature_unit": "fahrenheit",
                    "timezone": tz,
                }, timeout=30)
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                print(f"  {site}: Open-Meteo hourly request failed: {e}")
                continue

            api_times = data["hourly"]["time"]
            api_temps = data["hourly"]["temperature_2m"]

            for t, v in zip(api_times, api_temps):
                if v is None:
                    continue
                d = t[:10]
                h = int(t[11:13])
                if (site, d) not in existing_keys:
                    new_rows.append({
                        "site": site,
                        "date": d,
                        "hour": h,
                        "temperature_f": round(v, 1),
                    })
                    count += 1

            _time.sleep(0.3)

        print(f"  {site}: got {count} new hourly entries")

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = existing

    if not combined.empty:
        combined = combined.sort_values(["site", "date", "hour"]).reset_index(drop=True)
        combined.to_csv(FORECAST_HOURLY_CSV, index=False)
        print(f"\n  Saved {len(combined)} hourly entries to {FORECAST_HOURLY_CSV}")

    return combined


# ---------------------------------------------------------------------------
# Main analysis pipeline
# ---------------------------------------------------------------------------

def run_analysis(
    sites: Optional[list] = None,
    days: int = 180,
    fetch_only: bool = False,
    no_fetch: bool = False,
    force: bool = False,
) -> list:
    """Run the full analysis pipeline.

    Returns a list of per-site result dicts.
    """
    if sites is None:
        sites = ALL_SITES_WITH_TRAINING

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(CHARTS_DIR, exist_ok=True)

    # Phase 1: Fetch data
    obs_cache = {}
    for site in sites:
        print(f"\n{'='*60}")
        print(f"  {site} — Loading observations ({days} days)")
        print(f"{'='*60}")
        if no_fetch:
            csv_path = os.path.join(DATA_DIR, f"history_{site}.csv")
            if os.path.isfile(csv_path):
                obs_cache[site] = pd.read_csv(csv_path)
                print(f"  Loaded {len(obs_cache[site])} cached rows")
            else:
                print(f"  No cache found — skipping")
                obs_cache[site] = pd.DataFrame()
        else:
            obs_cache[site] = load_or_fetch(site, days=days, force=force)
            print(f"  {len(obs_cache[site])} total rows")

    if fetch_only:
        print("\n-- fetch-only mode, skipping analysis --")
        return []

    # Phase 2: Extract daily highs
    print(f"\n{'='*60}")
    print(f"  Extracting daily highs")
    print(f"{'='*60}")

    all_daily = []
    for site in sites:
        obs_df = obs_cache.get(site, pd.DataFrame())
        if obs_df.empty:
            print(f"  {site}: no data")
            continue
        daily = extract_daily_highs(obs_df, site)
        if not daily.empty:
            all_daily.append(daily)
            print(f"  {site}: {len(daily)} valid days")
        else:
            print(f"  {site}: no valid days (< 200 readings/day)")

    if not all_daily:
        print("No daily high data extracted — exiting")
        return []

    combined_daily = pd.concat(all_daily, ignore_index=True)
    daily_csv = os.path.join(DATA_DIR, "daily_highs.csv")
    combined_daily.to_csv(daily_csv, index=False)
    print(f"\nSaved {len(combined_daily)} daily records to {daily_csv}")

    # Phase 2b: Fetch solar noon for all extracted dates
    print(f"\n{'='*60}")
    print(f"  Fetching solar noon data")
    print(f"{'='*60}")

    dates_by_site = {}
    for site in sites:
        site_daily = combined_daily[combined_daily["site"] == site]
        if not site_daily.empty:
            dates_by_site[site] = site_daily["date"].astype(str).tolist()

    if not no_fetch:
        solar_df = load_or_fetch_solar_noon(sites, dates_by_site)
        n_solar = len(solar_df) if not solar_df.empty else 0
        print(f"  {n_solar} solar noon records cached")
    else:
        if os.path.isfile(SOLAR_NOON_CSV):
            solar_df = pd.read_csv(SOLAR_NOON_CSV)
            print(f"  Loaded {len(solar_df)} cached solar noon rows")
        else:
            print(f"  No solar noon cache — skipping")

    # Phase 3: Compute metrics
    print(f"\n{'='*60}")
    print(f"  Computing peak metrics")
    print(f"{'='*60}")

    results = []
    for site in sites:
        obs_df = obs_cache.get(site, pd.DataFrame())
        metrics = compute_peak_metrics(combined_daily, obs_df, site)
        results.append(metrics)
        print(f"  {site}: score={metrics['score']:.1f}, "
              f"peak_hour={metrics.get('peak_hour_mean', 0):.1f}±{metrics.get('peak_hour_std', 0):.1f}, "
              f"decline={metrics.get('decline_rate_mean', 0):.2f}±{metrics.get('decline_rate_std', 0):.2f}°F/hr")

    # Phase 3b: Forecast accuracy (Open-Meteo)
    print(f"\n{'='*60}")
    print(f"  Forecast accuracy (Open-Meteo vs actual)")
    print(f"{'='*60}")

    for r in results:
        site = r["site"]
        obs_df = obs_cache.get(site, pd.DataFrame())
        if obs_df.empty:
            continue
        acc = compute_forecast_accuracy(site, obs_df)
        if acc:
            r["forecast_accuracy"] = acc
            print(f"  {site}: MAE={acc['mae']:.1f}°F, bias={acc['bias']:+.1f}°F, "
                  f"within 1°F={acc['within_1f_pct']:.0f}%, "
                  f"within 2°F={acc['within_2f_pct']:.0f}% "
                  f"({acc['n_days']}d)")
            time.sleep(0.5)  # rate limit courtesy
        else:
            print(f"  {site}: no forecast data")

    # Save results JSON
    results_path = os.path.join(DATA_DIR, "analysis_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {results_path}")

    # Phase 4: Generate charts
    print(f"\n{'='*60}")
    print(f"  Generating charts")
    print(f"{'='*60}")

    for site in sites:
        obs_df = obs_cache.get(site, pd.DataFrame())
        if obs_df.empty:
            continue
        out1 = generate_peak_consistency_chart(combined_daily, site)
        if out1:
            print(f"  {site}: {out1}")
        out2 = generate_decline_profile_chart(combined_daily, obs_df, site)
        if out2:
            print(f"  {site}: {out2}")

    comparison_path = generate_comparison_chart(results)
    if comparison_path:
        print(f"  Comparison: {comparison_path}")

    # Print ranked summary
    print(f"\n{'='*60}")
    print(f"  PREDICTABILITY RANKING")
    print(f"{'='*60}")
    ranked = sorted(results, key=lambda r: r.get("score", 0), reverse=True)
    for i, r in enumerate(ranked, 1):
        print(f"  {i:2d}. {r['site']:6s}  score={r['score']:5.1f}  "
              f"peak={r.get('peak_hour_mean', 0):5.1f}h±{r.get('peak_hour_std', 0):.1f}  "
              f"decline={r.get('decline_rate_mean', 0):+.1f}°F/hr  "
              f"n={r.get('n_days', 0)} days")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Historical peak consistency analysis for temperature betting sites"
    )
    parser.add_argument(
        "--sites", type=str, default=None,
        help="Comma-separated ICAO codes (default: all sites)",
    )
    parser.add_argument(
        "--days", type=int, default=180,
        help="Days of history to analyse (default: 180)",
    )
    parser.add_argument(
        "--fetch-only", action="store_true",
        help="Just cache data, skip analysis",
    )
    parser.add_argument(
        "--no-fetch", action="store_true",
        help="Use cached data only, do not fetch",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force re-fetch all data (ignore cache)",
    )
    parser.add_argument(
        "--fetch-forecasts", action="store_true",
        help="Fetch per-day historical forecast highs from Open-Meteo and save to CSV",
    )
    parser.add_argument(
        "--fetch-hourly-forecasts", action="store_true",
        help="Fetch hourly historical forecasts from Open-Meteo and save to CSV",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    sites = None
    if args.sites:
        sites = [s.strip().upper() for s in args.sites.split(",")]
        # Validate
        for s in sites:
            if s not in _ALL_STATION_COORDS:
                parser.error(f"Unknown site {s}. Available: {', '.join(ALL_SITES_WITH_TRAINING)}")

    if args.fetch_hourly_forecasts:
        fetch_historical_hourly_forecasts(sites)
    elif args.fetch_forecasts:
        fetch_historical_forecasts(sites)
    else:
        run_analysis(
            sites=sites,
            days=args.days,
            fetch_only=args.fetch_only,
            no_fetch=args.no_fetch,
            force=args.force,
        )


if __name__ == "__main__":
    main()
