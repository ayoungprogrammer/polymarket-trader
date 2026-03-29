#!/usr/bin/env python3
"""Temperature momentum analysis, settlement prediction, and plotting."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Optional, List, Dict

import pandas as pd

log = logging.getLogger(__name__)


import numpy as np

# MA crossover parameters (minutes)
MA_SHORT_MIN = 30
MA_LONG_MIN = 90

# matplotlib is not thread-safe — serialize all plotting
_plot_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Settlement prediction: reverse-map auto-obs °C → possible °F
# ---------------------------------------------------------------------------

def c_to_possible_f(c_obs: int) -> List[int]:
    """Find all whole °F values that map to c_obs via round((F-32)*5/9).

    The 5-min station rounds the true temp to whole °F, then converts to
    whole °C via standard rounding.  This reverses that: given the °C we
    see, which °F values could have produced it?

    Typically returns 2 values (e.g. C=28 → [82, 83]), sometimes 1
    (when C*9/5+32 lands exactly on an integer, e.g. C=30 → [86]).
    """
    approx_f = c_obs * 9 / 5 + 32
    results = []
    for f in range(int(approx_f) - 2, int(approx_f) + 3):
        if round((f - 32) * 5 / 9) == c_obs:
            results.append(f)
    return sorted(results)


@dataclass
class RoundingPrediction:
    """Predicted settlement °F from a whole-°C auto-obs max."""
    possible_f: List[int]  # original whole °F values that map to this °C
    center_f: int          # best single prediction
    low_f: int             # lowest plausible settlement °F
    high_f: int            # highest plausible settlement °F
    probabilities: Dict[int, float]  # {whole_°F: probability}


def predict_settlement_f(
    c_obs: int,
    dwell_count: int = 1,
    metar_max_c: Optional[float] = None,
) -> RoundingPrediction:
    """Predict settlement whole °F from a max whole-°C auto reading.

    Uses reverse mapping to find the possible original °F values, then
    uses METAR T-group precision (0.1°C) to disambiguate the naive=low
    case where two °F values map to the same °C.

    Parameters
    ----------
    c_obs : int
        Maximum whole-°C reading observed so far (auto obs only).
    dwell_count : int
        Number of 5-min readings at c_obs.
    metar_max_c : float or None
        Maximum fractional-°C reading from METAR T-group observations.
        When available, disambiguates which °F value the auto reading
        corresponds to in the naive=low case.
    """
    possible_f = c_to_possible_f(c_obs)
    naive_f = round(c_obs * 9 / 5 + 32)

    if len(possible_f) == 1:
        f = possible_f[0]
        probs = {f: 0.98, f + 1: 0.02}
        return RoundingPrediction(
            possible_f=possible_f,
            center_f=f,
            low_f=f,
            high_f=f + 1,
            probabilities=probs,
        )

    f_low, f_high = possible_f[0], possible_f[-1]

    if naive_f == f_high:
        probs = {f_high: 0.99, f_high + 1: 0.01}
        center = f_high
    else:
        # naive=low case: METAR precision can resolve the ambiguity.
        threshold_c = (f_low + 0.5 - 32) * 5 / 9

        if metar_max_c is not None and metar_max_c > threshold_c:
            probs = {f_high: 0.92, f_high + 1: 0.06, f_low: 0.02}
            center = f_high
        elif metar_max_c is not None:
            probs = {f_low: 0.95, f_high: 0.04, f_high + 1: 0.01}
            center = f_low
        else:
            probs = {f_low: 0.74, f_high: 0.24, f_high + 1: 0.02}
            center = naive_f

    return RoundingPrediction(
        possible_f=possible_f,
        center_f=center,
        low_f=f_low,
        high_f=f_high + 1,
        probabilities=probs,
    )


def predict_settlement_from_obs(obs_df: pd.DataFrame) -> Optional[RoundingPrediction]:
    """Predict settlement °F from a Synoptic observation DataFrame.

    Extracts auto-obs max °C, dwell count, and METAR max °C from the
    observation data, then calls predict_settlement_f.  Handles the edge
    case where METAR max >> auto max (e.g. cold front, missed peak).

    Returns None if insufficient auto-obs data.
    """
    if obs_df.empty or "temperature_c" not in obs_df.columns:
        return None

    temp_c = pd.to_numeric(obs_df["temperature_c"], errors="coerce")

    # METAR reports at :53 past the hour; everything else is 5-min auto-obs
    minutes = pd.to_datetime(obs_df["timestamp"].str[:19]).dt.minute
    metar_mask = (minutes == 53) & temp_c.notna()
    auto_mask = (minutes != 53) & temp_c.notna()

    auto_c = temp_c[auto_mask].astype(float)
    if len(auto_c) < 5:
        return None

    max_c = int(auto_c.max())
    dwell_count = int((auto_c >= max_c).sum())

    # METAR T-group readings (0.1°C precision, may be whole numbers)
    metar_c = temp_c[metar_mask].astype(float)
    metar_max_c = float(metar_c.max()) if len(metar_c) > 0 else None

    return predict_settlement_f(max_c, dwell_count=dwell_count,
                                metar_max_c=metar_max_c)


def _rolling_mean(
    timestamps: pd.Series, values: pd.Series, window_min: int,
) -> list:
    """Time-based rolling mean using pandas .rolling() — O(n)."""
    try:
        idx = pd.DatetimeIndex(pd.to_datetime(timestamps))
    except (ValueError, TypeError):
        idx = pd.DatetimeIndex(pd.to_datetime(timestamps, utc=True))
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    s = pd.Series(values.values, index=idx, dtype=float)
    needs_sort = not s.index.is_monotonic_increasing
    if needs_sort:
        orig_order = np.arange(len(s))
        sort_idx = s.index.argsort()
        s = s.iloc[sort_idx]
    rolled = s.rolling(f"{window_min}min", min_periods=2).mean()
    if needs_sort:
        # Restore original order
        unsort = np.empty_like(sort_idx)
        unsort[sort_idx] = orig_order
        rolled = rolled.iloc[unsort]
    return [round(v, 2) if not np.isnan(v) else None for v in rolled.values]


def compute_momentum(
    df: pd.DataFrame, window_minutes: int = 60,
    ma_short_min: int = MA_SHORT_MIN, ma_long_min: int = MA_LONG_MIN,
) -> pd.DataFrame:
    """Dual moving-average crossover momentum.

    Adds columns:
    - ``ma_short`` — rolling mean over *ma_short_min* minutes
    - ``ma_long``  — rolling mean over *ma_long_min* minutes
    - ``ma_cross`` — ``ma_short - ma_long`` (negative = cooling)
    - ``rate_f_per_hr`` — OLS slope of ``ma_short`` over a 30-min trailing
      window (smoothed rate, preserves existing interface)
    - ``rate_window_min`` — kept for compatibility
    """
    if df.empty or "temperature_f" not in df.columns:
        return df

    df = df.copy()
    # Parse timestamps to naive datetime64 — strip tz if present
    _raw = df["timestamp"].astype(str).str[:19]  # truncate tz offset
    timestamps = pd.to_datetime(_raw)
    temps = df["temperature_f"].astype(float)

    ma_short_vals = _rolling_mean(timestamps, temps, ma_short_min)
    ma_long_vals = _rolling_mean(timestamps, temps, ma_long_min)

    df["ma_short"] = ma_short_vals
    df["ma_long"] = ma_long_vals
    df["ma_cross"] = [
        round(s - l, 2) if s is not None and l is not None else None
        for s, l in zip(ma_short_vals, ma_long_vals)
    ]

    # Rate: OLS slope of ma_short over trailing window — vectorized
    rate_window = min(ma_short_min, 30)
    ma_short_arr = np.array([v if v is not None else np.nan for v in ma_short_vals], dtype=float)
    ts_seconds = (timestamps - timestamps.iloc[0]).dt.total_seconds().values.astype(float)
    n = len(df)

    # Convert window to approximate row count for fast slicing
    # (data is ~5-min intervals but may vary; use time-based check)
    rates = np.full(n, np.nan)
    rate_window_sec = rate_window * 60.0
    for i in range(n):
        # Walk backward from i to find window start
        j = i
        while j > 0 and (ts_seconds[i] - ts_seconds[j - 1]) <= rate_window_sec:
            j -= 1
        x = ts_seconds[j:i + 1] / 3600.0  # hours
        y = ma_short_arr[j:i + 1]
        valid = ~np.isnan(y)
        x = x[valid]
        y = y[valid]
        if len(x) < 2 or (x[-1] - x[0]) < 0.05:
            continue
        xm = x.mean()
        dx = x - xm
        denom = np.sum(dx ** 2)
        if denom == 0:
            continue
        rates[i] = round(float(np.sum(dx * (y - y.mean())) / denom), 2)

    df["rate_f_per_hr"] = [None if np.isnan(r) else r for r in rates]
    df["rate_window_min"] = window_minutes
    return df


def extrapolate_momentum(
    df: pd.DataFrame, minutes: int = 60, step: int = 5
) -> pd.DataFrame:
    """Project temperature forward using momentum.

    Uses the latest rate for linear extrapolation.  Confidence band is
    derived from the std of rates over the last 3 hours.

    Returns a DataFrame with columns: timestamp, temperature_f, temp_lo, temp_hi.
    """
    rates = df["rate_f_per_hr"].dropna()
    temps = df["temperature_f"].dropna()
    if rates.empty or temps.empty:
        return pd.DataFrame()

    last_ts = pd.to_datetime(df["timestamp"].iloc[-1][:19])
    last_temp = float(temps.iloc[-1])
    last_rate = float(rates.iloc[-1])

    # Std of rates over last 3 hours
    recent_rates = rates.tail(36).astype(float)  # ~3h at 5-min intervals
    if len(recent_rates) > 2:
        rate_std = float(recent_rates.std())
    else:
        rate_std = abs(last_rate) * 0.5

    rows = []
    for i in range(1, minutes // step + 1):
        dt_hr = i * step / 60.0
        proj_ts = last_ts + pd.Timedelta(minutes=i * step)
        proj_temp = last_temp + last_rate * dt_hr
        spread = rate_std * (dt_hr ** 0.5)
        rows.append({
            "timestamp": proj_ts.isoformat(),
            "temperature_f": round(proj_temp, 1),
            "temp_lo": round(proj_temp - spread, 1),
            "temp_hi": round(proj_temp + spread, 1),
        })

    return pd.DataFrame(rows)


def plot_momentum(
    df: pd.DataFrame,
    site: str,
    output: str = "charts/weather/momentum.png",
    forecast_df: Optional[pd.DataFrame] = None,
    locked_rate: Optional[float] = None,
    likely_rate: Optional[float] = None,
    margin_threshold: Optional[float] = None,
    sun_times: Optional[Dict[str, float]] = None,
    metar_6h_f: Optional[float] = None,
    bracket: Optional[List[int]] = None,
    bracket_probs: Optional[List[dict]] = None,
    bracket_error: Optional[str] = None,
    peak_result: Optional[dict] = None,
    metar_6h_local_dt: Optional[object] = None,
    true_max_f: Optional[float] = None,
    cli_reports: Optional[list] = None,
    dsm_reports: Optional[list] = None,
):
    """Plot temperature + MA crossover and save to file.

    *bracket_probs*: list of dicts from bracket model, each with keys
    ``bracket`` (lo, hi), ``prob``, ``stage1_prob``, ``confidence``.

    *peak_result*: dict from peak_model.predict() with keys
    ``probability``, ``prediction``, ``cur_naive_f``.

    *true_max_f*: actual settlement high °F (e.g. from 24h ASOS max).
    Drawn as a horizontal line for backtesting / historical review.

    *cli_reports*: list of parsed CLI dicts from fetch_all_cli_today().
    Each drawn as a horizontal line showing the reported high.

    *dsm_reports*: list of parsed DSM dicts from fetch_dsm_today().

    Thread-safe: uses _plot_lock to serialize matplotlib calls.
    """
    with _plot_lock:
        _plot_momentum_impl(df, site, output, forecast_df,
                            locked_rate, likely_rate, margin_threshold,
                            sun_times, metar_6h_f, bracket, bracket_probs,
                            bracket_error, peak_result, metar_6h_local_dt,
                            true_max_f, cli_reports, dsm_reports)


def _plot_momentum_impl(
    df: pd.DataFrame,
    site: str,
    output: str,
    forecast_df: Optional[pd.DataFrame],
    locked_rate: Optional[float],
    likely_rate: Optional[float],
    margin_threshold: Optional[float],
    sun_times: Optional[Dict[str, float]] = None,
    metar_6h_f: Optional[float] = None,
    bracket: Optional[List[int]] = None,
    bracket_probs: Optional[List[dict]] = None,
    bracket_error: Optional[str] = None,
    peak_result: Optional[dict] = None,
    metar_6h_local_dt: Optional[object] = None,
    true_max_f: Optional[float] = None,
    cli_reports: Optional[list] = None,
    dsm_reports: Optional[list] = None,
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    timestamps = pd.to_datetime(df["timestamp"].str[:19]).values
    temps = df["temperature_f"].to_numpy(dtype=float)
    has_bracket_info = bracket_probs and len(bracket_probs) > 0
    has_bracket_footer = has_bracket_info or bracket_error
    fig_height = 9.5 if has_bracket_footer else 8
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, fig_height), sharex=True,
                                    gridspec_kw={"height_ratios": [2, 1]})

    # Sun times — vertical lines for sunrise, solar noon, sunset
    _sun_markers = []  # (sun_dt, color, label) — annotated after data is plotted
    if sun_times and len(timestamps) > 0:
        obs_date = pd.Timestamp(timestamps[0]).normalize()
        sun_styles = {
            "sunrise":    ("gold",       "--", "Sunrise"),
            "solar_noon": ("darkorange", "-",  "Solar noon"),
            "sunset":     ("gold",       "--", "Sunset"),
        }
        for key, (color, ls, lbl) in sun_styles.items():
            dec_hour = sun_times.get(key)
            if dec_hour is None:
                continue
            h = int(dec_hour)
            m = int((dec_hour - h) * 60)
            s = int(((dec_hour - h) * 60 - m) * 60)
            sun_dt = obs_date + pd.Timedelta(hours=h, minutes=m, seconds=s)
            for ax in (ax1, ax2):
                ax.axvline(sun_dt, color=color, linestyle=ls, alpha=0.5, linewidth=1)
            _sun_markers.append((sun_dt, color, lbl))

    # Temperature plot — observed + MA lines
    ax1.plot(timestamps, temps, color="tab:red", linewidth=1.5, alpha=0.6, label="Observed (°F)")
    if "ma_short" in df.columns:
        ax1.plot(timestamps, df["ma_short"].values, color="tab:orange", linewidth=2,
                 label=f"MA short ({MA_SHORT_MIN}m)")
    if "ma_long" in df.columns:
        ax1.plot(timestamps, df["ma_long"].values, color="tab:blue", linewidth=2,
                 label=f"MA long ({MA_LONG_MIN}m)")
    ax1.set_ylabel("Temperature (°F)")
    from zoneinfo import ZoneInfo as _ZI
    _pst_now = pd.Timestamp.now(tz=_ZI("America/Los_Angeles")).strftime("%-I:%M %p %Z")
    ax1.set_title(f"{site} — Temperature & MA Crossover  (generated {_pst_now})")
    ax1.grid(True, alpha=0.3)

    # Forecast overlay
    if forecast_df is not None and not forecast_df.empty and "temperature_f" in forecast_df.columns:
        fc_ts = pd.to_datetime(forecast_df["timestamp"].str[:19]).values
        fc_temps = forecast_df["temperature_f"].to_numpy(dtype=float)
        ax1.plot(fc_ts, fc_temps, color="tab:purple", linewidth=2, linestyle="--",
                 alpha=0.7, label="NWS Forecast (°F)")
        if np.any(~np.isnan(fc_temps)):
            fc_max_idx = int(np.nanargmax(fc_temps))
            peak_time = fc_ts[fc_max_idx]
            peak_lo = peak_time - np.timedelta64(30, 'm')
            peak_hi = peak_time + np.timedelta64(30, 'm')
            ax1.axvspan(peak_lo, peak_hi, color="tab:purple", alpha=0.1)
            ax1.annotate(f"Fcst peak: {fc_temps[fc_max_idx]:.0f}°F",
                         xy=(peak_time, fc_temps[fc_max_idx]),
                         xytext=(-60, 15), textcoords="offset points",
                         fontsize=9, color="tab:purple",
                         arrowprops=dict(arrowstyle="->", color="tab:purple", alpha=0.7))
            ax2.axvspan(peak_lo, peak_hi, color="tab:purple", alpha=0.1)

            fc_min_idx = int(np.nanargmin(fc_temps))
            low_time = fc_ts[fc_min_idx]
            low_lo = low_time - np.timedelta64(30, 'm')
            low_hi = low_time + np.timedelta64(30, 'm')
            ax1.axvspan(low_lo, low_hi, color="tab:blue", alpha=0.08)
            ax1.annotate(f"Fcst low: {fc_temps[fc_min_idx]:.0f}°F",
                         xy=(low_time, fc_temps[fc_min_idx]),
                         xytext=(10, -20), textcoords="offset points",
                         fontsize=9, color="tab:blue",
                         arrowprops=dict(arrowstyle="->", color="tab:blue", alpha=0.7))
            ax2.axvspan(low_lo, low_hi, color="tab:blue", alpha=0.08)

    # Momentum extrapolation (+60 min)
    extrap = extrapolate_momentum(df)
    if not extrap.empty:
        ex_ts = pd.to_datetime(extrap["timestamp"]).values
        ex_temps = extrap["temperature_f"].values
        ex_lo = extrap["temp_lo"].values
        ex_hi = extrap["temp_hi"].values
        bridge_ts = np.concatenate([[timestamps[-1]], ex_ts])
        bridge_temp = np.concatenate([[temps[-1]], ex_temps])
        bridge_lo = np.concatenate([[temps[-1]], ex_lo])
        bridge_hi = np.concatenate([[temps[-1]], ex_hi])
        ax1.plot(bridge_ts, bridge_temp, color="tab:green", linewidth=2, linestyle=":",
                 label=f"Momentum +60m ({ex_temps[-1]:.1f}°F)")
        ax1.fill_between(bridge_ts, bridge_lo, bridge_hi,
                         color="tab:green", alpha=0.1)
        ax1.annotate(f"{ex_temps[-1]:.1f}°F",
                     xy=(ex_ts[-1], ex_temps[-1]),
                     xytext=(10, 5), textcoords="offset points",
                     fontsize=9, color="tab:green",
                     arrowprops=dict(arrowstyle="->", color="tab:green", alpha=0.7))

    # Mark observed max + margin threshold
    if np.any(~np.isnan(temps)):
        max_idx = int(np.nanargmax(temps))
        obs_max_val = temps[max_idx]
        ax1.axhline(y=obs_max_val, color="tab:red", linestyle="--", alpha=0.5,
                     label=f"Obs max: {obs_max_val:.1f}°F")
        ax1.annotate(f"{obs_max_val:.1f}°F",
                     xy=(timestamps[max_idx], obs_max_val),
                     xytext=(10, 10), textcoords="offset points",
                     fontsize=9, color="tab:red",
                     arrowprops=dict(arrowstyle="->", color="tab:red", alpha=0.7))
        mt = margin_threshold if margin_threshold is not None else 2.0
        trigger_level = obs_max_val - mt
        ax1.axhline(y=trigger_level, color="tab:green", linestyle=":", alpha=0.4,
                     label=f"Margin trigger ({mt}°F below max)")

    # Mark observed min
    if np.any(~np.isnan(temps)):
        min_idx = int(np.nanargmin(temps))
        ax1.axhline(y=temps[min_idx], color="tab:blue", linestyle="--", alpha=0.5,
                     label=f"Obs min: {temps[min_idx]:.1f}°F")
        ax1.annotate(f"{temps[min_idx]:.1f}°F",
                     xy=(timestamps[min_idx], temps[min_idx]),
                     xytext=(10, -15), textcoords="offset points",
                     fontsize=9, color="tab:blue",
                     arrowprops=dict(arrowstyle="->", color="tab:blue", alpha=0.7))

    # True settlement max (24h ASOS) — for historical review
    if true_max_f is not None:
        ax1.axhline(y=true_max_f, color="red", linestyle="-", alpha=0.7, linewidth=1.5,
                     label=f"Settlement: {true_max_f:.0f}°F")

    # METAR 6h max — most precise observation (passed in as single value)
    if metar_6h_f is not None:
        ax1.axhline(y=metar_6h_f, color="magenta", linestyle="-", alpha=0.6, linewidth=1.5,
                     label=f"METAR 6h max: {metar_6h_f:.1f}°F")

    # CLI reports — vertical line at publish time, star at max time
    if cli_reports and len(timestamps) > 0:
        from datetime import datetime as _dt
        _today_date = pd.Timestamp(timestamps[-1]).normalize()
        _cli_labeled = False
        for cli in cli_reports:
            _cli_f = cli.get("max_temp_f")
            if _cli_f is None:
                continue
            _time_str = cli.get("max_temp_time", "")
            _tag = "prelim" if cli.get("is_preliminary") else "final"

            # Vertical line at issued/publish time
            _issued = cli.get("issued", "")
            if _issued:
                try:
                    import re as _re
                    _im = _re.match(r"(\d{1,4})\s+([AP]M)", _issued)
                    if _im:
                        _raw = _im.group(1).zfill(4)
                        _ih = int(_raw[:-2])
                        _imin = int(_raw[-2:])
                        _ampm = _im.group(2)
                        if _ampm == "PM" and _ih != 12:
                            _ih += 12
                        elif _ampm == "AM" and _ih == 12:
                            _ih = 0
                        _pub_dt = _today_date + pd.Timedelta(hours=_ih, minutes=_imin)
                        _lbl = f"CLI {_tag}: {_cli_f}°F" if not _cli_labeled else None
                        for ax in (ax1, ax2):
                            ax.axvline(_pub_dt, color="red", linestyle="-",
                                       alpha=0.7, linewidth=1.5,
                                       label=_lbl if ax is ax1 else None)
                        _cli_labeled = True
                except Exception:
                    pass

            # Star at max temp time
            if _time_str:
                try:
                    _t = _time_str.strip()
                    if ":" in _t:
                        _parsed = _dt.strptime(_t, "%I:%M %p")
                    else:
                        _parsed = _dt.strptime(_t, "%I%M %p")
                    _max_dt = _today_date + pd.Timedelta(hours=_parsed.hour, minutes=_parsed.minute)
                    ax1.scatter([_max_dt], [_cli_f + 0.3], color="#FF4444", marker="*",
                                s=150, zorder=7, edgecolors="darkred", linewidths=0.5)
                    ax1.annotate(f"CLI {_cli_f}°F", xy=(_max_dt, _cli_f + 0.3), xytext=(5, 8),
                                 textcoords="offset points", fontsize=7,
                                 color="#CC0000", fontweight="bold")
                except (ValueError, IndexError):
                    pass

    # DSM reports — scatter points at max time + vertical lines at publish time
    if dsm_reports and len(timestamps) > 0:
        _today_date = pd.Timestamp(timestamps[-1]).normalize()
        _dsm_labeled = False
        for dsm in dsm_reports:
            _dsm_f = dsm.get("max_temp_f")
            if _dsm_f is None:
                continue
            _time_str = dsm.get("max_temp_time", "")  # local HH:MM
            _utc_str = dsm.get("obs_time_utc", "")    # UTC HHMM

            # Vertical line at DSM publish time (use 'entered' field, not obs_time_utc)
            _entered = dsm.get("entered", "")
            if _entered:
                try:
                    from zoneinfo import ZoneInfo as _ZI
                    from weather.sites import FORECAST_STATIONS
                    _coords = FORECAST_STATIONS.get(site)
                    if _coords:
                        _tz = _ZI(_coords[2])
                        _entered_utc = pd.to_datetime(_entered)
                        if _entered_utc.tzinfo is None:
                            _entered_utc = _entered_utc.tz_localize("UTC")
                        _entered_local = _entered_utc.tz_convert(_tz)
                        _pub_naive = _entered_local.tz_localize(None)
                        _lbl = f"DSM: {_dsm_f}°F" if not _dsm_labeled else None
                        for ax in (ax1, ax2):
                            ax.axvline(_pub_naive, color="#44AAFF", linestyle="-",
                                       alpha=0.7, linewidth=1.5,
                                       label=_lbl if ax is ax1 else None)
                        _dsm_labeled = True
                except Exception:
                    pass

            # Star at max temp time
            if _time_str:
                try:
                    _parts = _time_str.split(":")
                    _h, _m = int(_parts[0]), int(_parts[1])
                    _max_dt = _today_date + pd.Timedelta(hours=_h, minutes=_m)
                    ax1.scatter([_max_dt], [_dsm_f - 0.3], color="#44AAFF", marker="*",
                                s=120, zorder=6, edgecolors="#0066CC", linewidths=0.5)
                    ax1.annotate(f"DSM {_dsm_f}°F", xy=(_max_dt, _dsm_f - 0.3), xytext=(5, -12),
                                 textcoords="offset points", fontsize=7,
                                 color="#0066CC", fontweight="bold")
                except (ValueError, IndexError):
                    pass

    # Projected CLI drop times from sites.yaml
    if len(timestamps) > 0:
        try:
            from weather.sites import get_site_config
            _cfg = get_site_config(site)
            _obs_date = pd.Timestamp(timestamps[-1]).normalize()
            for _key, _color, _lbl_pfx in [
                ("cli_prelim_local", "red", "CLI prelim"),
                ("cli_final_local", "red", "CLI final"),
            ]:
                _time_str = _cfg.get(_key)
                if _time_str:
                    _h, _m = int(_time_str.split(":")[0]), int(_time_str.split(":")[1])
                    _day = _obs_date if _h >= 12 else _obs_date + pd.Timedelta(days=1)
                    _proj_dt = _day + pd.Timedelta(hours=_h, minutes=_m)
                    for ax in (ax1, ax2):
                        ax.axvline(_proj_dt, color=_color, linestyle=":",
                                   alpha=0.35, linewidth=1)
        except Exception:
            pass

    # Projected 6h METAR report times — vertical lines for all 4 daily windows
    # Always compute from UTC schedule so all 4 times show (not just reported ones)
    _metar_6h_times_plotted = False
    if len(timestamps) > 0:
        _today_date = pd.Timestamp(timestamps[-1]).normalize()
        _m6h_times = []  # list of (local_datetime, reported_bool)
        try:
            from weather.sites import FORECAST_STATIONS
            from zoneinfo import ZoneInfo as _ZI
            from datetime import datetime as _dt
            coords = FORECAST_STATIONS.get(site)
            if coords:
                _tz = _ZI(coords[2])
                # Which hours have already reported?
                _reported_hours = set()
                if "max_temp_6h_f" in df.columns:
                    _m6h_rows = df[df["max_temp_6h_f"].notna()]
                    if not _m6h_rows.empty:
                        _reported_hours = set(pd.to_datetime(
                            _m6h_rows["timestamp"].str[:19]).dt.hour)
                for _uh in [0, 6, 12, 18]:
                    _utc = _dt(_today_date.year, _today_date.month, _today_date.day,
                               _uh, 53, tzinfo=_ZI("UTC"))
                    _local = _utc.astimezone(_tz)
                    _local_naive = pd.Timestamp(_local).tz_localize(None)
                    _reported = _local.hour in _reported_hours
                    _m6h_times.append((_local_naive, _reported))
        except Exception:
            pass
        if _m6h_times:
            for _mdt, _reported in _m6h_times:
                _alpha = 0.7 if _reported else 0.4
                _ls = "--" if _reported else ":"
                for ax in (ax1, ax2):
                    ax.axvline(_mdt, color="magenta", linestyle=_ls,
                               alpha=_alpha, linewidth=1.2)
            _metar_6h_times_plotted = True
            _sun_markers.append((_m6h_times[0][0], "magenta", "6h METAR windows"))

            # Projected DSM drop times from sites.yaml
            try:
                from weather.sites import get_site_config as _gsc
                _dsm_times = _gsc(site).get("dsm_drop_local", [])
                for _dt_str in _dsm_times:
                    _dh, _dm = int(_dt_str.split(":")[0]), int(_dt_str.split(":")[1])
                    _dsm_proj = _today_date + pd.Timedelta(hours=_dh, minutes=_dm)
                    for ax in (ax1, ax2):
                        ax.axvline(_dsm_proj, color="#44AAFF", linestyle=":",
                                   alpha=0.3, linewidth=0.8)
            except Exception:
                pass

    # Legacy single METAR line (only if projected times not shown)
    if not _metar_6h_times_plotted and metar_6h_local_dt is not None:
        metar_dt = pd.Timestamp(metar_6h_local_dt).tz_localize(None)
        for ax in (ax1, ax2):
            ax.axvline(metar_dt, color="magenta", linestyle="--", alpha=0.5, linewidth=1)
        _sun_markers.append((metar_dt, "magenta", "6h METAR"))

    # All 6h METAR observations from the data (max and min)
    if "max_temp_6h_f" in df.columns:
        metar_6h_rows = df[df["max_temp_6h_f"].notna()]
        # Skip early 6h METAR (before 02:00) — it reports yesterday's max
        if not metar_6h_rows.empty:
            _m6h_ts = pd.to_datetime(metar_6h_rows["timestamp"].str[:19])
            metar_6h_rows = metar_6h_rows[_m6h_ts.dt.hour >= 2]
        if not metar_6h_rows.empty:
            m_ts = pd.to_datetime(metar_6h_rows["timestamp"].str[:19]).values
            m_vals = metar_6h_rows["max_temp_6h_f"].to_numpy(dtype=float)
            ax1.scatter(m_ts, m_vals, color="magenta", marker="D", s=50,
                        zorder=5, edgecolors="white", linewidths=0.5,
                        label=f"6h METAR max ({len(m_vals)})")
            for t, v in zip(m_ts, m_vals):
                ax1.annotate(f"{v:.1f}", xy=(t, v), xytext=(5, 8),
                             textcoords="offset points", fontsize=7,
                             color="magenta", fontweight="bold")

    # Predicted bracket — shaded band (clamp sentinels to visible range)
    if bracket and len(bracket) == 2:
        ylims = ax1.get_ylim()
        b_lo = bracket[0] if bracket[0] > -1000 else ylims[0]
        b_hi = bracket[1] if bracket[1] < 1000 else ylims[1]
        label_lo = f"{bracket[0]}" if bracket[0] > -1000 else "≤"
        label_hi = f"{bracket[1]}" if bracket[1] < 1000 else "≥"
        ax1.axhspan(b_lo, b_hi, color="tab:green", alpha=0.12,
                     label=f"Predicted bracket: [{label_lo}, {label_hi}]°F")
        if bracket[0] > -1000:
            ax1.axhline(y=b_lo, color="tab:green", linestyle="-", alpha=0.3, linewidth=0.5)
        if bracket[1] < 1000:
            ax1.axhline(y=b_hi, color="tab:green", linestyle="-", alpha=0.3, linewidth=0.5)

    ax1.legend(loc="upper left", fontsize=8)

    # Sun time labels (placed after data so ylim is set)
    for sun_dt, color, lbl in _sun_markers:
        ax1.annotate(lbl, xy=(sun_dt, ax1.get_ylim()[1]),
                     xytext=(2, -12), textcoords="offset points",
                     fontsize=7, color=color, rotation=90, va="top")

    # MA crossover plot (replaces raw rate)
    ma_cross = df.get("ma_cross", pd.Series(dtype=float))
    if ma_cross.notna().any():
        cross_f = ma_cross.to_numpy(dtype=float)
        ax2.fill_between(timestamps, 0, cross_f,
                         where=cross_f <= 0,
                         color="tab:blue", alpha=0.3, label="Short < Long (cooling)")
        ax2.fill_between(timestamps, 0, cross_f,
                         where=cross_f > 0,
                         color="tab:orange", alpha=0.3, label="Short > Long (warming)")
        ax2.plot(timestamps, cross_f, color="tab:blue", linewidth=1.0, alpha=0.7)
    ax2.axhline(y=0, color="black", linewidth=0.5)
    fast_cross = locked_rate if locked_rate is not None else -0.5
    wide_cross = likely_rate if likely_rate is not None else -0.2
    ax2.axhline(y=fast_cross, color="green", linestyle="--", alpha=0.5,
                label=f"FAST ({fast_cross}°F)")
    ax2.axhline(y=wide_cross, color="gold", linestyle="--", alpha=0.5,
                label=f"WIDE ({wide_cross}°F)")
    ax2.set_ylabel("MA Cross (short − long °F)")
    ax2.set_xlabel("Time")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="lower left", fontsize=8)

    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%-I %p"))
    ax2.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    fig.autofmt_xdate(rotation=45)

    plt.tight_layout()

    # Bracket model probabilities table at bottom of figure
    if has_bracket_info:
        # Extract offset detail from first result (same for all brackets)
        od = bracket_probs[0].get("offset_detail") if bracket_probs else None
        # Current auto max °F (whole-degree C readings only, excludes METAR)
        tc = pd.to_numeric(df.get("temperature_c"), errors="coerce")
        auto_mask = (tc % 1 == 0) & tc.notna()
        if auto_mask.any():
            auto_max_c = int(round(tc[auto_mask].max()))
            cur_max_f = round(auto_max_c * 9.0 / 5.0 + 32.0)
        else:
            cur_max_f = int(round(float(df["temperature_f"].max())))
        # Rounding context: possible °F values and naive_is_high
        _possible_f = c_to_possible_f(auto_max_c) if auto_mask.any() else []
        _naive_is_high = (len(_possible_f) == 2 and cur_max_f == _possible_f[-1])
        _n_poss = len(_possible_f)
        if _n_poss == 1:
            _rounding_note = f"exact (only {_possible_f[0]}°F maps to {auto_max_c}°C)"
        elif _naive_is_high:
            _rounding_note = (f"rounds to {_possible_f[1]}°F (high) — "
                              f"likely settles down to {_possible_f[0]}°F")
        else:
            _rounding_note = (f"rounds to {_possible_f[0]}°F (low) — "
                              f"may settle up to {_possible_f[1]}°F")
        lines = [f"Bracket Model — Auto max: {cur_max_f}°F ({auto_max_c if auto_mask.any() else '?'}°C)  "
                 f"| {_rounding_note}"]
        if od:
            s1 = od["stage1"]
            s2 = od["stage2"]
            ov = od["override"]
            reasons = od.get("override_reasons", {})
            lines.append(f"  {'Offset':<8s} {'Stage1':>8s} {'Stage2':>8s} {'Final':>8s}  {'Override'}")
            lines.append(f"  {'─'*8} {'─'*8} {'─'*8} {'─'*8}  {'─'*30}")
            for k, label in [(-1, "P(−1)"), (0, "P(0)"), (1, "P(+1)")]:
                r = "; ".join(reasons.get(k, []))
                lines.append(f"  {label:<8s} {s1[k]:>8.1%} {s2[k]:>8.1%} {ov[k]:>8.1%}  {r}")
        lines.append("")
        lines.append(f"  {'Bracket':<12} {'S1':>6} {'S2':>6} {'Final':>6}")
        lines.append(f"  {'─'*12} {'─'*6} {'─'*6} {'─'*6}")
        for bp in bracket_probs:
            blo, bhi = bp["bracket"]
            s1 = bp.get("stage1_prob", 0.0)
            s2 = bp.get("stage2_prob", 0.0)
            final = bp["prob"]
            marker = " ◄" if bracket and [blo, bhi] == bracket else ""
            lo_s = f"≤" if blo <= -1000 else str(blo)
            hi_s = f"≥" if bhi >= 1000 else str(bhi)
            lines.append(f"  [{lo_s},{hi_s}]°F   {s1:>5.0%}  {s2:>5.0%}  {final:>5.0%}{marker}")
        fig.subplots_adjust(bottom=0.28)
        fig.text(0.02, 0.005, "\n".join(lines), fontsize=7.5, fontfamily="monospace",
                 va="bottom", ha="left",
                 bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.8))
    elif bracket_error:
        fig.subplots_adjust(bottom=0.15)
        fig.text(0.02, 0.005, f"Bracket Model: {bracket_error}", fontsize=8,
                 fontfamily="monospace", va="bottom", ha="left",
                 bbox=dict(boxstyle="round,pad=0.4", facecolor="mistyrose", alpha=0.8))

    # Peak model prediction — bottom right
    if peak_result is not None:
        prob = peak_result["probability"]
        prob_2f = peak_result.get("probability_2f", 0.0)
        naive_f = peak_result["cur_naive_f"]
        verdict = "YES" if peak_result["prediction"] else "NO"
        verdict_2f = "YES" if peak_result.get("prediction_2f") else "NO"
        color = "limegreen" if prob < 0.3 else ("orange" if prob < 0.7 else "tomato")
        peak_text = (f"Peak Model — Current: {naive_f}°F\n"
                     f"  P(>1°F): {prob:.0%}  → {verdict}\n"
                     f"  P(>2°F): {prob_2f:.0%}  → {verdict_2f}")
        fig.text(0.98, 0.005, peak_text, fontsize=8, fontfamily="monospace",
                 va="bottom", ha="right",
                 bbox=dict(boxstyle="round,pad=0.4", facecolor=color, alpha=0.25))

    plt.savefig(output, dpi=150)
    plt.close()
    log.info(f"Saved momentum chart to {output}")


def main():
    import argparse

    from weather.forecast import ForecastIngestion
    from weather.observations import SynopticIngestion

    parser = argparse.ArgumentParser(description="Temperature momentum analysis")
    parser.add_argument("site", nargs="?", default="KLAX",
                        help="ICAO station identifier (default: KLAX)")
    parser.add_argument("--hours", type=int, default=24,
                        help="Hours of observation data to fetch (default: 24)")
    parser.add_argument("--window", type=int, default=60,
                        help="Rate window in minutes (default: 60, for compat)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output chart filename (default: charts/weather/momentum_{SITE}.png)")
    parser.add_argument("--csv", type=str, default=None,
                        help="Also dump observations to CSV")
    args = parser.parse_args()
    if args.output is None:
        from paths import project_path
        import os
        args.output = project_path("charts", "weather", f"momentum_{args.site}.png")
        os.makedirs(os.path.dirname(args.output), exist_ok=True)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)

    # Fetch observations
    print(f"=== {args.site} — MA Crossover (short={MA_SHORT_MIN}m, long={MA_LONG_MIN}m, last {args.hours}h) ===")
    s = SynopticIngestion(site=args.site)
    df = s.fetch_live_weather(hours=args.hours)
    df = compute_momentum(df, window_minutes=args.window)

    print(f"Total obs: {len(df)}")

    latest_cross = df["ma_cross"].dropna().iloc[-1] if "ma_cross" in df.columns and df["ma_cross"].notna().any() else None
    latest_rate = df["rate_f_per_hr"].dropna().iloc[-1] if df["rate_f_per_hr"].notna().any() else None
    if latest_cross is not None:
        print(f"\nMA cross: {latest_cross:+.2f}°F (short − long)")
        print(f"Rate (smoothed): {latest_rate:+.2f}°F/hr" if latest_rate is not None else "Rate: N/A")
        if latest_cross <= -0.5:
            print("Signal: STRONG crossover (short well below long)")
        elif latest_cross <= -0.2:
            print("Signal: WEAK crossover (short slightly below long)")
        elif latest_cross < 0:
            print("Signal: MARGINAL (short barely below long)")
        else:
            print("Signal: NO crossover (short above long)")

    # Momentum extrapolation
    extrap = extrapolate_momentum(df)
    if not extrap.empty:
        print(f"\nMomentum +60m: {extrap['temperature_f'].iloc[-1]:.1f}°F"
              f" (range {extrap['temp_lo'].iloc[-1]:.1f}–{extrap['temp_hi'].iloc[-1]:.1f}°F)")

    # Fetch forecast for overlay
    forecast_df = None
    try:
        fi = ForecastIngestion(args.site)
        forecast_df = fi.fetch_forecast()
        print(f"Forecast peak: {forecast_df['temperature_f'].max():.0f}°F")
    except Exception as e:
        print(f"Could not fetch forecast: {e}")

    # Kalshi brackets + orderbook
    try:
        import os
        from dotenv import load_dotenv
        from paths import project_path
        from bot.app import KalshiClient, _discover_series_ticker

        load_dotenv(project_path(".env.demo"))
        api_key_id = os.getenv("KALSHI_API_KEY_ID")
        private_key_path = os.getenv("KALSHI_PRIVATE_KEY_PATH")
        kalshi_env = os.getenv("KALSHI_ENV", "demo")

        if api_key_id and private_key_path:
            from datetime import datetime
            from zoneinfo import ZoneInfo
            from weather.forecast import STATIONS as FORECAST_STATIONS
            station_tz = ZoneInfo(FORECAST_STATIONS[args.site][2]) if args.site in FORECAST_STATIONS else ZoneInfo("America/New_York")
            today_suffix = datetime.now(station_tz).strftime("-%y%b%d-").upper()
            client = KalshiClient(api_key_id, private_key_path, env="prod")
            for market_type in ("high", "low"):
                series_ticker = _discover_series_ticker(client, args.site, market_type)
                if not series_ticker:
                    print(f"\nNo open {market_type} markets for {args.site}")
                    continue
                resp = client.get_markets(series_ticker)
                markets = [m for m in resp.get("markets", []) if today_suffix in m.get("ticker", "")]
                if not markets:
                    print(f"\nNo open {series_ticker} markets")
                    continue
                print(f"\n=== Kalshi {series_ticker} ({market_type}) ===")
                for m in markets:
                    ticker = m.get("ticker", "")
                    title = (m.get("title", "") + " " + m.get("subtitle", "")).strip()
                    try:
                        ob = client.get_orderbook(ticker)
                        # yes/no arrays are BIDS sorted ascending by price
                        yes_bids = ob.get("yes", [])
                        no_bids = ob.get("no", [])
                        best_yes_bid = yes_bids[-1][0] if yes_bids else "none"
                        best_yes_ask = (100 - no_bids[-1][0]) if no_bids else "none"
                        print(f"  {ticker}: {title} — YES bid/ask: {best_yes_bid}/{best_yes_ask}c")
                    except Exception as e:
                        print(f"  {ticker}: {title} — orderbook error: {e}")
        else:
            print("\nKalshi credentials not set — skipping bracket display")
    except Exception as e:
        print(f"\nCould not fetch Kalshi brackets: {e}")

    # Fetch sun times for chart overlay
    sun_times = None
    try:
        from weather.forecast import STATIONS as FORECAST_STATIONS
        from weather.observations import fetch_sun_times
        coords = FORECAST_STATIONS.get(args.site)
        if coords:
            from datetime import datetime as _dt
            from zoneinfo import ZoneInfo as _ZI
            today_str = _dt.now(_ZI(coords[2])).strftime("%Y-%m-%d")
            sun_times = fetch_sun_times(coords[0], coords[1], today_str)
    except Exception as e:
        print(f"Could not fetch sun times: {e}")

    # 6h METAR max
    metar_6h_f = None
    if "max_temp_6h_f" in df.columns:
        m6h = pd.to_numeric(df["max_temp_6h_f"], errors="coerce").dropna()
        if not m6h.empty:
            metar_6h_f = float(m6h.max())

    # Bracket model
    bracket = None
    bracket_probs = None
    bracket_error = None
    try:
        from weather.bracket_model import load_model as _load_bm, get_probability as _get_prob
        from weather.backtest_rounding import extract_regression_features as _extract_rf
        _sn = sun_times.get("solar_noon") if sun_times else None
        _feats = _extract_rf(df, solar_noon_hour=_sn)
        if _feats is not None:
            _bmodel = _load_bm()
            _max_c = _feats.get("max_c")
            if _max_c is not None:
                _nf = round(float(_max_c) * 9.0 / 5.0 + 32.0)
                _cl = _nf if _nf % 2 == 0 else _nf - 1
                _btuples = [(_lo, _lo + 1) for _lo in [_cl - 4, _cl - 2, _cl, _cl + 2, _cl + 4]]
                bracket_probs = _get_prob(_bmodel, _feats, _btuples, metar_6h_f=metar_6h_f)
                bracket_probs = [bp for bp in bracket_probs if bp["prob"] >= 0.005]
                bracket_probs.sort(key=lambda x: -x["prob"])
                if bracket_probs:
                    bracket = list(bracket_probs[0]["bracket"])
    except Exception as e:
        bracket_error = str(e)

    # Peak model
    peak_result = None
    try:
        from weather.peak_model import load_model as _load_pm, predict as _peak_predict
        _pb = _load_pm()
        _fh = forecast_df["temperature_f"].max() if forecast_df is not None and not forecast_df.empty else 70.0
        _sn_h = sun_times.get("solar_noon", 12.0) if sun_times else 12.0
        peak_result = _peak_predict(_pb, df, forecast_high_f=float(_fh), solar_noon_hour=_sn_h)
    except Exception as e:
        print(f"Peak model: {e}")

    # CLI reports
    cli_reports = None
    try:
        from weather.observations import fetch_all_cli_today
        cli_reports = fetch_all_cli_today(args.site)
        if cli_reports:
            print(f"CLI reports: {len(cli_reports)}")
        else:
            print("CLI: no reports found")
    except Exception as e:
        print(f"CLI fetch error: {e}")

    # DSM reports
    dsm_reports = None
    try:
        from weather.observations import fetch_dsm_today
        dsm_reports = fetch_dsm_today(args.site)
        if dsm_reports:
            print(f"DSM reports: {len(dsm_reports)}")
    except Exception:
        pass

    # Plot
    try:
        from bot.app import MOMENTUM_PARAMS_FAST, MOMENTUM_PARAMS_WIDE
        lr = MOMENTUM_PARAMS_FAST[0]
        mt = MOMENTUM_PARAMS_FAST[1]
        ll = MOMENTUM_PARAMS_WIDE[0]
    except ImportError:
        lr, mt, ll = -0.5, 2.0, -0.2
    plot_momentum(df, args.site, output=args.output, forecast_df=forecast_df,
                  locked_rate=lr, likely_rate=ll, margin_threshold=mt,
                  sun_times=sun_times, metar_6h_f=metar_6h_f,
                  bracket=bracket, bracket_probs=bracket_probs,
                  bracket_error=bracket_error, peak_result=peak_result,
                  cli_reports=cli_reports, dsm_reports=dsm_reports)

    # Optional CSV
    if args.csv:
        df.to_csv(args.csv, index=False)
        print(f"Wrote {len(df)} rows to {args.csv}")


if __name__ == "__main__":
    main()
