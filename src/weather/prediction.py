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
    result = []
    for i in range(len(timestamps)):
        ts_i = timestamps.iloc[i]
        cutoff = ts_i - pd.Timedelta(minutes=window_min)
        mask = (timestamps >= cutoff) & (timestamps <= ts_i) & values.notna()
        w = values[mask]
        if len(w) < 2:
            result.append(None)
        else:
            result.append(round(float(w.mean()), 2))
    return result


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
    timestamps = pd.to_datetime(df["timestamp"])
    temps = df["temperature_f"].astype(float)

    ma_short_vals = _rolling_mean(timestamps, temps, ma_short_min)
    ma_long_vals = _rolling_mean(timestamps, temps, ma_long_min)

    df["ma_short"] = ma_short_vals
    df["ma_long"] = ma_long_vals
    df["ma_cross"] = [
        round(s - l, 2) if s is not None and l is not None else None
        for s, l in zip(ma_short_vals, ma_long_vals)
    ]

    rate_window = min(ma_short_min, 30)
    ma_short_series = pd.Series(ma_short_vals, dtype=float)
    rates = []
    for i in range(len(df)):
        ts_i = timestamps.iloc[i]
        cutoff = ts_i - pd.Timedelta(minutes=rate_window)
        mask = (timestamps >= cutoff) & (timestamps <= ts_i) & ma_short_series.notna()
        w_ts = timestamps[mask]
        w_vals = ma_short_series[mask]
        if len(w_ts) < 2:
            rates.append(None)
            continue
        t0 = w_ts.iloc[0]
        x = np.array([(t - t0).total_seconds() / 3600.0 for t in w_ts])
        y = np.array(w_vals)
        if x[-1] - x[0] < 0.05:
            rates.append(None)
            continue
        xm = x.mean()
        dx = x - xm
        denom = np.sum(dx ** 2)
        if denom == 0:
            rates.append(None)
            continue
        slope = np.sum(dx * (y - y.mean())) / denom
        rates.append(round(float(slope), 2))

    df["rate_f_per_hr"] = rates
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
):
    """Plot temperature + MA crossover and save to file.

    *bracket_probs*: list of dicts from bracket model, each with keys
    ``bracket`` (lo, hi), ``prob``, ``stage1_prob``, ``confidence``.

    Thread-safe: uses _plot_lock to serialize matplotlib calls.
    """
    with _plot_lock:
        _plot_momentum_impl(df, site, output, forecast_df,
                            locked_rate, likely_rate, margin_threshold,
                            sun_times, metar_6h_f, bracket, bracket_probs)


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
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    timestamps = pd.to_datetime(df["timestamp"].str[:19]).values
    temps = df["temperature_f"].to_numpy(dtype=float)
    has_bracket_info = bracket_probs and len(bracket_probs) > 0
    fig_height = 9.5 if has_bracket_info else 8
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
    ax1.set_title(f"{site} — Temperature & MA Crossover")
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
            ax1.axvspan(peak_lo, peak_hi, color="tab:purple", alpha=0.1,
                        label="Forecast peak window")
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
            ax1.axvspan(low_lo, low_hi, color="tab:blue", alpha=0.08,
                        label="Forecast low window")
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

    # METAR 6h max — most precise observation
    if metar_6h_f is not None:
        ax1.axhline(y=metar_6h_f, color="magenta", linestyle="-", alpha=0.6, linewidth=1.5,
                     label=f"METAR 6h max: {metar_6h_f:.1f}°F")

    # Predicted bracket — shaded band
    if bracket and len(bracket) == 2:
        ax1.axhspan(bracket[0], bracket[1], color="tab:green", alpha=0.12,
                     label=f"Predicted bracket: [{bracket[0]}, {bracket[1]}]°F")
        ax1.axhline(y=bracket[0], color="tab:green", linestyle="-", alpha=0.3, linewidth=0.5)
        ax1.axhline(y=bracket[1], color="tab:green", linestyle="-", alpha=0.3, linewidth=0.5)

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
        lines = ["Bracket Model Predictions:"]
        lines.append(f"  {'Bracket':<12} {'Stage1':>8} {'Final':>8}")
        lines.append(f"  {'─'*12} {'─'*8} {'─'*8}")
        for bp in bracket_probs:
            blo, bhi = bp["bracket"]
            s1 = bp.get("stage1_prob", 0.0)
            final = bp["prob"]
            marker = " ◄" if bracket and [blo, bhi] == bracket else ""
            lines.append(f"  [{blo}, {bhi}]°F   {s1:>7.0%}  {final:>7.0%}{marker}")
        fig.subplots_adjust(bottom=0.18)
        fig.text(0.02, 0.005, "\n".join(lines), fontsize=9, fontfamily="monospace",
                 va="bottom", ha="left",
                 bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.8))

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

    # Plot
    try:
        from bot.app import MOMENTUM_PARAMS_FAST, MOMENTUM_PARAMS_WIDE
        lr = MOMENTUM_PARAMS_FAST[0]   # cross threshold for FAST
        mt = MOMENTUM_PARAMS_FAST[1]   # margin for FAST
        ll = MOMENTUM_PARAMS_WIDE[0]   # cross threshold for WIDE
    except ImportError:
        lr, mt, ll = -0.5, 2.0, -0.2
    plot_momentum(df, args.site, output=args.output, forecast_df=forecast_df,
                  locked_rate=lr, likely_rate=ll, margin_threshold=mt,
                  sun_times=sun_times)

    # Optional CSV
    if args.csv:
        df.to_csv(args.csv, index=False)
        print(f"Wrote {len(df)} rows to {args.csv}")


if __name__ == "__main__":
    main()
