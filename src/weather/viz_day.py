#!/usr/bin/env python3
"""Render a momentum chart for a specific historical day and site.

Usage:
    python src/weather/viz_day.py KLAX 2026-03-10
    python src/weather/viz_day.py KDEN 2026-02-15 --output charts/kden_feb15.png
    python src/weather/viz_day.py KLAX 2026-03-10 --no-bracket --no-peak
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo

import pandas as pd

from paths import project_path
from weather.backtest import load_site_history
from weather.prediction import compute_momentum, plot_momentum


def render_day(
    site: str,
    date_str: str,
    output: Optional[str] = None,
    show_bracket: bool = True,
    show_peak: bool = True,
    cutoff_time: Optional[str] = None,
) -> str:
    """Load historical data for one day and render a momentum chart.

    Args:
        cutoff_time: If set (e.g. "14:30"), mask out all data after this
            local time.  Simulates what the model would see at that point
            in the day.

    Returns the output file path.
    """
    # Load full history and filter to the requested date
    df = load_site_history(site)
    if df.empty:
        raise FileNotFoundError(f"No history data for {site}")

    day_df = df[df["date"].astype(str) == date_str].copy()
    if day_df.empty:
        available = sorted(df["date"].astype(str).unique())
        raise ValueError(
            f"No data for {site} on {date_str}. "
            f"Available range: {available[0]} to {available[-1]}"
        )

    day_df = day_df.sort_values("ts").reset_index(drop=True)

    # Apply time cutoff — mask out all observations after the given local time
    if cutoff_time:
        cutoff_h, cutoff_m = (int(x) for x in cutoff_time.split(":"))
        cutoff_total_min = cutoff_h * 60 + cutoff_m
        ts_min = day_df["ts"].dt.hour * 60 + day_df["ts"].dt.minute
        day_df = day_df[ts_min <= cutoff_total_min].reset_index(drop=True)
        print(f"  {site} {date_str} (cutoff {cutoff_time}): {len(day_df)} observations")
    else:
        print(f"  {site} {date_str}: {len(day_df)} observations")

    # Compute momentum (adds ma_short, ma_long, ma_cross, rate columns)
    mom_df = compute_momentum(day_df)

    # Historical forecast (hourly)
    forecast_df = pd.DataFrame()
    try:
        fcst_csv = project_path("data", "weather", "forecast_hourly.csv")
        import os
        if os.path.isfile(fcst_csv):
            _fcst_all = pd.read_csv(fcst_csv)
            _fcst_day = _fcst_all[(_fcst_all["site"] == site) &
                                   (_fcst_all["date"] == date_str)]
            if not _fcst_day.empty:
                forecast_df = pd.DataFrame({
                    "timestamp": _fcst_day["hour"].apply(
                        lambda h: f"{date_str}T{int(h):02d}:00:00"),
                    "temperature_f": _fcst_day["temperature_f"].values,
                })
    except Exception:
        pass

    # Sun times
    sun_times = None
    try:
        from weather.observations import fetch_sun_times
        from weather.sites import FORECAST_STATIONS
        coords = FORECAST_STATIONS.get(site)
        if coords:
            sun_times = fetch_sun_times(coords[0], coords[1], date_str)
    except Exception:
        pass

    # METAR 6h value from observations
    metar_6h_f = None
    if "max_temp_6h_f" in mom_df.columns:
        m6h = pd.to_numeric(mom_df["max_temp_6h_f"], errors="coerce").dropna()
        if not m6h.empty:
            metar_6h_f = float(m6h.max())

    # (6h METAR times are now computed automatically in plot_momentum from UTC schedule)

    # Bracket model
    bracket = None
    bracket_probs = None
    bracket_error = None
    if show_bracket:
        try:
            from weather.bracket_model import load_model as load_bracket_model, get_probability
            from weather.backtest_rounding import extract_regression_features
            from weather.observations import fetch_sun_times as _fst
            from weather.sites import FORECAST_STATIONS as _FS

            solar_noon_hour = None
            if sun_times:
                solar_noon_hour = sun_times.get("solar_noon")

            feats = extract_regression_features(day_df, solar_noon_hour=solar_noon_hour)
            if feats is not None:
                bmodel = load_bracket_model()
                max_c = feats.get("max_c")
                if max_c is not None:
                    naive_f = round(float(max_c) * 9.0 / 5.0 + 32.0)
                    # Synthesize brackets around naive_f
                    if naive_f % 2 == 0:
                        center_lo = naive_f
                    else:
                        center_lo = naive_f - 1
                    bracket_tuples = []
                    for lo in [center_lo - 4, center_lo - 2, center_lo, center_lo + 2, center_lo + 4]:
                        bracket_tuples.append((lo, lo + 1))

                    bracket_probs = get_probability(
                        bmodel, feats, bracket_tuples, metar_6h_f=metar_6h_f
                    )
                    bracket_probs = [bp for bp in bracket_probs if bp["prob"] >= 0.005]
                    bracket_probs.sort(key=lambda x: -x["prob"])
                    if bracket_probs:
                        top = bracket_probs[0]
                        bracket = list(top["bracket"])
        except Exception as e:
            bracket_error = str(e)

    # Peak model
    peak_result = None
    if show_peak:
        try:
            from weather.peak_model import load_model as load_peak_model, predict as peak_predict
            peak_bundle = load_peak_model()
            solar_noon_hour = sun_times.get("solar_noon", 12.0) if sun_times else 12.0
            fcst_high = float(mom_df["temperature_f"].max())
            peak_result = peak_predict(
                peak_bundle, mom_df,
                forecast_high_f=fcst_high,
                solar_noon_hour=solar_noon_hour,
            )
        except Exception:
            pass

    # True settlement max from 6h METARs (skip before 01:00 — previous evening)
    if "max_temp_6h_f" not in day_df.columns:
        raise ValueError("max_temp_6h_f column missing — re-fetch history data")
    _m6h = day_df[(day_df["max_temp_6h_f"].notna()) & (day_df["ts"].dt.hour >= 1)]
    if _m6h.empty:
        raise ValueError("No daytime 6h METAR readings in day — cannot determine settlement")
    true_max_f = round(float(pd.to_numeric(_m6h["max_temp_6h_f"], errors="coerce").max()))

    # Output path
    if output is None:
        output = str(project_path("charts", "weather", f"viz_{site}_{date_str}.png"))

    plot_momentum(
        mom_df, site, output=output,
        forecast_df=forecast_df if not forecast_df.empty else None,
        sun_times=sun_times,
        metar_6h_f=metar_6h_f,
        bracket=bracket,
        bracket_probs=bracket_probs,
        bracket_error=bracket_error,
        peak_result=peak_result,
        true_max_f=true_max_f,
    )
    print(f"  Saved to {output}")
    return output


def main():
    parser = argparse.ArgumentParser(
        description="Render momentum chart for a historical day")
    parser.add_argument("site", help="ICAO station code (e.g. KLAX)")
    parser.add_argument("date", help="Date as YYYY-MM-DD")
    parser.add_argument("--output", "-o", default=None,
                        help="Output PNG path (default: charts/weather/viz_SITE_DATE.png)")
    parser.add_argument("--time", "-t", default=None,
                        help="Cutoff time as HH:MM — mask data after this local time")
    parser.add_argument("--no-bracket", action="store_true",
                        help="Skip bracket model overlay")
    parser.add_argument("--no-peak", action="store_true",
                        help="Skip peak model overlay")
    args = parser.parse_args()

    render_day(
        args.site.upper(),
        args.date,
        output=args.output,
        show_bracket=not args.no_bracket,
        show_peak=not args.no_peak,
        cutoff_time=args.time,
    )


if __name__ == "__main__":
    main()
