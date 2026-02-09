#!/usr/bin/env python3
"""Temperature momentum analysis and plotting."""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

log = logging.getLogger(__name__)


import numpy as np

METAR_WEIGHT = 5  # METAR observations weighted 5x vs ASOS


def is_metar(df: pd.DataFrame) -> pd.Series:
    """Return boolean mask for METAR rows (0.1°C precision)."""
    if "temperature_c" not in df.columns:
        return pd.Series(False, index=df.index)
    temp_c = df["temperature_c"].astype(float)
    return (temp_c % 1).abs() > 0.01


def compute_momentum(
    df: pd.DataFrame, window_minutes: int = 30
) -> pd.DataFrame:
    """Weighted rate-of-change over a sliding window.

    All observations are used, but METAR readings (0.1°C precision) receive
    5x weight in the weighted least-squares slope fit.  This gives a smooth
    rate anchored by accurate METAR data.

    Adds columns: ``rate_f_per_hr``, ``rate_window_min``, ``is_metar``.
    """
    if df.empty or "temperature_f" not in df.columns:
        return df

    timestamps = pd.to_datetime(df["timestamp"])
    temps = df["temperature_f"].astype(float)
    metar_mask = is_metar(df)

    rates = []
    for i in range(len(df)):
        ts_i = timestamps.iloc[i]
        cutoff = ts_i - pd.Timedelta(minutes=window_minutes)
        mask = (timestamps >= cutoff) & (timestamps <= ts_i) & temps.notna()
        w_ts = timestamps[mask]
        w_temps = temps[mask]
        w_metar = metar_mask[mask]

        if len(w_ts) < 2:
            rates.append(None)
            continue

        # Time in hours relative to window start
        t0 = w_ts.iloc[0]
        x = np.array([(t - t0).total_seconds() / 3600.0 for t in w_ts])
        y = np.array(w_temps)
        w = np.where(w_metar, METAR_WEIGHT, 1.0)

        if x[-1] - x[0] < 0.05:
            rates.append(None)
            continue

        # Weighted least squares: slope = Σw(x-x̄)(y-ȳ) / Σw(x-x̄)²
        xm = np.average(x, weights=w)
        ym = np.average(y, weights=w)
        dx = x - xm
        slope = np.sum(w * dx * (y - ym)) / np.sum(w * dx ** 2)
        rates.append(round(float(slope), 2))

    df = df.copy()
    df["rate_f_per_hr"] = rates
    df["rate_window_min"] = window_minutes
    df["is_metar"] = metar_mask
    return df


def extrapolate_momentum(
    df: pd.DataFrame, minutes: int = 60, step: int = 5
) -> pd.DataFrame:
    """Project temperature forward using weighted momentum.

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

    # Std of rates over last 3 hours (METAR-sampled points for stability)
    metar_mask = df.get("is_metar", pd.Series(False, index=df.index))
    metar_rates = df.loc[metar_mask, "rate_f_per_hr"].dropna().astype(float).tail(6)
    if len(metar_rates) > 2:
        rate_std = float(metar_rates.std())
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
    output: str = "momentum.png",
    forecast_df: Optional[pd.DataFrame] = None,
):
    """Plot temperature + weighted rate of change and save to file.

    *df* should already have ``rate_f_per_hr`` and ``is_metar`` columns
    from ``compute_momentum``.  If *forecast_df* is provided, overlays
    the NWS forecast as a dashed line on the temperature panel.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    # Timestamps are already in local time — strip the tz offset and parse as naive
    # so matplotlib shows local times on the axis without pandas tz conversion issues
    timestamps = pd.to_datetime(df["timestamp"].str[:19])
    temps = df["temperature_f"]
    rates = df["rate_f_per_hr"]
    metar_mask = df.get("is_metar", pd.Series(False, index=df.index))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                                    gridspec_kw={"height_ratios": [2, 1]})

    # Temperature plot — observed
    ax1.plot(timestamps, temps, color="tab:red", linewidth=1.5, label="Observed (°F)")
    ax1.set_ylabel("Temperature (°F)")
    ax1.set_title(f"{site} — Temperature & Momentum")
    ax1.grid(True, alpha=0.3)

    # Forecast overlay
    if forecast_df is not None and not forecast_df.empty and "temperature_f" in forecast_df.columns:
        fc_ts = pd.to_datetime(forecast_df["timestamp"].str[:19])
        fc_temps = forecast_df["temperature_f"]
        ax1.plot(fc_ts, fc_temps, color="tab:purple", linewidth=2, linestyle="--",
                 alpha=0.7, label="NWS Forecast (°F)")
        # Highlight forecast peak ±30 min
        if fc_temps.notna().any():
            fc_max_idx = fc_temps.idxmax()
            peak_time = fc_ts[fc_max_idx]
            peak_lo = peak_time - pd.Timedelta(minutes=30)
            peak_hi = peak_time + pd.Timedelta(minutes=30)
            ax1.axvspan(peak_lo, peak_hi, color="tab:purple", alpha=0.1,
                        label="Forecast peak window")
            ax1.annotate(f"Fcst peak: {fc_temps[fc_max_idx]:.0f}°F",
                         xy=(peak_time, fc_temps[fc_max_idx]),
                         xytext=(-60, 15), textcoords="offset points",
                         fontsize=9, color="tab:purple",
                         arrowprops=dict(arrowstyle="->", color="tab:purple", alpha=0.7))
            ax2.axvspan(peak_lo, peak_hi, color="tab:purple", alpha=0.1)

            # Highlight forecast low ±30 min
            fc_min_idx = fc_temps.idxmin()
            low_time = fc_ts[fc_min_idx]
            low_lo = low_time - pd.Timedelta(minutes=30)
            low_hi = low_time + pd.Timedelta(minutes=30)
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
        ex_ts = pd.to_datetime(extrap["timestamp"])
        ex_temps = extrap["temperature_f"]
        ex_lo = extrap["temp_lo"]
        ex_hi = extrap["temp_hi"]
        bridge_ts = pd.concat([pd.Series([timestamps.iloc[-1]]), ex_ts], ignore_index=True)
        bridge_temp = pd.concat([pd.Series([float(temps.iloc[-1])]), ex_temps], ignore_index=True)
        bridge_lo = pd.concat([pd.Series([float(temps.iloc[-1])]), ex_lo], ignore_index=True)
        bridge_hi = pd.concat([pd.Series([float(temps.iloc[-1])]), ex_hi], ignore_index=True)
        ax1.plot(bridge_ts, bridge_temp, color="tab:green", linewidth=2, linestyle=":",
                 label=f"Momentum +60m ({ex_temps.iloc[-1]:.1f}°F)")
        ax1.fill_between(bridge_ts, bridge_lo, bridge_hi,
                         color="tab:green", alpha=0.1)
        ax1.annotate(f"{ex_temps.iloc[-1]:.1f}°F",
                     xy=(ex_ts.iloc[-1], ex_temps.iloc[-1]),
                     xytext=(10, 5), textcoords="offset points",
                     fontsize=9, color="tab:green",
                     arrowprops=dict(arrowstyle="->", color="tab:green", alpha=0.7))

    # Mark observed max
    if temps.notna().any():
        max_idx = temps.idxmax()
        ax1.axhline(y=temps[max_idx], color="tab:red", linestyle="--", alpha=0.5,
                     label=f"Obs max: {temps[max_idx]:.1f}°F")
        ax1.annotate(f"{temps[max_idx]:.1f}°F",
                     xy=(timestamps[max_idx], temps[max_idx]),
                     xytext=(10, 10), textcoords="offset points",
                     fontsize=9, color="tab:red",
                     arrowprops=dict(arrowstyle="->", color="tab:red", alpha=0.7))

    # Mark observed min
    if temps.notna().any():
        min_idx = temps.idxmin()
        ax1.axhline(y=temps[min_idx], color="tab:blue", linestyle="--", alpha=0.5,
                     label=f"Obs min: {temps[min_idx]:.1f}°F")
        ax1.annotate(f"{temps[min_idx]:.1f}°F",
                     xy=(timestamps[min_idx], temps[min_idx]),
                     xytext=(10, -15), textcoords="offset points",
                     fontsize=9, color="tab:blue",
                     arrowprops=dict(arrowstyle="->", color="tab:blue", alpha=0.7))

    # METAR observation dots on temperature line
    if metar_mask.any():
        ax1.scatter(timestamps[metar_mask], temps[metar_mask],
                    color="tab:red", s=25, zorder=5, label="METAR (0.1°F)")

    ax1.legend(loc="upper left")

    # Weighted rate of change plot — continuous line with METAR dots highlighted
    ax2.fill_between(timestamps, 0, rates.astype(float),
                     where=rates.astype(float) <= 0,
                     color="tab:blue", alpha=0.3, label="Cooling")
    ax2.fill_between(timestamps, 0, rates.astype(float),
                     where=rates.astype(float) > 0,
                     color="tab:orange", alpha=0.3, label="Warming")
    ax2.plot(timestamps, rates, color="tab:blue", linewidth=1.0, alpha=0.7)
    # Highlight METAR-sampled rates as dots
    if metar_mask.any():
        m_rates_valid = rates[metar_mask].astype(float)
        ax2.scatter(timestamps[metar_mask], m_rates_valid,
                    color="tab:blue", s=30, zorder=5, label="METAR rate")
    ax2.axhline(y=0, color="black", linewidth=0.5)
    ax2.axhline(y=-2.0, color="green", linestyle="--", alpha=0.5, label="LOCKED (-2°F/hr)")
    ax2.axhline(y=-1.0, color="gold", linestyle="--", alpha=0.5, label="LIKELY (-1°F/hr)")
    ax2.set_ylabel("Rate (°F/hr)")
    ax2.set_xlabel("Time")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="lower left", fontsize=8)

    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%-I %p"))
    ax2.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    fig.autofmt_xdate(rotation=45)

    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()
    print(f"Saved momentum chart to {output}")


if __name__ == "__main__":
    import argparse

    from forecast import ForecastIngestion
    from weather import SynopticIngestion

    parser = argparse.ArgumentParser(description="Temperature momentum analysis")
    parser.add_argument("site", nargs="?", default="KLAX",
                        help="ICAO station identifier (default: KLAX)")
    parser.add_argument("--hours", type=int, default=24,
                        help="Hours of observation data to fetch (default: 24)")
    parser.add_argument("--window", type=int, default=30,
                        help="Momentum window in minutes (default: 30)")
    parser.add_argument("--output", type=str, default="momentum.png",
                        help="Output chart filename (default: momentum.png)")
    parser.add_argument("--csv", type=str, default=None,
                        help="Also dump observations to CSV")
    args = parser.parse_args()

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)

    # Fetch observations
    print(f"=== {args.site} — Weighted Momentum ({args.window}min window, last {args.hours}h) ===")
    s = SynopticIngestion(site=args.site)
    df = s.fetch_live_weather(hours=args.hours)
    df = compute_momentum(df, window_minutes=args.window)

    # Print METAR summary
    metar_rows = df[df["is_metar"] == True]
    print(f"Total obs: {len(df)}, METAR: {len(metar_rows)} (weighted {METAR_WEIGHT}x)")
    if not metar_rows.empty:
        cols = ["timestamp", "temperature_f", "rate_f_per_hr"]
        print(metar_rows[cols].to_string(index=False))

    latest_rate = df["rate_f_per_hr"].dropna().iloc[-1] if df["rate_f_per_hr"].notna().any() else None
    if latest_rate is not None:
        print(f"\nWeighted rate: {latest_rate:+.2f}°F/hr")
        if latest_rate <= -2.0:
            print("Status: LOCKED (rapid cooling)")
        elif latest_rate <= -1.0:
            print("Status: LIKELY (steady decline)")
        elif latest_rate <= -0.5:
            print("Status: POSSIBLE (slow decline)")
        else:
            print("Status: TOO_EARLY (flat or rising)")

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
        from bot import KalshiClient, SERIES_TICKERS

        load_dotenv(".env.demo")
        api_key_id = os.getenv("KALSHI_API_KEY_ID")
        private_key_path = os.getenv("KALSHI_PRIVATE_KEY_PATH")
        kalshi_env = os.getenv("KALSHI_ENV", "demo")

        if api_key_id and private_key_path:
            from datetime import datetime
            today_suffix = datetime.now().strftime("-%y%b%d-").upper()
            client = KalshiClient(api_key_id, private_key_path, env="prod")
            for market_type, series_ticker in SERIES_TICKERS.items():
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

    # Plot
    plot_momentum(df, args.site, output=args.output, forecast_df=forecast_df)

    # Optional CSV
    if args.csv:
        df.to_csv(args.csv, index=False)
        print(f"Wrote {len(df)} rows to {args.csv}")
