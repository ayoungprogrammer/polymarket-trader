#!/usr/bin/env python3
"""Backtest momentum strategy against historical data.

For each site and each day, pre-computes momentum at each window size,
then scans for trigger conditions across a grid of rate/margin params.

Key optimization: compute_momentum() is called once per (day, window_size),
not per param combo. Threshold checks are just array comparisons.
"""

from __future__ import annotations

import os
import sys
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional, List

import numpy as np
import pandas as pd

from paths import project_path
from weather.prediction import compute_momentum, predict_settlement_f

DATA_DIR = project_path("data")

ALL_SITES = [
    "KLAX", "KMIA", "KSFO", "KMDW", "KDEN", "KPHX",
    "KOKC", "KATL", "KDFW", "KSAT", "KHOU", "KMSP", "KDCA", "KAUS", "KBOS", "KPHL",
]

from weather.prediction import MA_SHORT_MIN, MA_LONG_MIN


def load_site_history(site: str) -> pd.DataFrame:
    csv_path = os.path.join(DATA_DIR, f"history_{site}.csv")
    if not os.path.isfile(csv_path):
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    df["ts"] = pd.to_datetime(df["timestamp"].str[:19])
    df["temperature_f"] = pd.to_numeric(df["temperature_f"], errors="coerce")
    df["temperature_c"] = pd.to_numeric(df.get("temperature_c"), errors="coerce")
    if "max_temp_24h_f" in df.columns:
        df["max_temp_24h_f"] = pd.to_numeric(df["max_temp_24h_f"], errors="coerce")
    for col in ["wind_speed_mph", "dewpoint_f", "relative_humidity_pct",
                "sea_level_pressure", "pressure_tendency"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["date"] = df["ts"].dt.date
    return df


def precompute_day_momentum(day_df: pd.DataFrame) -> dict:
    """Pre-compute MA crossover data for one day.

    Calls compute_momentum() once with fixed MA windows, then extracts
    arrays for fast scanning.  Also computes the actual peak hour for
    the forecast-peak cutoff logic (mirrors bot.py's ``peak + 2h`` gate).
    """
    temps = day_df["temperature_f"].values.astype(float)
    ts_arr = day_df["ts"].values
    hours = np.array([pd.Timestamp(t).hour + pd.Timestamp(t).minute / 60.0 for t in ts_arr])
    n = len(day_df)

    # Running max (observed max up to each point)
    running_max = np.maximum.accumulate(temps)

    # Actual end-of-day high and its hour (proxy for forecast peak)
    actual_high = float(np.nanmax(temps))
    peak_idx = int(np.nanargmax(temps))
    peak_hour = float(hours[peak_idx])

    # Compute MA crossover (fixed windows)
    mom_df = compute_momentum(day_df)
    ma_cross = mom_df["ma_cross"].values.astype(float)
    ma_short = mom_df["ma_short"].values.astype(float)
    ma_long = mom_df["ma_long"].values.astype(float)

    # Settlement prediction: running auto-obs max °C and METAR max °C
    # for applying the rounding model at trigger time.
    # METAR reports at :53 past the hour; everything else is 5-min auto-obs.
    temp_c = day_df["temperature_c"].values.astype(float) if "temperature_c" in day_df.columns else np.full(n, np.nan)
    obs_minutes = np.array([pd.Timestamp(t).minute for t in ts_arr])
    is_metar = ~np.isnan(temp_c) & (obs_minutes == 53)
    is_auto = ~np.isnan(temp_c) & (obs_minutes != 53)

    running_auto_max_c = np.full(n, np.nan)
    running_metar_max_c = np.full(n, np.nan)
    cur_auto_max = -np.inf
    cur_metar_max = -np.inf
    for i in range(n):
        if is_auto[i]:
            cur_auto_max = max(cur_auto_max, temp_c[i])
        if cur_auto_max > -np.inf:
            running_auto_max_c[i] = cur_auto_max
        if is_metar[i]:
            cur_metar_max = max(cur_metar_max, temp_c[i])
        if cur_metar_max > -np.inf:
            running_metar_max_c[i] = cur_metar_max

    # True daily high: prefer 24h ASOS max (settlement-grade), fall back to obs max
    true_high = actual_high
    if "max_temp_24h_f" in day_df.columns:
        val_24h = pd.to_numeric(day_df["max_temp_24h_f"], errors="coerce").max()
        if not np.isnan(val_24h):
            true_high = float(val_24h)

    actual_settlement_f = round(true_high)

    return {
        "temps": temps,
        "hours": hours,
        "running_max": running_max,
        "actual_high": true_high,
        "actual_settlement_f": actual_settlement_f,
        "peak_hour": peak_hour,
        "ma_cross": ma_cross,
        "ma_short": ma_short,
        "ma_long": ma_long,
        "running_auto_max_c": running_auto_max_c,
        "running_metar_max_c": running_metar_max_c,
        "n": n,
    }


def scan_trigger(
    precomp: dict,
    cross_threshold: float,
    margin: float,
    confirm_count: int = 1,
    min_hour: float = 10.0,
    max_hour: float = 20.0,
    peak_before_hours: float = 1.0,
    peak_after_hours: float = 2.0,
) -> Optional[dict]:
    """Scan pre-computed day data for first MA crossover trigger.

    *cross_threshold*: ma_cross must be <= this value.
    *margin*: ma_long peak - ma_short current must be >= this.
    *confirm_count*: consecutive readings ma_cross must stay below threshold.
    *min_hour*: absolute earliest hour (floor, default 10 AM).
    *max_hour*: absolute latest hour (ceiling, default 8 PM).
    *peak_before_hours*: start watching this many hours before the peak.
    *peak_after_hours*: stop watching this many hours after the peak.

    The effective trigger window is::

        [max(min_hour, peak - before), min(max_hour, peak + after)]

    Returns trigger info dict or None.
    """
    hours = precomp["hours"]
    actual_high = precomp["actual_high"]
    actual_settlement_f = precomp.get("actual_settlement_f", round(actual_high))
    peak_hour = precomp["peak_hour"]
    ma_cross = precomp["ma_cross"]
    ma_short = precomp["ma_short"]
    ma_long = precomp["ma_long"]
    running_max = precomp["running_max"]
    running_auto_max_c = precomp.get("running_auto_max_c")
    running_metar_max_c = precomp.get("running_metar_max_c")
    n = precomp["n"]

    # Trigger window based on peak time
    start_hour = max(min_hour, peak_hour - peak_before_hours)
    cutoff_hour = min(max_hour, peak_hour + peak_after_hours)

    consecutive = 0
    ma_long_peak = np.nan

    for i in range(n):
        if not np.isnan(ma_long[i]):
            if np.isnan(ma_long_peak) or ma_long[i] > ma_long_peak:
                ma_long_peak = ma_long[i]

        if hours[i] < start_hour or hours[i] > cutoff_hour:
            consecutive = 0
            continue

        if np.isnan(ma_cross[i]) or np.isnan(ma_short[i]) or np.isnan(ma_long_peak):
            consecutive = 0
            continue

        if ma_cross[i] <= cross_threshold:
            consecutive += 1
        else:
            consecutive = 0
            continue

        m = ma_long_peak - ma_short[i]
        if m < margin:
            continue

        if consecutive < confirm_count:
            continue

        obs_max = running_max[i]
        error = obs_max - actual_high
        if np.isnan(error):
            continue

        # Settlement prediction via rounding model
        settlement_f = round(float(obs_max))  # fallback: naive rounding
        settlement_prob = 0.0
        if running_auto_max_c is not None and not np.isnan(running_auto_max_c[i]):
            auto_max_c = int(running_auto_max_c[i])
            metar_max_c = (float(running_metar_max_c[i])
                           if running_metar_max_c is not None
                           and not np.isnan(running_metar_max_c[i])
                           else None)
            pred = predict_settlement_f(auto_max_c, metar_max_c=metar_max_c)
            settlement_f = pred.center_f
            settlement_prob = max(pred.probabilities.values())

        settlement_error = settlement_f - actual_settlement_f

        return {
            "trigger_hour": round(float(hours[i]), 2),
            "observed_max": round(float(obs_max), 1),
            "settlement_f": settlement_f,
            "settlement_prob": round(settlement_prob, 2),
            "ma_cross": round(float(ma_cross[i]), 2),
            "margin": round(float(m), 1),
            "error_f": round(float(error), 1),
            "settlement_error_f": settlement_error,
            "within_1f": abs(error) <= 1.0,
            "within_2f": abs(error) <= 2.0,
            "settlement_within_1f": abs(settlement_error) <= 1,
            "settlement_exact": settlement_error == 0,
        }

    return None


def run_full_backtest():
    """Run backtest across all sites with default MA crossover params, then grid search."""

    # ---- Phase 1: Default strategy results ----
    default_cross = -0.5
    default_margin = 1.5

    print("=" * 70)
    print(f"  BACKTEST: MA Crossover (short={MA_SHORT_MIN}m, long={MA_LONG_MIN}m)")
    print(f"  Default: cross={default_cross}, margin={default_margin}")
    print("=" * 70)

    site_precomp = {}  # cache for grid search

    # Collect all day DataFrames for parallel precomputation
    _work_items = []
    _no_data_sites = set()
    for site in ALL_SITES:
        df = load_site_history(site)
        if df.empty:
            print(f"  {site}: no data")
            _no_data_sites.add(site)
            continue
        for date, day_df in df.groupby("date"):
            if len(day_df) < 200:
                continue
            day_df = day_df.sort_values("ts").reset_index(drop=True)
            _work_items.append((site, str(date), day_df))

    # Parallel momentum precomputation (CPU-bound, uses multiprocessing)
    n_workers = min(os.cpu_count() or 4, len(_work_items))
    print(f"  Pre-computing momentum for {len(_work_items)} site-days "
          f"({n_workers} workers)...")
    sys.stdout.flush()
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(precompute_day_momentum, w[2]): (w[0], w[1])
            for w in _work_items
        }
        for future in as_completed(futures):
            site, date_str = futures[future]
            site_precomp.setdefault(site, []).append(
                (date_str, future.result()))
    del _work_items

    for site in site_precomp:
        site_precomp[site].sort(key=lambda x: x[0])

    print()
    for site in ALL_SITES:
        if site in _no_data_sites:
            continue
        days = site_precomp.get(site, [])
        if not days:
            print(f"  {site}: no valid days")
            continue

        # Scan with defaults
        triggered = 0
        wins_1f = 0
        wins_2f = 0
        settle_exact = 0
        settle_w1 = 0
        errors = []
        rebounds = 0
        trigger_hours = []

        for date_str, precomp in days:
            result = scan_trigger(precomp, default_cross, default_margin)
            if result:
                triggered += 1
                errors.append(result["error_f"])
                if result["within_1f"]:
                    wins_1f += 1
                if result["within_2f"]:
                    wins_2f += 1
                if result.get("settlement_exact"):
                    settle_exact += 1
                if result.get("settlement_within_1f"):
                    settle_w1 += 1
                if result["error_f"] < -1.0:
                    rebounds += 1
                trigger_hours.append(result["trigger_hour"])

        total = len(days)
        trig_pct = triggered / total * 100 if total > 0 else 0
        win1_pct = wins_1f / triggered * 100 if triggered > 0 else 0
        win2_pct = wins_2f / triggered * 100 if triggered > 0 else 0
        reb_pct = rebounds / triggered * 100 if triggered > 0 else 0
        settle_exact_pct = settle_exact / triggered * 100 if triggered > 0 else 0
        settle_w1_pct = settle_w1 / triggered * 100 if triggered > 0 else 0
        mean_err = np.mean(errors) if errors else 0
        std_err = np.std(errors) if errors else 0
        avg_hr = np.mean(trigger_hours) if trigger_hours else 0

        print(f"  {site}: {total:3d} days | "
              f"trig {triggered:2d}/{total} ({trig_pct:4.1f}%) | "
              f"raw±1°F {win1_pct:5.1f}% | "
              f"settle_exact {settle_exact_pct:5.1f}% | settle±1 {settle_w1_pct:5.1f}% | "
              f"err {mean_err:+.1f}±{std_err:.1f}°F | "
              f"avg_hr {avg_hr:.1f}")

        sys.stdout.flush()

    # ---- Phase 2: Grid search ----
    print(f"\n{'=' * 70}")
    print(f"  GRID SEARCH: Optimal params per site")
    print(f"  Grid: cross ∈ [-2,-1.5,-1,-0.5,-0.2], margin = 1.5, confirm ∈ [1,3]")
    print(f"{'=' * 70}")

    cross_vals = [-2.0, -1.5, -1.0, -0.5, -0.2]
    margin_vals = [1.5]
    confirm_vals = [1, 3]

    best_per_site = {}

    for site in ALL_SITES:
        days = site_precomp.get(site, [])
        if not days:
            continue

        total = len(days)
        best_score = -1
        best_info = None

        for cross in cross_vals:
            for margin in margin_vals:
                for confirm in confirm_vals:
                    triggered = 0
                    wins_1f = 0
                    wins_2f = 0
                    errors = []
                    rebounds = 0
                    trigger_hours = []

                    settle_exact = 0
                    settle_w1 = 0
                    for date_str, precomp in days:
                        result = scan_trigger(precomp, cross, margin, confirm)
                        if result:
                            triggered += 1
                            errors.append(result["error_f"])
                            if result["within_1f"]:
                                wins_1f += 1
                            if result["within_2f"]:
                                wins_2f += 1
                            if result.get("settlement_exact"):
                                settle_exact += 1
                            if result.get("settlement_within_1f"):
                                settle_w1 += 1
                            if result["error_f"] < -1.0:
                                rebounds += 1
                            trigger_hours.append(result["trigger_hour"])

                    trig_pct = triggered / total * 100 if total > 0 else 0
                    win1_pct = wins_1f / triggered * 100 if triggered > 0 else 0
                    reb_pct = rebounds / triggered * 100 if triggered > 0 else 0
                    settle_exact_pct = settle_exact / triggered * 100 if triggered > 0 else 0

                    if trig_pct < 20:
                        score = 0
                    else:
                        score = (win1_pct * 0.5 +
                                 trig_pct * 0.2 +
                                 (100 - reb_pct) * 0.3)

                    if score > best_score:
                        best_score = score
                        best_info = {
                            "cross": cross,
                            "margin": margin,
                            "confirm": confirm,
                            "triggered": triggered,
                            "total": total,
                            "trig_pct": round(trig_pct, 1),
                            "win1_pct": round(win1_pct, 1),
                            "win2_pct": round(wins_2f / triggered * 100 if triggered > 0 else 0, 1),
                            "settle_exact_pct": round(settle_exact_pct, 1),
                            "settle_w1_pct": round(settle_w1 / triggered * 100 if triggered > 0 else 0, 1),
                            "reb_pct": round(reb_pct, 1),
                            "mean_err": round(float(np.mean(errors)) if errors else 0, 2),
                            "std_err": round(float(np.std(errors)) if errors else 0, 2),
                            "avg_hr": round(float(np.mean(trigger_hours)) if trigger_hours else 0, 1),
                            "score": round(best_score, 1),
                        }

        best_per_site[site] = best_info
        b = best_info
        print(f"  {site}: cross={b['cross']:+.1f} margin={b['margin']:.1f} confirm={b['confirm']} | "
              f"trig {b['triggered']}/{b['total']} ({b['trig_pct']}%) | "
              f"raw±1°F {b['win1_pct']}% | "
              f"settle_exact {b['settle_exact_pct']}% | settle±1 {b['settle_w1_pct']}% | "
              f"err {b['mean_err']:+.1f}±{b['std_err']:.1f}°F | "
              f"score={b['score']}")
        sys.stdout.flush()

    # ---- Phase 3: Comparison table ----
    print(f"\n{'=' * 70}")
    print(f"  DEFAULT vs OPTIMAL COMPARISON")
    print(f"{'=' * 70}")
    print(f"  {'Site':<6} {'Def Trig%':>9} {'Def Win%':>9} {'Opt Trig%':>9} {'Opt Win%':>9} "
          f"{'Δ Win':>6} {'cross':>6} {'margin':>7} {'conf':>5}")
    print(f"  {'-'*6} {'-'*9} {'-'*9} {'-'*9} {'-'*9} {'-'*6} {'-'*6} {'-'*7} {'-'*5}")

    for site in ALL_SITES:
        days = site_precomp.get(site, [])
        if not days:
            continue
        total = len(days)

        # Default
        d_trig = 0
        d_wins = 0
        for _, precomp in days:
            r = scan_trigger(precomp, default_cross, default_margin)
            if r:
                d_trig += 1
                if r["within_1f"]:
                    d_wins += 1
        d_trig_pct = d_trig / total * 100 if total > 0 else 0
        d_win_pct = d_wins / d_trig * 100 if d_trig > 0 else 0

        # Optimal
        b = best_per_site.get(site)
        if not b:
            continue

        delta = b["win1_pct"] - d_win_pct
        print(f"  {site:<6} {d_trig_pct:>8.1f}% {d_win_pct:>8.1f}% "
              f"{b['trig_pct']:>8.1f}% {b['win1_pct']:>8.1f}% "
              f"{delta:>+5.1f} {b['cross']:>+5.1f} {b['margin']:>6.1f} {b['confirm']:>4}")

    # ---- Phase 4: Per-site day-by-day detail for optimal params ----
    print(f"\n{'=' * 70}")
    print(f"  DAY-BY-DAY RESULTS (optimal params, showing misses)")
    print(f"{'=' * 70}")

    for site in ALL_SITES:
        days = site_precomp.get(site, [])
        b = best_per_site.get(site)
        if not days or not b:
            continue

        misses = []
        for date_str, precomp in days:
            r = scan_trigger(precomp, b["cross"], b["margin"], b["confirm"])
            if r and not r.get("settlement_within_1f", r["within_1f"]):
                misses.append((date_str, r))

        if misses:
            print(f"\n  {site} — {len(misses)} settlement misses (error > ±1°F):")
            for date_str, r in misses:
                actual = precomp.get("actual_settlement_f", round(r['observed_max'] - r['error_f']))
                print(f"    {date_str}: raw {r['observed_max']:.1f}°F → "
                      f"settle {r.get('settlement_f', '?')}°F, "
                      f"actual {actual}°F, "
                      f"settle_err {r.get('settlement_error_f', '?'):+}°F, "
                      f"ma_cross {r['ma_cross']:+.2f}°F @ {r['trigger_hour']:.1f}h")

    # ---- Phase 5: Peak-relative window comparison ----
    print(f"\n{'=' * 70}")
    print(f"  PEAK-RELATIVE WINDOW TEST")
    print(f"  Compare fixed 10AM start vs peak-relative [peak-Bh, peak+Ah]")
    print(f"  Using per-site optimal cross/confirm from Phase 2, margin={default_margin}")
    print(f"{'=' * 70}")

    windows = [
        ("fixed 10AM",    None, None),     # baseline: min_hour=10, no peak-relative start
        ("peak-1h/+2h",   1.0,  2.0),
        ("peak-1h/+3h",   1.0,  3.0),
        ("peak-0.5h/+2h", 0.5,  2.0),
        ("peak-1.5h/+2h", 1.5,  2.0),
        ("peak-2h/+2h",   2.0,  2.0),
        ("peak-2h/+3h",   2.0,  3.0),
    ]

    print(f"\n  {'Site':<6} {'peak_hr':>7}", end="")
    for label, _, _ in windows:
        print(f"  {label:>15}", end="")
    print()
    print(f"  {'-'*6} {'-'*7}", end="")
    for _ in windows:
        print(f"  {'-'*15}", end="")
    print()
    print(f"  {'':6} {'':7}", end="")
    for _ in windows:
        print(f"  {'trig%/win±1°F':>15}", end="")
    print()

    for site in ALL_SITES:
        days = site_precomp.get(site, [])
        b = best_per_site.get(site)
        if not days or not b:
            continue

        total = len(days)
        cross = b["cross"]
        confirm = b["confirm"]

        # Average peak hour for display
        avg_peak = np.mean([p["peak_hour"] for _, p in days])

        line = f"  {site:<6} {avg_peak:>6.1f}h"

        for label, before, after in windows:
            trig = 0
            wins = 0
            for _, precomp in days:
                if before is None:
                    # Baseline: fixed 10 AM, peak+2h cutoff
                    r = scan_trigger(precomp, cross, default_margin, confirm,
                                     min_hour=10.0, peak_before_hours=24.0,
                                     peak_after_hours=2.0)
                else:
                    r = scan_trigger(precomp, cross, default_margin, confirm,
                                     peak_before_hours=before,
                                     peak_after_hours=after)
                if r:
                    trig += 1
                    if r["within_1f"]:
                        wins += 1
            trig_pct = trig / total * 100 if total > 0 else 0
            win_pct = wins / trig * 100 if trig > 0 else 0
            line += f"  {trig_pct:>5.1f}/{win_pct:>5.1f}%"
        print(line)

    sys.stdout.flush()

    # Save results
    output = {
        "default_params": {"cross": default_cross, "margin": default_margin},
        "optimal_per_site": best_per_site,
    }
    out_path = os.path.join(DATA_DIR, "backtest_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    run_full_backtest()
