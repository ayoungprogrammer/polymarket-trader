#!/usr/bin/env python3
"""Backtest settlement prediction using ONLY whole-°C auto observations.

The 5-minute station conversion chain:
  true temp → 1-min avg → 5x avg → round to whole °F → round to whole °C
NWS then displays °C and a lossy back-conversion to °F.

To recover the original whole °F, we reverse-map: find all integer °F
values where round((F-32)*5/9) == observed °C.  Typically 2 values.

Settlement uses the max 1-MINUTE reading (not 5-min avg) from BOTH the
5-min and hourly stations, rounded to whole °F with no C conversion.
So settlement can exceed any 5-min reported value.
"""

from __future__ import annotations

import os
import sys


from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd

from weather.backtest import load_site_history
from weather.sites import ALL_SITES, ALL_SITES_WITH_TRAINING, TRAINING_SITES
from weather.prediction import (
    c_to_possible_f,
    RoundingPrediction,
    predict_settlement_f,
)

from paths import project_path

SOLAR_NOON_CSV = project_path("data", "weather", "solar_noon.csv")


# ---------------------------------------------------------------------------
# Per-day evaluation
# ---------------------------------------------------------------------------

def evaluate_day(day_df: pd.DataFrame) -> Optional[dict]:
    """Evaluate settlement prediction for one day.

    Filters to auto-obs only (whole °C), takes the max °C reading
    (what we'd know right before METAR drops), runs predict_settlement_f,
    and compares to actual max from all readings.
    """
    df = day_df.copy()
    df["temperature_c"] = pd.to_numeric(df["temperature_c"], errors="coerce")
    df["temperature_f"] = pd.to_numeric(df["temperature_f"], errors="coerce")

    # Identify METAR minute: the minute with the most fractional °C readings.
    # Varies by station (:51, :52, :53, :56 etc). Auto-obs are on 5-min marks.
    minutes = pd.to_datetime(df["timestamp"].str[:19]).dt.minute
    temp_c_notna = df["temperature_c"].notna()
    frac_mask = (df["temperature_c"] % 1 != 0) & temp_c_notna
    if frac_mask.any():
        metar_minute = int(minutes[frac_mask].mode().iloc[0])
    else:
        metar_minute = 53  # fallback
    # Auto-obs: whole °C only, exclude METAR minute.
    # Also excludes SPECI (special obs) at non-METAR minutes with fractional °C.
    auto_mask = (df["temperature_c"] % 1 == 0) & (minutes != metar_minute) & temp_c_notna
    # METAR/precision: METAR minute readings + any fractional °C (SPECI etc.)
    metar_mask = ((minutes == metar_minute) | (df["temperature_c"] % 1 != 0)) & temp_c_notna

    auto_df = df[auto_mask]
    metar_df = df[metar_mask]
    if len(auto_df) < 20:
        return None

    auto_c = auto_df["temperature_c"].values.astype(float)
    metar_c = metar_df["temperature_c"].values.astype(float)
    max_c = int(round(auto_c.max()))
    dwell_count = int(np.sum(auto_c >= max_c))
    n_auto = len(auto_c)
    dwell_ratio = dwell_count / n_auto

    # Transition ratio: fraction of readings near peak that are AT max_c
    # (vs one step below). High ratio = sustained peak = likely f_high.
    n_at_max = int(np.sum(auto_c == max_c))
    n_at_prev = int(np.sum(auto_c == max_c - 1))
    trans_denom = n_at_max + n_at_prev
    trans_ratio = n_at_max / trans_denom if trans_denom > 0 else 0

    possible_f = c_to_possible_f(max_c)
    naive_f = round(max_c * 9 / 5 + 32)

    # METAR hard rule: if any METAR T-group reading exceeds the rounding
    # boundary, settlement is guaranteed >= naive_f (offset never -1).
    metar_above = False
    if len(possible_f) == 2 and len(metar_c) > 0:
        f_low = possible_f[0]
        boundary_c = (f_low + 0.5 - 32) * 5.0 / 9.0
        if float(metar_c.max()) > boundary_c:
            metar_above = True

    # Dwell + transition prediction (no METAR)
    if len(possible_f) == 1:
        center = possible_f[0]
        low_f = center
        high_f = center + 1
    elif naive_f == possible_f[-1]:
        # naive=high: almost always correct
        center = possible_f[-1]
        low_f = possible_f[0]
        high_f = possible_f[-1] + 1
    else:
        # naive=low: use linear score of dwell ratio + transition ratio.
        # trans_ratio captures sustained peak (readings at max_c vs max_c-1).
        # dwell_ratio captures overall time at peak (readings at max_c / total).
        # Linear boundary allows one to compensate for the other.
        f_low, f_high = possible_f[0], possible_f[-1]
        peak_score = 1.1 * trans_ratio + 1.7 * dwell_ratio
        if peak_score > 0.70:
            center = f_high
        else:
            center = f_low
        # Hard rule: METAR above boundary → floor prediction at naive_f
        if metar_above and center < naive_f:
            center = naive_f
        low_f = f_low
        high_f = f_high + 1

    # True daily high from 6h METARs (settlement-grade, correctly day-scoped)
    # Skip any 6h METAR before 01:00 — it covers the previous evening window
    if "max_temp_6h_f" not in df.columns:
        raise ValueError("max_temp_6h_f column missing — re-fetch history data")
    _m6h = df[["ts", "max_temp_6h_f"]].copy()
    _m6h["max_temp_6h_f"] = pd.to_numeric(_m6h["max_temp_6h_f"], errors="coerce")
    _m6h = _m6h[(_m6h["max_temp_6h_f"].notna()) & (_m6h["ts"].dt.hour >= 1)]
    if _m6h.empty:
        return None  # no daytime 6h METAR readings this day
    actual_max_f = round(float(_m6h["max_temp_6h_f"].max()))

    return {
        "max_c_obs": max_c,
        "dwell_count": dwell_count,
        "dwell_ratio": dwell_ratio,
        "trans_ratio": trans_ratio,
        "metar_max_c": float(metar_c.max()) if len(metar_c) > 0 else None,
        "n_metar_obs": len(metar_c),
        "possible_f": possible_f,
        "naive_f": naive_f,
        "predicted_f": center,
        "pred_low_f": low_f,
        "pred_high_f": high_f,
        "probabilities": {},
        "actual_max_f": actual_max_f,
        "error": center - actual_max_f,
        "naive_error": naive_f - actual_max_f,
        "in_range": low_f <= actual_max_f <= high_f,
        "n_auto_obs": n_auto,
    }


# ---------------------------------------------------------------------------
# Backtest harness
# ---------------------------------------------------------------------------

def run_backtest(sites: Optional[List[str]] = None):
    """Backtest settlement prediction across sites."""
    if sites is None:
        sites = ALL_SITES

    print("=" * 78)
    print("  SETTLEMENT PREDICTION BACKTEST")
    print("  Reverse-map auto-obs °C → possible °F, predict settlement")
    print("=" * 78)

    all_results: Dict[str, List[Tuple[str, dict]]] = {}

    for site in sites:
        df = load_site_history(site)
        if df.empty:
            print(f"  {site}: no data")
            continue

        days: List[Tuple[str, dict]] = []
        for date, day_df in df.groupby("date"):
            if len(day_df) < 200:
                continue
            day_df = day_df.sort_values("ts").reset_index(drop=True)
            result = evaluate_day(day_df)
            if result is None:
                continue
            days.append((str(date), result))

        all_results[site] = days

        if not days:
            print(f"  {site}: no valid days")
            continue

        total = len(days)
        exact = sum(1 for _, r in days if r["error"] == 0)
        within_1 = sum(1 for _, r in days if abs(r["error"]) <= 1)
        in_range = sum(1 for _, r in days if r["in_range"])
        naive_exact = sum(1 for _, r in days if r["naive_error"] == 0)

        errors = [r["error"] for _, r in days]
        mean_err = np.mean(errors)
        std_err = np.std(errors)

        print(f"  {site}: {total:3d} days | "
              f"model {exact:3d}/{total} ({exact/total*100:5.1f}%) | "
              f"naive {naive_exact:3d}/{total} ({naive_exact/total*100:5.1f}%) | "
              f"±1°F {within_1/total*100:5.1f}% | "
              f"range {in_range/total*100:5.1f}% | "
              f"err {mean_err:+.2f}±{std_err:.2f}°F")
        sys.stdout.flush()

    # ---- Position analysis: does naive_f position affect accuracy? ----
    print(f"\n{'=' * 78}")
    print(f"  POSITION ANALYSIS: naive_f vs possible_f")
    print(f"  When naive = high of pair vs low of pair vs only value")
    print(f"{'=' * 78}")

    pos_stats: Dict[str, Dict[str, int]] = {
        "high": {"total": 0, "exact": 0, "actual_above": 0},
        "low":  {"total": 0, "exact": 0, "actual_above": 0},
        "only": {"total": 0, "exact": 0, "actual_above": 0},
    }

    for site in sites:
        for _, r in all_results.get(site, []):
            pf = r["possible_f"]
            naive = r["naive_f"]
            actual = r["actual_max_f"]

            if len(pf) == 1:
                pos = "only"
            elif naive == pf[-1]:
                pos = "high"
            else:
                pos = "low"

            pos_stats[pos]["total"] += 1
            if naive == actual:
                pos_stats[pos]["exact"] += 1
            if actual > naive:
                pos_stats[pos]["actual_above"] += 1

    for pos, st in pos_stats.items():
        t = st["total"]
        if t == 0:
            continue
        print(f"  naive={pos:>4}: {t:4d} days | "
              f"naive_exact {st['exact']/t*100:5.1f}% | "
              f"actual > naive {st['actual_above']/t*100:5.1f}%")

    # ---- METAR signal analysis (naive=low cases) ----
    print(f"\n{'=' * 78}")
    print(f"  METAR SIGNAL ANALYSIS (naive=low cases only)")
    print(f"  Does METAR T-group precision predict actual > naive?")
    print(f"{'=' * 78}")

    metar_above_total = 0
    metar_above_correct = 0  # predicted f_high, actual was f_high+
    metar_below_total = 0
    metar_below_correct = 0  # predicted f_low, actual was f_low
    no_metar_total = 0
    no_metar_correct = 0

    for site in sites:
        for _, r in all_results.get(site, []):
            pf = r["possible_f"]
            if len(pf) != 2 or r["naive_f"] != pf[0]:
                continue  # only naive=low cases
            f_low, f_high = pf[0], pf[-1]
            actual = r["actual_max_f"]
            metar_c = r.get("metar_max_c")

            if metar_c is not None:
                threshold = (f_low + 0.5 - 32) * 5 / 9
                if metar_c > threshold:
                    metar_above_total += 1
                    if actual >= f_high:
                        metar_above_correct += 1
                else:
                    metar_below_total += 1
                    if actual == f_low:
                        metar_below_correct += 1
            else:
                no_metar_total += 1
                if actual == r["naive_f"]:
                    no_metar_correct += 1

    if metar_above_total:
        print(f"  METAR > threshold (predict f_high): "
              f"{metar_above_total:3d} days, "
              f"{metar_above_correct}/{metar_above_total} correct "
              f"({metar_above_correct/metar_above_total*100:.1f}%)")
    if metar_below_total:
        print(f"  METAR ≤ threshold (predict f_low):  "
              f"{metar_below_total:3d} days, "
              f"{metar_below_correct}/{metar_below_total} correct "
              f"({metar_below_correct/metar_below_total*100:.1f}%)")
    if no_metar_total:
        print(f"  No METAR data    (predict f_low):  "
              f"{no_metar_total:3d} days, "
              f"{no_metar_correct}/{no_metar_total} correct "
              f"({no_metar_correct/no_metar_total*100:.1f}%)")
    naive_low_total = metar_above_total + metar_below_total + no_metar_total
    naive_low_correct = metar_above_correct + metar_below_correct + no_metar_correct
    if naive_low_total:
        print(f"  Combined naive=low accuracy: "
              f"{naive_low_correct}/{naive_low_total} "
              f"({naive_low_correct/naive_low_total*100:.1f}%) "
              f"[was 74.2% without METAR]")

    # ---- Where actual lands relative to possible_f ----
    print(f"\n{'=' * 78}")
    print(f"  SETTLEMENT vs POSSIBLE_F: where does actual land?")
    print(f"{'=' * 78}")

    landing: Dict[str, int] = {}
    for site in sites:
        for _, r in all_results.get(site, []):
            pf = r["possible_f"]
            actual = r["actual_max_f"]
            if actual < pf[0]:
                key = "below_low"
            elif actual == pf[0] and len(pf) > 1:
                key = "at_f_low"
            elif actual == pf[-1]:
                key = "at_f_high"
            elif actual == pf[-1] + 1:
                key = "f_high+1"
            elif actual > pf[-1] + 1:
                key = f"f_high+{actual - pf[-1]}"
            elif len(pf) == 1 and actual == pf[0]:
                key = "at_only"
            else:
                key = f"other({actual} vs {pf})"
            landing[key] = landing.get(key, 0) + 1

    total_all = sum(landing.values())
    for key in ["at_only", "at_f_low", "at_f_high", "f_high+1"]:
        ct = landing.get(key, 0)
        print(f"  {key:>12}: {ct:4d} ({ct/total_all*100:5.1f}%)")
    for key, ct in sorted(landing.items()):
        if key not in ("at_only", "at_f_low", "at_f_high", "f_high+1"):
            print(f"  {key:>12}: {ct:4d} ({ct/total_all*100:5.1f}%)")

    # ---- Error distribution ----
    print(f"\n{'=' * 78}")
    print(f"  ERROR DISTRIBUTION: model (predicted_f - actual_f)")
    print(f"{'=' * 78}")

    for site in sites:
        days = all_results.get(site, [])
        if not days:
            continue
        errors = [r["error"] for _, r in days]
        counts: Dict[int, int] = {}
        for e in errors:
            counts[e] = counts.get(e, 0) + 1
        dist = " | ".join(f"{k:+d}:{v}" for k, v in sorted(counts.items()))
        print(f"  {site}: {dist}")

    # ---- Misses ----
    print(f"\n{'=' * 78}")
    print(f"  DAY-BY-DAY MISSES (model error > ±1°F)")
    print(f"{'=' * 78}")

    for site in sites:
        days = all_results.get(site, [])
        if not days:
            continue
        misses = [(d, r) for d, r in days if abs(r["error"]) > 1]
        if misses:
            print(f"\n  {site} — {len(misses)} misses:")
            for date_str, r in misses:
                metar_str = (f"metar_max_c={r['metar_max_c']:.1f}"
                             if r.get('metar_max_c') is not None
                             else "no_metar")
                print(f"    {date_str}: max_c={r['max_c_obs']}°C "
                      f"possible_f={r['possible_f']} "
                      f"→ pred {r['predicted_f']}°F | "
                      f"actual {r['actual_max_f']}°F | err {r['error']:+d}°F | "
                      f"dwell={r['dwell_count']} {metar_str} "
                      f"obs={r['n_auto_obs']}")

    # ---- Summary ----
    print(f"\n{'=' * 78}")
    print(f"  OVERALL SUMMARY")
    print(f"{'=' * 78}")

    total_days = 0
    model_exact = 0
    model_w1 = 0
    model_range = 0
    naive_exact_total = 0
    all_errors: List[int] = []

    for site in sites:
        for _, r in all_results.get(site, []):
            total_days += 1
            if r["error"] == 0:
                model_exact += 1
            if abs(r["error"]) <= 1:
                model_w1 += 1
            if r["in_range"]:
                model_range += 1
            if r["naive_error"] == 0:
                naive_exact_total += 1
            all_errors.append(r["error"])

    if total_days > 0:
        print(f"  Total days: {total_days}")
        print(f"  Model exact:  {model_exact}/{total_days} ({model_exact/total_days*100:.1f}%)")
        print(f"  Model ±1°F:   {model_w1}/{total_days} ({model_w1/total_days*100:.1f}%)")
        print(f"  Model range:  {model_range}/{total_days} ({model_range/total_days*100:.1f}%)")
        print(f"  Naive exact:  {naive_exact_total}/{total_days} ({naive_exact_total/total_days*100:.1f}%)")
        print(f"  Mean error: {np.mean(all_errors):+.2f}°F  Std: {np.std(all_errors):.2f}°F")

    # Save results
    import json
    from paths import project_path
    out_path = project_path("data", "weather", "backtest_rounding_results.json")
    output = {
        "summary": {
            "total_days": total_days,
            "model_exact_pct": round(model_exact / total_days * 100, 1) if total_days else 0,
            "naive_exact_pct": round(naive_exact_total / total_days * 100, 1) if total_days else 0,
            "w1_pct": round(model_w1 / total_days * 100, 1) if total_days else 0,
            "range_pct": round(model_range / total_days * 100, 1) if total_days else 0,
            "mean_error": round(float(np.mean(all_errors)), 2) if all_errors else 0,
            "std_error": round(float(np.std(all_errors)), 2) if all_errors else 0,
        },
        "per_site": {
            site: {
                "days": len(days),
                "model_exact_pct": round(sum(1 for _, r in days if r["error"] == 0) / len(days) * 100, 1),
                "naive_exact_pct": round(sum(1 for _, r in days if r["naive_error"] == 0) / len(days) * 100, 1),
                "w1_pct": round(sum(1 for _, r in days if abs(r["error"]) <= 1) / len(days) * 100, 1),
                "range_pct": round(sum(1 for _, r in days if r["in_range"]) / len(days) * 100, 1),
                "mean_error": round(float(np.mean([r["error"] for _, r in days])), 2),
            }
            for site in sites
            if (days := all_results.get(site, []))
        },
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")


# ---------------------------------------------------------------------------
# ML feature extraction + model exploration
# ---------------------------------------------------------------------------

_MET_FEATURE_NAMES = [
    "wind_at_peak", "wind_pre_peak", "wind_calm_frac",
    "wind_gust_range", "wind_change_at_peak",
    "rh_at_peak", "rh_min_near_peak",
    "dewpoint_at_peak", "dewpoint_depression_at_peak", "dewpoint_change_near_peak",
    "heat_index_at_peak", "heat_index_excess",
    "pressure_at_peak", "pressure_change_near_peak", "pressure_tendency_at_peak",
    "cloud_cover_at_peak", "cloud_variability",
]


def _met_features(
    day_df: pd.DataFrame,
    auto_ts: np.ndarray,
    max_idxs_auto: np.ndarray,
) -> dict:
    """Compute meteorological features from wind/dewpoint/RH/cloud/pressure.

    Gracefully returns NaN defaults when columns are missing from the data.

    ``auto_ts``: numpy datetime64 timestamps of auto-obs readings.
    ``max_idxs_auto``: indices into ``auto_ts`` where auto temp == max_c.
    Uses timestamps to align with ``day_df`` rows (which includes all obs).
    """
    n = len(day_df)
    if n == 0 or len(max_idxs_auto) == 0:
        return {k: np.nan for k in _MET_FEATURE_NAMES}

    # Build timestamp index into day_df for alignment
    df_ts = pd.to_datetime(day_df["timestamp"].str[:19]).values

    # Find day_df rows closest to auto-obs peak timestamps
    peak_timestamps = auto_ts[max_idxs_auto]
    first_peak_ts = peak_timestamps[0]
    last_peak_ts = peak_timestamps[-1]

    # Map peak timestamps to day_df indices
    peak_df_idxs = []
    for pt in peak_timestamps:
        diffs = np.abs(df_ts - pt)
        peak_df_idxs.append(int(np.argmin(diffs)))
    peak_df_idxs = np.array(peak_df_idxs)

    # Map first/last peak to day_df index
    first_max = peak_df_idxs[0]
    last_max = peak_df_idxs[-1]

    # Near-peak window: first_max-6 to last_max+6 (~30 min each side)
    near_start = max(0, first_max - 6)
    near_end = min(n, last_max + 7)

    # Pre-peak window: 12 readings (~1hr) before first_max
    pre_start = max(0, first_max - 12)
    pre_end = first_max

    # --- Category A: Wind ---
    wind = pd.to_numeric(day_df.get("wind_speed_mph"), errors="coerce").values if "wind_speed_mph" in day_df.columns else np.full(n, np.nan)

    wind_near = wind[near_start:near_end]
    wind_at_peak_vals = wind[peak_df_idxs]
    wind_at_peak = float(np.nanmean(wind_at_peak_vals)) if np.any(~np.isnan(wind_at_peak_vals)) else np.nan
    wind_pre = wind[pre_start:pre_end]
    wind_pre_peak = float(np.nanmean(wind_pre)) if len(wind_pre) > 0 and np.any(~np.isnan(wind_pre)) else np.nan
    wind_calm_frac = float(np.nanmean(wind_near < 5)) if len(wind_near) > 0 and np.any(~np.isnan(wind_near)) else np.nan
    wind_near_valid = wind_near[~np.isnan(wind_near)]
    wind_gust_range = float(wind_near_valid.max() - wind_near_valid.min()) if len(wind_near_valid) >= 2 else np.nan
    wind_change_at_peak = (wind_at_peak - wind_pre_peak) if not (np.isnan(wind_at_peak) or np.isnan(wind_pre_peak)) else np.nan

    # --- Category B: Moisture ---
    rh = pd.to_numeric(day_df.get("relative_humidity_pct"), errors="coerce").values if "relative_humidity_pct" in day_df.columns else np.full(n, np.nan)

    rh_peak = rh[peak_df_idxs]
    rh_at_peak = float(np.nanmean(rh_peak)) if np.any(~np.isnan(rh_peak)) else np.nan
    rh_near = rh[near_start:near_end]
    rh_min_near_peak = float(np.nanmin(rh_near)) if len(rh_near) > 0 and np.any(~np.isnan(rh_near)) else np.nan

    # --- Category C: Dewpoint ---
    dp = pd.to_numeric(day_df.get("dewpoint_c"), errors="coerce").values if "dewpoint_c" in day_df.columns else np.full(n, np.nan)
    temp_c_all = pd.to_numeric(day_df.get("temperature_c"), errors="coerce").values if "temperature_c" in day_df.columns else np.full(n, np.nan)

    dp_peak = dp[peak_df_idxs]
    dewpoint_at_peak = float(np.nanmean(dp_peak)) if np.any(~np.isnan(dp_peak)) else np.nan

    # Dewpoint depression = temp - dewpoint (larger = drier air)
    temp_at_peak_vals = temp_c_all[peak_df_idxs]
    if np.any(~np.isnan(dp_peak)) and np.any(~np.isnan(temp_at_peak_vals)):
        dewpoint_depression_at_peak = float(np.nanmean(temp_at_peak_vals - dp_peak))
    else:
        dewpoint_depression_at_peak = np.nan

    # Dewpoint change near peak (trend: rising dp = moisture advection)
    dp_near = dp[near_start:near_end]
    dp_near_valid = dp_near[~np.isnan(dp_near)]
    if len(dp_near_valid) >= 2:
        dewpoint_change_near_peak = float(dp_near_valid[-1] - dp_near_valid[0])
    else:
        dewpoint_change_near_peak = np.nan

    # --- Category D: Heat index (Rothfusz regression, NWS formula) ---
    # Only defined for T >= 80°F and RH >= 40%; NaN otherwise.
    temp_f_peak = temp_at_peak_vals * 9.0 / 5.0 + 32.0 if temp_c_all is not None else np.full(len(peak_df_idxs), np.nan)
    rh_peak_vals = rh[peak_df_idxs]
    t_f = float(np.nanmean(temp_f_peak))
    r_h = float(np.nanmean(rh_peak_vals))
    if not np.isnan(t_f) and not np.isnan(r_h) and t_f >= 80.0 and r_h >= 40.0:
        # Rothfusz regression
        hi = (-42.379
              + 2.04901523 * t_f
              + 10.14333127 * r_h
              - 0.22475541 * t_f * r_h
              - 0.00683783 * t_f ** 2
              - 0.05481717 * r_h ** 2
              + 0.00122874 * t_f ** 2 * r_h
              + 0.00085282 * t_f * r_h ** 2
              - 0.00000199 * t_f ** 2 * r_h ** 2)
        heat_index_at_peak = hi
        heat_index_excess = hi - t_f  # how much hotter it "feels"
    else:
        heat_index_at_peak = np.nan
        heat_index_excess = np.nan

    # --- Category E: Pressure ---
    slp = pd.to_numeric(day_df.get("sea_level_pressure"), errors="coerce").values if "sea_level_pressure" in day_df.columns else np.full(n, np.nan)
    ptend = pd.to_numeric(day_df.get("pressure_tendency"), errors="coerce").values if "pressure_tendency" in day_df.columns else np.full(n, np.nan)

    slp_peak = slp[peak_df_idxs]
    pressure_at_peak = float(np.nanmean(slp_peak)) if np.any(~np.isnan(slp_peak)) else np.nan

    # Pressure change in near-peak window (falling = approaching front)
    slp_near = slp[near_start:near_end]
    slp_near_valid = slp_near[~np.isnan(slp_near)]
    if len(slp_near_valid) >= 2:
        pressure_change_near_peak = float(slp_near_valid[-1] - slp_near_valid[0])
    else:
        pressure_change_near_peak = np.nan

    ptend_peak = ptend[peak_df_idxs]
    pressure_tendency_at_peak = float(np.nanmean(ptend_peak)) if np.any(~np.isnan(ptend_peak)) else np.nan

    # --- Category F: Cloud cover ---
    # cloud_layer_code is a string like "CLR", "FEW", "SCT", "BKN", "OVC"
    # Encode as ordinal: CLR=0, FEW=1, SCT=2, BKN=3, OVC=4
    cloud_order = {"CLR": 0, "FEW": 1, "SCT": 2, "BKN": 3, "OVC": 4}
    if "cloud_layer_code" in day_df.columns:
        cloud_raw = day_df["cloud_layer_code"].values
        cloud_vals = np.array([cloud_order.get(str(v).strip().upper()[:3], np.nan) for v in cloud_raw])
    else:
        cloud_vals = np.full(n, np.nan)

    cloud_near_peak = cloud_vals[near_start:near_end]
    cloud_near_peak_valid = cloud_near_peak[~np.isnan(cloud_near_peak)]
    if len(cloud_near_peak_valid) > 0:
        # Mode: most frequent cloud cover code near peak
        vals, counts = np.unique(cloud_near_peak_valid.astype(int), return_counts=True)
        cloud_cover_at_peak = float(vals[np.argmax(counts)])
    else:
        cloud_cover_at_peak = np.nan

    cloud_near = cloud_vals[near_start:near_end]
    cloud_near_valid = cloud_near[~np.isnan(cloud_near)]
    if len(cloud_near_valid) >= 2:
        cloud_variability = float(np.std(cloud_near_valid))
    else:
        cloud_variability = np.nan

    return {
        "wind_at_peak": wind_at_peak,
        "wind_pre_peak": wind_pre_peak,
        "wind_calm_frac": wind_calm_frac,
        "wind_gust_range": wind_gust_range,
        "wind_change_at_peak": wind_change_at_peak,
        "rh_at_peak": rh_at_peak,
        "rh_min_near_peak": rh_min_near_peak,
        "dewpoint_at_peak": dewpoint_at_peak,
        "dewpoint_depression_at_peak": dewpoint_depression_at_peak,
        "dewpoint_change_near_peak": dewpoint_change_near_peak,
        "heat_index_at_peak": heat_index_at_peak,
        "heat_index_excess": heat_index_excess,
        "pressure_at_peak": pressure_at_peak,
        "pressure_change_near_peak": pressure_change_near_peak,
        "pressure_tendency_at_peak": pressure_tendency_at_peak,
        "cloud_cover_at_peak": cloud_cover_at_peak,
        "cloud_variability": cloud_variability,
    }


def _extract_auto_metar(day_df: pd.DataFrame):
    """Split a day's data into auto-obs and METAR/precision readings.

    Returns (auto_c, auto_ts, metar_c, n_auto) or None if insufficient data.
    auto_c/auto_ts are numpy arrays of whole-°C auto-obs values and timestamps.
    metar_c is a numpy array of precision (fractional) °C readings.
    """
    df = day_df.copy()
    df["temperature_c"] = pd.to_numeric(df["temperature_c"], errors="coerce")
    df["temperature_f"] = pd.to_numeric(df["temperature_f"], errors="coerce")

    minutes = pd.to_datetime(df["timestamp"].str[:19]).dt.minute
    tc = df["temperature_c"]
    temp_c_notna = tc.notna()
    frac_mask = (tc % 1 != 0) & temp_c_notna
    metar_minute = int(minutes[frac_mask].mode().iloc[0]) if frac_mask.any() else 53

    auto_mask = (tc % 1 == 0) & (minutes != metar_minute) & temp_c_notna
    metar_mask = ((minutes == metar_minute) | (tc % 1 != 0)) & temp_c_notna

    ts = pd.to_datetime(df["timestamp"].str[:19])
    auto_c = tc[auto_mask].values.astype(float)
    auto_ts = ts[auto_mask].values
    metar_c = tc[metar_mask].values.astype(float)
    metar_ts = ts[metar_mask].values

    # True daily high from 6h METARs (settlement-grade, correctly day-scoped)
    # Skip any 6h METAR before 01:00 — it covers the previous evening window
    if "max_temp_6h_f" not in df.columns:
        raise ValueError("max_temp_6h_f column missing — re-fetch history data")
    _m6h_mask = df["max_temp_6h_f"].notna() & (ts.dt.hour >= 1)
    _m6h_vals = pd.to_numeric(df.loc[_m6h_mask, "max_temp_6h_f"], errors="coerce")
    actual_max_f = round(float(_m6h_vals.max())) if not _m6h_vals.empty else None

    return auto_c, auto_ts, metar_c, metar_ts, actual_max_f


def _metar_features(
    metar_c: np.ndarray,
    metar_ts: np.ndarray,
    auto_c: np.ndarray,
    auto_ts: np.ndarray,
    max_c: int,
    possible_f: list,
    naive_f: int,
    dwell_ratio: float = 0.0,
    trans_ratio: float = 0.0,
) -> dict:
    """Compute METAR + combined auto/METAR features.

    All METAR values are expressed relative to ``max_c`` (the auto-obs
    whole-°C max).

    Returns a dict of feature name → value.  All features default to 0
    when no METAR data is available (``has_metar`` = 0).
    """
    n_metar = len(metar_c)

    # Rounding boundary: °C value where round(C*9/5+32) flips between
    # the two candidate °F values.  Only meaningful for 2-value pairs.
    if len(possible_f) == 2:
        f_low = possible_f[0]
        boundary_c = (f_low + 0.5 - 32) * 5.0 / 9.0
    else:
        boundary_c = np.nan

    if n_metar > 0:
        metar_max_c = float(metar_c.max())
        # All gaps relative to max_c
        metar_gap_c = metar_max_c - float(max_c)
        metar_mean_gap_c = float(metar_c.mean()) - float(max_c)

        if not np.isnan(boundary_c):
            metar_boundary_margin_c = metar_max_c - boundary_c
            metar_above_boundary = 1.0 if metar_max_c > boundary_c else 0.0
            metar_n_above = int(np.sum(metar_c > boundary_c))
        else:
            metar_boundary_margin_c = 0.0
            metar_above_boundary = 0.0
            metar_n_above = 0

        # METAR confirmation strength: how far above max_c, clamped to [0, 1]
        # 0 = at or below max_c, 1 = ≥1°C above max_c
        metar_confirm = min(max(metar_gap_c, 0.0), 1.0)

        # Time between METAR peak and auto peak (hours, +ve = METAR peaked later)
        metar_peak_idx = int(np.argmax(metar_c))
        metar_peak_ts = metar_ts[metar_peak_idx]
        auto_peak_idx = int(np.argmax(auto_c))
        auto_peak_ts = auto_ts[auto_peak_idx]
        peak_lag_min = float(
            (metar_peak_ts - auto_peak_ts) / np.timedelta64(1, "m")
        )
    else:
        metar_gap_c = 0.0
        metar_mean_gap_c = 0.0
        metar_boundary_margin_c = 0.0
        metar_above_boundary = 0.0
        metar_n_above = 0
        metar_confirm = 0.0
        peak_lag_min = 0.0

    has_metar = 1.0 if n_metar > 0 else 0.0

    return {
        "has_metar": has_metar,                          # 1.0 if any METAR T-group readings exist
        "n_metar": n_metar,                              # count of precision METAR readings
        "metar_gap_c": metar_gap_c,                      # metar_max - max_c (°C above auto peak)
        "metar_mean_gap_c": metar_mean_gap_c,            # mean(all METAR) - max_c (consistent vs outlier)
        "metar_above_boundary": metar_above_boundary,    # 1.0 if metar_max > C-to-F rounding boundary
        "metar_boundary_margin_c": metar_boundary_margin_c,  # metar_max - boundary (°C, +ve = favors f_high)
        "metar_n_above": metar_n_above,                    # count of METAR readings above boundary
        "metar_confirm": metar_confirm,                  # clamp(metar_gap, 0, 1) — METAR peak strength
        "peak_lag_min": peak_lag_min,                    # metar_peak - auto_peak (min, +ve = METAR peaked later)
    }


def extract_features(day_df: pd.DataFrame) -> Optional[dict]:
    """Extract all candidate features for a single day.

    Returns a dict of features + target, or None if insufficient data.
    Features marked '(unused)' were explored but did not improve accuracy.
    """
    result = _extract_auto_metar(day_df)
    if result is None:
        return None
    auto_c, auto_ts, metar_c, metar_ts, actual_max_f = result
    if len(auto_c) < 20 or actual_max_f is None:
        return None

    max_c = int(round(auto_c.max()))
    possible_f = c_to_possible_f(max_c)
    naive_f = round(max_c * 9 / 5 + 32)

    # Only evaluate naive=low cases (the ambiguous ones)
    if len(possible_f) != 2 or naive_f != possible_f[0]:
        return None

    f_low, f_high = possible_f[0], possible_f[-1]
    ac = auto_c
    ats = auto_ts
    n_auto = len(ac)

    # All features are normalized to [0, 1] ratios for comparability.

    # --- Core features (used in production model) --------------------------

    # Dwell ratio: fraction of all readings at max_c
    dwell_count = int(np.sum(ac >= max_c))
    dwell_ratio = dwell_count / n_auto

    # Transition ratio: readings at max_c / (readings at max_c + max_c-1)
    # Measures how much of the near-peak time is AT max vs one step below.
    n_at_max = int(np.sum(ac == max_c))
    n_at_prev = int(np.sum(ac == max_c - 1))
    trans_denom = n_at_max + n_at_prev
    trans_ratio = n_at_max / trans_denom if trans_denom > 0 else 0

    # --- Additional features explored --------------------------------------

    # Consecutive dwell ratio: longest consecutive run at max_c / total readings
    max_consec = 0
    cur = 0
    for v in ac:
        if v == max_c:
            cur += 1
        else:
            cur = 0
        max_consec = max(max_consec, cur)
    consec_ratio = max_consec / n_auto

    # Time at max ratio: hours at max_c / total observation hours
    max_idxs = np.where(ac == max_c)[0]
    time_at_max_hrs = float(
        (ats[max_idxs[-1]] - ats[max_idxs[0]]) / np.timedelta64(1, "h")
    )
    total_hrs = float(
        (ats[-1] - ats[0]) / np.timedelta64(1, "h")
    ) if len(ats) > 1 else 1.0
    time_at_max_ratio = time_at_max_hrs / total_hrs if total_hrs > 0 else 0

    # Fraction of day above max_c-1
    n_above_prev = int(np.sum(ac >= max_c - 1))
    frac_above_prev = n_above_prev / n_auto

    # --- Features explored that did NOT help (commented out) ----------------

    # # Momentum around peak — approach/departure rates at whole-°C resolution
    # # All ~zero correlation: approach_rate=-0.015, departure_rate=+0.006,
    # # momentum_ratio=+0.023, rise_streak_ratio=+0.058
    # # With whole-°C granularity, all peaks look similar (~2°C/hr rise, ~2.5 drop).
    # # The sub-degree signal is destroyed by rounding.
    # first_max_idx = int(max_idxs[0])
    # last_max_idx = int(max_idxs[-1])
    # pre_n = 6
    # pre_start = max(0, first_max_idx - pre_n)
    # pre_vals = ac[pre_start:first_max_idx + 1]
    # pre_ts_arr = ats[pre_start:first_max_idx + 1]
    # if len(pre_vals) >= 2:
    #     dt_h = float((pre_ts_arr[-1] - pre_ts_arr[0]) / np.timedelta64(1, "h"))
    #     approach_rate = (float(pre_vals[-1] - pre_vals[0]) / dt_h) if dt_h > 0 else 0
    # else:
    #     approach_rate = 0
    # post_n = 6
    # post_end = min(len(ac), last_max_idx + post_n + 1)
    # post_vals = ac[last_max_idx:post_end]
    # post_ts_arr = ats[last_max_idx:post_end]
    # if len(post_vals) >= 2:
    #     dt_h = float((post_ts_arr[-1] - post_ts_arr[0]) / np.timedelta64(1, "h"))
    #     departure_rate = (float(post_vals[-1] - post_vals[0]) / dt_h) if dt_h > 0 else 0
    # else:
    #     departure_rate = 0
    # abs_sum = abs(approach_rate) + abs(departure_rate)
    # momentum_ratio = approach_rate / abs_sum if abs_sum > 0 else 0.5
    # rise_streak = 1
    # for i in range(first_max_idx - 1, -1, -1):
    #     if ac[i] <= ac[i + 1]:
    #         rise_streak += 1
    #     else:
    #         break
    # rise_streak_ratio = rise_streak / n_auto

    # # Oscillation count: transitions between max_c and max_c-1
    # # (corr=+0.004 — no signal)
    # near_peak = ac[(ac == max_c) | (ac == max_c - 1)]
    # oscillations = sum(1 for i in range(1, len(near_peak))
    #                    if near_peak[i] != near_peak[i-1])

    # # Block count: separate blocks of consecutive max_c readings
    # # (corr=+0.012 — no signal)
    # blocks = 0
    # in_block = False
    # for v in ac:
    #     if v == max_c:
    #         if not in_block:
    #             blocks += 1
    #             in_block = True
    #     else:
    #         in_block = False

    # # Readings at max_c-2 (how far below max is the bulk)
    # # (corr=-0.041 — no signal)
    # n_at_m2 = int(np.sum(ac == max_c - 2))

    # # Late peak position: last max_c reading as fraction of day
    # # (corr=+0.152 — weak, doesn't help in combination)
    # last_max_pos = max_idxs[-1] / n_auto

    # # Daily temperature range (max_c - min_c)
    # # (corr=-0.052 — no signal)
    # daily_range_c = max_c - int(round(ac.min()))

    # # Pre-peak slope: °C/hr over 12 readings before first max_c
    # # (corr=+0.032 — no signal)
    # pre_start = max(0, max_idxs[0] - 12)
    # pre_vals = ac[pre_start:max_idxs[0] + 1]
    # pre_ts = ats[pre_start:max_idxs[0] + 1]
    # if len(pre_vals) >= 3:
    #     dt = (pre_ts[-1] - pre_ts[0]) / np.timedelta64(1, "h")
    #     pre_slope = (pre_vals[-1] - pre_vals[0]) / dt if dt > 0 else 0
    # else:
    #     pre_slope = 0

    # # Post-peak slope: °C/hr over 12 readings after last max_c
    # # (corr=-0.070 — weak, doesn't help in combination)
    # post_end = min(len(ac), max_idxs[-1] + 13)
    # post_vals = ac[max_idxs[-1]:post_end]
    # post_ts = ats[max_idxs[-1]:post_end]
    # if len(post_vals) >= 3:
    #     dt = (post_ts[-1] - post_ts[0]) / np.timedelta64(1, "h")
    #     post_slope = (post_vals[-1] - post_vals[0]) / dt if dt > 0 else 0
    # else:
    #     post_slope = 0

    # --- METAR + combined features -------------------------------------------
    metar_feats = _metar_features(metar_c, metar_ts, ac, ats, max_c, possible_f,
                                  naive_f, dwell_ratio=dwell_ratio,
                                  trans_ratio=trans_ratio)

    # --- Target -------------------------------------------------------------
    is_high = 1 if actual_max_f >= f_high else 0

    return {
        # All features normalized to [0, 1]
        "dwell_ratio": dwell_ratio,
        "trans_ratio": trans_ratio,
        "consec_ratio": consec_ratio,
        "time_at_max_ratio": time_at_max_ratio,
        "frac_above_prev": frac_above_prev,
        # METAR features
        **metar_feats,
        # Metadata
        "max_c": max_c,
        "n_auto": n_auto,
        "f_low": f_low,
        "f_high": f_high,
        "naive_f": naive_f,
        "actual_max_f": actual_max_f,
        "is_high": is_high,
    }


def extract_regression_features(
    day_df: pd.DataFrame,
    solar_noon_hour: Optional[float] = None,
) -> Optional[dict]:
    """Extract regression features for ALL days (not just naive=low).

    Predicts actual settlement °F from auto-obs peak shape/momentum.
    If *solar_noon_hour* is provided (decimal hours from cached API data),
    computes peak-relative-to-solar-noon features.
    Returns a dict of features + target, or None if insufficient data.
    """
    result = _extract_auto_metar(day_df)
    if result is None:
        return None
    auto_c, auto_ts, metar_c, metar_ts, actual_max_f = result
    if len(auto_c) < 20 or actual_max_f is None:
        return None

    ac = auto_c
    ats = auto_ts
    n_auto = len(ac)
    max_c = int(round(ac.max()))

    possible_f = c_to_possible_f(max_c)
    naive_f_float = max_c * 9.0 / 5.0 + 32.0
    naive_f = round(naive_f_float)

    # --- Baseline features ---------------------------------------------------

    # Dwell count: number of readings at max_c
    n_at_max = int(np.sum(ac == max_c))
    dwell_count = n_at_max

    # Transition count: readings at max_c-1 (near-peak neighbors)
    n_at_prev = int(np.sum(ac == max_c - 1))
    trans_count = n_at_prev

    # Consecutive count: longest consecutive run at max_c
    max_consec = 0
    cur = 0
    for v in ac:
        if v == max_c:
            cur += 1
        else:
            cur = 0
        max_consec = max(max_consec, cur)
    consec_count = max_consec


    max_idxs = np.where(ac == max_c)[0]
    first_max_idx = int(max_idxs[0])
    last_max_idx = int(max_idxs[-1])

    # Count of readings at or above max_c-1
    above_prev_count = int(np.sum(ac >= max_c - 1))

    # --- Peak momentum features ----------------------------------------------

    def _endpoint_rate(vals, ts):
        """°C/hr from first to last element."""
        if len(vals) < 2:
            return 0.0
        dt_h = float((ts[-1] - ts[0]) / np.timedelta64(1, "h"))
        return float(vals[-1] - vals[0]) / dt_h if dt_h > 0 else 0.0

    # Approach rates at multiple horizons before first max_c
    approach_rate_15m = _endpoint_rate(
        ac[max(0, first_max_idx - 3):first_max_idx + 1],
        ats[max(0, first_max_idx - 3):first_max_idx + 1])
    approach_rate_30m = _endpoint_rate(
        ac[max(0, first_max_idx - 6):first_max_idx + 1],
        ats[max(0, first_max_idx - 6):first_max_idx + 1])
    approach_rate_1h = _endpoint_rate(
        ac[max(0, first_max_idx - 12):first_max_idx + 1],
        ats[max(0, first_max_idx - 12):first_max_idx + 1])

    # Departure rates at multiple horizons after last max_c
    departure_rate_15m = _endpoint_rate(
        ac[last_max_idx:min(len(ac), last_max_idx + 4)],
        ats[last_max_idx:min(len(ac), last_max_idx + 4)])
    departure_rate_30m = _endpoint_rate(
        ac[last_max_idx:min(len(ac), last_max_idx + 7)],
        ats[last_max_idx:min(len(ac), last_max_idx + 7)])


    # --- Peak shape features -------------------------------------------------

    # Rise hours: time from first (max_c - 2) reading to first max_c
    above_m2 = np.where(ac >= max_c - 2)[0]
    if len(above_m2) > 0:
        first_m2_idx = int(above_m2[0])
        rise_hrs = float(
            (ats[first_max_idx] - ats[first_m2_idx]) / np.timedelta64(1, "h")
        )
    else:
        rise_hrs = 0.0

    # Readings at max_c-1 before peak
    before_peak = ac[:first_max_idx]
    readings_at_prev_before = int(np.sum(before_peak == max_c - 1))

    # --- Additional peak features --------------------------------------------

    # Time of first/last max_c reading as HHMM integer (e.g. 1430)
    first_max_time = int(pd.Timestamp(ats[first_max_idx]).hour * 100
                         + pd.Timestamp(ats[first_max_idx]).minute)
    last_max_time = int(pd.Timestamp(ats[last_max_idx]).hour * 100
                        + pd.Timestamp(ats[last_max_idx]).minute)

    # Number of separate consecutive blocks of max_c readings
    at_max_mask = (ac == max_c).astype(int)
    n_max_blocks = int(np.sum(np.diff(at_max_mask) == 1))
    if at_max_mask[0] == 1:
        n_max_blocks += 1

    # Oscillation count: transitions between max_c and max_c-1 among near-peak readings
    near_peak = ac[ac >= max_c - 1]
    if len(near_peak) > 1:
        transitions = int(np.sum(np.abs(np.diff(near_peak)) >= 1))
        oscillation_count = transitions / len(near_peak)
    else:
        oscillation_count = 0.0

    # Single reading peak: 1.0 if very few readings at max
    single_reading_peak = 1.0 if n_at_max <= 2 else 0.0

    # Daily range normalized
    min_c = int(round(ac.min()))
    daily_range_c = float(max_c - min_c) / max_c if max_c != 0 else 0.0

    # Approach acceleration: 30m rate - 1h rate (positive = accelerating into peak)
    approach_accel = approach_rate_30m - approach_rate_1h

    # Std of °C readings in 12 readings (~1hr) before first max_c
    pre_peak_start = max(0, first_max_idx - 12)
    pre_peak_vals = ac[pre_peak_start:first_max_idx]
    pre_peak_std = float(np.std(pre_peak_vals)) if len(pre_peak_vals) >= 2 else 0.0

    # Pre-peak momentum: OLS slope (°C/hr) over 24 readings (~2hrs) before first max_c
    pre24_start = max(0, first_max_idx - 24)
    pre24_vals = ac[pre24_start:first_max_idx + 1]
    pre24_ts = ats[pre24_start:first_max_idx + 1]
    if len(pre24_vals) >= 4:
        t_hrs = np.array([(t - pre24_ts[0]) / np.timedelta64(1, "h")
                          for t in pre24_ts], dtype=float)
        # Simple OLS slope
        t_mean = t_hrs.mean()
        v_mean = pre24_vals.mean()
        denom = np.sum((t_hrs - t_mean) ** 2)
        pre_peak_momentum = float(
            np.sum((t_hrs - t_mean) * (pre24_vals - v_mean)) / denom
        ) if denom > 0 else 0.0
    else:
        pre_peak_momentum = 0.0

    # --- First-order derivative features --------------------------------------
    dt_hr = 1.0 / 12.0  # 5 min in hours

    ac_float = ac.astype(float)
    d1_full = np.gradient(ac_float, dt_hr)  # °C/hr at every 5-min reading

    # Approach rate over 2h before peak (longer horizon than 30m/1h)
    pre24_rate_start = max(0, first_max_idx - 24)
    pre24_rate_vals = ac[pre24_rate_start:first_max_idx + 1]
    pre24_rate_ts = ats[pre24_rate_start:first_max_idx + 1]
    if len(pre24_rate_vals) >= 2:
        dt_h = float((pre24_rate_ts[-1] - pre24_rate_ts[0]) / np.timedelta64(1, "h"))
        approach_rate_2h = float(pre24_rate_vals[-1] - pre24_rate_vals[0]) / dt_h if dt_h > 0 else 0.0
    else:
        approach_rate_2h = 0.0

    # Departure rate over 1h after last max_c
    post12_end = min(len(ac), last_max_idx + 13)
    post12_vals = ac[last_max_idx:post12_end]
    post12_ts = ats[last_max_idx:post12_end]
    if len(post12_vals) >= 2:
        dt_h = float((post12_ts[-1] - post12_ts[0]) / np.timedelta64(1, "h"))
        departure_rate_1h = float(post12_vals[-1] - post12_vals[0]) / dt_h if dt_h > 0 else 0.0
    else:
        departure_rate_1h = 0.0

    # --- d1 stats over 1hr windows ---
    # Max dT/dt in 1hr before peak (steepest instantaneous climb)
    d1_pre_start = max(0, first_max_idx - 12)
    d1_pre_slice = d1_full[d1_pre_start:first_max_idx + 1]
    max_d1_pre = float(np.max(d1_pre_slice)) if len(d1_pre_slice) > 0 else 0.0

    # Min dT/dt in 1hr after peak (steepest instantaneous drop)
    d1_post_start = last_max_idx
    d1_post_end = min(len(d1_full), last_max_idx + 13)
    d1_post_slice = d1_full[d1_post_start:d1_post_end]
    min_d1_post = float(np.min(d1_post_slice)) if len(d1_post_slice) > 0 else 0.0

    # Mean dT/dt across peak region (near zero = plateau, negative = declining through peak)
    d1_peak_start = max(0, first_max_idx - 3)
    d1_peak_end = min(len(d1_full), last_max_idx + 4)
    d1_peak_slice = d1_full[d1_peak_start:d1_peak_end]
    mean_d1_peak = float(np.mean(d1_peak_slice)) if len(d1_peak_slice) > 0 else 0.0

    # Std of dT/dt in 1hr around peak (noisy derivative = unstable peak)
    d1_vol_start = max(0, first_max_idx - 6)
    d1_vol_end = min(len(d1_full), last_max_idx + 7)
    d1_vol_slice = d1_full[d1_vol_start:d1_vol_end]
    std_d1_peak = float(np.std(d1_vol_slice)) if len(d1_vol_slice) >= 2 else 0.0

    # --- d1 stats over 30min windows (shorter horizon) ---
    d1_pre_30m_start = max(0, first_max_idx - 6)
    d1_pre_30m_slice = d1_full[d1_pre_30m_start:first_max_idx + 1]
    max_d1_pre_30m = float(np.max(d1_pre_30m_slice)) if len(d1_pre_30m_slice) > 0 else 0.0

    d1_post_30m_end = min(len(d1_full), last_max_idx + 7)
    d1_post_30m_slice = d1_full[last_max_idx:d1_post_30m_end]
    min_d1_post_30m = float(np.min(d1_post_30m_slice)) if len(d1_post_30m_slice) > 0 else 0.0

    d1_vol_30m_start = max(0, first_max_idx - 3)
    d1_vol_30m_end = min(len(d1_full), last_max_idx + 4)
    d1_vol_30m_slice = d1_full[d1_vol_30m_start:d1_vol_30m_end]
    std_d1_peak_30m = float(np.std(d1_vol_30m_slice)) if len(d1_vol_30m_slice) >= 2 else 0.0
    mean_d1_peak_30m = float(np.mean(d1_vol_30m_slice)) if len(d1_vol_30m_slice) > 0 else 0.0

    # --- Second-order derivative features ------------------------------------

    d2_full = np.gradient(d1_full, dt_hr)  # °C/hr² at every 5-min reading

    # Mean d²T/dt² over 12 readings (~1hr) before first max_c
    # Negative = decelerating into peak (plateau), positive = accelerating (spike)
    if first_max_idx >= 4:
        pre_d2_start = max(0, first_max_idx - 12)
        pre_d2_vals = ac[pre_d2_start:first_max_idx + 1]
        if len(pre_d2_vals) >= 3:
            d2 = np.gradient(np.gradient(pre_d2_vals.astype(float), dt_hr), dt_hr)
            pre_peak_d2 = float(np.mean(d2))
        else:
            pre_peak_d2 = 0.0
    else:
        pre_peak_d2 = 0.0

    # Mean d²T/dt² across peak region (first_max to last_max ± 3)
    # Sharp peak = large negative d2, flat plateau = near zero
    peak_region_start = max(0, first_max_idx - 3)
    peak_region_end = min(len(ac), last_max_idx + 4)
    peak_region_vals = ac[peak_region_start:peak_region_end]
    if len(peak_region_vals) >= 3:
        d2 = np.gradient(np.gradient(peak_region_vals.astype(float), dt_hr), dt_hr)
        peak_d2 = float(np.mean(d2))
    else:
        peak_d2 = 0.0

    # Max |d²T/dt²| in 1hr before peak (sharpest inflection)
    pre_inflect_start = max(0, first_max_idx - 12)
    pre_inflect_vals = ac[pre_inflect_start:first_max_idx + 1]
    if len(pre_inflect_vals) >= 3:
        d2 = np.gradient(np.gradient(pre_inflect_vals.astype(float), dt_hr), dt_hr)
        max_d2_pre = float(np.max(np.abs(d2)))
    else:
        max_d2_pre = 0.0

    # Std of d²T/dt² across peak region (curvature variability)
    d2_peak_slice = d2_full[peak_region_start:peak_region_end]
    std_d2_peak = float(np.std(d2_peak_slice)) if len(d2_peak_slice) >= 2 else 0.0

    # Min d²T/dt² in 1hr after peak (sharpest deceleration post-peak)
    d2_post_start = last_max_idx
    d2_post_end = min(len(d2_full), last_max_idx + 13)
    d2_post_slice = d2_full[d2_post_start:d2_post_end]
    min_d2_post = float(np.min(d2_post_slice)) if len(d2_post_slice) > 0 else 0.0

    # --- d2 stats over 30min windows (shorter horizon) ---
    # Mean d²T/dt² over 30min before peak
    pre_d2_30m_start = max(0, first_max_idx - 6)
    pre_d2_30m_vals = ac[pre_d2_30m_start:first_max_idx + 1]
    if len(pre_d2_30m_vals) >= 3:
        d2_30m = np.gradient(np.gradient(pre_d2_30m_vals.astype(float), dt_hr), dt_hr)
        pre_peak_d2_30m = float(np.mean(d2_30m))
    else:
        pre_peak_d2_30m = 0.0

    # Max |d²T/dt²| in 30min before peak
    if len(pre_d2_30m_vals) >= 3:
        max_d2_pre_30m = float(np.max(np.abs(d2_30m)))
    else:
        max_d2_pre_30m = 0.0

    # Std of d²T/dt² in 30min around peak
    d2_vol_30m_start = max(0, first_max_idx - 3)
    d2_vol_30m_end = min(len(d2_full), last_max_idx + 4)
    d2_vol_30m_slice = d2_full[d2_vol_30m_start:d2_vol_30m_end]
    std_d2_peak_30m = float(np.std(d2_vol_30m_slice)) if len(d2_vol_30m_slice) >= 2 else 0.0

    # Min d²T/dt² in 30min after peak
    d2_post_30m_end = min(len(d2_full), last_max_idx + 7)
    d2_post_30m_slice = d2_full[last_max_idx:d2_post_30m_end]
    min_d2_post_30m = float(np.min(d2_post_30m_slice)) if len(d2_post_30m_slice) > 0 else 0.0

    # --- Temporal features (from timestamps, no re-fetch needed) ---------------

    peak_ts_val = ats[first_max_idx]
    peak_dt = pd.Timestamp(peak_ts_val)
    day_of_year = peak_dt.day_of_year
    day_of_year_sin = float(np.sin(2 * np.pi * day_of_year / 365.25))
    day_of_year_cos = float(np.cos(2 * np.pi * day_of_year / 365.25))

    peak_hour_decimal = peak_dt.hour + peak_dt.minute / 60.0

    # Solar noon from cached API data (sunrisesunset.io)
    solar_noon = solar_noon_hour if solar_noon_hour is not None else 12.0
    peak_minus_solar_noon = peak_hour_decimal - solar_noon

    # --- Volatility features (from auto-obs °C converted to °F) ----------------

    # Use auto-obs array (ac) converted to °F — indices align with first_max_idx
    ac_f = ac * 9.0 / 5.0 + 32.0

    # Std of temp_f in 30min around peak (±3 readings)
    vol30_start = max(0, first_max_idx - 3)
    vol30_end = min(n_auto, last_max_idx + 4)
    vol30_window = ac_f[vol30_start:vol30_end]
    vol30_valid = vol30_window[~np.isnan(vol30_window)] if len(vol30_window) > 0 else np.array([])
    temp_f_volatility_30m = float(np.std(vol30_valid)) if len(vol30_valid) >= 2 else 0.0

    # Std of temp_f in 1hr around peak (±6 readings)
    vol_start = max(0, first_max_idx - 6)
    vol_end = min(n_auto, last_max_idx + 7)
    vol_window = ac_f[vol_start:vol_end]
    vol_valid = vol_window[~np.isnan(vol_window)] if len(vol_window) > 0 else np.array([])
    temp_f_volatility_1hr = float(np.std(vol_valid)) if len(vol_valid) >= 2 else 0.0

    # Sub-hour oscillation rate: sign-change rate in consecutive 5-min diffs
    osc_vals = ac_f[vol_start:vol_end]
    osc_valid = osc_vals[~np.isnan(osc_vals)] if len(osc_vals) > 0 else np.array([])
    if len(osc_valid) >= 3:
        diffs = np.diff(osc_valid)
        sign_changes = np.sum(diffs[:-1] * diffs[1:] < 0)
        sub_hour_oscillation_rate = float(sign_changes / (len(diffs) - 1)) if len(diffs) > 1 else 0.0
    else:
        sub_hour_oscillation_rate = 0.0

    # Peak temp_f minus 30-min moving average at same time
    ma_window = 6  # 30 min = 6 readings
    if n_auto >= ma_window:
        ma_start = max(0, first_max_idx - ma_window + 1)
        ma_vals = ac_f[ma_start:first_max_idx + 1]
        ma_valid = ma_vals[~np.isnan(ma_vals)]
        if len(ma_valid) > 0:
            max_minus_ma30 = float(ac_f[first_max_idx] - np.mean(ma_valid))
        else:
            max_minus_ma30 = 0.0
    else:
        max_minus_ma30 = 0.0

    # Volatility in 2hr window around peak (±12 readings)
    vol2_start = max(0, first_max_idx - 12)
    vol2_end = min(n_auto, last_max_idx + 13)
    vol2_window = ac_f[vol2_start:vol2_end]
    vol2_valid = vol2_window[~np.isnan(vol2_window)] if len(vol2_window) > 0 else np.array([])
    temp_f_volatility_2hr = float(np.std(vol2_valid)) if len(vol2_valid) >= 2 else 0.0

    # Range of °F values in 1hr around peak (max - min)
    temp_f_range_1hr = float(np.ptp(vol_valid)) if len(vol_valid) >= 2 else 0.0

    # IQR of °F values in 2hr around peak (robust spread measure)
    if len(vol2_valid) >= 4:
        temp_f_iqr_2hr = float(np.percentile(vol2_valid, 75) - np.percentile(vol2_valid, 25))
    else:
        temp_f_iqr_2hr = 0.0

    # Mean absolute successive difference in °F (captures jitter independent of trend)
    if len(vol_valid) >= 2:
        temp_f_mean_abs_diff = float(np.mean(np.abs(np.diff(vol_valid))))
    else:
        temp_f_mean_abs_diff = 0.0

    # Peak temp_f minus 1hr moving average (wider context than 30min)
    ma60_window = 12  # 1hr = 12 readings
    if n_auto >= ma60_window:
        ma60_start = max(0, first_max_idx - ma60_window + 1)
        ma60_vals = ac_f[ma60_start:first_max_idx + 1]
        ma60_valid = ma60_vals[~np.isnan(ma60_vals)]
        max_minus_ma60 = float(ac_f[first_max_idx] - np.mean(ma60_valid)) if len(ma60_valid) > 0 else 0.0
    else:
        max_minus_ma60 = 0.0

    # Peak temp_f minus 15min moving average (tightest context)
    ma15_window = 3  # 15 min = 3 readings
    if n_auto >= ma15_window:
        ma15_start = max(0, first_max_idx - ma15_window + 1)
        ma15_vals = ac_f[ma15_start:first_max_idx + 1]
        ma15_valid = ma15_vals[~np.isnan(ma15_vals)]
        max_minus_ma15 = float(ac_f[first_max_idx] - np.mean(ma15_valid)) if len(ma15_valid) > 0 else 0.0
    else:
        max_minus_ma15 = 0.0

    # Max jump between consecutive readings within ±1hr of peak
    jump_start = max(0, first_max_idx - 12)
    jump_end = min(n_auto, last_max_idx + 13)
    jump_window = ac[jump_start:jump_end]
    if len(jump_window) >= 2:
        max_jump_near_peak = float(np.max(np.abs(np.diff(jump_window))))
    else:
        max_jump_near_peak = 0.0

    # Did any consecutive auto obs jump ≥2°F to the current auto max °F?
    # Signals a transient spike rather than sustained climb.
    has_2f_jump_to_peak = 0.0
    if n_auto >= 2:
        auto_f = np.round(ac * 9.0 / 5.0 + 32.0).astype(int)
        max_f = int(auto_f.max())
        for j in range(1, n_auto):
            if auto_f[j] >= max_f and auto_f[j] - auto_f[j - 1] >= 2:
                has_2f_jump_to_peak = 1.0
                break

    # --- Rounding gap at peak edge -------------------------------------------
    # Does dropping from max_c to max_c-1 cause a 2°F shift in settlement?
    # e.g. 27°C→81°F, 26°C→79°F = 2°F gap. This makes the peak fragile:
    # losing 1°C costs 2°F instead of the usual 1°F.
    naive_f_at_max = round(max_c * 9.0 / 5.0 + 32.0)
    naive_f_at_prev = round((max_c - 1) * 9.0 / 5.0 + 32.0)
    naive_f_at_next = round((max_c + 1) * 9.0 / 5.0 + 32.0)
    peak_edge_f_gap_below = naive_f_at_max - naive_f_at_prev  # 1 or 2 (cost of losing 1°C)
    peak_edge_f_gap_above = naive_f_at_next - naive_f_at_max  # 1 or 2 (gain from +1°C)

    # Distance in °C from max_c to the rounding boundary for next/prev °F
    # Upper boundary: °C where round(C*9/5+32) flips from naive_f to naive_f+1
    # Lower boundary: °C where round(C*9/5+32) flips from naive_f to naive_f-1
    c_upper_boundary = (naive_f + 0.5 - 32.0) * 5.0 / 9.0
    c_lower_boundary = (naive_f - 0.5 - 32.0) * 5.0 / 9.0
    dist_c_to_next_f = c_upper_boundary - max_c   # °C margin before rounding up
    dist_c_to_prev_f = max_c - c_lower_boundary   # °C margin before rounding down

    # Range of °F in 30min around peak
    temp_f_range_30m = float(np.ptp(vol30_valid)) if len(vol30_valid) >= 2 else 0.0

    # Mean absolute successive difference in 30min window
    if len(vol30_valid) >= 2:
        temp_f_mean_abs_diff_30m = float(np.mean(np.abs(np.diff(vol30_valid))))
    else:
        temp_f_mean_abs_diff_30m = 0.0

    # IQR of °F in 1hr around peak
    if len(vol_valid) >= 4:
        temp_f_iqr_1hr = float(np.percentile(vol_valid, 75) - np.percentile(vol_valid, 25))
    else:
        temp_f_iqr_1hr = 0.0

    # --- Meteorological features (from day_df columns) ------------------------
    met_feats = _met_features(day_df, ats, max_idxs)

    return {
        # --- Baseline ---
        "naive_f_float": naive_f_float,              # max_c converted to °F (fractional)
        "max_c": float(max_c),                       # highest whole-°C auto reading
        "max_c_mod5": float(max_c % 5),              # 5°C cycle position (rounding pattern)
        "dwell_count": dwell_count,                   # readings at max_c
        "trans_count": trans_count,                   # readings at max_c - 1
        "consec_count": consec_count,                 # longest consecutive run at max_c
        "above_prev_count": above_prev_count,         # readings >= max_c - 1
        # --- Peak momentum ---
        "approach_rate_15m": approach_rate_15m,        # °C/hr over 15min before first max_c
        "approach_rate_30m": approach_rate_30m,        # °C/hr over 30min before first max_c
        "approach_rate_1h": approach_rate_1h,          # °C/hr over 1hr before first max_c
        "departure_rate_15m": departure_rate_15m,      # °C/hr over 15min after last max_c
        "departure_rate_30m": departure_rate_30m,      # °C/hr over 30min after last max_c
        # --- Peak shape ---
        "rise_hrs": rise_hrs,                          # hours from first (max_c-2) to first max_c
        "readings_at_prev_before": readings_at_prev_before,  # max_c-1 readings before peak
        # --- Rounding boundary ---
        "c_to_f_frac": naive_f_float % 1.0,           # fractional part of C-to-F conversion
        "boundary_distance": min(naive_f_float % 1.0, 1.0 - naive_f_float % 1.0),  # symmetric distance to nearest rounding flip
        "naive_is_high": 1.0 if (len(possible_f) == 2 and naive_f == possible_f[-1]) else 0.0,  # 1 if naive round(C→F) gives the higher candidate
        "n_possible_f": float(len(possible_f)),        # 1 (exact) or 2 (ambiguous)
        # --- Rounding × peak interactions (help linear models) ---
        "frac_x_dwell": (naive_f_float % 1.0) * dwell_count,
        "frac_x_consec": (naive_f_float % 1.0) * consec_count,
        # --- Additional peak features ---
        "last_max_time": last_max_time,                # HHMM of last max_c reading
        "first_max_time": first_max_time,              # HHMM of first max_c reading
        "n_max_blocks": n_max_blocks,                  # separate consecutive blocks at max_c
        "oscillation_count": oscillation_count,        # transitions between max_c and max_c-1
        "single_reading_peak": single_reading_peak,    # 1.0 if n_at_max <= 2
        "daily_range_c": daily_range_c,                # (max_c - min_c) / max_c
        "approach_accel": approach_accel,               # approach_rate_30m - approach_rate_1h

        "pre_peak_std": pre_peak_std,                  # std of °C in 1hr before first max_c
        "pre_peak_momentum": pre_peak_momentum,        # OLS slope °C/hr over 2hr before peak
        # --- First-order derivatives (1hr) ---
        "approach_rate_2h": approach_rate_2h,            # °C/hr over 2hr before first max_c
        "departure_rate_1h": departure_rate_1h,          # °C/hr over 1hr after last max_c
        "max_d1_pre": max_d1_pre,                        # max dT/dt in 1hr before peak
        "min_d1_post": min_d1_post,                      # min dT/dt in 1hr after peak
        "mean_d1_peak": mean_d1_peak,                    # mean dT/dt across peak region
        "std_d1_peak": std_d1_peak,                      # std of dT/dt in 1hr around peak
        # --- First-order derivatives (30min) ---
        "max_d1_pre_30m": max_d1_pre_30m,                # max dT/dt in 30min before peak
        "min_d1_post_30m": min_d1_post_30m,              # min dT/dt in 30min after peak
        "mean_d1_peak_30m": mean_d1_peak_30m,            # mean dT/dt in 30min around peak
        "std_d1_peak_30m": std_d1_peak_30m,              # std of dT/dt in 30min around peak
        # --- Second-order derivatives (1hr) ---
        "pre_peak_d2": pre_peak_d2,                    # mean d²T/dt² over 1hr before peak
        "peak_d2": peak_d2,                            # mean d²T/dt² across peak region ±3
        "max_d2_pre": max_d2_pre,                      # max |d²T/dt²| in 1hr before peak
        "std_d2_peak": std_d2_peak,                      # std of d²T/dt² across peak region
        "min_d2_post": min_d2_post,                      # min d²T/dt² in 1hr after peak (sharpest decel)
        # --- Second-order derivatives (30min) ---
        "pre_peak_d2_30m": pre_peak_d2_30m,              # mean d²T/dt² over 30min before peak
        "max_d2_pre_30m": max_d2_pre_30m,                # max |d²T/dt²| in 30min before peak
        "std_d2_peak_30m": std_d2_peak_30m,              # std of d²T/dt² in 30min around peak
        "min_d2_post_30m": min_d2_post_30m,              # min d²T/dt² in 30min after peak
        # --- METAR features ---
        **_metar_features(metar_c, metar_ts, ac, ats, max_c, possible_f,
                         naive_f, dwell_ratio=dwell_count / n_auto,
                         trans_ratio=(n_at_max / (n_at_max + n_at_prev)
                                      if (n_at_max + n_at_prev) > 0 else 0.0)),
        # --- Rounding × METAR interaction ---
        "frac_x_metar_confirm": (naive_f_float % 1.0) * min(max(
            (float(metar_c.max()) - float(max_c)) if len(metar_c) > 0 else 0.0,
            0.0), 1.0),
        # --- Meteorological features ---
        **met_feats,
        # --- Temporal features ---
        "day_of_year_sin": day_of_year_sin,
        "day_of_year_cos": day_of_year_cos,
        "solar_noon": solar_noon,
        "peak_minus_solar_noon": peak_minus_solar_noon,
        # --- Volatility features ---
        "temp_f_volatility_30m": temp_f_volatility_30m,
        "temp_f_volatility_1hr": temp_f_volatility_1hr,
        "temp_f_volatility_2hr": temp_f_volatility_2hr,
        "temp_f_range_30m": temp_f_range_30m,
        "temp_f_range_1hr": temp_f_range_1hr,
        "temp_f_iqr_1hr": temp_f_iqr_1hr,
        "temp_f_iqr_2hr": temp_f_iqr_2hr,
        "temp_f_mean_abs_diff": temp_f_mean_abs_diff,
        "temp_f_mean_abs_diff_30m": temp_f_mean_abs_diff_30m,
        "sub_hour_oscillation_rate": sub_hour_oscillation_rate,
        "max_jump_near_peak": max_jump_near_peak,
        "has_2f_jump_to_peak": has_2f_jump_to_peak,   # 1.0 if consecutive auto obs jumped ≥2°F to max °F
        "max_minus_ma15": max_minus_ma15,
        "max_minus_ma30": max_minus_ma30,
        "max_minus_ma60": max_minus_ma60,
        # --- Peak edge rounding ---
        "peak_edge_f_gap_below": float(peak_edge_f_gap_below),        # °F gap when dropping from max_c to max_c-1 (1 or 2)
        "peak_edge_f_gap_above": float(peak_edge_f_gap_above),  # °F gap when gaining max_c to max_c+1 (1 or 2)
        "dist_c_to_next_f": dist_c_to_next_f,                    # °C margin to round up to naive_f+1
        "dist_c_to_prev_f": dist_c_to_prev_f,                    # °C margin to round down to naive_f-1
        # --- Meta / target ---
        "naive_f": naive_f,
        "possible_f": possible_f,
        "actual_max_f": actual_max_f,
        "offset": max(min(actual_max_f - naive_f, 1), -1),
        "n_auto": n_auto,
    }


# Auto-obs only features (no METAR data).  Used as stage-1 model inputs;
# predicted probabilities feed into the combined model.
FEATURE_COLS = [
    # --- Rounding boundary ---
    "c_to_f_frac", "boundary_distance", "naive_is_high", "n_possible_f",
    "frac_x_dwell", "frac_x_consec",
    # --- Peak shape ---
    "dwell_count", "trans_count", "consec_count",
    "above_prev_count",
    "approach_rate_15m", "approach_rate_30m", "approach_rate_1h",
    "departure_rate_15m", "departure_rate_30m",
    "rise_hrs",
    "readings_at_prev_before",
    "last_max_time", "first_max_time",
    "n_max_blocks", "oscillation_count",
    "single_reading_peak", "daily_range_c",
    "approach_accel",
    "pre_peak_std",
    "pre_peak_momentum",
    # --- First-order derivatives (1hr) ---
    "approach_rate_2h", "departure_rate_1h",
    "max_d1_pre", "min_d1_post", "mean_d1_peak", "std_d1_peak",
    # --- First-order derivatives (30min) ---
    "max_d1_pre_30m", "min_d1_post_30m", "mean_d1_peak_30m", "std_d1_peak_30m",
    # --- Second-order derivatives (1hr) ---
    "pre_peak_d2", "peak_d2", "max_d2_pre", "std_d2_peak", "min_d2_post",
    # --- Second-order derivatives (30min) ---
    "pre_peak_d2_30m", "max_d2_pre_30m", "std_d2_peak_30m", "min_d2_post_30m",
    # --- Meteorological ---
    "wind_at_peak", "wind_calm_frac",
    "wind_gust_range",
    "rh_at_peak", "rh_min_near_peak",
    "dewpoint_at_peak", "dewpoint_depression_at_peak", "dewpoint_change_near_peak",
    "heat_index_at_peak", "heat_index_excess",
    "pressure_at_peak", "pressure_change_near_peak", "pressure_tendency_at_peak",
    "cloud_cover_at_peak", "cloud_variability",
    # --- Absolute temperature ---
    "max_c", "max_c_mod5",
    # --- Temporal ---
    "solar_noon", "peak_minus_solar_noon",
    # --- Volatility ---
    "temp_f_volatility_30m", "temp_f_volatility_1hr", "temp_f_volatility_2hr",
    "temp_f_range_30m", "temp_f_range_1hr",
    "temp_f_iqr_1hr", "temp_f_iqr_2hr",
    "temp_f_mean_abs_diff", "temp_f_mean_abs_diff_30m",
    "sub_hour_oscillation_rate",
    "max_jump_near_peak",
    "has_2f_jump_to_peak",
    "max_minus_ma15", "max_minus_ma30", "max_minus_ma60",
    # --- Peak edge rounding ---
    "peak_edge_f_gap_below",
    "peak_edge_f_gap_above",
    "dist_c_to_next_f",
    "dist_c_to_prev_f",
    # --- METAR-derived (T-group precision) ---
    "has_metar", "metar_above_boundary", "metar_boundary_margin_c",
    "metar_gap_c", "metar_mean_gap_c", "metar_n_above",
    "metar_confirm", "peak_lag_min",
    "frac_x_metar_confirm",
]

METAR_FEATURE_COLS = [
    "has_metar", "metar_above_boundary", "metar_boundary_margin_c",
    "metar_gap_c", "metar_mean_gap_c", "metar_n_above",
    "metar_confirm", "peak_lag_min",
    "frac_x_metar_confirm",
]

DOY_COLS = ["day_of_year_sin", "day_of_year_cos"]

FORECAST_COLS = ["yest_high_vs_naive_f", "forecast_vs_naive_f"]

AUTO_FEATURE_COLS = [c for c in FEATURE_COLS if c not in METAR_FEATURE_COLS]


def tune_histgbm(sites: Optional[List[str]] = None, since: Optional[str] = None):
    """Grid search HistGradientBoostingClassifier hyperparameters."""
    from sklearn.experimental import enable_hist_gradient_boosting  # noqa: F401
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import GridSearchCV, StratifiedKFold

    if sites is None:
        sites = ALL_SITES

    # --- Load cached solar noon data ------------------------------------------
    solar_noon_lookup: Dict[Tuple[str, str], float] = {}
    if os.path.isfile(SOLAR_NOON_CSV):
        sn_df = pd.read_csv(SOLAR_NOON_CSV)
        for _, row in sn_df.iterrows():
            solar_noon_lookup[(row["site"], str(row["date"]))] = float(row["solar_noon_hour"])

    # --- Collect features ----------------------------------------------------
    rows: List[dict] = []
    for site in sites:
        df = load_site_history(site)
        if df.empty:
            continue
        for date, day_df in df.groupby("date"):
            if len(day_df) < 200:
                continue
            day_df = day_df.sort_values("ts").reset_index(drop=True)
            sn = solar_noon_lookup.get((site, str(date)))
            feats = extract_regression_features(day_df, solar_noon_hour=sn)
            if feats is None:
                continue
            feats["site"] = site
            feats["date"] = str(date)
            rows.append(feats)

    rdf = pd.DataFrame(rows)
    if since:
        rdf = rdf[rdf["date"] >= since].reset_index(drop=True)
        print(f"  Filtered to dates >= {since}: {len(rdf)} rows")
    feature_cols = list(AUTO_FEATURE_COLS)
    required_cols = [c for c in feature_cols if c not in _MET_FEATURE_NAMES]
    feature_mask = rdf[required_cols].notna().all(axis=1)
    rdf = rdf[feature_mask].reset_index(drop=True)
    n = len(rdf)

    X = rdf[feature_cols].values
    y = rdf["offset"].values.astype(int)

    print("=" * 78)
    print(f"  HISTGBM HYPERPARAMETER TUNING ({n} rows, {len(feature_cols)} features)")
    print("=" * 78)

    param_grid = {
        "max_iter": [100, 200, 400],
        "max_depth": [5, 7, 9, 12],
        "learning_rate": [0.05, 0.1, 0.2],
        "min_samples_leaf": [10, 20, 40],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    gs = GridSearchCV(
        HistGradientBoostingClassifier(random_state=42, early_stopping=True,
                                       n_iter_no_change=10),
        param_grid, cv=cv, scoring="accuracy", n_jobs=-1, verbose=1,
    )
    gs.fit(X, y)

    print(f"\n  Best params: {gs.best_params_}")
    print(f"  Best CV accuracy: {gs.best_score_:.1%}")

    # Show top 10 results
    results = pd.DataFrame(gs.cv_results_)
    results = results.sort_values("rank_test_score")
    print(f"\n  Top 10 configurations:")
    print(f"  {'Rank':>4s} {'Accuracy':>9s} {'max_iter':>9s} {'depth':>5s} {'lr':>6s} {'min_leaf':>8s}")
    print(f"  {'-'*4} {'-'*9} {'-'*9} {'-'*5} {'-'*6} {'-'*8}")
    for _, row in results.head(10).iterrows():
        print(f"  {int(row['rank_test_score']):4d} {row['mean_test_score']:9.1%}"
              f" {int(row['param_max_iter']):9d}"
              f" {int(row['param_max_depth']):5d}"
              f" {row['param_learning_rate']:6.2f}"
              f" {int(row['param_min_samples_leaf']):8d}")


def _load_regression_data(sites: Optional[List[str]] = None,
                          stage1_only: bool = False,
                          max_rows: Optional[int] = None,
                          since: Optional[str] = None,
                          use_forecast: bool = False) -> pd.DataFrame:
    """Load and prepare feature dataframe for regression backtest."""
    if sites is None:
        sites = ALL_SITES

    solar_noon_lookup: Dict[Tuple[str, str], float] = {}
    if os.path.isfile(SOLAR_NOON_CSV):
        sn_df = pd.read_csv(SOLAR_NOON_CSV)
        for _, row in sn_df.iterrows():
            solar_noon_lookup[(row["site"], str(row["date"]))] = float(row["solar_noon_hour"])
        print(f"  Loaded {len(solar_noon_lookup)} solar noon entries from cache")
    else:
        print(f"  No solar noon cache found — using fallback (12.0)")

    # Build yesterday actual high lookup: {(site, date_str): high_f}
    yest_high_lookup: Dict[Tuple[str, str], float] = {}
    if use_forecast:
        for site in sites:
            df_tmp = load_site_history(site)
            if df_tmp.empty:
                continue
            for date, day_df in df_tmp.groupby("date"):
                if "max_temp_6h_f" in day_df.columns:
                    _m6h = day_df[day_df["max_temp_6h_f"].notna() & (day_df["ts"].dt.hour >= 1)]
                    if not _m6h.empty:
                        yest_high_lookup[(site, str(date))] = round(float(
                            pd.to_numeric(_m6h["max_temp_6h_f"], errors="coerce").max()))

    # Load forecast highs: {(site, date_str): forecast_high_f}
    forecast_lookup: Dict[Tuple[str, str], float] = {}
    if use_forecast:
        fcst_csv = os.path.join(os.path.dirname(SOLAR_NOON_CSV), "forecast_highs.csv")
        if os.path.isfile(fcst_csv):
            fcst_df = pd.read_csv(fcst_csv)
            for _, row in fcst_df.iterrows():
                forecast_lookup[(row["site"], str(row["date"]))] = float(row["forecast_high_f"])
            print(f"  Loaded {len(forecast_lookup)} forecast high entries")
        else:
            print(f"  WARNING: forecast_highs.csv not found — forecast_vs_naive_f will be NaN")

    rows: List[dict] = []
    for site in sites:
        df = load_site_history(site)
        if df.empty:
            continue
        sorted_dates = sorted(df["date"].unique())
        date_to_prev = {d: sorted_dates[i - 1] if i > 0 else None
                        for i, d in enumerate(sorted_dates)}
        for date, day_df in df.groupby("date"):
            if len(day_df) < 200:
                continue
            day_df = day_df.sort_values("ts").reset_index(drop=True)
            sn = solar_noon_lookup.get((site, str(date)))
            feats = extract_regression_features(day_df, solar_noon_hour=sn)
            if feats is None:
                continue
            feats["site"] = site
            feats["date"] = str(date)

            if use_forecast:
                naive_f = feats.get("naive_f")
                # Yesterday's actual high relative to today's naive_f
                prev_date = date_to_prev.get(date)
                yest_high = yest_high_lookup.get((site, str(prev_date))) if prev_date else None
                feats["yest_high_vs_naive_f"] = float(yest_high - naive_f) if yest_high is not None and naive_f is not None else np.nan
                # Forecast high relative to today's naive_f
                fcst_high = forecast_lookup.get((site, str(date)))
                feats["forecast_vs_naive_f"] = float(round(fcst_high) - naive_f) if fcst_high is not None and naive_f is not None else np.nan

            rows.append(feats)

    rdf = pd.DataFrame(rows)
    if since:
        rdf = rdf[rdf["date"] >= since].reset_index(drop=True)
        print(f"  Filtered to dates >= {since}: {len(rdf)} rows")
    n = len(rdf)

    sn_na = int((rdf["solar_noon"] == 12.0).sum())
    print(f"  Solar noon: {n - sn_na}/{n} from API, {sn_na} fallback (NA)")

    feature_cols = list(AUTO_FEATURE_COLS if stage1_only else FEATURE_COLS)
    # Only require non-NaN for core features; met features can be NaN
    # (HistGBM handles NaN natively)
    required_cols = [c for c in feature_cols if c not in _MET_FEATURE_NAMES]
    feature_mask = rdf[required_cols].notna().all(axis=1)
    n_dropped = int((~feature_mask).sum())
    rdf = rdf[feature_mask].reset_index(drop=True)
    print(f"  Dropped {n_dropped} rows with NA features, {len(rdf)} remaining")

    if max_rows and len(rdf) > max_rows:
        rdf = rdf.sample(n=max_rows, random_state=42).reset_index(drop=True)
        print(f"  Dry-run: sampled {max_rows} rows")

    return rdf


def _plot_pr_curves(pr_data: list, stage1_only: bool = False):
    """Plot precision-recall curves from LOSO test predictions.

    pr_data: list of dicts with keys:
        site, y_test, s1_proba, s1_classes, [s2_proba, s2_classes]
    Plots upper (offset>=0) and middle (offset=+1) P/R curves.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, average_precision_score

    def _get_score(proba, classes, mode):
        """Convert 3-class probabilities to binary score."""
        cls_list = list(classes)
        if mode == "upper":
            # P(offset >= 0) = P(0) + P(+1)
            idx_0 = cls_list.index(0) if 0 in cls_list else None
            idx_p1 = cls_list.index(1) if 1 in cls_list else None
            score = np.zeros(len(proba))
            if idx_0 is not None:
                score += proba[:, idx_0]
            if idx_p1 is not None:
                score += proba[:, idx_p1]
            return score
        elif mode == "middle":
            # P(offset = +1)
            idx_p1 = cls_list.index(1) if 1 in cls_list else None
            if idx_p1 is not None:
                return proba[:, idx_p1]
            return np.zeros(len(proba))
        elif mode == "temp_m1":
            idx = cls_list.index(-1) if -1 in cls_list else None
            return proba[:, idx] if idx is not None else np.zeros(len(proba))
        elif mode == "temp_0":
            idx = cls_list.index(0) if 0 in cls_list else None
            return proba[:, idx] if idx is not None else np.zeros(len(proba))
        elif mode == "temp_p1":
            idx = cls_list.index(1) if 1 in cls_list else None
            return proba[:, idx] if idx is not None else np.zeros(len(proba))

    out_dir = project_path("charts")
    os.makedirs(out_dir, exist_ok=True)

    stages = ["S1"]
    if not stage1_only:
        stages.append("S2")

    for mode, mode_label in [("upper", "Upper (offset≥0)"), ("middle", "Middle (offset=+1)"),
                                ("temp_m1", "Exact: offset=-1"), ("temp_0", "Exact: offset=0"), ("temp_p1", "Exact: offset=+1")]:
        fig, axes = plt.subplots(1, len(stages), figsize=(7 * len(stages), 6))
        if len(stages) == 1:
            axes = [axes]

        for ax, stage in zip(axes, stages):
            # Per-site curves
            all_y = []
            all_scores = []

            for entry in pr_data:
                y_true = entry["y_test"]
                if mode == "upper":
                    y_bin = (y_true >= 0).astype(int)
                elif mode == "middle":
                    y_bin = (y_true > 0).astype(int)
                elif mode == "temp_m1":
                    y_bin = (y_true == -1).astype(int)
                elif mode == "temp_0":
                    y_bin = (y_true == 0).astype(int)
                elif mode == "temp_p1":
                    y_bin = (y_true == 1).astype(int)

                proba_key = "s1_proba" if stage == "S1" else "s2_proba"
                classes_key = "s1_classes" if stage == "S1" else "s2_classes"
                if proba_key not in entry:
                    continue

                scores = _get_score(entry[proba_key], entry[classes_key], mode)
                site = entry["site"]

                all_y.append(y_bin)
                all_scores.append(scores)

                if len(y_bin) >= 20:
                    prec, rec, _ = precision_recall_curve(y_bin, scores)
                    ax.plot(rec, prec, alpha=0.3, linewidth=1, label=f"{site} (n={len(y_bin)})")

            # Combined curve
            if all_y:
                y_all = np.concatenate(all_y)
                s_all = np.concatenate(all_scores)
                prec, rec, _ = precision_recall_curve(y_all, s_all)
                ap = average_precision_score(y_all, s_all)
                ax.plot(rec, prec, color="black", linewidth=2.5,
                        label=f"Combined (AP={ap:.3f}, n={len(y_all)})")

                # Baseline: prevalence
                prevalence = y_all.mean()
                ax.axhline(prevalence, color="gray", linestyle="--", linewidth=1,
                           label=f"Baseline ({prevalence:.1%})")

            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
            ax.set_title(f"{stage} — {mode_label}")
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            ax.legend(fontsize=7, loc="lower left")
            ax.grid(True, alpha=0.3)

        fig.tight_layout()
        path = os.path.join(out_dir, f"pr_curve_{mode}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved {path}")


def backtest_regression(sites: Optional[List[str]] = None,
                        stage1_only: bool = False,
                        rdf: Optional[pd.DataFrame] = None):
    """Train 3-class offset model, eval on exact temp + upper + middle brackets.

    Model always predicts offset {-1, 0, +1}. Evals remap predictions:
      - Exact: 3-class accuracy
      - Upper: YES (offset >= 0) vs NO (offset = -1)
      - Middle: UPPER (offset = +1) vs LOWER (offset <= 0)
    """
    from sklearn.experimental import enable_hist_gradient_boosting  # noqa: F401
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import accuracy_score

    if rdf is None:
        rdf = _load_regression_data(sites, stage1_only)

    n = len(rdf)
    y_3class = rdf["offset"].values.astype(int)
    site_labels = rdf["site"].values
    auto_cols = list(AUTO_FEATURE_COLS)

    def _to_upper(y_raw):
        return (y_raw >= 0).astype(int)

    def _to_middle(y_raw):
        return (y_raw > 0).astype(int)

    # --- Class distribution --------------------------------------------------
    print("=" * 78)
    print(f"  3-CLASS OFFSET MODEL → eval on exact / upper / middle")
    print("=" * 78)
    print(f"  Total days: {n}  |  Sites: {len(set(site_labels))}")
    for cls in [-1, 0, 1]:
        cnt = int(np.sum(y_3class == cls))
        print(f"    offset={cls:+d}: {cnt:5d} ({cnt/n*100:5.1f}%)")

    # --- Baselines -----------------------------------------------------------
    always_0 = int(np.sum(y_3class == 0))
    y_upper = _to_upper(y_3class)
    y_middle = _to_middle(y_3class)
    print(f"\n  Baselines:")
    print(f"    Exact (always-0):   {always_0/n:.1%}")
    print(f"    Upper (always-YES): {int(np.sum(y_upper == 1))/n:.1%}")
    print(f"    Middle (always-0):  {int(np.sum(y_middle == 0))/n:.1%}")

    X_auto = rdf[auto_cols].values

    # --- SHAP feature importances (fit on full data) --------------------------
    import shap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    chart_dir = project_path("charts")
    os.makedirs(chart_dir, exist_ok=True)

    print(f"\n{'=' * 78}")
    print(f"  SHAP FEATURE IMPORTANCES")
    print(f"{'=' * 78}")

    s1_model = HistGradientBoostingClassifier(
        max_iter=200, max_depth=7, learning_rate=0.05,
        min_samples_leaf=40, random_state=42)
    s1_model.fit(X_auto, y_3class)
    s1_explainer = shap.TreeExplainer(s1_model)
    s1_shap = s1_explainer(pd.DataFrame(X_auto, columns=auto_cols))

    # Mean |SHAP| across all classes
    s1_mean_abs = np.mean(np.mean(np.abs(s1_shap.values), axis=2), axis=0)
    sorted_idx = np.argsort(s1_mean_abs)[::-1]
    print(f"\n  Stage 1 (auto-only, mean |SHAP|):")
    for i in sorted_idx[:15]:
        print(f"    {auto_cols[i]:30s}: {s1_mean_abs[i]:.4f}")

    # S1 SHAP summary_plot per class
    colors = {-1: "#F44336", 0: "#FFC107", 1: "#4CAF50"}
    class_labels = {-1: "minus1", 0: "zero", 1: "plus1"}
    for ci, cls in enumerate(s1_model.classes_):
        plt.figure(figsize=(10, 8))
        shap.summary_plot(s1_shap.values[:, :, ci], pd.DataFrame(X_auto, columns=auto_cols),
                          max_display=20, show=False, plot_type="dot")
        plt.title(f"S1 SHAP — offset={cls:+d}")
        plt.tight_layout()
        path = os.path.join(chart_dir, f"shap_s1_offset_{class_labels[cls]}.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  Saved {path}")

    # S1 bar chart (mean |SHAP| across classes)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(np.mean(np.abs(s1_shap.values), axis=2),
                      pd.DataFrame(X_auto, columns=auto_cols),
                      max_display=20, show=False, plot_type="bar")
    plt.title("S1 SHAP — mean |SHAP| (all classes)")
    plt.tight_layout()
    path = os.path.join(chart_dir, "shap_s1_bar.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")

    # S1 per-feature dependence plots (top 15 by mean |SHAP|)
    s1_feat_dir = os.path.join(chart_dir, "shap_s1_features")
    os.makedirs(s1_feat_dir, exist_ok=True)
    X_auto_df = pd.DataFrame(X_auto, columns=auto_cols)
    for rank, fi in enumerate(sorted_idx[:15]):
        feat_name = auto_cols[fi]
        for ci, cls in enumerate(s1_model.classes_):
            plt.figure(figsize=(8, 5))
            shap.dependence_plot(feat_name, s1_shap.values[:, :, ci],
                                 X_auto_df, show=False)
            plt.title(f"S1 SHAP — {feat_name} (offset={cls:+d})")
            plt.tight_layout()
            path = os.path.join(s1_feat_dir,
                                f"{rank:02d}_{feat_name}_{class_labels[cls]}.png")
            plt.savefig(path, dpi=120)
            plt.close()
    print(f"  Saved {len(sorted_idx[:15]) * 3} per-feature plots to {s1_feat_dir}")

    if not stage1_only:
        # Fit S2 on full data for SHAP
        s1_classes = s1_model.classes_
        prob_col_names = [f"auto_prob_{c:+d}" for c in s1_classes]
        auto_proba_full = s1_model.predict_proba(X_auto)
        rdf_s2 = rdf.copy()
        for i, col_name in enumerate(prob_col_names):
            rdf_s2[col_name] = auto_proba_full[:, i]

        metar_confirm = rdf_s2["metar_confirm"].values
        auto_prob_p1 = rdf_s2.get("auto_prob_+1", pd.Series(np.zeros(n))).values
        rdf_s2["consensus"] = auto_prob_p1 * metar_confirm
        rdf_s2["auto_metar_divergence"] = auto_prob_p1 - metar_confirm

        metar_raw_cols = ["metar_confirm", "metar_gap_c"]
        consensus_cols = ["consensus", "auto_metar_divergence"]
        stage2_cols = prob_col_names + metar_raw_cols + consensus_cols
        X_s2 = rdf_s2[stage2_cols].values

        s2_model = HistGradientBoostingClassifier(
            max_iter=200, max_depth=5, learning_rate=0.1,
            min_samples_leaf=5, random_state=42)
        s2_model.fit(X_s2, y_3class)

        s2_explainer = shap.TreeExplainer(s2_model)
        s2_shap = s2_explainer(pd.DataFrame(X_s2, columns=stage2_cols))

        s2_mean_abs = np.mean(np.mean(np.abs(s2_shap.values), axis=2), axis=0)
        sorted_idx = np.argsort(s2_mean_abs)[::-1]
        print(f"\n  Stage 2 (S1 probs + consensus, mean |SHAP|):")
        for i in sorted_idx[:len(stage2_cols)]:
            print(f"    {stage2_cols[i]:30s}: {s2_mean_abs[i]:.4f}")

        for ci, cls in enumerate(s2_model.classes_):
            plt.figure(figsize=(8, 5))
            shap.summary_plot(s2_shap.values[:, :, ci], pd.DataFrame(X_s2, columns=stage2_cols),
                              max_display=10, show=False, plot_type="dot")
            plt.title(f"S2 SHAP — offset={cls:+d}")
            plt.tight_layout()
            path = os.path.join(chart_dir, f"shap_s2_offset_{class_labels[cls]}.png")
            plt.savefig(path, dpi=150)
            plt.close()
            print(f"  Saved {path}")

        plt.figure(figsize=(8, 5))
        shap.summary_plot(np.mean(np.abs(s2_shap.values), axis=2),
                          pd.DataFrame(X_s2, columns=stage2_cols),
                          max_display=10, show=False, plot_type="bar")
        plt.title("S2 SHAP — mean |SHAP| (all classes)")
        plt.tight_layout()
        path = os.path.join(chart_dir, "shap_s2_bar.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  Saved {path}")

        # S2 per-feature dependence plots
        s2_feat_dir = os.path.join(chart_dir, "shap_s2_features")
        os.makedirs(s2_feat_dir, exist_ok=True)
        X_s2_df = pd.DataFrame(X_s2, columns=stage2_cols)
        for rank, fi in enumerate(sorted_idx[:len(stage2_cols)]):
            feat_name = stage2_cols[fi]
            for ci, cls in enumerate(s2_model.classes_):
                plt.figure(figsize=(8, 5))
                shap.dependence_plot(feat_name, s2_shap.values[:, :, ci],
                                     X_s2_df, show=False)
                plt.title(f"S2 SHAP — {feat_name} (offset={cls:+d})")
                plt.tight_layout()
                path = os.path.join(s2_feat_dir,
                                    f"{rank:02d}_{feat_name}_{class_labels[cls]}.png")
                plt.savefig(path, dpi=120)
                plt.close()
        print(f"  Saved {len(stage2_cols) * 3} per-feature plots to {s2_feat_dir}")

    # --- Leave-One-Site-Out CV (Kalshi sites only) ----------------------------
    # Test set: only days from 2026-01-01 onward
    test_date_cutoff = "2026-01-01"
    is_test_eligible = rdf["date"].astype(str) >= test_date_cutoff
    n_test_eligible = int(is_test_eligible.sum())

    print(f"\n{'=' * 78}")
    loso_label = "LEAVE-ONE-SITE-OUT" + (" (stage1-only)" if stage1_only else " (2-stage)")
    print(f"  {loso_label}")
    print(f"  Training sites ({len(TRAINING_SITES)}) always in training set")
    print(f"  Test set: {n_test_eligible} days from {test_date_cutoff} onward")
    print(f"{'=' * 78}")

    kalshi_sites_in_data = sorted(s for s in set(site_labels) if s in ALL_SITES)
    if stage1_only:
        print(f"  {'Site':<8s} {'N':>5s} {'A0%':>6s} {'Exact':>7s} {'Upper':>7s} {'Middle':>7s}")
        print(f"  {'-'*8} {'-'*5} {'-'*6} {'-'*7} {'-'*7} {'-'*7}")
    else:
        print(f"  {'Site':<8s} {'N':>5s} {'A0%':>6s} {'S1 Ex':>7s} {'S1 Up':>7s} {'S1 Mi':>7s} {'S2 Ex':>7s} {'S2 Up':>7s} {'S2 Mi':>7s}")
        print(f"  {'-'*8} {'-'*5} {'-'*6} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")

    loso_totals = {"n": 0, "a0": 0, "s1_exact": 0, "s1_upper": 0, "s1_middle": 0,
                   "s2_exact": 0, "s2_upper": 0, "s2_middle": 0}

    # Collect per-site LOSO probabilities for P/R curves
    pr_data = []  # list of {site, y_test, s1_proba, s2_proba}

    for site in kalshi_sites_in_data:
        test_mask = (site_labels == site) & is_test_eligible.values
        train_mask = ~test_mask
        n_test = int(test_mask.sum())
        if n_test == 0:
            continue

        y_test = y_3class[test_mask]
        y_train = y_3class[train_mask]
        a0_correct = int(np.sum(y_test == 0))
        a0_acc = a0_correct / n_test

        # S1
        X_auto_train = X_auto[train_mask]
        X_auto_test = X_auto[test_mask]

        s1 = HistGradientBoostingClassifier(
            max_iter=200, max_depth=5, learning_rate=0.1,
            min_samples_leaf=5, random_state=42)
        s1.fit(X_auto_train, y_train)
        s1_pred = s1.predict(X_auto_test)
        s1_proba = s1.predict_proba(X_auto_test)  # (n_test, 3)

        s1_ex = accuracy_score(y_test, s1_pred)
        s1_up = accuracy_score(_to_upper(y_test), _to_upper(s1_pred))
        s1_mi = accuracy_score(_to_middle(y_test), _to_middle(s1_pred))

        loso_totals["n"] += n_test
        loso_totals["a0"] += a0_correct
        loso_totals["s1_exact"] += int(np.sum(s1_pred == y_test))
        loso_totals["s1_upper"] += int(np.sum(_to_upper(s1_pred) == _to_upper(y_test)))
        loso_totals["s1_middle"] += int(np.sum(_to_middle(s1_pred) == _to_middle(y_test)))

        site_pr = {"site": site, "y_test": y_test, "s1_proba": s1_proba,
                   "s1_classes": s1.classes_}

        if stage1_only:
            pr_data.append(site_pr)
            print(f"  {site:<8s} {n_test:5d} {a0_acc:5.1%} {s1_ex:6.1%} {s1_up:6.1%} {s1_mi:6.1%}")
        else:
            # S2
            s1_classes_loso = s1.classes_
            prob_names = [f"auto_prob_{c:+d}" for c in s1_classes_loso]

            train_proba = cross_val_predict(
                s1, X_auto_train, y_train,
                cv=min(5, len(set(y_train))),
                method='predict_proba')
            s1.fit(X_auto_train, y_train)

            rdf_train = rdf.loc[train_mask].copy()
            for i, col_name in enumerate(prob_names):
                rdf_train[col_name] = train_proba[:, i]
            mc_train = rdf_train["metar_confirm"].values
            ap1_train = rdf_train.get("auto_prob_+1", pd.Series(np.zeros(int(train_mask.sum())))).values
            rdf_train["consensus"] = ap1_train * mc_train
            rdf_train["auto_metar_divergence"] = ap1_train - mc_train

            test_proba = s1.predict_proba(X_auto_test)
            rdf_test = rdf.loc[test_mask].copy()
            for i, col_name in enumerate(prob_names):
                rdf_test[col_name] = test_proba[:, i]
            mc_test = rdf_test["metar_confirm"].values
            ap1_test = rdf_test.get("auto_prob_+1", pd.Series(np.zeros(n_test))).values
            rdf_test["consensus"] = ap1_test * mc_test
            rdf_test["auto_metar_divergence"] = ap1_test - mc_test

            s2_cols = prob_names + ["metar_confirm", "metar_gap_c", "consensus", "auto_metar_divergence"]
            X_s2_train = rdf_train[s2_cols].values
            X_s2_test = rdf_test[s2_cols].values

            s2 = HistGradientBoostingClassifier(
                max_iter=200, max_depth=5, learning_rate=0.1,
                min_samples_leaf=5, random_state=42)
            s2.fit(X_s2_train, y_train)
            s2_pred = s2.predict(X_s2_test)
            s2_proba = s2.predict_proba(X_s2_test)

            s2_ex = accuracy_score(y_test, s2_pred)
            s2_up = accuracy_score(_to_upper(y_test), _to_upper(s2_pred))
            s2_mi = accuracy_score(_to_middle(y_test), _to_middle(s2_pred))

            loso_totals["s2_exact"] += int(np.sum(s2_pred == y_test))
            loso_totals["s2_upper"] += int(np.sum(_to_upper(s2_pred) == _to_upper(y_test)))
            loso_totals["s2_middle"] += int(np.sum(_to_middle(s2_pred) == _to_middle(y_test)))

            site_pr["s2_proba"] = s2_proba
            site_pr["s2_classes"] = s2.classes_
            pr_data.append(site_pr)

            print(f"  {site:<8s} {n_test:5d} {a0_acc:5.1%} {s1_ex:6.1%} {s1_up:6.1%} {s1_mi:6.1%} {s2_ex:6.1%} {s2_up:6.1%} {s2_mi:6.1%}")

    nt = loso_totals["n"]
    if nt > 0:
        print(f"\n  LOSO totals (n={nt}):")
        print(f"    Always-0:  {loso_totals['a0']/nt:.1%}")
        print(f"    S1 exact:  {loso_totals['s1_exact']/nt:.1%}  upper: {loso_totals['s1_upper']/nt:.1%}  middle: {loso_totals['s1_middle']/nt:.1%}")
        if not stage1_only:
            print(f"    S2 exact:  {loso_totals['s2_exact']/nt:.1%}  upper: {loso_totals['s2_upper']/nt:.1%}  middle: {loso_totals['s2_middle']/nt:.1%}")

    # --- Precision-Recall curves (LOSO test set) -----------------------------
    if pr_data:
        _plot_pr_curves(pr_data, stage1_only)


def backtest_ml_model(sites: Optional[List[str]] = None):
    """Train and evaluate ML models for naive=low settlement prediction.

    Extracts features from all site-days, trains logistic regression and
    decision tree classifiers with cross-validation, and prints results.
    """
    from sklearn.experimental import enable_hist_gradient_boosting  # noqa: F401
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import cross_val_score, cross_val_predict

    if sites is None:
        sites = ALL_SITES

    # --- Collect features ---------------------------------------------------
    rows = []
    for site in sites:
        df = load_site_history(site)
        if df.empty:
            continue
        for date, day_df in df.groupby("date"):
            if len(day_df) < 200:
                continue
            day_df = day_df.sort_values("ts").reset_index(drop=True)
            feats = extract_features(day_df)
            if feats is None:
                continue
            feats["site"] = site
            feats["date"] = str(date)
            rows.append(feats)

    rdf = pd.DataFrame(rows)
    y = rdf["is_high"].values
    n = len(rdf)
    n_high = y.sum()

    print("=" * 78)
    print("  ML MODEL EXPLORATION (naive=low cases only)")
    print("=" * 78)
    print(f"  Total days: {n}  |  f_high: {n_high} ({n_high/n*100:.1f}%)"
          f"  |  f_low: {n - n_high} ({(n - n_high)/n*100:.1f}%)")

    # --- Feature correlations -----------------------------------------------
    feature_cols = [
        "dwell_ratio", "trans_ratio", "consec_ratio",
        "time_at_max_ratio", "frac_above_prev",
        "has_metar", "metar_above_boundary", "metar_boundary_margin_c",
        "metar_gap_c", "metar_mean_gap_c", "metar_n_above",
        "metar_confirm", "peak_lag_min", "consensus", "auto_metar_divergence",
    ]
    print(f"\n  Feature correlations with is_high:")
    for f in feature_cols:
        corr = rdf[f].corr(rdf["is_high"])
        m0 = rdf.loc[rdf["is_high"] == 0, f].mean()
        m1 = rdf.loc[rdf["is_high"] == 1, f].mean()
        print(f"    {f:20s}: corr={corr:+.3f}"
              f"  mean(low)={m0:7.2f}  mean(high)={m1:7.2f}")

    # --- Hand-crafted rules -------------------------------------------------
    print(f"\n{'=' * 78}")
    print(f"  RULE-BASED MODELS")
    print(f"{'=' * 78}")

    def eval_rule(name, pred_h):
        tp = (pred_h & (y == 1)).sum()
        fp = (pred_h & (y == 0)).sum()
        fn = (~pred_h & (y == 1)).sum()
        tn = (~pred_h & (y == 0)).sum()
        acc = (tp + tn) / n
        ov = 588 + 119 + tp + tn
        print(f"    {name:45s}: {acc:.1%}  tp={tp:3d} fp={fp:3d}"
              f" fn={fn:3d} → {ov}/1290 ({ov/1290*100:.1f}%)")

    eval_rule("always f_low (baseline)",
              pd.Series([False] * n))
    eval_rule("ratio>0.09 & trans>0.42 (AND)",
              (rdf["dwell_ratio"] > 0.09) & (rdf["trans_ratio"] > 0.42))
    eval_rule("1.1*trans + 1.7*ratio > 0.70 (linear)",
              1.1 * rdf["trans_ratio"] + 1.7 * rdf["dwell_ratio"] > 0.70)

    # --- HistGBM ------------------------------------------------------------
    print(f"\n{'=' * 78}")
    print(f"  HIST GRADIENT BOOSTING")
    print(f"{'=' * 78}")

    X = rdf[feature_cols].fillna(0).values

    hgb = HistGradientBoostingClassifier(
        max_iter=200, max_depth=5, learning_rate=0.1,
        min_samples_leaf=5, random_state=42,
    )
    scores = cross_val_score(hgb, X, y, cv=5, scoring="accuracy")
    print(f"    5-fold CV: {scores.mean():.1%} ± {scores.std():.1%}")

    # HistGBM with tuned probability threshold
    proba_cv = cross_val_predict(hgb, X, y, cv=5, method="predict_proba")[:, 1]
    best_t, best_acc = 0.5, 0
    for t100 in range(20, 80):
        t = t100 / 100
        pred = (proba_cv >= t).astype(int)
        acc = (pred == y).mean()
        if acc > best_acc:
            best_t, best_acc = t, acc
    print(f"    Best probability threshold: {best_t:.2f} → {best_acc:.1%}")

    # --- Leave-one-site-out for best linear rule ----------------------------
    print(f"{'=' * 78}")
    print(f"  LEAVE-ONE-SITE-OUT: 1.1*trans + 1.7*ratio > 0.70")
    print(f"{'=' * 78}")

    score = 1.1 * rdf["trans_ratio"] + 1.7 * rdf["dwell_ratio"]
    total_correct = 0
    for site in sorted(rdf["site"].unique()):
        mask = rdf["site"] == site
        pred_h = (score > 0.70) & mask
        pred_l = (score <= 0.70) & mask
        tp = (pred_h & (rdf["is_high"] == 1)).sum()
        fp = (pred_h & (rdf["is_high"] == 0)).sum()
        fn = (pred_l & (rdf["is_high"] == 1)).sum()
        tn = (pred_l & (rdf["is_high"] == 0)).sum()
        n_site = mask.sum()
        correct = tp + tn
        total_correct += correct
        print(f"    {site}: {correct}/{n_site} ({correct/n_site*100:.0f}%)"
              f"  tp={tp} fp={fp} fn={fn}")
    print(f"    Overall: {total_correct}/{n} ({total_correct/n*100:.1f}%)")


def verify_bracket_model(
    sites: Optional[List[str]] = None,
    model_path: Optional[str] = None,
    stage1_only: bool = False,
    since: Optional[str] = None,
):
    """Load a trained bracket_model and verify accuracy on historical data.

    Evaluates all three metrics in one pass:
      - Exact: 3-class offset {-1, 0, +1}
      - Upper: bracket [naive_f, naive_f+1] YES (offset >= 0) vs NO
      - Middle: upper (offset = +1) vs lower (offset <= 0)

    stage1_only:
      If True, use stage-1 (auto-only GBM) probabilities instead of
      the full 2-stage pipeline.
    """
    from weather.bracket_model import load_model, get_probability

    if sites is None:
        sites = list(ALL_SITES)

    model = load_model(model_path)
    prob_key = "stage1_prob" if stage1_only else "prob"
    stage_label = "stage-1 only (auto GBM)" if stage1_only else "full 2-stage"
    print(f"  Loaded model: {model.get('trained_at', '?')} "
          f"({model.get('n_rows', '?')} rows, {len(model.get('sites', []))} sites)")
    print(f"  Pipeline: {stage_label}")
    if since:
        print(f"  Since: {since}")

    # Load solar noon cache
    solar_noon_lookup: Dict[Tuple[str, str], float] = {}
    if os.path.isfile(SOLAR_NOON_CSV):
        sn_df = pd.read_csv(SOLAR_NOON_CSV)
        for _, row in sn_df.iterrows():
            solar_noon_lookup[(row["site"], str(row["date"]))] = float(row["solar_noon_hour"])

    # Evaluate each site-day
    results: List[dict] = []
    for site in sites:
        df = load_site_history(site)
        if df.empty:
            continue
        for date, day_df in df.groupby("date"):
            if since and str(date) < since:
                continue
            if len(day_df) < 200:
                continue
            day_df = day_df.sort_values("ts").reset_index(drop=True)
            sn = solar_noon_lookup.get((site, str(date)))
            feats = extract_regression_features(day_df, solar_noon_hour=sn)
            if feats is None:
                continue

            max_c = feats.get("max_c")
            if max_c is None:
                continue
            naive_f = round(float(max_c) * 9.0 / 5.0 + 32.0)
            actual_offset = feats["offset"]

            # 3-class: single-degree brackets isolating each offset
            brackets = [
                (naive_f - 1, naive_f - 1),
                (naive_f, naive_f),
                (naive_f + 1, naive_f + 1),
            ]
            br_results = get_probability(model, feats, brackets)
            offset_probs = {}
            for br in br_results:
                low, _ = br["bracket"]
                if low == naive_f - 1:
                    offset_probs[-1] = br[prob_key]
                elif low == naive_f:
                    offset_probs[0] = br[prob_key]
                elif low == naive_f + 1:
                    offset_probs[1] = br[prob_key]

            pred_offset = max(offset_probs, key=lambda k: (offset_probs[k], -abs(k)))

            # Derive upper/middle from offset probs
            prob_upper_yes = offset_probs.get(0, 0.0) + offset_probs.get(1, 0.0)
            actual_upper = int(actual_offset >= 0)
            pred_upper = int(prob_upper_yes >= 0.5)

            prob_middle_up = offset_probs.get(1, 0.0)
            prob_middle_lo = offset_probs.get(-1, 0.0) + offset_probs.get(0, 0.0)
            actual_middle = int(actual_offset > 0)
            pred_middle = int(prob_middle_up > prob_middle_lo)

            results.append({
                "site": site, "date": str(date),
                "actual_offset": actual_offset,
                "pred_offset": pred_offset,
                "exact_correct": pred_offset == actual_offset,
                "prob_minus1": offset_probs.get(-1, 0.0),
                "prob_0": offset_probs.get(0, 0.0),
                "prob_plus1": offset_probs.get(1, 0.0),
                "actual_upper": actual_upper,
                "pred_upper": pred_upper,
                "upper_correct": pred_upper == actual_upper,
                "actual_middle": actual_middle,
                "pred_middle": pred_middle,
                "middle_correct": pred_middle == actual_middle,
            })

    if not results:
        print("  No results generated.")
        return

    rdf = pd.DataFrame(results)
    n = len(rdf)

    # === Header ===============================================================
    s1_tag = " (STAGE-1 ONLY)" if stage1_only else ""
    print(f"\n{'=' * 78}")
    print(f"  BRACKET MODEL VERIFICATION{s1_tag}")
    print(f"{'=' * 78}")
    print(f"  Total days: {n}  |  Sites: {rdf['site'].nunique()}")
    for cls in [-1, 0, 1]:
        cnt = int((rdf["actual_offset"] == cls).sum())
        print(f"    offset={cls:+d}: {cnt:5d} ({cnt / n * 100:5.1f}%)")

    # === Summary table ========================================================
    exact_correct = int(rdf["exact_correct"].sum())
    upper_correct = int(rdf["upper_correct"].sum())
    middle_correct = int(rdf["middle_correct"].sum())
    always_0 = int((rdf["actual_offset"] == 0).sum())
    always_upper_yes = int((rdf["actual_upper"] == 1).sum())
    always_middle_lo = int((rdf["actual_middle"] == 0).sum())

    print(f"\n  {'Metric':<12s} {'Accuracy':>10s} {'Baseline':>10s} {'Lift':>8s}")
    print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*8}")
    print(f"  {'Exact':<12s} {exact_correct / n:>10.1%} {always_0 / n:>10.1%} "
          f"{(exact_correct - always_0) / n:>+8.1%}")
    print(f"  {'Upper':<12s} {upper_correct / n:>10.1%} "
          f"{max(always_upper_yes, n - always_upper_yes) / n:>10.1%} "
          f"{(upper_correct - max(always_upper_yes, n - always_upper_yes)) / n:>+8.1%}")
    print(f"  {'Middle':<12s} {middle_correct / n:>10.1%} "
          f"{max(always_middle_lo, n - always_middle_lo) / n:>10.1%} "
          f"{(middle_correct - max(always_middle_lo, n - always_middle_lo)) / n:>+8.1%}")

    # === Per-offset exact accuracy ============================================
    print(f"\n  Per-offset accuracy:")
    print(f"  {'Offset':>8s} {'N':>6s} {'Correct':>8s} {'Acc':>7s}")
    print(f"  {'-'*8} {'-'*6} {'-'*8} {'-'*7}")
    for offset in [-1, 0, 1]:
        mask = rdf["actual_offset"] == offset
        cnt = int(mask.sum())
        if cnt == 0:
            continue
        correct = int(rdf.loc[mask, "exact_correct"].sum())
        print(f"  {offset:>+8d} {cnt:>6d} {correct:>8d} {correct / cnt:>7.1%}")

    # === Confusion matrix =====================================================
    print(f"\n  Confusion matrix (rows=actual, cols=predicted):")
    print(f"  {'':>8s}  pred-1  pred 0  pred+1")
    for actual in [-1, 0, 1]:
        row = []
        for pred in [-1, 0, 1]:
            cnt = int(((rdf["actual_offset"] == actual) & (rdf["pred_offset"] == pred)).sum())
            row.append(f"{cnt:>6d}")
        print(f"  actual{actual:+d}  {'  '.join(row)}")

    # === Per-site accuracy ====================================================
    print(f"\n  Per-site accuracy:")
    print(f"  {'Site':<8s} {'N':>5s} {'Exact':>7s} {'Upper':>7s} {'Middle':>7s} {'A0%':>6s}")
    print(f"  {'-'*8} {'-'*5} {'-'*7} {'-'*7} {'-'*7} {'-'*6}")
    for site in sorted(rdf["site"].unique()):
        mask = rdf["site"] == site
        site_df = rdf[mask]
        n_s = len(site_df)
        ex = int(site_df["exact_correct"].sum())
        up = int(site_df["upper_correct"].sum())
        mi = int(site_df["middle_correct"].sum())
        a0 = int((site_df["actual_offset"] == 0).sum())
        print(f"  {site:<8s} {n_s:>5d} {ex / n_s:>7.1%} {up / n_s:>7.1%} "
              f"{mi / n_s:>7.1%} {a0 / n_s:>6.1%}")

    # === Calibration ==========================================================
    mean_correct_prob = 0.0
    for _, row in rdf.iterrows():
        ao = row["actual_offset"]
        if ao == -1:
            mean_correct_prob += row["prob_minus1"]
        elif ao == 0:
            mean_correct_prob += row["prob_0"]
        else:
            mean_correct_prob += row["prob_plus1"]
    mean_correct_prob /= n
    print(f"\n  Avg predicted prob for correct offset: {mean_correct_prob:.3f}")


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Backtest settlement prediction (auto-obs only)")
    parser.add_argument("--site", type=str, default=None,
                        help="Comma-separated ICAO codes (default: all)")
    parser.add_argument("--ml", action="store_true",
                        help="Run ML model exploration instead of backtest")
    parser.add_argument("--regression", action="store_true",
                        help="Run regression model to predict settlement °F")
    parser.add_argument("--bracket-upper", action="store_true",
                        help="Binary bracket mode: YES (offset>=0) vs NO (offset=-1)")
    parser.add_argument("--bracket-middle", action="store_true",
                        help="Binary bracket mode: UPPER (offset=+1) vs LOWER (offset<=0)")
    parser.add_argument("--verify-model", action="store_true",
                        help="Verify latest bracket_model against historical data")
    parser.add_argument("--stage1-only", action="store_true",
                        help="With --verify-model: test stage-1 (auto GBM) only")
    parser.add_argument("--tune", action="store_true",
                        help="Grid search HistGBM hyperparameters")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to bracket_model pickle (default: latest)")
    parser.add_argument("--use-all-sites", action="store_true",
                        help="Include all training sites (default: Kalshi sites only)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Sample 1000 rows for quick iteration")
    parser.add_argument("--since", type=str, default=None,
                        help="Only use training data from this date (YYYY-MM-DD)")
    parser.add_argument("--use-doy", action="store_true",
                        help="Include day_of_year_sin/cos in feature set")
    parser.add_argument("--use-forecast", action="store_true",
                        help="Include forecast features (yesterday high, forecast high vs naive_f)")
    args = parser.parse_args()

    sites = args.site.split(",") if args.site else None
    if sites is None and args.use_all_sites:
        sites = list(ALL_SITES_WITH_TRAINING)
    max_rows = 1000 if args.dry_run else None
    if args.use_doy:
        for col in DOY_COLS:
            if col not in FEATURE_COLS:
                FEATURE_COLS.append(col)
        AUTO_FEATURE_COLS.extend(c for c in DOY_COLS if c not in AUTO_FEATURE_COLS)
        print(f"  Including day-of-year sin/cos features")
    if args.use_forecast:
        for col in FORECAST_COLS:
            if col not in FEATURE_COLS:
                FEATURE_COLS.append(col)
        AUTO_FEATURE_COLS.extend(c for c in FORECAST_COLS if c not in AUTO_FEATURE_COLS)
        print(f"  Including forecast features: {FORECAST_COLS}")
    if args.tune:
        tune_histgbm(sites, since=args.since)
    elif args.verify_model:
        verify_bracket_model(sites, model_path=args.model_path,
                             stage1_only=args.stage1_only, since=args.since or "2026-01-01")
    elif args.regression or args.bracket_upper or args.bracket_middle:
        rdf = _load_regression_data(sites, stage1_only=args.stage1_only, max_rows=max_rows, since=args.since, use_forecast=args.use_forecast)
        backtest_regression(sites, stage1_only=args.stage1_only, rdf=rdf)
    elif args.ml:
        backtest_ml_model(sites)
    else:
        run_backtest(sites)


if __name__ == "__main__":
    main()
