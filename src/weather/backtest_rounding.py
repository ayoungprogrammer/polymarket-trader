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

from weather.backtest import load_site_history, ALL_SITES
from weather.prediction import (
    c_to_possible_f,
    RoundingPrediction,
    predict_settlement_f,
)

from paths import project_path

SOLAR_NOON_CSV = project_path("data", "solar_noon.csv")


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

    # True daily high: prefer 24h ASOS max (settlement-grade), fall back to obs max
    all_temps_f = df["temperature_f"].dropna()
    if all_temps_f.empty:
        return None
    actual_max_f = round(float(all_temps_f.max()))
    if "max_temp_24h_f" in df.columns:
        val_24h = pd.to_numeric(df["max_temp_24h_f"], errors="coerce").max()
        if not np.isnan(val_24h):
            actual_max_f = round(float(val_24h))

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
    out_path = project_path("data", "backtest_rounding_results.json")
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

    return {
        "wind_at_peak": wind_at_peak,
        "wind_pre_peak": wind_pre_peak,
        "wind_calm_frac": wind_calm_frac,
        "wind_gust_range": wind_gust_range,
        "wind_change_at_peak": wind_change_at_peak,
        "rh_at_peak": rh_at_peak,
        "rh_min_near_peak": rh_min_near_peak,
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

    # True daily high: prefer 24h ASOS max (settlement-grade), fall back to obs max
    all_tf = df["temperature_f"].dropna()
    actual_max_f = round(float(all_tf.max())) if len(all_tf) > 0 else None
    if actual_max_f is not None and "max_temp_24h_f" in df.columns:
        val_24h = pd.to_numeric(df["max_temp_24h_f"], errors="coerce").max()
        if not np.isnan(val_24h):
            actual_max_f = round(float(val_24h))

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
AUTO_FEATURE_COLS = [
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
    # --- Temporal ---
    "day_of_year_sin", "day_of_year_cos",
    "solar_noon", "peak_minus_solar_noon",
    # --- Volatility ---
    "temp_f_volatility_30m", "temp_f_volatility_1hr", "temp_f_volatility_2hr",
    "temp_f_range_30m", "temp_f_range_1hr",
    "temp_f_iqr_1hr", "temp_f_iqr_2hr",
    "temp_f_mean_abs_diff", "temp_f_mean_abs_diff_30m",
    "sub_hour_oscillation_rate",
    "max_minus_ma15", "max_minus_ma30", "max_minus_ma60",
    # --- Peak edge rounding ---
    "peak_edge_f_gap_below",
    "peak_edge_f_gap_above",
    "dist_c_to_next_f",
    "dist_c_to_prev_f",
]

# METAR-derived features (require T-group precision readings).
METAR_FEATURE_COLS = [
    "has_metar", "metar_above_boundary", "metar_boundary_margin_c",
    "metar_gap_c", "metar_mean_gap_c", "metar_n_above",
    "metar_confirm", "peak_lag_min",
    "frac_x_metar_confirm",
]

# Combined feature set for the final model (auto + METAR + stacked auto-model
# probabilities).  The stacked columns (auto_prob_minus1, auto_prob_0,
# auto_prob_plus1) are added dynamically in backtest_regression().
FEATURE_COLS = AUTO_FEATURE_COLS + METAR_FEATURE_COLS


def backtest_regression(sites: Optional[List[str]] = None,
                        bracket_mode: Optional[str] = None):
    """Train classification models to predict settlement offset {-1, 0, +1}.

    Offset = actual_max_f - naive_f (clamped to [-1, +1]).
    Drops absolute-temp features (naive_f_float, max_c) and uses peak shape,
    momentum, and c_to_f_frac (rounding boundary proximity) only.

    bracket_mode:
      None     — 3-class offset {-1, 0, +1} (default)
      "upper"  — binary: YES (offset >= 0) vs NO (offset = -1)
                 Bracket is [naive_f, naive_f+1].
      "middle" — binary: upper bracket (offset = +1) vs lower bracket (offset <= 0)
                 Lower bracket [naive_f-1, naive_f], upper [naive_f+1, naive_f+2].
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_predict, StratifiedKFold
    from sklearn.metrics import (
        accuracy_score, classification_report, confusion_matrix,
    )

    if sites is None:
        sites = ALL_SITES

    # --- Load cached solar noon data ------------------------------------------
    solar_noon_lookup: Dict[Tuple[str, str], float] = {}
    if os.path.isfile(SOLAR_NOON_CSV):
        sn_df = pd.read_csv(SOLAR_NOON_CSV)
        for _, row in sn_df.iterrows():
            solar_noon_lookup[(row["site"], str(row["date"]))] = float(row["solar_noon_hour"])
        print(f"  Loaded {len(solar_noon_lookup)} solar noon entries from cache")
    else:
        print(f"  No solar noon cache found — using fallback (12.0)")

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
    n = len(rdf)

    # --- Solar noon coverage ---------------------------------------------------
    sn_na = int((rdf["solar_noon"] == 12.0).sum())
    print(f"  Solar noon: {n - sn_na}/{n} from API, {sn_na} fallback (NA)")

    # Target: offset clamped to {-1, 0, +1}
    y = rdf["offset"].values.astype(int)

    # Remap to binary for bracket modes
    if bracket_mode == "upper":
        y = (y >= 0).astype(int)  # 1 if offset in {0,+1}, 0 if -1
    elif bracket_mode == "middle":
        y = (y > 0).astype(int)   # 1 if offset = +1, 0 if offset in {-1,0}

    site_labels = rdf["site"].values

    # --- NA summary per column ------------------------------------------------
    na_counts = rdf.isna().sum()
    cols_with_na = na_counts[na_counts > 0]
    if len(cols_with_na) > 0:
        print(f"\n  Columns with NA values ({len(cols_with_na)}/{len(rdf.columns)}):")
        for col, cnt in cols_with_na.sort_values(ascending=False).items():
            print(f"    {col:40s} {cnt:5d} / {n}  ({cnt/n*100:5.1f}%)")
    else:
        print(f"\n  No NA values in any column ({len(rdf.columns)} cols, {n} rows)")

    # --- Class distribution --------------------------------------------------
    print("=" * 78)
    if bracket_mode is None:
        print("  CLASSIFICATION MODEL: predict offset (actual - naive) in {-1, 0, +1}")
    elif bracket_mode == "upper":
        print("  BRACKET-UPPER: predict YES (offset >= 0) vs NO (offset = -1)")
        print("  Bracket [naive_f, naive_f+1]: will true temp be at or above naive?")
    elif bracket_mode == "middle":
        print("  BRACKET-MIDDLE: predict UPPER (offset = +1) vs LOWER (offset <= 0)")
        print("  Lower [naive_f-1, naive_f] vs Upper [naive_f+1, naive_f+2]")
    print("=" * 78)
    print(f"  Total days: {n}  |  Sites: {len(set(site_labels))}")

    if bracket_mode == "upper":
        for cls, label in [(1, "YES (offset >= 0)"), (0, "NO  (offset = -1)")]:
            cnt = int(np.sum(y == cls))
            print(f"    {label}: {cnt:5d} ({cnt/n*100:5.1f}%)")
    elif bracket_mode == "middle":
        for cls, label in [(1, "UPPER (offset = +1)"), (0, "LOWER (offset <= 0)")]:
            cnt = int(np.sum(y == cls))
            print(f"    {label}: {cnt:5d} ({cnt/n*100:5.1f}%)")
    else:
        for cls in [-1, 0, 1]:
            cnt = int(np.sum(y == cls))
            print(f"    offset={cls:+d}: {cnt:5d} ({cnt/n*100:5.1f}%)")

    # --- Feature list (no absolute-temp features) ----------------------------
    feature_cols = list(FEATURE_COLS)

    # Drop rows where any feature is NA
    feature_mask = rdf[feature_cols].notna().all(axis=1)
    n_dropped = int((~feature_mask).sum())
    rdf = rdf[feature_mask].reset_index(drop=True)
    y = rdf["offset"].values.astype(int)
    if bracket_mode == "upper":
        y = (y >= 0).astype(int)
    elif bracket_mode == "middle":
        y = (y > 0).astype(int)
    site_labels = rdf["site"].values
    n = len(rdf)
    print(f"\n  Dropped {n_dropped} rows with NA features, {n} remaining")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # --- Baselines -----------------------------------------------------------
    print(f"\n  Baselines:")
    if bracket_mode is not None:
        majority_cls = int(np.argmax(np.bincount(y)))
        majority_count = int(np.sum(y == majority_cls))
        always_0_acc = majority_count / n
        majority_label = {
            "upper": {1: "always-YES", 0: "always-NO"},
            "middle": {1: "always-UPPER", 0: "always-LOWER"},
        }[bracket_mode][majority_cls]
        print(f"    {majority_label}:  {always_0_acc:.1%} ({majority_count}/{n})")
    else:
        always_0_acc = np.sum(y == 0) / n
        print(f"    Always-0:  {always_0_acc:.1%} ({np.sum(y == 0)}/{n})")

    # =========================================================================
    # Stage 1: Auto-only model (no METAR features)
    # =========================================================================
    auto_cols = [c for c in AUTO_FEATURE_COLS if c in rdf.columns]
    X_auto = rdf[auto_cols].values

    # Always train on 3-class offset for stage 1 (probabilities are richer)
    y_3class = rdf["offset"].values.astype(int)

    auto_model = GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        min_samples_leaf=5, random_state=42)

    # Out-of-fold predicted probabilities (no data leakage)
    auto_proba = cross_val_predict(
        auto_model, X_auto, y_3class, cv=cv, method="predict_proba")
    auto_pred = cross_val_predict(auto_model, X_auto, y_3class, cv=cv)
    auto_acc = accuracy_score(y_3class, auto_pred)

    # Map probability columns to class labels
    auto_model.fit(X_auto, y_3class)
    auto_classes = auto_model.classes_  # e.g. [-1, 0, 1]
    prob_col_names = [f"auto_prob_{c:+d}" for c in auto_classes]

    for i, col_name in enumerate(prob_col_names):
        rdf[col_name] = auto_proba[:, i]

    # Consensus features: stage-1 probability × METAR confirmation
    rdf["consensus"] = rdf["auto_prob_+1"] * rdf["metar_confirm"]
    rdf["auto_metar_divergence"] = rdf["auto_prob_+1"] - rdf["metar_confirm"]

    print(f"\n  Stage 1 — Auto-only GBM (no METAR): {auto_acc:.1%} (3-class)")

    # Auto-only feature importances
    importances = auto_model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    print(f"    Top features:")
    for i in sorted_idx[:8]:
        print(f"      {auto_cols[i]:30s}: {importances[i]:.4f}")

    # =========================================================================
    # Stage 2: Combined model (METAR + stacked auto-model probabilities)
    # =========================================================================
    metar_cols = [c for c in METAR_FEATURE_COLS if c in rdf.columns]
    consensus_cols = ["consensus", "auto_metar_divergence"]
    combined_cols = metar_cols + prob_col_names + consensus_cols
    feature_cols = combined_cols

    X = rdf[feature_cols].values
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    lr_kwargs = {"max_iter": 1000, "random_state": 42}
    if bracket_mode is None:
        lr_kwargs["multi_class"] = "multinomial"
    models = [
        ("LogisticRegression", LogisticRegression(**lr_kwargs)),
        ("RandomForest", RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=5,
            random_state=42, n_jobs=-1)),
        ("GradientBoosting", GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            min_samples_leaf=5, random_state=42)),
    ]

    print(f"\n  Stage 2 — Combined (auto + METAR + stacked probs):")
    print(f"  {'Model':<22s} {'Accuracy':>10s}")
    print(f"  {'-'*22} {'-'*10}")

    # Hard-rule masks for structural overrides
    metar_above_mask = rdf["metar_above_boundary"].values == 1.0
    naive_is_high_mask = rdf["naive_is_high"].values == 1.0
    naive_is_low_mask = (rdf["naive_is_high"].values == 0.0) & (rdf["n_possible_f"].values == 2.0)
    exact_mask = rdf["n_possible_f"].values == 1.0
    metar_gap_high_mask = rdf["metar_gap_c"].values >= 0.25

    # Structural override rules: (name, mask, clamped_from, clamped_to, confidence)
    # Confidence = historical accuracy of the forced value under this condition.
    # Rules are applied in order; later rules can override earlier ones.
    if bracket_mode is None:
        clamp_rules = [
            # naive_is_low: offset is never -1 (1/679 = 0.1%)
            ("naive_is_low→0",      naive_is_low_mask,  -1, 0, 0.999),
            # naive_is_high: offset is rarely +1 (40/740 = 5.4%)
            ("naive_is_high→0",     naive_is_high_mask,  1, 0, 0.946),
            # exact (n_possible_f == 1): offset ≈ always 0 (92.8%)
            ("exact→0",             exact_mask,         -1, 0, 0.928),
            ("exact→0",             exact_mask,          1, 0, 0.928),
            # metar_above_boundary: offset is never -1 (0/396)
            ("metar_above→0",       metar_above_mask,   -1, 0, 1.000),
            # metar_gap_c >= 0.25: offset is always +1 (188/188 = 100%)
            ("metar_gap≥.25→+1",    metar_gap_high_mask, 0, 1, 1.000),
            ("metar_gap≥.25→+1",    metar_gap_high_mask,-1, 1, 1.000),
        ]
    elif bracket_mode == "upper":
        clamp_rules = [
            # naive_is_low: offset never -1 → always >=0 → 1
            ("naive_is_low→1",      naive_is_low_mask,   0, 1, 0.999),
            # exact: offset=0 → >=0 → 1
            ("exact→1",             exact_mask,          0, 1, 0.928),
            # metar_above_boundary: offset never -1 → 1
            ("metar_above→1",       metar_above_mask,    0, 1, 1.000),
            # metar_gap_c >= 0.25: always +1 → 1
            ("metar_gap≥.25→1",     metar_gap_high_mask, 0, 1, 1.000),
        ]
    else:
        clamp_rules = []

    best_model_name = None
    best_acc = always_0_acc
    best_preds = np.zeros(n, dtype=int)
    best_confidence = np.full(n, always_0_acc)

    for name, model in models:
        use_scaled = name == "LogisticRegression"
        X_in = X_s if use_scaled else X
        cv_pred = cross_val_predict(model, X_in, y, cv=cv)
        cv_conf = np.zeros(n, dtype=float)

        # Apply structural overrides in order
        n_clamped = 0
        for rule_name, rule_mask, from_val, to_val, conf in clamp_rules:
            m = rule_mask & (cv_pred == from_val)
            cnt = int(np.sum(m))
            n_clamped += cnt
            cv_pred[m] = to_val
            cv_conf[m] = conf

        acc = accuracy_score(y, cv_pred)
        suffix = f"  ({n_clamped} clamped)" if n_clamped else ""
        print(f"  {name:<22s} {acc:9.1%}{suffix}")

        if acc > best_acc:
            best_acc = acc
            best_model_name = name
            best_preds = cv_pred
            best_confidence = cv_conf

    baseline_label = "majority-class" if bracket_mode else "always-0"
    if best_model_name:
        print(f"\n  Best model: {best_model_name} ({best_acc:.1%} vs {baseline_label} {always_0_acc:.1%})")
    else:
        print(f"\n  No model beat {baseline_label} baseline ({always_0_acc:.1%})")

    # --- Clamp confidence breakdown ---------------------------------------------
    clamped_mask = best_confidence > 0
    n_clamped_total = int(np.sum(clamped_mask))
    if n_clamped_total > 0:
        clamped_correct = int(np.sum((best_preds == y) & clamped_mask))
        clamped_acc = clamped_correct / n_clamped_total
        unclamped_mask = ~clamped_mask
        n_unclamped = int(np.sum(unclamped_mask))
        unclamped_correct = int(np.sum((best_preds[unclamped_mask] == y[unclamped_mask]))) if n_unclamped else 0
        unclamped_acc = unclamped_correct / n_unclamped if n_unclamped else 0.0
        print(f"\n  Clamp breakdown: {n_clamped_total} clamped ({clamped_acc:.1%} acc) "
              f"| {n_unclamped} unclamped ({unclamped_acc:.1%} acc)")
        # Per-confidence-level breakdown
        for conf_val in sorted(set(best_confidence[clamped_mask])):
            cm = clamped_mask & (best_confidence == conf_val)
            nc = int(np.sum(cm))
            correct = int(np.sum((best_preds == y) & cm))
            print(f"    conf={conf_val:.3f}: {nc:>4d} clamped, {correct}/{nc} correct ({correct/nc:.1%})")

    # --- Per-class metrics for best model ------------------------------------
    print(f"\n{'=' * 78}")
    print(f"  CLASSIFICATION REPORT (5-fold CV, best: {best_model_name or baseline_label})")
    print(f"{'=' * 78}")
    labels = sorted(set(y))
    if bracket_mode == "upper":
        target_names = ["NO (offset=-1)", "YES (offset>=0)"]
    elif bracket_mode == "middle":
        target_names = ["LOWER (offset<=0)", "UPPER (offset=+1)"]
    else:
        target_names = [f"offset={c:+d}" for c in labels]
    print(classification_report(y, best_preds, labels=labels,
                                target_names=target_names, zero_division=0))

    # --- Confusion matrix ----------------------------------------------------
    print(f"  Confusion matrix (rows=true, cols=predicted):")
    cm = confusion_matrix(y, best_preds, labels=labels)
    header = "        " + "  ".join(f"{c:+d}" for c in labels)
    print(f"  {header}")
    for i, cls in enumerate(labels):
        row = "  ".join(f"{cm[i, j]:4d}" for j in range(len(labels)))
        print(f"    {cls:+d}:  {row}")

    # --- Feature importances (fit on full data) ------------------------------
    print(f"\n{'=' * 78}")
    print(f"  FEATURE IMPORTANCES")
    print(f"{'=' * 78}")

    for name, model in models:
        use_scaled = name == "LogisticRegression"
        X_in = X_s if use_scaled else X
        model.fit(X_in, y)

        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            # For multiclass LR, average absolute coefficients across classes
            importances = np.mean(np.abs(model.coef_), axis=0)
        else:
            continue

        sorted_idx = np.argsort(importances)[::-1]
        print(f"\n  {name}:")
        for i in sorted_idx[:10]:
            print(f"    {feature_cols[i]:30s}: {importances[i]:.4f}")

    # --- Leave-One-Site-Out CV -----------------------------------------------
    print(f"\n{'=' * 78}")
    print(f"  LEAVE-ONE-SITE-OUT CROSS-VALIDATION")
    print(f"{'=' * 78}")

    unique_sites = sorted(set(site_labels))
    base_col = "Majority%" if bracket_mode else "Always0%"
    print(f"  {'Site':<8s} {'N':>5s} {base_col:>10s}"
          f" {'Best Acc%':>10s} {'Best':>18s}")
    print(f"  {'-'*8} {'-'*5} {'-'*10} {'-'*10} {'-'*18}")

    loso_total = 0
    loso_always0_correct = 0
    loso_best_correct = 0

    for site in unique_sites:
        test_mask = site_labels == site
        train_mask = ~test_mask
        n_test = int(test_mask.sum())
        if n_test == 0:
            continue

        y_test = y[test_mask]
        if bracket_mode is not None:
            majority_cls_site = int(np.argmax(np.bincount(y)))
            a0_correct = int(np.sum(y_test == majority_cls_site))
        else:
            a0_correct = int(np.sum(y_test == 0))
        a0_acc = a0_correct / n_test

        site_best_name = baseline_label.capitalize()
        site_best_acc = a0_acc
        site_best_correct = a0_correct

        for name, model in models:
            use_scaled = name == "LogisticRegression"
            X_train = (X_s if use_scaled else X)[train_mask]
            X_test = (X_s if use_scaled else X)[test_mask]
            y_train = y[train_mask]

            model.fit(X_train, y_train)
            pred = model.predict(X_test)

            acc = accuracy_score(y_test, pred)
            correct = int(np.sum(pred == y_test))

            if acc > site_best_acc:
                site_best_acc = acc
                site_best_correct = correct
                site_best_name = name

        print(f"  {site:<8s} {n_test:5d} {a0_acc:9.1%}"
              f" {site_best_acc:9.1%}"
              f" {site_best_name:>18s}")

        loso_total += n_test
        loso_always0_correct += a0_correct
        loso_best_correct += site_best_correct

    if loso_total > 0:
        print(f"\n  LOSO totals: {baseline_label} {loso_always0_correct}/{loso_total}"
              f" ({loso_always0_correct/loso_total*100:.1f}%)"
              f"  |  best {loso_best_correct}/{loso_total}"
              f" ({loso_best_correct/loso_total*100:.1f}%)")


def backtest_ml_model(sites: Optional[List[str]] = None):
    """Train and evaluate ML models for naive=low settlement prediction.

    Extracts features from all site-days, trains logistic regression and
    decision tree classifiers with cross-validation, and prints results.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier, export_text
    from sklearn.preprocessing import StandardScaler
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

    # --- Logistic regression ------------------------------------------------
    print(f"\n{'=' * 78}")
    print(f"  LOGISTIC REGRESSION")
    print(f"{'=' * 78}")

    X = rdf[feature_cols].fillna(0).values
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    lr = LogisticRegression(max_iter=1000)
    scores = cross_val_score(lr, X_s, y, cv=5, scoring="accuracy")
    print(f"    5-fold CV: {scores.mean():.1%} ± {scores.std():.1%}")

    lr.fit(X_s, y)
    coefs = sorted(zip(feature_cols, lr.coef_[0]),
                   key=lambda x: abs(x[1]), reverse=True)
    print(f"    Coefficients (standardized):")
    for name, c in coefs:
        print(f"      {name:20s}: {c:+.3f}")

    # LR with tuned probability threshold
    proba_cv = cross_val_predict(lr, X_s, y, cv=5, method="predict_proba")[:, 1]
    best_t, best_acc = 0.5, 0
    for t100 in range(20, 80):
        t = t100 / 100
        pred = (proba_cv >= t).astype(int)
        acc = (pred == y).mean()
        if acc > best_acc:
            best_t, best_acc = t, acc
    print(f"    Best probability threshold: {best_t:.2f} → {best_acc:.1%}")

    # --- Decision tree ------------------------------------------------------
    print(f"\n{'=' * 78}")
    print(f"  DECISION TREE (depth=3)")
    print(f"{'=' * 78}")

    dt = DecisionTreeClassifier(max_depth=3)
    dt_scores = cross_val_score(dt, X, y, cv=5, scoring="accuracy")
    print(f"    5-fold CV: {dt_scores.mean():.1%} ± {dt_scores.std():.1%}")

    dt.fit(X, y)
    print(export_text(dt, feature_names=feature_cols))

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
    bracket_mode: Optional[str] = None,
    stage1_only: bool = False,
):
    """Load a trained bracket_model and verify its accuracy on historical data.

    For each site-day, extracts features via extract_regression_features(),
    runs get_probability() with brackets matching the real Kalshi structure,
    and checks whether the model picks the correct bracket.

    bracket_mode:
      None     — 3-class offset {-1, 0, +1} via single-degree brackets
      "upper"  — binary: bracket [naive_f, naive_f+1] YES vs NO
      "middle" — binary: upper [naive_f+1, naive_f+2] vs lower [naive_f-1, naive_f]
    stage1_only:
      If True, use stage-1 (auto-only GBM) probabilities instead of
      the full 2-stage pipeline. Useful for isolating stage-1 performance.
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
    if bracket_mode:
        print(f"  Bracket mode: {bracket_mode}")

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
            actual_f = naive_f + actual_offset

            if bracket_mode == "upper":
                # Binary: does actual land in [naive_f, naive_f+1]?
                actual_yes = int(actual_offset >= 0)
                brackets = [(naive_f, naive_f + 1)]
                br_results = get_probability(model, feats, brackets)
                yes_prob = br_results[0][prob_key]
                pred_yes = int(yes_prob >= 0.5)
                results.append({
                    "site": site, "date": str(date),
                    "actual": actual_yes, "pred": pred_yes,
                    "correct": pred_yes == actual_yes,
                    "prob_yes": yes_prob,
                    "prob_no": 1.0 - yes_prob,
                    "actual_offset": actual_offset,
                })

            elif bracket_mode == "middle":
                # Binary: actual in upper [naive_f+1, naive_f+2] vs lower [naive_f-1, naive_f]?
                actual_upper = int(actual_offset > 0)
                brackets = [
                    (naive_f - 1, naive_f),      # lower
                    (naive_f + 1, naive_f + 2),   # upper
                ]
                br_results = get_probability(model, feats, brackets)
                lower_prob = br_results[0][prob_key]
                upper_prob = br_results[1][prob_key]
                pred_upper = int(upper_prob > lower_prob)
                results.append({
                    "site": site, "date": str(date),
                    "actual": actual_upper, "pred": pred_upper,
                    "correct": pred_upper == actual_upper,
                    "prob_lower": lower_prob,
                    "prob_upper": upper_prob,
                    "actual_offset": actual_offset,
                })

            else:
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
                results.append({
                    "site": site, "date": str(date),
                    "actual": actual_offset, "pred": pred_offset,
                    "correct": pred_offset == actual_offset,
                    "prob_minus1": offset_probs.get(-1, 0.0),
                    "prob_0": offset_probs.get(0, 0.0),
                    "prob_plus1": offset_probs.get(1, 0.0),
                    "actual_offset": actual_offset,
                })

    if not results:
        print("  No results generated.")
        return

    rdf = pd.DataFrame(results)
    n = len(rdf)
    n_correct = int(rdf["correct"].sum())

    # --- Header ---
    s1_tag = " (STAGE-1 ONLY)" if stage1_only else ""
    print(f"\n{'=' * 70}")
    if bracket_mode == "upper":
        print(f"  BRACKET MODEL VERIFICATION — UPPER [naive, naive+1] YES/NO{s1_tag}")
    elif bracket_mode == "middle":
        print(f"  BRACKET MODEL VERIFICATION — MIDDLE upper vs lower{s1_tag}")
    else:
        print(f"  BRACKET MODEL VERIFICATION — 3-CLASS OFFSET{s1_tag}")
    print(f"{'=' * 70}")
    print(f"  Total days: {n}  |  Sites: {rdf['site'].nunique()}")

    if bracket_mode == "upper":
        actual_yes = int((rdf["actual"] == 1).sum())
        baseline = max(actual_yes, n - actual_yes)
        base_label = "always-YES" if actual_yes >= n - actual_yes else "always-NO"
        print(f"  Model accuracy: {n_correct}/{n} ({n_correct / n:.1%})")
        print(f"  {base_label} baseline: {baseline}/{n} ({baseline / n:.1%})")
        print(f"  Lift: {(n_correct - baseline) / n:+.1%}")

        # Class breakdown
        print(f"\n  {'Class':>8s} {'N':>6s} {'Correct':>8s} {'Acc':>7s} {'Avg prob':>9s}")
        print(f"  {'-'*8} {'-'*6} {'-'*8} {'-'*7} {'-'*9}")
        for cls, label in [(1, "YES"), (0, "NO")]:
            mask = rdf["actual"] == cls
            cnt = int(mask.sum())
            if cnt == 0:
                continue
            correct = int(rdf.loc[mask, "correct"].sum())
            avg_p = rdf.loc[mask, "prob_yes" if cls == 1 else "prob_no"].mean()
            print(f"  {label:>8s} {cnt:>6d} {correct:>8d} {correct / cnt:>7.1%} {avg_p:>9.3f}")

        # Confusion matrix
        print(f"\n  Confusion matrix (rows=actual, cols=predicted):")
        print(f"  {'':>10s}  pred_NO  pred_YES")
        for actual, label in [(0, "actual_NO"), (1, "actual_YES")]:
            row = []
            for pred in [0, 1]:
                cnt = int(((rdf["actual"] == actual) & (rdf["pred"] == pred)).sum())
                row.append(f"{cnt:>8d}")
            print(f"  {label:>10s}  {'  '.join(row)}")

    elif bracket_mode == "middle":
        actual_upper = int((rdf["actual"] == 1).sum())
        baseline = max(actual_upper, n - actual_upper)
        base_label = "always-UPPER" if actual_upper >= n - actual_upper else "always-LOWER"
        print(f"  Model accuracy: {n_correct}/{n} ({n_correct / n:.1%})")
        print(f"  {base_label} baseline: {baseline}/{n} ({baseline / n:.1%})")
        print(f"  Lift: {(n_correct - baseline) / n:+.1%}")

        # Class breakdown
        print(f"\n  {'Class':>8s} {'N':>6s} {'Correct':>8s} {'Acc':>7s} {'Avg prob':>9s}")
        print(f"  {'-'*8} {'-'*6} {'-'*8} {'-'*7} {'-'*9}")
        for cls, label, col in [(0, "LOWER", "prob_lower"), (1, "UPPER", "prob_upper")]:
            mask = rdf["actual"] == cls
            cnt = int(mask.sum())
            if cnt == 0:
                continue
            correct = int(rdf.loc[mask, "correct"].sum())
            avg_p = rdf.loc[mask, col].mean()
            print(f"  {label:>8s} {cnt:>6d} {correct:>8d} {correct / cnt:>7.1%} {avg_p:>9.3f}")

        # Confusion matrix
        print(f"\n  Confusion matrix (rows=actual, cols=predicted):")
        print(f"  {'':>12s}  pred_LOW  pred_UP")
        for actual, label in [(0, "actual_LOW"), (1, "actual_UP")]:
            row = []
            for pred in [0, 1]:
                cnt = int(((rdf["actual"] == actual) & (rdf["pred"] == pred)).sum())
                row.append(f"{cnt:>8d}")
            print(f"  {label:>12s}  {'  '.join(row)}")

    else:
        always_0 = int((rdf["actual_offset"] == 0).sum())
        print(f"  Model accuracy: {n_correct}/{n} ({n_correct / n:.1%})")
        print(f"  Always-0 baseline: {always_0}/{n} ({always_0 / n:.1%})")
        print(f"  Lift: {(n_correct - always_0) / n:+.1%}")

        # Per-offset breakdown
        print(f"\n  Per-offset accuracy:")
        print(f"  {'Offset':>8s} {'N':>6s} {'Correct':>8s} {'Acc':>7s}")
        print(f"  {'-'*8} {'-'*6} {'-'*8} {'-'*7}")
        for offset in [-1, 0, 1]:
            mask = rdf["actual_offset"] == offset
            cnt = int(mask.sum())
            if cnt == 0:
                continue
            correct = int(rdf.loc[mask, "correct"].sum())
            print(f"  {offset:>+8d} {cnt:>6d} {correct:>8d} {correct / cnt:>7.1%}")

        # Confusion matrix
        print(f"\n  Confusion matrix (rows=actual, cols=predicted):")
        print(f"  {'':>8s}  pred-1  pred 0  pred+1")
        for actual in [-1, 0, 1]:
            row = []
            for pred in [-1, 0, 1]:
                cnt = int(((rdf["actual"] == actual) & (rdf["pred"] == pred)).sum())
                row.append(f"{cnt:>6d}")
            print(f"  actual{actual:+d}  {'  '.join(row)}")

    # Per-site accuracy (all modes)
    print(f"\n  Per-site accuracy:")
    base_col = "Baseline" if bracket_mode else "Always0"
    print(f"  {'Site':<8s} {'N':>5s} {'Acc':>7s} {base_col:>8s}")
    print(f"  {'-'*8} {'-'*5} {'-'*7} {'-'*8}")
    for site in sorted(rdf["site"].unique()):
        mask = rdf["site"] == site
        site_df = rdf[mask]
        n_s = len(site_df)
        correct = int(site_df["correct"].sum())
        if bracket_mode == "upper":
            majority = max(int((site_df["actual"] == 1).sum()),
                           int((site_df["actual"] == 0).sum()))
        elif bracket_mode == "middle":
            majority = max(int((site_df["actual"] == 1).sum()),
                           int((site_df["actual"] == 0).sum()))
        else:
            majority = int((site_df["actual_offset"] == 0).sum())
        print(f"  {site:<8s} {n_s:>5d} {correct / n_s:>7.1%} {majority / n_s:>8.1%}")

    # Calibration
    if bracket_mode is None:
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
    elif bracket_mode == "upper":
        avg_p = rdf.apply(
            lambda r: r["prob_yes"] if r["actual"] == 1 else r["prob_no"], axis=1).mean()
        print(f"\n  Avg predicted prob for correct class: {avg_p:.3f}")
    elif bracket_mode == "middle":
        avg_p = rdf.apply(
            lambda r: r["prob_upper"] if r["actual"] == 1 else r["prob_lower"], axis=1).mean()
        print(f"\n  Avg predicted prob for correct class: {avg_p:.3f}")


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
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to bracket_model pickle (default: latest)")
    args = parser.parse_args()

    sites = args.site.split(",") if args.site else None
    if args.verify_model:
        bm = None
        if args.bracket_upper:
            bm = "upper"
        elif args.bracket_middle:
            bm = "middle"
        verify_bracket_model(sites, model_path=args.model_path, bracket_mode=bm,
                             stage1_only=args.stage1_only)
    elif args.bracket_upper or args.bracket_middle:
        bracket_mode = "upper" if args.bracket_upper else "middle"
        backtest_regression(sites, bracket_mode=bracket_mode)
    elif args.regression:
        backtest_regression(sites)
    elif args.ml:
        backtest_ml_model(sites)
    else:
        run_backtest(sites)


if __name__ == "__main__":
    main()
