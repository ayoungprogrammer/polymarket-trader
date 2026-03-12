"""Exploratory analysis of structural offset rules.

Scans all numeric features for conditions where offset values are
near-impossible or dominant, then drills into combinatorial patterns.

Usage:
    PYTHONPATH=src python -m weather.eda_conditions
"""
from __future__ import annotations

import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from weather.backtest_rounding import (
    ALL_SITES,
    SOLAR_NOON_CSV,
    extract_regression_features,
    load_site_history,
)


def build_df(sites=None):
    if sites is None:
        sites = ALL_SITES
    solar_noon_lookup: Dict[Tuple[str, str], float] = {}
    if os.path.isfile(SOLAR_NOON_CSV):
        sn_df = pd.read_csv(SOLAR_NOON_CSV)
        for _, row in sn_df.iterrows():
            solar_noon_lookup[(row["site"], str(row["date"]))] = float(row["solar_noon_hour"])

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
    return pd.DataFrame(rows)


def scan_features(rdf: pd.DataFrame):
    """Scan all numeric features for strict-rule patterns."""
    n = len(rdf)
    print(f"Total rows: {n}")
    print(f"Offset distribution:\n{rdf['offset'].value_counts().sort_index()}\n")

    num_cols = rdf.select_dtypes(include=[np.number]).columns.tolist()
    exclude = {"offset", "naive_f", "actual_max_f", "n_auto", "naive_f_float", "max_c"}
    num_cols = [c for c in num_cols if c not in exclude]

    print(f"Scanning {len(num_cols)} numeric features...\n")
    print("=" * 100)
    print(f"  {'Feature':<35s} {'Condition':<25s} {'N':>5s} {'off=-1':>8s} {'off=0':>8s} {'off=+1':>8s}  Notes")
    print("=" * 100)

    results = []
    for col in num_cols:
        vals = rdf[col].dropna()
        if len(vals) == 0:
            continue
        uniq = sorted(vals.unique())

        # Low-cardinality: check each value
        if len(uniq) <= 15:
            for v in uniq:
                mask = rdf[col] == v
                sub = rdf.loc[mask, "offset"]
                n_sub = len(sub)
                if n_sub < 15:
                    continue
                counts = sub.value_counts()
                pct_m1 = counts.get(-1, 0) / n_sub
                pct_0 = counts.get(0, 0) / n_sub
                pct_p1 = counts.get(1, 0) / n_sub

                notes = []
                if pct_m1 < 0.03 and counts.get(-1, 0) <= 3:
                    notes.append(f"-1 NEAR-ZERO ({counts.get(-1,0)}/{n_sub})")
                if pct_p1 < 0.03 and counts.get(1, 0) <= 3:
                    notes.append(f"+1 NEAR-ZERO ({counts.get(1,0)}/{n_sub})")
                if pct_0 > 0.90:
                    notes.append(f"0 DOMINANT ({pct_0:.1%})")
                if pct_m1 > 0.90:
                    notes.append(f"-1 DOMINANT ({pct_m1:.1%})")
                if pct_p1 > 0.90:
                    notes.append(f"+1 DOMINANT ({pct_p1:.1%})")

                if notes:
                    cond = f"== {v}"
                    print(f"  {col:<33s} {cond:<25s} {n_sub:>5d} {pct_m1:>7.1%} {pct_0:>7.1%} {pct_p1:>7.1%}  {'; '.join(notes)}")
                    results.append((col, cond, n_sub, pct_m1, pct_0, pct_p1, notes))
        else:
            # Continuous: check extreme quantiles
            for q_lo, q_hi, label in [
                (0, 0.05, "bot 5%"), (0.95, 1.0, "top 5%"),
                (0, 0.10, "bot 10%"), (0.90, 1.0, "top 10%"),
                (0, 0.15, "bot 15%"), (0.85, 1.0, "top 15%"),
            ]:
                if q_lo == 0:
                    threshold = vals.quantile(q_hi)
                    mask = rdf[col] <= threshold
                    cond = f"<= {threshold:.4f} ({label})"
                else:
                    threshold = vals.quantile(q_lo)
                    mask = rdf[col] >= threshold
                    cond = f">= {threshold:.4f} ({label})"

                sub = rdf.loc[mask, "offset"]
                n_sub = len(sub)
                if n_sub < 15:
                    continue
                counts = sub.value_counts()
                pct_m1 = counts.get(-1, 0) / n_sub
                pct_0 = counts.get(0, 0) / n_sub
                pct_p1 = counts.get(1, 0) / n_sub

                notes = []
                if pct_m1 < 0.02 and counts.get(-1, 0) <= 2:
                    notes.append(f"-1 NEAR-ZERO ({counts.get(-1,0)}/{n_sub})")
                if pct_p1 < 0.02 and counts.get(1, 0) <= 2:
                    notes.append(f"+1 NEAR-ZERO ({counts.get(1,0)}/{n_sub})")
                if pct_0 > 0.90:
                    notes.append(f"0 DOMINANT ({pct_0:.1%})")

                if notes:
                    print(f"  {col:<33s} {cond:<25s} {n_sub:>5d} {pct_m1:>7.1%} {pct_0:>7.1%} {pct_p1:>7.1%}  {'; '.join(notes)}")
                    results.append((col, cond, n_sub, pct_m1, pct_0, pct_p1, notes))

    print(f"\n{'=' * 100}")
    print(f"Found {len(results)} candidate patterns total")
    return results


def deep_analysis(rdf: pd.DataFrame):
    """Drill into the strongest combinatorial patterns."""

    print("\n" + "=" * 90)
    print("DEEP PATTERN ANALYSIS")
    print("=" * 90)

    # --- c_to_f_frac groups ---
    print("\n--- c_to_f_frac groups ---")
    for frac_val in sorted(rdf["c_to_f_frac"].dropna().unique()):
        mask = rdf["c_to_f_frac"] == frac_val
        sub = rdf.loc[mask, "offset"]
        n_sub = len(sub)
        if n_sub < 10:
            continue
        counts = sub.value_counts()
        pct_m1 = counts.get(-1, 0) / n_sub
        pct_0 = counts.get(0, 0) / n_sub
        pct_p1 = counts.get(1, 0) / n_sub
        print(f"  frac={frac_val:.4f}  n={n_sub:>4d}  -1={pct_m1:5.1%}  0={pct_0:5.1%}  +1={pct_p1:5.1%}  "
              f"(-1: {counts.get(-1,0)}, +1: {counts.get(1,0)})")

    # --- metar_confirm thresholds ---
    print("\n--- metar_confirm >= threshold → offset +1? ---")
    for thr in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]:
        mask = rdf["metar_confirm"] >= thr
        sub = rdf.loc[mask, "offset"]
        n_sub = len(sub)
        if n_sub < 5:
            continue
        counts = sub.value_counts()
        pct_p1 = counts.get(1, 0) / n_sub
        pct_0 = counts.get(0, 0) / n_sub
        pct_m1 = counts.get(-1, 0) / n_sub
        print(f"  metar_confirm >= {thr:.2f}  n={n_sub:>4d}  -1={pct_m1:5.1%}  0={pct_0:5.1%}  +1={pct_p1:5.1%}")

    # --- metar_confirm + naive_is_high combos ---
    print("\n--- metar_confirm >= threshold AND naive_is_high ---")
    for nih in [0, 1]:
        for thr in [0.2, 0.25, 0.3]:
            mask = (rdf["metar_confirm"] >= thr) & (rdf["naive_is_high"] == nih)
            sub = rdf.loc[mask, "offset"]
            n_sub = len(sub)
            if n_sub < 5:
                continue
            counts = sub.value_counts()
            print(f"  metar_confirm >= {thr}, naive_is_high={nih}  n={n_sub:>4d}  "
                  f"-1={counts.get(-1,0):>3d} ({counts.get(-1,0)/n_sub:5.1%})  "
                  f" 0={counts.get(0,0):>3d} ({counts.get(0,0)/n_sub:5.1%})  "
                  f"+1={counts.get(1,0):>3d} ({counts.get(1,0)/n_sub:5.1%})")

    # --- single_reading_peak ---
    print("\n--- single_reading_peak ---")
    for val in [0, 1]:
        mask = rdf["single_reading_peak"] == val
        sub = rdf.loc[mask, "offset"]
        n_sub = len(sub)
        counts = sub.value_counts()
        pct_m1 = counts.get(-1, 0) / n_sub
        pct_0 = counts.get(0, 0) / n_sub
        pct_p1 = counts.get(1, 0) / n_sub
        print(f"  single_reading_peak={val}  n={n_sub:>4d}  -1={pct_m1:5.1%}  0={pct_0:5.1%}  +1={pct_p1:5.1%}")
    # with naive_is_high
    for nih in [0, 1]:
        mask = (rdf["single_reading_peak"] == 1) & (rdf["naive_is_high"] == nih)
        sub = rdf.loc[mask, "offset"]
        n_sub = len(sub)
        if n_sub < 5:
            continue
        counts = sub.value_counts()
        print(f"  single_peak=1, naive_is_high={nih}: n={n_sub}  "
              f"{dict(counts.sort_index())}")

    # --- dwell_count bins ---
    print("\n--- dwell_count (readings at max_c) ---")
    for low, high in [(1, 2), (3, 5), (6, 10), (11, 20), (21, 50), (51, 999)]:
        mask = (rdf["dwell_count"] >= low) & (rdf["dwell_count"] <= high)
        sub = rdf.loc[mask, "offset"]
        n_sub = len(sub)
        if n_sub < 10:
            continue
        counts = sub.value_counts()
        pct_m1 = counts.get(-1, 0) / n_sub
        pct_0 = counts.get(0, 0) / n_sub
        pct_p1 = counts.get(1, 0) / n_sub
        print(f"  dwell_count [{low:>3d}-{high:>3d}]  n={n_sub:>4d}  -1={pct_m1:5.1%}  0={pct_0:5.1%}  +1={pct_p1:5.1%}")

    # --- dwell_count + naive_is_high ---
    print("\n--- dwell_count >= threshold & naive_is_high ---")
    for nih in [0, 1]:
        for dwell_thr in [20, 30, 40, 50]:
            mask = (rdf["dwell_count"] >= dwell_thr) & (rdf["naive_is_high"] == nih)
            sub = rdf.loc[mask, "offset"]
            n_sub = len(sub)
            if n_sub < 10:
                continue
            counts = sub.value_counts()
            pct_m1 = counts.get(-1, 0) / n_sub
            pct_0 = counts.get(0, 0) / n_sub
            pct_p1 = counts.get(1, 0) / n_sub
            print(f"  dwell>={dwell_thr:>2d}, naive_is_high={nih}  n={n_sub:>4d}  "
                  f"-1={pct_m1:5.1%}  0={pct_0:5.1%}  +1={pct_p1:5.1%}  "
                  f"(-1:{counts.get(-1,0)}, +1:{counts.get(1,0)})")

    # --- consec_count + naive_is_high ---
    print("\n--- consec_count >= threshold & naive_is_high ---")
    for nih in [0, 1]:
        for cc_thr in [10, 15, 20, 30]:
            mask = (rdf["consec_count"] >= cc_thr) & (rdf["naive_is_high"] == nih)
            sub = rdf.loc[mask, "offset"]
            n_sub = len(sub)
            if n_sub < 10:
                continue
            counts = sub.value_counts()
            pct_m1 = counts.get(-1, 0) / n_sub
            pct_0 = counts.get(0, 0) / n_sub
            pct_p1 = counts.get(1, 0) / n_sub
            print(f"  consec>={cc_thr:>2d}, naive_is_high={nih}  n={n_sub:>4d}  "
                  f"-1={pct_m1:5.1%}  0={pct_0:5.1%}  +1={pct_p1:5.1%}  "
                  f"(-1:{counts.get(-1,0)}, +1:{counts.get(1,0)})")

    # --- Volatility + naive_is_high ---
    print("\n--- temp_f_volatility_1hr <= threshold & naive_is_high ---")
    for thr in [0.4, 0.5, 0.6, 0.7, 0.8]:
        for nih in [0, 1]:
            mask = (rdf["temp_f_volatility_1hr"] <= thr) & (rdf["naive_is_high"] == nih)
            sub = rdf.loc[mask, "offset"]
            n_sub = len(sub)
            if n_sub < 10:
                continue
            counts = sub.value_counts()
            pct_m1 = counts.get(-1, 0) / n_sub
            pct_0 = counts.get(0, 0) / n_sub
            pct_p1 = counts.get(1, 0) / n_sub
            if pct_p1 < 0.03 or pct_m1 < 0.03 or pct_0 > 0.90:
                print(f"  vol_1hr<={thr}, naive_is_high={nih}  n={n_sub:>4d}  "
                      f"-1={pct_m1:5.1%}  0={pct_0:5.1%}  +1={pct_p1:5.1%}")

    # --- metar_gap_c thresholds ---
    print("\n--- metar_gap_c >= threshold ---")
    for thr in [0.1, 0.15, 0.2, 0.25, 0.3]:
        mask = rdf["metar_gap_c"] >= thr
        sub = rdf.loc[mask, "offset"]
        n_sub = len(sub)
        if n_sub < 5:
            continue
        counts = sub.value_counts()
        pct_m1 = counts.get(-1, 0) / n_sub
        pct_0 = counts.get(0, 0) / n_sub
        pct_p1 = counts.get(1, 0) / n_sub
        print(f"  metar_gap_c >= {thr:.2f}  n={n_sub:>4d}  -1={pct_m1:5.1%}  0={pct_0:5.1%}  +1={pct_p1:5.1%}")

    print("\n--- metar_gap_c <= threshold (METAR < auto max) ---")
    for thr in [-0.1, -0.2, -0.3, -0.4, -0.5]:
        mask = rdf["metar_gap_c"] <= thr
        sub = rdf.loc[mask, "offset"]
        n_sub = len(sub)
        if n_sub < 5:
            continue
        counts = sub.value_counts()
        pct_m1 = counts.get(-1, 0) / n_sub
        pct_0 = counts.get(0, 0) / n_sub
        pct_p1 = counts.get(1, 0) / n_sub
        print(f"  metar_gap_c <= {thr:.2f}  n={n_sub:>4d}  -1={pct_m1:5.1%}  0={pct_0:5.1%}  +1={pct_p1:5.1%}")

    # --- metar_gap_c + naive_is_high ---
    print("\n--- metar_gap_c >= 0.2 + naive_is_high ---")
    for nih in [0, 1]:
        mask = (rdf["metar_gap_c"] >= 0.2) & (rdf["naive_is_high"] == nih)
        sub = rdf.loc[mask, "offset"]
        n_sub = len(sub)
        if n_sub < 5:
            continue
        counts = sub.value_counts()
        print(f"  metar_gap_c >= 0.2, naive_is_high={nih}  n={n_sub:>4d}  "
              f"-1={counts.get(-1,0):>3d}  0={counts.get(0,0):>3d}  +1={counts.get(1,0):>3d}")

    print("\n--- metar_gap_c <= threshold AND naive_is_high == 1 ---")
    for thr in [-0.3, -0.4, -0.5, -0.6, -0.7]:
        mask = (rdf["metar_gap_c"] <= thr) & (rdf["naive_is_high"] == 1)
        sub = rdf.loc[mask, "offset"]
        n_sub = len(sub)
        if n_sub < 5:
            continue
        counts = sub.value_counts()
        pct_m1 = counts.get(-1, 0) / n_sub
        print(f"  gap<={thr}, nih=1: n={n_sub}  -1={pct_m1:.1%}  "
              f"counts={dict(counts.sort_index())}")

    # --- metar_confirm vs metar_gap_c ---
    print("\n--- metar_confirm vs metar_gap_c correlation ---")
    both = rdf[["metar_confirm", "metar_gap_c"]].dropna()
    print(f"  Pearson r = {both['metar_confirm'].corr(both['metar_gap_c']):.4f}")
    print(f"  metar_gap_c >= 0.25: {int((rdf['metar_gap_c'] >= 0.25).sum())}")
    print(f"  metar_confirm >= 0.25: {int((rdf['metar_confirm'] >= 0.25).sum())}")
    print(f"  Both >= 0.25: {int(((rdf['metar_confirm'] >= 0.25) & (rdf['metar_gap_c'] >= 0.25)).sum())}")

    # --- Summary of strict rules ---
    print("\n" + "=" * 90)
    print("SUMMARY: PERFECT OR NEAR-PERFECT RULES")
    print("=" * 90)

    rules = [
        ("n_possible_f == 1 (exact)", rdf["n_possible_f"] == 1, "clamp to 0 (92.8%)"),
        ("naive_is_low & n_poss==2", (rdf["naive_is_high"] == 0) & (rdf["n_possible_f"] == 2), "clamp -1→0 (0.1%)"),
        ("naive_is_high", rdf["naive_is_high"] == 1, "clamp +1→0 (5.4%)"),
        ("metar_above_boundary", rdf["metar_above_boundary"] == 1, "clamp -1→0 (0.0%)"),
        ("metar_gap_c >= 0.25", rdf["metar_gap_c"] >= 0.25, "force +1 (100%)"),
        ("metar_gap_c >= 0.2 & nih==0", (rdf["metar_gap_c"] >= 0.2) & (rdf["naive_is_high"] == 0), "force +1 (100%)"),
        ("single_peak==1 & nih==0", (rdf["single_reading_peak"] == 1) & (rdf["naive_is_high"] == 0), "clamp +1→0?"),
        ("single_peak==1 & nih==1", (rdf["single_reading_peak"] == 1) & (rdf["naive_is_high"] == 1), "strong -1 signal"),
        ("vol_1hr<=0.5 & nih==0 & n_poss==2", (rdf["temp_f_volatility_1hr"] <= 0.5) & (rdf["naive_is_high"] == 0) & (rdf["n_possible_f"] == 2), "force 0 (100%?)"),
        ("vol_1hr<=0.5 & nih==1", (rdf["temp_f_volatility_1hr"] <= 0.5) & (rdf["naive_is_high"] == 1), "force -1 (92.3%)"),
        ("consec>=30 & nih==0", (rdf["consec_count"] >= 30) & (rdf["naive_is_high"] == 0), "force +1 (90.4%)"),
    ]

    for name, mask, desc in rules:
        sub = rdf.loc[mask, "offset"]
        n_sub = len(sub)
        if n_sub == 0:
            continue
        counts = sub.value_counts()
        pct_m1 = counts.get(-1, 0) / n_sub
        pct_0 = counts.get(0, 0) / n_sub
        pct_p1 = counts.get(1, 0) / n_sub
        print(f"  {name:<45s} n={n_sub:>4d}  -1={pct_m1:5.1%}  0={pct_0:5.1%}  +1={pct_p1:5.1%}  {desc}")

    # --- Overlap analysis ---
    print("\n--- Overlap analysis ---")
    gap_025 = rdf["metar_gap_c"] >= 0.25
    gap_02_nih0 = (rdf["metar_gap_c"] >= 0.2) & (rdf["naive_is_high"] == 0)
    metar_above = rdf["metar_above_boundary"] == 1
    nih0 = (rdf["naive_is_high"] == 0) & (rdf["n_possible_f"] == 2)
    print(f"  metar_gap_c >= 0.25 overlaps with metar_above: {int((gap_025 & metar_above).sum())}/{int(gap_025.sum())}")
    print(f"  metar_gap_c >= 0.25 overlaps with naive_is_low: {int((gap_025 & nih0).sum())}/{int(gap_025.sum())}")
    print(f"  gap_02_nih0 is subset of naive_is_low: {int((gap_02_nih0 & nih0).sum())}/{int(gap_02_nih0.sum())}")


if __name__ == "__main__":
    rdf = build_df()
    scan_features(rdf)
    deep_analysis(rdf)
