"""Bracket probability model for Kalshi temperature markets.

Trains a 2-stage model (auto-only GBM → combined RF with structural clamps)
to predict settlement offset {-1, 0, +1} from auto-obs peak features, then
converts offset probabilities to bracket probabilities for live betting.

Usage:
    # Train and persist model
   python src/weather/bracket_model.py

    # Load and predict
    from weather.bracket_model import load_model, get_probability
    model = load_model()
    probs = get_probability(model, features, [(80, 81), (81, 82)])

Current progress

  Site         N   Always0%  S1 Acc%  S2 Acc%  S2+Clamp%   Upper%  Middle%
  -------- ----- ---------- -------- -------- ---------- -------- --------
  KATL        99     51.5%   75.8%   75.8%     75.8%   88.9%   86.9%
  KAUS        99     53.5%   78.8%   79.8%     79.8%   91.9%   87.9%
  KBOS        99     52.5%   85.9%   89.9%     89.9%   96.0%   93.9%
  KDCA        98     56.1%   75.5%   82.7%     82.7%   92.9%   89.8%
  KDEN        99     46.5%   81.8%   80.8%     80.8%   92.9%   87.9%
  KDFW        99     50.5%   82.8%   81.8%     81.8%   93.9%   86.9%
  KHOU        94     53.2%   83.0%   84.0%     84.0%   91.5%   92.6%
  KLAX        98     49.0%   77.6%   82.7%     82.7%   94.9%   87.8%
  KMDW        99     55.6%   77.8%   79.8%     79.8%   86.9%   92.9%
  KMIA        99     54.5%   77.8%   80.8%     80.8%   93.9%   86.9%
  KMSP        99     50.5%   75.8%   82.8%     82.8%   90.9%   91.9%
  KOKC        98     51.0%   79.6%   84.7%     84.7%   92.9%   91.8%
  KPHL        92     47.8%   81.5%   82.6%     82.6%   92.4%   89.1%
  KPHX        98     45.9%   74.5%   70.4%     70.4%   90.8%   78.6%
  KSAT        98     45.9%   85.7%   87.8%     87.8%   96.9%   90.8%
  KSEA        99     51.5%   83.8%   82.8%     82.8%   89.9%   92.9%
  KSFO        99     61.6%   73.7%   75.8%     75.8%   88.9%   85.9%

  TOTAL     1666     51.6%   79.5%   81.5%     81.5%   92.1%   89.1%

  

"""

from __future__ import annotations

import os
import pickle
import sys
from datetime import datetime
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd

from weather.backtest_rounding import (
    ALL_SITES,
    AUTO_FEATURE_COLS,
    METAR_FEATURE_COLS,
    SOLAR_NOON_CSV,
    extract_regression_features,
)
from weather.backtest import load_site_history
from weather.prediction import c_to_possible_f
from paths import project_path


# ---------------------------------------------------------------------------
# Clamp rules — structural overrides applied after stage-2 prediction
# ---------------------------------------------------------------------------
# Each rule: (name, condition_dict, from_val, to_val, confidence)
#   condition_dict: {feature: value} for equality,
#                   {feature__gte: value} for >=, {feature__lte: value} for <=

# # OLD CLAMP RULES (v1) — superseded by EDA-derived rules below
# CLAMP_RULES_V1 = [
#     ("naive_is_low→0",   {"naive_is_high": 0, "n_possible_f": 2}, -1, 0, 0.999),
#     ("naive_is_high→0",  {"naive_is_high": 1},                     1, 0, 0.946),
#     ("exact→0",          {"n_possible_f": 1},                     -1, 0, 0.928),
#     ("exact→0",          {"n_possible_f": 1},                      1, 0, 0.928),
#     ("metar_above→0",    {"metar_above_boundary": 1},             -1, 0, 1.000),
#     ("metar_gap≥.25→+1", {"metar_gap_c__gte": 0.25},              0, 1, 1.000),
#     ("metar_gap≥.25→+1", {"metar_gap_c__gte": 0.25},             -1, 1, 1.000),
# ]

CLAMP_RULES = [
    # --- Structural: rounding geometry ---
    # n_possible_f == 1: only one °F maps to this °C, offset ≠ 0 near-impossible
    ("exact→0 (clamp -1)",        {"n_possible_f": 1},                                -1, 0, 1.0),
    # naive_is_low with 2 possible °F: -1 never happens (0.1%)
    ("nih0+n2→0 (clamp -1)",      {"naive_is_high": 0, "n_possible_f": 2},            -1, 0, 0.999),
    # naive_is_high == 1: +1 rare (5.4%)
    # ("nih1→0 (clamp +1)",         {"naive_is_high": 1},                                1, 0, 0.95),
    # gap_below == 1: losing 1°C only costs 1°F, -1 near-impossible (0.3%)
    ("gap_below1→0 (clamp -1)",   {"peak_edge_f_gap_below": 1},                       -1, 0, 0.997),
    # gap_above == 1: gaining 1°C only adds 1°F, +1 rare (6.1%)
    # ("gap_above1→0 (clamp +1)",   {"peak_edge_f_gap_above": 1},                        1, 0, 0.94),

    # --- METAR-based ---
    # metar_above_boundary: METAR confirmed above rounding boundary, -1 impossible
    ("metar_above→0 (clamp -1)",  {"metar_above_boundary": 1},                        -1, 0, 1.0),
    # metar_gap_c >= 0.25: 100% +1 in data (n=188)
    ("metar_gap≥.25→+1 (force)",  {"metar_gap_c__gte": 0.25},                          0, 1, 1.0),
    ("metar_gap≥.25→+1 (force)",  {"metar_gap_c__gte": 0.25},                         -1, 1, 1.0),
    # metar_gap_c >= 0.2 & nih==0: 100% +1 (n=183)
    ("gap≥.2+nih0→+1 (force)",    {"metar_gap_c__gte": 0.2, "naive_is_high": 0},       0, 1, 1.0),
    ("gap≥.2+nih0→+1 (force)",    {"metar_gap_c__gte": 0.2, "naive_is_high": 0},      -1, 1, 1.0),
    # metar_confirm >= 0.2 & nih==0: 100% +1 (n=183)
    ("cfm≥.2+nih0→+1 (force)",    {"metar_confirm__gte": 0.2, "naive_is_high": 0},     0, 1, 1.0),
    ("cfm≥.2+nih0→+1 (force)",    {"metar_confirm__gte": 0.2, "naive_is_high": 0},    -1, 1, 1.0),
    # metar_gap_c <= -0.5 & nih==1: 89.3% -1 (n=178)
    # ("gap≤-.5+nih1→-1 (force)",   {"metar_gap_c__lte": -0.5, "naive_is_high": 1},      0, -1, 0.893),
    # ("gap≤-.5+nih1→-1 (force)",   {"metar_gap_c__lte": -0.5, "naive_is_high": 1},      1, -1, 0.893),

    # --- Peak shape ---
    # single_reading_peak==1 & nih==0: peaked once, +1 nearly impossible (2.9%)
    ("single+nih0→0 (clamp +1)",  {"single_reading_peak": 1, "naive_is_high": 0},      1, 0, 0.96),
    # single_reading_peak==1 & nih==1: strong -1 (93.7%)
    # ("single+nih1→-1 (force)",    {"single_reading_peak": 1, "naive_is_high": 1},       0, -1, 0.94),
    # ("single+nih1→-1 (force)",    {"single_reading_peak": 1, "naive_is_high": 1},       1, -1, 0.94),
    # consec>=30 & nih==0: sustained at peak, force +1 (90.4%)
#     ("consec≥30+nih0→+1 (force)", {"consec_count__gte": 30, "naive_is_high": 0},        0, 1, 0.904),
#     ("consec≥30+nih0→+1 (force)", {"consec_count__gte": 30, "naive_is_high": 0},       -1, 1, 0.904),
]


def _bin_probs(proba, bins=[0, 0.33, 0.66, 1.01]):
    """Discretize probabilities into bins (0=low, 1=med, 2=high)."""
    return np.digitize(proba, bins[1:])  # returns 0, 1, or 2


def _matches_rule(features: dict, condition: dict) -> bool:
    """Check if a single row's features match a clamp rule condition."""
    for key, threshold in condition.items():
        if key.endswith("__gte"):
            feat_key = key[:-5]
            val = features.get(feat_key)
            if val is None or np.isnan(val) or val < threshold:
                return False
        elif key.endswith("__lte"):
            feat_key = key[:-5]
            val = features.get(feat_key)
            if val is None or np.isnan(val) or val > threshold:
                return False
        else:
            val = features.get(key)
            if val is None or val != threshold:
                return False
    return True


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(sites: Optional[List[str]] = None) -> str:
    """Train the 2-stage bracket model on full dataset and persist to disk.

    Returns the path to the saved model pickle.
    """
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.preprocessing import StandardScaler

    if sites is None:
        sites = list(ALL_SITES)

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

    # --- Drop NA rows on feature columns --------------------------------------
    auto_cols = [c for c in AUTO_FEATURE_COLS if c in rdf.columns]
    metar_cols = [c for c in METAR_FEATURE_COLS if c in rdf.columns]

    feature_mask = rdf[auto_cols + metar_cols].notna().all(axis=1)
    rdf = rdf[feature_mask].reset_index(drop=True)
    n = len(rdf)

    y_3class = rdf["offset"].values.astype(int)

    # =========================================================================
    # Stage 1: Auto-only GBM (3-class offset)
    # =========================================================================
    X_auto = rdf[auto_cols].values

    stage1_model = GradientBoostingClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        min_samples_leaf=5, random_state=42,
    )
    stage1_model.fit(X_auto, y_3class)

    # Out-of-fold S1 predictions for S2 training (eliminates leakage)
    from sklearn.model_selection import cross_val_predict
    auto_proba = cross_val_predict(
        stage1_model, X_auto, y_3class,
        cv=5, method='predict_proba', n_jobs=-1,
    )
    # Re-fit S1 on all data for the saved inference model
    stage1_model.fit(X_auto, y_3class)
    auto_classes = stage1_model.classes_  # e.g. [-1, 0, 1]
    prob_col_names = [f"auto_prob_{c:+d}" for c in auto_classes]

    for i, col_name in enumerate(prob_col_names):
        rdf[col_name] = _bin_probs(auto_proba[:, i])

    # Consensus features (using binned values)
    rdf["consensus"] = rdf["auto_prob_+1"] * rdf["metar_confirm"]
    rdf["auto_metar_divergence"] = rdf["auto_prob_+1"] - rdf["metar_confirm"]

    # =========================================================================
    # Stage 2: Combined RF (METAR + stacked auto probs + consensus)
    # =========================================================================
    consensus_cols = ["consensus", "auto_metar_divergence"]
    stage2_cols = metar_cols + prob_col_names + consensus_cols

    X_stage2 = rdf[stage2_cols].values
    stage2_scaler = StandardScaler()
    X_stage2_s = stage2_scaler.fit_transform(X_stage2)

    stage2_model = RandomForestClassifier(
        n_estimators=200, max_depth=8, min_samples_leaf=5,
        random_state=42, n_jobs=-1,
    )
    stage2_model.fit(X_stage2_s, y_3class)

    # =========================================================================
    # Save bundle
    # =========================================================================
    out_dir = project_path("models", "weather")
    os.makedirs(out_dir, exist_ok=True)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"{timestamp}.pkl")

    bundle = {
        "stage1_model": stage1_model,
        "stage2_model": stage2_model,
        "stage2_scaler": stage2_scaler,
        "auto_cols": auto_cols,
        "stage2_cols": stage2_cols,
        "trained_at": datetime.utcnow().isoformat(),
        "n_rows": n,
        "sites": sites,
    }

    with open(out_path, "wb") as f:
        pickle.dump(bundle, f)

    print(f"Saved model to {out_path} ({n} rows, {len(sites)} sites)")
    return out_path


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_model(path: Optional[str] = None) -> dict:
    """Load a trained model bundle. Latest by filename if path is None."""
    if path is None:
        model_dir = project_path("models", "weather")
        import glob
        files = sorted(glob.glob(os.path.join(model_dir, "*.pkl")))
        if not files:
            raise FileNotFoundError(f"No model files in {model_dir}")
        path = files[-1]

    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def get_probability(
    model: dict,
    features: dict,
    brackets: List[Tuple[int, int]],
    metar_6h_f: Optional[float] = None,
) -> List[dict]:
    """Compute bracket probabilities from extracted features.

    Args:
        model: dict from load_model()
        features: dict from extract_regression_features() (single day)
        brackets: list of (low, high) tuples — Kalshi bracket bounds
        metar_6h_f: optional 6h METAR max in °F (0.1° precision).
            If present and round(metar_6h_f) is above the model's predicted
            bracket, lock in to the bracket containing round(metar_6h_f).

    Returns:
        list of {"bracket": (low, high), "prob": float, "confidence": float | None}
    """
    auto_cols = model["auto_cols"]
    stage2_cols = model["stage2_cols"]
    stage1 = model["stage1_model"]
    stage2 = model["stage2_model"]
    scaler = model["stage2_scaler"]

    # --- Stage 1: auto-only probabilities ------------------------------------
    auto_vec = np.array([[features.get(c, 0.0) for c in auto_cols]])
    # Replace NaN with 0 for robustness
    auto_vec = np.nan_to_num(auto_vec, nan=0.0)
    auto_proba = stage1.predict_proba(auto_vec)[0]
    auto_classes = stage1.classes_

    # Build prob dict: {offset: probability}
    stage1_offset_probs = {int(c): float(p) for c, p in zip(auto_classes, auto_proba)}
    offset_probs = dict(stage1_offset_probs)

    # --- Build stage-2 feature vector ----------------------------------------
    # Add binned auto_prob columns to features dict for stage-2 lookup
    stage2_features = dict(features)
    for c, p in zip(auto_classes, auto_proba):
        stage2_features[f"auto_prob_{c:+d}"] = int(_bin_probs(np.array([p]))[0])

    # Consensus features (using binned values)
    auto_prob_plus1 = stage2_features.get("auto_prob_+1", 0.0)
    metar_confirm = stage2_features.get("metar_confirm", 0.0)
    stage2_features["consensus"] = auto_prob_plus1 * metar_confirm
    stage2_features["auto_metar_divergence"] = auto_prob_plus1 - metar_confirm

    stage2_vec = np.array([[stage2_features.get(c, 0.0) for c in stage2_cols]])
    stage2_vec = np.nan_to_num(stage2_vec, nan=0.0)
    stage2_vec_s = scaler.transform(stage2_vec)

    stage2_proba = stage2.predict_proba(stage2_vec_s)[0]
    stage2_classes = stage2.classes_

    # Raw 3-class probs from stage 2
    stage2_offset_probs = {int(c): float(p) for c, p in zip(stage2_classes, stage2_proba)}
    offset_probs = dict(stage2_offset_probs)

    # --- Apply clamp rules ---------------------------------------------------
    # Find the single best matching rule set (grouped by name, ranked by
    # confidence). Only ONE rule set fires — no cascading.
    clamp_confidence = None
    offset_clamp_reasons: Dict[int, List[str]] = {-1: [], 0: [], 1: []}

    # Group rules by name — rules with the same name form one "set"
    # (e.g. a force rule has two entries: 0→target and opposite→target)
    from collections import OrderedDict
    rule_groups: OrderedDict[str, list] = OrderedDict()
    for entry in CLAMP_RULES:
        name = entry[0]
        rule_groups.setdefault(name, []).append(entry)

    # Find matching groups, pick the one with highest confidence
    best_group = None
    best_conf = -1.0
    for group_name, entries in rule_groups.items():
        conf = entries[0][4]  # confidence is same for all entries in group
        if conf > best_conf:
            if _matches_rule(features, entries[0][1]):
                best_group = entries
                best_conf = conf

    if best_group is not None:
        for name, condition, from_val, to_val, confidence in best_group:
            moved = offset_probs.get(from_val, 0.0) * confidence
            if moved > 0:
                offset_probs[from_val] = offset_probs.get(from_val, 0.0) - moved
                offset_probs[to_val] = offset_probs.get(to_val, 0.0) + moved
                clamp_confidence = confidence
                reason_str = f"clamp {name} ({from_val:+d}→{to_val:+d}, {confidence:.0%})"
                offset_clamp_reasons[to_val].append(reason_str)

    # Ensure all three offsets exist
    for k in [-1, 0, 1]:
        offset_probs.setdefault(k, 0.0)
        stage1_offset_probs.setdefault(k, 0.0)
        stage2_offset_probs.setdefault(k, 0.0)

    # Snapshot offset probs at each stage for chart display
    _offset_detail = {
        "stage1": {k: stage1_offset_probs.get(k, 0.0) for k in [-1, 0, 1]},
        "stage2": {k: stage2_offset_probs.get(k, 0.0) for k in [-1, 0, 1]},
        "override": {k: offset_probs.get(k, 0.0) for k in [-1, 0, 1]},
        "override_reasons": {k: offset_clamp_reasons.get(k, []) for k in [-1, 0, 1]},
    }

    # --- Map offsets to candidate temps and then to brackets ------------------
    max_c = features.get("max_c")
    if max_c is None:
        return [{"bracket": b, "prob": 0.0, "confidence": None} for b in brackets]

    naive_f = round(float(max_c) * 9.0 / 5.0 + 32.0)
    # Candidate temps for each offset
    candidates = {
        -1: naive_f - 1,
         0: naive_f,
         1: naive_f + 1,
    }

    results = []
    for low, high in brackets:
        prob = 0.0
        offsets_in_bracket = []
        for offset, temp in candidates.items():
            if low <= temp <= high:
                prob += offset_probs.get(offset, 0.0)
                offsets_in_bracket.append(offset)
        # Map stage1 and stage2 offsets to this bracket too
        s1_prob = 0.0
        s2_prob = 0.0
        for offset, temp in candidates.items():
            if low <= temp <= high:
                s1_prob += stage1_offset_probs.get(offset, 0.0)
                s2_prob += stage2_offset_probs.get(offset, 0.0)
        # Per-bracket reason: show clamp rules that moved prob INTO this bracket's offsets
        bracket_clamp_reasons = []
        for o in sorted(offsets_in_bracket):
            bracket_clamp_reasons.extend(offset_clamp_reasons.get(o, []))
        if bracket_clamp_reasons:
            reason = "; ".join(bracket_clamp_reasons)
        else:
            offset_strs = [f"P({o:+d})={offset_probs.get(o, 0.0):.0%}"
                           for o in sorted(offsets_in_bracket)]
            reason = f"model: {' + '.join(offset_strs)}" if offset_strs else "model: no matching offset"
        results.append({
            "bracket": (low, high),
            "prob": prob,
            "stage1_prob": s1_prob,
            "stage2_prob": s2_prob,
            "confidence": clamp_confidence,
            "reason": reason,
            "offset_detail": _offset_detail,
        })

    # --- 6h METAR lock-in override -------------------------------------------
    # The 6h METAR max is settlement-grade (0.1°C precision, tracked by ASOS
    # continuously). When available it supersedes all model predictions —
    # round(metar_6h_f) is the best estimate of the CLI settlement value.
    if metar_6h_f is not None:
        metar_settled_f = round(metar_6h_f)
        model_top = max(results, key=lambda r: r["prob"])
        lock_reason = (f"6h METAR override: round({metar_6h_f:.1f})={metar_settled_f}°F "
                       f"(model predicted [{model_top['bracket'][0]}, {model_top['bracket'][1]}])")
        for r in results:
            lo, hi = r["bracket"]
            if lo <= metar_settled_f <= hi:
                r["prob"] = 0.99
                r["stage1_prob"] = 0.99
                r["confidence"] = 1.0
                r["metar_6h_lock"] = True
                r["reason"] = lock_reason
            else:
                r["prob"] = (1.0 - 0.99) / max(len(results) - 1, 1)
                r["stage1_prob"] = r["prob"]

    return results


# ---------------------------------------------------------------------------
# LOSO backtest
# ---------------------------------------------------------------------------

def train_loso(sites: Optional[List[str]] = None):
    """Leave-one-site-out backtest of the full 2-stage pipeline.

    For each site, trains stage1 (auto GBM) + stage2 (combined RF) on
    all other sites, predicts on the held-out site, and applies clamp
    rules.  Reports per-site and overall accuracy.
    """
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    from collections import OrderedDict

    if sites is None:
        sites = list(ALL_SITES)

    # --- Load solar noon cache ------------------------------------------------
    solar_noon_lookup: Dict[Tuple[str, str], float] = {}
    if os.path.isfile(SOLAR_NOON_CSV):
        sn_df = pd.read_csv(SOLAR_NOON_CSV)
        for _, row in sn_df.iterrows():
            solar_noon_lookup[(row["site"], str(row["date"]))] = float(row["solar_noon_hour"])

    # --- Collect features for all sites ---------------------------------------
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
    auto_cols = [c for c in AUTO_FEATURE_COLS if c in rdf.columns]
    metar_cols = [c for c in METAR_FEATURE_COLS if c in rdf.columns]

    feature_mask = rdf[auto_cols + metar_cols].notna().all(axis=1)
    rdf = rdf[feature_mask].reset_index(drop=True)
    n = len(rdf)

    y = rdf["offset"].values.astype(int)
    site_labels = rdf["site"].values
    unique_sites = sorted(set(site_labels))

    print("=" * 78)
    print("  BRACKET MODEL — LEAVE-ONE-SITE-OUT BACKTEST")
    print(f"  {n} days across {len(unique_sites)} sites")
    print("=" * 78)

    # Class distribution
    for cls in [-1, 0, 1]:
        cnt = int(np.sum(y == cls))
        print(f"  offset={cls:+d}: {cnt:5d} ({cnt / n * 100:5.1f}%)")

    always_0_total = int(np.sum(y == 0))
    print(f"  Baseline (always-0): {always_0_total / n:.1%}")

    # --- LOSO loop ------------------------------------------------------------
    print(f"\n  {'Site':<8s} {'N':>5s} {'Always0%':>10s}"
          f" {'S1 Acc%':>8s} {'S2 Acc%':>8s} {'S2+Clamp%':>10s}"
          f" {'Upper%':>8s} {'Middle%':>8s}")
    print(f"  {'-' * 8} {'-' * 5} {'-' * 10} {'-' * 8} {'-' * 8} {'-' * 10}"
          f" {'-' * 8} {'-' * 8}")

    _clamp_stats = {}
    all_y_true = []
    all_s1_pred = []
    all_s2_pred = []
    all_s2c_pred = []

    for site in unique_sites:
        test_mask = site_labels == site
        train_mask = ~test_mask
        n_test = int(test_mask.sum())
        if n_test == 0:
            continue

        y_train = y[train_mask]
        y_test = y[test_mask]

        # Stage 1: auto-only GBM
        X_auto_train = rdf.loc[train_mask, auto_cols].values
        X_auto_test = rdf.loc[test_mask, auto_cols].values

        s1 = GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            min_samples_leaf=5, random_state=42,
        )
        s1.fit(X_auto_train, y_train)
        s1_pred = s1.predict(X_auto_test)
        s1_acc = accuracy_score(y_test, s1_pred)

        # Build stage-2 features (stacked auto probs + METAR)
        auto_classes = s1.classes_
        prob_col_names = [f"auto_prob_{c:+d}" for c in auto_classes]

        # Out-of-fold S1 predictions for S2 training (eliminates leakage)
        from sklearn.model_selection import cross_val_predict
        train_proba = cross_val_predict(
            s1, X_auto_train, y_train,
            cv=5, method='predict_proba', n_jobs=-1,
        )
        s1.fit(X_auto_train, y_train)  # re-fit for test predictions
        rdf_train = rdf.loc[train_mask].copy()
        for i, col_name in enumerate(prob_col_names):
            rdf_train[col_name] = _bin_probs(train_proba[:, i])
        rdf_train["consensus"] = rdf_train.get("auto_prob_+1", 0.0) * rdf_train["metar_confirm"]
        rdf_train["auto_metar_divergence"] = rdf_train.get("auto_prob_+1", 0.0) - rdf_train["metar_confirm"]

        # Test proba (binned with same thresholds)
        test_proba = s1.predict_proba(X_auto_test)
        rdf_test = rdf.loc[test_mask].copy()
        for i, col_name in enumerate(prob_col_names):
            rdf_test[col_name] = _bin_probs(test_proba[:, i])
        rdf_test["consensus"] = rdf_test.get("auto_prob_+1", 0.0) * rdf_test["metar_confirm"]
        rdf_test["auto_metar_divergence"] = rdf_test.get("auto_prob_+1", 0.0) - rdf_test["metar_confirm"]

        consensus_cols = ["consensus", "auto_metar_divergence"]
        stage2_cols = metar_cols + prob_col_names + consensus_cols

        X_s2_train = rdf_train[stage2_cols].values
        X_s2_test = rdf_test[stage2_cols].values

        scaler = StandardScaler()
        X_s2_train_s = scaler.fit_transform(X_s2_train)
        X_s2_test_s = scaler.transform(X_s2_test)

        # Stage 2: combined RF
        s2 = RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=5,
            random_state=42, n_jobs=-1,
        )
        s2.fit(X_s2_train_s, y_train)
        s2_pred = s2.predict(X_s2_test_s)
        s2_acc = accuracy_score(y_test, s2_pred)

        # Apply clamp rules
        s2c_pred = s2_pred.copy()
        s2_proba = s2.predict_proba(X_s2_test_s)
        s2_classes = s2.classes_
        s2_offset_probs = np.zeros((n_test, 3))
        for ci, c in enumerate(s2_classes):
            col_idx = {-1: 0, 0: 1, 1: 2}.get(int(c))
            if col_idx is not None:
                s2_offset_probs[:, col_idx] = s2_proba[:, ci]

        # Build rule groups once per fold (not per row)
        rule_groups: OrderedDict[str, list] = OrderedDict()
        for entry in CLAMP_RULES:
            rule_groups.setdefault(entry[0], []).append(entry)

        for idx_in_test in range(n_test):
            row_feats = rdf_test.iloc[idx_in_test].to_dict()
            probs = {-1: s2_offset_probs[idx_in_test, 0],
                     0: s2_offset_probs[idx_in_test, 1],
                     1: s2_offset_probs[idx_in_test, 2]}

            s2_argmax = max(probs, key=probs.get)

            # Find best matching clamp group
            best_group = None
            best_conf = -1.0
            for group_name, entries in rule_groups.items():
                conf = entries[0][4]
                if conf > best_conf and _matches_rule(row_feats, entries[0][1]):
                    best_group = entries
                    best_conf = conf

            if best_group is not None:
                for _, _, from_val, to_val, confidence in best_group:
                    moved = probs.get(from_val, 0.0) * confidence
                    if moved > 0:
                        probs[from_val] -= moved
                        probs[to_val] = probs.get(to_val, 0.0) + moved

            clamped_argmax = max(probs, key=probs.get)
            s2c_pred[idx_in_test] = clamped_argmax

            # Track per-rule impact
            if best_group is not None and clamped_argmax != s2_argmax:
                rule_name = best_group[0][0]
                true_label = y_test[idx_in_test]
                was_right = (s2_argmax == true_label)
                now_right = (clamped_argmax == true_label)
                _clamp_stats.setdefault(rule_name, {"helped": 0, "hurt": 0, "neutral": 0})
                if now_right and not was_right:
                    _clamp_stats[rule_name]["helped"] += 1
                elif was_right and not now_right:
                    _clamp_stats[rule_name]["hurt"] += 1
                else:
                    _clamp_stats[rule_name]["neutral"] += 1

        s2c_acc = accuracy_score(y_test, s2c_pred)
        a0_acc = int(np.sum(y_test == 0)) / n_test

        # Bracket-upper: offset >= 0 (YES) vs offset == -1 (NO)
        upper_true = (y_test >= 0).astype(int)
        upper_pred = (s2c_pred >= 0).astype(int)
        upper_acc = accuracy_score(upper_true, upper_pred)

        # Bracket-middle: offset == +1 (UPPER) vs offset <= 0 (LOWER)
        mid_true = (y_test > 0).astype(int)
        mid_pred = (s2c_pred > 0).astype(int)
        mid_acc = accuracy_score(mid_true, mid_pred)

        print(f"  {site:<8s} {n_test:5d} {a0_acc:9.1%}"
              f" {s1_acc:7.1%} {s2_acc:7.1%} {s2c_acc:9.1%}"
              f" {upper_acc:7.1%} {mid_acc:7.1%}")

        all_y_true.extend(y_test)
        all_s1_pred.extend(s1_pred)
        all_s2_pred.extend(s2_pred)
        all_s2c_pred.extend(s2c_pred)

    # --- Overall summary ------------------------------------------------------
    all_y_true = np.array(all_y_true)
    all_s1_pred = np.array(all_s1_pred)
    all_s2_pred = np.array(all_s2_pred)
    all_s2c_pred = np.array(all_s2c_pred)
    n_total = len(all_y_true)

    if n_total > 0:
        a0_total = int(np.sum(all_y_true == 0))

        upper_true_all = (all_y_true >= 0).astype(int)
        upper_pred_all = (all_s2c_pred >= 0).astype(int)
        mid_true_all = (all_y_true > 0).astype(int)
        mid_pred_all = (all_s2c_pred > 0).astype(int)

        print(f"\n  {'TOTAL':<8s} {n_total:5d} {a0_total / n_total:9.1%}"
              f" {accuracy_score(all_y_true, all_s1_pred):7.1%}"
              f" {accuracy_score(all_y_true, all_s2_pred):7.1%}"
              f" {accuracy_score(all_y_true, all_s2c_pred):9.1%}"
              f" {accuracy_score(upper_true_all, upper_pred_all):7.1%}"
              f" {accuracy_score(mid_true_all, mid_pred_all):7.1%}")

        # Classification report for final (stage2 + clamp)
        print(f"\n{'=' * 78}")
        print(f"  CLASSIFICATION REPORT (LOSO, Stage2 + Clamp)")
        print(f"{'=' * 78}")
        labels = sorted(set(all_y_true))
        target_names = [f"offset={c:+d}" for c in labels]
        print(classification_report(all_y_true, all_s2c_pred, labels=labels,
                                    target_names=target_names, zero_division=0))

        print(f"  Confusion matrix (rows=true, cols=predicted):")
        cm = confusion_matrix(all_y_true, all_s2c_pred, labels=labels)
        header = "        " + "  ".join(f"{c:+d}" for c in labels)
        print(f"  {header}")
        for i, cls in enumerate(labels):
            row_str = "  ".join(f"{cm[i, j]:4d}" for j in range(len(labels)))
            print(f"    {cls:+d}:  {row_str}")

        # Per-rule clamp impact
        if _clamp_stats:
            print(f"\n{'=' * 78}")
            print(f"  CLAMP RULE IMPACT (only rows where clamp changed prediction)")
            print(f"{'=' * 78}")
            print(f"  {'Rule':<35s} {'Helped':>7s} {'Hurt':>7s} {'Neutral':>8s} {'Net':>5s}")
            print(f"  {'-'*35} {'-'*7} {'-'*7} {'-'*8} {'-'*5}")
            total_helped = total_hurt = total_neutral = 0
            for rule, stats in sorted(_clamp_stats.items()):
                h, hu, n = stats["helped"], stats["hurt"], stats["neutral"]
                total_helped += h
                total_hurt += hu
                total_neutral += n
                print(f"  {rule:<35s} {h:7d} {hu:7d} {n:8d} {h - hu:+5d}")
            print(f"  {'-'*35} {'-'*7} {'-'*7} {'-'*8} {'-'*5}")
            print(f"  {'TOTAL':<35s} {total_helped:7d} {total_hurt:7d} {total_neutral:8d}"
                  f" {total_helped - total_hurt:+5d}")

    # --- Feature importance (train full model on all data) --------------------
    if n_total > 0:
        print(f"\n{'=' * 78}")
        print(f"  FEATURE IMPORTANCE")
        print(f"{'=' * 78}")

        # Stage 1: auto-only GBM on full data
        X_auto_full = rdf[auto_cols].values
        s1_full = GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            min_samples_leaf=5, random_state=42,
        )
        s1_full.fit(X_auto_full, y)

        s1_imp = sorted(zip(auto_cols, s1_full.feature_importances_),
                        key=lambda x: -x[1])
        print(f"\n  Stage 1 (GBM, auto-only) — top 20:")
        print(f"  {'Feature':<35s} {'Importance':>10s}")
        print(f"  {'-'*35} {'-'*10}")
        for feat, imp in s1_imp[:20]:
            print(f"  {feat:<35s} {imp:10.4f}")

        # Stage 2: build stacked features and train RF on full data
        auto_proba_full = s1_full.predict_proba(X_auto_full)
        auto_classes_full = s1_full.classes_
        prob_col_names_full = [f"auto_prob_{c:+d}" for c in auto_classes_full]

        rdf_full = rdf.copy()
        for i, col_name in enumerate(prob_col_names_full):
            rdf_full[col_name] = _bin_probs(auto_proba_full[:, i])
        rdf_full["consensus"] = rdf_full["auto_prob_+1"] * rdf_full["metar_confirm"]
        rdf_full["auto_metar_divergence"] = rdf_full["auto_prob_+1"] - rdf_full["metar_confirm"]

        consensus_cols = ["consensus", "auto_metar_divergence"]
        stage2_cols_full = metar_cols + prob_col_names_full + consensus_cols
        X_s2_full = rdf_full[stage2_cols_full].values
        scaler_full = StandardScaler()
        X_s2_full_s = scaler_full.fit_transform(X_s2_full)

        s2_full = RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=5,
            random_state=42, n_jobs=-1,
        )
        s2_full.fit(X_s2_full_s, y)

        s2_imp = sorted(zip(stage2_cols_full, s2_full.feature_importances_),
                        key=lambda x: -x[1])
        print(f"\n  Stage 2 (RF, METAR + binned auto probs) — all {len(stage2_cols_full)} features:")
        print(f"  {'Feature':<35s} {'Importance':>10s}")
        print(f"  {'-'*35} {'-'*10}")
        for feat, imp in s2_imp:
            print(f"  {feat:<35s} {imp:10.4f}")

        # --- Feature distributions per label ----------------------------------
        print(f"\n{'=' * 78}")
        print(f"  FEATURE DISTRIBUTIONS PER LABEL (mean ± std)")
        print(f"{'=' * 78}")

        print(f"\n  Stage 1 features (top 15):")
        top_s1_feats = [f for f, _ in s1_imp[:15]]
        print(f"  {'Feature':<30s} {'off=-1':>18s} {'off=0':>18s} {'off=+1':>18s}")
        print(f"  {'-'*30} {'-'*18} {'-'*18} {'-'*18}")
        for feat in top_s1_feats:
            vals = rdf[feat]
            parts = []
            for cls in [-1, 0, 1]:
                v = vals[y == cls]
                parts.append(f"{v.mean():8.3f} ± {v.std():6.3f}")
            print(f"  {feat:<30s} {parts[0]:>18s} {parts[1]:>18s} {parts[2]:>18s}")

        print(f"\n  Stage 2 features:")
        print(f"  {'Feature':<30s} {'off=-1':>18s} {'off=0':>18s} {'off=+1':>18s}")
        print(f"  {'-'*30} {'-'*18} {'-'*18} {'-'*18}")
        for feat, _ in s2_imp:
            if feat in rdf_full.columns:
                vals = rdf_full[feat]
                parts = []
                for cls in [-1, 0, 1]:
                    v = vals[y == cls]
                    parts.append(f"{v.mean():8.3f} ± {v.std():6.3f}")
                print(f"  {feat:<30s} {parts[0]:>18s} {parts[1]:>18s} {parts[2]:>18s}")


# ---------------------------------------------------------------------------
# S1 model sweep
# ---------------------------------------------------------------------------

def sweep_s1(sites: Optional[List[str]] = None):
    """Sweep different Stage-1 models via LOSO and compare end-to-end accuracy.

    Loads data once, then runs the full S1→CV→S2→clamp LOSO pipeline for each
    candidate S1 model.  Prints a comparison table.
    """
    import warnings
    warnings.filterwarnings("ignore")

    from sklearn.ensemble import (
        GradientBoostingClassifier, RandomForestClassifier,
        ExtraTreesClassifier, AdaBoostClassifier,
    )
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import accuracy_score
    from collections import OrderedDict

    try:
        from sklearn.experimental import enable_hist_gradient_boosting  # noqa: F401
        from sklearn.ensemble import HistGradientBoostingClassifier
        has_histgbm = True
    except ImportError:
        has_histgbm = False

    if sites is None:
        sites = list(ALL_SITES)

    # --- Load data (same as train_loso) ---------------------------------------
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

    rdf = pd.DataFrame(rows)
    auto_cols = [c for c in AUTO_FEATURE_COLS if c in rdf.columns]
    metar_cols = [c for c in METAR_FEATURE_COLS if c in rdf.columns]

    feature_mask = rdf[auto_cols + metar_cols].notna().all(axis=1)
    rdf = rdf[feature_mask].reset_index(drop=True)
    n = len(rdf)
    y = rdf["offset"].values.astype(int)
    site_labels = rdf["site"].values
    unique_sites = sorted(set(site_labels))

    # --- Candidate S1 models --------------------------------------------------
    candidates: List[Tuple[str, object]] = [
        # Baselines
        ("LogReg", LogisticRegression(
            max_iter=1000, multi_class="multinomial", random_state=42)),
        # Current
        ("GBM d4/200/lr0.1 (current)", GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            min_samples_leaf=5, random_state=42)),
        # GBM variants (depth + learning rate)
        ("GBM d3/200/lr0.1", GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.1,
            min_samples_leaf=5, random_state=42)),
        ("GBM d5/200/lr0.1", GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            min_samples_leaf=5, random_state=42)),
        ("GBM d4/300/lr0.05", GradientBoostingClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            min_samples_leaf=5, random_state=42)),
        # Tree ensembles (parallel-friendly → fast)
        ("RF 200/d10", RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_leaf=5,
            random_state=42, n_jobs=-1)),
        ("RF 300/dNone", RandomForestClassifier(
            n_estimators=300, max_depth=None, min_samples_leaf=5,
            random_state=42, n_jobs=-1)),
        ("ExtraTrees 300/d10", ExtraTreesClassifier(
            n_estimators=300, max_depth=10, min_samples_leaf=5,
            random_state=42, n_jobs=-1)),
        ("ExtraTrees 300/dNone", ExtraTreesClassifier(
            n_estimators=300, max_depth=None, min_samples_leaf=5,
            random_state=42, n_jobs=-1)),
        # AdaBoost
        ("AdaBoost 200", AdaBoostClassifier(
            n_estimators=200, learning_rate=0.1, random_state=42)),
    ]

    if has_histgbm:
        candidates.extend([
            ("HistGBM d4/200/lr0.1", HistGradientBoostingClassifier(
                max_iter=200, max_depth=4, learning_rate=0.1,
                min_samples_leaf=5, random_state=42)),
            ("HistGBM d6/300/lr0.05", HistGradientBoostingClassifier(
                max_iter=300, max_depth=6, learning_rate=0.05,
                min_samples_leaf=5, random_state=42)),
        ])

    # --- S1-only LOSO evaluation ------------------------------------------------
    def _loso_s1(s1_factory):
        """LOSO with S1 only — just 17 fits, no S2/CV/clamps."""
        from sklearn.base import clone
        all_y_true, all_s1_pred = [], []

        for site in unique_sites:
            test_mask = site_labels == site
            train_mask = ~test_mask
            if test_mask.sum() == 0:
                continue

            s1 = clone(s1_factory)
            s1.fit(X_auto[train_mask], y[train_mask])
            all_y_true.extend(y[test_mask])
            all_s1_pred.extend(s1.predict(X_auto[test_mask]))

        all_y_true = np.array(all_y_true)
        all_s1_pred = np.array(all_s1_pred)
        acc = accuracy_score(all_y_true, all_s1_pred)
        upper_true = (all_y_true >= 0).astype(int)
        upper_pred = (all_s1_pred >= 0).astype(int)
        upper_acc = accuracy_score(upper_true, upper_pred)
        return acc, upper_acc

    X_auto = rdf[auto_cols].values

    # --- Run sweep ------------------------------------------------------------
    import time as _time

    print("=" * 78)
    print("  S1 MODEL SWEEP — LOSO (Stage 1 only, no S2)")
    print(f"  {n} days across {len(unique_sites)} sites")
    print("=" * 78)
    print(f"\n  {'Model':<30s} {'S1 Acc%':>8s} {'Upper%':>8s}")
    print(f"  {'-'*30} {'-'*8} {'-'*8}", flush=True)

    results = []
    for name, model in candidates:
        t0 = _time.time()
        try:
            acc, upper_acc = _loso_s1(model)
            elapsed = _time.time() - t0
            print(f"  {name:<30s} {acc:7.1%} {upper_acc:7.1%}  ({elapsed:.0f}s)",
                  flush=True)
            results.append((name, acc, upper_acc))
        except Exception as e:
            elapsed = _time.time() - t0
            print(f"  {name:<30s} {'FAILED':>8s}  {str(e)[:50]}  ({elapsed:.0f}s)",
                  flush=True)

    # --- Summary --------------------------------------------------------------
    if results:
        print(f"\n{'=' * 78}")
        best = max(results, key=lambda r: r[1])
        best_upper = max(results, key=lambda r: r[2])
        print(f"  Best S1 Acc:   {best[0]:<30s} {best[1]:.1%}")
        print(f"  Best Upper:    {best_upper[0]:<30s} {best_upper[2]:.1%}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train bracket probability model")
    parser.add_argument("--sites", type=str, default=None,
                        help="Comma-separated ICAO codes (default: all)")
    parser.add_argument("--save", action="store_true",
                        help="Train on all data and save model to disk")
    parser.add_argument("--sweep-s1", action="store_true",
                        help="Sweep different S1 models via LOSO")
    args = parser.parse_args()

    sites = args.sites.split(",") if args.sites else None
    if args.sweep_s1:
        sweep_s1(sites=sites)
    elif args.save:
        path = train(sites=sites)
        print(path)
    else:
        train_loso(sites=sites)
