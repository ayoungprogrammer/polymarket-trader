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
# Clamp rules — structural overrides applied after model prediction
# ---------------------------------------------------------------------------
# Each rule: (name, condition_dict, from_val, to_val, confidence)
#   condition_dict: {feature: value} for equality, {feature__gte: value} for >=
CLAMP_RULES = [
    ("naive_is_low→0",   {"naive_is_high": 0, "n_possible_f": 2}, -1, 0, 0.999),
    ("naive_is_high→0",  {"naive_is_high": 1},                     1, 0, 0.946),
    ("exact→0",          {"n_possible_f": 1},                     -1, 0, 0.928),
    ("exact→0",          {"n_possible_f": 1},                      1, 0, 0.928),
    ("metar_above→0",    {"metar_above_boundary": 1},             -1, 0, 1.000),
    ("metar_gap≥.25→+1", {"metar_gap_c__gte": 0.25},              0, 1, 1.000),
    ("metar_gap≥.25→+1", {"metar_gap_c__gte": 0.25},             -1, 1, 1.000),
]


def _matches_rule(features: dict, condition: dict) -> bool:
    """Check if a single row's features match a clamp rule condition."""
    for key, threshold in condition.items():
        if key.endswith("__gte"):
            feat_key = key[:-5]
            val = features.get(feat_key)
            if val is None or np.isnan(val) or val < threshold:
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
        n_estimators=200, max_depth=4, learning_rate=0.1,
        min_samples_leaf=5, random_state=42,
    )
    stage1_model.fit(X_auto, y_3class)

    # Get stage-1 predict_proba on training data for stage-2 features
    auto_proba = stage1_model.predict_proba(X_auto)
    auto_classes = stage1_model.classes_  # e.g. [-1, 0, 1]
    prob_col_names = [f"auto_prob_{c:+d}" for c in auto_classes]

    for i, col_name in enumerate(prob_col_names):
        rdf[col_name] = auto_proba[:, i]

    # Consensus features
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
    # Add auto_prob columns to features dict for stage-2 lookup
    stage2_features = dict(features)
    for c, p in zip(auto_classes, auto_proba):
        stage2_features[f"auto_prob_{c:+d}"] = p

    # Consensus features
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
    offset_probs = {int(c): float(p) for c, p in zip(stage2_classes, stage2_proba)}

    # --- Apply clamp rules ---------------------------------------------------
    clamp_confidence = None
    clamp_reasons: List[str] = []
    for name, condition, from_val, to_val, confidence in CLAMP_RULES:
        if _matches_rule(features, condition):
            moved = offset_probs.get(from_val, 0.0) * confidence
            if moved > 0:
                offset_probs[from_val] = offset_probs.get(from_val, 0.0) - moved
                offset_probs[to_val] = offset_probs.get(to_val, 0.0) + moved
                clamp_confidence = confidence
                clamp_reasons.append(f"clamp {name} (offset {from_val:+d}→{to_val:+d}, {confidence:.0%})")

    # Ensure all three offsets exist
    for k in [-1, 0, 1]:
        offset_probs.setdefault(k, 0.0)

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

    # Build reason string for model-only predictions
    if clamp_reasons:
        model_reason = "; ".join(clamp_reasons)
    else:
        model_reason = None  # will be filled per-bracket below

    results = []
    for low, high in brackets:
        prob = 0.0
        offsets_in_bracket = []
        for offset, temp in candidates.items():
            if low <= temp <= high:
                prob += offset_probs.get(offset, 0.0)
                offsets_in_bracket.append(offset)
        # Map stage1 offsets to this bracket too
        s1_prob = 0.0
        for offset, temp in candidates.items():
            if low <= temp <= high:
                s1_prob += stage1_offset_probs.get(offset, 0.0)
        # Per-bracket reason
        if model_reason:
            reason = model_reason
        else:
            offset_strs = [f"P(offset={o:+d})={offset_probs.get(o, 0.0):.0%}"
                           for o in sorted(offsets_in_bracket)]
            reason = f"model: {' + '.join(offset_strs)}" if offset_strs else "model: no matching offset"
        results.append({
            "bracket": (low, high),
            "prob": prob,
            "stage1_prob": s1_prob,
            "confidence": clamp_confidence,
            "reason": reason,
        })

    # --- 6h METAR lock-in override -------------------------------------------
    # The 6h METAR max is settlement-grade (0.1°C precision, tracked by ASOS
    # continuously). If round(metar_6h_f) lands in a bracket above the model's
    # top prediction, lock to that bracket — the model's auto-obs features
    # can't see beyond whole-°C, but the 6h report already confirms the peak.
    if metar_6h_f is not None:
        metar_settled_f = round(metar_6h_f)
        # Find which bracket the model currently predicts
        model_top = max(results, key=lambda r: r["prob"])
        model_top_hi = model_top["bracket"][1]
        # Only override if 6h METAR settles above the model's predicted bracket
        if metar_settled_f > model_top_hi:
            lock_reason = (f"6h METAR lock: round({metar_6h_f:.1f})={metar_settled_f}°F "
                           f"> model top bracket [{model_top['bracket'][0]}, {model_top_hi}]")
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
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train bracket probability model")
    parser.add_argument("--sites", type=str, default=None,
                        help="Comma-separated ICAO codes (default: all)")
    args = parser.parse_args()

    sites = args.sites.split(",") if args.sites else None
    path = train(sites=sites)
    print(path)
