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
import re
import sys
from datetime import datetime
from typing import Optional, List, Dict, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from weather.backtest_rounding import (
    FEATURE_COLS,
    AUTO_FEATURE_COLS,
    SOLAR_NOON_CSV,
    extract_regression_features,
)
from weather.sites import ALL_SITES, ALL_SITES_WITH_TRAINING
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

# Post-model probability overrides based on empirical patterns in the data.
# Each rule moves probability mass between offset classes {-1, 0, +1}.
#
# Tuple format: (name, condition, from_class, to_class, confidence[, max_model_prob])
#   name         — human-readable label
#   condition    — dict of feature checks (exact match, __gte, __lte suffixes)
#   from_class   — offset class to take probability FROM (-1, 0, or +1)
#   to_class     — offset class to give probability TO (or None to redistribute proportionally)
#   confidence   — fraction of from_class's probability to move (0.0–1.0)
#   max_model_prob (optional) — only fire if model's P(to_class) is below this;
#                               skips the rule when the model is already confident
#
# Rules are grouped by name. Only the highest-confidence matching group fires.
# Two entries per "force" rule: one moves P(0)→target, the other P(opposite)→target.
CLAMP_RULES = [
    # --- Force +1: METAR confirms true high above auto max ---
    # Two entries per group: zero P(0) and clamp P(-1) → all to +1
    # Ordered by specificity (most specific first); only one group fires.
    ("cfm≥.25→+1",          {"metar_confirm__gte": 0.25},                           0,  1, 0.986),  # n=1084, 1.4% stays 0
    ("cfm≥.25→+1",          {"metar_confirm__gte": 0.25},                          -1,  1, 0.977),  # n=1084, 2.3% stays -1
    ("cfm≥.2+nih0→+1",     {"metar_confirm__gte": 0.2, "naive_is_high": 0},       0,  1, 1.0),   # n=1070
    ("cfm≥.2+nih0→+1",     {"metar_confirm__gte": 0.2, "naive_is_high": 0},      -1,  1, 0.964),  # n=1070, 96.4% +1
    # ("mod5=3+cfm≥.1→+1",   {"max_c_mod5": 3.0, "metar_confirm__gte": 0.1},       0,  1, 1.0),   # n=521
    # ("mod5=3+cfm≥.1→+1",   {"max_c_mod5": 3.0, "metar_confirm__gte": 0.1},      -1,  1, 0.96),  # n=521
    # ("mod5=4+cfm≥.1→+1",   {"max_c_mod5": 4.0, "metar_confirm__gte": 0.1},       0,  1, 1.0),   # n=472
    # ("mod5=4+cfm≥.1→+1",   {"max_c_mod5": 4.0, "metar_confirm__gte": 0.1},      -1,  1, 0.96),  # n=472
    # ("gap≥.25→+1",          {"metar_gap_c__gte": 0.25},                            0,  1, 1.0),   # n=1084
    # ("gap≥.25→+1",          {"metar_gap_c__gte": 0.25},                           -1,  1, 0.963),  # n=1084, 96.3% +1
    # --- METAR confirm + naive_is_low: cap P(-1) (data: 2.8% -1 across all thresholds,
    #     but that's driven by nih=0 cases; for nih=1, -1 is still the likely outcome) ---
    ("cfm>0+nih0: cap -1",  {"metar_confirm__gte": 0.01, "naive_is_high": 0},     -1, None, 0.97),  # n~1200
    # --- Cap P(-1): redistribute excess to 0/+1 ---
    ("dwell≥20+nih0: cap -1", {"dwell_count__gte": 10, "naive_is_high": 0},        -1, None, 0.97),  # n=2984
    ("consec≥30+nih0: cap -1", {"consec_count__gte": 10, "naive_is_high": 0},      -1, None, 0.98),  # n=719
    # --- Cap P(+1): METAR confirms below & naive rounds high ---
    ("metar_gap≤-.5+nih1: cap +1", {"metar_gap_c__lte": -0.5, "naive_is_high": 1},       1, None, 0.95),  # n=973
    # # --- Conditional force -1: only override when model P(-1) < 94.7% ---
    # ("single+nih1→-1",      {"single_reading_peak": 1, "naive_is_high": 1},        0, -1, 0.947, 0.947),  # n=550, 94.7% -1
    # ("single+nih1→-1",      {"single_reading_peak": 1, "naive_is_high": 1},        1, -1, 0.947, 0.947),  # n=550, 94.7% -1
    # --- Backstops: cap the rare class based on rounding position ---
    ("nih0: cap -1",         {"naive_is_high": 0},                                 -1, None, 0.965),  # n=5052, only 3.5% -1
    ("nih1: cap +1",         {"naive_is_high": 1},                                  1, None, 0.945),  # n=4091, only 5.5% +1
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

def train(sites: Optional[List[str]] = None, since: Optional[str] = None) -> str:
    """Train the 2-stage bracket model on full dataset and persist to disk.

    Args:
        sites: ICAO codes to include (default: Kalshi sites).
        since: Only include days on or after this date (YYYY-MM-DD).

    Returns the path to the saved model pickle.
    """
    from sklearn.experimental import enable_hist_gradient_boosting  # noqa: F401
    from sklearn.ensemble import HistGradientBoostingClassifier

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
            if since and str(date) < since:
                continue
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
    feature_cols = [c for c in FEATURE_COLS if c in rdf.columns]
    auto_cols = [c for c in AUTO_FEATURE_COLS if c in rdf.columns]

    # Only require non-NaN for core features; met features can be NaN
    from weather.backtest_rounding import _MET_FEATURE_NAMES
    required_cols = [c for c in feature_cols if c not in _MET_FEATURE_NAMES]
    feature_mask = rdf[required_cols].notna().all(axis=1)
    rdf = rdf[feature_mask].reset_index(drop=True)
    n = len(rdf)

    y_3class = rdf["offset"].values.astype(int)

    # =========================================================================
    # Stage 1: HistGBM on auto-only features (no METAR)
    # =========================================================================
    X_auto = rdf[auto_cols].values

    stage1_model = HistGradientBoostingClassifier(
        max_iter=200, max_depth=7, learning_rate=0.05,
        min_samples_leaf=40, random_state=42,
    )
    stage1_model.fit(X_auto, y_3class)

    # In-sample S1 predictions for S2 training (NOT out-of-fold).
    #
    # backtest_rounding.py uses OOF S1 probs → OOF S2 eval, which is
    # consistent because both stages see OOF distributions. But a saved
    # model can't do OOF at inference — it runs the full-fit S1, producing
    # sharper/more confident probs than OOF. If S2 was trained on OOF S1
    # probs, it sees a different distribution at inference → worse accuracy.
    #
    # Using in-sample S1 probs here means S2 trains on the same distribution
    # it will see in production. Overfitting risk is low because S1 is
    # well-regularized (max_depth=7, min_samples_leaf=40, 15k+ rows).
    auto_proba = stage1_model.predict_proba(X_auto)
    auto_classes = stage1_model.classes_  # e.g. [-1, 0, 1]
    prob_col_names = [f"auto_prob_{c:+d}" for c in auto_classes]

    for i, col_name in enumerate(prob_col_names):
        rdf[col_name] = auto_proba[:, i]

    # Consensus features (using raw S1 probabilities)
    rdf["consensus"] = rdf["auto_prob_+1"] * rdf["metar_confirm"]
    rdf["auto_metar_divergence"] = rdf["auto_prob_+1"] - rdf["metar_confirm"]

    # =========================================================================
    # Stage 2: HistGBM on stacked S1 probs + METAR + consensus (no scaler)
    # =========================================================================
    metar_raw_cols = ["metar_confirm", "metar_gap_c"]
    consensus_cols = ["consensus", "auto_metar_divergence"]
    stage2_cols = prob_col_names + metar_raw_cols + consensus_cols

    X_stage2 = rdf[stage2_cols].values

    stage2_model = HistGradientBoostingClassifier(
        max_iter=200, max_depth=5, learning_rate=0.1,
        min_samples_leaf=5, random_state=42,
    )
    stage2_model.fit(X_stage2, y_3class)

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


def map_offsets_to_brackets(
    candidates: Dict[int, int],
    naive_f: int,
    probabilities: Dict[int, float],
    brackets: List[Tuple[int, int]],
    stage1_offset_probs: Optional[Dict[int, float]] = None,
    stage2_offset_probs: Optional[Dict[int, float]] = None,
    offset_clamp_reasons: Optional[Dict[int, List[str]]] = None,
    clamp_confidence: Optional[float] = None,
    offset_detail: Optional[dict] = None,
) -> List[dict]:
    """Map offset probabilities to Kalshi bracket results.

    Parameters
    ----------
    candidates : dict
        {offset: temp_f} mapping, e.g. {-1: 76, 0: 77, 1: 78}.
    naive_f : int
        Current naive settlement °F (round(max_c * 9/5 + 32)).
    probabilities : dict
        {offset: probability} from the model pipeline.
    brackets : list of (low, high)
        Kalshi bracket bounds. Sentinels: -1000 = "or below", 1000 = "or above".
    stage1_offset_probs : dict, optional
        Stage-1 offset probs for display.
    stage2_offset_probs : dict, optional
        Stage-2 offset probs for display.
    offset_clamp_reasons : dict, optional
        {offset: [reason_str, ...]} from clamp rules.
    clamp_confidence : float or None
        Confidence of the clamp rule that fired.
    offset_detail : dict or None
        Per-stage offset probability snapshot for chart display.

    Returns
    -------
    list of {"bracket": (low, high), "prob": float, "confidence": float | None, ...}
    """
    s1 = stage1_offset_probs or {}
    s2 = stage2_offset_probs or {}
    clamp_reasons = offset_clamp_reasons or {}

    # Clamp open-ended bracket sentinels so offset mapping works correctly.
    # Sentinels (-1000 for "or below", 1000 for "or above") would otherwise
    # match all candidate temps and absorb 100% of probability.
    clamped_brackets = []
    for low, high in brackets:
        lo = low if low > -1000 else naive_f - 10
        hi = high if high < 1000 else naive_f + 10
        clamped_brackets.append((lo, hi))

    results = []
    for (low, high), (orig_lo, orig_hi) in zip(clamped_brackets, brackets):
        prob = 0.0
        offsets_in_bracket = []
        for offset, temp in candidates.items():
            if low <= temp <= high:
                prob += probabilities.get(offset, 0.0)
                offsets_in_bracket.append(offset)
        # Map stage1 and stage2 offsets to this bracket too
        s1_prob = 0.0
        s2_prob = 0.0
        for offset, temp in candidates.items():
            if low <= temp <= high:
                s1_prob += s1.get(offset, 0.0)
                s2_prob += s2.get(offset, 0.0)
        # Per-bracket reason: show clamp rules that moved prob INTO this bracket's offsets
        bracket_clamp_reasons = []
        for o in sorted(offsets_in_bracket):
            bracket_clamp_reasons.extend(clamp_reasons.get(o, []))
        if bracket_clamp_reasons:
            reason = "; ".join(bracket_clamp_reasons)
        else:
            offset_strs = [f"P({o:+d})={probabilities.get(o, 0.0):.0%}"
                           for o in sorted(offsets_in_bracket)]
            reason = f"model: {' + '.join(offset_strs)}" if offset_strs else "model: no matching offset"
        entry: dict = {
            "bracket": (orig_lo, orig_hi),
            "prob": prob,
            "stage1_prob": s1_prob,
            "stage2_prob": s2_prob,
            "confidence": clamp_confidence,
            "reason": reason,
        }
        if offset_detail is not None:
            entry["offset_detail"] = offset_detail
        results.append(entry)

    return results


def _find_peak_metar(
    readings: List[dict],
    site_timezone: str,
    forecast_peak_hour: float,
) -> Optional[dict]:
    """Find the METAR reading whose 6h window contains the forecast peak hour.

    Each METAR at local hour H covers the window [H-6, H).
    Returns the reading whose window contains forecast_peak_hour, or None.
    """
    tz = ZoneInfo(site_timezone)
    for r in readings:
        ts_str = r["timestamp_utc"]
        # Parse UTC timestamp
        # Normalize tz offset: -0400 → -04:00 (Python 3.9 needs the colon)
        ts_str = re.sub(r'([+-]\d{2})(\d{2})$', r'\1:\2', ts_str)
        ts_utc = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        local_hour = ts_utc.astimezone(tz).hour + ts_utc.astimezone(tz).minute / 60.0
        window_start = local_hour - 6.0
        # Handle wrap-around (e.g. METAR at 5 AM covers 23:00-05:00)
        if window_start < 0:
            # Window wraps midnight: [24+window_start, 24) or [0, local_hour)
            if forecast_peak_hour >= (24.0 + window_start) or forecast_peak_hour < local_hour:
                return r
        else:
            if window_start <= forecast_peak_hour < local_hour:
                return r
    return None


def get_probability(
    model: dict,
    features: dict,
    brackets: List[Tuple[int, int]],
    metar_6h_f: Optional[float] = None,
    metar_6h_readings: Optional[List[dict]] = None,
    site_timezone: Optional[str] = None,
    auto_max_since_last_metar_f: Optional[float] = None,
    forecast_peak_hour: Optional[float] = None,
    cli_high_f: Optional[int] = None,
) -> List[dict]:
    """Compute bracket probabilities from extracted features.

    Args:
        model: dict from load_model()
        features: dict from extract_regression_features() (single day)
        brackets: list of (low, high) tuples — Kalshi bracket bounds
        metar_6h_f: optional 6h METAR max in °F (0.1° precision).
            If present and round(metar_6h_f) is above the model's predicted
            bracket, lock in to the bracket containing round(metar_6h_f).
        metar_6h_readings: optional list of all 6h METAR readings today,
            each {"timestamp_utc": str, "value_f": float}.
        site_timezone: IANA timezone for the station (e.g. "America/Los_Angeles").
        auto_max_since_last_metar_f: highest auto obs temp (°F) since the
            last 6h METAR timestamp.
        forecast_peak_hour: NWS forecast peak hour in local time (decimal,
            e.g. 14.5 = 2:30 PM).
        cli_high_f: optional CLI (Daily Summary) reported high in whole °F.
            This is the actual settlement value — if present, lock directly
            to the bracket containing this value at 99%.

    Returns:
        list of {"bracket": (low, high), "prob": float, "confidence": float | None}
    """
    auto_cols = model["auto_cols"]
    stage2_cols = model["stage2_cols"]
    stage1 = model["stage1_model"]
    stage2 = model["stage2_model"]
    stage2_scaler = model.get("stage2_scaler")

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
    # Add raw S1 probabilities to features dict for stage-2 lookup
    stage2_features = dict(features)
    for c, p in zip(auto_classes, auto_proba):
        stage2_features[f"auto_prob_{c:+d}"] = float(p)

    # Consensus features (using raw S1 probabilities)
    auto_prob_plus1 = stage2_features.get("auto_prob_+1", 0.0)
    metar_confirm = stage2_features.get("metar_confirm", 0.0)
    stage2_features["consensus"] = auto_prob_plus1 * metar_confirm
    stage2_features["auto_metar_divergence"] = auto_prob_plus1 - metar_confirm

    stage2_vec = np.array([[stage2_features.get(c, 0.0) for c in stage2_cols]])
    stage2_vec = np.nan_to_num(stage2_vec, nan=0.0)
    if stage2_scaler is not None:
        stage2_vec = stage2_scaler.transform(stage2_vec)

    stage2_proba = stage2.predict_proba(stage2_vec)[0]
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
                # Check max_model_prob gate (6th element if present)
                max_model_prob = entries[0][5] if len(entries[0]) > 5 else None
                if max_model_prob is not None:
                    target_class = entries[0][3]
                    if offset_probs.get(target_class, 0.0) >= max_model_prob:
                        continue  # model already confident enough, skip
                best_group = entries
                best_conf = conf

    if best_group is not None:
        for entry in best_group:
            name, condition, from_val, to_val, confidence = entry[0], entry[1], entry[2], entry[3], entry[4]
            moved = offset_probs.get(from_val, 0.0) * confidence
            if moved > 0:
                offset_probs[from_val] = offset_probs.get(from_val, 0.0) - moved
                if to_val is None:
                    # Redistribute proportionally to remaining classes
                    others = [k for k in [-1, 0, 1] if k != from_val]
                    total = sum(offset_probs.get(k, 0.0) for k in others)
                    for k in others:
                        share = offset_probs.get(k, 0.0) / total if total > 0 else 0.5
                        offset_probs[k] = offset_probs.get(k, 0.0) + moved * share
                    clamp_confidence = confidence
                    reason_str = f"clamp {name} (zero {from_val:+d}, {confidence:.0%})"
                    offset_clamp_reasons[from_val].append(reason_str)
                else:
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

    results = map_offsets_to_brackets(
        candidates, naive_f, offset_probs, brackets,
        stage1_offset_probs=stage1_offset_probs,
        stage2_offset_probs=stage2_offset_probs,
        offset_clamp_reasons=offset_clamp_reasons,
        clamp_confidence=clamp_confidence,
        offset_detail=_offset_detail,
    )

    # --- 6h METAR override ----------------------------------------------------
    # The 6h METAR max is settlement-grade (0.1°C precision, tracked by ASOS
    # continuously over the 6h window).  When available:
    #   1. Every bracket below the effective floor gets prob=0
    #   2. If the peak-window METAR has reported and auto obs haven't exceeded
    #      it, lock to the METAR bracket at ~100%.
    if metar_6h_readings is not None and len(metar_6h_readings) > 0:
        # --- Multi-METAR logic ---
        # Skip the first 6h METAR — it covers the overnight window and
        # its max is from yesterday's evening, not today's peak.
        daytime_readings = metar_6h_readings[1:] if len(metar_6h_readings) > 1 else []

        if daytime_readings:
            # Floor from daytime 6h METARs only (settlement-grade, 0.1°C precision).
            effective_floor = round(max(r["value_f"] for r in daytime_readings))

            # Zero all brackets below the effective floor
            remaining_prob = 0.0
            for r in results:
                lo, hi = r["bracket"]
                if hi < effective_floor:
                    remaining_prob += r["prob"]
                    r["prob"] = 0.0
                    r["reason"] = f"6h METAR floor: {effective_floor}°F"

            # Latest daytime METAR value
            latest_metar_val = daytime_readings[-1]["value_f"]

            # Check if auto obs since last METAR exceed it
            auto_exceeds_metar = (
                auto_max_since_last_metar_f is not None
                and auto_max_since_last_metar_f > latest_metar_val
            )

            # Lock decision based on forecast peak hour
            peak_metar = None
            if forecast_peak_hour is not None and site_timezone is not None:
                peak_metar = _find_peak_metar(daytime_readings, site_timezone, forecast_peak_hour)

            if peak_metar is not None and not auto_exceeds_metar:
                # Peak-window METAR has reported and auto obs didn't exceed it → lock
                lock_val = round(max(r["value_f"] for r in daytime_readings))
                lock_reason = (f"6h METAR lock (peak window): floor={lock_val}°F, "
                               f"{len(daytime_readings)} daytime readings")
                for r in results:
                    lo, hi = r["bracket"]
                    if lo <= lock_val <= hi:
                        r["prob"] = 0.99
                        r["confidence"] = 1.0
                        r["metar_6h_lock"] = True
                        r["reason"] = lock_reason
                    else:
                        r["prob"] = (1.0 - 0.99) / max(len(results) - 1, 1)
                for r in results:
                    r["metar_state"] = "locked"
                    r["metar_floor_f"] = effective_floor
                    r["metar_readings_count"] = len(daytime_readings)
            elif remaining_prob > 0:
                # Redistribute zeroed-out probability to remaining brackets
                active = [r for r in results if r["prob"] > 0]
                total_active = sum(r["prob"] for r in active)
                if total_active > 0:
                    for r in active:
                        r["prob"] = r["prob"] / total_active * (total_active + remaining_prob)
                state = "floor_only"
                if auto_exceeds_metar:
                    state = "auto_exceeds"
                for r in results:
                    r["metar_state"] = state
                    r["metar_floor_f"] = effective_floor
                    r["metar_readings_count"] = len(daytime_readings)
            else:
                for r in results:
                    r["metar_state"] = "floor_only"
                    r["metar_floor_f"] = effective_floor
                    r["metar_readings_count"] = len(daytime_readings)

    elif metar_6h_f is not None:
        # --- Legacy single-value logic (backward compat) ---
        metar_settled_f = round(metar_6h_f)

        # Auto max_c in °F for comparison
        auto_max_f = round(float(max_c) * 9.0 / 5.0 + 32.0) if max_c is not None else None
        # If auto peak is below METAR, the peak has passed — lock to METAR bracket
        peak_passed = auto_max_f is not None and auto_max_f < metar_settled_f

        # Zero out every bracket below the METAR settlement value
        remaining_prob = 0.0
        for r in results:
            lo, hi = r["bracket"]
            if hi < metar_settled_f:
                remaining_prob += r["prob"]
                r["prob"] = 0.0
                r["reason"] = f"6h METAR floor: {metar_settled_f}°F"

        if peak_passed:
            # Lock to the METAR bracket
            lock_reason = (f"6h METAR lock: round({metar_6h_f:.1f})={metar_settled_f}°F, "
                           f"auto peak={auto_max_f}°F (below)")
            for r in results:
                lo, hi = r["bracket"]
                if lo <= metar_settled_f <= hi:
                    r["prob"] = 0.99
                    r["confidence"] = 1.0
                    r["metar_6h_lock"] = True
                    r["reason"] = lock_reason
                else:
                    r["prob"] = (1.0 - 0.99) / max(len(results) - 1, 1)
        elif remaining_prob > 0:
            # Redistribute zeroed-out probability to remaining brackets
            active = [r for r in results if r["prob"] > 0]
            total_active = sum(r["prob"] for r in active)
            if total_active > 0:
                for r in active:
                    r["prob"] = r["prob"] / total_active * (total_active + remaining_prob)

    # --- CLI/DSM high floor (settlement value sets a hard floor) ---
    # Zero candidate temps below the floor, then rebuild bracket probs.
    # This handles the mid-bracket case: if floor=83 and bracket is (82,83),
    # candidate 82 is eliminated but 83 survives.
    if cli_high_f is not None:
        # Zero offsets whose candidate temp is below the floor
        floored_probs = {}
        zeroed_prob = 0.0
        for offset, temp in candidates.items():
            if temp < cli_high_f:
                zeroed_prob += offset_probs.get(offset, 0.0)
                floored_probs[offset] = 0.0
            else:
                floored_probs[offset] = offset_probs.get(offset, 0.0)

        # Renormalize surviving offsets to absorb zeroed mass
        surviving_total = sum(floored_probs.values())
        if surviving_total > 0 and zeroed_prob > 0:
            scale = 1.0 / surviving_total
            for offset in floored_probs:
                floored_probs[offset] *= scale

        # Rebuild bracket probs from floored offsets
        if zeroed_prob > 0:
            for r in results:
                lo, hi = r["bracket"]
                # Clamp sentinels for matching
                blo = lo if lo > -1000 else naive_f - 10
                bhi = hi if hi < 1000 else naive_f + 10
                new_prob = 0.0
                for offset, temp in candidates.items():
                    if blo <= temp <= bhi:
                        new_prob += floored_probs.get(offset, 0.0)
                if new_prob != r["prob"]:
                    r["reason"] = f"CLI floor: high={cli_high_f}°F"
                r["prob"] = new_prob

    return results


# ---------------------------------------------------------------------------
# LOSO backtest
# ---------------------------------------------------------------------------

def train_loso(sites: Optional[List[str]] = None, since: Optional[str] = None):
    """Leave-one-site-out backtest of the full 2-stage pipeline.

    For each site, trains stage1 (auto GBM) + stage2 (combined RF) on
    all other sites, predicts on the held-out site, and applies clamp
    rules.  Reports per-site and overall accuracy.

    Args:
        sites: ICAO codes to include (default: Kalshi sites).
        since: Only include days on or after this date (YYYY-MM-DD).
    """
    from sklearn.experimental import enable_hist_gradient_boosting  # noqa: F401
    from sklearn.ensemble import HistGradientBoostingClassifier
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
            if since and str(date) < since:
                continue
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
    feature_cols = [c for c in FEATURE_COLS if c in rdf.columns]
    auto_cols = [c for c in AUTO_FEATURE_COLS if c in rdf.columns]

    required_cols = [c for c in feature_cols if c not in _MET_FEATURE_NAMES]
    feature_mask = rdf[required_cols].notna().all(axis=1)
    rdf = rdf[feature_mask].reset_index(drop=True)

    # Test set: only days from 2026-01-01 onward
    test_date_cutoff = "2026-01-01"
    is_test_eligible = rdf["date"].astype(str) >= test_date_cutoff
    n_test_eligible = int(is_test_eligible.sum())
    n = len(rdf)

    y = rdf["offset"].values.astype(int)
    site_labels = rdf["site"].values
    unique_sites = sorted(set(site_labels))

    print("=" * 78)
    print("  BRACKET MODEL — LEAVE-ONE-SITE-OUT BACKTEST")
    print(f"  {n} total days across {len(unique_sites)} sites")
    print(f"  Test set: {n_test_eligible} days from {test_date_cutoff} onward")
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
        test_mask = (site_labels == site) & is_test_eligible.values
        train_mask = ~test_mask
        n_test = int(test_mask.sum())
        if n_test == 0:
            continue

        y_train = y[train_mask]
        y_test = y[test_mask]

        # Stage 1: HistGBM on auto-only features (no METAR)
        X_auto_train = rdf.loc[train_mask, auto_cols].values
        X_auto_test = rdf.loc[test_mask, auto_cols].values

        s1 = HistGradientBoostingClassifier(
            max_iter=200, max_depth=7, learning_rate=0.05,
            min_samples_leaf=40, random_state=42,
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
            rdf_train[col_name] = train_proba[:, i]
        rdf_train["consensus"] = rdf_train.get("auto_prob_+1", 0.0) * rdf_train["metar_confirm"]
        rdf_train["auto_metar_divergence"] = rdf_train.get("auto_prob_+1", 0.0) - rdf_train["metar_confirm"]

        # Test proba (raw S1 probabilities)
        test_proba = s1.predict_proba(X_auto_test)
        rdf_test = rdf.loc[test_mask].copy()
        for i, col_name in enumerate(prob_col_names):
            rdf_test[col_name] = test_proba[:, i]
        rdf_test["consensus"] = rdf_test.get("auto_prob_+1", 0.0) * rdf_test["metar_confirm"]
        rdf_test["auto_metar_divergence"] = rdf_test.get("auto_prob_+1", 0.0) - rdf_test["metar_confirm"]

        metar_raw_cols = ["metar_confirm", "metar_gap_c"]
        consensus_cols = ["consensus", "auto_metar_divergence"]
        stage2_cols = prob_col_names + metar_raw_cols + consensus_cols

        X_s2_train = rdf_train[stage2_cols].values
        X_s2_test = rdf_test[stage2_cols].values

        # Stage 2: HistGBM (no scaler)
        s2 = HistGradientBoostingClassifier(
            max_iter=200, max_depth=5, learning_rate=0.1,
            min_samples_leaf=5, random_state=42,
        )
        s2.fit(X_s2_train, y_train)
        s2_pred = s2.predict(X_s2_test)
        s2_acc = accuracy_score(y_test, s2_pred)

        # Apply clamp rules
        s2c_pred = s2_pred.copy()
        s2_proba = s2.predict_proba(X_s2_test)
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
                    max_model_prob = entries[0][5] if len(entries[0]) > 5 else None
                    if max_model_prob is not None:
                        target_class = entries[0][3]
                        if probs.get(target_class, 0.0) >= max_model_prob:
                            continue
                    best_group = entries
                    best_conf = conf

            if best_group is not None:
                for entry in best_group:
                    _, _, from_val, to_val, confidence = entry[0], entry[1], entry[2], entry[3], entry[4]
                    moved = probs.get(from_val, 0.0) * confidence
                    if moved > 0:
                        probs[from_val] -= moved
                        if to_val is None:
                            others = [k for k in [-1, 0, 1] if k != from_val]
                            total = sum(probs.get(k, 0.0) for k in others)
                            for k in others:
                                share = probs.get(k, 0.0) / total if total > 0 else 0.5
                                probs[k] = probs.get(k, 0.0) + moved * share
                        else:
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

        # Stage 1: HistGBM on auto-only features (full data)
        X_auto_full = rdf[auto_cols].values
        s1_full = HistGradientBoostingClassifier(
            max_iter=200, max_depth=7, learning_rate=0.05,
            min_samples_leaf=40, random_state=42,
        )
        s1_full.fit(X_auto_full, y)

        # HistGBM doesn't have feature_importances_ by default, use permutation
        from sklearn.inspection import permutation_importance
        perm_imp = permutation_importance(s1_full, X_auto_full, y, n_repeats=5,
                                          random_state=42, n_jobs=-1)
        s1_imp = sorted(zip(auto_cols, perm_imp.importances_mean),
                        key=lambda x: -x[1])
        print(f"\n  Stage 1 (HistGBM, auto-only features) — top 20:")
        print(f"  {'Feature':<35s} {'Importance':>10s}")
        print(f"  {'-'*35} {'-'*10}")
        for feat, imp in s1_imp[:20]:
            print(f"  {feat:<35s} {imp:10.4f}")

        # Stage 2: build stacked features and train HistGBM on full data
        auto_proba_full = s1_full.predict_proba(X_auto_full)
        auto_classes_full = s1_full.classes_
        prob_col_names_full = [f"auto_prob_{c:+d}" for c in auto_classes_full]

        rdf_full = rdf.copy()
        for i, col_name in enumerate(prob_col_names_full):
            rdf_full[col_name] = auto_proba_full[:, i]
        rdf_full["consensus"] = rdf_full["auto_prob_+1"] * rdf_full["metar_confirm"]
        rdf_full["auto_metar_divergence"] = rdf_full["auto_prob_+1"] - rdf_full["metar_confirm"]

        metar_raw_cols = ["metar_confirm", "metar_gap_c"]
        consensus_cols = ["consensus", "auto_metar_divergence"]
        stage2_cols_full = prob_col_names_full + metar_raw_cols + consensus_cols
        X_s2_full = rdf_full[stage2_cols_full].values

        s2_full = HistGradientBoostingClassifier(
            max_iter=200, max_depth=5, learning_rate=0.1,
            min_samples_leaf=5, random_state=42,
        )
        s2_full.fit(X_s2_full, y)

        perm_imp_s2 = permutation_importance(s2_full, X_s2_full, y, n_repeats=5,
                                              random_state=42, n_jobs=-1)
        s2_imp = sorted(zip(stage2_cols_full, perm_imp_s2.importances_mean),
                        key=lambda x: -x[1])
        print(f"\n  Stage 2 (HistGBM, METAR + raw S1 probs) — all {len(stage2_cols_full)} features:")
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

def sweep_s1(sites: Optional[List[str]] = None, since: Optional[str] = None):
    """Sweep different Stage-1 models via LOSO and compare end-to-end accuracy.

    Loads data once, then runs the full S1→CV→S2→clamp LOSO pipeline for each
    candidate S1 model.  Prints a comparison table.

    Args:
        sites: ICAO codes to include (default: Kalshi sites).
        since: Only include days on or after this date (YYYY-MM-DD).
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
            if since and str(date) < since:
                continue
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
    feature_cols = [c for c in FEATURE_COLS if c in rdf.columns]

    required_cols = [c for c in feature_cols if c not in _MET_FEATURE_NAMES]
    feature_mask = rdf[required_cols].notna().all(axis=1)
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

    X_all = rdf[feature_cols].values

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
            s1.fit(X_all[train_mask], y[train_mask])
            all_y_true.extend(y[test_mask])
            all_s1_pred.extend(s1.predict(X_all[test_mask]))

        all_y_true = np.array(all_y_true)
        all_s1_pred = np.array(all_s1_pred)
        acc = accuracy_score(all_y_true, all_s1_pred)
        upper_true = (all_y_true >= 0).astype(int)
        upper_pred = (all_s1_pred >= 0).astype(int)
        upper_acc = accuracy_score(upper_true, upper_pred)
        return acc, upper_acc

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
                        help="Comma-separated ICAO codes (default: Kalshi sites)")
    parser.add_argument("--all-sites", action="store_true",
                        help="Include training-only sites (default: Kalshi sites only)")
    parser.add_argument("--since", type=str, default=None,
                        help="Only include days on or after this date (YYYY-MM-DD)")
    parser.add_argument("--save", action="store_true",
                        help="Train on all data and save model to disk")
    parser.add_argument("--sweep-s1", action="store_true",
                        help="Sweep different S1 models via LOSO")
    args = parser.parse_args()

    if args.sites:
        sites = args.sites.split(",")
    elif args.all_sites:
        sites = list(ALL_SITES_WITH_TRAINING)
    else:
        sites = None

    if args.sweep_s1:
        sweep_s1(sites=sites, since=args.since)
    else:
        path = train(sites=sites, since=args.since)
        print(path)
