"""Peak detection model — will the settlement °F increase by +1?

Trains a GBM classifier on historical observation data and provides
inference for live prediction. The model predicts whether the final
naive_f (round(max_c * 9/5 + 32)) will be ≥ current naive_f + 1.

Usage:
    # Train and save model
    cd src && python -m weather.peak_model

    # Load and predict from live observations
    from weather.peak_model import load_model, predict
    model = load_model()
    result = predict(model, obs_df, forecast_high_f=82.0)
"""

from __future__ import annotations

import os
import pickle
import sys
from datetime import datetime
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd

from paths import project_path
from weather.backtest_peak import (
    FEATURE_COLS,
    MODEL_DIR,
    MODEL_PATH,
    LABEL,
    extract_peak_features,
    load_all_samples,
    load_forecast_hourly,
    _print_feature_importances,
    _print_calibration,
)
from weather.backtest import precompute_day_momentum
from weather.prediction import compute_momentum

# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------


SAMPLES_CACHE = project_path("data", "peak_samples.pkl")


def _make_model():
    """Create the HistGBM classifier used for training."""
    from sklearn.experimental import enable_hist_gradient_boosting  # noqa: F401
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(
        max_iter=200, max_depth=4, learning_rate=0.1,
        min_samples_leaf=5, random_state=42,
    )


def _load_samples(sites: Optional[List[str]] = None,
                  no_cache: bool = False) -> pd.DataFrame:
    """Load samples, using a pickle cache to skip recomputation."""
    cache = SAMPLES_CACHE
    if not no_cache and os.path.isfile(cache):
        print(f"  Loading cached samples from {cache}")
        df = pd.read_pickle(cache)
        if sites is not None:
            df = df[df["site"].isin(sites)]
        return df

    print("Loading and generating samples...")
    df = load_all_samples(sites)
    if not df.empty and sites is None:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        df.to_pickle(cache)
        print(f"  Cached {len(df)} samples to {cache}")
    return df


def train(sites: Optional[List[str]] = None,
          no_cache: bool = False) -> dict:
    """Train HistGBM on all historical data and return model bundle."""
    df = _load_samples(sites, no_cache=no_cache)
    if df.empty:
        raise RuntimeError("No samples generated. Check data files.")

    n_pos = (df[LABEL] == 1).sum()
    n_neg = (df[LABEL] == 0).sum()
    print(f"  Dataset: {len(df)} samples, {df['site'].nunique()} sites")
    print(f"  Class balance: {n_pos} positive ({n_pos / len(df):.1%}), "
          f"{n_neg} negative ({n_neg / len(df):.1%})")

    X = df[FEATURE_COLS].values
    y = df[LABEL].values

    model = _make_model()
    print("  Training HistGBM...")
    model.fit(X, y)

    # Feature importances
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        top_idx = np.argsort(importances)[::-1][:10]
        print("  Top 10 features:")
        for rank, idx in enumerate(top_idx, 1):
            print(f"    {rank:>2}. {FEATURE_COLS[idx]:<30s} {importances[idx]:.4f}")

    bundle = {
        "model": model,
        "features": FEATURE_COLS,
        "trained_at": datetime.utcnow().isoformat(),
        "n_samples": len(df),
        "n_sites": int(df["site"].nunique()),
    }
    return bundle


def _run_one_fold(args_tuple):
    """Train/eval one LOSO fold. Designed for joblib.Parallel."""
    test_site, X_all, y_all, site_labels, feature_cols = args_tuple
    from sklearn.base import clone
    from sklearn.metrics import accuracy_score, f1_score

    train_mask = site_labels != test_site
    test_mask = site_labels == test_site

    X_train, y_train = X_all[train_mask], y_all[train_mask]
    X_test, y_test = X_all[test_mask], y_all[test_mask]

    if len(X_test) < 10:
        return None

    model = _make_model()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return {
        "site": test_site,
        "n_test": len(X_test),
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "pos_pct": float(y_test.mean()),
        "y_true": y_test,
        "y_pred": y_pred,
        "y_prob": y_prob,
    }


def train_loso(sites: Optional[List[str]] = None,
               no_cache: bool = False) -> dict:
    """Leave-one-site-out CV with HistGBM. Default CLI action."""
    from joblib import Parallel, delayed
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        classification_report,
    )

    df = _load_samples(sites, no_cache=no_cache)
    if df.empty:
        raise RuntimeError("No samples generated. Check data files.")

    all_sites = sorted(df["site"].unique())
    n_pos = (df[LABEL] == 1).sum()
    n_neg = (df[LABEL] == 0).sum()
    n_days = df.groupby(["site", "date"]).ngroups
    print(f"\n  Dataset: {len(df)} samples, {n_days} site-days, {len(all_sites)} sites")
    print(f"  Class balance: {n_pos} positive ({n_pos / len(df):.1%}), "
          f"{n_neg} negative ({n_neg / len(df):.1%})")

    pos_rate = df[LABEL].mean()
    baseline_acc = max(pos_rate, 1 - pos_rate)
    baseline_label = "always-1" if pos_rate >= 0.5 else "always-0"

    print(f"\n{'=' * 70}")
    print(f"  LEAVE-ONE-SITE-OUT CV ({len(all_sites)} sites)")
    print(f"{'=' * 70}")
    print(f"  Baseline ({baseline_label}): {baseline_acc:.1%}\n")

    # Pre-extract arrays once for all folds
    X_all = df[FEATURE_COLS].values
    y_all = df[LABEL].values
    site_labels = df["site"].values

    # Run folds in parallel
    fold_args = [
        (site, X_all, y_all, site_labels, FEATURE_COLS)
        for site in all_sites
    ]
    print(f"  Training {len(all_sites)} folds in parallel...")
    results = Parallel(n_jobs=-1, prefer="threads")(
        delayed(_run_one_fold)(a) for a in fold_args
    )

    # Collect results
    per_site = []
    all_y_true = []
    all_y_pred = []
    all_y_prob = []

    print(f"  {'Site':<6} {'N':>6} {'Acc':>7} {'F1':>7} {'Pos%':>7}")
    print(f"  {'-'*6} {'-'*6} {'-'*7} {'-'*7} {'-'*7}")

    for r in results:
        if r is None:
            continue
        per_site.append({
            "site": r["site"],
            "n_test": r["n_test"],
            "accuracy": r["accuracy"],
            "f1": r["f1"],
            "pos_pct": r["pos_pct"],
        })
        print(f"  {r['site']:<6} {r['n_test']:>6} {r['accuracy']:>7.4f} "
              f"{r['f1']:>7.4f} {r['pos_pct']:>6.1%}")
        all_y_true.extend(r["y_true"])
        all_y_pred.extend(r["y_pred"])
        all_y_prob.extend(r["y_prob"])

    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    all_y_prob = np.array(all_y_prob)

    overall_acc = accuracy_score(all_y_true, all_y_pred)
    overall_f1 = f1_score(all_y_true, all_y_pred, zero_division=0)
    overall_prec = precision_score(all_y_true, all_y_pred, zero_division=0)
    overall_rec = recall_score(all_y_true, all_y_pred, zero_division=0)

    accs = [ps["accuracy"] for ps in per_site]
    f1s = [ps["f1"] for ps in per_site]

    print(f"\n  Overall:  Acc={overall_acc:.4f}  Prec={overall_prec:.4f}  "
          f"Rec={overall_rec:.4f}  F1={overall_f1:.4f}")
    print(f"  Mean acc: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"  Mean F1:  {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")

    print(f"\n{classification_report(all_y_true, all_y_pred, target_names=['no_increase_f', 'will_increase_f'])}")

    # Feature importances — train on full data
    full_model = _make_model()
    full_model.fit(X_all, y_all)
    if hasattr(full_model, "feature_importances_"):
        _print_feature_importances(full_model, FEATURE_COLS)
    else:
        # HistGBM in sklearn <1.0 lacks feature_importances_;
        # use permutation importance instead.
        from sklearn.inspection import permutation_importance
        perm = permutation_importance(
            full_model, X_all, y_all, n_repeats=5,
            random_state=42, n_jobs=-1,
        )
        importances = perm.importances_mean
        top_idx = np.argsort(importances)[::-1][:15]
        print(f"\n  TOP 15 FEATURE IMPORTANCES (permutation)")
        print(f"  {'-' * 40}")
        for rank, idx in enumerate(top_idx, 1):
            print(f"  {rank:>2}. {FEATURE_COLS[idx]:<30s} {importances[idx]:.4f}")
    _print_calibration(all_y_true, all_y_prob)

    return {
        "accuracy": overall_acc,
        "f1": overall_f1,
        "per_site": per_site,
    }


def save_model(bundle: dict, path: Optional[str] = None) -> str:
    """Persist model bundle to disk with timestamped copy.

    Saves to both a timestamped file (e.g. model_20260312_183045.pkl)
    and a symlink/copy at model.pkl pointing to the latest.
    """
    import shutil
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Timestamped filename
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    ts_path = os.path.join(MODEL_DIR, f"model_{ts}.pkl")
    with open(ts_path, "wb") as f:
        pickle.dump(bundle, f)
    print(f"  Model saved to {ts_path}")

    # Copy as model.pkl (latest)
    latest_path = path or MODEL_PATH
    shutil.copy2(ts_path, latest_path)
    print(f"  Copied to {latest_path}")

    return ts_path


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------


def load_model(path: Optional[str] = None) -> dict:
    """Load a trained peak model bundle from disk."""
    if path is None:
        path = MODEL_PATH
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No peak model at {path}. "
                                "Run: cd src && python -m weather.peak_model")
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    return bundle


# ---------------------------------------------------------------------------
# Inference: extract features from live observations
# ---------------------------------------------------------------------------


def extract_live_features(
    obs_df: pd.DataFrame,
    forecast_high_f: float,
    solar_noon_hour: float = 12.0,
    forecast_hourly: Optional[List[Tuple[int, float]]] = None,
) -> Optional[dict]:
    """Extract peak features from live observation data.

    Parameters
    ----------
    obs_df : DataFrame
        Today's observations with columns: timestamp, temperature_f,
        temperature_c, and optionally wind_speed_mph, dewpoint_f, etc.
    forecast_high_f : float
        NWS or Open-Meteo forecast high for today in °F.
    solar_noon_hour : float
        Solar noon in decimal hours (default 12.0).
    forecast_hourly : list of (hour, temp_f), optional
        Hourly forecast curve for today.

    Returns
    -------
    dict or None
        Feature dict ready for predict(), or None if insufficient data.
    """
    if obs_df.empty or len(obs_df) < 10:
        return None

    # Ensure required columns
    if "temperature_f" not in obs_df.columns:
        return None

    df = obs_df.copy()

    # Add 'ts' column (used by precompute_day_momentum)
    if "ts" not in df.columns:
        df["ts"] = pd.to_datetime(df["timestamp"].str[:19])

    # Add 'date' column
    if "date" not in df.columns:
        df["date"] = df["ts"].dt.date

    df = df.sort_values("ts").reset_index(drop=True)

    # Precompute momentum arrays
    precomp = precompute_day_momentum(df)
    mom_df = compute_momentum(df)
    precomp["_mom_df"] = mom_df

    # Use the last observation as the sample point
    as_of_idx = len(df) - 1

    # We don't know actual_max_c at inference time — use 0 as placeholder.
    # The label won't be meaningful but extract_peak_features uses it only
    # for the label, which we discard.
    actual_max_c_placeholder = 0.0

    feats = extract_peak_features(
        df, precomp, as_of_idx, solar_noon_hour,
        actual_max_c_placeholder, forecast_high_f,
        forecast_hourly=forecast_hourly,
    )
    if feats is None:
        return None

    # Remove label — not meaningful at inference time
    feats.pop(LABEL, None)
    return feats


# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------


def predict(
    model_bundle: dict,
    obs_df: pd.DataFrame,
    forecast_high_f: float,
    solar_noon_hour: float = 12.0,
    forecast_hourly: Optional[List[Tuple[int, float]]] = None,
) -> Optional[dict]:
    """Predict whether the settlement °F will increase by +1.

    Parameters
    ----------
    model_bundle : dict
        Output of load_model().
    obs_df : DataFrame
        Today's live observations.
    forecast_high_f : float
        Forecast high temperature in °F.
    solar_noon_hour : float
        Solar noon in decimal hours.
    forecast_hourly : list of (hour, temp_f), optional
        Hourly forecast curve for today.

    Returns
    -------
    dict or None
        {
            "probability": float,   # P(will_increase_f=1)
            "prediction": bool,     # probability >= 0.5
            "cur_naive_f": int,     # current naive_f from auto-obs
            "features": dict,       # extracted features
        }
    """
    feats = extract_live_features(
        obs_df, forecast_high_f, solar_noon_hour,
        forecast_hourly=forecast_hourly,
    )
    if feats is None:
        return None

    clf = model_bundle["model"]
    feature_cols = model_bundle["features"]

    # Build feature vector in model's expected order
    X = np.array([[feats.get(col, 0.0) for col in feature_cols]])
    prob = float(clf.predict_proba(X)[0, 1])

    # Recover cur_naive_f from features
    cur_f_fractional = feats.get("cur_f_fractional", 0.0)
    # cur_f_fractional is the fractional part of cur_f_float
    # We need to reconstruct naive_f from the observation data
    # Use the forecast_gap_f: forecast_high_f - cur_max = forecast_gap_f
    # So cur_max = forecast_high_f - forecast_gap_f
    cur_max_f = forecast_high_f - feats.get("forecast_gap_f", 0.0)
    # cur_max_f is in °F (running max of observations)
    cur_naive_f = round(cur_max_f)

    return {
        "probability": prob,
        "prediction": prob >= 0.5,
        "cur_naive_f": cur_naive_f,
        "features": feats,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Peak detection model — LOSO eval (default) or train & save")
    parser.add_argument("--site", type=str, default=None,
                        help="Comma-separated ICAO codes (default: all)")
    parser.add_argument("--save", action="store_true",
                        help="Train on all data and persist model to disk")
    parser.add_argument("--no-cache", action="store_true",
                        help="Rebuild samples from CSVs (ignore pickle cache)")
    args = parser.parse_args()

    sites = args.site.split(",") if args.site else None

    if args.save:
        print("=" * 60)
        print("  PEAK MODEL — TRAIN & SAVE")
        print("=" * 60)
        print()
        bundle = train(sites, no_cache=args.no_cache)
        path = save_model(bundle)
        print(f"\n  {path}")
    else:
        print("=" * 70)
        print("  PEAK MODEL — LEAVE-ONE-SITE-OUT EVALUATION")
        print(f"  Label: {LABEL}")
        print("=" * 70)
        train_loso(sites, no_cache=args.no_cache)

    print("\nDone.")


if __name__ == "__main__":
    main()
