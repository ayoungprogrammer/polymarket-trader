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
from weather.sites import ALL_SITES

# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------


SAMPLES_CACHE = project_path("data", "weather", "peak_samples.pkl")


def _make_model():
    """Create the HistGBM classifier used for training."""
    from sklearn.experimental import enable_hist_gradient_boosting  # noqa: F401
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(
        max_iter=400, max_depth=7, learning_rate=0.2,
        min_samples_leaf=20, random_state=42,
    )


def _load_samples(sites: Optional[List[str]] = None,
                  no_cache: bool = False,
                  since: Optional[str] = None) -> pd.DataFrame:
    """Load samples, using a pickle cache to skip recomputation."""
    cache = SAMPLES_CACHE
    if not no_cache and os.path.isfile(cache):
        print(f"  Loading cached samples from {cache}")
        df = pd.read_pickle(cache)
        if sites is not None:
            df = df[df["site"].isin(sites)]
        if since:
            df = df[df["date"] >= since].reset_index(drop=True)
            print(f"  Filtered to dates >= {since}: {len(df)} rows")
        return df

    print("Loading and generating samples...")
    df = load_all_samples(sites, since=since)
    if not df.empty and sites is None and since is None:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        df.to_pickle(cache)
        print(f"  Cached {len(df)} samples to {cache}")
    return df


def train(sites: Optional[List[str]] = None,
          no_cache: bool = False,
          since: Optional[str] = None) -> dict:
    """Train HistGBM on all historical data and return model bundle."""
    df = _load_samples(sites, no_cache=no_cache, since=since)
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
    print("  Training HistGBM (>1°F)...")
    model.fit(X, y)

    # Feature importances
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        top_idx = np.argsort(importances)[::-1][:10]
        print("  Top 10 features (>1°F):")
        for rank, idx in enumerate(top_idx, 1):
            print(f"    {rank:>2}. {FEATURE_COLS[idx]:<30s} {importances[idx]:.4f}")

    # Train >2°F model
    LABEL_2 = "will_increase_2f"
    model_2f = None
    if LABEL_2 in df.columns:
        y2 = df[LABEL_2].values
        n_pos2 = int(y2.sum())
        n_neg2 = len(y2) - n_pos2
        print(f"\n  >2°F class balance: {n_pos2} positive ({n_pos2 / len(df):.1%}), "
              f"{n_neg2} negative ({n_neg2 / len(df):.1%})")
        model_2f = _make_model()
        print("  Training HistGBM (>2°F)...")
        model_2f.fit(X, y2)
        if hasattr(model_2f, "feature_importances_"):
            importances2 = model_2f.feature_importances_
            top_idx2 = np.argsort(importances2)[::-1][:10]
            print("  Top 10 features (>2°F):")
            for rank, idx in enumerate(top_idx2, 1):
                print(f"    {rank:>2}. {FEATURE_COLS[idx]:<30s} {importances2[idx]:.4f}")

    bundle = {
        "model": model,
        "model_2f": model_2f,
        "features": FEATURE_COLS,
        "trained_at": datetime.utcnow().isoformat(),
        "n_samples": len(df),
        "n_sites": int(df["site"].nunique()),
    }
    return bundle


def _run_loso_label(df: pd.DataFrame, label_col: str, label_name: str) -> dict:
    """LOSO eval for one label. Tests only Kalshi sites, 2026+ data."""
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        classification_report,
    )

    all_sites = sorted(df["site"].unique())
    test_sites = sorted(s for s in all_sites if s in ALL_SITES)
    train_only = sorted(s for s in all_sites if s not in ALL_SITES)

    pos_rate = df[label_col].mean()
    baseline_acc = max(pos_rate, 1 - pos_rate)
    baseline_label = "always-0" if pos_rate < 0.5 else "always-1"

    print(f"\n{'=' * 70}")
    print(f"  LEAVE-ONE-SITE-OUT CV — {label_name}")
    print(f"  Test sites: {len(test_sites)} (Kalshi)  |  Train-only: {len(train_only)}")
    print(f"  Test data: 2026-01-01 onward")
    print(f"{'=' * 70}")

    n_pos = int((df[label_col] == 1).sum())
    n_neg = len(df) - n_pos
    print(f"  Class balance: {n_pos} positive ({n_pos / len(df):.1%}), "
          f"{n_neg} negative ({n_neg / len(df):.1%})")
    print(f"  Baseline ({baseline_label}): {baseline_acc:.1%}\n")

    per_site = []
    all_y_true = []
    all_y_pred = []
    all_y_prob = []

    print(f"  {'Site':<6} {'N':>6} {'Acc':>7} {'F1':>7} {'Pos%':>7}")
    print(f"  {'-'*6} {'-'*6} {'-'*7} {'-'*7} {'-'*7}")

    for test_site in test_sites:
        test_mask = (df["site"] == test_site) & (df["date"] >= "2026-01-01")
        train_df = df[~test_mask]
        test_df = df[test_mask]
        if len(test_df) < 10:
            continue

        X_train = train_df[FEATURE_COLS].values
        y_train = train_df[label_col].values
        X_test = test_df[FEATURE_COLS].values
        y_test = test_df[label_col].values

        model = _make_model()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        per_site.append({
            "site": test_site,
            "n_test": len(test_df),
            "accuracy": acc,
            "f1": f1,
            "pos_pct": float(y_test.mean()),
        })
        print(f"  {test_site:<6} {len(test_df):>6} {acc:>7.4f} "
              f"{f1:>7.4f} {y_test.mean():>6.1%}")

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        all_y_prob.extend(y_prob)

    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    all_y_prob = np.array(all_y_prob)

    if len(all_y_true) == 0:
        print("  No test results.")
        return {"accuracy": 0, "f1": 0, "per_site": []}

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

    print(f"\n{classification_report(all_y_true, all_y_pred, target_names=[f'no_{label_name}', label_name])}")

    # Feature importances — train on full data
    X_all = df[FEATURE_COLS].values
    y_all = df[label_col].values
    full_model = _make_model()
    full_model.fit(X_all, y_all)
    if hasattr(full_model, "feature_importances_"):
        _print_feature_importances(full_model, FEATURE_COLS)

    _print_calibration(all_y_true, all_y_prob)

    return {
        "accuracy": overall_acc,
        "f1": overall_f1,
        "per_site": per_site,
    }


def train_loso(sites: Optional[List[str]] = None,
               no_cache: bool = False,
               since: Optional[str] = None) -> dict:
    """Leave-one-site-out CV for both >1°F and >2°F labels."""
    df = _load_samples(sites, no_cache=no_cache, since=since)
    if df.empty:
        raise RuntimeError("No samples generated. Check data files.")

    n_days = df.groupby(["site", "date"]).ngroups
    print(f"\n  Dataset: {len(df)} samples, {n_days} site-days, {df['site'].nunique()} sites")

    result_1f = _run_loso_label(df, LABEL, ">1°F")

    LABEL_2 = "will_increase_2f"
    result_2f = None
    if LABEL_2 in df.columns:
        result_2f = _run_loso_label(df, LABEL_2, ">2°F")

    return {
        "accuracy": result_1f["accuracy"],
        "f1": result_1f["f1"],
        "per_site": result_1f["per_site"],
        "result_2f": result_2f,
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
    yesterday: Optional[Dict[str, float]] = None,
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
    yesterday : dict, optional
        Prior day summary: {"high_f", "low_f", "peak_hour", "high_vs_forecast_f"}.

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

    # Precompute momentum arrays (precompute_day_momentum already calls
    # compute_momentum and caches _mom_df in the returned dict)
    precomp = precompute_day_momentum(df)

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
        yesterday=yesterday,
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
    yesterday: Optional[Dict[str, float]] = None,
) -> Optional[dict]:
    """Predict whether the settlement °F will increase by +1 and +2.

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
            "probability": float,       # P(will_increase >= 1°F)
            "probability_2f": float,    # P(will_increase >= 2°F)
            "prediction": bool,         # probability >= 0.5
            "prediction_2f": bool,      # probability_2f >= 0.5
            "cur_naive_f": int,         # current naive_f from auto-obs
            "features": dict,           # extracted features
        }
    """
    feats = extract_live_features(
        obs_df, forecast_high_f, solar_noon_hour,
        forecast_hourly=forecast_hourly,
        yesterday=yesterday,
    )
    if feats is None:
        return None

    clf = model_bundle["model"]
    feature_cols = model_bundle["features"]

    # Build feature vector in model's expected order
    X = np.array([[feats.get(col, 0.0) for col in feature_cols]])
    prob = float(clf.predict_proba(X)[0, 1])

    # >2°F model (may not exist in older bundles)
    clf_2f = model_bundle.get("model_2f")
    if clf_2f is not None:
        prob_2f = float(clf_2f.predict_proba(X)[0, 1])
    else:
        prob_2f = 0.0

    # Recover cur_naive_f from features
    cur_max_f = forecast_high_f - feats.get("forecast_gap_f", 0.0)
    cur_naive_f = round(cur_max_f)

    return {
        "probability": prob,
        "probability_2f": prob_2f,
        "prediction": prob >= 0.5,
        "prediction_2f": prob_2f >= 0.5,
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
    parser.add_argument("--use-all-sites", action="store_true",
                        help="Include all training sites (default: Kalshi sites only)")
    parser.add_argument("--since", type=str, default=None,
                        help="Only use data from this date (YYYY-MM-DD)")
    args = parser.parse_args()

    sites = args.site.split(",") if args.site else None
    if sites is None and args.use_all_sites:
        from weather.backtest_rounding import ALL_SITES_WITH_TRAINING
        sites = list(ALL_SITES_WITH_TRAINING)

    if args.save:
        print("=" * 60)
        print("  PEAK MODEL — TRAIN & SAVE")
        print("=" * 60)
        print()
        bundle = train(sites, no_cache=args.no_cache, since=args.since)
        path = save_model(bundle)
        print(f"\n  {path}")
    else:
        print("=" * 70)
        print("  PEAK MODEL — LEAVE-ONE-SITE-OUT EVALUATION")
        print(f"  Label: {LABEL}")
        print("=" * 70)
        train_loso(sites, no_cache=args.no_cache, since=args.since)

    print("\nDone.")


if __name__ == "__main__":
    main()
