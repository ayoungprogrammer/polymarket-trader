#!/usr/bin/env python3
"""Feature exploration & correlation analysis for settlement prediction.

Visualises feature distributions, correlations with the target (offset),
and inter-feature relationships.  Reuses the data pipeline from
backtest_rounding.py.

Usage:
    PYTHONPATH=src python -m weather.eda                    # all sites
    PYTHONPATH=src python -m weather.eda --site KLAX,KMIA   # specific sites
    PYTHONPATH=src python -m weather.eda --bracket-upper    # binary: offset >= 0
    PYTHONPATH=src python -m weather.eda --bracket-middle   # binary: offset > 0
    PYTHONPATH=src python -m weather.eda --top 10           # top N features
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from weather.backtest import load_site_history, ALL_SITES
from weather.backtest_rounding import extract_regression_features, FEATURE_COLS, SOLAR_NOON_CSV
from paths import project_path

EDA_DIR = project_path("charts", "eda")


# ---------------------------------------------------------------------------
# Data loading (mirrors backtest_regression's pipeline)
# ---------------------------------------------------------------------------

def load_feature_dataframe(
    sites: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Load per-day feature rows for the given sites."""
    if sites is None:
        sites = ALL_SITES

    # Solar noon cache
    solar_noon_lookup: Dict[Tuple[str, str], float] = {}
    if os.path.isfile(SOLAR_NOON_CSV):
        sn_df = pd.read_csv(SOLAR_NOON_CSV)
        for _, row in sn_df.iterrows():
            solar_noon_lookup[(row["site"], str(row["date"]))] = float(row["solar_noon_hour"])
        print(f"  Loaded {len(solar_noon_lookup)} solar noon entries")

    rows: List[dict] = []
    for site in sites:
        print(f"  Loading {site}...", end="", flush=True)
        df = load_site_history(site)
        if df.empty:
            print(" no data")
            continue
        n_days = 0
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
            n_days += 1
            rows.append(feats)
        print(f" {n_days} days")

    rdf = pd.DataFrame(rows)
    if rdf.empty:
        return rdf

    # Cross-day lagged features (same as backtest_regression)
    rdf = rdf.sort_values(["site", "date"]).reset_index(drop=True)
    rdf["yesterday_offset"] = np.nan
    rdf["prev_3day_mean_offset"] = np.nan
    rdf["temp_trend_3day"] = np.nan

    for site in rdf["site"].unique():
        mask = rdf["site"] == site
        site_idx = rdf.index[mask]
        offsets = rdf.loc[site_idx, "offset"].values.astype(float)
        max_c_vals = rdf.loc[site_idx, "max_c"].values.astype(float)

        rdf.loc[site_idx[1:], "yesterday_offset"] = offsets[:-1]
        for i in range(3, len(site_idx)):
            rdf.loc[site_idx[i], "prev_3day_mean_offset"] = float(
                np.mean(offsets[i - 3:i]))
            rdf.loc[site_idx[i], "temp_trend_3day"] = float(
                (max_c_vals[i] - max_c_vals[i - 3]) / 3.0)

    return rdf


# ---------------------------------------------------------------------------
# 1. Feature–target correlation bar chart (Spearman)
# ---------------------------------------------------------------------------

def plot_feature_target_correlation(
    rdf: pd.DataFrame,
    feature_cols: List[str],
    target: np.ndarray,
    target_label: str,
):
    """Spearman correlation of each feature with the target."""
    from scipy.stats import spearmanr

    corrs = {}
    for col in feature_cols:
        vals = rdf[col].values.astype(float)
        valid = ~(np.isnan(vals) | np.isnan(target))
        if valid.sum() < 10:
            continue
        # Skip constant features (correlation undefined)
        if np.std(vals[valid]) == 0:
            continue
        r, _ = spearmanr(vals[valid], target[valid])
        if np.isnan(r):
            continue
        corrs[col] = r

    # Sort by absolute correlation
    sorted_cols = sorted(corrs, key=lambda c: abs(corrs[c]), reverse=True)

    print(f"\n{'=' * 70}")
    print(f"  Feature–target Spearman correlation ({target_label})")
    print(f"{'=' * 70}")
    for col in sorted_cols:
        print(f"    {col:40s}  {corrs[col]:+.4f}")

    # Bar chart
    fig, ax = plt.subplots(figsize=(8, max(6, len(sorted_cols) * 0.28)))
    colors = ["#2196F3" if corrs[c] >= 0 else "#F44336" for c in sorted_cols]
    y_pos = np.arange(len(sorted_cols))
    ax.barh(y_pos, [corrs[c] for c in sorted_cols], color=colors, height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_cols, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Spearman correlation")
    ax.set_title(f"Feature–target correlation ({target_label})")
    ax.axvline(0, color="black", linewidth=0.5)
    fig.tight_layout()
    path = os.path.join(EDA_DIR, "correlation_target.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# 2. Correlation matrix heatmap (Pearson, hierarchically clustered)
# ---------------------------------------------------------------------------

def plot_correlation_matrix(
    rdf: pd.DataFrame,
    feature_cols: List[str],
):
    """Pairwise Pearson correlation heatmap with hierarchical clustering."""
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import squareform

    # Drop constant features (no variance → NaN correlations)
    X = rdf[feature_cols].astype(float)
    non_const = [c for c in feature_cols if X[c].std() > 0]
    X = X[non_const]
    corr = X.corr().fillna(0)

    # Hierarchical clustering on 1 - |corr| distance
    dist = 1 - corr.abs().values
    np.fill_diagonal(dist, 0)
    # Force exact symmetry and clamp negatives from floating-point noise
    dist = (dist + dist.T) / 2
    dist = np.clip(dist, 0, None)
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="average")
    order = leaves_list(Z)

    ordered_cols = [non_const[i] for i in order]
    corr_ordered = corr.loc[ordered_cols, ordered_cols]

    print(f"\n{'=' * 70}")
    print(f"  Correlation matrix ({len(non_const)} features)")
    print(f"{'=' * 70}")

    # Find highly correlated pairs
    pairs = []
    for i in range(len(non_const)):
        for j in range(i + 1, len(non_const)):
            r = corr.iloc[i, j]
            if abs(r) > 0.7:
                pairs.append((non_const[i], non_const[j], r))
    pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    if pairs:
        print(f"  Highly correlated pairs (|r| > 0.7):")
        for a, b, r in pairs[:20]:
            print(f"    {a:30s} <-> {b:30s}  r={r:+.3f}")

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(corr_ordered.values, cmap="RdBu_r", vmin=-1, vmax=1,
                   aspect="auto")
    ax.set_xticks(np.arange(len(ordered_cols)))
    ax.set_yticks(np.arange(len(ordered_cols)))
    ax.set_xticklabels(ordered_cols, rotation=90, fontsize=5)
    ax.set_yticklabels(ordered_cols, fontsize=5)
    ax.set_title("Feature correlation matrix (clustered)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path = os.path.join(EDA_DIR, "correlation_matrix.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# 3. Feature distributions by offset class (box plots)
# ---------------------------------------------------------------------------

def plot_distributions_by_offset(
    rdf: pd.DataFrame,
    feature_cols: List[str],
    target: np.ndarray,
    target_label: str,
    top_n: int = 8,
):
    """Box plots of top-N correlated features split by offset class."""
    from scipy.stats import spearmanr

    classes = sorted(set(target.astype(int)))

    # Rank features by |Spearman r|
    corrs = {}
    for col in feature_cols:
        vals = rdf[col].values.astype(float)
        valid = ~(np.isnan(vals) | np.isnan(target))
        if valid.sum() < 10:
            continue
        if np.std(vals[valid]) == 0:
            continue
        r, _ = spearmanr(vals[valid], target[valid])
        if np.isnan(r):
            continue
        corrs[col] = abs(r)

    top_cols = sorted(corrs, key=corrs.get, reverse=True)[:top_n]

    print(f"\n{'=' * 70}")
    print(f"  Distributions by class — top {len(top_cols)} features")
    print(f"{'=' * 70}")
    for col in top_cols:
        parts = []
        for cls in classes:
            vals = rdf.loc[target == cls, col].dropna()
            parts.append(f"class {cls}: mean={vals.mean():.3f} std={vals.std():.3f}")
        print(f"  {col}: {' | '.join(parts)}")

    n_cols = min(4, len(top_cols))
    n_rows = (len(top_cols) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[np.newaxis, :]
    elif n_cols == 1:
        axes = axes[:, np.newaxis]

    for idx, col in enumerate(top_cols):
        r = idx // n_cols
        c = idx % n_cols
        ax = axes[r, c]
        data_groups = [rdf.loc[target == cls, col].dropna().values for cls in classes]
        bp = ax.boxplot(data_groups, labels=[str(c) for c in classes],
                        patch_artist=True, widths=0.6)
        colors_map = {-1: "#F44336", 0: "#FFC107", 1: "#4CAF50"}
        default_colors = ["#2196F3", "#FF9800", "#4CAF50", "#F44336"]
        for i, patch in enumerate(bp["boxes"]):
            cls_val = classes[i]
            color = colors_map.get(cls_val, default_colors[i % len(default_colors)])
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.set_title(col, fontsize=8)
        ax.set_xlabel(target_label, fontsize=7)
        ax.tick_params(labelsize=7)

    # Hide unused axes
    for idx in range(len(top_cols), n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].set_visible(False)

    fig.suptitle(f"Feature distributions by {target_label}", fontsize=11)
    fig.tight_layout()
    path = os.path.join(EDA_DIR, "distributions_by_offset.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# 4. Per-site offset distribution (stacked bar chart)
# ---------------------------------------------------------------------------

def plot_offset_by_site(rdf: pd.DataFrame):
    """Stacked bar chart of offset class proportions per site."""
    sites = sorted(rdf["site"].unique())
    offsets = sorted(rdf["offset"].unique())

    print(f"\n{'=' * 70}")
    print(f"  Offset distribution by site")
    print(f"{'=' * 70}")

    counts = {}
    for site in sites:
        site_data = rdf[rdf["site"] == site]["offset"]
        n = len(site_data)
        counts[site] = {}
        parts = []
        for o in offsets:
            cnt = int((site_data == o).sum())
            counts[site][o] = cnt
            parts.append(f"{o:+d}:{cnt}({cnt/n*100:.0f}%)")
        print(f"  {site}: n={n:4d}  {' '.join(parts)}")

    fig, ax = plt.subplots(figsize=(max(8, len(sites) * 0.8), 5))
    x = np.arange(len(sites))
    width = 0.7
    colors_map = {-1: "#F44336", 0: "#FFC107", 1: "#4CAF50"}
    default_colors = ["#2196F3", "#FF9800", "#4CAF50", "#F44336"]

    bottom = np.zeros(len(sites))
    for o in offsets:
        vals = []
        for site in sites:
            total = sum(counts[site].values())
            vals.append(counts[site].get(o, 0) / total * 100 if total > 0 else 0)
        color = colors_map.get(int(o), default_colors[len(offsets) % len(default_colors)])
        ax.bar(x, vals, width, bottom=bottom, label=f"offset={int(o):+d}",
               color=color, alpha=0.8)
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(sites, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Offset class distribution by site")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(EDA_DIR, "offset_by_site.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# 5. Mutual information bar chart
# ---------------------------------------------------------------------------

def plot_mutual_information(
    rdf: pd.DataFrame,
    feature_cols: List[str],
    target: np.ndarray,
    target_label: str,
):
    """Mutual information of each feature with the target (handles both
    classification and regression targets)."""
    from sklearn.feature_selection import (
        mutual_info_classif, mutual_info_regression,
    )

    available = [c for c in feature_cols if c in rdf.columns]
    X = rdf[available].astype(float).values
    valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(target))
    X = X[valid_mask]
    y = target[valid_mask]

    if len(X) < 20:
        print("  Too few rows for mutual information — skipping.")
        return

    # Classification if target is discrete (few unique values), else regression
    n_unique = len(np.unique(y))
    if n_unique <= 10:
        mi = mutual_info_classif(X, y.astype(int), random_state=42, n_neighbors=5)
    else:
        mi = mutual_info_regression(X, y, random_state=42, n_neighbors=5)

    mi_dict = {col: float(v) for col, v in zip(available, mi)}
    sorted_cols = sorted(mi_dict, key=mi_dict.get, reverse=True)

    print(f"\n{'=' * 70}")
    print(f"  Mutual information with {target_label}")
    print(f"{'=' * 70}")
    for col in sorted_cols:
        print(f"    {col:40s}  {mi_dict[col]:.4f}")

    # Bar chart
    fig, ax = plt.subplots(figsize=(8, max(6, len(sorted_cols) * 0.28)))
    y_pos = np.arange(len(sorted_cols))
    vals = [mi_dict[c] for c in sorted_cols]
    ax.barh(y_pos, vals, color="#4CAF50", height=0.7, alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_cols, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Mutual information (nats)")
    ax.set_title(f"Mutual information with {target_label}")
    fig.tight_layout()
    path = os.path.join(EDA_DIR, "mutual_information.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# 6. True temperature vs auto max_c scatter (°C and °F)
# ---------------------------------------------------------------------------

def plot_true_vs_auto(rdf: pd.DataFrame):
    """Side-by-side scatter of actual settlement temp vs auto-obs max,
    in both °C and °F, coloured by offset class."""
    if "max_c" not in rdf.columns or "actual_max_f" not in rdf.columns:
        print("  Missing max_c or actual_max_f — skipping true-vs-auto plot.")
        return

    max_c = rdf["max_c"].values.astype(float)
    naive_f_float = rdf["naive_f_float"].values.astype(float)
    actual_f = rdf["actual_max_f"].values.astype(float)
    offset = rdf["offset"].values.astype(int)

    # Derive actual °C from actual °F:  C = (F - 32) * 5/9
    actual_c = (actual_f - 32.0) * 5.0 / 9.0

    colors_map = {-1: "#F44336", 0: "#FFC107", 1: "#4CAF50"}
    labels_map = {-1: "offset −1", 0: "offset 0", 1: "offset +1"}
    point_colors = [colors_map.get(o, "#999999") for o in offset]

    fig, (ax_c, ax_f) = plt.subplots(1, 2, figsize=(14, 6))

    # --- °C panel ---
    for cls in [-1, 0, 1]:
        mask = offset == cls
        ax_c.scatter(max_c[mask], actual_c[mask], c=colors_map[cls],
                     label=labels_map[cls], alpha=0.4, s=12, edgecolors="none")
    lo_c = min(max_c.min(), actual_c.min()) - 1
    hi_c = max(max_c.max(), actual_c.max()) + 1
    ax_c.plot([lo_c, hi_c], [lo_c, hi_c], "k--", lw=0.8, alpha=0.5)
    ax_c.set_xlabel("Auto-obs max (°C, whole-degree)")
    ax_c.set_ylabel("Actual settlement (°C)")
    ax_c.set_title("True temp vs auto max — °C")
    ax_c.legend(fontsize=7, loc="upper left")
    ax_c.set_aspect("equal", adjustable="datalim")

    # --- °F panel ---
    for cls in [-1, 0, 1]:
        mask = offset == cls
        ax_f.scatter(naive_f_float[mask], actual_f[mask], c=colors_map[cls],
                     label=labels_map[cls], alpha=0.4, s=12, edgecolors="none")
    lo_f = min(naive_f_float.min(), actual_f.min()) - 2
    hi_f = max(naive_f_float.max(), actual_f.max()) + 2
    ax_f.plot([lo_f, hi_f], [lo_f, hi_f], "k--", lw=0.8, alpha=0.5)
    ax_f.set_xlabel("Auto-obs max (°F, fractional C→F)")
    ax_f.set_ylabel("Actual settlement (°F, whole-degree)")
    ax_f.set_title("True temp vs auto max — °F")
    ax_f.legend(fontsize=7, loc="upper left")
    ax_f.set_aspect("equal", adjustable="datalim")

    fig.suptitle("Settlement temperature vs auto-obs maximum", fontsize=12)
    fig.tight_layout()
    path = os.path.join(EDA_DIR, "true_vs_auto.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_eda(
    sites: Optional[List[str]] = None,
    bracket_mode: Optional[str] = None,
    top_n: int = 100,
):
    os.makedirs(EDA_DIR, exist_ok=True)

    print("Loading feature data...")
    rdf = load_feature_dataframe(sites)
    if rdf.empty:
        print("No data loaded — check data/ directory for history CSVs.")
        return

    feature_cols = list(FEATURE_COLS)

    # Drop rows with NA in any feature column
    available_cols = [c for c in feature_cols if c in rdf.columns]
    feature_mask = rdf[available_cols].notna().all(axis=1)
    n_dropped = int((~feature_mask).sum())
    rdf = rdf[feature_mask].reset_index(drop=True)
    print(f"  {len(rdf)} rows ({n_dropped} dropped for NA features)")

    if len(rdf) < 10:
        print("Insufficient data after NA filtering.")
        return

    # Target
    y = rdf["offset"].values.astype(float)
    if bracket_mode == "upper":
        target = (y >= 0).astype(float)
        target_label = "bracket-upper (offset >= 0)"
    elif bracket_mode == "middle":
        target = (y > 0).astype(float)
        target_label = "bracket-middle (offset > 0)"
    else:
        target = y
        target_label = "offset"

    # Run all analyses
    plot_feature_target_correlation(rdf, available_cols, target, target_label)
    plot_correlation_matrix(rdf, available_cols)
    plot_distributions_by_offset(rdf, available_cols, target, target_label, top_n=top_n)
    plot_mutual_information(rdf, available_cols, target, target_label)
    plot_true_vs_auto(rdf)
    plot_offset_by_site(rdf)

    print(f"\nAll charts saved to {EDA_DIR}/")


def main():
    parser = argparse.ArgumentParser(
        description="Feature exploration & correlation analysis")
    parser.add_argument("--site", type=str, default=None,
                        help="Comma-separated ICAO codes (default: all)")
    parser.add_argument("--bracket-upper", action="store_true",
                        help="Binary target: offset >= 0")
    parser.add_argument("--bracket-middle", action="store_true",
                        help="Binary target: offset > 0")
    parser.add_argument("--top", type=int, default=100,
                        help="Top N features for distribution plots (default: 100)")
    args = parser.parse_args()

    sites = args.site.split(",") if args.site else None
    bracket_mode = None
    if args.bracket_upper:
        bracket_mode = "upper"
    elif args.bracket_middle:
        bracket_mode = "middle"

    run_eda(sites, bracket_mode=bracket_mode, top_n=args.top)


if __name__ == "__main__":
    main()
