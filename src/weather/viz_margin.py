#!/usr/bin/env python3
"""Visualize C-to-F rounding margin by max_c.

For each whole-°C value, shows how much °C change is needed to flip
the rounded °F settlement value.  Values near 0 are fragile (a tiny
temperature change flips the °F); values near 0.28°C are stable.

Usage:
    python src/weather/viz_margin.py
    python src/weather/viz_margin.py --range 15 35
"""

from __future__ import annotations

import argparse

import numpy as np
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Plot C-to-F rounding margin")
    parser.add_argument("--range", nargs=2, type=int, default=[-10, 45],
                        metavar=("LO", "HI"),
                        help="°C range to plot (default: -10 45)")
    parser.add_argument("--output", type=str, default=None,
                        help="Save to file instead of showing")
    args = parser.parse_args()

    lo, hi = args.range
    max_c_vals = np.arange(lo, hi + 1)

    f_float = max_c_vals * 9.0 / 5.0 + 32.0
    naive_f = np.round(f_float).astype(int)
    frac = f_float % 1.0
    # Distance to nearest °F rounding boundary (in °F)
    dist_f = np.minimum(frac, 1.0 - frac)
    # Convert to °C
    dist_c = dist_f * 5.0 / 9.0

    fig, ax1 = plt.subplots(figsize=(14, 5))

    colors = ["#d62728" if d < 0.1 else "#ff7f0e" if d < 0.2 else "#2ca02c"
              for d in dist_c]
    ax1.bar(max_c_vals, dist_c, color=colors, edgecolor="white", linewidth=0.5)

    ax1.set_xlabel("max_c (°C)")
    ax1.set_ylabel("°C margin to nearest °F flip")
    ax1.set_title("C-to-F Rounding Margin: how much °C change flips the settlement °F")
    ax1.axhline(y=5.0 / 18.0, color="gray", linestyle="--", alpha=0.5,
                label=f"max margin = 5/18 = {5/18:.3f}°C")
    ax1.legend(loc="upper right")

    # Label each bar with the naive °F
    for c, f, d in zip(max_c_vals, naive_f, dist_c):
        ax1.text(c, d + 0.005, str(f), ha="center", va="bottom",
                 fontsize=6, rotation=90, color="gray")

    ax1.set_xticks(max_c_vals[::2])
    ax1.set_xlim(lo - 0.7, hi + 0.7)
    ax1.set_ylim(0, 0.35)

    plt.tight_layout()

    if args.output:
        plt.savefig(args.output, dpi=150)
        print(f"Saved to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
