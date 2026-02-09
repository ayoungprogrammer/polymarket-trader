#!/usr/bin/env python3
"""Generate temp_conversions.csv showing possible F values for each whole-degree C reading.

A whole-degree C reading of X means the true temp is X-0.5 to X+0.4 C.
This shows what that range maps to in whole-degree F.
"""

import csv

with open("temp_conversions.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["c", "min_f", "max_f"])
    for c in range(-30, 41):
        lo_f = round((c - 0.5) * 9 / 5 + 32)
        hi_f = round((c + 0.4) * 9 / 5 + 32)
        w.writerow([c, lo_f, hi_f])

print("Wrote temp_conversions.csv")
