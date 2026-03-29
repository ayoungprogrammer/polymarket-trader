"""Unified site configuration loaded from configs/weather/sites.yaml.

Provides two backward-compatible dicts so existing code keeps working:

- ``KALSHI_STATIONS``  – ``{site: (city, kalshi_suffix)}`` (was app.py STATIONS)
- ``FORECAST_STATIONS`` – ``{site: (lat, lon, timezone)}`` (was forecast.py STATIONS)
- ``ALL_SITES``         – sorted list of all site codes
"""
from __future__ import annotations

import os
from typing import Dict, List, Tuple

import yaml

_CONFIG_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "configs", "weather")
)
_CONFIG_PATH = os.path.join(_CONFIG_DIR, "sites.yaml")
_TRAINING_CONFIG_PATH = os.path.join(_CONFIG_DIR, "training_sites.yaml")


def _load() -> dict:
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _load_training() -> dict:
    if not os.path.isfile(_TRAINING_CONFIG_PATH):
        return {}
    with open(_TRAINING_CONFIG_PATH) as f:
        return yaml.safe_load(f) or {}


_DATA: dict = _load()
_TRAINING_DATA: dict = _load_training()

# {site: (city, kalshi_suffix)} — drop-in replacement for bot/app.py STATIONS
KALSHI_STATIONS: Dict[str, Tuple[str, str]] = {
    site: (cfg["city"], cfg["kalshi_suffix"])
    for site, cfg in _DATA.items()
}

# {site: (lat, lon, timezone)} — drop-in replacement for weather/forecast.py STATIONS
FORECAST_STATIONS: Dict[str, Tuple[float, float, str]] = {
    site: (cfg["lat"], cfg["lon"], cfg["timezone"])
    for site, cfg in _DATA.items()
}

ALL_SITES: List[str] = sorted(_DATA.keys())

# Training-only sites (no Kalshi markets, used for model training data)
TRAINING_STATIONS: Dict[str, Tuple[float, float, str]] = {
    site: (cfg["lat"], cfg["lon"], cfg["timezone"])
    for site, cfg in _TRAINING_DATA.items()
}

TRAINING_SITES: List[str] = sorted(_TRAINING_DATA.keys())

# Combined: Kalshi sites + training-only sites
ALL_SITES_WITH_TRAINING: List[str] = sorted(set(ALL_SITES) | set(TRAINING_SITES))


def get_site_config(site: str) -> dict:
    """Return the full config dict for a site, or raise KeyError."""
    if site in _DATA:
        return _DATA[site]
    if site in _TRAINING_DATA:
        return _TRAINING_DATA[site]
    raise KeyError(f"Unknown site: {site}")


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Show site configuration and live Kalshi market info")
    parser.add_argument("site", nargs="?", default="KLAX", help="ICAO site code (default: KLAX)")
    parser.add_argument("--all", action="store_true", help="List all configured sites")
    parser.add_argument("--market", choices=["high", "low"], default="high", help="Market type (default: high)")
    args = parser.parse_args()

    if args.all:
        print(f"{'Site':<8} {'City':<20} {'Kalshi':<8} {'Lat':>8} {'Lon':>10} {'Timezone'}")
        print(f"{'─'*8} {'─'*20} {'─'*8} {'─'*8} {'─'*10} {'─'*25}")
        for site in ALL_SITES:
            cfg = _DATA[site]
            print(f"{site:<8} {cfg['city']:<20} {cfg['kalshi_suffix']:<8} "
                  f"{cfg['lat']:>8.4f} {cfg['lon']:>10.4f} {cfg['timezone']}")
        sys.exit(0)

    site = args.site.upper()
    if not site.startswith("K"):
        site = "K" + site
    if site not in _DATA:
        print(f"Unknown site: {site}")
        print(f"Available: {', '.join(ALL_SITES)}")
        sys.exit(1)

    cfg = _DATA[site]
    city, suffix = cfg["city"], cfg["kalshi_suffix"]
    lat, lon, tz = cfg["lat"], cfg["lon"], cfg["timezone"]

    print(f"\n{'='*60}")
    print(f"  {site} — {city}")
    print(f"{'='*60}")
    print(f"  Coordinates:    {lat}, {lon}")
    print(f"  Timezone:       {tz}")
    print(f"  Kalshi suffix:  {suffix}")
    print(f"  High ticker:    KXHIGH{suffix}")
    print(f"  Low ticker:     KXLOW{suffix}")
    print(f"  Kalshi URL:     https://kalshi.com/markets/kxhigh{suffix.lower()}")
    print(f"  NWS timeseries: https://www.weather.gov/wrh/timeseries?site={site}")
    print()
    print(f"  For live brackets: python -m weather.market {site} --market {args.market}")
