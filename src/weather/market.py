"""Kalshi market bracket lookup for weather sites.

Fetches today's brackets for a given site, parses bounds, and optionally
includes orderbook prices.

Usage:
    cd src && python -m weather.market KLAX
    cd src && python -m weather.market KMIA --market low
"""
from __future__ import annotations

import logging
import os
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

from weather.sites import KALSHI_STATIONS, FORECAST_STATIONS

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Bracket parsing
# ---------------------------------------------------------------------------

def parse_bracket(market: dict) -> Tuple[str, int, int, str]:
    """Parse bracket bounds from a Kalshi market dict.

    Returns (label, lo, hi, ticker).  Open-ended brackets use sentinel
    values: ``-1000`` for "or below", ``1000`` for "or above".

    For open-ended brackets, ``floor_strike`` / ``cap_strike`` fields
    are used when available to get precise thresholds:
    - ``>X°`` with ``floor_strike=X`` → lo = X+1  (strictly above X)
    - ``<X°`` with ``cap_strike=X``  → hi = X-1  (strictly below X)
    """
    ticker = market.get("ticker", "")
    title = (market.get("title", "") + " " + market.get("subtitle", "")).strip()

    label = title
    lo: int = 0
    hi: int = 0

    range_match = re.search(r"(\d+)°?\s*(?:to|-)\s*(\d+)°?", title, re.IGNORECASE)
    if range_match:
        lo, hi = int(range_match.group(1)), int(range_match.group(2))
        label = f"{lo}° to {hi}°"
    else:
        # "82° or above", ">82°", "≥82°"
        above = re.search(r"(?:>|≥|>=)\s*(\d+)°?|(\d+)°?\s*or\s*(?:above|higher|more)", title, re.IGNORECASE)
        if above:
            raw = int(above.group(1) or above.group(2))
            fs = market.get("floor_strike")
            lo = int(float(fs)) + 1 if fs is not None else raw
            hi = 1000
            label = f"≥{lo}°"
        else:
            # "68° or below", "<68°", "≤68°"
            below = re.search(r"(?:<|≤|<=)\s*(\d+)°?|(\d+)°?\s*or\s*(?:below|lower|less|under)", title, re.IGNORECASE)
            if below:
                raw = int(below.group(1) or below.group(2))
                cs = market.get("cap_strike")
                hi = int(float(cs)) - 1 if cs is not None else raw
                lo = -1000
                label = f"≤{hi}°"
            else:
                # Last resort: floor_strike / cap_strike fields
                floor_strike = market.get("floor_strike")
                cap_strike = market.get("cap_strike")
                if floor_strike is not None and cap_strike is not None:
                    lo = int(float(floor_strike))
                    hi = int(float(cap_strike))
                    label = f"{lo}° to {hi}°"

    return label, lo, hi, ticker


def parse_all_brackets(markets: List[Dict]) -> List[Tuple[str, int, int]]:
    """Parse all markets into (ticker, lo, hi) tuples.

    Open-ended brackets keep sentinel values: ``-1000`` for "or below",
    ``1000`` for "or above".
    """
    results: List[Tuple[str, int, int]] = []
    for market in markets:
        _, lo, hi, ticker = parse_bracket(market)
        if lo == 0 and hi == 0:
            continue
        results.append((ticker, lo, hi))
    return results


def find_matching_bracket(markets: List[Dict], temp_f: float) -> Optional[Dict]:
    """Find the market bracket that contains *temp_f* (rounded to int).

    Returns the first matching market dict, or ``None``.
    """
    temp_int = int(round(temp_f))
    log.info(f"Looking for bracket containing {temp_int}°F (raw: {temp_f}°F)")

    for market in markets:
        _, lo, hi, _ = parse_bracket(market)
        if lo == 0 and hi == 0:
            continue
        if lo <= temp_int <= hi:
            log.info(f"Matched bracket: {market.get('ticker', '')} ({lo}–{hi}°F)")
            return market

    return None


def get_today_brackets(
    client,
    site: str,
    market_type: str = "high",
    include_prices: bool = False,
) -> List[dict]:
    """Fetch today's market brackets for a site.

    Args:
        client: KalshiClient instance.
        site: ICAO code (e.g. "KLAX").
        market_type: "high" or "low".
        include_prices: if True, fetch orderbook bid/ask for each bracket.

    Returns list of dicts sorted by bracket lower bound::

        {"ticker": str, "label": str, "lo": int, "hi": int,
         "yes_bid": int, "yes_ask": int}

    Open-ended brackets use sentinels: lo=-1000 for "or below",
    hi=1000 for "or above".
    """
    from bot.app import _discover_series_ticker

    series = _discover_series_ticker(client, site, market_type)
    if not series:
        log.warning(f"[{site}] Could not discover {market_type} series ticker")
        return []

    coords = FORECAST_STATIONS.get(site)
    tz_name = coords[2] if coords else "America/Los_Angeles"
    station_tz = ZoneInfo(tz_name)
    today_suffix = datetime.now(station_tz).strftime("%y%b%d").upper()

    resp = client.get_markets(series)
    markets = resp.get("markets", [])
    today_markets = [m for m in markets if today_suffix in m.get("ticker", "")]

    brackets = []
    for m in today_markets:
        label, lo, hi, ticker = parse_bracket(m)

        yes_bid, yes_ask = 0, 0
        if include_prices:
            try:
                ob = client.get_orderbook(ticker)
                yes_bids = ob.get("yes", [])
                no_bids = ob.get("no", [])
                if yes_bids:
                    yes_bid = yes_bids[-1][0]
                if no_bids:
                    yes_ask = 100 - no_bids[-1][0]
            except Exception:
                pass

        brackets.append({
            "ticker": ticker,
            "label": label,
            "lo": lo,
            "hi": hi,
            "yes_bid": yes_bid,
            "yes_ask": yes_ask,
        })

    brackets.sort(key=lambda b: (b["lo"], b["hi"]))
    return brackets


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys

    from dotenv import load_dotenv
    from paths import project_path

    load_dotenv(project_path(".env.demo"))

    parser = argparse.ArgumentParser(description="Show today's Kalshi brackets for a weather site")
    parser.add_argument("site", nargs="?", default="KLAX", help="ICAO site code (default: KLAX)")
    parser.add_argument("--market", choices=["high", "low"], default="high", help="Market type (default: high)")
    parser.add_argument("--no-prices", action="store_true", help="Skip orderbook price lookup")
    args = parser.parse_args()

    site = args.site.upper()
    if not site.startswith("K"):
        site = "K" + site
    if site not in KALSHI_STATIONS:
        print(f"Unknown site: {site}")
        from weather.sites import ALL_SITES
        print(f"Available: {', '.join(ALL_SITES)}")
        sys.exit(1)

    city, suffix = KALSHI_STATIONS[site]

    from bot.app import KalshiClient

    api_key = os.getenv("KALSHI_API_KEY_ID")
    pk_path = os.getenv("KALSHI_PRIVATE_KEY_PATH")
    env = os.getenv("KALSHI_ENV", "demo")
    if not api_key or not pk_path:
        print("Set KALSHI_API_KEY_ID and KALSHI_PRIVATE_KEY_PATH in .env.demo")
        sys.exit(1)

    try:
        client = KalshiClient(api_key, pk_path, env=env)
    except Exception as e:
        print(f"Could not initialize Kalshi client: {e}")
        sys.exit(1)
    brackets = get_today_brackets(client, site, args.market, include_prices=not args.no_prices)

    print(f"\n{site} — {city} ({args.market}, {env})")
    if not brackets:
        print("  No brackets found for today")
        sys.exit(0)

    print(f"  {'[lo, hi]':<16} {'Label':<20} {'Bid':>6} {'Ask':>6}  {'Ticker'}")
    print(f"  {'─'*16} {'─'*20} {'─'*6} {'─'*6}  {'─'*40}")
    for b in brackets:
        bid_s = f"{b['yes_bid']}¢" if b["yes_bid"] else "—"
        ask_s = f"{b['yes_ask']}¢" if b["yes_ask"] else "—"
        bounds = f"[{b['lo']}, {b['hi']}]"
        print(f"  {bounds:<16} {b['label']:<20} {bid_s:>6} {ask_s:>6}  {b['ticker']}")
    print()
