#!/usr/bin/env python3
"""Kalshi NBA market discovery and bracket matching.

Probes candidate series tickers on the Kalshi API, parses market titles
(over/under lines, player props, spreads), and matches projections to
the correct bracket for order placement.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional

log = logging.getLogger(__name__)

# Candidate series ticker prefixes to probe.  Kalshi NBA tickers are not
# publicly documented so we brute-force plausible patterns.
_CANDIDATE_PREFIXES = [
    # Game totals
    "KXNBA",
    "KXNBAOU",
    "KXNBATOTAL",
    "KXNBAPTS",
    # Spreads
    "KXNBASPREAD",
    "KXNBASP",
    # Player props
    "KXNBAPROPS",
    "KXNBAPROP",
    "KXNBAREB",
    "KXNBAAST",
    "KXNBA3PT",
    "KXNBA3PM",
    "KXNBAPR",
    # Quarters / halves
    "KXNBA1Q",
    "KXNBA1H",
    "KXNBAQ1",
    "KXNBAH1",
]


def discover_nba_series(client) -> Dict[str, str]:
    """Try candidate tickers via client.get_markets(series_ticker=...).

    Returns {description: series_ticker} for every ticker that has at least
    one market (open or otherwise).
    """
    found: Dict[str, str] = {}
    for prefix in _CANDIDATE_PREFIXES:
        try:
            # Check open first
            resp = client.get_markets(prefix, status="open")
            markets = resp.get("markets", [])
            if markets:
                sample = markets[0].get("title", prefix)
                found[prefix] = f"open ({len(markets)} markets) — e.g. {sample}"
                continue
            # Fall back to any status
            resp = client.get_markets(prefix, status=None)
            markets = resp.get("markets", [])
            if markets:
                sample = markets[0].get("title", prefix)
                found[prefix] = f"any ({len(markets)} markets) — e.g. {sample}"
        except Exception as exc:
            log.debug("Ticker %s lookup failed: %s", prefix, exc)
    return found


def get_game_markets(client, series_ticker: str,
                     status: Optional[str] = "open") -> List[dict]:
    """All markets under a series ticker.

    Each dict includes ticker, title, subtitle, yes_price, no_price,
    floor_strike, cap_strike, etc. — whatever Kalshi returns.
    """
    resp = client.get_markets(series_ticker, status=status)
    return resp.get("markets", [])


def parse_market_line(market: dict) -> Optional[dict]:
    """Extract line / threshold from a market's title and subtitle.

    Returns a dict with keys like:
        side:   'over' | 'under'
        line:   float (e.g. 215.5)
        player: str or None
        stat:   str or None (points, rebounds, assists, 3-pointers)

    Returns None if the title can't be parsed.
    """
    title = (market.get("title", "") + " " + market.get("subtitle", "")).strip()
    if not title:
        return None

    result: dict = {"raw_title": title, "ticker": market.get("ticker", "")}

    # --- Player prop: "LeBron James Over 25.5 points" ---
    player_match = re.search(
        r"(.+?)\s+(over|under)\s+([\d.]+)\s+(points?|rebounds?|assists?|3-?pointers?|threes?|steals?|blocks?)",
        title, re.IGNORECASE,
    )
    if player_match:
        result["player"] = player_match.group(1).strip()
        result["side"] = player_match.group(2).lower()
        result["line"] = float(player_match.group(3))
        result["stat"] = player_match.group(4).lower()
        return result

    # --- Game total: "Over 215.5" / "Under 215.5" ---
    total_match = re.search(r"(over|under)\s+([\d.]+)", title, re.IGNORECASE)
    if total_match:
        result["side"] = total_match.group(1).lower()
        result["line"] = float(total_match.group(2))
        return result

    # --- Spread: "Team -4.5" or "Team +4.5" ---
    spread_match = re.search(r"(.+?)\s+([+-][\d.]+)", title)
    if spread_match:
        result["team"] = spread_match.group(1).strip()
        result["line"] = float(spread_match.group(2))
        result["side"] = "spread"
        return result

    # --- Range bracket: "210 to 215" ---
    range_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:to|-)\s*(\d+(?:\.\d+)?)", title)
    if range_match:
        result["range_lo"] = float(range_match.group(1))
        result["range_hi"] = float(range_match.group(2))
        result["side"] = "range"
        return result

    # --- "X or above" / "X or below" ---
    above_match = re.search(r"([\d.]+)\s*or\s*(above|higher|more)", title, re.IGNORECASE)
    if above_match:
        result["line"] = float(above_match.group(1))
        result["side"] = "above"
        return result

    below_match = re.search(r"([\d.]+)\s*or\s*(below|lower|less)", title, re.IGNORECASE)
    if below_match:
        result["line"] = float(below_match.group(1))
        result["side"] = "below"
        return result

    # Fall back to floor_strike / cap_strike
    fs = market.get("floor_strike")
    cs = market.get("cap_strike")
    if fs is not None and cs is not None:
        result["range_lo"] = float(fs)
        result["range_hi"] = float(cs)
        result["side"] = "range"
        return result

    return None


def match_bracket(markets: List[dict], target: float) -> Optional[dict]:
    """Find the bracket where *target* falls.

    For over/under markets, returns the market whose line is closest to
    the target, along with the recommended side (over/under).

    For range brackets, returns the bracket that contains the target.

    Returns a dict with 'market' and 'buy_side' keys, or None.
    """
    # First try range/above/below brackets (exact containment)
    for mkt in markets:
        parsed = parse_market_line(mkt)
        if not parsed:
            continue

        if parsed.get("side") == "range":
            lo = parsed.get("range_lo", 0)
            hi = parsed.get("range_hi", 0)
            if lo <= target <= hi:
                return {"market": mkt, "buy_side": "yes", "parsed": parsed}

        elif parsed.get("side") == "above":
            if target >= parsed["line"]:
                return {"market": mkt, "buy_side": "yes", "parsed": parsed}

        elif parsed.get("side") == "below":
            if target <= parsed["line"]:
                return {"market": mkt, "buy_side": "yes", "parsed": parsed}

    # For over/under lines, find the line closest to target
    best = None
    best_dist = float("inf")
    for mkt in markets:
        parsed = parse_market_line(mkt)
        if not parsed:
            continue
        if parsed.get("side") in ("over", "under"):
            line = parsed["line"]
            dist = abs(target - line)
            if dist < best_dist:
                best_dist = dist
                best = (mkt, parsed)

    if best:
        mkt, parsed = best
        buy_side = "yes" if target > parsed["line"] else "no"
        direction = "over" if target > parsed["line"] else "under"
        return {"market": mkt, "buy_side": buy_side, "direction": direction,
                "parsed": parsed}

    return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    from paths import project_path
    load_dotenv(project_path(".env.demo"))

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    from bot.app import _get_client  # noqa: E402
    client = _get_client()

    print("Probing Kalshi for NBA series tickers...\n")
    found = discover_nba_series(client)
    if not found:
        print("No NBA series tickers found. Kalshi may not have NBA markets right now.")
    else:
        for ticker, desc in sorted(found.items()):
            print(f"  {ticker:20s}  {desc}")

        # Show sample markets for the first found series
        first_ticker = sorted(found.keys())[0]
        print(f"\nSample markets for {first_ticker}:")
        markets = get_game_markets(client, first_ticker)
        for m in markets[:10]:
            title = m.get("title", "")
            sub = m.get("subtitle", "")
            ticker = m.get("ticker", "")
            parsed = parse_market_line(m)
            print(f"  {ticker:40s} {title} | {sub}")
            if parsed:
                print(f"    -> parsed: {parsed}")
