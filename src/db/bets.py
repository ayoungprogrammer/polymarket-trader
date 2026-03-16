"""SQLite bet log — records all bets (including dry runs) and prevents duplicates.

Schema:
    bets (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,          -- ISO 8601 in America/Los_Angeles
        strategy TEXT NOT NULL,           -- "claude", "momentum", etc.
        market TEXT NOT NULL,             -- "KXHIGHMIA-26FEB28-T80:yes"
        price_cents INTEGER NOT NULL,     -- price per contract
        count INTEGER NOT NULL,           -- number of contracts
        bet_size_cents INTEGER NOT NULL,  -- price_cents * count
        dry_run INTEGER NOT NULL,         -- 1 if dry run, 0 if live
        metadata TEXT                     -- JSON: target_temp, confidence, reasoning
    )
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime
from typing import List, Optional
from zoneinfo import ZoneInfo

from paths import project_path

log = logging.getLogger(__name__)

DB_PATH = project_path("bets.db")


def init_db() -> None:
    """Create the bets table if it doesn't exist."""
    con = sqlite3.connect(DB_PATH)
    try:
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("""
            CREATE TABLE IF NOT EXISTS bets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                strategy TEXT NOT NULL,
                market TEXT NOT NULL,
                price_cents INTEGER NOT NULL,
                count INTEGER NOT NULL,
                bet_size_cents INTEGER NOT NULL,
                dry_run INTEGER NOT NULL,
                metadata TEXT
            )
        """)
        con.commit()
    finally:
        con.close()


def log_bet(
    strategy: str,
    market: str,
    price_cents: int,
    count: int,
    dry_run: bool,
    metadata: Optional[dict] = None,
) -> None:
    """Insert a bet row into the database."""
    con = sqlite3.connect(DB_PATH)
    try:
        con.execute(
            """INSERT INTO bets (timestamp, strategy, market, price_cents, count,
               bet_size_cents, dry_run, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                datetime.now(ZoneInfo("America/Los_Angeles")).isoformat(),
                strategy,
                market,
                price_cents,
                count,
                price_cents * count,
                1 if dry_run else 0,
                json.dumps(metadata) if metadata else None,
            ),
        )
        con.commit()
        log.info("Logged bet to DB: %s %s @ %dc x%d", strategy, market,
                 price_cents, count)
    finally:
        con.close()


def is_locked_today(strategy: str, series_ticker: str) -> bool:
    """Check if a bet was already placed today for this strategy + series ticker."""
    today_str = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d")
    con = sqlite3.connect(DB_PATH)
    try:
        row = con.execute(
            """SELECT COUNT(*) FROM bets
               WHERE strategy = ? AND market LIKE ? AND timestamp LIKE ?""",
            (strategy, f"{series_ticker}%", f"{today_str}%"),
        ).fetchone()
        return row[0] > 0
    finally:
        con.close()


def get_recent_bets(limit: int = 50) -> List[dict]:
    """Fetch recent bets from the database."""
    con = sqlite3.connect(DB_PATH)
    try:
        con.row_factory = sqlite3.Row
        rows = con.execute(
            "SELECT * FROM bets ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        con.close()
