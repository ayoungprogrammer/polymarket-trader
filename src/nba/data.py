#!/usr/bin/env python3
"""Historical NBA data for quarter-by-quarter analysis.

Uses nba_api (free, no key) for:
  - Live scoreboard (today's games, scores, period, clock)
  - Live box scores (player stats for in-progress games)
  - Historical team game logs with quarter scores
  - Player recent game stats

Caches quarter profiles to data/nba/quarter_profiles.json (refreshed daily).
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime, date
from typing import Dict, List, Optional

from paths import project_path

log = logging.getLogger(__name__)

DATA_DIR = project_path("data")
NBA_DATA_DIR = os.path.join(DATA_DIR, "nba")
TEAM_SCORES_PATH = os.path.join(NBA_DATA_DIR, "team_scores.csv")
QUARTER_PROFILES_PATH = os.path.join(NBA_DATA_DIR, "quarter_profiles.json")
QUARTER_PROFILES_HA_PATH = os.path.join(NBA_DATA_DIR, "quarter_profiles_ha.json")
PLAYER_GAME_LOGS_PATH = os.path.join(NBA_DATA_DIR, "player_game_logs.csv")

# nba_api rate-limit: add small delays between requests
_REQUEST_DELAY = 0.6  # seconds



def _ensure_data_dir() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)


def _ensure_nba_data_dir() -> None:
    os.makedirs(NBA_DATA_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Live scoreboard
# ---------------------------------------------------------------------------

def get_live_scoreboard() -> List[dict]:
    """Today's live games from NBA.com.

    Returns list of dicts:
        game_id, status (str), period (int), clock (str),
        home_team, away_team, home_score, away_score,
        home_q1..q4, away_q1..q4
    """
    from nba_api.live.nba.endpoints import ScoreBoard

    sb = ScoreBoard()
    games_raw = sb.get_dict()

    games = []
    for g in games_raw.get("scoreboard", {}).get("games", []):
        home = g.get("homeTeam", {})
        away = g.get("awayTeam", {})

        # Quarter scores from periods array
        home_periods = {p["period"]: p["score"] for p in home.get("periods", [])}
        away_periods = {p["period"]: p["score"] for p in away.get("periods", [])}

        game = {
            "game_id": g.get("gameId", ""),
            "status": _game_status_text(g),
            "period": g.get("period", 0),
            "clock": g.get("gameClock", ""),
            "home_team": home.get("teamTricode", ""),
            "away_team": away.get("teamTricode", ""),
            "home_score": home.get("score", 0),
            "away_score": away.get("score", 0),
            "home_q1": home_periods.get(1, 0),
            "home_q2": home_periods.get(2, 0),
            "home_q3": home_periods.get(3, 0),
            "home_q4": home_periods.get(4, 0),
            "away_q1": away_periods.get(1, 0),
            "away_q2": away_periods.get(2, 0),
            "away_q3": away_periods.get(3, 0),
            "away_q4": away_periods.get(4, 0),
        }
        games.append(game)

    return games


def _game_status_text(game: dict) -> str:
    """Convert NBA API game status to a human-readable string."""
    status = game.get("gameStatus", 1)
    if status == 1:
        return "scheduled"
    elif status == 2:
        return "live"
    elif status == 3:
        return "final"
    return f"unknown({status})"


# ---------------------------------------------------------------------------
# Live box score
# ---------------------------------------------------------------------------

def get_live_box_score(game_id: str) -> dict:
    """Live player stats for an in-progress game.

    Returns:
        {
            home_team: str,
            away_team: str,
            home_players: [{name, pts, reb, ast, fg3m, min_played}],
            away_players: [{name, pts, reb, ast, fg3m, min_played}],
        }
    """
    from nba_api.live.nba.endpoints import BoxScore

    bs = BoxScore(game_id=game_id)
    data = bs.get_dict()
    game = data.get("game", {})

    def _parse_players(team_data: dict) -> List[dict]:
        players = []
        for p in team_data.get("players", []):
            stats = p.get("statistics", {})
            # minutes come as "PT12M30.00S" ISO 8601 duration
            minutes_str = stats.get("minutesCalculated", "PT0M0S")
            minutes = _parse_iso_minutes(minutes_str)
            players.append({
                "name": p.get("name", ""),
                "pts": stats.get("points", 0),
                "reb": stats.get("reboundsTotal", 0),
                "ast": stats.get("assists", 0),
                "fg3m": stats.get("threePointersMade", 0),
                "min_played": minutes,
            })
        return players

    return {
        "home_team": game.get("homeTeam", {}).get("teamTricode", ""),
        "away_team": game.get("awayTeam", {}).get("teamTricode", ""),
        "home_players": _parse_players(game.get("homeTeam", {})),
        "away_players": _parse_players(game.get("awayTeam", {})),
    }


def _parse_iso_minutes(iso_str: str) -> float:
    """Parse ISO 8601 duration like 'PT12M30.00S' to minutes as float."""
    import re
    m = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:([\d.]+)S)?", iso_str or "")
    if not m:
        return 0.0
    hours = int(m.group(1) or 0)
    mins = int(m.group(2) or 0)
    secs = float(m.group(3) or 0)
    return hours * 60 + mins + secs / 60.0


# ---------------------------------------------------------------------------
# Season quarter scores (per-game, per-team)
# ---------------------------------------------------------------------------

def fetch_season_scores(season: str = "2025-26",
                        force_refresh: bool = False) -> "pd.DataFrame":
    """Fetch real per-game quarter scores for the entire season.

    Uses the BallDontLie API (free tier, 5 req/min) which returns quarter
    scores for all games from 2023+. Paginates through all games in ~13
    requests (100 per page).

    Requires BALLDONTLIE_API_KEY env var (free at https://app.balldontlie.io).

    Saves to data/nba/team_scores.csv with columns:
        game_id, game_date, team, opponent, home_away,
        q1, q2, q3, q4, ot1, ot2, ot3, total

    Returns the DataFrame.
    """
    import pandas as pd

    _ensure_nba_data_dir()

    if not force_refresh and os.path.isfile(TEAM_SCORES_PATH):
        log.info("Loading cached team scores from %s", TEAM_SCORES_PATH)
        return pd.read_csv(TEAM_SCORES_PATH)

    import requests

    api_key = os.environ.get("BALLDONTLIE_API_KEY", "")
    if not api_key:
        log.error("BALLDONTLIE_API_KEY not set. Get a free key at https://app.balldontlie.io")
        return pd.DataFrame()

    # season "2025-26" -> BallDontLie uses the start year: 2025
    season_year = int(season.split("-")[0])
    base_url = "https://api.balldontlie.io/v1/games"
    headers = {"Authorization": api_key}

    all_records = []
    cursor = None
    page = 0

    while True:
        params = {
            "seasons[]": season_year,
            "per_page": 100,
        }
        if cursor:
            params["cursor"] = cursor

        time.sleep(12)  # free tier: 5 req/min = 12s between requests
        try:
            resp = requests.get(base_url, headers=headers, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            log.error("BallDontLie API error: %s", e)
            break

        games = data.get("data", [])
        if not games:
            break

        for g in games:
            game_id = g.get("id", "")
            game_date = g.get("date", "")[:10]  # "2025-10-22T00:00:00.000Z"
            home = g.get("home_team", {})
            visitor = g.get("visitor_team", {})
            home_abbr = home.get("abbreviation", "")
            visitor_abbr = visitor.get("abbreviation", "")
            home_score = g.get("home_team_score", 0)
            visitor_score = g.get("visitor_team_score", 0)

            # Skip unplayed games
            if not home_score and not visitor_score:
                continue

            hq1 = g.get("home_q1", 0) or 0
            hq2 = g.get("home_q2", 0) or 0
            hq3 = g.get("home_q3", 0) or 0
            hq4 = g.get("home_q4", 0) or 0
            hot1 = g.get("home_ot1", 0) or 0
            hot2 = g.get("home_ot2", 0) or 0
            hot3 = g.get("home_ot3", 0) or 0

            vq1 = g.get("visitor_q1", 0) or 0
            vq2 = g.get("visitor_q2", 0) or 0
            vq3 = g.get("visitor_q3", 0) or 0
            vq4 = g.get("visitor_q4", 0) or 0
            vot1 = g.get("visitor_ot1", 0) or 0
            vot2 = g.get("visitor_ot2", 0) or 0
            vot3 = g.get("visitor_ot3", 0) or 0

            # Home team row
            all_records.append({
                "game_id": game_id,
                "game_date": game_date,
                "team": home_abbr,
                "opponent": visitor_abbr,
                "home_away": "home",
                "q1": hq1, "q2": hq2, "q3": hq3, "q4": hq4,
                "ot1": hot1, "ot2": hot2, "ot3": hot3,
                "total": home_score,
            })
            # Away team row
            all_records.append({
                "game_id": game_id,
                "game_date": game_date,
                "team": visitor_abbr,
                "opponent": home_abbr,
                "home_away": "away",
                "q1": vq1, "q2": vq2, "q3": vq3, "q4": vq4,
                "ot1": vot1, "ot2": vot2, "ot3": vot3,
                "total": visitor_score,
            })

        page += 1
        meta = data.get("meta", {})
        cursor = meta.get("next_cursor")
        log.info("Page %d: %d games fetched (%d records so far)",
                 page, len(games), len(all_records))

        if not cursor:
            break

    df = pd.DataFrame(all_records)
    if not df.empty:
        df.to_csv(TEAM_SCORES_PATH, index=False)
        log.info("Saved %d rows to %s", len(df), TEAM_SCORES_PATH)
    else:
        log.warning("No games fetched — CSV not written")
    return df


# ---------------------------------------------------------------------------
# Historical team quarter profiles
# ---------------------------------------------------------------------------

def build_team_quarter_profiles(season: str = "2025-26",
                                last_n: int = 20,
                                force_refresh: bool = False) -> Dict[str, dict]:
    """Per-quarter scoring averages for each team over last N games.

    Returns {team_abbr: {q1_avg, q2_avg, q3_avg, q4_avg, total_avg,
                          q1_std, q2_std, q3_std, q4_std, games_used}}.

    Cached to data/nba/quarter_profiles.json.  Refreshes if file is
    older than 24 hours or force_refresh is True.
    """
    _ensure_nba_data_dir()

    # Check cache freshness
    if not force_refresh and os.path.isfile(QUARTER_PROFILES_PATH):
        mtime = os.path.getmtime(QUARTER_PROFILES_PATH)
        age_hours = (time.time() - mtime) / 3600
        if age_hours < 24:
            log.info("Loading cached quarter profiles (%.1f hours old)", age_hours)
            with open(QUARTER_PROFILES_PATH) as f:
                return json.load(f)

    log.info("Building team quarter profiles (season=%s, last_n=%d)...", season, last_n)

    import pandas as pd
    import statistics

    # Load real quarter scores (fetch if needed)
    scores_df = fetch_season_scores(season=season)
    if scores_df.empty:
        log.warning("No season scores available, returning empty profiles")
        return {}

    profiles: Dict[str, dict] = {}

    for abbr, team_df in scores_df.groupby("team"):
        # Take last N games (sorted by date, most recent first)
        team_sorted = team_df.sort_values("game_date", ascending=False).head(last_n)

        q1s = team_sorted["q1"].tolist()
        q2s = team_sorted["q2"].tolist()
        q3s = team_sorted["q3"].tolist()
        q4s = team_sorted["q4"].tolist()
        totals = team_sorted["total"].tolist()

        if not totals:
            continue

        avg_total = statistics.mean(totals)
        std_total = statistics.stdev(totals) if len(totals) > 1 else 5.0

        profiles[abbr] = {
            "q1_avg": round(statistics.mean(q1s), 1),
            "q2_avg": round(statistics.mean(q2s), 1),
            "q3_avg": round(statistics.mean(q3s), 1),
            "q4_avg": round(statistics.mean(q4s), 1),
            "total_avg": round(avg_total, 1),
            "q1_std": round(statistics.stdev(q1s), 1) if len(q1s) > 1 else 2.0,
            "q2_std": round(statistics.stdev(q2s), 1) if len(q2s) > 1 else 2.0,
            "q3_std": round(statistics.stdev(q3s), 1) if len(q3s) > 1 else 2.0,
            "q4_std": round(statistics.stdev(q4s), 1) if len(q4s) > 1 else 2.0,
            "total_std": round(std_total, 1),
            "games_used": len(team_sorted),
        }

    # Save cache
    with open(QUARTER_PROFILES_PATH, "w") as f:
        json.dump(profiles, f, indent=2)
    log.info("Cached quarter profiles for %d teams", len(profiles))

    return profiles


# ---------------------------------------------------------------------------
# Home/away-split quarter profiles
# ---------------------------------------------------------------------------

def build_team_quarter_profiles_ha(
    season: str = "2025-26",
    last_n: int = 20,
    force_refresh: bool = False,
) -> Dict[str, dict]:
    """Per-quarter scoring averages split by home/away for each team.

    Returns dict keyed by "TEAM|home" or "TEAM|away" -> profile dict.
    (Using pipe separator because JSON keys must be strings.)

    Cached to data/nba/quarter_profiles_ha.json.
    Refreshes if file is older than 24 hours or force_refresh is True.
    """
    _ensure_nba_data_dir()

    if not force_refresh and os.path.isfile(QUARTER_PROFILES_HA_PATH):
        mtime = os.path.getmtime(QUARTER_PROFILES_HA_PATH)
        age_hours = (time.time() - mtime) / 3600
        if age_hours < 24:
            log.info("Loading cached home/away profiles (%.1f hours old)", age_hours)
            with open(QUARTER_PROFILES_HA_PATH) as f:
                return json.load(f)

    log.info("Building home/away quarter profiles (season=%s, last_n=%d)...", season, last_n)

    import statistics as _stats
    import pandas as pd

    scores_df = fetch_season_scores(season=season)
    if scores_df.empty:
        log.warning("No season scores available, returning empty profiles")
        return {}

    profiles: Dict[str, dict] = {}

    for (abbr, ha), group_df in scores_df.groupby(["team", "home_away"]):
        team_sorted = group_df.sort_values("game_date", ascending=False).head(last_n)

        q1s = team_sorted["q1"].tolist()
        q2s = team_sorted["q2"].tolist()
        q3s = team_sorted["q3"].tolist()
        q4s = team_sorted["q4"].tolist()
        totals = team_sorted["total"].tolist()

        if not totals:
            continue

        key = f"{abbr}|{ha}"
        profiles[key] = {
            "q1_avg": round(_stats.mean(q1s), 1),
            "q2_avg": round(_stats.mean(q2s), 1),
            "q3_avg": round(_stats.mean(q3s), 1),
            "q4_avg": round(_stats.mean(q4s), 1),
            "total_avg": round(_stats.mean(totals), 1),
            "q1_std": round(_stats.stdev(q1s), 1) if len(q1s) > 1 else 2.0,
            "q2_std": round(_stats.stdev(q2s), 1) if len(q2s) > 1 else 2.0,
            "q3_std": round(_stats.stdev(q3s), 1) if len(q3s) > 1 else 2.0,
            "q4_std": round(_stats.stdev(q4s), 1) if len(q4s) > 1 else 2.0,
            "total_std": round(_stats.stdev(totals), 1) if len(totals) > 1 else 5.0,
            "games_used": len(team_sorted),
        }

    with open(QUARTER_PROFILES_HA_PATH, "w") as f:
        json.dump(profiles, f, indent=2)
    log.info("Cached home/away quarter profiles: %d entries", len(profiles))

    return profiles


# ---------------------------------------------------------------------------
# Player recent stats
# ---------------------------------------------------------------------------

def get_player_recent_stats(player_name: str, last_n: int = 10,
                            season: str = "2025-26") -> Optional[dict]:
    """Last N games for a player: avg pts, reb, ast, 3pm, minutes.

    Returns None if player not found.
    """
    from nba_api.stats.static import players as nba_players
    from nba_api.stats.endpoints import PlayerGameLog

    import statistics

    # Find player ID
    matches = nba_players.find_players_by_full_name(player_name)
    if not matches:
        # Try partial match
        matches = [p for p in nba_players.get_active_players()
                   if player_name.lower() in p["full_name"].lower()]
    if not matches:
        log.warning("Player not found: %s", player_name)
        return None

    player_id = matches[0]["id"]
    player_full_name = matches[0]["full_name"]

    time.sleep(_REQUEST_DELAY)
    gl = PlayerGameLog(
        player_id=player_id,
        season=season,
        season_type_all_star="Regular Season",
    )
    rows = gl.get_normalized_dict().get("PlayerGameLog", [])

    if not rows:
        return None

    recent = rows[:last_n]

    def _avg(key):
        vals = [r.get(key, 0) for r in recent]
        return round(statistics.mean(vals), 1) if vals else 0.0

    return {
        "player_name": player_full_name,
        "player_id": player_id,
        "games": len(recent),
        "avg_pts": _avg("PTS"),
        "avg_reb": _avg("REB"),
        "avg_ast": _avg("AST"),
        "avg_fg3m": _avg("FG3M"),
        "avg_min": _avg("MIN"),
        "avg_stl": _avg("STL"),
        "avg_blk": _avg("BLK"),
    }


# ---------------------------------------------------------------------------
# Player game logs (bulk fetch)
# ---------------------------------------------------------------------------

def fetch_player_game_logs(
    season: str = "2025-26",
    min_games: int = 10,
    force_refresh: bool = False,
) -> "pd.DataFrame":
    """Fetch game logs for all active players who played >= min_games.

    Uses nba_api PlayerGameLog endpoint. Rate-limited to ~1 req/sec.
    Saves to data/nba/player_game_logs.csv.

    Columns: player_id, player_name, game_id, game_date, team,
             opponent, home_away, minutes, started, pts, reb, ast
    """
    import pandas as pd

    _ensure_nba_data_dir()

    if not force_refresh and os.path.isfile(PLAYER_GAME_LOGS_PATH):
        log.info("Loading cached player game logs from %s", PLAYER_GAME_LOGS_PATH)
        return pd.read_csv(PLAYER_GAME_LOGS_PATH)

    from nba_api.stats.endpoints import PlayerGameLog
    from nba_api.stats.static import players as nba_players

    active = nba_players.get_active_players()
    log.info("Fetching game logs for %d active players...", len(active))
    print(f"Fetching game logs for {len(active)} active players...", flush=True)

    all_rows: List[dict] = []
    fetched = 0
    skipped = 0

    for i, player in enumerate(active):
        pid = player["id"]
        pname = player["full_name"]

        rows = None
        for attempt in range(3):
            try:
                time.sleep(1.0 + attempt * 0.5)
                gl = PlayerGameLog(
                    player_id=pid,
                    season=season,
                    season_type_all_star="Regular Season",
                    timeout=60,
                )
                rows = gl.get_normalized_dict().get("PlayerGameLog", [])
                break
            except Exception as e:
                if attempt < 2:
                    continue
                log.warning("[%d/%d] Error for %s: %s", i + 1, len(active), pname, e)

        if rows is None:
            skipped += 1
            continue

        if len(rows) < min_games:
            skipped += 1
            continue

        fetched += 1
        for r in rows:
            matchup = r.get("MATCHUP", "")
            home_away = "home" if "vs." in matchup else "away"
            if "vs." in matchup:
                opponent = matchup.split("vs.")[-1].strip()
            elif "@" in matchup:
                opponent = matchup.split("@")[-1].strip()
            else:
                opponent = ""
            team = matchup.split(" ")[0] if matchup else ""

            all_rows.append({
                "player_id": pid,
                "player_name": pname,
                "game_id": r.get("Game_ID", ""),
                "game_date": r.get("GAME_DATE", ""),
                "team": team,
                "opponent": opponent,
                "home_away": home_away,
                "minutes": r.get("MIN", 0) or 0,
                "started": 1 if r.get("START_POSITION", "") else 0,
                "pts": r.get("PTS", 0) or 0,
                "reb": r.get("REB", 0) or 0,
                "ast": r.get("AST", 0) or 0,
            })

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(active)}] fetched={fetched} skipped={skipped} rows={len(all_rows)}",
                  flush=True)

    df = pd.DataFrame(all_rows)
    log.info("Done: %d players, %d game logs, %d skipped", fetched, len(df), skipped)
    print(f"  Done: {fetched} players, {len(df)} game logs, {skipped} skipped")

    df.to_csv(PLAYER_GAME_LOGS_PATH, index=False)
    log.info("Saved to %s", PLAYER_GAME_LOGS_PATH)
    print(f"  Saved to {PLAYER_GAME_LOGS_PATH}")
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    cmd = sys.argv[1] if len(sys.argv) > 1 else "scoreboard"

    if cmd == "scoreboard":
        print("Fetching live scoreboard...\n")
        games = get_live_scoreboard()
        if not games:
            print("No games today.")
        for g in games:
            print(f"  {g['away_team']} {g['away_score']}  @  "
                  f"{g['home_team']} {g['home_score']}  "
                  f"({g['status']} P{g['period']} {g['clock']})")
            print(f"    Away Q: {g['away_q1']}/{g['away_q2']}/{g['away_q3']}/{g['away_q4']}")
            print(f"    Home Q: {g['home_q1']}/{g['home_q2']}/{g['home_q3']}/{g['home_q4']}")

    elif cmd == "profiles":
        profiles = build_team_quarter_profiles(force_refresh="--refresh" in sys.argv)
        team = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("-") else None
        if team:
            p = profiles.get(team.upper())
            if p:
                print(f"\n{team.upper()} quarter profile:")
                for k, v in p.items():
                    print(f"  {k}: {v}")
            else:
                print(f"Team {team.upper()} not found. Available: {', '.join(sorted(profiles.keys()))}")
        else:
            for abbr in sorted(profiles.keys()):
                p = profiles[abbr]
                print(f"  {abbr}: total={p['total_avg']:.1f} "
                      f"Q1={p['q1_avg']:.1f} Q2={p['q2_avg']:.1f} "
                      f"Q3={p['q3_avg']:.1f} Q4={p['q4_avg']:.1f}")

    elif cmd == "scores":
        import pandas as pd
        df = fetch_season_scores(force_refresh="--refresh" in sys.argv)
        print(f"\nTotal rows: {len(df)}")
        teams = df["team"].nunique() if not df.empty else 0
        dates = df["game_date"].nunique() if not df.empty else 0
        print(f"Teams: {teams}, Game dates: {dates}")
        if not df.empty:
            print(f"\nSample rows:")
            print(df.head(6).to_string(index=False))

    elif cmd == "player":
        name = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else "LeBron James"
        print(f"Fetching recent stats for {name}...")
        stats = get_player_recent_stats(name)
        if stats:
            for k, v in stats.items():
                print(f"  {k}: {v}")
        else:
            print("Player not found.")

    elif cmd == "player-logs":
        import pandas as pd
        df = fetch_player_game_logs(force_refresh="--refresh" in sys.argv)
        print(f"\nTotal rows: {len(df)}")
        if not df.empty:
            players = df["player_name"].nunique()
            games = df["game_id"].nunique()
            print(f"Players: {players}, Games: {games}")
            print(f"Date range: {df['game_date'].min()} to {df['game_date'].max()}")

    elif cmd == "boxscore":
        game_id = sys.argv[2] if len(sys.argv) > 2 else ""
        if not game_id:
            print("Usage: python nba_data.py boxscore <game_id>")
            print("Get game_id from: python nba_data.py scoreboard")
            sys.exit(1)
        bs = get_live_box_score(game_id)
        for side in ["home", "away"]:
            team = bs[f"{side}_team"]
            print(f"\n{team} ({side}):")
            for p in bs[f"{side}_players"]:
                if p["min_played"] > 0:
                    print(f"  {p['name']:20s} {p['pts']:3d}pts {p['reb']:2d}reb "
                          f"{p['ast']:2d}ast {p['fg3m']:2d}3pm {p['min_played']:.0f}min")
    else:
        print("Usage: python -m nba.data [scoreboard|profiles|scores|player|player-logs|boxscore] [args]")
