# Kalshi Temperature Betting Bot

## Project Overview
Multi-city bot that bets on Kalshi daily high/low temperature markets (KXHIGH*/KXLOW*) using NWS weather data. Supports 12 US cities with parallel watch mode. One-shot or continuous polling via `--watch`.

## Project Structure
```
polymarket-trader/
  run.py                          # Main entry point (replaces `python bot.py`)
  conftest.py                     # Adds src/ to pytest path
  .env.demo                       # Kalshi demo API credentials
  bets.db                         # SQLite bet log (created at runtime)
  momentum_*.png                  # Per-site momentum charts (written at runtime)

  tests/
    test_core.py                  # Unit tests (58 tests)

  src/
    paths.py                      # PROJECT_ROOT + project_path() helper
    bot/
      app.py                      # Kalshi API client, strategies, Flask dashboard,
                                  #   Telegram notifications, SQLite bet log (~2400 lines)
    weather/
      weather.py                  # Synoptic/MesoWest API + NWS HTML observation parsing
      forecast.py                 # NWS hourly forecast (gridpoint API + DWML fallback),
                                  #   STATIONS dict with (lat, lon, timezone) per site
      prediction.py               # Momentum computation (WLS), extrapolation, charting
      analysis.py                 # Historical peak consistency analysis (fetches ~100 days)
      backtest.py                 # Grid-search backtest of momentum strategy params
      temp_conversions.py         # Generates C-to-F rounding reference CSV
    nba/
      data.py                     # NBA live scoreboard + team quarter profiles (nba_api)
      markets.py                  # Kalshi NBA market discovery + ticker parsing
      strategy.py                 # NBA quarter projection + edge detection

  data/                           # Cached data files (gitignored runtime output)
    history_{SITE}.csv            # ~100 days of 5-min Synoptic observations per site
                                  #   columns: timestamp, temperature_f, temperature_c
                                  #   ~31k rows each, 15 sites total
    daily_highs.csv               # Extracted daily highs/lows per site
                                  #   columns: site, date, daily_high_f, daily_low_f,
                                  #   daily_range_f, peak_time, peak_hour, n_readings
    analysis_results.json         # Per-site peak consistency metrics (peak_hour_mean/std,
                                  #   decline_rate, time_to_3deg, forecast_accuracy, score)
    backtest_results.json         # Grid-search results: optimal momentum params per site
                                  #   keys: default_params, optimal_per_site
    nba_quarter_profiles.json     # Cached NBA team quarter scoring profiles (refreshed daily)

  charts/                         # Analysis output charts
    peak_consistency_{SITE}.png   # Peak timing consistency scatter plots (per site)
    decline_profile_{SITE}.png    # Post-peak decline rate profiles (per site)
    comparison_summary.png        # Cross-site comparison summary chart

  html/                           # Flask dashboard static files
    dashboard.html                # Main dashboard — all sites overview
    site.html                     # Single-site detail view
    nba.html                      # NBA betting dashboard
    analysis.html                 # Analysis overview page
    analysis_site.html            # Per-site analysis detail page
    css/                          # Stylesheets (style.css, dashboard.css, site.css,
                                  #   nba.css, analysis.css)
    js/                           # Client JS (dashboard.js, site.js, nba.js,
                                  #   analysis.js, analysis_site.js)
```

## Architecture

### Source Code (`src/`)
- `src/paths.py` — `PROJECT_ROOT` and `project_path()` for resolving paths relative to project root
- `src/bot/app.py` — main script: Kalshi API client, multi-market orchestration, strategies, order placement, Telegram notifications, SQLite bet log, Flask dashboard
- `src/weather/weather.py` — weather data ingestion (Synoptic/MesoWest API, NWS API, NWS HTML parsing)
- `src/weather/forecast.py` — NWS hourly forecast ingestion (gridpoint API + DWML fallback), station coordinates/timezones
- `src/weather/prediction.py` — momentum computation (WLS), extrapolation, charting, CLI diagnostic tool
- `src/weather/analysis.py` — historical peak consistency analysis (~100 days per site)
- `src/weather/backtest.py` — grid-search backtest of momentum strategy params against historical data
- `src/weather/temp_conversions.py` — reference table for C-to-F rounding ranges
- `src/nba/data.py` — NBA live scoreboard + team quarter profiles via nba_api
- `src/nba/markets.py` — Kalshi NBA market discovery + parsing
- `src/nba/strategy.py` — NBA quarter-by-quarter projection + edge detection

## Import Conventions
- All cross-module imports use **absolute package imports**: `from weather.forecast import ...`, `from bot.app import ...`, `from nba.data import ...`
- Do NOT manipulate `sys.path` — run scripts as modules from `src/`: `cd src && python -m weather.strategy --site KLAX`
- File paths use `from paths import project_path` to resolve relative to project root (e.g. `project_path("data", "file.json")`)

## Multi-Market Support

### Station Config (`src/bot/app.py` STATIONS dict)
```python
# (city_name, kalshi_suffix)
STATIONS = {
    "KLAX": ("Los Angeles", "LAX"),       # KXHIGHLAX
    "KMIA": ("Miami", "MIA"),             # KXHIGHMIA
    "KSFO": ("San Francisco", "TSFO"),    # KXHIGHTSFO
    "KORD": ("Chicago", "CHI"),           # KXHIGHCHI
    "KDEN": ("Denver", "DEN"),            # KXHIGHDEN (low: KXLOWTDEN)
    "KPHX": ("Phoenix", "TPHX"),          # KXHIGHTPHX
    "KOKC": ("Oklahoma City", "TOKC"),    # KXHIGHTOKC
    "KATL": ("Atlanta", "TATL"),          # KXHIGHTATL
    "KDFW": ("Dallas", "TDAL"),           # KXHIGHTDAL
    "KSAT": ("San Antonio", "TSATX"),     # KXHIGHTSATX
    "KHOU": ("Houston", "THOU"),          # KXHIGHTHOU
    "KMSP": ("Minneapolis", "TMIN"),      # KXHIGHTMIN
}
```
Kalshi suffixes do NOT reliably match ICAO codes. Most were verified from kalshi.com URLs. High and low tickers can differ for the same city (e.g. Denver: high=`DEN`, low=`TDEN`).

### Station Coordinates (`src/weather/forecast.py` STATIONS dict)
Separate dict with `(lat, lon, timezone)` for each station. Used by `ForecastIngestion` for NWS gridpoint lookups and by `src/bot/app.py` for timezone-aware date/time operations.

### Ticker Discovery (`_discover_series_ticker`)
At startup, queries the Kalshi API for each site to find valid series tickers:
1. Tries configured suffix first (from STATIONS dict)
2. Falls back to ICAO[1:] derivation
3. Also tries T-prefixed variant (e.g. `TMIA` for low markets)
4. Two-pass: checks `status="open"` first, then `status=None` (any status) to confirm series exists even when today's contract isn't open yet

### Timezone Handling
- **Station timezone** (from `forecast.py` STATIONS): used for `today_suffix` date computation (matching Kalshi ticker dates), forecast remaining-hour filtering, observation timestamps
- **Eastern Time**: used for market close (11:59 PM ET = Kalshi standard)
- **Important**: All `today_suffix` computations use the station's local timezone, NOT system time or Pacific. This matters for east coast stations where the date rolls over earlier.

## SQLite Bet Log + Daily Lock

### Database (`bets.db`)
```sql
CREATE TABLE bets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,          -- ISO 8601 in America/Los_Angeles
    strategy TEXT NOT NULL,           -- "claude", "momentum", etc.
    market TEXT NOT NULL,             -- "KXHIGHMIA-26FEB28-T80:yes"
    price_cents INTEGER NOT NULL,     -- price per contract
    count INTEGER NOT NULL,           -- number of contracts
    bet_size_cents INTEGER NOT NULL,  -- price_cents * count
    dry_run INTEGER NOT NULL,         -- 1 if dry run, 0 if live
    metadata TEXT                     -- JSON: target_temp, confidence, reasoning
);
```

### `_log_bet()` — records every bet (including dry runs) to the DB
### `_is_locked_today()` — prevents duplicate bets
- Checks if a bet was already placed today for a given `(strategy, series_ticker)` pair
- Uses `America/Los_Angeles` for the date check (consistent with bet timestamps)
- Queried at the start of `_watch_loop` — if locked, that site's thread exits immediately

## Watch Mode (`--watch`)

### Flow
1. Parse `--site` (comma-separated ICAO codes, default: all 12)
2. Discover series tickers for all requested sites
3. Start Telegram listener thread (handles `/predict [SITE]` commands)
4. Spawn one daemon thread per site via `_run_single_market_watch()`
5. Each thread runs `_watch_loop()` independently → on trigger, calls `_place_claude_order()` or `_place_bracket_order()`
6. Main thread joins all watch threads
7. SIGINT handler for clean Ctrl+C exit

### `_watch_loop()` per-site polling
1. **Daily lock check**: skip if already bet today for this strategy+ticker
2. **Forecast-based sleep**: fetch forecast peak time, sleep until 2h before peak (avoids wasting API calls in the morning)
3. **Poll loop** (every `--watch-interval` seconds, default 300 = 5 min):
   - Check market close (11:59 PM ET) — stop if reached
   - Run the selected strategy for this site
   - If strategy returns a trigger (LOCKED / Claude says bet), return result
   - If no trigger, sleep and retry
4. Returns `float` (target temp) for non-claude strategies, `dict` (decision) for claude strategy, or `None` if market close reached

### Log Prefixing
All watch-mode log messages are prefixed with `[SITE]` (e.g. `[KLAX]`, `[KMIA]`) so output from parallel threads is distinguishable.

## One-Shot Mode (no `--watch`)
Iterates over each site sequentially:
1. Run the selected strategy
2. If bet triggered, place order (or log dry run)
3. Move to next site

## Order Placement

### `_place_claude_order()` — for claude strategy
- Claude's decision includes exact ticker and side (yes/no)
- Looks up orderbook: best ask = `100 - opposite_bids[-1][0]`
- Sanity checks: reject if ask > 95c (no margin) or < 30c (bad match)
- Contract count: `$10 / ask_price`, min 1
- Logs to DB, sends Telegram notification
- **Currently DISABLED** — order placement code is commented out, logs what it would do

### `_place_bracket_order()` — for non-claude strategies
- Queries open markets, filters to today's date suffix (station timezone)
- Calls `find_matching_bracket()` to match target temp to a bracket
- Same orderbook/pricing/sanity logic as claude order placement
- Always buys YES on the matching bracket

### Bracket Matching (`find_matching_bracket()`)
Parses market title/subtitle for:
- Range brackets: "68° to 69°" → matches if temp in [68, 69]
- Above brackets: "82° or above" → matches if temp ≥ 82
- Below brackets: "68° or below" → matches if temp ≤ 68
- Falls back to `floor_strike`/`cap_strike` fields

## Strategies

### metar6h (default, `--strategy metar6h`)
- Waits for 6h METAR report (3:53 PM PT for high, 9:53 AM PT for low)
- Reads max/min from NWS HTML page (`fetch_nws_observation()` / `parse_6hr_section()`)
- Confirms with forecast confidence check (HIGH/MEDIUM/LOW)
- **LAX-specific timing** — the 3 PM PT check doesn't apply to other timezones

### peak-track (`--strategy peak-track`)
- Checks if last 3 Synoptic readings are monotonically decreasing AND below forecast peak
- Status: LOCKED / NEAR_PEAK / TOO_EARLY / ERROR
- High-only (no low support)

### momentum (`--strategy momentum`)
- Computes °F/hr rate of change over 30-min sliding window
- Grades: LOCKED (≤ -2°F/hr) / LIKELY (≤ -1) / POSSIBLE (≤ -0.5) / TOO_EARLY
- High-only

### claude (`--strategy claude`)
- Sends all weather data + market prices to Claude API for analysis
- Claude uses `make_bet_decision` tool to return structured decision
- Data sent: live observations (24h), 6h METAR max/min, momentum rates, forecast, Kalshi bracket prices
- Momentum chart saved per-site: `momentum_{SITE}.png`
- System prompt is station-agnostic — user message specifies the exact market/ticker

## Telegram Integration

### Notifications (`_send_telegram()`)
- Sends chart image + caption on bet triggers
- Uses `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` env vars
- Silently logs on failure — never blocks betting

### Command Listener (`_telegram_listener()`)
- Long-polls `getUpdates` in a background daemon thread
- Handles `/predict [SITE]` — runs Claude strategy for the specified site (default: KLAX), sends chart + decision
- Looks up site in `site_tickers` dict, falls back to discovery if not found

## Kalshi API

### Authentication
- RSA-PSS signature of `timestamp_ms + METHOD + path` (no query params)
- Headers: `KALSHI-ACCESS-KEY`, `KALSHI-ACCESS-SIGNATURE`, `KALSHI-ACCESS-TIMESTAMP`
- Salt length: `PSS.MAX_LENGTH` (not `DIGEST_LENGTH` — unavailable in cryptography 3.x)

### Endpoints
- Demo: `https://demo-api.kalshi.co`, Prod: `https://api.elections.kalshi.com`
- API prefix: `/trade-api/v2`
- `GET /markets` — list markets by series ticker (optional status filter)
- `GET /markets/{ticker}/orderbook` — yes/no bid arrays sorted ascending
- `GET /markets/trades` — trade history with pagination
- `POST /markets/{ticker}/orders` — place order

### Orderbook Structure
- `yes[]` = resting YES bids, `no[]` = resting NO bids, both `[price, quantity]` sorted ascending
- Best YES bid = `yes[-1][0]`
- Best YES ask = `100 - no[-1][0]` (selling NO = buying YES)

## Kalshi Settlement Rules

### Settlement Source
Markets settle on the **NWS Climatological Report (Daily)** for each city's station.

### Critical: Which CLI Report Settles
- **"Only the first official non-preliminary report"** is used for resolution
- The preliminary CLI (e.g. 5 PM PST for LAX) does NOT count
- The **final CLI** publishes the **morning after** (~7-8 AM ET next day)
- CLI reports whole-degree Fahrenheit (rounded from precise °C observations)

### Timeline (example: LAX Feb 8 market)
```
Feb 8 (measurement day):
  ~3:53 PM PST    6h METAR max drops (probable daily high)
  ~5:00 PM PST    Preliminary CLI published — NOT used for settlement
   8:59 PM PST    Market closes (11:59 PM ET)
  11:59 PM PST    Measurement day ends (midnight-to-midnight)

Feb 9:
  ~7-8 AM ET      Final CLI published — THIS settles the market
  10:00 AM ET     Market expires
```

### Bracket Types
- "between" (e.g. "70 to 71" is inclusive: ≥ 70 and ≤ 71)
- "greater than" (e.g. >82 means 83+ resolves Yes)
- "less than" (e.g. <68 means 67 or below resolves Yes)

## Weather Data Sources

### 1. Synoptic/MesoWest API (best) — `SynopticIngestion` in `src/weather/weather.py`
- 5-min resolution, dedicated `air_temp_high_6_hour_set_1` field
- T-group precision (0.1°F) at :53 METAR observations
- NWS public token, requires `Referer: weather.gov` header
- Accepts any ICAO station code

### 2. NWS Gridpoint API — `ForecastIngestion` in `src/weather/forecast.py`
- Hourly forecast including past hours of today
- Requires lat/lon → grid resolution step
- Falls back to DWML XML endpoint

### 3. NWS HTML page — `fetch_nws_observation()` in `src/weather/weather.py`
- 6h and 24h max/min from tgftp.nws.noaa.gov
- Regex parsing (BeautifulSoup fails on nested tables)

## Temperature Precision
- **Hourly METAR** (:53): 0.1°C via T-group — most precise
- **5-min auto obs**: whole-degree C only (~2°F uncertainty)
- **6h METAR max/min** (`1snTTT`/`2snTTT`): 0.1°C, continuously tracked by ASOS
- **CLI report**: rounds to whole °F

## CLI Usage
```bash
# Watch all 12 markets in parallel (default)
python run.py --strategy claude --market high --watch --dry-run

# Watch specific sites only
python run.py --strategy claude --market high --watch --dry-run --site KLAX,KMIA

# One-shot for a single site
python run.py --strategy claude --market high --dry-run --site KPHX

# One-shot with simulated time
python run.py --strategy claude --market high --dry-run --site KLAX --simulate-time 2026-02-08T14:00:00-0800

# Other strategies
python run.py --market high --dry-run --site KLAX              # metar6h (default)
python run.py --strategy peak-track --dry-run --site KLAX      # peak-track
python run.py --strategy momentum --dry-run --site KLAX        # momentum
python run.py --strategy momentum --force --dry-run --site KLAX  # override if not LOCKED

# Prediction diagnostic (standalone)
cd src && python -m weather.prediction KLAX
cd src && python -m weather.prediction KLAX --csv out.csv

# Claude strategy (standalone)
cd src && python -m weather.strategy --site KLAX
cd src && python -m weather.strategy --site KMIA --market low

# NBA tools
cd src && python -m nba.data scoreboard
cd src && python -m nba.strategy --scan
```

## Python Notes
- Target: Python 3.9 — use `from __future__ import annotations` and `typing.Optional`
- All timezone-aware datetime operations use `zoneinfo.ZoneInfo`
