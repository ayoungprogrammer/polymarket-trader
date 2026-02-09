# Kalshi LA High Temp Bot

## Project Overview
Bot that bets on Kalshi KXHIGHLAX markets (LA daily high temperature) using NWS weather data. One-shot cron-friendly script.

## Architecture
- `bot.py` — main betting script with Kalshi API client, bracket matching, order placement
- `weather.py` — weather data ingestion (NWS API, Synoptic/MesoWest API, NWS HTML parsing)
- `.env.demo` — Kalshi demo API credentials
- `temp_conversions.py` — reference table for C-to-F rounding ranges

## Kalshi KXHIGHLAX Settlement Rules

### Settlement Source
The market settles on the **NWS Climatological Report (Daily)** for LAX:
https://forecast.weather.gov/product.php?site=LOX&product=CLI&issuedby=LAX

### Critical: Which CLI Report Settles
From the GLOBALTEMPERATURE rulebook:
- **"Only the first official non-preliminary report"** is used for resolution
- The 5 PM PST preliminary CLI does NOT count — it can be updated if temps change later
- The **final CLI** is published the **morning after** (~7-8 AM ET next day), covering full midnight-to-midnight
- "Revisions after the Expiration Date are not included" — only the first non-preliminary report matters
- **"Contract resolution is based on the full precision reported by the Source Agency"** — but CLI only reports whole °F, so full precision = whole degrees F

### Timeline (for a Feb 8 market)
```
Feb 8 (measurement day):
  ~3:53 PM PST    6h METAR max drops (probable daily high)
  ~5:00 PM PST    Preliminary CLI published — NOT used for settlement
   8:59 PM PST    Market closes (11:59 PM ET)
  11:59 PM PST    Measurement day ends (midnight-to-midnight)

Feb 9:
  ~7-8 AM ET      Final (non-preliminary) CLI published — THIS settles the market
  10:00 AM ET     Market expires
```

### Key Implication
You CANNOT read the final settlement value before the market closes. You are always
betting on a prediction. The 6h METAR max + confirming temps are dropping is your best
signal, but a late-day temperature spike is a real risk.

### Bracket Types
- "greater than" (e.g. >82 means 83+ resolves Yes)
- "between" (e.g. "70 to 71" is inclusive: >= 70 and <= 71)
- "less than" (e.g. <68 means 67 or below resolves Yes)

### Rounding
The CLI whole-degree F value comes from rounding the precise C observation:
- 6h METAR max gives true temp to 0.1C (e.g. 20.6C)
- 20.6C = 69.08F -> CLI rounds to **69F**
- Whole-degree C readings (from 5-min obs) have a +/-0.5C uncertainty = 2-3F range
- The 6h max is the accurate one; use it to predict the CLI value

## Weather Data Sources (ranked by usefulness)

### 1. Synoptic/MesoWest API (best)
- Same source as weather.gov/wrh/timeseries page
- `SynopticIngestion` class in weather.py
- Has dedicated `air_temp_high_6_hour_set_1` field — no METAR decoding needed
- T-group precision (0.1F) at :53 METAR observations
- NWS public token, requires `Referer: weather.gov` header
- Endpoint: `https://api.synopticdata.com/v2/stations/timeseries?STID=KLAX&...&token=7c76618b66c74aee913bdbae4b448bdd`

### 2. NWS api.weather.gov
- `WeatherIngestion` class in weather.py
- `maxTemperatureLast24Hours` field exists but is always null for KLAX
- 6h/24h data only in raw METAR `rawMessage` field — must decode remarks
- METAR remark groups: `1snTTT` (6h max), `2snTTT` (6h min), `4snTTTsnTTT` (24h max/min)
- 5-min auto obs have whole-degree C precision; hourly METARs have 0.1C via T-group

### 3. NWS HTML page (tgftp.nws.noaa.gov)
- `fetch_nws_observation()` / `parse_6hr_section()` in weather.py
- Shows 6h and 24h max/min in a table
- Must use regex on raw HTML — BeautifulSoup mis-parses the nested tables (all TDs end up under body)

## Temperature Precision Notes
- **Hourly METAR** (:53 past hour): T-group gives 0.1C precision. API uses this value.
- **5-minute auto obs**: whole-degree C only. A reading of 21C could be 20.5-21.4C = 69-71F.
- **6h max/min** (METAR `1snTTT`/`2snTTT`): continuously tracked by ASOS sensor to 0.1C. Most accurate.
- **CLI report**: rounds to whole F. Use the 6h max (0.1C) to predict this before CLI publishes.

## Betting Strategy — Daily High (KXHIGHLAX)
1. Wait for afternoon 6h METAR report (~3:53 PM PST / 23:53 UTC)
2. Read the 6h max — this is the probable daily high to 0.1C
3. Fetch NWS hourly forecast → check remaining hours are all below observed max
4. Confidence gate: HIGH (>3°F margin) / MEDIUM (<3°F margin) / LOW (forecast ≥ observed max)
5. Convert the 6h max to F and round to nearest whole degree — this predicts the CLI value
6. Find the Kalshi bracket containing that temp
7. Buy YES before the market fully prices it in

## Betting Strategy — Daily Low (KXLOWLAX)
1. Wait for morning 6h METAR report (~9:53 AM PST / 17:53 UTC) — captures overnight low
2. Read the 6h min — this is the probable daily low to 0.1C
3. Fetch NWS hourly forecast → check remaining hours (including evening) are all above observed min
4. Confidence gate: HIGH (>3°F margin) / MEDIUM (<3°F margin) / LOW (forecast ≤ observed min)
5. Convert the 6h min to F and round to nearest whole degree — this predicts the CLI value
6. Find the Kalshi bracket containing that temp
7. Buy YES before the market fully prices it in

### Low Temp Timing Notes
- Daily low typically occurs around sunrise (~6-7 AM PST for LA)
- The 9:53 AM 6h METAR covers 4-10 AM, usually capturing the overnight minimum
- Risk: evening/night cold front could push temps below the morning low
- The forecast evening hours are critical — if they show temps near the observed min, confidence is LOW
- Best to bet in the afternoon when daytime warming confirms the low is well above the morning min

## Betting Strategy — Peak Track (--strategy peak-track)
Real-time strategy that doesn't wait for the 6h METAR report. Can bet earlier in the day.

1. Fetch NWS hourly forecast → find peak temp and when it's expected
2. Fetch live Synoptic observations (5-min resolution, last 6h)
3. Find today's observed max from live data
4. Check if the **last 3 readings** are:
   - **Monotonically decreasing** (each lower than the previous)
   - **All below the forecast peak**
5. If both conditions met → peak has passed → status = **LOCKED** → bet on observed max
6. If below peak but not declining → **NEAR_PEAK** → wait (or --force)
7. If still at/above peak → **TOO_EARLY** → wait (or --force)

### Advantages over metar6h
- Can trigger earlier in the day (doesn't need to wait for 3:53 PM report)
- Uses 5-minute obs resolution instead of 6-hour windows
- Directly tracks the temperature curve in real-time

### Limitations
- Uses forecast peak as reference (forecasts can be wrong by 2-3°F)
- Observed max from 5-min obs is whole-degree C (~2°F uncertainty)
- The 6h METAR max (0.1°C precision) is more accurate for the bet target

## Betting Strategy — Momentum (--strategy momentum)
Rate-of-change strategy. Computes °F/hour over a 30-minute sliding window of live
Synoptic observations, grades confidence based on how fast temps are dropping.

1. Fetch NWS hourly forecast → find peak temp
2. Fetch live Synoptic observations (5-min resolution, last 6h)
3. Compute rate of change over last 30 min: `(last_temp - first_temp) / hours`
4. Require current temp below forecast peak, then grade by rate:
   - **LOCKED** — rate ≤ -2°F/hr (rapid cooling, peak definitely passed) → auto-bet
   - **LIKELY** — rate ≤ -1°F/hr (steady decline) → exit unless --force
   - **POSSIBLE** — rate ≤ -0.5°F/hr (slow decline, could plateau) → exit unless --force
   - **TOO_EARLY** — rate > -0.5°F/hr (flat or rising) → exit unless --force
5. Bets on today's observed max from live data

### Momentum vs Peak-Track
- Peak-track checks discrete direction (are last 3 ticks decreasing?)
- Momentum measures actual speed of decline — more nuanced
- Momentum can distinguish "fast crash" from "slow drift down" from "plateau"
- Both use forecast peak as reference and observed max as bet target

### Risks (all strategies)
- Late-day temperature spike (Santa Ana winds) — check forecast before betting on high
- Evening cold front — check forecast before betting on low
- The final CLI covers midnight-to-midnight, so any temp after the report still counts
- Rounding edge case: 69.5F could round either way depending on NWS method
- Thin liquidity on some brackets
- 5-min obs are whole-degree C (~2°F uncertainty) — less precise than 6h METAR max

### CLI Usage
```
python bot.py --market high --dry-run                  # metar6h high (default)
python bot.py --market low --dry-run                   # metar6h low
python bot.py --strategy peak-track --dry-run          # peak-track high
python bot.py --strategy peak-track --force --dry-run  # override if not yet LOCKED
python bot.py --strategy momentum --dry-run            # momentum high
python bot.py --strategy momentum --force --dry-run    # override if not yet LOCKED
```

## Kalshi API
- Demo: `https://demo-api.kalshi.co`, Prod: `https://api.elections.kalshi.com`
- Auth: RSA-PSS signature of `timestamp_ms + METHOD + path` (no query params)
- Headers: `KALSHI-ACCESS-KEY`, `KALSHI-ACCESS-SIGNATURE`, `KALSHI-ACCESS-TIMESTAMP`
- Salt length: use `PSS.MAX_LENGTH` (not `DIGEST_LENGTH` — unavailable in cryptography 3.x)

## Python Notes
- Target: Python 3.9 — use `from __future__ import annotations` and `typing.Optional`
