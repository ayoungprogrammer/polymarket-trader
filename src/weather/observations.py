"""Weather data ingestion from NWS and Synoptic Data APIs."""

from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Dict, Optional, Tuple
from zoneinfo import ZoneInfo

import pandas as pd
import requests

log = logging.getLogger(__name__)

NWS_API_BASE = "https://api.weather.gov"
NWS_CURRENT_URL = "https://tgftp.nws.noaa.gov/weather/current/{site}.html"
NWS_CLI_URL = (
    "https://forecast.weather.gov/product.php"
    "?site=NWS&issuedby={suffix}&product=CLI&format=CI&version={version}&glossary=0"
)
SYNOPTIC_API_BASE = "https://api.synopticdata.com/v2"
SYNOPTIC_TOKEN = "7c76618b66c74aee913bdbae4b448bdd"  # NWS public token


class WeatherIngestion:
    """Fetch live weather observations from api.weather.gov.

    Parameters
    ----------
    site : str
        ICAO station identifier, e.g. "KLAX", "KJFK".
    """

    def __init__(self, site: str = "KLAX"):
        self.site = site.upper()
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "(polymarket-trader, contact@example.com)",
            "Accept": "application/geo+json",
        })

    def fetch_live_weather(
        self,
        start: Optional[str] = None,
        end: Optional[str] = None,
        limit: int = 500,
    ) -> pd.DataFrame:
        """Fetch recent observations and return as a DataFrame.

        Parameters
        ----------
        start : str, optional
            ISO 8601 start time, e.g. "2026-02-07T00:00:00Z".
        end : str, optional
            ISO 8601 end time.
        limit : int
            Max number of observations to return (API max ~500).

        Returns
        -------
        pd.DataFrame
            One row per observation with columns for timestamp, temperature,
            dewpoint, humidity, wind, pressure, visibility, etc.
        """
        url = f"{NWS_API_BASE}/stations/{self.site}/observations"
        params = {"limit": limit}
        if start:
            params["start"] = start
        if end:
            params["end"] = end

        log.info(f"Fetching observations for {self.site} (limit={limit})")
        print(url)
        resp = self.session.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        features = data.get("features", [])
        if not features:
            log.warning(f"No observations returned for {self.site}")
            return pd.DataFrame()

        rows = []
        for feat in features:
            props = feat.get("properties", {})
            rows.append(_extract_observation(props))

        df = pd.DataFrame(rows)
        df = df.sort_values("timestamp").reset_index(drop=True)
        log.info(f"Fetched {len(df)} observations for {self.site}")
        return df


def _extract_observation(props: dict) -> dict:
    """Flatten one observation's properties into a dict of scalar values."""

    def val(field: dict) -> Optional[float]:
        if field is None:
            return None
        return field.get("value")

    temp_c = val(props.get("temperature"))
    dewpoint_c = val(props.get("dewpoint"))
    wind_kmh = val(props.get("windSpeed"))
    gust_kmh = val(props.get("windGust"))
    pressure_pa = val(props.get("barometricPressure"))
    visibility_m = val(props.get("visibility"))

    row = {
        "timestamp": props.get("timestamp"),
        "text_description": props.get("textDescription", ""),
        "temperature_c": temp_c,
        "temperature_f": _c_to_f(temp_c),
        "dewpoint_c": dewpoint_c,
        "dewpoint_f": _c_to_f(dewpoint_c),
        "relative_humidity_pct": val(props.get("relativeHumidity")),
        "wind_direction_deg": val(props.get("windDirection")),
        "wind_speed_kmh": wind_kmh,
        "wind_speed_mph": _kmh_to_mph(wind_kmh),
        "wind_gust_kmh": gust_kmh,
        "wind_gust_mph": _kmh_to_mph(gust_kmh),
        "barometric_pressure_pa": pressure_pa,
        "barometric_pressure_inhg": _pa_to_inhg(pressure_pa),
        "visibility_m": visibility_m,
        "visibility_mi": round(visibility_m / 1609.344, 2) if visibility_m is not None else None,
        "cloud_layers": _format_clouds(props.get("cloudLayers", [])),
    }

    # Decode 6h/24h temps from METAR remarks
    raw_metar = props.get("rawMessage", "")
    if raw_metar:
        row.update(decode_metar_remarks(raw_metar))

    return row


def _c_to_f(c: Optional[float]) -> Optional[float]:
    return round(c * 9 / 5 + 32, 1) if c is not None else None


def _kmh_to_mph(kmh: Optional[float]) -> Optional[float]:
    return round(kmh / 1.60934, 1) if kmh is not None else None


def _pa_to_inhg(pa: Optional[float]) -> Optional[float]:
    return round(pa / 3386.389, 2) if pa is not None else None


def _format_clouds(layers: list) -> str:
    if not layers:
        return ""
    parts = []
    for layer in layers:
        base = layer.get("base", {})
        amount = layer.get("amount", "")
        height_m = base.get("value") if base else None
        height_ft = round(height_m * 3.28084) if height_m is not None else "?"
        parts.append(f"{amount} {height_ft}ft")
    return ", ".join(parts)


class SynopticIngestion:
    """Fetch live weather observations from the Synoptic Data (MesoWest) API.

    This is the same data source used by the NWS timeseries page
    (weather.gov/wrh/timeseries). It provides higher-precision temps
    (T-group tenths at :53 METARs) and dedicated 6h/24h max/min fields.

    Parameters
    ----------
    site : str
        ICAO station identifier, e.g. "KLAX", "KJFK".
    """

    def __init__(self, site: str = "KLAX"):
        self.site = site.upper()
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0",
            "Referer": f"https://www.weather.gov/wrh/timeseries?site={self.site}",
            "Origin": "https://www.weather.gov",
        })

    def fetch_live_weather(
        self,
        hours: int = 72,
    ) -> pd.DataFrame:
        """Fetch recent observations and return as a DataFrame.

        Parameters
        ----------
        hours : int
            Number of hours of recent data to fetch (max 720).

        Returns
        -------
        pd.DataFrame
            One row per observation with all available sensor fields.
            Temperatures are always in metric (°C source) with both _c
            and _f columns derived.
        """
        recent_minutes = hours * 60
        url = f"{SYNOPTIC_API_BASE}/stations/timeseries"
        params = {
            "STID": self.site,
            "showemptystations": "1",
            "recent": recent_minutes,
            "complete": "1",
            "units": "metric",
            "token": SYNOPTIC_TOKEN,
            "obtimezone": "local",
        }

        log.info(f"Fetching Synoptic observations for {self.site} (last {hours}h)")
        resp = self.session.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        summary = data.get("SUMMARY", {})
        if summary.get("RESPONSE_CODE") != 1:
            msg = summary.get("RESPONSE_MESSAGE", "Unknown error")
            raise RuntimeError(f"Synoptic API error: {msg}")

        stations = data.get("STATION", [])
        if not stations:
            log.warning(f"No station data returned for {self.site}")
            return pd.DataFrame()

        obs = stations[0].get("OBSERVATIONS", {})
        df = _synoptic_obs_to_df(obs)

        # Filter to current day only for short requests (<=24h)
        if not df.empty and hours <= 24:
            today = df["timestamp"].iloc[-1][:10]
            df = df[df["timestamp"].str[:10] == today].reset_index(drop=True)

        # Trim rows up to and including the 24h METAR if it occurs before 1 AM.
        # The 24h METAR reports yesterday's max — same trim as load_site_history.
        if not df.empty and "max_temp_24h_f" in df.columns:
            ts = pd.to_datetime(df["timestamp"].str[:19])
            early_24h = df[df["max_temp_24h_f"].notna() & (ts.dt.hour < 1)]
            if not early_24h.empty:
                first_pos = early_24h.index[0]
                df = df.loc[df.index > first_pos].reset_index(drop=True)

        return df


def _synoptic_obs_to_df(obs: dict) -> pd.DataFrame:
    """Convert Synoptic OBSERVATIONS dict to a DataFrame.

    Source data is metric (°C, km/h). Always produces both _c and _f
    columns for temperatures, and both _kmh and _mph for wind speed.
    """
    dates = obs.get("date_time", [])
    if not dates:
        return pd.DataFrame()

    # Map Synoptic field names to clean column names
    field_map = {
        "date_time": "timestamp",
        "air_temp_set_1": "temperature",
        "dew_point_temperature_set_1": "dewpoint",
        "relative_humidity_set_1": "relative_humidity_pct",
        "wind_speed_set_1": "wind_speed",
        "wind_direction_set_1": "wind_direction_deg",
        "wind_cardinal_direction_set_1d": "wind_cardinal",
        "altimeter_set_1": "altimeter",
        "sea_level_pressure_set_1": "sea_level_pressure",
        "visibility_set_1": "visibility",
        "cloud_layer_1_code_set_1": "cloud_layer_code",
        "cloud_layer_1_set_1d": "cloud_layer",
        "metar_set_1": "metar",
        "weather_summary_set_1d": "weather_summary",
        "air_temp_high_6_hour_set_1": "max_temp_6h",
        "air_temp_low_6_hour_set_1": "min_temp_6h",
        "air_temp_high_24_hour_set_1": "max_temp_24h",
        "air_temp_low_24_hour_set_1": "min_temp_24h",
        "pressure_tendency_set_1": "pressure_tendency",
    }

    df_data = {}
    for synoptic_key, col_name in field_map.items():
        if synoptic_key in obs:
            df_data[col_name] = obs[synoptic_key]

    df = pd.DataFrame(df_data)

    # Add both C and F columns for all temperature fields (source is metric)
    temp_cols = ["temperature", "dewpoint", "max_temp_6h", "min_temp_6h",
                 "max_temp_24h", "min_temp_24h"]
    for col in temp_cols:
        if col not in df.columns:
            continue
        numeric = pd.to_numeric(df[col], errors="coerce")
        df[f"{col}_c"] = numeric
        df[f"{col}_f"] = (numeric * 9 / 5 + 32).round(2)
        df.drop(columns=[col], inplace=True)

    if "wind_speed" in df.columns:
        numeric = pd.to_numeric(df["wind_speed"], errors="coerce")
        df["wind_speed_kmh"] = numeric
        df["wind_speed_mph"] = (numeric / 1.60934).round(2)
        df.drop(columns=["wind_speed"], inplace=True)

    df = df.sort_values("timestamp").reset_index(drop=True)
    log.info(f"Fetched {len(df)} observations from Synoptic API")
    return df


# ---------------------------------------------------------------------------
# METAR remarks decoder for 6h / 24h temperature groups
# ---------------------------------------------------------------------------

def _decode_metar_temp(group: str) -> Optional[float]:
    """Decode a METAR snTTT temperature group to degrees C.

    Format: 1 sign digit + 3 digits = tenths of °C.
    Sign: 0 = positive, 1 = negative.
    Example: "0206" -> +20.6°C, "1003" -> -0.3°C
    """
    if len(group) != 4 or not group.isdigit():
        return None
    sign = -1 if group[0] == "1" else 1
    return sign * int(group[1:]) / 10.0


def decode_metar_remarks(raw: str) -> dict:
    """Extract 6h/24h max/min temps from METAR RMK section.

    Remark groups (appear after "RMK" in the METAR string):
      1snTTT  — 6-hour maximum temperature
      2snTTT  — 6-hour minimum temperature
      4snTTTsnTTT — 24-hour max and min temperature

    Returns dict with keys (value in °C, or None if not present):
      max_temp_6h_c, min_temp_6h_c, max_temp_24h_c, min_temp_24h_c
      max_temp_6h_f, min_temp_6h_f, max_temp_24h_f, min_temp_24h_f
    """
    result = {
        "max_temp_6h_c": None, "max_temp_6h_f": None,
        "min_temp_6h_c": None, "min_temp_6h_f": None,
        "max_temp_24h_c": None, "max_temp_24h_f": None,
        "min_temp_24h_c": None, "min_temp_24h_f": None,
    }

    if "RMK" not in raw:
        return result
    rmk = raw.split("RMK", 1)[1]

    # 6-hour max: group starting with 1, followed by sign+3 digits
    m = re.search(r"\b1([01]\d{3})\b", rmk)
    if m:
        val = _decode_metar_temp(m.group(1))
        if val is not None:
            result["max_temp_6h_c"] = val
            result["max_temp_6h_f"] = _c_to_f(val)

    # 6-hour min: group starting with 2
    m = re.search(r"\b2([01]\d{3})\b", rmk)
    if m:
        val = _decode_metar_temp(m.group(1))
        if val is not None:
            result["min_temp_6h_c"] = val
            result["min_temp_6h_f"] = _c_to_f(val)

    # 24-hour max/min: group starting with 4, 8 digits after
    m = re.search(r"\b4([01]\d{3})([01]\d{3})\b", rmk)
    if m:
        max_val = _decode_metar_temp(m.group(1))
        min_val = _decode_metar_temp(m.group(2))
        if max_val is not None:
            result["max_temp_24h_c"] = max_val
            result["max_temp_24h_f"] = _c_to_f(max_val)
        if min_val is not None:
            result["min_temp_24h_c"] = min_val
            result["min_temp_24h_f"] = _c_to_f(min_val)

    return result


# ---------------------------------------------------------------------------
# NWS current-observation HTML parsing
# ---------------------------------------------------------------------------

_TZ_ABBREVS = {
    "EST": "-0500", "EDT": "-0400", "CST": "-0600", "CDT": "-0500",
    "MST": "-0700", "MDT": "-0600", "PST": "-0800", "PDT": "-0700",
}


def fetch_nws_observation(site: str = "KLAX") -> str:
    """Download the NWS current-conditions HTML page for *site*."""
    url = NWS_CURRENT_URL.format(site=site.upper())
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    return resp.text


def parse_6hr_section(html: str) -> Tuple[datetime, float, float]:
    """Return (timestamp_utc, max_temp_f, min_temp_f) from the 6-hour section.

    The NWS page has a <TR> with three <TD> cells:
      <TD ALIGN=CENTER>69.1 (20.6)</TD>   -- max temp F (C)
      <TD ALIGN=CENTER>64.0 (17.8)</TD>   -- min temp F (C)
      <TD>In the <B>6 hours</B> preceding Feb 07, 2026 - 06:53 PM EST ...</TD>

    We use regex on the raw HTML since BeautifulSoup mis-parses the nested tables.
    """
    match = re.search(
        r"<TR>\s*"
        r"<TD[^>]*>.*?([\d.]+)\s*\([\d.]+\).*?</TD>\s*"
        r"<TD[^>]*>.*?([\d.]+)\s*\([\d.]+\).*?</TD>\s*"
        r"<TD[^>]*>.*?6 hours.*?preceding\s+"
        r"(\w+ \d{1,2}, \d{4} - \d{1,2}:\d{2} [AP]M \w+)"
        r".*?</TD>\s*</TR>",
        html,
        re.DOTALL | re.IGNORECASE,
    )
    if not match:
        raise ValueError("Could not parse 6-hour section from NWS page")

    max_temp = float(match.group(1))
    min_temp = float(match.group(2))
    ts_str = match.group(3)

    ts_clean = ts_str.replace(" - ", " ")
    for abbr, offset in _TZ_ABBREVS.items():
        if abbr in ts_clean:
            ts_clean = ts_clean.replace(abbr, offset)
            break

    ts = datetime.strptime(ts_clean, "%b %d, %Y %I:%M %p %z")
    return ts, max_temp, min_temp


def is_past_3pm_pacific(ts: datetime) -> bool:
    """Check if the timestamp is at or past 3:00 PM Pacific."""
    pacific = ZoneInfo("America/Los_Angeles")
    ts_pacific = ts.astimezone(pacific)
    return ts_pacific.hour >= 15

# west coast PT
# report drops at  3:53, 9:53
# KSEA
# KSFO
# KLOX


# East cost
# reprot drops at 12:54, 6:54
# KPHL

# KNYC -> diff scheduoe
# https://www.weather.gov/wrh/timeseries?site=KNYC


# ---------------------------------------------------------------------------
# NWS CLI (Climatological Report) — preliminary + final daily summary
# ---------------------------------------------------------------------------

# CLI issuedby suffix for each ICAO station (strip leading K)
_CLI_SUFFIX = {
    "KLAX": "LAX", "KMIA": "MIA", "KSFO": "SFO", "KORD": "ORD",
    "KDEN": "DEN", "KPHX": "PHX", "KOKC": "OKC", "KATL": "ATL",
    "KDFW": "DFW", "KSAT": "SAT", "KHOU": "HOU", "KMSP": "MSP",
}


def fetch_cli(site: str, version: int = 1) -> Optional[str]:
    """Fetch a CLI product for *site*.

    Parameters
    ----------
    site : str
        ICAO station identifier (e.g. "KHOU").
    version : int
        1 = most recent (usually final, morning-after),
        2 = previous (usually preliminary same-day ~4-7 PM local).
        Higher versions go further back in history.

    Returns
    -------
    The raw CLI text, or None if not found.
    """
    suffix = _CLI_SUFFIX.get(site.upper(), site.upper().lstrip("K"))
    url = NWS_CLI_URL.format(suffix=suffix, version=version)
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
    except Exception as e:
        log.warning(f"CLI fetch failed for {site} v{version}: {e}")
        return None

    m = re.search(r"<pre[^>]*>(.*?)</pre>", resp.text, re.DOTALL)
    if not m:
        return None
    return m.group(1)


def parse_cli(text: str) -> Optional[Dict]:
    """Parse a CLI product text into structured data.

    Returns
    -------
    Dict with keys:
        site_name : str — e.g. "HOUSTON/HOBBY AIRPORT"
        date : str — e.g. "MARCH 27 2026"
        issued : str — e.g. "424 PM CDT FRI MAR 27 2026"
        is_preliminary : bool — True if "VALID TODAY" / "VALID AS OF"
        max_temp_f : int or None
        max_temp_time : str or None — e.g. "2:59 PM"
        min_temp_f : int or None
        min_temp_time : str or None
    Or None if parsing fails.
    """
    if not text:
        return None

    result: Dict = {
        "site_name": None,
        "date": None,
        "issued": None,
        "is_preliminary": False,
        "max_temp_f": None,
        "max_temp_time": None,
        "min_temp_f": None,
        "min_temp_time": None,
    }

    # Preliminary flag
    result["is_preliminary"] = "VALID TODAY" in text or "VALID AS OF" in text

    # Site name + date from "...THE <SITE> CLIMATE SUMMARY FOR <DATE>..."
    m = re.search(
        r"\.\.\.THE\s+(.+?)\s+CLIMATE SUMMARY FOR\s+(.+?)\.\.\.",
        text,
    )
    if m:
        result["site_name"] = m.group(1).strip()
        result["date"] = m.group(2).strip()

    # Issuance time — line with AM/PM and a year
    for line in text.split("\n"):
        line = line.strip()
        if re.search(r"\d{1,2}\s+[AP]M\s+\w+\s+\w+\s+\w+\s+\d+\s+\d{4}", line):
            result["issued"] = line
            break

    # Temperature section — MAXIMUM and MINIMUM lines
    # Format:  "  MAXIMUM         86   2:59 PM  88    1935  76     10       73"
    # or:      "  MAXIMUM         54    359 PM  78    1988  59     -5       76"
    # or:      "  MAXIMUM         80R  2:52 PM  ..." (R = record set/tied)
    # or:      "  MAXIMUM         39        MM  ..." (MM = missing time)
    m = re.search(
        r"MAXIMUM\s+(-?\d+)R?\s+([\d:]+\s*[AP]M|MM)",
        text,
    )
    if m:
        result["max_temp_f"] = int(m.group(1))
        time_str = m.group(2).strip()
        if time_str != "MM":
            result["max_temp_time"] = time_str

    m = re.search(
        r"MINIMUM\s+(-?\d+)R?\s+([\d:]+\s*[AP]M|MM)",
        text,
    )
    if m:
        result["min_temp_f"] = int(m.group(1))
        time_str = m.group(2).strip()
        if time_str != "MM":
            result["min_temp_time"] = time_str

    return result


def fetch_all_cli_today(
    site: str,
    max_versions: int = 5,
) -> list:
    """Fetch all CLI reports for today's date.

    Scans versions 1..max_versions and collects every report whose
    date matches the most recent report's date (i.e. today).

    Returns a list of parsed dicts, ordered most-recent-first.
    Each dict has keys: site_name, date, issued, is_preliminary,
    max_temp_f, max_temp_time, min_temp_f, min_temp_time.
    """
    reports = []
    target_date = None
    for v in range(1, max_versions + 1):
        text = fetch_cli(site, version=v)
        if not text:
            continue
        parsed = parse_cli(text)
        if not parsed or parsed.get("max_temp_f") is None:
            continue
        if target_date is None:
            target_date = parsed.get("date")
        if parsed.get("date") != target_date:
            break  # older day — stop
        parsed["version"] = v
        reports.append(parsed)

    if reports:
        log.info(f"[{site}] Found {len(reports)} CLI report(s) for {target_date}: "
                 + ", ".join(f"v{r['version']}={r['max_temp_f']}F"
                             + (" (prelim)" if r["is_preliminary"] else "")
                             for r in reports))
    return reports


# ---------------------------------------------------------------------------
# DSM (ASOS Daily Summary Message) via IEM AFOS archive
# ---------------------------------------------------------------------------

IEM_AFOS_LIST_URL = "https://mesonet.agron.iastate.edu/api/1/nws/afos/list.json"


def _parse_dsm_text(text: str) -> Optional[Dict]:
    """Parse a raw DSM coded message.

    Format example:
        KATL DS 1500 28/03 701441/ 490743// 70/ 49//...

    Fields (slash-delimited):
        station DS time dd/mm maxHHMM/ minHHMM// max/ min//...
    where max/min are °F, HHMM is local time of occurrence.
    'M' means missing.
    """
    if not text:
        return None

    # Find the DS line — starts with K followed by station ID
    ds_line = None
    for line in text.strip().split("\n"):
        line = line.strip()
        if " DS " in line and line.startswith("K"):
            ds_line = line
            break
    if not ds_line:
        return None

    # Parse: KXXX DS HHMM dd/mm {max}{HHMM}/ {min}{HHMM}// ...
    # Temp is 2-3 digits, time is always 4 digits (HHMM local).
    # M = missing for either field.
    m = re.match(
        r"(K\w+)\s+DS\s+(?:COR\s+)?(\d{4}|\w+)\s+(\d{1,2}/\d{2})\s+"
        r"(-?\d{2,3}|M)(\d{4})?/\s*(-?\d{2,3}|M)(\d{4})?/",
        ds_line,
    )
    if not m:
        return None

    station = m.group(1)
    obs_time = m.group(2)  # UTC time of report e.g. "1500"
    date_str = m.group(3)  # dd/mm

    max_raw = m.group(4)
    max_time_raw = m.group(5)  # HHMM local or None
    min_raw = m.group(6)
    min_time_raw = m.group(7)

    result: Dict = {
        "station": station,
        "obs_time_utc": obs_time,
        "date": date_str,
        "max_temp_f": int(max_raw) if max_raw != "M" else None,
        "max_temp_time": None,
        "min_temp_f": int(min_raw) if min_raw != "M" else None,
        "min_temp_time": None,
    }

    if max_time_raw and len(max_time_raw) == 4:
        result["max_temp_time"] = f"{int(max_time_raw[:2])}:{max_time_raw[2:]}"
    if min_time_raw and len(min_time_raw) == 4:
        result["min_temp_time"] = f"{int(min_time_raw[:2])}:{min_time_raw[2:]}"

    return result


def fetch_dsm_today(site: str, station_tz: Optional[str] = None) -> list:
    """Fetch all DSM reports for today from IEM AFOS archive.

    Parameters
    ----------
    site : str
        ICAO station identifier (e.g. "KHOU").
    station_tz : str, optional
        IANA timezone (e.g. "America/Chicago").  Used to determine
        today's date for filtering.  Falls back to UTC.

    Returns
    -------
    List of parsed DSM dicts, ordered oldest-first.
    Each dict has: station, obs_time_utc, date, max_temp_f,
    max_temp_time, min_temp_f, min_temp_time.
    """
    suffix = site.upper().lstrip("K")
    pil = f"DSM{suffix}"

    if station_tz:
        _tz = ZoneInfo(station_tz)
        _now = datetime.now(_tz)
    else:
        _now = datetime.utcnow()
    today_str = _now.strftime("%Y-%m-%d")
    # DSM date field is dd/mm — compute expected value for today
    today_ddmm = _now.strftime("%d/%m")

    try:
        resp = requests.get(IEM_AFOS_LIST_URL, params={
            "pil": pil,
            "date": today_str,
        }, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        log.warning(f"[{site}] IEM DSM list failed: {e}")
        return []

    rows = data.get("data", [])
    if not rows:
        return []

    reports = []
    for row in rows:
        text_url = row.get("text_link")
        if not text_url:
            continue
        try:
            r = requests.get(text_url, timeout=10)
            r.raise_for_status()
            parsed = _parse_dsm_text(r.text)
            if parsed and parsed["max_temp_f"] is not None:
                # Filter: DSM date must match today (station local)
                if parsed.get("date") != today_ddmm:
                    log.debug(f"[{site}] DSM date mismatch: {parsed.get('date')} vs {today_ddmm}")
                    continue
                parsed["entered"] = row.get("entered")
                reports.append(parsed)
        except Exception as e:
            log.debug(f"[{site}] DSM text fetch failed: {e}")

    if reports:
        log.info(f"[{site}] Found {len(reports)} DSM report(s): "
                 + ", ".join(f"{r['obs_time_utc']}={r['max_temp_f']}F" for r in reports))
    return reports


# ---------------------------------------------------------------------------
# Solar noon (sunrisesunset.io API)
# ---------------------------------------------------------------------------

def fetch_solar_noon(lat: float, lon: float, date_str: str) -> Optional[float]:
    """Fetch solar noon for a single date and location.

    Parameters
    ----------
    lat, lon : float
        Station coordinates.
    date_str : str
        Date as "YYYY-MM-DD".

    Returns
    -------
    Decimal hours (e.g. 12.25 = 12:15 PM), or None on failure.
    """
    try:
        resp = requests.get("https://api.sunrisesunset.io/json", params={
            "lat": lat,
            "lng": lon,
            "date": date_str,
            "time_format": "24",
        }, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        log.debug(f"Solar noon API error for {date_str}: {e}")
        return None

    if data.get("status") != "OK":
        return None

    results = data.get("results", {})
    if isinstance(results, list):
        results = results[0] if results else {}
    noon_str = results.get("solar_noon", "")
    if not noon_str:
        return None

    parts = noon_str.split(":")
    if len(parts) < 2:
        return None
    hour = int(parts[0]) + int(parts[1]) / 60.0
    if len(parts) >= 3:
        hour += int(parts[2]) / 3600.0
    return round(hour, 4)


def fetch_sun_times(lat: float, lon: float, date_str: str) -> Optional[Dict[str, float]]:
    """Fetch sunrise, sunset, and solar noon for a single date/location.

    Returns dict with decimal hours: {"sunrise": 6.5, "solar_noon": 12.25, "sunset": 18.1}
    or None on failure.
    """
    try:
        resp = requests.get("https://api.sunrisesunset.io/json", params={
            "lat": lat,
            "lng": lon,
            "date": date_str,
            "time_format": "24",
        }, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        log.debug(f"Sun times API error for {date_str}: {e}")
        return None

    if data.get("status") != "OK":
        return None

    results = data.get("results", {})
    if isinstance(results, list):
        results = results[0] if results else {}

    def _parse_hms(s: str) -> Optional[float]:
        parts = s.split(":")
        if len(parts) < 2:
            return None
        h = int(parts[0]) + int(parts[1]) / 60.0
        if len(parts) >= 3:
            h += int(parts[2]) / 3600.0
        return round(h, 4)

    sunrise = _parse_hms(results.get("sunrise", ""))
    sunset = _parse_hms(results.get("sunset", ""))
    solar_noon = _parse_hms(results.get("solar_noon", ""))

    if solar_noon is None:
        return None

    out: Dict[str, float] = {"solar_noon": solar_noon}
    if sunrise is not None:
        out["sunrise"] = sunrise
    if sunset is not None:
        out["sunset"] = sunset
    return out


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Weather data ingestion")
    parser.add_argument("site", nargs="?", default="KLAX",
                        help="ICAO station identifier (default: KLAX)")
    parser.add_argument("--hours", type=int, default=24,
                        help="Hours of data to fetch (default: 24)")
    parser.add_argument("--csv", type=str, default="weather.csv",
                        help="Output CSV filename (default: weather.csv)")
    parser.add_argument("--cli", action="store_true",
                        help="Fetch and parse NWS CLI report instead of Synoptic data")
    parser.add_argument("--dsm", action="store_true",
                        help="Fetch DSM reports from IEM AFOS archive")
    args = parser.parse_args()

    if args.dsm:
        print(f"=== DSM (IEM AFOS) — {args.site} ===")
        reports = fetch_dsm_today(args.site)
        if reports:
            for r in reports:
                print(f"  [{r['obs_time_utc']} UTC] {r['station']}")
                print(f"    Max: {r['max_temp_f']}F at {r['max_temp_time']}")
                print(f"    Min: {r['min_temp_f']}F at {r['min_temp_time']}")
        else:
            print("  No DSM reports found")
        return

    if args.cli:
        print(f"=== NWS CLI — {args.site} ===")
        reports = fetch_all_cli_today(args.site)
        if reports:
            for r in reports:
                tag = "PRELIMINARY" if r["is_preliminary"] else "FINAL"
                print(f"  [{tag}] v{r.get('version', '?')}")
                print(f"    Site:    {r['site_name']}")
                print(f"    Date:    {r['date']}")
                print(f"    Issued:  {r['issued']}")
                print(f"    Max:     {r['max_temp_f']}F at {r['max_temp_time']}")
                print(f"    Min:     {r['min_temp_f']}F at {r['min_temp_time']}")
        else:
            print("  No CLI reports found")
        return

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)

    print(f"=== Synoptic API — {args.site} (last {args.hours}h) ===")
    s = SynopticIngestion(site=args.site)
    df = s.fetch_live_weather(hours=args.hours)
    df.to_csv(args.csv, index=False)
    print(f"Wrote {len(df)} rows to {args.csv}")


if __name__ == '__main__':
    main()