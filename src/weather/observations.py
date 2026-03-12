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
        units: str = "metric",
    ) -> pd.DataFrame:
        """Fetch recent observations and return as a DataFrame.

        Parameters
        ----------
        hours : int
            Number of hours of recent data to fetch (max 720).
        units : str
            "english" for F/mph, "metric" for C/km/h.

        Returns
        -------
        pd.DataFrame
            One row per observation with all available sensor fields.
        """
        recent_minutes = hours * 60
        unit_str = "temp|F,speed|mph,english" if units == "english" else ""
        url = f"{SYNOPTIC_API_BASE}/stations/timeseries"
        params = {
            "STID": self.site,
            "showemptystations": "1",
            "units": unit_str,
            "recent": recent_minutes,
            "complete": "1",
            "token": SYNOPTIC_TOKEN,
            "obtimezone": "local",
        }
        if not unit_str:
            del params["units"]

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
        df = _synoptic_obs_to_df(obs, units)

        # Filter to current day using the local date in the timestamps
        if not df.empty:
            today = df["timestamp"].iloc[-1][:10]
            df = df[df["timestamp"].str[:10] == today].reset_index(drop=True)

        return df


def _synoptic_obs_to_df(obs: dict, units: str) -> pd.DataFrame:
    """Convert Synoptic OBSERVATIONS dict to a DataFrame.

    Always produces both _c and _f columns for temperatures regardless
    of the source unit system.
    """
    dates = obs.get("date_time", [])
    if not dates:
        return pd.DataFrame()

    is_english = units == "english"

    # Map Synoptic field names to clean column names
    field_map = {
        "date_time": "timestamp",
        "air_temp_set_1": "temperature",
        "dew_point_temperature_set_1": "dewpoint",
        "relative_humidity_set_1": "relative_humidity_pct",
        "wind_speed_set_1": "wind_speed",
        "wind_direction_set_1": "wind_direction_deg",
        "wind_cardinal_direction_set_1d": "wind_cardinal",
        "altimeter_set_1": "altimeter_inhg" if is_english else "altimeter",
        "sea_level_pressure_set_1": "sea_level_pressure",
        "visibility_set_1": "visibility_mi" if is_english else "visibility",
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

    # Add both C and F columns for all temperature fields
    temp_cols = ["temperature", "dewpoint", "max_temp_6h", "min_temp_6h",
                 "max_temp_24h", "min_temp_24h"]
    for col in temp_cols:
        if col not in df.columns:
            continue
        numeric = pd.to_numeric(df[col], errors="coerce")
        if is_english:
            df[f"{col}_f"] = numeric
            df[f"{col}_c"] = ((numeric - 32) * 5 / 9).round(2)
        else:
            df[f"{col}_c"] = numeric
            df[f"{col}_f"] = (numeric * 9 / 5 + 32).round(2)
        df.drop(columns=[col], inplace=True)

    if "wind_speed" in df.columns:
        numeric = pd.to_numeric(df["wind_speed"], errors="coerce")
        if is_english:
            df["wind_speed_mph"] = numeric
            df["wind_speed_kmh"] = (numeric * 1.60934).round(2)
        else:
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
    args = parser.parse_args()

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)

    print(f"=== Synoptic API — {args.site} (last {args.hours}h) ===")
    s = SynopticIngestion(site=args.site)
    df = s.fetch_live_weather(hours=args.hours)
    df.to_csv(args.csv, index=False)
    print(f"Wrote {len(df)} rows to {args.csv}")


if __name__ == '__main__':
    main()