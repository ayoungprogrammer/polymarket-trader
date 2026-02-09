"""NWS hourly forecast ingestion."""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

import pandas as pd
import requests

log = logging.getLogger(__name__)

# (lat, lon, timezone)
STATIONS = {
    "KLAX": (33.9381, -118.3889, "America/Los_Angeles"),
    "KSFO": (37.6197, -122.3647, "America/Los_Angeles"),
    "KJFK": (40.6413, -73.7781, "America/New_York"),
    "KPHL": (39.8721, -75.2411, "America/New_York"),
    "KSEA": (47.4502, -122.3088, "America/Los_Angeles"),
    "KSHN": (47.2336, -123.1464, "America/Los_Angeles"),
}

FORECAST_URL = "https://forecast.weather.gov/MapClick.php"
NWS_API_BASE = "https://api.weather.gov"


class ForecastIngestion:
    """Fetch NWS hourly digital forecast for a station.

    Parameters
    ----------
    site : str
        ICAO station identifier, e.g. "KLAX". Must be in STATIONS dict,
        or provide lat/lon directly.
    lat, lon : float, optional
        Override coordinates (used if site not in STATIONS).
    """

    def __init__(
        self,
        site: str = "KSFO",
        lat: Optional[float] = None,
        lon: Optional[float] = None,
        tz: Optional[str] = None,
    ):
        self.site = site.upper()
        if lat is not None and lon is not None:
            self.lat = lat
            self.lon = lon
            self.tz = tz or "America/Los_Angeles"
        elif self.site in STATIONS:
            self.lat, self.lon, self.tz = STATIONS[self.site]
        else:
            raise ValueError(
                f"Unknown site {self.site}. Provide lat/lon or add to STATIONS dict."
            )

    def fetch_forecast(self) -> pd.DataFrame:
        """Fetch the hourly digital forecast and return as a DataFrame.

        Tries the NWS gridpoint API first (returns full day including past
        hours), falls back to the DWML XML endpoint if that fails.

        Returns a DataFrame with one row per hour, filtered to the current day.
        """
        try:
            df = self._fetch_gridpoint_forecast()
            if not df.empty:
                return df
        except Exception as e:
            log.warning(f"Gridpoint forecast failed, falling back to DWML: {e}")

        return self._fetch_dwml_forecast()

    def _fetch_gridpoint_forecast(self) -> pd.DataFrame:
        """Fetch from the NWS gridpoint API — includes past hours of today."""
        session = requests.Session()
        session.headers.update({
            "User-Agent": "(polymarket-trader, contact@example.com)",
            "Accept": "application/geo+json",
        })

        # Step 1: Resolve lat/lon to grid office/x/y
        points_url = f"{NWS_API_BASE}/points/{self.lat},{self.lon}"
        log.info(f"Resolving grid point for {self.site} ({self.lat}, {self.lon})")
        resp = session.get(points_url, timeout=15)
        resp.raise_for_status()
        grid_url = resp.json()["properties"]["forecastGridData"]

        # Step 2: Fetch raw gridpoint data
        log.info(f"Fetching gridpoint forecast: {grid_url}")
        resp = session.get(grid_url, timeout=15)
        resp.raise_for_status()
        props = resp.json()["properties"]

        # Step 3: Parse temperature series (in °C, with ISO 8601 duration intervals)
        df = _parse_gridpoint_series(props.get("temperature", {}), self.tz)

        # Filter to current day in local timezone
        if not df.empty:
            today = datetime.now(ZoneInfo(self.tz)).strftime("%Y-%m-%d")
            df = df[df["timestamp"].str[:10] == today].reset_index(drop=True)
            log.info(f"Gridpoint forecast: {len(df)} hourly rows for {today}")

        return df

    def _fetch_dwml_forecast(self) -> pd.DataFrame:
        """Fallback: fetch from the DWML XML endpoint (future hours only)."""
        url = FORECAST_URL
        params = {
            "lat": self.lat,
            "lon": self.lon,
            "FcstType": "digitalDWML",
        }

        log.info(f"Fetching DWML forecast for {self.site} ({self.lat}, {self.lon})")
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()

        df = _parse_dwml(resp.text)

        # Filter to current day using the local date in timestamps
        if not df.empty:
            today = datetime.now(ZoneInfo(self.tz)).strftime("%Y-%m-%d")
            df = df[df["timestamp"].str[:10] == today].reset_index(drop=True)
            log.info(f"DWML forecast: {len(df)} hourly rows for {today}")
        return df


def _parse_iso_duration(duration: str) -> int:
    """Parse ISO 8601 duration like 'PT1H', 'PT2H', 'PT4H' to hours."""
    m = re.match(r"PT(\d+)H", duration)
    return int(m.group(1)) if m else 1


def _parse_gridpoint_series(temp_data: dict, tz: str) -> pd.DataFrame:
    """Parse a gridpoint temperature series into hourly rows.

    The gridpoint API returns values with ISO 8601 time intervals like:
      {"validTime": "2026-02-08T15:00:00+00:00/PT1H", "value": 12.78}

    Multi-hour intervals (PT2H, PT4H) are expanded into individual hourly rows.
    Temperatures are converted from °C to °F.
    """
    values = temp_data.get("values", [])
    if not values:
        return pd.DataFrame()

    local_tz = ZoneInfo(tz)
    rows = []
    for entry in values:
        valid_time = entry["validTime"]
        temp_c = entry["value"]
        if temp_c is None:
            continue

        # Split "2026-02-08T15:00:00+00:00/PT1H" into timestamp and duration
        ts_str, duration = valid_time.split("/")
        start = datetime.fromisoformat(ts_str)
        hours = _parse_iso_duration(duration)
        temp_f = round(temp_c * 9 / 5 + 32, 1)

        # Expand multi-hour intervals into individual rows
        for h in range(hours):
            ts = start + timedelta(hours=h)
            local_ts = ts.astimezone(local_tz)
            rows.append({
                "timestamp": local_ts.isoformat(),
                "temperature_f": temp_f,
            })

    return pd.DataFrame(rows)


def _parse_dwml(xml_text: str) -> pd.DataFrame:
    """Parse NWS Digital Weather Markup Language XML into a DataFrame."""
    root = ET.fromstring(xml_text)
    data = root.find("data")
    if data is None:
        return pd.DataFrame()

    # Parse time layouts — map layout-key to list of start times
    time_layouts = {}
    for tl in data.findall("time-layout"):
        key = tl.findtext("layout-key", "")
        starts = [sv.text for sv in tl.findall("start-valid-time")]
        time_layouts[key] = starts

    # Weather parameters are inside <parameters>
    params = data.find("parameters")
    if params is None:
        return pd.DataFrame()

    # Parse each weather parameter
    columns = {}

    # Temperature
    for temp in params.findall("temperature"):
        layout = temp.get("time-layout", "")
        ttype = temp.get("type", "")
        col = "temperature_f" if ttype == "hourly" else f"temperature_{ttype.replace(' ', '_')}_f"
        values = [_to_num(v.text) for v in temp.findall("value")]
        columns[col] = (layout, values)

    # Wind speed
    for ws in params.findall("wind-speed"):
        layout = ws.get("time-layout", "")
        ttype = ws.get("type", "")
        col = "wind_speed_mph" if ttype == "sustained" else f"wind_{ttype}_mph"
        values = [_to_num(v.text) for v in ws.findall("value")]
        columns[col] = (layout, values)

    # Wind direction
    for wd in params.findall("direction"):
        layout = wd.get("time-layout", "")
        values = [_to_num(v.text) for v in wd.findall("value")]
        columns["wind_direction_deg"] = (layout, values)

    # Cloud cover
    for cc in params.findall("cloud-amount"):
        layout = cc.get("time-layout", "")
        values = [_to_num(v.text) for v in cc.findall("value")]
        columns["cloud_cover_pct"] = (layout, values)

    # Precipitation probability
    for pop in params.findall("probability-of-precipitation"):
        layout = pop.get("time-layout", "")
        values = [_to_num(v.text) for v in pop.findall("value")]
        columns["precip_prob_pct"] = (layout, values)

    # Humidity
    for rh in params.findall("humidity"):
        layout = rh.get("time-layout", "")
        values = [_to_num(v.text) for v in rh.findall("value")]
        columns["relative_humidity_pct"] = (layout, values)

    if not columns:
        return pd.DataFrame()

    # Use the first layout as the primary time axis
    primary_layout = list(columns.values())[0][0]
    timestamps = time_layouts.get(primary_layout, [])

    df_data = {"timestamp": timestamps}
    for col_name, (layout, values) in columns.items():
        if layout == primary_layout and len(values) == len(timestamps):
            df_data[col_name] = values

    return pd.DataFrame(df_data)


def _to_num(text: Optional[str]) -> Optional[float]:
    if text is None or text.strip() == "" or text.strip() == "--":
        return None
    try:
        return float(text)
    except ValueError:
        return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch NWS hourly forecast")
    parser.add_argument("site", nargs="?", default="KLAX",
                        help="ICAO station identifier (default: KLAX)")
    args = parser.parse_args()

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)

    f = ForecastIngestion(args.site)
    df = f.fetch_forecast()
    print(f"Forecast high: {df['temperature_f'].max()}F, low: {df['temperature_f'].min()}F")
    print()
    print(df.to_string())
