"""
Hourly weather at each stadium from Open-Meteo (free, no key needed).

Past games use the historical archive. Upcoming games, and games from the
last week (the archive runs a few days behind), use the forecast API.
Responses are saved to data/raw/weather/ like the CFBD pulls.
"""

import json
import time
from datetime import date, timedelta

import pandas as pd
import requests

from ticketing.config import RAW_DIR, WEATHER_ARCHIVE_URL, WEATHER_FORECAST_URL


CACHE_DIR = RAW_DIR / "weather"

HOURLY_FIELDS = "temperature_2m,precipitation,wind_speed_10m"

# Rain that falls in the hours before kickoff changes walk-up decisions as
# much as rain during the game, so the window starts well before kickoff.
RAIN_HOURS_BEFORE = 6
RAIN_HOURS_AFTER = 3

# The archive runs a few days behind; the forecast reaches 16 days ahead.
ARCHIVE_LAG_DAYS = 7
FORECAST_DAYS = 15


def hourly_weather(latitude, longitude, start, end, cache_name):
    """Hourly temperature (F), precipitation (in) and wind (mph) in UTC, cached."""

    recent = pd.Timestamp(start).date() >= date.today() - timedelta(days=ARCHIVE_LAG_DAYS + 2)

    # Forecasts change daily, so recent and future windows get dated cache files.
    if recent:
        cache_name = f"{cache_name}_forecast_{date.today().isoformat()}"

    path = CACHE_DIR / f"{cache_name}.json"

    if path.exists():
        data = json.loads(path.read_text())
    else:
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": str(start),
            "end_date": str(end),
            "hourly": HOURLY_FIELDS,
            "temperature_unit": "fahrenheit",
            "wind_speed_unit": "mph",
            "precipitation_unit": "inch",
            "timezone": "GMT",
        }
        data = _get(WEATHER_FORECAST_URL if recent else WEATHER_ARCHIVE_URL, params)

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))

    hourly = data["hourly"]

    return pd.DataFrame({
        "time": pd.to_datetime(hourly["time"], utc=True),
        "temp_f": hourly["temperature_2m"],
        "precip_in": hourly["precipitation"],
        "wind_mph": hourly["wind_speed_10m"],
    }).set_index("time")


def at_kickoff(hourly, kickoff_utc):
    """Kickoff temperature, rain around the game window, and average game-time wind."""

    hour = kickoff_utc.floor("h")

    if hour not in hourly.index:
        return {"temp_f": None, "precip_in": None, "wind_mph": None}

    rain_window = hourly.loc[hour - pd.Timedelta(hours=RAIN_HOURS_BEFORE): hour + pd.Timedelta(hours=RAIN_HOURS_AFTER)]
    game_window = hourly.loc[hour: hour + pd.Timedelta(hours=3)]

    return {
        "temp_f": hourly.at[hour, "temp_f"],
        "precip_in": rain_window["precip_in"].sum(min_count=1),
        "wind_mph": game_window["wind_mph"].mean(),
    }


def _get(url, params):
    """GET with patient retries: Open-Meteo's free tier throttles by the minute."""

    for attempt in range(6):
        response = requests.get(url, params=params, timeout=60)

        if response.status_code == 429:
            time.sleep(30 * (attempt + 1))
            continue

        response.raise_for_status()
        return response.json()

    raise RuntimeError(f"Open-Meteo kept rate-limiting {params['latitude']},{params['longitude']}")
