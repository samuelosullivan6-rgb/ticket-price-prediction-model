"""
Cached client for the CollegeFootballData.com API.

Every response is saved to data/raw/cfbd/ as JSON. Later runs read the file
instead of calling the API, which keeps the free tier's monthly call limit
safe and makes every result reproducible from the saved pulls.
"""

import json
import os
import time
from datetime import date

import requests

from ticketing.config import CFBD_URL, PROJECT_DIR, RAW_DIR, LIVE_SEASON


CACHE_DIR = RAW_DIR / "cfbd"


def api_key():
    """Read CFBD_API_KEY from the environment, or from a .env file in the project."""

    if os.environ.get("CFBD_API_KEY"):
        return os.environ["CFBD_API_KEY"]

    env_file = PROJECT_DIR / ".env"

    if env_file.exists():
        for line in env_file.read_text().splitlines():
            name, _, value = line.partition("=")
            if name.strip() == "CFBD_API_KEY":
                return value.strip().strip('"').strip("'")

    raise SystemExit(
        "No CFBD API key found. Get a free key at "
        "https://collegefootballdata.com/key and put it in a .env file:\n"
        "    CFBD_API_KEY=your-key-here"
    )


def fetch(endpoint, cache_name, refresh=False, **params):
    """
    Return the JSON for one endpoint, from disk if it was pulled before.

    The live season changes every week, so its files are stamped with
    today's date: one fresh pull per day, cached after that.
    """

    if params.get("year") == LIVE_SEASON:
        cache_name = f"{cache_name}_{date.today().isoformat()}"

    path = CACHE_DIR / f"{cache_name}.json"

    if path.exists() and not refresh:
        return json.loads(path.read_text())

    headers = {"Authorization": f"Bearer {api_key()}", "Accept": "application/json"}

    for attempt in range(4):
        response = requests.get(f"{CFBD_URL}{endpoint}", headers=headers, params=params, timeout=60)

        # 429 means "slow down"; anything else unexpected is a real error.
        if response.status_code == 429:
            time.sleep(10 * (attempt + 1))
            continue

        response.raise_for_status()
        break
    else:
        raise RuntimeError(f"CFBD kept rate-limiting {endpoint} {params}")

    data = response.json()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))

    return data


# --------------------------------------------------
# ONE FUNCTION PER DATASET
# --------------------------------------------------
# Regular season only: bowls and conference title games are neutral-site
# events with their own ticketing, so they never enter the sample.

def games(season, **kw):
    return fetch("/games", f"games_{season}", year=season, seasonType="regular", **kw)


def media(season, **kw):
    return fetch("/games/media", f"media_{season}", year=season, seasonType="regular", **kw)


def lines(season, **kw):
    return fetch("/lines", f"lines_{season}", year=season, seasonType="regular", **kw)


def rankings(season, **kw):
    return fetch("/rankings", f"rankings_{season}", year=season, seasonType="regular", **kw)


def calendar(season, **kw):
    return fetch("/calendar", f"calendar_{season}", year=season, **kw)


def venues(**kw):
    return fetch("/venues", "venues", **kw)


def teams(**kw):
    return fetch("/teams", "teams", **kw)
