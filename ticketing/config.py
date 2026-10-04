"""
Project-wide settings. Every number or choice that changes a result lives
here, so the scripts never hide an assumption.
"""

from pathlib import Path


# --------------------------------------------------
# PATHS
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
RESULTS_DIR = PROJECT_DIR / "results"
PREDICTIONS_DIR = PROJECT_DIR / "predictions"

MODEL_TABLE = PROCESSED_DIR / "home_games.parquet"


# --------------------------------------------------
# SEASONS
# --------------------------------------------------

# Ten modeled seasons. 2020 is left out because COVID capped or banned crowds.
MODEL_SEASONS = [2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025]

# Never used for fitting. The predictive model is judged on this season.
HOLDOUT_SEASON = 2025

# Season being predicted live, week by week.
LIVE_SEASON = 2026


# --------------------------------------------------
# DEFINITIONS
# --------------------------------------------------

# A game counts as a sellout at 98% of listed capacity. Announced crowds
# hover just under or over capacity on sold-out days, so an exact 100%
# cut would miss most real sellouts.
SELLOUT_THRESHOLD = 0.98

# Fill rates outside this range are almost always data-entry errors
# (a missing digit, or the capacity of a different stadium).
VALID_FILL_RANGE = (0.10, 1.50)

# Local kickoff hour that starts each time slot.
AFTERNOON_STARTS = 13   # before 1 PM is a "noon" kickoff
NIGHT_STARTS = 18       # 6 PM or later is a night kickoff

# College football final margins have a standard deviation of roughly
# 15 points around the closing spread. Used to turn a spread into a
# home win probability.
SPREAD_SD = 15.0


# --------------------------------------------------
# DOLLAR ASSUMPTION (not data)
# --------------------------------------------------

# Schools don't publish revenue per seat, so dollars are an ASSUMPTION:
# average ticket revenue per distributed ticket, blended across season
# tickets, single games and discounted student seats. Change these to
# re-price every dollar figure in the project.
ASSUMED_TICKET_PRICE = {
    "power": 75,    # SEC, Big Ten, Big 12, ACC, Pac-12 (through 2023), Notre Dame
    "other": 25,    # Group of Five and the remaining independents
}

POWER_CONFERENCES = {"SEC", "Big Ten", "Big 12", "ACC", "Pac-12"}
POWER_INDEPENDENTS = {"Notre Dame"}


# --------------------------------------------------
# API
# --------------------------------------------------

CFBD_URL = "https://api.collegefootballdata.com"
WEATHER_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
WEATHER_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
