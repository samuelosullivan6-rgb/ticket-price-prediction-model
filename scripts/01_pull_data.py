"""
Step 1: pull every dataset once and save it to data/raw/.

CollegeFootballData.com: games (attendance, kickoff), venues (capacity,
location), teams (home towns), TV networks, betting lines and polls.
Open-Meteo: hourly weather at each stadium around every home game.

Anything already on disk is skipped, so re-running costs no API calls.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ticketing import cfbd, features
from ticketing.config import MODEL_SEASONS


# Seasons whose games are needed only for each team's previous-season
# record and attendance.
history = {features.previous_season(s) for s in MODEL_SEASONS} | {s - 1 for s in MODEL_SEASONS}

print("CollegeFootballData.com")
for season in sorted(set(MODEL_SEASONS) | history):
    cfbd.games(season)
    if season in MODEL_SEASONS:
        cfbd.media(season)
        cfbd.lines(season)
        cfbd.rankings(season)
    print(f"  {season} done")

cfbd.venues()
cfbd.teams()

print("Open-Meteo")
games = features.load_games(sorted(set(MODEL_SEASONS) | history))
home = features.home_game_table(games, features.load_venues())
features.add_weather(home[home["season"].isin(MODEL_SEASONS)], verbose=True)

print("All pulls saved under data/raw/")
