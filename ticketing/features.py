"""
Turns the raw API pulls into one row per home game, holding everything
known before kickoff that might move ticket demand.

The same build() feeds the historical models and the live weekly
predictions, so a feature can't mean one thing in training and another
on game week.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm

from ticketing import cfbd, weather
from ticketing.config import (
    AFTERNOON_STARTS, ASSUMED_TICKET_PRICE, NIGHT_STARTS,
    POWER_CONFERENCES, POWER_INDEPENDENTS, SELLOUT_THRESHOLD, SPREAD_SD,
    VALID_FILL_RANGE,
)


# --------------------------------------------------
# BUILD
# --------------------------------------------------

def build(seasons, verbose=True):
    """Every FBS home game in `seasons`, with features and (once played) attendance."""

    # Earlier seasons are loaded only so each team's previous-season record
    # and attendance can be looked up.
    lookback = {previous_season(s) for s in seasons} | {s - 1 for s in seasons}
    all_games = load_games(sorted(set(seasons) | lookback))

    df = home_game_table(all_games, load_venues(), verbose)
    df = add_demand_history(df)
    df = df[df["season"].isin(seasons)].copy()

    df = add_kickoff(df)
    df = add_team_form(df, all_games)
    df = add_tv(df, seasons)
    df = add_rankings(df, seasons)
    df = add_opponent(df)
    df = add_lines(df, seasons)
    df = add_weather(df, verbose)
    df = add_price(df)

    return df.sort_values(["season", "start_utc"]).reset_index(drop=True)


def previous_season(season):
    """The last season with normal crowds before this one. 2020 doesn't count."""

    return 2019 if season == 2021 else season - 1


# --------------------------------------------------
# LOADING
# --------------------------------------------------

def load_games(seasons):
    games = pd.DataFrame([g for s in seasons for g in cfbd.games(s)])
    games["start_utc"] = pd.to_datetime(games["startDate"], utc=True)
    return games


def load_venues():
    venues = pd.DataFrame(cfbd.venues())
    return venues.rename(columns={"id": "venue_id", "name": "venue_name"})[
        ["venue_id", "venue_name", "capacity", "timezone", "latitude", "longitude", "state", "dome"]
    ]


def home_game_table(games, venues, verbose=False):
    """Regular-season games on an FBS team's own field, with fill rate where known."""

    on_campus = games[
        (games["homeClassification"] == "fbs")
        & ~games["neutralSite"].fillna(False).astype(bool)
        & ~games["season"].isin([2020])
    ]

    df = pd.DataFrame({
        "game_id": on_campus["id"],
        "season": on_campus["season"],
        "week": on_campus["week"],
        "start_utc": on_campus["start_utc"],
        "tbd": on_campus["startTimeTBD"].fillna(False).astype(bool),
        "completed": on_campus["completed"].fillna(False).astype(bool),
        "home_team": on_campus["homeTeam"],
        "home_conf": on_campus["homeConference"],
        "away_team": on_campus["awayTeam"],
        "away_conf": on_campus["awayConference"],
        "away_fcs": on_campus["awayClassification"] != "fbs",
        "conf_game": on_campus["conferenceGame"].fillna(False).astype(bool),
        "home_elo": on_campus["homePregameElo"],
        "away_elo": on_campus["awayPregameElo"],
        "venue_id": on_campus["venueId"],
        "attendance": on_campus["attendance"],
    }).merge(venues, on="venue_id", how="left")

    df["fill"] = df["attendance"] / df["capacity"]

    # A missing or zero crowd on a completed game means "not reported",
    # and an impossible fill rate means a bad capacity or a typo.
    valid = df["fill"].between(*VALID_FILL_RANGE)
    if verbose:
        reported = df["completed"] & df["attendance"].gt(0)
        print(f"  {len(df):,} home games; {(reported & ~valid).sum()} dropped for impossible fill rates, "
              f"{(df['completed'] & ~df['attendance'].gt(0)).sum()} had no reported attendance")
    df.loc[~valid, ["attendance", "fill"]] = np.nan

    # Above 100% means standing room or generous counting, not extra demand
    # the stadium could have sold, so models see fill capped at capacity.
    df["fill_capped"] = df["fill"].clip(upper=1.0)
    df["sellout"] = (df["fill"] >= SELLOUT_THRESHOLD).astype(float).where(df["fill"].notna())

    df = df.sort_values("start_utc")
    df["home_game_number"] = df.groupby(["home_team", "season"]).cumcount() + 1

    return df


# --------------------------------------------------
# DEMAND HISTORY
# --------------------------------------------------

def add_demand_history(df):
    """How full the stadium was last season, and so far this season."""

    history = (
        df.dropna(subset=["fill"])
        .groupby(["home_team", "season"])
        .agg(team_prev_fill=("fill_capped", "mean"), team_prev_sellout_rate=("sellout", "mean"))
        .reset_index()
        .rename(columns={"season": "prev_season"})
    )

    df["prev_season"] = df["season"].map(previous_season)
    df = df.merge(history, on=["home_team", "prev_season"], how="left")

    # Shift by one so a game never sees its own crowd.
    df = df.sort_values("start_utc")
    df["team_fill_to_date"] = (
        df.groupby(["home_team", "season"])["fill_capped"]
        .transform(lambda s: s.shift().expanding().mean())
    )

    return df


# --------------------------------------------------
# KICKOFF
# --------------------------------------------------

def add_kickoff(df):
    """Local kickoff hour, time slot and day of week."""

    zones = df["timezone"].where(df["timezone"].notna() & df["timezone"].ne(""), df.apply(_guess_timezone, axis=1))
    local = [t.tz_convert(z) for t, z in zip(df["start_utc"], zones)]

    # TBD games carry a placeholder time in CFBD, not a real kickoff.
    df["kick_hour"] = [t.hour + t.minute / 60 for t in local]
    df.loc[df["tbd"], "kick_hour"] = np.nan
    df["kickoff_local"] = np.where(df["tbd"], "TBD", [t.strftime("%a %b %d, %I:%M %p %Z") for t in local])
    df["month"] = [t.month for t in local]
    weekday = pd.Series([t.dayofweek for t in local], index=df.index)

    df["kick_slot"] = np.select(
        [df["tbd"], df["kick_hour"] < AFTERNOON_STARTS, df["kick_hour"] < NIGHT_STARTS],
        ["tbd", "noon", "afternoon"],
        "night",
    )
    df["day_type"] = np.select([weekday == 5, weekday.between(1, 4)], ["saturday", "weeknight"], "sun_mon")

    df["home_opener"] = df["home_game_number"] == 1
    df["late_season"] = df["month"] >= 11

    # Non-conference games cluster in September, when crowds run high
    # anyway. A phase control keeps that from posing as a conference effect.
    df["season_phase"] = np.select([df["month"] <= 9, df["month"] == 10], ["september", "october"], "november")

    return df


def _guess_timezone(row):
    """Fallback when CFBD has no time zone for a venue: infer it from longitude."""

    if row["state"] == "AZ":
        return "America/Phoenix"
    if row["state"] == "HI":
        return "Pacific/Honolulu"

    lon = row["longitude"]
    if pd.isna(lon) or lon > -87:
        return "America/New_York"
    if lon > -101:
        return "America/Chicago"
    if lon > -114:
        return "America/Denver"
    return "America/Los_Angeles"


# --------------------------------------------------
# TEAM FORM
# --------------------------------------------------

def add_team_form(df, all_games):
    """Each team's record entering the game, and last season's record."""

    played = all_games[all_games["completed"].fillna(False).astype(bool) & all_games["homePoints"].notna()]

    # One row per team per game, from that team's side.
    long = pd.concat([
        pd.DataFrame({"season": played["season"], "team": played["homeTeam"], "start_utc": played["start_utc"],
                      "win": played["homePoints"] > played["awayPoints"]}),
        pd.DataFrame({"season": played["season"], "team": played["awayTeam"], "start_utc": played["start_utc"],
                      "win": played["awayPoints"] > played["homePoints"]}),
    ]).sort_values("start_utc")

    long["wins"] = long.groupby(["team", "season"])["win"].cumsum()
    long["games"] = long.groupby(["team", "season"]).cumcount() + 1

    season_pct = long.groupby(["team", "season"])["win"].mean()

    for side in ["home", "away"]:
        # As-of join: the record from games strictly before this kickoff.
        to_date = pd.merge_asof(
            df[["start_utc", "season", f"{side}_team"]].reset_index().sort_values("start_utc"),
            long[["start_utc", "season", "team", "wins", "games"]].rename(columns={"team": f"{side}_team"}),
            on="start_utc", by=["season", f"{side}_team"], allow_exact_matches=False,
        ).set_index("index")
        wins, games = to_date["wins"].fillna(0), to_date["games"].fillna(0)

        # One win and one loss of padding, so a 1-0 team isn't treated as perfect.
        df[f"{side}_winpct"] = (wins + 1) / (games + 2)
        df[f"{side}_games_played"] = games

        # 2020's record counts here: the team still played, only the crowds were missing.
        keys = list(zip(df[f"{side}_team"], df["season"] - 1))
        df[f"{side}_prev_winpct"] = [season_pct.get(k, np.nan) for k in keys]

    return df


# --------------------------------------------------
# TV
# --------------------------------------------------

# Ordered best-to-worst exposure. A game carried on several outlets is
# labeled by its biggest one.
TV_TIERS = ["broadcast", "espn", "cable", "conference_net", "stream"]

BROADCAST = {"ABC", "CBS", "NBC", "FOX"}
ESPN = {"ESPN", "ESPN2"}
CABLE = {"ESPNU", "FS1", "FS2", "CBSSN", "CBS SPORTS NETWORK", "ESPNEWS", "ESPNN", "ESPNC", "NBCSN",
         "FSN", "TRUTV", "TNT", "TBS", "NFL NETWORK", "THE CW", "CW", "USA", "USA NETWORK"}
CONFERENCE_NETS = ("SEC NETWORK", "SECN", "BTN", "BIG TEN NETWORK", "ACC NETWORK", "ACCN",
                   "PAC-12", "PAC12", "P12N", "LHN", "LONGHORN NETWORK")


def tv_tier(outlet, media_type):
    name = str(outlet).strip().upper()

    if media_type == "web" or name.endswith("+") or "ACCNX" in name:
        return "stream"
    if name in BROADCAST:
        return "broadcast"
    if name in ESPN:
        return "espn"
    if name in CABLE:
        return "cable"
    if name.startswith(CONFERENCE_NETS):
        return "conference_net"
    return "stream"


def add_tv(df, seasons):
    media = pd.DataFrame([m for s in seasons for m in cfbd.media(s)])
    media = media[media["mediaType"].isin(["tv", "web"])].copy()
    media["tv"] = [tv_tier(o, t) for o, t in zip(media["outlet"], media["mediaType"])]
    media["rank"] = media["tv"].map(TV_TIERS.index)

    best = media.sort_values("rank").drop_duplicates("id")[["id", "tv", "outlet"]]
    df = df.merge(best.rename(columns={"id": "game_id", "outlet": "tv_outlet"}), on="game_id", how="left")
    # A game with no listing was streamed or shown regionally, if at all.
    df["tv"] = df["tv"].fillna("stream")

    return df


# --------------------------------------------------
# RANKINGS
# --------------------------------------------------

# CFBD numbers its polls so that the week-N poll is the one released
# before week N games, so a game uses the latest poll whose week is no
# later than its own.
POLL_PREFERENCE = ["AP Top 25", "Coaches Poll"]


def add_rankings(df, seasons):
    rows = []
    for s in seasons:
        for week in cfbd.rankings(s):
            polls = {p["poll"]: p["ranks"] for p in week["polls"]}
            name = next((p for p in POLL_PREFERENCE if p in polls), None)
            if name:
                rows += [(week["season"], week["week"], r["school"], r["rank"]) for r in polls[name]]

    polls = pd.DataFrame(rows, columns=["season", "poll_week", "school", "rank"])

    # Find the poll each game should use first, then look teams up in that
    # poll only. A team that dropped out shouldn't inherit an old ranking.
    weeks = polls[["season", "poll_week"]].drop_duplicates().sort_values("poll_week")
    df = pd.merge_asof(
        df.reset_index().sort_values("week"), weeks,
        left_on="week", right_on="poll_week", by="season",
    ).set_index("index").sort_index()

    for side in ["home", "away"]:
        side_ranks = polls.rename(columns={"school": f"{side}_team", "rank": f"{side}_rank"})
        df = df.reset_index().merge(side_ranks, on=["season", "poll_week", f"{side}_team"], how="left").set_index("index")
        df[f"{side}_ranked"] = df[f"{side}_rank"].notna()

    df.index.name = None
    return df.drop(columns="poll_week")


# --------------------------------------------------
# OPPONENT
# --------------------------------------------------

def add_opponent(df):
    """Opponent prestige and how far its fans have to travel."""

    location = pd.DataFrame(
        [{"away_team": t["school"], **(t.get("location") or {})} for t in cfbd.teams()]
    ).reindex(columns=["away_team", "latitude", "longitude", "state"]).rename(
        columns={"latitude": "away_lat", "longitude": "away_lon", "state": "away_state"}
    )

    df = df.merge(location.drop_duplicates("away_team"), on="away_team", how="left")

    df["distance_mi"] = _haversine_miles(df["away_lat"], df["away_lon"], df["latitude"], df["longitude"])
    df["same_state"] = df["away_state"] == df["state"]

    df["home_power"] = df.apply(lambda r: is_power(r["home_conf"], r["home_team"], r["season"]), axis=1)
    df["away_power"] = df.apply(lambda r: is_power(r["away_conf"], r["away_team"], r["season"]), axis=1)

    return df.drop(columns=["away_lat", "away_lon", "away_state"])


def is_power(conference, team, season):
    # The Pac-12 was down to two schools from 2024 on.
    if conference == "Pac-12" and season >= 2024:
        return False
    return conference in POWER_CONFERENCES or team in POWER_INDEPENDENTS


def _haversine_miles(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = (np.radians(pd.to_numeric(x, errors="coerce")) for x in (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 3959 * 2 * np.arcsin(np.sqrt(a))


# --------------------------------------------------
# BETTING LINES
# --------------------------------------------------

def add_lines(df, seasons):
    """
    Point spread (negative = home team favored) and the home win
    probability it implies. The spread is the market's read on how
    lopsided a game will be, which is what the uncertainty-of-outcome
    test needs.
    """

    spreads = {}
    for s in seasons:
        for game in cfbd.lines(s):
            values = [line["spread"] for line in game["lines"] if line.get("spread") is not None]
            if values:
                # Median across sportsbooks, so one stale book can't move it.
                spreads[game["id"]] = float(np.median(values))

    df["spread"] = df["game_id"].map(spreads)
    df["spread_source"] = np.where(df["spread"].notna(), "market", None)

    # No line posted (common against FCS teams): estimate one from the
    # Elo gap, fit on games that have both.
    elo_gap = df["home_elo"] - df["away_elo"]
    known = df["spread"].notna() & elo_gap.notna()
    slope, intercept = np.polyfit(elo_gap[known], df.loc[known, "spread"], 1)

    from_elo = df["spread"].isna() & elo_gap.notna()
    df.loc[from_elo, "spread"] = intercept + slope * elo_gap[from_elo]
    df.loc[from_elo, "spread_source"] = "elo"

    # Still missing: use the typical spread for the same kind of opponent.
    typical = df[df["spread_source"] == "market"].groupby("away_fcs")["spread"].median()
    rest = df["spread"].isna()
    df.loc[rest, "spread"] = df.loc[rest, "away_fcs"].map(typical)
    df.loc[rest, "spread_source"] = "typical"

    df["home_win_prob"] = norm.cdf(-df["spread"] / SPREAD_SD)
    df["abs_spread"] = df["spread"].abs()
    df["spread_bucket"] = pd.cut(
        df["spread"],
        bins=[-np.inf, -28, -17, -7, 7, np.inf],
        labels=["fav_28plus", "fav_17_28", "fav_7_17", "close", "underdog_7plus"],
    ).astype(str)

    return df


# --------------------------------------------------
# WEATHER
# --------------------------------------------------

def add_weather(df, verbose=False):
    """Kickoff temperature, rain around the game, and wind at each stadium."""

    out = pd.DataFrame(index=df.index, columns=["temp_f", "precip_in", "wind_mph"], dtype=float)

    # Past games come from the archive, recent and upcoming ones from the
    # forecast, which only reaches about two weeks out. Later games wait.
    today = pd.Timestamp.now(tz="UTC").normalize()
    source = np.where(df["start_utc"] < today - pd.Timedelta(days=weather.ARCHIVE_LAG_DAYS), "archive", "forecast")
    in_range = df["start_utc"] < today + pd.Timedelta(days=weather.FORECAST_DAYS)

    # One request per stadium per season (its first to last home date)
    # keeps the Open-Meteo call count in the low thousands.
    eligible = ~df["tbd"] & df["latitude"].notna() & in_range
    groups = df[eligible].groupby(["venue_id", "season", source[eligible]])

    for i, ((venue_id, season, _), games) in enumerate(groups):
        if verbose and i % 100 == 0:
            print(f"  weather: {i:,} of {groups.ngroups:,} stadium-seasons")

        start = games["start_utc"].min().date() - pd.Timedelta(days=1)
        end = games["start_utc"].max().date() + pd.Timedelta(days=1)
        lat, lon = games["latitude"].iloc[0], games["longitude"].iloc[0]

        hourly = weather.hourly_weather(lat, lon, start, end, f"{int(venue_id)}_{season}_{start}_{end}")

        for idx, kickoff in games["start_utc"].items():
            out.loc[idx] = pd.Series(weather.at_kickoff(hourly, kickoff), dtype=float)

    df = df.join(out)

    # Under a roof the weather outside doesn't reach the seats.
    dome = df["dome"].fillna(False).astype(bool)
    df.loc[dome, ["temp_f", "precip_in", "wind_mph"]] = [70.0, 0.0, 0.0]

    df["temp_bin"] = pd.cut(
        df["temp_f"], bins=[-np.inf, 40, 55, 75, 85, np.inf],
        labels=["under_40", "40_55", "55_75", "75_85", "85_plus"], right=False,
    ).astype(str)
    df["rain"] = df["precip_in"] >= 0.10
    df["heavy_rain"] = df["precip_in"] >= 0.50
    df["windy"] = df["wind_mph"] >= 15

    return df


# --------------------------------------------------
# PRICE
# --------------------------------------------------

def add_price(df):
    """Attach the assumed ticket price used to turn seats into dollars."""

    df["price_tier"] = np.where(df["home_power"], "power", "other")
    df["assumed_price"] = df["price_tier"].map(ASSUMED_TICKET_PRICE)
    return df
