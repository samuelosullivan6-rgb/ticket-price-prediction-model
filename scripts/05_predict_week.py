"""
Step 5: predict this coming weekend's home crowds before kickoff.

Run once a week, after the networks announce kickoff times (usually 6 or
12 days out) and before the first game. With --commit the prediction
files go straight into git, so the commit timestamp shows they existed
before the games were played. Files are never overwritten: running again
later in the week (after "6-day hold" kickoff times are announced) adds
a part-2 file covering only the newly announced games.

  python scripts/05_predict_week.py              # the next week, files only
  python scripts/05_predict_week.py --commit     # ...and commit them
  python scripts/05_predict_week.py --week 8

Each game gets the boosted model's forecast (with a 10-90% band and a
sellout chance), its gap from the school's normal crowd, and the
factors behind that gap from the step-3 regression.
"""

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from ticketing import cfbd, features, models
from ticketing.config import LIVE_SEASON, MODEL_TABLE, PREDICTIONS_DIR, PROJECT_DIR


AT_RISK_GAP = 0.05   # same definition that was tested on the 2025 holdout

# Plain-English names for each regression group, used in the "why" column.
DRIVER_NAMES = {
    "kickoff": "kickoff time",
    "day": "day of week",
    "tv": "TV slot",
    "competitiveness": "lopsided matchup",
    "opponent": "opponent draw",
    "home_team_form": "home team's record",
    "weather": "weather forecast",
    "calendar": "point in the season",
}


def next_week(season, now):
    """The first regular-season week whose games haven't started yet."""

    for week in sorted(cfbd.calendar(season), key=lambda w: w["week"]):
        if week.get("seasonType") == "regular" and pd.Timestamp(week["firstGameStart"]) > now:
            return week["week"]
    raise SystemExit("No regular-season weeks left to predict.")


def describe(row, group, points):
    """One driver as readable text, e.g. 'kickoff time: 12:00 PM (-3.1 pts)'."""

    detail = {
        "kickoff": row["kickoff_local"].split(", ")[-1],
        "day": row["day_type"].replace("_", "/"),
        "tv": row["tv_outlet"] if pd.notna(row["tv_outlet"]) else "stream only",
        "competitiveness": (f"{row['home_team']} favored by {-row['spread']:.1f}" if row["spread"] < 0
                            else f"{row['home_team']} +{row['spread']:.1f}"),
        "opponent": row["away_team"] + (" (FCS)" if row["away_fcs"] else ""),
        "home_team_form": f"{row['home_winpct'] * (row['home_games_played'] + 2) - 1:.0f}-"
                          f"{(1 - row['home_winpct']) * (row['home_games_played'] + 2) - 1:.0f} so far",
        "weather": f"{row['temp_f']:.0f} F, {row['precip_in']:.2f} in rain" if pd.notna(row["temp_f"]) else "no forecast yet",
        "calendar": "home opener" if row["home_opener"] else row["season_phase"],
    }[group]
    return f"{DRIVER_NAMES[group]}: {detail} ({100 * points:+.1f} pts)"


# --------------------------------------------------
# SETUP
# --------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--week", type=int)
parser.add_argument("--commit", action="store_true", help="git-commit the prediction files")
args = parser.parse_args()

now = pd.Timestamp.now(tz="UTC")
week = args.week or next_week(LIVE_SEASON, now)

out_dir = PREDICTIONS_DIR / str(LIVE_SEASON)
earlier = sorted(out_dir.glob(f"week_{week:02d}.csv")) + sorted(out_dir.glob(f"week_{week:02d}_part*.csv"))
already_predicted = set(pd.concat([pd.read_csv(p)["game_id"] for p in earlier])) if earlier else set()

name = f"week_{week:02d}" + (f"_part{len(earlier) + 1}" if earlier else "")
out_csv, out_md = out_dir / f"{name}.csv", out_dir / f"{name}.md"


# --------------------------------------------------
# DATA
# --------------------------------------------------
# History comes from step 2; this season is pulled fresh, so its finished
# games (with attendance) join the training data and upcoming ones get
# the latest lines, polls and forecast.

history = pd.read_parquet(MODEL_TABLE)
live = features.build([LIVE_SEASON], verbose=False)
both = models.prepare(pd.concat([history, live], ignore_index=True))

train = both[both["fill"].notna()]
upcoming = both[(both["season"] == LIVE_SEASON) & (both["week"] == week) & (both["start_utc"] > now)
                & ~both["game_id"].isin(already_predicted)]
games, no_time = upcoming[~upcoming["tbd"]].copy(), upcoming[upcoming["tbd"]]

if games.empty:
    raise SystemExit(f"No new home games with announced kickoff times to predict in week {week}.")


# --------------------------------------------------
# PREDICT
# --------------------------------------------------

# CFBD posts attendance weeks after the games, so early in a season the
# "fill so far" feature is blank for every upcoming game. A feature the
# games being predicted can't have is dropped from training as well, so
# the model never leans on information it won't get.
usable = [f for f in models.GBM_FEATURES if games[f].notna().any()]
dropped = sorted(set(models.GBM_FEATURES) - set(usable))
gbm = models.fit_gbm(train, features=usable)
games = games.join(models.predict_gbm(gbm, games))

# "Normal" is the school's average fill last season, the same yardstick
# the at-risk flag was tested against on the 2025 holdout.
games["school_normal_fill"] = games["team_prev_fill"].fillna(train["fill_capped"].mean())
games["pred_attendance"] = (games["pred_fill"] * games["capacity"]).round(-2)
games["gap_pts"] = 100 * (games["pred_fill"] - games["school_normal_fill"])
games["gap_seats"] = (games["gap_pts"] / 100 * games["capacity"]).round(-2)
games["gap_dollars_at_assumed_price"] = (games["gap_seats"] * games["assumed_price"]).round(-3)
games["at_risk"] = games["gap_pts"] <= -100 * AT_RISK_GAP


# --------------------------------------------------
# WHY: DRIVERS FROM THE FIXED-EFFECTS REGRESSION
# --------------------------------------------------

fitted = train[train["kick_slot"] != "tbd"]
X_all, groups = models.design_matrix(pd.concat([fitted, games]), models.MAIN_SPEC)
X_hist, X_new = X_all.iloc[: len(fitted)], X_all.iloc[len(fitted):]
keep = X_hist.columns[X_hist.std() > 0].union(["const"])
ols = models.fit_ols(fitted["fill_capped"], X_hist[keep], fitted["home_team"])

contrib = models.drivers(ols.params, X_new[keep], X_hist[keep], games["home_team"], fitted["home_team"], groups)
contrib.index = games.index

games["why"] = [
    "; ".join(describe(games.loc[i], g, v) for g, v in contrib.loc[i].sort_values().items() if v <= -0.01)
    or "nothing unusual"
    for i in games.index
]


# --------------------------------------------------
# WRITE
# --------------------------------------------------

columns = ["game_id", "kickoff_local", "home_team", "away_team", "tv_outlet", "spread", "home_win_prob",
           "temp_f", "precip_in", "capacity", "school_normal_fill", "pred_fill", "pred_fill_low",
           "pred_fill_high", "pred_attendance", "sellout_prob", "gap_pts", "gap_seats",
           "gap_dollars_at_assumed_price", "at_risk", "why"]
games = games.sort_values("gap_seats")

out_dir.mkdir(parents=True, exist_ok=True)
games[columns].round(3).to_csv(out_csv, index=False)

stamp = now.strftime("%Y-%m-%d %H:%M UTC")
lines = [
    f"# {LIVE_SEASON} week {week}: home crowd predictions" + (f" (part {len(earlier) + 1})" if earlier else ""),
    "",
    f"Made {stamp}, before any of these games kicked off. {len(games)} games; "
    f"{int(games['at_risk'].sum())} flagged at risk (forecast {100 * AT_RISK_GAP:.0f}+ pts of capacity "
    f"below the school's average last season)."
    + (f" Not yet available this week, so left out of the model: {', '.join(dropped)}." if dropped else ""),
    "",
    "| Kickoff (local) | Game | Forecast fill (10-90%) | Seats vs. normal | Sellout chance | Why |",
    "|---|---|---|---:|---:|---|",
]
for _, g in games.iterrows():
    flag = "**at risk** " if g["at_risk"] else ""
    lines.append(
        f"| {g['kickoff_local']} | {flag}{g['away_team']} at {g['home_team']} | "
        f"{g['pred_fill']:.0%} ({g['pred_fill_low']:.0%}-{g['pred_fill_high']:.0%}) | "
        f"{g['gap_seats']:+,.0f} | {g['sellout_prob']:.0%} | {g['why']} |"
    )
if len(no_time):
    lines += ["", "Not predicted, kickoff time still TBD: "
              + ", ".join(f"{r.away_team} at {r.home_team}" for r in no_time.itertuples())]
lines += ["", "Dollar figures in the CSV use the assumed ticket prices in ticketing/config.py, not real revenue."]
out_md.write_text("\n".join(lines) + "\n")

print("\n".join(lines))

if args.commit:
    paths = [str(p.relative_to(PROJECT_DIR)) for p in (out_csv, out_md)]
    subprocess.run(["git", "add", *paths], cwd=PROJECT_DIR, check=True)
    subprocess.run(["git", "commit", "-m", f"Predict {LIVE_SEASON} week {week} home crowds (before kickoff)"],
                   cwd=PROJECT_DIR, check=True)
