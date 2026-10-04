"""
Step 6: grade a week's committed predictions once the games are played.

Joins the predictions file with the attendance CFBD reports afterward,
writes week_NN_scored.csv next to it, and rebuilds the season scorecard
from every scored week.

  python scripts/06_score_week.py --week 7 [--commit]
"""

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from ticketing import cfbd
from ticketing.config import LIVE_SEASON, PREDICTIONS_DIR, PROJECT_DIR, SELLOUT_THRESHOLD


parser = argparse.ArgumentParser()
parser.add_argument("--week", type=int, required=True)
parser.add_argument("--commit", action="store_true")
args = parser.parse_args()

folder = PREDICTIONS_DIR / str(LIVE_SEASON)
parts = sorted(folder.glob(f"week_{args.week:02d}.csv")) + sorted(folder.glob(f"week_{args.week:02d}_part*.csv"))
predictions = pd.concat([pd.read_csv(p) for p in parts])

# Always re-pull: attendance is often posted a day or two after the game.
results = pd.DataFrame(cfbd.games(LIVE_SEASON, refresh=True))[["id", "completed", "attendance"]]
scored = predictions.merge(results.rename(columns={"id": "game_id"}), on="game_id", how="left")
scored = scored[scored["completed"].fillna(False).astype(bool) & scored["attendance"].gt(0)].copy()

if scored.empty:
    raise SystemExit("No attendance posted yet for this week. Try again in a day or two.")

scored["actual_fill"] = (scored["attendance"] / scored["capacity"]).clip(upper=1.0)
scored["error_pts"] = 100 * (scored["pred_fill"] - scored["actual_fill"])
scored["error_seats"] = scored["error_pts"] / 100 * scored["capacity"]
scored["inside_band"] = scored["actual_fill"].between(scored["pred_fill_low"], scored["pred_fill_high"])
scored["sold_out"] = scored["actual_fill"] >= SELLOUT_THRESHOLD
scored["actually_low"] = scored["actual_fill"] - scored["school_normal_fill"] <= -0.05
scored["flagged_and_low"] = scored["at_risk"] & scored["actually_low"]

out = folder / f"week_{args.week:02d}_scored.csv"
scored.round(4).to_csv(out, index=False)

# The scorecard is rebuilt from every scored week, so it can't drift from them.
every_week = pd.concat([pd.read_csv(p).assign(week=int(p.stem.split("_")[1]))
                        for p in sorted(folder.glob("week_*_scored.csv"))])
scorecard = every_week.groupby("week").agg(
    games=("game_id", "size"),
    mae_pts=("error_pts", lambda e: e.abs().mean()),
    bias_pts=("error_pts", "mean"),
    band_coverage=("inside_band", "mean"),
    flagged_at_risk=("at_risk", "sum"),
    flagged_and_low=("flagged_and_low", "sum"),
    actually_low=("actually_low", "sum"),
).round(3)
scorecard.to_csv(folder / "scorecard.csv")

print(scored[["home_team", "away_team", "pred_fill", "actual_fill", "error_pts", "at_risk", "actually_low"]]
      .round(3).to_string(index=False))
print()
print(scorecard.to_string())

if args.commit:
    paths = [str(p.relative_to(PROJECT_DIR)) for p in (out, folder / "scorecard.csv")]
    subprocess.run(["git", "add", *paths], cwd=PROJECT_DIR, check=True)
    subprocess.run(["git", "commit", "-m", f"Score {LIVE_SEASON} week {args.week} predictions"], cwd=PROJECT_DIR, check=True)
