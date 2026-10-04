"""
Step 2: build the modeling table, one row per FBS home game.

Reads only the saved pulls from step 1. Writes
data/processed/home_games.parquet and a season-by-season summary to
results/data/.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ticketing import features
from ticketing.config import MODEL_SEASONS, MODEL_TABLE, RESULTS_DIR


df = features.build(MODEL_SEASONS)

MODEL_TABLE.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(MODEL_TABLE, index=False)

# Kickoff-time effects need a known kickoff, so TBD games are counted
# here and left out of the models.
played = df[df["fill"].notna()]
summary = played.groupby("season").agg(
    home_games=("game_id", "size"),
    median_fill=("fill", "median"),
    sellout_rate=("sellout", "mean"),
    tbd_kickoffs=("tbd", "sum"),
    market_spread_share=("spread_source", lambda s: (s == "market").mean()),
    weather_coverage=("temp_f", lambda s: s.notna().mean()),
).round(3)

out = RESULTS_DIR / "data"
out.mkdir(parents=True, exist_ok=True)
summary.to_csv(out / "season_summary.csv")

print(summary.to_string())
print(f"\n{len(played):,} home games with attendance across {played['home_team'].nunique()} schools")
print(f"Saved {MODEL_TABLE.relative_to(MODEL_TABLE.parents[2])}")
