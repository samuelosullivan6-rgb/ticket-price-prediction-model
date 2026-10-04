"""
Step 3: what drives attendance, and what each factor is worth.

Four views of the same question, each answering a different worry:

  OLS, team + season fixed effects   the clean, readable answer
  Tobit (censored)                    sellouts hide demand above capacity
  Two-part                            "will it sell out?" + "how full if not?"
  OLS, team x season fixed effects    rules out a school's good or bad year

Effects are reported in points of capacity, seats per game, and dollars
per game at the ASSUMED ticket price in ticketing/config.py.

Also tests the uncertainty-of-outcome hypothesis (do lopsided games draw
fewer fans?) and estimates the demand hidden behind sellouts.

Writes to results/explain/.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy.stats import norm

from ticketing import charts, models
from ticketing.config import ASSUMED_TICKET_PRICE, MODEL_TABLE, RESULTS_DIR, SELLOUT_THRESHOLD


OUT = RESULTS_DIR / "explain"
OUT.mkdir(parents=True, exist_ok=True)

FOCUS_SCHOOL = "Clemson"


# --------------------------------------------------
# EFFECTS TO REPORT
# --------------------------------------------------
# Each effect is a weighted sum of coefficients, so "per doubling of
# distance" or "a heavy favorite vs. a toss-up" come out with proper
# standard errors like any single coefficient.

def toss_up_vs(p):
    """Weights for moving the home win probability from 50% to p."""
    return {"home_win_prob": p - 0.5, "home_win_prob_sq": p ** 2 - 0.25}


EFFECTS = {
    "Noon kickoff (vs. night)": {"kick_slot=noon": 1},
    "Afternoon kickoff (vs. night)": {"kick_slot=afternoon": 1},
    "Weeknight game (vs. Saturday)": {"day_type=weeknight": 1},
    "ABC / CBS / NBC / FOX (vs. stream only)": {"tv=broadcast": 1},
    "ESPN / ESPN2 (vs. stream only)": {"tv=espn": 1},
    "Other national cable (vs. stream only)": {"tv=cable": 1},
    "Conference network (vs. stream only)": {"tv=conference_net": 1},
    "Ranked opponent": {"away_ranked": 1},
    "Power-conference opponent": {"away_power": 1},
    "FCS opponent": {"away_fcs": 1},
    "Conference game": {"conf_game": 1},
    "Opponent's campus twice as far away": {"log_distance": np.log(2)},
    "Home team ranked": {"home_ranked": 1},
    "Home win % so far, +25 pts": {"home_winpct": 0.25},
    "Home win % last season, +25 pts": {"home_prev_winpct": 0.25},
    "Home team a heavy favorite (85% vs. 50%)": toss_up_vs(0.85),
    "Home team an underdog (30% vs. 50%)": toss_up_vs(0.30),
    "Kickoff under 40 F (vs. 55-75 F)": {"temp_bin=under_40": 1},
    "Kickoff 85 F or hotter (vs. 55-75 F)": {"temp_bin=85_plus": 1},
    "Rain, 0.1 in+ around kickoff": {"rain": 1},
    "Heavy rain, 0.5 in+ (added to rain)": {"heavy_rain": 1},
    "Wind 15 mph+": {"windy": 1},
    "Home opener": {"home_opener": 1},
    "October game (vs. September)": {"season_phase=october": 1},
    "November game (vs. September)": {"season_phase=november": 1},
}


def combine(params, cov, weights):
    """Estimate and standard error of a weighted sum of coefficients."""

    w = pd.Series(weights).reindex(params.index).fillna(0)
    return float(w @ params), float(np.sqrt(w @ cov @ w))


def effect_rows(name, params, cov, games, observed_share=1.0):
    """
    One row per effect. `observed_share` scales latent demand to the
    crowd actually reported (a Tobit effect only shows on games that
    weren't selling out anyway); it's 1 for the OLS models.
    """

    share = np.broadcast_to(observed_share, len(games))
    points_per_unit = 100 * share.mean()
    seats_per_unit = (games["capacity"] * share).mean()
    dollars_per_unit = (games["capacity"] * games["assumed_price"] * share).mean()

    rows = []
    for label, weights in EFFECTS.items():
        if not set(weights) <= set(params.index):
            continue
        estimate, se = combine(params, cov, weights)
        rows.append({
            "effect": label,
            "model": name,
            "pts_of_capacity": estimate * points_per_unit,
            "ci_low": (estimate - 1.96 * se) * points_per_unit,
            "ci_high": (estimate + 1.96 * se) * points_per_unit,
            "p_value": 2 * norm.sf(abs(estimate / se)),
            "seats_per_game": estimate * seats_per_unit,
            "dollars_per_game_at_assumed_price": estimate * dollars_per_unit,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------
# DATA
# --------------------------------------------------

df = models.prepare(pd.read_parquet(MODEL_TABLE))

# Kickoff-time effects need a known kickoff, and every model needs a crowd.
games = df[df["fill"].notna() & (df["kick_slot"] != "tbd")].copy()
games["team_season"] = games["home_team"] + " " + games["season"].astype(str)

print(f"{len(games):,} home games, {games['home_team'].nunique()} schools, "
      f"{games['sellout'].mean():.0%} sold out")


# --------------------------------------------------
# 1. OLS WITH TEAM AND SEASON FIXED EFFECTS
# --------------------------------------------------

main_spec = models.MAIN_SPEC

X, groups = models.design_matrix(games, main_spec)
ols = models.fit_ols(games["fill_capped"], X, games["home_team"])

ols_rows = effect_rows("OLS", ols.params, ols.cov_params(), games)


# --------------------------------------------------
# 2. TOBIT: DEMAND BEHIND SELLOUTS
# --------------------------------------------------

X_tobit, _ = models.design_matrix(games, main_spec, keep_all_levels=("home_team",))
tobit = models.Tobit().fit(games["fill"], X_tobit, games["home_team"])
tobit_cov = pd.DataFrame(tobit.cov[:-1, :-1], index=tobit.columns, columns=tobit.columns)

tobit_rows = effect_rows("Tobit (latent demand)", tobit.table["coef"], tobit_cov, games)
tobit_observed_rows = effect_rows(
    "Tobit (reported crowd)", tobit.table["coef"], tobit_cov, games,
    observed_share=norm.cdf((SELLOUT_THRESHOLD - tobit.latent_demand(X_tobit)) / tobit.sigma),
)

print(f"Tobit: sigma = {tobit.sigma:.3f} of capacity, {tobit.share_censored:.0%} of games censored")


# --------------------------------------------------
# 3. TWO-PART MODEL
# --------------------------------------------------
# Part one is a linear probability model: with team fixed effects, a
# logit would drop every school that always (or never) sells out.

sellout_lpm = models.fit_ols(games["sellout"], X, games["home_team"])
sellout_rows = effect_rows("Two-part: sellout chance", sellout_lpm.params, sellout_lpm.cov_params(), games)
sellout_rows = sellout_rows[["effect", "model", "pts_of_capacity", "ci_low", "ci_high", "p_value"]].rename(
    columns={"pts_of_capacity": "pts_of_probability"}
)

unsold = games[games["sellout"] == 0]
X_unsold, _ = models.design_matrix(unsold, main_spec)
X_unsold = X_unsold.loc[:, X_unsold.std() > 0].assign(const=1.0)
fill_if_unsold = models.fit_ols(unsold["fill_capped"], X_unsold, unsold["home_team"])
unsold_rows = effect_rows("Two-part: fill if not sold out", fill_if_unsold.params, fill_if_unsold.cov_params(), unsold)


# --------------------------------------------------
# 4. ROBUSTNESS
# --------------------------------------------------

# Team x season effects absorb anything about a school's year (a coaching
# change, a hot start, a price change), leaving only game-to-game
# differences inside one season. Last season's record is fixed within a
# school-season, so the fixed effects already contain it.
ts_spec = {**main_spec, "home_team_form": ["home_ranked", "home_winpct"]}
X_ts, _ = models.design_matrix(games, ts_spec, fixed_effects=("team_season",))
ols_ts = models.fit_ols(games["fill_capped"], X_ts, games["home_team"])
ts_rows = effect_rows("OLS, team x season FE", ols_ts.params, ols_ts.cov_params(), games)

X_elo, _ = models.design_matrix(games, models.ELO_SPEC)
ols_elo = models.fit_ols(games["fill_capped"], X_elo, games["home_team"])
elo_rows = effect_rows("OLS + Elo ratings", ols_elo.params, ols_elo.cov_params(), games)


effects = pd.concat([ols_rows, tobit_rows, tobit_observed_rows, unsold_rows, ts_rows, elo_rows])
effects.round(4).to_csv(OUT / "effects_all_models.csv", index=False)
sellout_rows.round(4).to_csv(OUT / "effects_sellout_probability.csv", index=False)

headline = ols_rows.merge(
    tobit_rows[["effect", "pts_of_capacity", "seats_per_game", "dollars_per_game_at_assumed_price"]],
    on="effect", suffixes=("", "_tobit_latent"),
)
headline.round(3).to_csv(OUT / "effects_headline.csv", index=False)

charts.effect_plot(
    pd.concat([ols_rows, tobit_rows]).rename(columns={"effect": "label", "pts_of_capacity": "estimate"}),
    OUT / "effects.png",
    "What moves a home crowd (same school, other things equal)",
    "Points of stadium capacity (line = 95% interval)",
)


# --------------------------------------------------
# 5. UNCERTAINTY OF OUTCOME
# --------------------------------------------------
# The hypothesis: fans want a contest, so demand peaks near a 50/50 game
# and falls as the result becomes a foregone conclusion. The rival idea
# is that home fans just want to see their team win, so demand keeps
# rising with the home win probability. A downward-bending curve with a
# peak inside the range supports the first; a still-rising one the second.

def peak(params, cov):
    """Win probability where demand tops out (-b1 / 2 b2), with a delta-method SE."""

    b1, b2 = params["home_win_prob"], params["home_win_prob_sq"]
    gradient = pd.Series({"home_win_prob": -1 / (2 * b2), "home_win_prob_sq": b1 / (2 * b2 ** 2)})
    sub = cov.loc[gradient.index, gradient.index]
    return -b1 / (2 * b2), float(np.sqrt(gradient @ sub @ gradient))


uoh = []
for name, params, cov in [
    ("OLS", ols.params, ols.cov_params()),
    ("OLS + Elo ratings", ols_elo.params, ols_elo.cov_params()),
    ("OLS, team x season FE", ols_ts.params, ols_ts.cov_params()),
    ("Tobit (latent demand)", tobit.table["coef"], tobit_cov),
]:
    top, top_se = peak(params, cov)
    curvature, curvature_se = params["home_win_prob_sq"], np.sqrt(cov.loc["home_win_prob_sq", "home_win_prob_sq"])
    uoh.append({
        "model": name,
        "curvature": curvature,
        "curvature_p_value": 2 * norm.sf(abs(curvature / curvature_se)),
        "peak_home_win_prob": top,
        "peak_ci_low": top - 1.96 * top_se,
        "peak_ci_high": top + 1.96 * top_se,
    })

uoh = pd.DataFrame(uoh)
uoh.round(4).to_csv(OUT / "uncertainty_of_outcome.csv", index=False)

# The same test without assuming a quadratic: spread buckets vs. a close game.
bucket_spec = models.SPREAD_BUCKET_SPEC
X_b, _ = models.design_matrix(games, bucket_spec)
ols_b = models.fit_ols(games["fill_capped"], X_b, games["home_team"])
buckets = pd.DataFrame({
    "spread_bucket": [c.split("=")[1] for c in X_b.columns if c.startswith("spread_bucket=")],
    "pts_of_capacity_vs_close_game": [100 * ols_b.params[c] for c in X_b.columns if c.startswith("spread_bucket=")],
    "se": [100 * ols_b.bse[c] for c in X_b.columns if c.startswith("spread_bucket=")],
    "games": [int(X_b[c].sum()) for c in X_b.columns if c.startswith("spread_bucket=")],
})
buckets.round(3).to_csv(OUT / "uncertainty_of_outcome_spread_buckets.csv", index=False)

grid = np.linspace(0.05, 0.98, 60)
curves = {}
for name, params, cov in [("OLS", ols.params, ols.cov_params()), ("Tobit (latent demand)", tobit.table["coef"], tobit_cov)]:
    W = np.column_stack([grid - 0.5, grid ** 2 - 0.25])
    b = params[["home_win_prob", "home_win_prob_sq"]].to_numpy()
    V = cov.loc[["home_win_prob", "home_win_prob_sq"], ["home_win_prob", "home_win_prob_sq"]].to_numpy()
    estimate = 100 * W @ b
    se = 100 * np.sqrt(np.einsum("ij,jk,ik->i", W, V, W))
    curves[name] = (estimate, estimate - 1.96 * se, estimate + 1.96 * se)

charts.curve_plot(
    grid, curves, OUT / "uncertainty_of_outcome.png",
    "Does a lopsided game cost fans?",
    "Home win probability implied by the point spread",
    "Attendance vs. a 50/50 game (pts of capacity)",
)


# --------------------------------------------------
# 6. DEMAND HIDDEN BEHIND SELLOUTS
# --------------------------------------------------
# For a sold-out game the Tobit gives E[demand | demand >= capacity]: how
# many people likely wanted in. Demand well above capacity on a regular
# basis is a sign tickets were priced below what the market would bear.

games["latent_demand"] = tobit.latent_demand(X_tobit)
games["demand_if_sold_out"] = np.where(games["sellout"] == 1, tobit.demand_given_sellout(X_tobit), np.nan)
games["unmet_seats"] = ((games["demand_if_sold_out"] - 1).clip(lower=0) * games["capacity"])
games["unmet_dollars_at_assumed_price"] = games["unmet_seats"] * games["assumed_price"]

# A school that sold out every game in the sample has no unsold game to
# anchor its level, so its excess demand reflects the prior, not data.
sellout_rate = games.groupby("home_team")["sellout"].mean()
always_sold_out = sorted(sellout_rate[sellout_rate == 1].index)
games["identified"] = ~games["home_team"].isin(always_sold_out)

by_school = (
    games[games["identified"]]
    .groupby("home_team")
    .agg(
        home_games=("game_id", "size"),
        sellout_rate=("sellout", "mean"),
        sellouts=("sellout", "sum"),
        avg_demand_when_sold_out=("demand_if_sold_out", "mean"),
        avg_unmet_seats_per_sellout=("unmet_seats", lambda s: s[s > 0].mean()),
        unmet_dollars_per_season_at_assumed_price=("unmet_dollars_at_assumed_price", "sum"),
    )
)
by_school["unmet_dollars_per_season_at_assumed_price"] /= games.groupby("home_team")["season"].nunique()
by_school = by_school[by_school["sellouts"] >= 5].sort_values("avg_demand_when_sold_out", ascending=False)
by_school.round(3).to_csv(OUT / "hidden_demand_by_school.csv")

top_games = (
    games[games["identified"] & (games["sellout"] == 1)]
    .nlargest(40, "demand_if_sold_out")
    [["season", "week", "home_team", "away_team", "kick_slot", "tv", "away_rank",
      "capacity", "attendance", "demand_if_sold_out", "unmet_seats", "unmet_dollars_at_assumed_price"]]
)
top_games.round(3).to_csv(OUT / "hidden_demand_top_games.csv", index=False)

focus = games[games["home_team"] == FOCUS_SCHOOL][
    ["season", "week", "away_team", "kick_slot", "tv", "away_rank", "home_win_prob", "capacity",
     "attendance", "fill", "sellout", "latent_demand", "demand_if_sold_out", "unmet_seats",
     "unmet_dollars_at_assumed_price"]
].sort_values(["season", "week"])
focus.round(3).to_csv(OUT / f"hidden_demand_{FOCUS_SCHOOL.lower()}.csv", index=False)

pd.Series(always_sold_out, name="home_team").to_csv(OUT / "always_sold_out_schools.csv", index=False)


# --------------------------------------------------
# SUMMARY
# --------------------------------------------------

show = headline.set_index("effect")[["pts_of_capacity", "ci_low", "ci_high", "seats_per_game",
                                      "dollars_per_game_at_assumed_price", "pts_of_capacity_tobit_latent"]]
print("\nEffects (OLS, team + season FE; last column = Tobit latent demand)")
print(show.round(1).to_string())
print("\nUncertainty of outcome")
print(uoh.round(3).to_string(index=False))
print(buckets.round(2).to_string(index=False))
print(f"\nSchools that sold out every game (excess demand not identified): {', '.join(always_sold_out) or 'none'}")
print(f"\nHidden demand, top schools with 5+ sellouts")
print(by_school.head(12).round(3).to_string())
print(f"\nAssumed ticket price: {ASSUMED_TICKET_PRICE} (not data; see ticketing/config.py)")
print(f"Saved to {OUT}")
