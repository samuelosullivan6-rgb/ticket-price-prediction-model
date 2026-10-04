"""
Step 4: can a gradient-boosted model predict a season it never saw?

Settings are chosen by leave-one-season-out validation on 2015-2024,
then the model is trained once on 2015-2024 and scored on 2025. It has
to beat three simpler forecasts to earn its place:

  last season     the school's average fill last season
  season so far   this season's average so far (last season before the opener)
  fixed effects   the step-3 regression, refit without 2025

Writes to results/predict/, including the validated settings that the
live weekly predictions reuse.
"""

import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import brier_score_loss, roc_auc_score

from ticketing import charts, models
from ticketing.config import HOLDOUT_SEASON, MODEL_TABLE, RESULTS_DIR


OUT = RESULTS_DIR / "predict"
OUT.mkdir(parents=True, exist_ok=True)

# A game is "at risk" when it's forecast at least this far below the
# school's typical crowd (its average fill last season).
AT_RISK_GAP = 0.05


df = models.prepare(pd.read_parquet(MODEL_TABLE))
df = df[df["fill"].notna()].copy()

train = df[df["season"] < HOLDOUT_SEASON]
test = df[df["season"] == HOLDOUT_SEASON].copy()
print(f"Train: {len(train):,} games (2015-{HOLDOUT_SEASON - 1}, no 2020).  Test: {len(test):,} games ({HOLDOUT_SEASON}).")


# --------------------------------------------------
# 1. CHOOSE SETTINGS (LEAVE ONE SEASON OUT)
# --------------------------------------------------
# Each candidate predicts every training season from the others. Holding
# out whole seasons mimics the real task: forecasting a season ahead.

GRID = {
    "learning_rate": [0.05],
    "max_iter": [250, 500],
    "max_leaf_nodes": [15, 31],
    "min_samples_leaf": [20, 60],
    "l2_regularization": [0.0, 1.0],
}

results = []
for values in itertools.product(*GRID.values()):
    params = dict(zip(GRID, values))
    errors = []
    for season in sorted(train["season"].unique()):
        fit_on, held = train[train["season"] != season], train[train["season"] == season]
        model = models.gbm_regressor(params).fit(models.gbm_frame(fit_on), fit_on["fill_capped"])
        errors.append((model.predict(models.gbm_frame(held)).clip(0, 1) - held["fill_capped"]).abs())
    results.append({**params, "cv_mae_pts": 100 * pd.concat(errors).mean()})

cv = pd.DataFrame(results).sort_values("cv_mae_pts")
cv.round(3).to_csv(OUT / "cv_grid.csv", index=False)

best = cv.iloc[0].drop("cv_mae_pts").to_dict()
best = {k: (int(v) if k in ("max_iter", "max_leaf_nodes", "min_samples_leaf") else float(v)) for k, v in best.items()}
models.GBM_PARAMS_FILE.write_text(json.dumps(best, indent=2))
print(f"Best settings (CV MAE {cv.iloc[0]['cv_mae_pts']:.2f} pts): {best}")


# --------------------------------------------------
# 2. FIT ON 2015-2024, PREDICT 2025
# --------------------------------------------------

gbm = models.fit_gbm(train, best)
test = test.join(models.predict_gbm(gbm, test))

# The live run can't see this season's attendance until CFBD posts it, so
# also score the model the live run actually uses until then.
without_to_date = [f for f in models.GBM_FEATURES if f != "team_fill_to_date"]
gbm_live = models.fit_gbm(train, best, features=without_to_date)
test["pred_fill_no_to_date"] = models.predict_gbm(gbm_live, test)["pred_fill"]

# Baselines. A school new to FBS has no history, so it gets the league average.
league_average = train["fill_capped"].mean()
test["pred_last_season"] = test["team_prev_fill"].fillna(league_average)
test["pred_season_so_far"] = test["team_fill_to_date"].fillna(test["pred_last_season"])

# The fixed-effects model has no 2025 season effect, so 2025 games borrow 2024's.
both = pd.concat([train, test.assign(season=HOLDOUT_SEASON - 1)])
X_all, _ = models.design_matrix(both, models.MAIN_SPEC)
X_train, X_test = X_all.iloc[: len(train)], X_all.iloc[len(train):]
keep = X_train.columns[X_train.std() > 0].union(["const"])
ols = models.fit_ols(train["fill_capped"], X_train[keep], train["home_team"])
known_school = test["home_team"].isin(train["home_team"])
test["pred_fixed_effects"] = np.where(
    known_school, (X_test[keep] @ ols.params).clip(0, 1).to_numpy(), test["pred_last_season"]
)


# --------------------------------------------------
# 3. SCORE
# --------------------------------------------------

def score(pred, actual, capacity):
    error = pred - actual
    return {
        "mae_pts_of_capacity": 100 * error.abs().mean(),
        "rmse_pts_of_capacity": 100 * np.sqrt((error ** 2).mean()),
        "mae_seats": (error.abs() * capacity).mean(),
        "r_squared": 1 - (error ** 2).sum() / ((actual - actual.mean()) ** 2).sum(),
    }


forecasts = {
    "Last season's average": "pred_last_season",
    "Season so far": "pred_season_so_far",
    "Fixed-effects regression": "pred_fixed_effects",
    "Gradient boosting": "pred_fill",
    "Gradient boosting, no fill so far": "pred_fill_no_to_date",
}
scores = pd.DataFrame(
    {name: score(test[col], test["fill_capped"], test["capacity"]) for name, col in forecasts.items()}
).T
scores.round(3).to_csv(OUT / f"holdout_{HOLDOUT_SEASON}_scores.csv")

# Sellout chance: is the probability calibrated and does it rank games well?
sellout = {
    "gbm_brier": brier_score_loss(test["sellout"], test["sellout_prob"]),
    "baseline_brier": brier_score_loss(test["sellout"], test["team_prev_sellout_rate"].fillna(train["sellout"].mean())),
    "gbm_auc": roc_auc_score(test["sellout"], test["sellout_prob"]),
    "band_coverage_10_90": test["fill_capped"].between(test["pred_fill_low"], test["pred_fill_high"]).mean(),
}

# The business question: does an at-risk flag pick out games that really
# came in well below normal?
norm = test["pred_last_season"]
actually_low = test["fill_capped"] - norm <= -AT_RISK_GAP
sellout["share_of_games_actually_low"] = actually_low.mean()
for tag, col in [("", "pred_fill"), ("no_fill_so_far_", "pred_fill_no_to_date")]:
    flagged = test[col] - norm <= -AT_RISK_GAP
    sellout.update({
        f"{tag}at_risk_flagged": int(flagged.sum()),
        f"{tag}at_risk_precision": actually_low[flagged].mean(),
        f"{tag}at_risk_recall": flagged[actually_low].mean(),
    })
pd.Series(sellout).round(4).to_csv(OUT / f"holdout_{HOLDOUT_SEASON}_sellout_and_risk.csv", header=["value"])

test[["game_id", "season", "week", "home_team", "away_team", "kick_slot", "tv", "capacity", "attendance",
      "fill_capped", "pred_fill", "pred_fill_low", "pred_fill_high", "sellout", "sellout_prob",
      "pred_last_season", "pred_season_so_far", "pred_fixed_effects"]
     ].round(4).to_csv(OUT / f"holdout_{HOLDOUT_SEASON}_predictions.csv", index=False)


# --------------------------------------------------
# 4. WHAT THE MODEL LEANS ON
# --------------------------------------------------
# Permutation importance: how much worse 2025 predictions get when one
# feature's values are shuffled. Measured on the test season, not training.

importance = permutation_importance(
    gbm["fill"], models.gbm_frame(test), test["fill_capped"],
    scoring="neg_mean_absolute_error", n_repeats=10, random_state=0,
)
importance = pd.DataFrame({
    "feature": models.GBM_FEATURES,
    "mae_increase_pts": 100 * importance.importances_mean,
    "sd": 100 * importance.importances_std,
}).sort_values("mae_increase_pts", ascending=False)
importance.round(4).to_csv(OUT / f"holdout_{HOLDOUT_SEASON}_feature_importance.csv", index=False)


# --------------------------------------------------
# 5. CHART
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(5.5, 5.5))
ax.plot([0.2, 1], [0.2, 1], color=charts.COLORS["neutral"], lw=0.8)
ax.scatter(test["pred_fill"], test["fill_capped"], s=9, alpha=0.5, color=charts.COLORS["ols"], lw=0)
ax.set_xlabel(f"Predicted fill rate (model never saw {HOLDOUT_SEASON})")
ax.set_ylabel("Actual fill rate (capped at 100%)")
ax.set_title(f"{HOLDOUT_SEASON} holdout: MAE {scores.loc['Gradient boosting', 'mae_pts_of_capacity']:.1f} pts of capacity",
             loc="left", fontsize=11)
fig.tight_layout()
fig.savefig(OUT / f"holdout_{HOLDOUT_SEASON}_predicted_vs_actual.png")
plt.close(fig)


print(f"\n{HOLDOUT_SEASON} holdout")
print(scores.round(2).to_string())
print()
print(pd.Series(sellout).round(3).to_string())
print("\nTop features (permutation importance on the holdout)")
print(importance.head(10).round(2).to_string(index=False))
print(f"\nSaved to {OUT}")
