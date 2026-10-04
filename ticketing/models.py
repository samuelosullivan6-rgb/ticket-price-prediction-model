"""
The models: a fixed-effects regression for clean, explainable effects,
a Tobit regression that accounts for sellouts hiding demand, and a
gradient-boosted model built purely to predict.
"""

import json

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import optimize
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from ticketing.config import RESULTS_DIR, SELLOUT_THRESHOLD


# --------------------------------------------------
# SHARED PREPARATION
# --------------------------------------------------

def prepare(df):
    """Derived columns and gap-filling shared by every model."""

    df = df.copy()

    # Elo is missing for some lower-division opponents. Filling with the
    # typical value for their level keeps those games in the sample, and
    # the away_fcs flag still marks them.
    for side in ["home", "away"]:
        df[f"{side}_elo"] = df[f"{side}_elo"].fillna(df.groupby("away_fcs")[f"{side}_elo"].transform("median"))
        df[f"{side}_prev_winpct"] = df[f"{side}_prev_winpct"].fillna(0.5)

    df["home_win_prob_sq"] = df["home_win_prob"] ** 2
    df["log_distance"] = np.log1p(df["distance_mi"].fillna(df["distance_mi"].median()))
    df["same_state"] = df["same_state"].fillna(False)
    df["elo_home_z"] = (df["home_elo"] - 1500) / 300
    df["elo_away_z"] = (df["away_elo"] - 1500) / 300

    return df


# --------------------------------------------------
# REGRESSION DESIGN
# --------------------------------------------------
# Each group is a business question ("what is kickoff time worth?").
# Categorical terms are (column, reference level): every effect is read
# against that reference. For kickoff, "noon" means "noon compared with
# a night game".

MAIN_SPEC = {
    "kickoff": [("kick_slot", "night")],
    "day": [("day_type", "saturday")],
    "tv": [("tv", "stream")],
    "competitiveness": ["home_win_prob", "home_win_prob_sq"],
    "opponent": ["away_ranked", "away_power", "away_fcs", "conf_game", "log_distance", "same_state"],
    "home_team_form": ["home_ranked", "home_winpct", "home_prev_winpct"],
    "weather": [("temp_bin", "55_75"), "rain", "heavy_rain", "windy"],
    "calendar": ["home_opener", ("season_phase", "september")],
}

# Same model with the win probability swapped for spread buckets, so the
# uncertainty-of-outcome result doesn't depend on a quadratic shape.
SPREAD_BUCKET_SPEC = {**MAIN_SPEC, "competitiveness": [("spread_bucket", "close")]}

# Adds both teams' Elo ratings, so the competitiveness terms are compared
# within games of similar team quality.
ELO_SPEC = {**MAIN_SPEC, "team_quality": ["elo_home_z", "elo_away_z"]}


def design_matrix(df, spec, fixed_effects=("home_team", "season"), keep_all_levels=()):
    """
    Numeric matrix for a spec, plus a column -> group map.

    Team fixed effects compare each school with itself, so the effects
    say what a factor does to a given stadium, not that big programs
    draw big crowds. `keep_all_levels` keeps every level of a fixed
    effect (for the Tobit, whose prior pins them down instead).
    """

    columns, groups = {}, {}

    for group, terms in spec.items():
        for term in terms:
            if isinstance(term, tuple):
                name, reference = term
                for level in sorted(df[name].dropna().unique()):
                    if level != reference:
                        columns[f"{name}={level}"] = (df[name] == level).astype(float)
                        groups[f"{name}={level}"] = group
            else:
                columns[term] = df[term].astype(float)
                groups[term] = group

    X = pd.DataFrame(columns, index=df.index)

    # One level of each fixed effect is dropped to sit in the intercept.
    for fe in fixed_effects:
        dummies = pd.get_dummies(df[fe], prefix=fe, prefix_sep="=", drop_first=fe not in keep_all_levels, dtype=float)
        X = X.join(dummies)
        groups.update({c: "fixed_effect" for c in dummies.columns})

    X.insert(0, "const", 1.0)
    groups["const"] = "fixed_effect"

    return X, groups


def fit_ols(y, X, clusters):
    """OLS with standard errors clustered by school (games at one stadium aren't independent)."""

    return sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(clusters)[0]})


# --------------------------------------------------
# TOBIT (CENSORED) REGRESSION
# --------------------------------------------------

class Tobit:
    """
    Regression for demand that is only partly visible.

    A sellout reports capacity, not demand: the true number of people who
    wanted in is somewhere at or above it. Plain regression treats a
    sellout as "demand = capacity" and so understates every effect that
    pushes demand up. The Tobit uses a sellout only as "demand was at least
    this", and fits the latent demand behind both kinds of game.

    A school that sells out every game gives no upper bound at all, so its
    team effect would run off to infinity. Team effects therefore get a
    mild normal prior (sd = `prior_sd` of capacity): it barely moves
    schools with any unsold games and keeps the rest finite. Those
    always-sold-out schools are flagged downstream rather than ranked.
    """

    def __init__(self, limit=SELLOUT_THRESHOLD, prior_sd=0.25, prior_prefix="home_team="):
        self.limit = limit
        self.prior_sd = prior_sd
        self.prior_prefix = prior_prefix

    def fit(self, y, X, clusters):
        self.columns = list(X.columns)
        self.prior = np.array([c.startswith(self.prior_prefix) for c in self.columns] + [False])
        X, y = X.to_numpy(float), np.asarray(y, float)
        censored = y >= self.limit
        y = np.where(censored, self.limit, y)

        start_beta = np.linalg.lstsq(X, y, rcond=None)[0]
        start = np.append(start_beta, np.log(np.std(y - X @ start_beta)))

        result = optimize.minimize(
            lambda t: -self._loglik(t, X, y, censored),
            start,
            jac=lambda t: -self._gradient(t, X, y, censored),
            method="L-BFGS-B",
            options={"maxiter": 5000},
        )
        if not result.success:
            raise RuntimeError(f"Tobit did not converge: {result.message}")

        theta = result.x
        self.beta, self.sigma = theta[:-1], np.exp(theta[-1])
        self.loglik = self._loglik(theta, X, y, censored)
        self.share_censored = censored.mean()

        # Sandwich variance: the curvature of the likelihood (bread) around
        # the spread of each school's summed scores (meat), so errors are
        # clustered by school like the OLS.
        hessian = self._numeric_hessian(theta, X, y, censored)
        bread = np.linalg.inv(-hessian)
        cluster_scores = pd.DataFrame(self._scores(theta, X, y, censored)).groupby(np.asarray(clusters)).sum().to_numpy()
        g = len(cluster_scores)
        meat = cluster_scores.T @ cluster_scores * g / (g - 1)
        self.cov = bread @ meat @ bread

        se = np.sqrt(np.diag(self.cov))[:-1]
        self.table = pd.DataFrame({"coef": self.beta, "se": se}, index=self.columns)
        self.table["p_value"] = 2 * norm.sf(np.abs(self.table["coef"] / self.table["se"]))

        # How a factor moves the crowd actually reported, not latent demand:
        # it only shows up when the game wasn't going to sell out anyway.
        self.p_not_sold_out = norm.cdf((self.limit - X @ self.beta) / self.sigma).mean()

        return self

    def latent_demand(self, X):
        """Predicted demand as a share of capacity, ignoring the capacity ceiling."""

        return X.to_numpy(float) @ self.beta

    def demand_given_sellout(self, X):
        """Expected demand for games that did sell out: E[demand | demand >= limit]."""

        mean = self.latent_demand(X)
        a = (mean - self.limit) / self.sigma
        return mean + self.sigma * np.exp(norm.logpdf(a) - norm.logcdf(a))

    def _loglik(self, theta, X, y, censored):
        beta, sigma = theta[:-1], np.exp(theta[-1])
        mean = X @ beta
        z = (y - mean) / sigma
        data = np.where(censored, norm.logcdf((mean - self.limit) / sigma), norm.logpdf(z) - np.log(sigma)).sum()
        return data - 0.5 * np.sum((theta[self.prior] / self.prior_sd) ** 2)

    def _gradient(self, theta, X, y, censored):
        return self._scores(theta, X, y, censored).sum(axis=0) - np.where(self.prior, theta / self.prior_sd ** 2, 0)

    def _scores(self, theta, X, y, censored):
        """Per-game gradient of the log-likelihood (columns: betas, then log sigma)."""

        beta, sigma = theta[:-1], np.exp(theta[-1])
        mean = X @ beta
        z = (y - mean) / sigma
        a = (mean - self.limit) / sigma
        mills = np.exp(norm.logpdf(a) - norm.logcdf(a))

        d_mean = np.where(censored, mills, z) / sigma
        d_log_sigma = np.where(censored, -a * mills, z ** 2 - 1)

        return np.column_stack([X * d_mean[:, None], d_log_sigma])

    def _numeric_hessian(self, theta, X, y, censored, step=1e-5):
        k = len(theta)
        hessian = np.empty((k, k))
        for j in range(k):
            shift = np.zeros(k)
            shift[j] = step
            hessian[:, j] = (
                self._gradient(theta + shift, X, y, censored) - self._gradient(theta - shift, X, y, censored)
            ) / (2 * step)
        return (hessian + hessian.T) / 2


# --------------------------------------------------
# DRIVERS (WHY A GAME IS PREDICTED LOW)
# --------------------------------------------------

def drivers(coefs, X_new, X_history, teams_new, teams_history, groups):
    """
    Break a game's demand into factor groups, relative to that school's
    typical home game.

    Each group's contribution is coef x (this game's value - the school's
    average value). The fixed effects cancel, so what's left reads as
    "this game is down 4 points of capacity because of the noon kickoff
    and 2 more because of the weak opponent".
    """

    usable = [c for c in X_new.columns if groups.get(c) not in (None, "fixed_effect") and c in coefs.index]

    typical = X_history[usable].groupby(np.asarray(teams_history)).mean()
    baseline = typical.reindex(teams_new).fillna(X_history[usable].mean())
    baseline.index = X_new.index

    contributions = (X_new[usable] - baseline) * coefs[usable]
    return contributions.T.groupby(pd.Series(groups)[usable]).sum().T


# --------------------------------------------------
# GRADIENT-BOOSTED MODEL
# --------------------------------------------------
# No team identity on purpose. Each school is described by its own demand
# history (last season's fill, this season's fill so far), which carries
# over to seasons and schools the model never saw.

GBM_NUMERIC = [
    "kick_hour", "capacity", "spread", "home_win_prob",
    "home_rank", "away_rank", "home_elo", "away_elo",
    "home_winpct", "home_games_played", "home_prev_winpct", "away_winpct", "away_prev_winpct",
    "home_power", "away_power", "away_fcs", "conf_game", "distance_mi", "same_state",
    "week", "home_game_number", "late_season",
    "temp_f", "precip_in", "wind_mph", "dome",
    "team_prev_fill", "team_prev_sellout_rate", "team_fill_to_date",
    "season",
]
GBM_CATEGORICAL = ["kick_slot", "day_type", "tv"]
GBM_FEATURES = GBM_NUMERIC + GBM_CATEGORICAL

# Fixed category lists, so a week with no weeknight games still encodes
# "weeknight" the same way the training data did.
CATEGORY_LEVELS = {
    "kick_slot": ["noon", "afternoon", "night", "tbd"],
    "day_type": ["saturday", "weeknight", "sun_mon"],
    "tv": ["broadcast", "espn", "cable", "conference_net", "stream"],
}

# Starting point only. 04_test_predictive_model.py picks the real settings
# by leave-one-season-out validation and saves them for the live run.
GBM_PARAMS = {"learning_rate": 0.05, "max_iter": 400, "max_leaf_nodes": 15, "min_samples_leaf": 40, "l2_regularization": 1.0}
GBM_PARAMS_FILE = RESULTS_DIR / "predict" / "gbm_params.json"


def gbm_params():
    """Validated settings if step 4 has been run, the defaults otherwise."""

    if GBM_PARAMS_FILE.exists():
        return json.loads(GBM_PARAMS_FILE.read_text())
    return GBM_PARAMS


def gbm_frame(df, features=GBM_FEATURES):
    """Feature table for the boosted models. Missing values stay NaN, which the model handles natively."""

    X = df[[f for f in features if f in GBM_NUMERIC]].astype(float)
    for col in [f for f in features if f in GBM_CATEGORICAL]:
        X[col] = pd.Categorical(df[col], categories=CATEGORY_LEVELS[col])
    return X


def fit_gbm(df, params=None, features=GBM_FEATURES):
    """
    Four boosted models on one feature set: the expected fill rate, the
    10th and 90th percentiles around it, and the chance of a sellout.
    """

    X, y = gbm_frame(df, features), df["fill_capped"]
    params = params or gbm_params()

    return {
        "features": list(features),
        "fill": gbm_regressor(params).fit(X, y),
        "low": gbm_regressor(params, loss="quantile", quantile=0.10).fit(X, y),
        "high": gbm_regressor(params, loss="quantile", quantile=0.90).fit(X, y),
        "sellout": HistGradientBoostingClassifier(**params, categorical_features="from_dtype", random_state=0)
        .fit(X, df["sellout"].astype(int)),
    }


def gbm_regressor(params, **loss):
    return HistGradientBoostingRegressor(**params, **loss, categorical_features="from_dtype", random_state=0)


def predict_gbm(models, df):
    X = gbm_frame(df, models["features"])
    return pd.DataFrame({
        "pred_fill": models["fill"].predict(X).clip(0, 1),
        "pred_fill_low": models["low"].predict(X).clip(0, 1),
        "pred_fill_high": models["high"].predict(X).clip(0, 1),
        "sellout_prob": models["sellout"].predict_proba(X)[:, 1],
    }, index=df.index)
