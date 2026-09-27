"""The two models the project needs, and the baselines they have to beat.

Model 1 is a ridge regression fitted separately for each campaign, because the
rain a January reading responds to is not the rain an August reading responds
to, and separate fits make the coefficients readable: metres per 100 mm.

Model 2 is LightGBM, one global model over all wells and campaigns, with
monotone constraints so that more rain can never make it predict deeper water.
Without that, a what-if answer could come back saying extra rain lowers the
water table, which is nonsense and would discredit the simulator.

Model 3 is a small neural network, present as the benchmark the course asks for.

Baselines: "normal" predicts that every well sits at its own normal level, and
"last year" repeats the same campaign a year earlier. A model that cannot beat
the first has learnt nothing about rain. The second is a reference only: it uses
past water levels, which carry pumping history and are banned from the
attribution models.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import RANDOM_SEED
from .dataset import CATEGORICAL, FEATURES, MONOTONE, RAIN_FEATURES, WELL_FEATURES

ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0)
NUMERIC = RAIN_FEATURES + WELL_FEATURES


def _numeric_pipeline() -> Pipeline:
    return Pipeline([("fill", SimpleImputer(strategy="median")), ("scale", StandardScaler())])


def _preprocessor(numeric=NUMERIC, categorical=CATEGORICAL) -> ColumnTransformer:
    """Built from the columns actually present, so a smaller feature set still fits."""
    steps = []
    if list(numeric):
        steps.append(("num", _numeric_pipeline(), list(numeric)))
    if list(categorical):
        steps.append(("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), list(categorical)))
    return ColumnTransformer(steps)


def _columns_present(X) -> tuple[list[str], list[str]]:
    """Numeric and categorical inputs this frame carries; campaign is the grouping, not an input."""
    return ([c for c in NUMERIC if c in X.columns],
            [c for c in CATEGORICAL if c in X.columns and c != "campaign"])


class PerCampaignRidge(BaseEstimator, RegressorMixin):
    """One ridge per campaign, each choosing its own penalty by cross-validation."""

    def __init__(self, alphas=ALPHAS):
        self.alphas = alphas

    def fit(self, X: pd.DataFrame, y):
        y = np.asarray(y, dtype=float)
        self.numeric_, categorical = _columns_present(X)
        base = Pipeline([("prep", _preprocessor(self.numeric_, categorical)),
                         ("ridge", RidgeCV(alphas=list(self.alphas)))])
        # the pooled fit is both the model for a campaign with too little data and
        # the whole model when the target has no campaigns to split on
        self.fallback_ = clone(base).fit(X, y)
        self.models_ = {}
        if "campaign" in X.columns:
            for campaign, rows in X.groupby("campaign", observed=True).groups.items():
                idx = X.index.get_indexer(rows)
                if len(idx) >= 50:
                    self.models_[campaign] = clone(base).fit(X.iloc[idx], y[idx])
        self.campaigns_ = list(self.models_)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.models_ or "campaign" not in X.columns:
            return self.fallback_.predict(X)
        out = np.empty(len(X))
        for campaign, rows in X.groupby("campaign", observed=True).groups.items():
            idx = X.index.get_indexer(rows)
            model = self.models_.get(campaign, self.fallback_)
            out[idx] = model.predict(X.iloc[idx])
        return out

    def coefficients(self) -> pd.DataFrame:
        """Fitted response per campaign, in metres of water level per unit of the feature.

        Rain totals are reported per 100 mm and percentage-of-normal features per
        10 percentage points, so the numbers are readable side by side. Negative
        means the water stands shallower, which is what rain should do.
        """
        rows = []
        for campaign, model in (self.models_ or {"all": self.fallback_}).items():
            names = model[:-1].get_feature_names_out()
            scales = model["prep"].named_transformers_["num"]["scale"].scale_
            for name, coef in zip(names, model["ridge"].coef_):
                plain = name.split("__", 1)[1]
                unit, step = ("category", np.nan)
                if plain in self.numeric_:
                    per_unit = coef / scales[self.numeric_.index(plain)]
                    unit, step = ("per 10% of normal", 10.0) if plain.endswith("_pct") else ("per 100 mm", 100.0)
                    if plain in ("sy", "well_depth_m", "rain_normal_mm"):
                        unit, step = "per unit", 1.0
                    response = per_unit * step
                else:
                    response = np.nan
                rows.append({"campaign": campaign, "feature": plain, "standardised": coef,
                             "response_m": response, "unit": unit})
        return pd.DataFrame(rows)


class MonotoneLGBM(lgb.LGBMRegressor):
    """LightGBM that reads its monotone constraints off the columns it is handed.

    Two things have to be right. LightGBM wants one constraint per training
    column, so a shorter feature set needs a shorter vector, worked out at fit
    time. And the direction depends on which way the target is written: a depth
    gets *smaller* when rain arrives, a rise gets *bigger*. Forcing one sign on
    both makes the model fit backwards and it flatlines, which is exactly what
    happened before `wetter_is_higher_` existed.
    """

    wetter_is_higher_ = False  # default suits a depth target; clones keep it

    def fit(self, X, y, **kwargs):
        if hasattr(X, "columns"):
            flip = -1 if self.wetter_is_higher_ else 1
            self.set_params(monotone_constraints=[flip * MONOTONE.get(column, 0) for column in X.columns])
        return super().fit(X, y, **kwargs)


def lgbm(monotone: bool = True, wetter_is_higher: bool = False, **params):
    """Gradient boosting over every well at once, categoricals handled natively.

    wetter_is_higher says which way the target runs: False for a depth below
    ground (rain makes it smaller), True for a rise or a net gain (rain makes it
    larger). See targets.WETTER_IS_HIGHER.
    """
    settings = dict(
        objective="regression", n_estimators=600, learning_rate=0.05, num_leaves=63,
        min_child_samples=200, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        reg_lambda=1.0, random_state=RANDOM_SEED, n_jobs=-1, verbose=-1,
    )
    settings.update(params)
    if not monotone:
        return lgb.LGBMRegressor(**settings)
    settings["monotone_constraints_method"] = "advanced"
    model = MonotoneLGBM(**settings)
    model.wetter_is_higher_ = wetter_is_higher
    return model


def mlp(**params):
    """Small neural network benchmark on the same inputs."""
    settings = dict(hidden_layer_sizes=(64, 32), alpha=1e-3, learning_rate_init=5e-3, batch_size=512,
                    max_iter=60, early_stopping=True, n_iter_no_change=5, random_state=RANDOM_SEED)
    settings.update(params)
    return Pipeline([("prep", _preprocessor()), ("mlp", MLPRegressor(**settings))])


def build(name: str, **params):
    """Model by name: ridge, lgbm, lgbm_free (no monotone constraints), mlp."""
    builders = {
        "ridge": lambda: PerCampaignRidge(**params),
        "lgbm": lambda: lgbm(monotone=True, **params),
        "lgbm_free": lambda: lgbm(monotone=False, **params),
        "mlp": lambda: mlp(**params),
    }
    if name not in builders:
        raise ValueError(f"unknown model {name!r}; choose from {sorted(builders)}")
    return builders[name]()


def baseline_normal(table: pd.DataFrame) -> np.ndarray:
    """Every well sits at its own normal: anomaly zero."""
    return np.zeros(len(table))


def baseline_last_year(table: pd.DataFrame) -> np.ndarray:
    """The same campaign one year earlier; falls back to the normal when that reading is missing.

    Reference only: it uses past water levels, which carry pumping history.
    """
    previous = table[["well_uid", "campaign", "year", "anomaly_m"]].copy()
    previous["year"] = previous["year"] + 1
    merged = table[["well_uid", "campaign", "year"]].merge(
        previous, on=["well_uid", "campaign", "year"], how="left", suffixes=("", "_prev"))
    return merged["anomaly_m"].fillna(0.0).to_numpy()
