"""Model candidates, baselines, and a rolling-origin backtest for the PM2.5 forecast.

Several model types compete on the same backtest. The best one that beats the persistence
baseline is trained on all data and becomes the challenger.

Every candidate predicts log1p(PM2.5). The p10-p90 range comes from the same method for all
of them (split conformal): fit on the older 90% of the training data, measure the model's
errors on the newest 10%, then use the 10th/90th percentile of those errors as the range.
That keeps the comparison fair, since every model gets its range the same way.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import timedelta

import numpy as np
import pandas as pd
import polars as pl
from sklearn.base import RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from ml.features import CATEGORICAL_COLS, FEATURE_COLS
from shared.aqi import pm25_to_aqi

CALIBRATION_SHARE = 0.10  # newest share of training rows used to measure errors for the range


# --- Candidate models --------------------------------------------------------------------


def _numeric(cols: list[str]) -> list[str]:
    return [c for c in cols if c not in CATEGORICAL_COLS]


def _log_cols(cols: list[str]) -> list[str]:
    """Concentration columns are skewed; a linear model fits their log much better."""
    return [c for c in cols if c.startswith(("pm25_lag", "pm25_roll_mean", "pm10_"))]


def _log1p_nonneg(x):
    """log1p that tolerates bad negative inputs instead of producing NaN."""
    return np.log1p(np.clip(x, 0, None))


def make_hgb(cols: list[str]) -> RegressorMixin:
    """Gradient-boosted trees. Handles missing values and the continent category natively."""
    return HistGradientBoostingRegressor(
        loss="absolute_error",
        max_iter=400,
        learning_rate=0.05,
        categorical_features="from_dtype",
        random_state=0,
    )


def make_random_forest(cols: list[str]) -> RegressorMixin:
    """Bagged deep trees: a different bias than boosting, often robust on noisy data."""
    prep = ColumnTransformer(
        [
            ("num", SimpleImputer(strategy="median"), _numeric(cols)),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
        ]
    )
    forest = RandomForestRegressor(
        n_estimators=200, min_samples_leaf=5, max_features=0.5, n_jobs=-1, random_state=0
    )
    return make_pipeline(prep, forest)


def make_ridge(cols: list[str]) -> RegressorMixin:
    """Regularized linear regression: fast, explainable, and a sanity check for the trees."""
    log_cols = _log_cols(cols)
    other = [c for c in _numeric(cols) if c not in log_cols]
    prep = ColumnTransformer(
        [
            (
                "log",
                make_pipeline(
                    SimpleImputer(strategy="median"),
                    FunctionTransformer(_log1p_nonneg),
                    StandardScaler(),
                ),
                log_cols,
            ),
            ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), other),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
        ]
    )
    return make_pipeline(prep, Ridge(alpha=1.0))


CANDIDATES: dict[str, Callable[[list[str]], RegressorMixin]] = {
    "hgb": make_hgb,
    "random_forest": make_random_forest,
    "ridge": make_ridge,
}


# --- Fitted model ------------------------------------------------------------------------


@dataclass
class PM25Model:
    name: str
    estimator: RegressorMixin
    feature_cols: list[str]
    # 10th / 90th percentile of calibration errors, in log space
    resid_lo: float
    resid_hi: float
    meta: dict = field(default_factory=dict)

    def predict(self, df: pl.DataFrame) -> pd.DataFrame:
        X, _ = to_xy(df, self.feature_cols)
        log_pred = self.estimator.predict(X)
        return pd.DataFrame(
            {
                "pm25_pred": np.expm1(log_pred).clip(min=0),
                "pm25_p10": np.expm1(log_pred + min(self.resid_lo, 0)).clip(min=0),
                "pm25_p90": np.expm1(log_pred + max(self.resid_hi, 0)).clip(min=0),
            }
        )


def to_xy(df: pl.DataFrame, cols: list[str] = FEATURE_COLS) -> tuple[pd.DataFrame, np.ndarray]:
    X = df.select(cols).to_pandas()
    for c in CATEGORICAL_COLS:
        if c in X:
            X[c] = X[c].astype("category")
    y = np.log1p(df["target_pm25"].to_numpy()) if "target_pm25" in df.columns else None
    return X, y


def fit(train: pl.DataFrame, name: str = "hgb", cols: list[str] = FEATURE_COLS) -> PM25Model:
    """Fit one candidate, with a p10-p90 range from its errors on the newest training rows."""
    train = train.drop_nulls("target_pm25").sort("as_of_utc")
    n_cal = max(1, int(train.height * CALIBRATION_SHARE))
    early, cal = train.head(train.height - n_cal), train.tail(n_cal)

    X_early, y_early = to_xy(early, cols)
    X_cal, y_cal = to_xy(cal, cols)
    resid = y_cal - CANDIDATES[name](cols).fit(X_early, y_early).predict(X_cal)
    lo, hi = np.quantile(resid, [0.1, 0.9])

    X, y = to_xy(train, cols)
    estimator = CANDIDATES[name](cols).fit(X, y)
    return PM25Model(name, estimator, list(cols), float(lo), float(hi))


# --- Evaluation --------------------------------------------------------------------------


def score(actual: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    aqi_a = np.array([pm25_to_aqi(v) for v in actual])
    aqi_p = np.array([pm25_to_aqi(v) for v in pred])
    return {
        "mae": float(np.mean(np.abs(actual - pred))),
        "rmse": float(np.sqrt(np.mean((actual - pred) ** 2))),
        "aqi_acc": float(np.mean(aqi_a == aqi_p)),
    }


def backtest(
    features: pl.DataFrame,
    candidates: list[str] | None = None,
    cols: list[str] = FEATURE_COLS,
    n_folds: int = 4,
    test_days: int = 14,
) -> pl.DataFrame:
    """Expanding-window backtest. Never split randomly: the test set is always in the future.

    Returns one row per fold per model, including the two baselines.
    """
    candidates = candidates or list(CANDIDATES)
    data = features.drop_nulls("target_pm25").sort("as_of_utc")
    end = data["as_of_utc"].max()
    rows = []
    for k in range(n_folds, 0, -1):
        test_start = end - timedelta(days=test_days * k)
        test_end = test_start + timedelta(days=test_days)
        train = data.filter(pl.col("as_of_utc") < test_start)
        test = data.filter(pl.col("as_of_utc").is_between(test_start, test_end, closed="left"))
        if train.is_empty() or test.is_empty():
            continue

        actual = test["target_pm25"].to_numpy()
        fold = dict(fold=k, test_start=test_start, n_test=len(actual))

        rows.append(
            {"model": "persistence", **fold, **score(actual, test["pm25_lag_0"].to_numpy())}
        )
        city_mean_7 = test["pm25_roll_mean_7"].fill_null(test["pm25_lag_0"]).to_numpy()
        rows.append({"model": "city_mean_7", **fold, **score(actual, city_mean_7)})

        for name in candidates:
            preds = fit(train, name, cols).predict(test)
            covered = (actual >= preds["pm25_p10"]) & (actual <= preds["pm25_p90"])
            rows.append(
                {
                    "model": name,
                    **fold,
                    **score(actual, preds["pm25_pred"].to_numpy()),
                    "interval_coverage": float(covered.mean()),
                }
            )
    return pl.DataFrame(rows)


def summarize(results: pl.DataFrame) -> pl.DataFrame:
    """Mean metrics per model across folds, best first, with the persistence MAE to beat."""
    baseline_mae = results.filter(pl.col("model") == "persistence")["mae"].mean()
    return (
        results.group_by("model")
        .agg(pl.col("mae", "rmse", "aqi_acc", "interval_coverage").mean())
        .with_columns(pl.lit(baseline_mae).alias("baseline_mae"))
        .sort("mae")
    )


def passes_rule(m: dict) -> bool:
    """Beat persistence, and keep the p10-p90 range honest (should cover roughly 80%)."""
    coverage = m.get("interval_coverage")
    return m["mae"] < m["baseline_mae"] and coverage is not None and 0.70 <= coverage <= 0.90


def pick_best(summary: pl.DataFrame, candidates: list[str]) -> dict | None:
    """Lowest-MAE candidate that passes the rule, or None if none do."""
    for row in summary.filter(pl.col("model").is_in(candidates)).iter_rows(named=True):
        if passes_rule(row):
            return row
    return None


def should_promote(challenger: dict, champion: dict | None) -> bool:
    """Promotion rule from the plan: pass the rule and beat the champion by at least 2%."""
    if not passes_rule(challenger):
        return False
    return champion is None or challenger["mae"] <= 0.98 * champion["mae"]
