"""Baselines, gradient-boosted PM2.5 model, and a rolling-origin backtest.

Uses scikit-learn's HistGradientBoostingRegressor (no system libraries needed on macOS).
Swap in LightGBM later if you want; the interface below stays the same.
"""

from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import pandas as pd
import polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor

from ml.features import CATEGORICAL_COLS, FEATURE_COLS
from shared.aqi import pm25_to_aqi

QUANTILES = {"p10": 0.1, "p90": 0.9}


@dataclass
class PM25Model:
    median: HistGradientBoostingRegressor
    quantiles: dict[str, HistGradientBoostingRegressor]

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=X.index)
        out["pm25_pred"] = np.expm1(self.median.predict(X)).clip(min=0)
        for name, m in self.quantiles.items():
            out[f"pm25_{name}"] = np.expm1(m.predict(X)).clip(min=0)
        # Quantile models are trained separately, so enforce p10 <= pred <= p90
        out["pm25_p10"] = out[["pm25_p10", "pm25_pred"]].min(axis=1)
        out["pm25_p90"] = out[["pm25_p90", "pm25_pred"]].max(axis=1)
        return out


def to_xy(df: pl.DataFrame) -> tuple[pd.DataFrame, np.ndarray | None]:
    X = df.select(FEATURE_COLS).to_pandas()
    for c in CATEGORICAL_COLS:
        X[c] = X[c].astype("category")
    y = np.log1p(df["target_pm25"].to_numpy()) if "target_pm25" in df.columns else None
    return X, y


def fit(train: pl.DataFrame) -> PM25Model:
    X, y = to_xy(train)
    common = dict(max_iter=400, learning_rate=0.05, categorical_features="from_dtype")
    median = HistGradientBoostingRegressor(loss="absolute_error", **common).fit(X, y)
    quantiles = {
        name: HistGradientBoostingRegressor(loss="quantile", quantile=q, **common).fit(X, y)
        for name, q in QUANTILES.items()
    }
    return PM25Model(median, quantiles)


def score(actual: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    aqi_a = np.array([pm25_to_aqi(v) for v in actual])
    aqi_p = np.array([pm25_to_aqi(v) for v in pred])
    return {
        "mae": float(np.mean(np.abs(actual - pred))),
        "rmse": float(np.sqrt(np.mean((actual - pred) ** 2))),
        "aqi_acc": float(np.mean(aqi_a == aqi_p)),
    }


def backtest(features: pl.DataFrame, n_folds: int = 4, test_days: int = 14) -> pl.DataFrame:
    """Expanding-window backtest. Never split randomly: the test set is always in the future."""
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
        model = fit(train)
        preds = model.predict(to_xy(test)[0])
        covered = (actual >= preds["pm25_p10"]) & (actual <= preds["pm25_p90"])
        persistence = test["pm25_lag_0"].to_numpy()
        city_mean_7 = test["pm25_roll_mean_7"].fill_null(test["pm25_lag_0"]).to_numpy()

        for name, pred in [
            ("model", preds["pm25_pred"].to_numpy()),
            ("persistence", persistence),
            ("city_mean_7", city_mean_7),
        ]:
            m = score(actual, pred)
            m.update(fold=k, model=name, test_start=test_start, n_test=len(actual))
            if name == "model":
                m["interval_coverage"] = float(covered.mean())
            rows.append(m)
    return pl.DataFrame(rows)


def summarize(results: pl.DataFrame) -> dict[str, float]:
    """Mean metrics across folds, plus the persistence baseline to beat."""
    by_model = results.group_by("model").agg(pl.col("mae", "rmse", "aqi_acc").mean())
    model = by_model.filter(pl.col("model") == "model").row(0, named=True)
    base = by_model.filter(pl.col("model") == "persistence").row(0, named=True)
    return {
        "mae": model["mae"],
        "rmse": model["rmse"],
        "aqi_acc": model["aqi_acc"],
        "baseline_mae": base["mae"],
        "interval_coverage": results.filter(pl.col("model") == "model")["interval_coverage"].mean(),
    }


def should_promote(challenger: dict, champion: dict | None) -> bool:
    """Promotion rule from the plan: beat persistence, keep intervals honest, beat the champion."""
    ok = challenger["mae"] < challenger["baseline_mae"] and (
        0.70 <= challenger["interval_coverage"] <= 0.90
    )
    if champion is None:
        return ok
    return ok and challenger["mae"] <= 0.98 * champion["mae"]
