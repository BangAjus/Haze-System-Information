from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from ml.features import CATEGORICAL_COLS, FEATURE_COLS
from ml.train import CANDIDATES, backtest, fit, pick_best, should_promote, summarize


def make_features(n_cities: int = 4, n_days: int = 90, seed: int = 0) -> pl.DataFrame:
    """Synthetic feature table: tomorrow's PM2.5 is mostly today's, plus noise."""
    rng = np.random.default_rng(seed)
    start = datetime(2026, 5, 1, 21, tzinfo=UTC)
    rows = []
    for city in range(n_cities):
        level = rng.uniform(5, 40)
        for d in range(n_days):
            row = {c: float(rng.normal()) for c in FEATURE_COLS if c not in CATEGORICAL_COLS}
            row["pm25_lag_0"] = level * rng.uniform(0.7, 1.3)
            row["target_pm25"] = max(0.1, row["pm25_lag_0"] * 0.9 + rng.normal(0, 2))
            row.update(
                geonameid=city,
                as_of_utc=start + timedelta(days=d),
                continent=["AS", "EU"][city % 2],
                pm25_roll_mean_7=level,
            )
            rows.append(row)
    return pl.DataFrame(rows)


@pytest.mark.parametrize("name", list(CANDIDATES))
def test_every_candidate_fits_and_predicts(name):
    df = make_features()
    preds = fit(df, name).predict(df.head(20))
    assert len(preds) == 20
    assert (preds["pm25_pred"] >= 0).all()
    assert (preds["pm25_p10"] <= preds["pm25_pred"]).all()
    assert (preds["pm25_pred"] <= preds["pm25_p90"]).all()


def test_backtest_scores_every_model_and_baseline():
    results = backtest(make_features(), candidates=["ridge", "hgb"], n_folds=2, test_days=7)
    assert set(results["model"]) == {"persistence", "city_mean_7", "ridge", "hgb"}
    summary = summarize(results)
    assert summary["mae"].is_sorted()


def test_pick_best_skips_candidates_that_fail_the_rule():
    summary = pl.DataFrame(
        {
            "model": ["hgb", "ridge", "persistence"],
            "mae": [3.0, 3.5, 4.0],
            "interval_coverage": [0.40, 0.80, None],  # hgb's range is far too narrow
            "baseline_mae": [4.0, 4.0, 4.0],
        }
    )
    assert pick_best(summary, ["hgb", "ridge"])["model"] == "ridge"


def test_promotion_needs_a_2_percent_gain_over_the_champion():
    challenger = {"mae": 3.0, "baseline_mae": 4.0, "interval_coverage": 0.8}
    assert should_promote(challenger, None)
    assert should_promote(challenger, {"mae": 3.1})
    assert not should_promote(challenger, {"mae": 3.02})
