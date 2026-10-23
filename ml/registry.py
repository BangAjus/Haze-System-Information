"""Saving models, finding champions, and recording training runs.

Shared by both forecast tracks:
  - daily:  next-reading forecast from the daily feature tables (features_pm25/pm10)
  - hourly: 24h-ahead forecast from hourly sequences (LSTM, GRU, gradient boosting)
Each (track, target) pair has its own champion in ml.model_runs.
"""

import pickle
from datetime import datetime
from pathlib import Path

import polars as pl
from sqlalchemy import text

from shared.config import get_settings
from shared.db import get_engine, read_sql, write_table

# ------ PHASE 0: SETTINGS --------

METRIC_KEYS = ("mae", "rmse", "aqi_acc", "baseline_mae", "interval_coverage")


# ------ PHASE 1: MODEL FILES --------


def model_path(version: str) -> Path:
    path = get_settings().haze_data_dir / "models" / f"{version}.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_model(model, version: str) -> Path:
    path = model_path(version)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    return path


def load_model(version: str, expected_type: type, required_attr: str = "feature_cols"):
    """Load a saved model, or None if it's missing or saved by an incompatible older version."""
    try:
        with open(model_path(version), "rb") as f:
            model = pickle.load(f)
    except (FileNotFoundError, AttributeError, ModuleNotFoundError, EOFError):
        return None
    ok = isinstance(model, expected_type) and hasattr(model, required_attr)
    return model if ok else None


# ------ PHASE 2: CHAMPIONS AND RUN HISTORY --------


def current_champion(track: str, target: str) -> dict | None:
    df = read_sql(
        """
        select model_version, model_type, mae from ml.model_runs
        where alias = 'champion' and track = :track and target = :target
        order by trained_at desc limit 1
        """,
        track=track,
        target=target,
    )
    return df.row(0, named=True) if df.height else None


def log_candidates(
    summary: pl.DataFrame,
    track: str,
    target: str,
    trained_at: datetime,
    mlflow_run_id: str | None,
    selected: str | None,
) -> None:
    cols = ["model", "mae", "rmse", "aqi_acc", "interval_coverage", "baseline_mae"]
    write_table(
        summary.select([c for c in cols if c in summary.columns]).with_columns(
            pl.lit(trained_at).alias("trained_at"),
            pl.lit(track).alias("track"),
            pl.lit(target).alias("target"),
            (pl.col("model") == (selected or "")).alias("selected"),
            pl.lit(mlflow_run_id).alias("mlflow_run_id"),
        ),
        "model_candidates",
        schema="ml",
        if_exists="append",
    )


def record_run(
    *,
    version: str,
    model_type: str,
    track: str,
    target: str,
    mlflow_run_id: str | None,
    trained_at: datetime,
    train_start,
    train_end,
    metrics: dict,
    promote: bool,
) -> None:
    """Insert the run into ml.model_runs; if promoted, archive the previous champion."""
    with get_engine().begin() as conn:
        # job 1: if promoted, archive the current champion for this track and target
        if promote:
            conn.execute(
                text(
                    """
                    update ml.model_runs set alias = 'archived'
                    where alias = 'champion' and track = :track and target = :target
                    """
                ),
                dict(track=track, target=target),
            )
        # job 2: record this run
        conn.execute(
            text(
                """
                insert into ml.model_runs (model_version, model_type, track, target,
                    mlflow_run_id, trained_at, train_start, train_end, mae, rmse, aqi_acc,
                    baseline_mae, interval_coverage, alias)
                values (:v, :type, :track, :target, :run, :trained_at, :start, :end, :mae,
                    :rmse, :aqi_acc, :baseline_mae, :interval_coverage, :alias)
                """
            ),
            dict(
                v=version,
                type=model_type,
                track=track,
                target=target,
                run=mlflow_run_id,
                trained_at=trained_at,
                start=train_start,
                end=train_end,
                alias="champion" if promote else "rejected",
                **{k: metrics.get(k) for k in METRIC_KEYS},
            ),
        )
