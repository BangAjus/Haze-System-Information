"""ML assets: weekly model training, daily forecasts and accuracy tracking.

The feature table ml.features_pm25 is a dbt model. The model logic lives in the ml/ package;
these assets only load inputs, call it and save outputs.
"""

import pickle
from datetime import UTC, datetime, timedelta

import polars as pl
from dagster import (
    AssetCheckResult,
    AssetExecutionContext,
    AssetKey,
    Config,
    FreshnessPolicy,
    MaterializeResult,
    asset,
    asset_check,
)
from sqlalchemy import text

from ml.train import CANDIDATES, PM25Model, backtest, fit, pick_best, should_promote, summarize
from shared.aqi import pm25_to_aqi_expr
from shared.config import get_settings
from shared.db import get_engine, read_sql, write_table

FCT = AssetKey(["core", "fct_observation"])
FEATURES = AssetKey(["ml", "features_pm25"])  # built by dbt: dbt/models/ml/features_pm25.sql


def _load_model(version: str) -> PM25Model | None:
    """Load a saved model, or None if it was saved by an incompatible older ml/train.py."""
    try:
        with open(_model_path(version), "rb") as f:
            model = pickle.load(f)
    except (FileNotFoundError, AttributeError, ModuleNotFoundError):
        return None
    return model if isinstance(model, PM25Model) and hasattr(model, "feature_cols") else None


def _model_path(version: str):
    path = get_settings().haze_data_dir / "models" / f"{version}.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


class TrainConfig(Config):
    """Which model types compete in the backtest. Editable in the Dagster launchpad."""

    candidates: list[str] = list(CANDIDATES)


@asset(group_name="ml", compute_kind="sklearn", deps=[FEATURES], key_prefix="ml")
def pm25_model(context: AssetExecutionContext, config: TrainConfig) -> MaterializeResult:
    """Backtest every candidate, train the best one on all data, promote it if it passes."""
    import mlflow

    unknown = set(config.candidates) - set(CANDIDATES)
    if unknown:
        raise ValueError(f"Unknown candidates {sorted(unknown)}. Options: {sorted(CANDIDATES)}")

    features = read_sql("select * from ml.features_pm25")
    summary = summarize(backtest(features, candidates=config.candidates))
    context.log.info(f"Backtest (mean of folds, best first):\n{summary}")

    trained_at = datetime.now(UTC)
    best = pick_best(summary, config.candidates)
    mlflow.set_tracking_uri(get_settings().mlflow_tracking_uri)
    mlflow.set_experiment("pm25_forecast")

    with mlflow.start_run(run_name="train-" + trained_at.strftime("%Y%m%d%H%M")) as parent:
        # One child run per candidate, so MLflow shows them side by side
        for row in summary.iter_rows(named=True):
            with mlflow.start_run(run_name=row["model"], nested=True):
                mlflow.log_metrics({k: v for k, v in row.items() if isinstance(v, float)})
        write_table(
            summary.with_columns(
                pl.lit(trained_at).alias("trained_at"),
                pl.lit(parent.info.run_id).alias("mlflow_run_id"),
                (pl.col("model") == (best or {}).get("model")).alias("selected"),
            ),
            "model_candidates",
            schema="ml",
            if_exists="append",
        )
        if best is None:
            mlflow.log_param("selected", "none")
            context.log.warning(
                "No candidate beat persistence with an honest range. Keeping champion."
            )
            return MaterializeResult(metadata={"selected": "none", "candidates": len(summary)})

        labelled = features.drop_nulls("target_pm25")
        model = fit(labelled, best["model"])
        version = f"pm25-{best['model'].replace('_', '')}-" + trained_at.strftime("%Y%m%d%H%M")
        with open(_model_path(version), "wb") as f:
            pickle.dump(model, f)

        champion = read_sql(
            "select model_version, mae from ml.model_runs where alias = 'champion' limit 1"
        )
        current = champion.row(0, named=True) if champion.height else None
        if current and _load_model(current["model_version"]) is None:
            # The champion can't be loaded by this code (older format), so it can't forecast
            context.log.warning(f"Champion {current['model_version']} is unloadable; replacing it.")
            current = None
        promote = should_promote(best, current)
        mlflow.log_params(
            {"selected": best["model"], "model_version": version, "promoted": promote}
        )
        mlflow.log_artifact(str(_model_path(version)))

    metrics = {k: best[k] for k in ("mae", "rmse", "aqi_acc", "baseline_mae", "interval_coverage")}
    with get_engine().begin() as conn:
        if promote:
            conn.execute(
                text("update ml.model_runs set alias = 'archived' where alias = 'champion'")
            )
        conn.execute(
            text(
                """
                insert into ml.model_runs (model_version, model_type, mlflow_run_id, trained_at,
                    train_start, train_end, mae, rmse, aqi_acc, baseline_mae, interval_coverage,
                    alias)
                values (:v, :type, :run, :trained_at, :start, :end, :mae, :rmse, :aqi_acc,
                    :baseline_mae, :interval_coverage, :alias)
                """
            ),
            dict(
                v=version,
                type=best["model"],
                run=parent.info.run_id,
                trained_at=trained_at,
                start=labelled["as_of_utc"].min().date(),
                end=labelled["as_of_utc"].max().date(),
                alias="champion" if promote else "rejected",
                **metrics,
            ),
        )
    return MaterializeResult(
        metadata={
            "model_version": version,
            "selected": best["model"],
            "promoted": promote,
            **metrics,
        }
    )


@asset(
    group_name="ml",
    compute_kind="sklearn",
    deps=[FEATURES, pm25_model],
    key_prefix="ml",
    # The apps show these; stale forecasts should be visible in Dagster and on the dashboard
    freshness_policy=FreshnessPolicy.time_window(
        warn_window=timedelta(hours=26), fail_window=timedelta(hours=50)
    ),
)
def forecasts() -> MaterializeResult:
    """Next-reading PM2.5 forecast for every city, from the champion model."""
    champion = read_sql("select model_version from ml.model_runs where alias = 'champion' limit 1")
    if champion.is_empty():
        raise RuntimeError("No champion model yet. Materialize ml/pm25_model first.")
    version = champion.item()
    model = _load_model(version)
    if model is None:
        raise RuntimeError(f"Can't load {version} (missing or older format). Run weekly_retrain.")

    latest = read_sql(
        """
        select distinct on (geonameid) *
        from ml.features_pm25
        -- Only cities with a recent reading; stale cities get no forecast
        where as_of_utc > (select max(as_of_utc) from ml.features_pm25) - interval '3 days'
        order by geonameid, as_of_utc desc
        """
    )
    preds = pl.from_pandas(model.predict(latest))
    issued_at = datetime.now(UTC)
    out = pl.concat([latest.select("geonameid", "as_of_utc"), preds], how="horizontal").select(
        "geonameid",
        pl.lit(issued_at).alias("issued_at_utc"),
        pl.lit(24, dtype=pl.Int16).alias("horizon_h"),
        (pl.col("as_of_utc") + pl.duration(hours=24)).alias("target_time_utc"),
        "pm25_pred",
        "pm25_p10",
        "pm25_p90",
        pm25_to_aqi_expr("pm25_pred").alias("aqi_pred"),
        pl.lit(version).alias("model_version"),
    )
    rows = write_table(out, "forecasts", schema="ml", if_exists="append")
    return MaterializeResult(metadata={"rows_written": rows, "model_version": version})


@asset_check(asset=forecasts)
def one_forecast_per_active_city() -> AssetCheckResult:
    df = read_sql(
        """
        select
          (select count(*) from core.dim_city where is_active) as active_cities,
          (select count(distinct geonameid) from ml.forecasts
            where issued_at_utc = (select max(issued_at_utc) from ml.forecasts)) as forecast_cities
        """
    )
    active, forecast = df.row(0)
    return AssetCheckResult(
        passed=forecast >= active,
        metadata={"active_cities": active, "forecast_cities": forecast},
    )


@asset(group_name="ml", compute_kind="postgres", deps=[forecasts, FCT], key_prefix="ml")
def forecast_accuracy() -> MaterializeResult:
    """Each forecast next to the reading that actually arrived (within 6h of the target)."""
    with get_engine().begin() as conn:
        conn.execute(
            text(
                """
                drop table if exists ml.forecast_accuracy;
                create table ml.forecast_accuracy as
                select f.geonameid, f.target_time_utc, f.model_version,
                       f.pm25_pred, o.pm2_5 as pm25_actual,
                       abs(f.pm25_pred - o.pm2_5) as abs_error,
                       f.aqi_pred, o.aqi as aqi_actual,
                       o.pm2_5 between f.pm25_p10 and f.pm25_p90 as in_interval
                from ml.forecasts f
                join lateral (
                    select pm2_5, aqi from core.fct_observation o
                    where o.geonameid = f.geonameid
                      and o.observed_at_utc between f.target_time_utc - interval '6 hours'
                                                and f.target_time_utc + interval '6 hours'
                    order by abs(extract(epoch from o.observed_at_utc - f.target_time_utc))
                    limit 1
                ) o on true;
                """
            )
        )
        n = conn.execute(text("select count(*) from ml.forecast_accuracy")).scalar_one()
    return MaterializeResult(metadata={"rows_written": n})
