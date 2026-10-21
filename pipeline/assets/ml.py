"""ML assets: weekly model training, daily forecasts and accuracy tracking.

The feature table ml.features_pm25 is a dbt model. The model logic lives in the ml/ package;
these assets only load inputs, call it and save outputs.
"""

import pickle
from datetime import UTC, datetime

import polars as pl
from dagster import (
    AssetCheckResult,
    AssetExecutionContext,
    AssetKey,
    MaterializeResult,
    asset,
    asset_check,
)
from sqlalchemy import text

from ml.train import backtest, fit, should_promote, summarize, to_xy
from shared.aqi import pm25_to_aqi_expr
from shared.config import get_settings
from shared.db import get_engine, read_sql, write_table

FCT = AssetKey(["core", "fct_observation"])
FEATURES = AssetKey(["ml", "features_pm25"])  # built by dbt: dbt/models/ml/features_pm25.sql


def _model_path(version: str):
    path = get_settings().haze_data_dir / "models" / f"{version}.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


@asset(group_name="ml", compute_kind="sklearn", deps=[FEATURES], key_prefix="ml")
def pm25_model(context: AssetExecutionContext) -> MaterializeResult:
    """Backtest a challenger, fit it on all data, and promote it if it passes the rule."""
    import mlflow

    features = read_sql("select * from ml.features_pm25")
    results = backtest(features)
    metrics = summarize(results)
    context.log.info(f"Backtest: {metrics}")

    labelled = features.drop_nulls("target_pm25")
    model = fit(labelled)
    version = "pm25-hgb-" + datetime.now(UTC).strftime("%Y%m%d%H%M")
    with open(_model_path(version), "wb") as f:
        pickle.dump(model, f)

    champion = read_sql("select mae from ml.model_runs where alias = 'champion' limit 1")
    promote = should_promote(metrics, champion.row(0, named=True) if champion.height else None)

    mlflow.set_tracking_uri(get_settings().mlflow_tracking_uri)
    mlflow.set_experiment("pm25_forecast")
    with mlflow.start_run(run_name=version) as run:
        mlflow.log_metrics(metrics)
        mlflow.log_param("promoted", promote)
        mlflow.log_artifact(str(_model_path(version)))

    with get_engine().begin() as conn:
        if promote:
            conn.execute(
                text("update ml.model_runs set alias = 'archived' where alias = 'champion'")
            )
        conn.execute(
            text(
                """
                insert into ml.model_runs (model_version, mlflow_run_id, train_start, train_end,
                    mae, rmse, aqi_acc, baseline_mae, interval_coverage, alias)
                values (:v, :run, :start, :end, :mae, :rmse, :aqi_acc, :baseline_mae,
                    :interval_coverage, :alias)
                """
            ),
            dict(
                v=version,
                run=run.info.run_id,
                start=labelled["as_of_utc"].min().date(),
                end=labelled["as_of_utc"].max().date(),
                alias="champion" if promote else "rejected",
                **metrics,
            ),
        )
    return MaterializeResult(metadata={"model_version": version, "promoted": promote, **metrics})


@asset(group_name="ml", compute_kind="sklearn", deps=[FEATURES, pm25_model], key_prefix="ml")
def forecasts() -> MaterializeResult:
    """Next-reading PM2.5 forecast for every city, from the champion model."""
    champion = read_sql("select model_version from ml.model_runs where alias = 'champion' limit 1")
    if champion.is_empty():
        raise RuntimeError("No champion model yet. Materialize ml/pm25_model first.")
    version = champion.item()
    with open(_model_path(version), "rb") as f:
        model = pickle.load(f)

    latest = read_sql(
        """
        select distinct on (geonameid) *
        from ml.features_pm25
        -- Only cities with a recent reading; stale cities get no forecast
        where as_of_utc > (select max(as_of_utc) from ml.features_pm25) - interval '3 days'
        order by geonameid, as_of_utc desc
        """
    )
    preds = pl.from_pandas(model.predict(to_xy(latest)[0]))
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
