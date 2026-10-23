"""Daily-track ML assets: per-pollutant model training, forecasts and accuracy tracking.

Feature tables ml.features_pm25 and ml.features_pm10 are dbt models. The model logic lives in
ml/train.py, and selection/promotion in pipeline/assets/training.py; these assets wire it up.
"""

from datetime import UTC, datetime, timedelta

import polars as pl
from dagster import (
    AssetCheckResult,
    AssetCheckSeverity,
    AssetExecutionContext,
    AssetKey,
    AssetsDefinition,
    Config,
    FreshnessPolicy,
    MaterializeResult,
    asset,
    asset_check,
)
from sqlalchemy import text

from ml.drift import psi
from ml.drift import status as drift_status
from ml.features import FEATURE_COLS
from ml.registry import current_champion, load_model
from ml.train import CANDIDATES, ForecastModel, backtest, fit, summarize
from pipeline.assets.training import select_and_register
from shared.aqi import BREAKPOINTS, to_aqi_expr
from shared.db import get_engine, read_sql, read_sql_copy, replace_rows, write_table

TARGETS = ["pm25", "pm10"]
TRACK = "daily"
FCT_AIR = AssetKey(["core", "fct_air_hourly"])


def features_key(target: str) -> AssetKey:
    """Built by dbt: dbt/models/ml/features_<target>.sql"""
    return AssetKey(["ml", f"features_{target}"])


class TrainConfig(Config):
    """Which model types compete in the backtest. Editable in the Dagster launchpad."""

    candidates: list[str] = list(CANDIDATES)


def make_model_asset(target: str) -> AssetsDefinition:
    @asset(
        name=f"{target}_model",
        group_name="ml",
        compute_kind="sklearn",
        deps=[features_key(target)],
        key_prefix="ml",
    )
    def _model(context: AssetExecutionContext, config: TrainConfig) -> MaterializeResult:
        unknown = set(config.candidates) - set(CANDIDATES)
        if unknown:
            raise ValueError(f"Unknown candidates {sorted(unknown)}. Options: {sorted(CANDIDATES)}")

        features = read_sql(f"select * from ml.features_{target}")
        summary = summarize(backtest(features, candidates=config.candidates, target=target))
        labelled = features.drop_nulls("target")

        def fit_winner(name: str):
            model = fit(labelled, name, target=target)
            return model, labelled["as_of_utc"].min().date(), labelled["as_of_utc"].max().date()

        return select_and_register(
            context,
            track=TRACK,
            target=target,
            summary=summary,
            candidates=config.candidates,
            fit_winner=fit_winner,
            model_type_check=ForecastModel,
        )

    _model.__doc__ = f"Backtest every candidate for {target}, train the best, maybe promote it."
    return _model


pm25_model = make_model_asset("pm25")
pm10_model = make_model_asset("pm10")


def _predict_latest(target: str) -> tuple[pl.DataFrame, str]:
    champion = current_champion(TRACK, target)
    if champion is None:
        raise RuntimeError(f"No {target} champion yet. Materialize ml/{target}_model first.")
    model = load_model(champion["model_version"], ForecastModel)
    if model is None:
        raise RuntimeError(f"Can't load {champion['model_version']}. Run weekly_retrain.")
    latest = read_sql(
        f"""
        select distinct on (geonameid) *
        from ml.features_{target}
        -- Only cities with a recent reading; stale cities get no forecast
        where as_of_utc > (select max(as_of_utc) from ml.features_{target}) - interval '3 days'
        order by geonameid, as_of_utc desc
        """
    )
    preds = pl.from_pandas(model.predict(latest))
    out = pl.concat([latest.select("geonameid", "as_of_utc"), preds], how="horizontal")
    return out, champion["model_version"]


@asset(
    group_name="ml",
    compute_kind="sklearn",
    deps=[features_key("pm25"), features_key("pm10"), pm25_model, pm10_model],
    key_prefix="ml",
    # The apps show these; stale forecasts should be visible in Dagster and on the dashboard
    freshness_policy=FreshnessPolicy.time_window(
        warn_window=timedelta(hours=26), fail_window=timedelta(hours=50)
    ),
)
def forecasts() -> MaterializeResult:
    """Next-reading PM2.5 and PM10 forecasts for every city, from each pollutant's champion.

    The forecast AQI is the worse of the two pollutants' levels.
    """
    pm25, pm25_version = _predict_latest("pm25")
    pm10, pm10_version = _predict_latest("pm10")
    issued_at = datetime.now(UTC)
    out = pm25.join(
        pm10.select(
            "geonameid",
            pl.col("pred").alias("pm10_pred"),
            pl.col("p10").alias("pm10_p10"),
            pl.col("p90").alias("pm10_p90"),
        ),
        on="geonameid",
        how="left",
    ).select(
        "geonameid",
        pl.lit(issued_at).alias("issued_at_utc"),
        pl.lit(24, dtype=pl.Int16).alias("horizon_h"),
        (pl.col("as_of_utc") + pl.duration(hours=24)).alias("target_time_utc"),
        pl.col("pred").alias("pm25_pred"),
        pl.col("p10").alias("pm25_p10"),
        pl.col("p90").alias("pm25_p90"),
        pl.max_horizontal(
            to_aqi_expr("pred", "pm25"), to_aqi_expr(pl.col("pm10_pred").fill_null(0), "pm10")
        ).alias("aqi_pred"),
        pl.lit(pm25_version).alias("model_version"),
        "pm10_pred",
        "pm10_p10",
        "pm10_p90",
        pl.lit(pm10_version).alias("pm10_model_version"),
    )
    rows = write_table(out, "forecasts", schema="ml", if_exists="append")
    return MaterializeResult(
        metadata={"rows_written": rows, "pm25_model": pm25_version, "pm10_model": pm10_version}
    )


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


def aqi_case_sql(col: str, pollutant: str) -> str:
    """SQL CASE for a pollutant's AQI level, from the same breakpoints as shared/aqi.py."""
    whens = " ".join(
        f"when {col} < {b} then {level}" for level, b in enumerate(BREAKPOINTS[pollutant], 1)
    )
    return f"(case {whens} else 5 end)"


def _accuracy_select(
    track: str, target: str, table: str, pred: str, lo: str, hi: str, version: str, where: str = ""
) -> str:
    actual = "pm2_5" if target == "pm25" else "pm10"
    return f"""
        select '{track}' as track, '{target}' as target, f.geonameid, f.target_time_utc,
               f.{version} as model_version, f.{pred} as pred, o.{actual} as actual,
               abs(f.{pred} - o.{actual}) as abs_error,
               o.{actual} between f.{lo} and f.{hi} as in_interval,
               {aqi_case_sql(f"f.{pred}", target)} as aqi_pred,
               {aqi_case_sql(f"o.{actual}", target)} as aqi_actual
        from {table} f
        join lateral (
            select pm2_5, pm10 from core.fct_air_hourly o
            where o.geonameid = f.geonameid
              and o.hour_utc between f.target_time_utc - interval '6 hours'
                                 and f.target_time_utc + interval '6 hours'
            order by abs(extract(epoch from o.hour_utc - f.target_time_utc))
            limit 1
        ) o on true
        where f.{pred} is not null {where}
    """


@asset(group_name="ml", compute_kind="postgres", deps=[forecasts, FCT_AIR], key_prefix="ml")
def forecast_accuracy() -> MaterializeResult:
    """Every past forecast (both tracks, both pollutants) next to the reading that arrived.

    One row per forecast; the actual is the hourly reading nearest the target time (±6h).
    """
    parts = [
        _accuracy_select(
            "daily", "pm25", "ml.forecasts", "pm25_pred", "pm25_p10", "pm25_p90", "model_version"
        ),
        _accuracy_select(
            "daily",
            "pm10",
            "ml.forecasts",
            "pm10_pred",
            "pm10_p10",
            "pm10_p90",
            "pm10_model_version",
        ),
        _accuracy_select(
            "hourly",
            "pm25",
            "ml.forecasts_hourly",
            "pred",
            "p10",
            "p90",
            "model_version",
            "and f.target = 'pm25'",
        ),
        _accuracy_select(
            "hourly",
            "pm10",
            "ml.forecasts_hourly",
            "pred",
            "p10",
            "p90",
            "model_version",
            "and f.target = 'pm10'",
        ),
    ]
    with get_engine().begin() as conn:
        conn.execute(text("drop table if exists ml.forecast_accuracy"))
        conn.execute(text("create table ml.forecast_accuracy as " + " union all ".join(parts)))
        counts = conn.execute(
            text("select track || '/' || target, count(*) from ml.forecast_accuracy group by 1")
        ).all()
    return MaterializeResult(metadata={k: v for k, v in counts} or {"rows_written": 0})


def _drift_rows(
    track: str, target: str, reference: pl.DataFrame, current: pl.DataFrame, features: list[str]
) -> list[dict]:
    today = datetime.now(UTC).date()
    out = []
    for f in features:
        value = psi(
            reference[f].cast(pl.Float64).to_numpy(), current[f].cast(pl.Float64).to_numpy()
        )
        out.append(
            dict(
                date_utc=today,
                track=track,
                target=target,
                feature=f,
                psi=value,
                status=drift_status(value),
            )
        )
    return out


@asset(group_name="ml", compute_kind="numpy", deps=[forecasts, FCT_AIR], key_prefix="ml")
def drift() -> MaterializeResult:
    """PSI of each input feature: last 7 days vs. the champion's training period."""
    rows = []
    for target in TARGETS:
        champ = read_sql(
            """
            select train_start, train_end from ml.model_runs
            where alias = 'champion' and track = 'daily' and target = :t
            """,
            t=target,
        )
        if champ.height:
            start, end = champ.row(0)
            feats = read_sql(f"select * from ml.features_{target}")
            ref = feats.filter(pl.col("as_of_utc").dt.date().is_between(start, end))
            cur = feats.filter(pl.col("as_of_utc") > pl.col("as_of_utc").max() - timedelta(days=7))
            # Calendar features shift by construction (a recent week is always one month),
            # and continent is categorical: neither says anything about data drift
            skip = {"continent", "dow", "month_sin", "month_cos"}
            numeric = [c for c in FEATURE_COLS if c not in skip]
            rows += _drift_rows("daily", target, ref, cur, numeric)

    # Hourly track inputs are the pollutant series themselves
    hourly_champ = read_sql(
        "select min(train_start), max(train_end) from ml.model_runs "
        "where alias = 'champion' and track = 'hourly'"
    )
    if hourly_champ.height and hourly_champ.row(0)[0] is not None:
        start, end = hourly_champ.row(0)
        cols = ["pm2_5", "pm10", "no2", "o3", "co", "so2", "nh3"]
        # Millions of rows: stream via COPY (read_sql's per-row tuples crashed the process)
        air = read_sql_copy(
            f"select hour_utc, {', '.join(cols)} from core.fct_air_hourly "
            "where hour_utc > now() - interval '400 days'"
        )
        ref = air.filter(pl.col("hour_utc").dt.date().is_between(start, end))
        cur = air.filter(pl.col("hour_utc") > pl.col("hour_utc").max() - timedelta(days=7))
        for target in TARGETS:
            rows += _drift_rows("hourly", target, ref, cur, cols)

    if not rows:
        return MaterializeResult(metadata={"skipped": "no champion yet"})
    df = pl.DataFrame(rows)
    replace_rows(df, "drift", "ml")
    worst = df.sort("psi", descending=True, nulls_last=True).row(0, named=True)
    return MaterializeResult(
        metadata={
            "features_checked": df.height,
            "drifted": int((df["status"] == "drift").sum()),
            "watch": int((df["status"] == "watch").sum()),
            "worst_feature": f"{worst['track']}/{worst['target']}/{worst['feature']}",
            "worst_psi": round(worst["psi"], 3) if worst["psi"] == worst["psi"] else None,
        }
    )


@asset_check(asset=drift)
def no_feature_drifted() -> AssetCheckResult:
    """Warns when any input feature's PSI reaches 0.25 (time to retrain or check the data)."""
    df = read_sql("select track, target, feature, psi from ml.drift where status = 'drift'")
    return AssetCheckResult(
        passed=df.is_empty(),
        severity=AssetCheckSeverity.WARN,
        metadata={"drifted": [f"{r[0]}/{r[1]}/{r[2]}={r[3]:.2f}" for r in df.rows()][:20]},
    )
