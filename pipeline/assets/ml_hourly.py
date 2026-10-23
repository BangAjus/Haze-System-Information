"""Hourly-track ML assets: 24h-ahead PM2.5 and PM10 from the last 72 hours.

Training (weekly) backtests hgb_lags, LSTM and GRU per pollutant and promotes the best;
forecasts run every hour after the OpenWeather collection, from each pollutant's champion.
"""

from datetime import UTC, datetime

import numpy as np
import polars as pl
from dagster import (
    AssetExecutionContext,
    AssetKey,
    AssetsDefinition,
    Config,
    MaterializeResult,
    asset,
)

from ml.hourly import (
    HORIZON,
    HOURLY_CANDIDATES,
    HourlyModel,
    anchors,
    backtest_hourly,
    build_hourly,
    fit_hourly,
)
from ml.registry import current_champion, load_model
from ml.train import summarize
from pipeline.assets.training import select_and_register
from shared.db import read_sql_copy, write_table

TRACK = "hourly"
FCT_AIR = AssetKey(["core", "fct_air_hourly"])
HISTORY_DAYS = 400  # a year of backfill plus margin


def load_hourly(days: int = HISTORY_DAYS):
    # Passed straight through (no local variable) so build_hourly can free it early
    return build_hourly(
        read_sql_copy(
            f"""
        select f.geonameid, f.hour_utc, f.pm2_5, f.pm10, f.no2, f.o3, f.co, f.so2, f.nh3, f."no",
               c.lat, c.lon, c.continent, c.population
        from core.fct_air_hourly f
        join core.dim_city c using (geonameid)
        where c.is_active and f.hour_utc > now() - interval '{int(days)} days'
        """
        )
    )


class HourlyTrainConfig(Config):
    """Which model types compete. LSTM/GRU take a few minutes each on CPU."""

    candidates: list[str] = list(HOURLY_CANDIDATES)


def make_hourly_model_asset(target: str) -> AssetsDefinition:
    @asset(
        name=f"hourly_{target}_model",
        group_name="ml_hourly",
        compute_kind="pytorch",
        deps=[FCT_AIR],
        key_prefix="ml",
    )
    def _model(context: AssetExecutionContext, config: HourlyTrainConfig) -> MaterializeResult:
        unknown = set(config.candidates) - set(HOURLY_CANDIDATES)
        if unknown:
            raise ValueError(f"Unknown candidates {sorted(unknown)}: {sorted(HOURLY_CANDIDATES)}")
        data = load_hourly()
        context.log.info(f"{len(data.seq):,} hourly rows, {int(data.valid.sum()):,} valid anchors")
        summary = summarize(backtest_hourly(data, target, config.candidates))

        def fit_winner(name: str):
            rows = anchors(data, None, None, stride=6)
            model = fit_hourly(data, rows, name, target)
            hours = data.hour[rows]
            return (
                model,
                hours.min().astype("datetime64[D]").item(),
                hours.max().astype("datetime64[D]").item(),
            )

        return select_and_register(
            context,
            track=TRACK,
            target=target,
            summary=summary,
            candidates=config.candidates,
            fit_winner=fit_winner,
            model_type_check=HourlyModel,
        )

    _model.__doc__ = f"Backtest hgb_lags, LSTM and GRU for {target} 24h ahead; promote the best."
    return _model


hourly_pm25_model = make_hourly_model_asset("pm25")
hourly_pm10_model = make_hourly_model_asset("pm10")


@asset(group_name="ml_hourly", compute_kind="pytorch", deps=[FCT_AIR], key_prefix="ml")
def forecasts_hourly(context: AssetExecutionContext) -> MaterializeResult:
    """24h-ahead PM2.5 and PM10 for every city with a complete last-72h window.

    Skips a pollutant (without failing) until its hourly champion exists.
    """
    data = load_hourly(days=7)  # only the newest windows are needed
    rows = data.latest
    issued_at = datetime.now(UTC)
    written, used = 0, {}
    for target in ("pm25", "pm10"):
        champion = current_champion(TRACK, target)
        model = champion and load_model(champion["model_version"], HourlyModel)
        if not model:
            context.log.warning(f"No loadable hourly {target} champion yet; skipping.")
            continue
        preds = model.predict(data, rows)
        hours = data.hour[rows].astype("datetime64[us]")
        out = pl.DataFrame(
            {
                "geonameid": data.geonameid[rows],
                "issued_at_utc": [issued_at] * len(rows),
                "target": [target] * len(rows),
                "horizon_h": np.full(len(rows), HORIZON, dtype=np.int16),
                "target_time_utc": pl.Series(
                    hours + np.timedelta64(HORIZON, "h")
                ).dt.replace_time_zone("UTC"),
                "pred": preds["pred"],
                "p10": preds["p10"],
                "p90": preds["p90"],
                "model_version": [champion["model_version"]] * len(rows),
            }
        )
        written += write_table(out, "forecasts_hourly", schema="ml", if_exists="append")
        used[f"{target}_model"] = champion["model_version"]
    return MaterializeResult(metadata={"rows_written": written, "cities": len(rows), **used})
