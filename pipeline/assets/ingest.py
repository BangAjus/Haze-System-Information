"""Raw layer: download the Kaggle file and load it into raw.weather_air_snapshots.

Each Kaggle version holds the full history (one row per city per daily collection since
2026-04-24), so the raw table is replaced on every load and dbt models handle the rest.
The downloaded file is kept on disk per version so any load can be replayed.
"""

from datetime import UTC, datetime, timedelta

import polars as pl
from dagster import (
    AssetCheckResult,
    AssetCheckSeverity,
    AssetExecutionContext,
    FreshnessPolicy,
    MaterializeResult,
    asset,
    asset_check,
)

from pipeline.resources import KaggleResource
from shared.config import get_settings
from shared.db import read_sql, replace_rows

# Columns staging relies on. If the author renames any of them the load fails loudly.
REQUIRED_COLUMNS = {
    "geonameid",
    "city_name",
    "country_code",
    "country_name",
    "continent",
    "city_population",
    "geo_lat",
    "geo_lon",
    "weather_time_unix",
    "temp",
    "feels_like",
    "humidity",
    "pressure",
    "wind_speed",
    "wind_deg",
    "wind_gust",
    "clouds_all",
    "visibility",
    "rain_1h",
    "snow_1h",
    "is_daytime",
    "aqi",
    "pm2_5",
    "pm10",
    "co",
    "no",
    "no2",
    "o3",
    "so2",
    "nh3",
    "units_weather",
}


@asset(group_name="ingest", compute_kind="kaggle")
def raw_kaggle_file(context: AssetExecutionContext, kaggle: KaggleResource) -> MaterializeResult:
    """Latest Kaggle file, saved under data/raw/source=kaggle/ingest_date=YYYY-MM-DD/."""
    ingest_date = datetime.now(UTC).date().isoformat()
    dest = get_settings().raw_dir / "source=kaggle" / f"ingest_date={ingest_date}"
    path = kaggle.download(dest)
    context.log.info(f"Downloaded {path}")
    return MaterializeResult(
        metadata={"path": str(path), "bytes": path.stat().st_size, "ingest_date": ingest_date}
    )


# "Has the pipeline loaded anything lately?" Evaluated by the daemon even when no run happens,
# so a dead sensor or a dataset that stopped updating still shows up as stale.
DAILY_FRESHNESS = FreshnessPolicy.time_window(
    warn_window=timedelta(hours=26), fail_window=timedelta(hours=50)
)

# "Is the data itself recent?" The newest observation's age when a load finishes. Catches
# the author republishing old data, which the policy above can't see.
OBSERVATION_WARN_AGE = timedelta(hours=36)
OBSERVATION_ERROR_AGE = timedelta(hours=60)


@asset(
    group_name="ingest",
    compute_kind="postgres",
    deps=[raw_kaggle_file],
    key_prefix="raw",
    freshness_policy=DAILY_FRESHNESS,
)
def weather_air_snapshots(context: AssetExecutionContext) -> MaterializeResult:
    """The Kaggle file loaded as-is into raw.weather_air_snapshots, plus lineage columns."""
    latest = sorted(get_settings().raw_dir.glob("source=kaggle/ingest_date=*/*.parquet"))[-1]
    df = pl.read_parquet(latest)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Source file is missing columns: {sorted(missing)}")

    df = df.with_columns(
        pl.lit("kaggle").alias("_source"),
        pl.lit(str(latest)).alias("_file"),
        pl.lit(datetime.now(UTC)).alias("_ingested_at"),
        pl.lit(context.run_id).alias("_dagster_run_id"),
    )
    # Truncate and reload: dropping the table would fail, dbt views depend on it
    rows = replace_rows(df, "weather_air_snapshots", schema="raw")
    return MaterializeResult(metadata={"rows_written": rows, "source_file": str(latest)})


@asset_check(asset=weather_air_snapshots, blocking=True)
def raw_row_count_stable() -> AssetCheckResult:
    """The full-history file should only grow. A drop means a bad or truncated file."""
    df = read_sql(
        """
        select count(*) as n_rows,
               count(distinct (geonameid, weather_time_unix)) as n_keys
        from raw.weather_air_snapshots
        """
    )
    n_rows, n_keys = df.row(0)
    return AssetCheckResult(
        passed=n_rows > 0 and n_rows == n_keys,
        severity=AssetCheckSeverity.ERROR,
        metadata={"n_rows": n_rows, "duplicate_keys": n_rows - n_keys},
    )


@asset_check(asset=weather_air_snapshots)
def observations_are_recent() -> AssetCheckResult:
    """Warns at 36h and fails at 60h since the newest observation in the file."""
    newest = read_sql("select max(weather_time_unix) as t from raw.weather_air_snapshots").item()
    age = datetime.now(UTC) - datetime.fromtimestamp(newest, UTC)
    severity = AssetCheckSeverity.ERROR if age > OBSERVATION_ERROR_AGE else AssetCheckSeverity.WARN
    return AssetCheckResult(
        passed=age <= OBSERVATION_WARN_AGE,
        severity=severity,
        metadata={
            "newest_observation_utc": datetime.fromtimestamp(newest, UTC).isoformat(),
            "age_hours": round(age.total_seconds() / 3600, 1),
        },
    )
