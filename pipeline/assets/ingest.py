"""Raw layer: download the Kaggle file and load it into raw.weather_air_snapshots.

Each Kaggle version holds the full history (one row per city per daily collection since
2026-04-24), so the raw table is replaced on every load and dbt models handle the rest.
The downloaded file is kept on disk per version so any load can be replayed.
"""

from datetime import UTC, datetime

import polars as pl
from dagster import (
    AssetCheckResult,
    AssetCheckSeverity,
    AssetExecutionContext,
    MaterializeResult,
    asset,
    asset_check,
)

from pipeline.resources import KaggleResource
from shared.config import get_settings
from shared.db import read_sql, write_table

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


@asset(group_name="ingest", compute_kind="postgres", deps=[raw_kaggle_file], key_prefix="raw")
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
    rows = write_table(df, "weather_air_snapshots", schema="raw", if_exists="replace")
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
