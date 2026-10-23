"""Hourly collection straight from OpenWeather, for every active city.

Rows use the same column names as the Kaggle file, so dbt can union the two sources.
The observation time is the Air Pollution API's timestamp (hourly), since PM2.5 is what
the model predicts; the weather timestamp is kept as weather_time_unix.

Free tier budget (checked 2026-10-22): 60 calls/min and 1,000,000 calls/month.
584 cities x 2 APIs x 24 runs/day is about 869k calls in a 31-day month, so each run first
checks this month's usage and skips if it would pass MONTHLY_CALL_BUDGET.
"""

from datetime import UTC, datetime, timedelta

import polars as pl
from dagster import (
    AssetCheckResult,
    AssetCheckSeverity,
    AssetExecutionContext,
    Config,
    FreshnessPolicy,
    MaterializeResult,
    asset,
    asset_check,
)

from pipeline.resources import OpenWeatherResource
from shared.config import get_settings
from shared.db import insert_new_rows, read_sql, write_table

# ------ PHASE 0: SETTINGS --------

CALLS_PER_CITY = 2  # Current Weather + Air Pollution
MONTHLY_CALL_BUDGET = 950_000  # free tier is 1,000,000; keep a margin for retries and tests
MIN_SUCCESS_SHARE = 0.98
POLLUTANTS = ["pm2_5", "pm10", "co", "no", "no2", "o3", "so2", "nh3"]


class CollectConfig(Config):
    """Set max_cities for a quick test run instead of all active cities."""

    max_cities: int | None = None


# ------ PHASE 1: HELPERS (BUDGET + TRANSFORM) --------


def _calls_this_month() -> int:
    """Approximate calls made this month: two per row collected (retries not counted)."""

    # job 1: count this month's collected rows, two calls each
    rows = read_sql(
        """
        select count(*) as n from raw.openweather_snapshots
        where _ingested_at >= date_trunc('month', now())
        """
    ).item()
    return rows * CALLS_PER_CITY


def to_row(city: dict, weather: dict, air: dict) -> dict:
    """Flatten one city's two API responses into the Kaggle file's column names."""

    # job 1: pick out the nested parts of both responses
    reading = air["list"][0]
    comp = reading["components"]
    sun = weather.get("sys", {})
    wind = weather.get("wind", {})

    # job 2: one flat row with the Kaggle file's column names
    return {
        "geonameid": city["geonameid"],
        "city_name": city["city_name"],
        "country_code": city["country_code"],
        "country_name": city["country_name"],
        "continent": city["continent"],
        "city_population": city["population"],
        "geo_lat": city["lat"],
        "geo_lon": city["lon"],
        "weather_time_unix": weather["dt"],
        "air_time_unix": reading["dt"],
        "temp": weather["main"]["temp"],
        "feels_like": weather["main"].get("feels_like"),
        "humidity": weather["main"].get("humidity"),
        "pressure": weather["main"].get("pressure"),
        "wind_speed": wind.get("speed"),
        "wind_deg": wind.get("deg"),
        "wind_gust": wind.get("gust"),
        "clouds_all": weather.get("clouds", {}).get("all"),
        "visibility": weather.get("visibility"),
        "rain_1h": weather.get("rain", {}).get("1h", 0.0),  # key absent when it isn't raining
        "snow_1h": weather.get("snow", {}).get("1h", 0.0),
        "is_daytime": int(sun.get("sunrise", 0) <= weather["dt"] < sun.get("sunset", 0)),
        "aqi": reading["main"]["aqi"],
        "pm2_5": comp.get("pm2_5"),
        "pm10": comp.get("pm10"),
        "co": comp.get("co"),
        "no": comp.get("no"),
        "no2": comp.get("no2"),
        "o3": comp.get("o3"),
        "so2": comp.get("so2"),
        "nh3": comp.get("nh3"),
    }


# ------ PHASE 2: HOURLY COLLECTION (EXTRACT + LOAD) --------


@asset(
    group_name="ingest_hourly",
    compute_kind="openweather",
    key_prefix="raw",
    # Reads its city list from core.dim_city, but doesn't declare it as a dependency:
    # dim_city is built from these rows too, so declaring it would make a loop.
    freshness_policy=FreshnessPolicy.time_window(
        warn_window=timedelta(hours=2), fail_window=timedelta(hours=6)
    ),
)
def openweather_snapshots(
    context: AssetExecutionContext, openweather: OpenWeatherResource, config: CollectConfig
) -> MaterializeResult:
    """Current weather and air pollution for every active city, appended each hour."""

    # job 1: city list (all active cities, or the first max_cities for a test run)
    cities = read_sql(
        """
        select geonameid, city_name, country_code, country_name, continent, population,
               lat, lon
        from core.dim_city where is_active order by population desc
        """
    ).to_dicts()
    if config.max_cities:
        cities = cities[: config.max_cities]

    # job 2: skip the run if it would pass the monthly call budget
    used = _calls_this_month()
    needed = len(cities) * CALLS_PER_CITY
    if used + needed > MONTHLY_CALL_BUDGET:
        context.log.warning(
            f"Skipping: {used:,} calls used this month, budget {MONTHLY_CALL_BUDGET:,}"
        )
        return MaterializeResult(metadata={"skipped": "monthly budget", "calls_this_month": used})

    # job 3: extract weather + air pollution per city (paced under 60 calls/min)
    collected_at = datetime.now(UTC)
    results, failures = openweather.fetch(cities, log=context.log)
    for f in failures[:5]:
        context.log.warning(f"City {f['geonameid']} failed: {f['error']}")

    # job 4: transform responses into rows with lineage columns
    rows = [to_row(r["city"], r["weather"], r["air"]) for r in results]
    df = pl.DataFrame(rows).with_columns(
        pl.lit("openweather").alias("_source"),
        pl.lit(collected_at).alias("_ingested_at"),
        pl.lit(context.run_id).alias("_dagster_run_id"),
    )

    # job 5: keep the rows on disk too, so any hour can be replayed
    raw_dir = (
        get_settings().raw_dir
        / "source=openweather"
        / f"ingest_date={collected_at.date().isoformat()}"
    )
    raw_dir.mkdir(parents=True, exist_ok=True)
    df.write_parquet(raw_dir / f"hour={collected_at:%H}.parquet")

    # job 6: load (append) into raw.openweather_snapshots
    written = write_table(df, "openweather_snapshots", schema="raw", if_exists="append")
    return MaterializeResult(
        metadata={
            "rows_written": written,
            "cities_requested": len(cities),
            "cities_failed": len(failures),
            "success_share": round(len(results) / max(len(cities), 1), 4),
            "calls_this_month": used + needed,
            "monthly_budget": MONTHLY_CALL_BUDGET,
        }
    )


# ------ PHASE 3: QUALITY CHECK --------


@asset_check(asset=openweather_snapshots)
def most_cities_collected() -> AssetCheckResult:
    """At least 98% of active cities in the latest collection."""

    # job 1: cities in the latest collection vs. active cities
    df = read_sql(
        """
        select
          (select count(distinct geonameid) from raw.openweather_snapshots
            where _ingested_at = (select max(_ingested_at) from raw.openweather_snapshots))
            as collected,
          (select count(*) from core.dim_city where is_active) as active
        """
    )
    collected, active = df.row(0)
    share = collected / max(active, 1)

    # job 2: warn below 98%
    return AssetCheckResult(
        passed=share >= MIN_SUCCESS_SHARE,
        severity=AssetCheckSeverity.WARN,
        metadata={"collected": collected, "active_cities": active, "share": round(share, 4)},
    )


# ------ PHASE 4: HISTORY BACKFILL (EXTRACT + LOAD) --------


class BackfillConfig(Config):
    """How far back to fetch. One API call per city covers the whole window (up to a year)."""

    days: int = 365
    max_cities: int | None = None


@asset(group_name="backfill", compute_kind="openweather", key_prefix="raw")
def openweather_air_history(
    context: AssetExecutionContext, openweather: OpenWeatherResource, config: BackfillConfig
) -> MaterializeResult:
    """Hourly pollution history per active city, for the hourly forecast models.

    Safe to re-run: rows already stored (same city and hour) are skipped. Uses one call per
    city, so a full backfill of ~584 cities costs ~600 calls of the 1M monthly budget.
    """
    import httpx

    # job 1: city list and the time window to fetch
    cities = read_sql(
        "select geonameid, lat, lon from core.dim_city where is_active order by population desc"
    ).to_dicts()
    if config.max_cities:
        cities = cities[: config.max_cities]

    end = int(datetime.now(UTC).timestamp())
    start = end - config.days * 86400
    inserted, fetched, failed, batch = 0, 0, [], []

    # job 2: load helper, inserting only (city, hour) rows not stored yet
    def flush():
        nonlocal inserted, batch
        if batch:
            df = pl.DataFrame(batch).with_columns(
                # Stamped per batch, so dbt's incremental model sees each batch as new
                pl.lit(datetime.now(UTC)).alias("_ingested_at"),
                pl.lit(context.run_id).alias("_dagster_run_id"),
            )
            inserted += insert_new_rows(
                df, "openweather_air_history", "raw", key=["geonameid", "air_time_unix"]
            )
            batch = []

    # job 3: extract a year of hourly readings per city, loading every 25 cities
    with httpx.Client(timeout=60) as client:
        for i, city in enumerate(cities, 1):
            try:
                readings = openweather.air_history(client, city["lat"], city["lon"], start, end)
            except PermissionError:
                raise
            except Exception as exc:
                failed.append(city["geonameid"])
                context.log.warning(f"City {city['geonameid']} failed: {exc}")
                continue
            fetched += len(readings)
            batch += [
                {"geonameid": city["geonameid"], "air_time_unix": r["dt"], "aqi": r["main"]["aqi"]}
                | {k: r["components"].get(k) for k in POLLUTANTS}
                for r in readings
            ]
            if i % 25 == 0:
                flush()
                context.log.info(f"{i}/{len(cities)} cities, {inserted:,} new rows")
    flush()

    # job 4: report what came in
    return MaterializeResult(
        metadata={
            "cities": len(cities),
            "cities_failed": len(failed),
            "readings_fetched": fetched,
            "rows_inserted": inserted,
            "window_days": config.days,
        }
    )
