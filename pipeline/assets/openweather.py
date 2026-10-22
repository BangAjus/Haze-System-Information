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
from shared.db import read_sql, write_table

CALLS_PER_CITY = 2  # Current Weather + Air Pollution
MONTHLY_CALL_BUDGET = 950_000  # free tier is 1,000,000; keep a margin for retries and tests
MIN_SUCCESS_SHARE = 0.98


class CollectConfig(Config):
    """Set max_cities for a quick test run instead of all active cities."""

    max_cities: int | None = None


def _calls_this_month() -> int:
    """Approximate calls made this month: two per row collected (retries not counted)."""
    rows = read_sql(
        """
        select count(*) as n from raw.openweather_snapshots
        where _ingested_at >= date_trunc('month', now())
        """
    ).item()
    return rows * CALLS_PER_CITY


def to_row(city: dict, weather: dict, air: dict) -> dict:
    """Flatten one city's two API responses into the Kaggle file's column names."""
    reading = air["list"][0]
    comp = reading["components"]
    sun = weather.get("sys", {})
    wind = weather.get("wind", {})
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
    cities = read_sql(
        """
        select geonameid, city_name, country_code, country_name, continent, population,
               lat, lon
        from core.dim_city where is_active order by population desc
        """
    ).to_dicts()
    if config.max_cities:
        cities = cities[: config.max_cities]

    used = _calls_this_month()
    needed = len(cities) * CALLS_PER_CITY
    if used + needed > MONTHLY_CALL_BUDGET:
        context.log.warning(
            f"Skipping: {used:,} calls used this month, budget {MONTHLY_CALL_BUDGET:,}"
        )
        return MaterializeResult(metadata={"skipped": "monthly budget", "calls_this_month": used})

    collected_at = datetime.now(UTC)
    results, failures = openweather.fetch(cities, log=context.log)
    for f in failures[:5]:
        context.log.warning(f"City {f['geonameid']} failed: {f['error']}")

    rows = [to_row(r["city"], r["weather"], r["air"]) for r in results]
    df = pl.DataFrame(rows).with_columns(
        pl.lit("openweather").alias("_source"),
        pl.lit(collected_at).alias("_ingested_at"),
        pl.lit(context.run_id).alias("_dagster_run_id"),
    )

    # Keep the exact responses on disk too, so any hour can be replayed
    raw_dir = (
        get_settings().raw_dir
        / "source=openweather"
        / f"ingest_date={collected_at.date().isoformat()}"
    )
    raw_dir.mkdir(parents=True, exist_ok=True)
    df.write_parquet(raw_dir / f"hour={collected_at:%H}.parquet")

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


@asset_check(asset=openweather_snapshots)
def most_cities_collected() -> AssetCheckResult:
    """At least 98% of active cities in the latest collection."""
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
    return AssetCheckResult(
        passed=share >= MIN_SUCCESS_SHARE,
        severity=AssetCheckSeverity.WARN,
        metadata={"collected": collected, "active_cities": active, "share": round(share, 4)},
    )
