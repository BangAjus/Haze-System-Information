# Data profile

Profiled on 2026-10-17 from Kaggle version 178 of `xjoannax88/global-weather-and-air-pollution-dataset`
(`output/openweather_weather_airpollution_top3cities_per_country.parquet`, 12.6 MB).

## Shape

| Fact | Value |
|---|---|
| Rows × columns | 103,211 × 88 |
| History | 2026-04-24 to 2026-10-16 (176 days) |
| Cities / countries | 596 cities (`geonameid`), 209 countries |
| Readings per day | ~584, one per active city |
| Collection time | Once a day, mostly 20:00–23:00 UTC |
| Days with two collections | A few (max 2 readings per city per day) |
| Duplicate `(geonameid, weather_time_unix)` | 0 |

Each Kaggle version contains the **full history**, not only the newest day. The raw load
replaces the table each run, and `fct_observation` merges incrementally.

## Decisions this changed

- **City key:** the file has `geonameid` (GeoNames ID), so the pipeline uses it instead of the
  md5 name hash proposed in the plan. It is stable and unique per city.
- **Observation time:** use `weather_time_unix` (never null). The `*_utc` text columns are null on 583 rows.
- **Units:** `units_weather` is always `metric`, so `temp` is already °C and wind is m/s.
- **Continent:** null on 13,008 rows, which are exactly the North American countries (US, CA, MX,
  Caribbean, Central America). The code `NA` was read as missing upstream. Staging restores it.
- **Target leakage confirmed:** `target_pm25 == pm2_5` and `target_aqi_class == aqi` on 100% of rows.
  They are same-time copies, so staging drops them and the model predicts the next reading instead.
- **Forecast history:** about 6 months, which covers spring to autumn but no full winter.
  Seasonality features will be weak until more data accumulates (or after a backfill).

## Nulls worth handling

| Column | Nulls | Handling |
|---|---|---|
| `wind_gust` | 30,394 (29%) | Filled with `wind_speed` in features |
| `visibility` | 97 | Left null |
| `continent` | 13,008 | Restored to `NA` |
| `*_utc` text columns | 583 | Not used; unix timestamps instead |

## PM2.5 distribution (µg/m³)

Median 4.4, p90 25.7, p99 86.7, max 406.6. No negative values. Temperature range −13.7 to 42.1 °C.

## Hourly history backfill (2026-10-22)

One year of hourly air pollution from OpenWeather's history endpoint: 584 cities, 4.9M rows
(2025-10-22 to 2026-10-22), about 96% of hours present per city.

- **Sahel dust storms are real, not errors.** Zinder and Maradi (Niger), Kano (Nigeria) and
  N'Djamena (Chad) reach PM10 up to ~10,000 µg/m³ and PM2.5 ~2,000 µg/m³ during the
  December–February Harmattan season. The data tests' upper limits (PM2.5 5,000, PM10 20,000)
  are set to catch broken values, not extreme weather. PM10 p99.99 is ~2,460 µg/m³.
- **Weather history isn't on the free tier**, so the hourly models use pollutants and calendar
  features only; the daily models keep the weather features.
