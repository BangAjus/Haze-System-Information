# Haze

A city air-quality pipeline, a PM2.5 forecast model, one API, and (later) a web operations
dashboard and an iOS app. Built on the
[Global Weather & Air Pollution Dataset](https://www.kaggle.com/datasets/xjoannax88/global-weather-and-air-pollution-dataset)
(OpenWeather and GeoNames data, CC BY 4.0).

- Plan: [docs/haze-plan.html](docs/haze-plan.html)
- Data profile and decisions: [docs/data-profile.md](docs/data-profile.md)

## Layout

```
shared/     config, database helpers, AQI scale
pipeline/   Dagster: assets (ingest, dbt, ml), jobs, schedules, sensors
dbt/        staging -> core -> marts, with tests
ml/         features, backtest, model training
api/        FastAPI /v1 (cities, ops, ml, health)
infra/      Docker Compose, Postgres init SQL, Dagster instance config
web/        Next.js dashboard (phase 4)
ios/        SwiftUI app (phase 5)
tests/      unit tests
```

## Data flow

Kaggle file → `data/raw/` (Parquet, per ingest date) → `raw.weather_air_snapshots` →
dbt `staging.stg_observations` → `core.dim_city`, `core.fct_observation` →
`marts.city_daily`, `marts.country_daily` → `ml.features_pm25` → `ml.forecasts` → API.
Every run is also recorded in `ops.pipeline_runs`, which the dashboard reads.

## Getting started

1. Copy the env file and fill in the Postgres password and your Kaggle token:
   ```bash
   cp .env.example .env
   ```
2. Install Python dependencies (or use `uv sync --all-extras` if you have uv):
   ```bash
   make setup
   ```
3. Start Postgres and MLflow in Docker:
   ```bash
   make db
   ```
4. Run Dagster locally and open http://localhost:3000. Materialize `daily_ingest`, then
   `weekly_retrain`, then `daily_ml`.
   ```bash
   make dev
   ```
5. Run the API and open http://localhost:8000/docs:
   ```bash
   make api
   ```

### Full stack in Docker

```bash
docker compose --env-file .env -f infra/docker-compose.yml up -d --build
```

(`make up` does the same.) Host ports come from `.env` (`DAGSTER_HOST_PORT`, `API_HOST_PORT`, …).
Your Kaggle token is mounted read-only from `~/.kaggle`. Sensors and schedules start switched
off: turn them on in the Dagster UI under Automation, or with `dagster sensor start <name>`.

## Jobs

| Job | Trigger | Does |
|---|---|---|
| `daily_ingest` | `kaggle_update_sensor` (new Kaggle version); fallback 06:00 UTC, skipped if an ingest ran in the last 20h | Download, load raw, `dbt build`. Only one runs at a time (queue limit) |
| `hourly_openweather` | Every hour at :15 (on automatically when `OPENWEATHER_API_KEY` is set) | Weather + air pollution for every active city from OpenWeather, `dbt build`, then the hourly 24h-ahead forecasts |
| `daily_ml` | 06:30 UTC | Daily features, PM2.5 + PM10 forecasts, accuracy, drift |
| `weekly_retrain` | Sunday 03:00 UTC | Both tracks × both pollutants: backtest every candidate, train the best, promote it if it passes the rule |
| `backfill_air_history` | Manual | One-off: a year of hourly pollution history per city |

## Hourly OpenWeather collection

Calls the Current Weather and Air Pollution APIs for every active city once an hour and
appends to `raw.openweather_snapshots`, using the Kaggle file's column names so dbt unions
both sources in `staging.stg_observations` (via the `observation_columns` macro).

Free tier budget (checked 2026-10-22): 60 calls/min and 1,000,000 calls/month.

| | Calls |
|---|---|
| One run (584 cities × 2 APIs) | 1,168 |
| Per day (24 runs) | 28,032 |
| Per 31-day month | ~869,000 (87% of the limit) |

Requests are paced at 50/min (a run takes ~23 minutes), and each run skips itself if it
would push this month's usage past 950,000 calls. For a quick test, launch
`hourly_openweather` with `ops: raw__openweather_snapshots: config: max_cities: 10`.
To halve the usage, change the schedule's cron in `pipeline/jobs.py` to every 2 hours.

## Freshness

- **Pipeline freshness:** `raw/weather_air_snapshots` and `ml/forecasts` have Dagster freshness
  policies (warn after 26h, fail after 50h without an update). The daemon evaluates them even
  when nothing runs, so a dead sensor shows up as stale in the UI.
- **Data freshness:** the `observations_are_recent` check runs on every ingest and warns when
  the newest observation is over 36h old (error over 60h). The same rule is in dbt:
  `make dbt-freshness`.

## Forecast models

Two tracks, each forecasting **PM2.5 and PM10 separately** (each pollutant has its own
models and champion). The forecast AQI is the worse of the two pollutants' levels.

| Track | Predicts | Inputs | Candidates | Retrained |
|---|---|---|---|---|
| daily | next reading (~24h ahead) | dbt `features_pm25` / `features_pm10`: lags, rolling means, weather, calendar, location | `hgb`, `random_forest`, `ridge` | weekly |
| hourly | value 24h ahead, refreshed every hour | last 72h of all 8 pollutants from `core.fct_air_hourly`, time of day/week, location | `hgb_lags`, `lstm`, `gru` (PyTorch, CPU) | weekly |

Both tracks use the same rules: an expanding-window backtest against persistence baselines,
p10–p90 ranges from split-conformal errors, and promotion only when a candidate beats
persistence, keeps its range honest (70–90% coverage) and beats the champion by 2%.
Every candidate's backtest goes to `ml.model_candidates` and to MLflow as nested runs.

Choose which candidates compete in the Dagster launchpad
(`ops: ml__pm25_model: config: candidates`, or `ml__hourly_pm25_model` for the hourly track).
To add a model, add it to `CANDIDATES` in `ml/train.py` or `HOURLY_CANDIDATES` in `ml/hourly.py`.

**Hourly history:** the hourly models need long series, so `backfill_air_history` (manual job)
fetches up to a year of hourly pollution per city from OpenWeather's history endpoint. One call
per city covers the whole year (~600 calls in total). Re-runs skip hours already stored.

**Monitoring:** `ml/forecast_accuracy` scores every past forecast (both tracks, both
pollutants) against the hourly reading nearest its target time; the API breaks errors down by
continent or AQI band. `ml/drift` computes PSI per input feature (last 7 days vs. the
champion's training period) and warns at PSI ≥ 0.25.

## Status

Phases 0, 1 and 3 done: ingestion (Kaggle daily, OpenWeather hourly, one-year hourly
backfill), the warehouse, and both forecast tracks for PM2.5 and PM10 with accuracy and drift
monitoring. Not yet built: check results and asset row
counts in `ops.*`, `/v1/ops/assets`, manual run trigger, device/alert endpoints, web and iOS apps.
