-- Runs once, the first time the Postgres volume is created.
-- Separate databases for Dagster's own storage and for MLflow.
CREATE DATABASE dagster;
CREATE DATABASE mlflow;

-- Warehouse schemas in the main database. dbt creates staging, core and marts itself.
CREATE SCHEMA IF NOT EXISTS raw;
CREATE SCHEMA IF NOT EXISTS ml;
CREATE SCHEMA IF NOT EXISTS ops;
CREATE SCHEMA IF NOT EXISTS app;

-- Hourly OpenWeather collection (pipeline/assets/openweather.py). Same column names as the
-- Kaggle file, so staging can union the two sources.
CREATE TABLE IF NOT EXISTS raw.openweather_snapshots (
    geonameid          bigint NOT NULL,
    city_name          text,
    country_code       text,
    country_name       text,
    continent          text,
    city_population    bigint,
    geo_lat            double precision,
    geo_lon            double precision,
    weather_time_unix  bigint,
    air_time_unix      bigint NOT NULL,
    temp               double precision,
    feels_like         double precision,
    humidity           double precision,
    pressure           double precision,
    wind_speed         double precision,
    wind_deg           double precision,
    wind_gust          double precision,
    clouds_all         double precision,
    visibility         double precision,
    rain_1h            double precision,
    snow_1h            double precision,
    is_daytime         bigint,
    aqi                bigint,
    pm2_5              double precision,
    pm10               double precision,
    co                 double precision,
    "no"               double precision,
    no2                double precision,
    o3                 double precision,
    so2                double precision,
    nh3                double precision,
    _source            text,
    _ingested_at       timestamptz NOT NULL,
    _dagster_run_id    text
);
CREATE INDEX IF NOT EXISTS openweather_snapshots_ingested_at
    ON raw.openweather_snapshots (_ingested_at);

-- Hourly air pollution history from OpenWeather (pipeline/assets/openweather.py, backfill).
-- Pollutants only: weather history isn't on the free tier.
CREATE TABLE IF NOT EXISTS raw.openweather_air_history (
    geonameid        bigint NOT NULL,
    air_time_unix    bigint NOT NULL,
    aqi              bigint,
    pm2_5            double precision,
    pm10             double precision,
    co               double precision,
    "no"             double precision,
    no2              double precision,
    o3               double precision,
    so2              double precision,
    nh3              double precision,
    _ingested_at     timestamptz NOT NULL,
    _dagster_run_id  text,
    PRIMARY KEY (geonameid, air_time_unix)
);

-- Pipeline run metadata, written by the ops_recorder sensor
CREATE TABLE IF NOT EXISTS ops.pipeline_runs (
    run_id         text PRIMARY KEY,
    job_name       text NOT NULL,
    partition_key  text,
    status         text NOT NULL,
    trigger        text,
    started_at     timestamptz,
    finished_at    timestamptz,
    duration_s     numeric,
    error_message  text,
    updated_at     timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS ops.asset_materializations (
    run_id           text NOT NULL,
    asset_key        text NOT NULL,
    partition_key    text,
    rows_written     bigint,
    materialized_at  timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (run_id, asset_key)
);

CREATE TABLE IF NOT EXISTS ops.check_results (
    run_id       text NOT NULL,
    asset_key    text NOT NULL,
    check_name   text NOT NULL,
    severity     text NOT NULL,
    passed       boolean NOT NULL,
    failed_rows  bigint,
    details      jsonb,
    checked_at   timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (run_id, asset_key, check_name)
);

-- Model outputs
CREATE TABLE IF NOT EXISTS ml.model_runs (
    model_version      text PRIMARY KEY,
    model_type         text,
    track              text NOT NULL DEFAULT 'daily',   -- daily | hourly
    target             text NOT NULL DEFAULT 'pm25',    -- pm25 | pm10
    mlflow_run_id      text,
    trained_at         timestamptz NOT NULL DEFAULT now(),
    train_start        date,
    train_end          date,
    mae                numeric,
    rmse               numeric,
    aqi_acc            numeric,
    baseline_mae       numeric,
    interval_coverage  numeric,
    alias              text NOT NULL DEFAULT 'challenger'
);

CREATE TABLE IF NOT EXISTS ml.forecasts (
    geonameid        bigint NOT NULL,
    issued_at_utc    timestamptz NOT NULL,
    horizon_h        smallint NOT NULL,
    target_time_utc  timestamptz NOT NULL,
    pm25_pred        numeric NOT NULL,
    pm25_p10         numeric,
    pm25_p90         numeric,
    aqi_pred         smallint,       -- worse of the PM2.5 and PM10 levels
    model_version    text NOT NULL,  -- PM2.5 model
    pm10_pred        numeric,
    pm10_p10         numeric,
    pm10_p90         numeric,
    pm10_model_version text,
    PRIMARY KEY (geonameid, issued_at_utc, horizon_h)
);

-- Every candidate's backtest from each training run, and the hourly forecasts
CREATE TABLE IF NOT EXISTS ml.model_candidates (
    trained_at         timestamptz NOT NULL,
    track              text NOT NULL,
    target             text NOT NULL,
    model              text NOT NULL,
    mae                double precision,
    rmse               double precision,
    aqi_acc            double precision,
    interval_coverage  double precision,
    baseline_mae       double precision,
    selected           boolean NOT NULL DEFAULT false,
    mlflow_run_id      text,
    PRIMARY KEY (trained_at, track, target, model)
);
CREATE TABLE IF NOT EXISTS ml.forecasts_hourly (
    geonameid        bigint NOT NULL,
    issued_at_utc    timestamptz NOT NULL,
    target           text NOT NULL,
    horizon_h        smallint NOT NULL,
    target_time_utc  timestamptz NOT NULL,
    pred             double precision NOT NULL,
    p10              double precision,
    p90              double precision,
    model_version    text NOT NULL,
    PRIMARY KEY (geonameid, issued_at_utc, target, horizon_h)
);

-- iOS push alerts (phase 5)
CREATE TABLE IF NOT EXISTS app.devices (
    device_id     uuid PRIMARY KEY,
    apns_token    text NOT NULL,
    created_at    timestamptz NOT NULL DEFAULT now(),
    last_seen_at  timestamptz
);

CREATE TABLE IF NOT EXISTS app.alert_subscriptions (
    device_id         uuid REFERENCES app.devices ON DELETE CASCADE,
    geonameid         bigint NOT NULL,
    aqi_threshold     smallint NOT NULL CHECK (aqi_threshold BETWEEN 1 AND 5),
    last_notified_at  timestamptz,
    PRIMARY KEY (device_id, geonameid)
);
