-- Runs once, the first time the Postgres volume is created.
-- Separate databases for Dagster's own storage and for MLflow.
CREATE DATABASE dagster;
CREATE DATABASE mlflow;

-- Warehouse schemas in the main database. dbt creates staging, core and marts itself.
CREATE SCHEMA IF NOT EXISTS raw;
CREATE SCHEMA IF NOT EXISTS ml;
CREATE SCHEMA IF NOT EXISTS ops;
CREATE SCHEMA IF NOT EXISTS app;

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
    aqi_pred         smallint,
    model_version    text NOT NULL,
    PRIMARY KEY (geonameid, issued_at_utc, horizon_h)
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
