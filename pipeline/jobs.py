from dagster import AssetSelection, ScheduleDefinition, define_asset_job

ingest_and_transform = AssetSelection.groups("ingest", "transform")
daily_ml_assets = AssetSelection.assets(
    ["ml", "features_pm25"], ["ml", "forecasts"], ["ml", "forecast_accuracy"]
)

daily_ingest = define_asset_job(
    "daily_ingest",
    selection=ingest_and_transform,
    description="Download the latest Kaggle file, load raw, run dbt build.",
)

daily_ml = define_asset_job(
    "daily_ml",
    selection=daily_ml_assets,
    description="Rebuild features, forecast every city, score past forecasts.",
)

weekly_retrain = define_asset_job(
    "weekly_retrain",
    selection=AssetSelection.assets(["ml", "features_pm25"], ["ml", "pm25_model"]),
    description="Backtest and train a challenger; promote it if it passes the rule.",
)

# Fallback in case the Kaggle sensor misses an update. The author publishes around 20-23 UTC.
daily_ingest_fallback = ScheduleDefinition(job=daily_ingest, cron_schedule="0 6 * * *")
daily_ml_schedule = ScheduleDefinition(job=daily_ml, cron_schedule="30 6 * * *")
weekly_retrain_schedule = ScheduleDefinition(job=weekly_retrain, cron_schedule="0 3 * * 0")
