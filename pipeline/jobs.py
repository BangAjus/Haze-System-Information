from datetime import UTC, datetime, timedelta

from dagster import (
    AssetSelection,
    DagsterRunStatus,
    DefaultScheduleStatus,
    RunRequest,
    RunsFilter,
    ScheduleDefinition,
    ScheduleEvaluationContext,
    SkipReason,
    define_asset_job,
    schedule,
)

from shared.config import get_settings

ingest_and_transform = AssetSelection.groups("ingest", "transform")
daily_ml_assets = AssetSelection.assets(
    ["ml", "features_pm25"], ["ml", "forecasts"], ["ml", "forecast_accuracy"]
)

# The run queue allows one run with this tag at a time (see infra/dagster/dagster.yaml), so
# two loads (Kaggle daily, OpenWeather hourly) never run dbt build or touch raw at once.
INGEST_TAGS = {"haze/exclusive": "ingest"}

daily_ingest = define_asset_job(
    "daily_ingest",
    selection=ingest_and_transform,
    tags=INGEST_TAGS,
    description="Download the latest Kaggle file, load raw, run dbt build.",
)

hourly_openweather = define_asset_job(
    "hourly_openweather",
    selection=AssetSelection.groups("ingest_hourly", "transform"),
    tags=INGEST_TAGS,
    description="Collect current weather and air pollution for every active city, run dbt build.",
)

daily_ml = define_asset_job(
    "daily_ml",
    selection=daily_ml_assets,
    description="Rebuild features, forecast every city, score past forecasts.",
)

weekly_retrain = define_asset_job(
    "weekly_retrain",
    selection=AssetSelection.assets(["ml", "features_pm25"], ["ml", "pm25_model"]),
    description="Backtest every candidate; train the best and promote it if it passes the rule.",
)

FALLBACK_SKIP_WINDOW = timedelta(hours=20)


@schedule(job=daily_ingest, cron_schedule="0 6 * * *", name="daily_ingest_schedule")
def daily_ingest_fallback(context: ScheduleEvaluationContext):
    """Fallback in case the Kaggle sensor misses an update (the author publishes ~20-23 UTC).

    Skips when an ingest is already queued or running, or succeeded recently, so the
    sensor and this schedule don't both load the same data.
    """
    since = datetime.now(UTC) - FALLBACK_SKIP_WINDOW
    recent = context.instance.get_runs(
        filters=RunsFilter(
            job_name="daily_ingest",
            statuses=[
                DagsterRunStatus.QUEUED,
                DagsterRunStatus.STARTING,
                DagsterRunStatus.STARTED,
                DagsterRunStatus.SUCCESS,
            ],
            created_after=since,
        ),
        limit=1,
    )
    if recent:
        return SkipReason(f"daily_ingest already ran or is running (run {recent[0].run_id[:8]})")
    return RunRequest(tags={"trigger": "schedule"})


daily_ml_schedule = ScheduleDefinition(job=daily_ml, cron_schedule="30 6 * * *")
weekly_retrain_schedule = ScheduleDefinition(job=weekly_retrain, cron_schedule="0 3 * * 0")

# Minute 15: the Air Pollution API's hourly reading is out by then. Switched on automatically
# only when OPENWEATHER_API_KEY is set; otherwise it starts off.
hourly_openweather_schedule = ScheduleDefinition(
    job=hourly_openweather,
    cron_schedule="15 * * * *",
    default_status=DefaultScheduleStatus.RUNNING
    if get_settings().openweather_api_key
    else DefaultScheduleStatus.STOPPED,
)
