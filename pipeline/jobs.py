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

# ------ PHASE 0: SELECTIONS AND QUEUE TAGS --------

ingest_and_transform = AssetSelection.groups("ingest", "transform")
daily_ml_assets = AssetSelection.assets(
    ["ml", "features_pm25"],
    ["ml", "features_pm10"],
    ["ml", "forecasts"],
    ["ml", "forecast_accuracy"],
    ["ml", "drift"],
)

# The run queue allows one run with this tag at a time (see infra/dagster/dagster.yaml), so
# two loads (Kaggle daily, OpenWeather hourly) never run dbt build or touch raw at once.
INGEST_TAGS = {"haze/exclusive": "ingest"}
# One OpenWeather job at a time too, so two jobs never share the 60 calls/minute limit
OPENWEATHER_TAGS = {"haze/openweather": "api"}


def max_runtime(minutes: int) -> dict:
    """Run monitoring cancels the run after this long (frees the queue if a worker died)."""
    return {"dagster/max_runtime": str(minutes * 60)}


# ------ PHASE 1: JOBS --------

daily_ingest = define_asset_job(
    "daily_ingest",
    selection=ingest_and_transform,
    tags={**INGEST_TAGS, **max_runtime(45)},
    description="Download the latest Kaggle file, load raw, run dbt build.",
)

hourly_openweather = define_asset_job(
    "hourly_openweather",
    # Collect, rebuild the warehouse, then refresh the hourly 24h-ahead forecasts
    selection=AssetSelection.groups("ingest_hourly", "transform")
    | AssetSelection.assets(["ml", "forecasts_hourly"]),
    tags={**INGEST_TAGS, **OPENWEATHER_TAGS, **max_runtime(55)},
    description="Collect current weather and air pollution for every active city, run dbt build.",
)

backfill_air_history = define_asset_job(
    "backfill_air_history",
    selection=AssetSelection.groups("backfill"),
    tags={**OPENWEATHER_TAGS, **max_runtime(180)},
    description="One-off: fetch hourly pollution history for every active city (manual).",
)

daily_ml = define_asset_job(
    "daily_ml",
    selection=daily_ml_assets,
    tags=max_runtime(60),
    description="Rebuild features, forecast every city, score past forecasts.",
)

weekly_retrain = define_asset_job(
    "weekly_retrain",
    selection=AssetSelection.assets(
        ["ml", "features_pm25"],
        ["ml", "features_pm10"],
        ["ml", "pm25_model"],
        ["ml", "pm10_model"],
        ["ml", "hourly_pm25_model"],
        ["ml", "hourly_pm10_model"],
    ),
    tags=max_runtime(240),
    # One model at a time: each hourly model holds ~5M rows in memory, and two in parallel
    # exceeded Docker's 8 GB (the run process was killed with no Python error).
    config={"execution": {"config": {"multiprocess": {"max_concurrent": 1}}}},
    description="Both tracks, both pollutants: backtest every candidate, promote the best.",
)

# ------ PHASE 2: SCHEDULES --------

FALLBACK_SKIP_WINDOW = timedelta(hours=20)


@schedule(job=daily_ingest, cron_schedule="0 6 * * *", name="daily_ingest_schedule")
def daily_ingest_fallback(context: ScheduleEvaluationContext):
    """Fallback in case the Kaggle sensor misses an update (the author publishes ~20-23 UTC).

    Skips when an ingest is already queued or running, or succeeded recently, so the
    sensor and this schedule don't both load the same data.
    """
    # job 1: look for an ingest queued, running or finished in the last 20h
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
    # job 2: skip if found, otherwise start one
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
