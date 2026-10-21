from datetime import UTC, datetime

from dagster import (
    DagsterRunStatus,
    RunFailureSensorContext,
    RunRequest,
    RunStatusSensorContext,
    SensorEvaluationContext,
    SkipReason,
    run_failure_sensor,
    run_status_sensor,
    sensor,
)
from sqlalchemy import text

from pipeline.jobs import daily_ingest
from pipeline.resources import KaggleResource
from shared.db import get_engine


@sensor(job=daily_ingest, minimum_interval_seconds=30 * 60)
def kaggle_update_sensor(context: SensorEvaluationContext, kaggle: KaggleResource):
    """Starts daily_ingest when the author publishes a new version of the file."""
    fingerprint = kaggle.fingerprint()
    if fingerprint == context.cursor:
        return SkipReason("No new Kaggle version")
    context.update_cursor(fingerprint)
    return RunRequest(run_key=fingerprint, tags={"trigger": "sensor", "source": "kaggle"})


def _record_run(context: RunStatusSensorContext, status: str) -> None:
    run = context.dagster_run
    stats = context.instance.get_run_stats(run.run_id)
    started = datetime.fromtimestamp(stats.start_time, UTC) if stats.start_time else None
    finished = datetime.fromtimestamp(stats.end_time, UTC) if stats.end_time else None
    # Only run_failure_sensor contexts carry the failure event
    failure_event = getattr(context, "failure_event", None)
    error = failure_event.message if failure_event else None

    with get_engine().begin() as conn:
        conn.execute(
            text(
                """
                insert into ops.pipeline_runs (run_id, job_name, partition_key, status, trigger,
                    started_at, finished_at, duration_s, error_message, updated_at)
                values (:run_id, :job, :partition, :status, :trigger, :started, :finished,
                    :duration, :error, now())
                on conflict (run_id) do update set
                    status = excluded.status, finished_at = excluded.finished_at,
                    duration_s = excluded.duration_s, error_message = excluded.error_message,
                    updated_at = now()
                """
            ),
            dict(
                run_id=run.run_id,
                job=run.job_name,
                partition=run.tags.get("dagster/partition"),
                status=status,
                trigger=run.tags.get(
                    "trigger", "schedule" if "dagster/schedule_name" in run.tags else "manual"
                ),
                started=started,
                finished=finished,
                duration=(stats.end_time - stats.start_time)
                if stats.start_time and stats.end_time
                else None,
                error=error,
            ),
        )


@run_status_sensor(run_status=DagsterRunStatus.SUCCESS, monitor_all_code_locations=True)
def record_run_success(context: RunStatusSensorContext):
    """Writes finished runs to ops.pipeline_runs, which the web dashboard reads."""
    _record_run(context, "success")


@run_failure_sensor(monitor_all_code_locations=True)
def record_run_failure(context: RunFailureSensorContext):
    _record_run(context, "failure")
