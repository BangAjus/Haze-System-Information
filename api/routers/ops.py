from fastapi import APIRouter, Query

from api.db import fetch_all, fetch_one
from shared.config import get_settings

router = APIRouter(prefix="/ops", tags=["pipeline operations"])


@router.get("/summary")
def summary():
    """Status tiles for the dashboard's Overview page."""
    return fetch_one(
        """
        select
          (select row_to_json(r) from (
              select run_id, job_name, status, finished_at from ops.pipeline_runs
              order by started_at desc nulls last limit 1) r) as last_run,
          (select max(ingested_at) from core.fct_observation) as last_ingested_at,
          (select max(observed_at_utc) from core.fct_observation) as latest_observation,
          (select count(*) from core.fct_observation
             where observed_at_utc::date = (select max(observed_at_utc)::date
                                            from core.fct_observation)) as rows_latest_day,
          (select count(*) from ops.check_results
             where not passed and checked_at > now() - interval '24 hours') as failed_checks_24h,
          (select avg(abs_error) from ml.forecast_accuracy
             where target_time_utc > now() - interval '7 days') as model_mae_7d
        """
    )


@router.get("/runs")
def runs(job: str | None = None, status: str | None = None, limit: int = Query(50, le=500)):
    rows = fetch_all(
        """
        select * from ops.pipeline_runs
        where (cast(:job as text) is null or job_name = :job)
          and (cast(:status as text) is null or status = :status)
        order by started_at desc nulls last
        limit :limit
        """,
        job=job,
        status=status,
        limit=limit,
    )
    ui = get_settings().dagster_ui_url
    for r in rows:
        r["dagster_url"] = f"{ui}/runs/{r['run_id']}"
    return rows


@router.get("/runs/{run_id}")
def run_detail(run_id: str):
    run = fetch_one("select * from ops.pipeline_runs where run_id = :id", id=run_id)
    run["assets"] = fetch_all(
        "select * from ops.asset_materializations where run_id = :id order by asset_key", id=run_id
    )
    run["checks"] = fetch_all(
        "select * from ops.check_results where run_id = :id order by asset_key, check_name",
        id=run_id,
    )
    run["dagster_url"] = f"{get_settings().dagster_ui_url}/runs/{run_id}"
    return run


@router.get("/checks")
def checks(failed_only: bool = False, limit: int = Query(200, le=2000)):
    return fetch_all(
        """
        select * from ops.check_results
        where (not :failed_only or not passed)
        order by checked_at desc
        limit :limit
        """,
        failed_only=failed_only,
        limit=limit,
    )


# TODO(phase 2): GET /ops/assets (freshness per asset) and
# POST /ops/jobs/{job}/runs (trigger through Dagster GraphQL, admin key required).
