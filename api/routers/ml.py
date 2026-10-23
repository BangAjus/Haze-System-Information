from datetime import date, timedelta
from typing import Literal

from fastapi import APIRouter, Query

from api.db import fetch_all

# ------ PHASE 0: ROUTER --------

router = APIRouter(prefix="/ml", tags=["model"])

Track = Literal["daily", "hourly"]
Target = Literal["pm25", "pm10"]


# ------ PHASE 1: MODELS --------


@router.get("/models")
def models(track: Track | None = None, target: Target | None = None):
    """Every trained model version, newest first. alias is champion, rejected or archived."""
    return fetch_all(
        """
        select * from ml.model_runs
        where (cast(:track as text) is null or track = :track)
          and (cast(:target as text) is null or target = :target)
        order by trained_at desc
        """,
        track=track,
        target=target,
    )


@router.get("/candidates")
def candidates(track: Track = "daily", target: Target = "pm25"):
    """Every candidate's backtest from the latest training run, best first."""
    return fetch_all(
        """
        select * from ml.model_candidates
        where track = :track and target = :target
          and trained_at = (select max(trained_at) from ml.model_candidates
                            where track = :track and target = :target)
        order by mae
        """,
        track=track,
        target=target,
    )


# ------ PHASE 2: MONITORING --------


@router.get("/accuracy")
def accuracy(
    track: Track = "daily",
    target: Target = "pm25",
    group_by: Literal["continent", "aqi_band", "none"] = "continent",
    start: date | None = None,
    end: date | None = None,
):
    """Realized forecast error per day, broken down by continent or by the actual AQI band."""
    end = end or date.today()
    start = start or end - timedelta(days=30)
    group = {"continent": "c.continent", "aqi_band": "a.aqi_actual::text", "none": "'all'"}[
        group_by
    ]
    return fetch_all(
        f"""
        select a.target_time_utc::date as date_utc, {group} as grp,
               avg(a.abs_error) as mae,
               avg((a.aqi_pred = a.aqi_actual)::int) as aqi_acc,
               avg(a.in_interval::int) as interval_coverage,
               count(*) as n
        from ml.forecast_accuracy a
        join core.dim_city c using (geonameid)
        where a.track = :track and a.target = :target
          and a.target_time_utc::date between :start and :end
        group by 1, 2
        order by 1, 2
        """,
        track=track,
        target=target,
        start=start,
        end=end,
    )


@router.get("/drift")
def drift(track: Track = "daily", target: Target = "pm25", limit: int = Query(30, le=365)):
    """Daily feature drift (PSI against the champion's training period), newest first."""
    return fetch_all(
        """
        select * from ml.drift
        where track = :track and target = :target
        order by date_utc desc, psi desc
        limit :limit
        """,
        track=track,
        target=target,
        limit=limit,
    )
