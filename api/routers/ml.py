from datetime import date, timedelta

from fastapi import APIRouter

from api.db import fetch_all

router = APIRouter(prefix="/ml", tags=["model"])


@router.get("/models")
def models():
    return fetch_all("select * from ml.model_runs order by trained_at desc")


@router.get("/accuracy")
def accuracy(start: date | None = None, end: date | None = None):
    """Daily realized error by continent."""
    end = end or date.today()
    start = start or end - timedelta(days=30)
    return fetch_all(
        """
        select a.target_time_utc::date as date_utc, c.continent,
               avg(a.abs_error) as mae,
               avg((a.aqi_pred = a.aqi_actual)::int) as aqi_acc,
               avg(a.in_interval::int) as interval_coverage,
               count(*) as n
        from ml.forecast_accuracy a
        join core.dim_city c using (geonameid)
        where a.target_time_utc::date between :start and :end
        group by 1, 2
        order by 1, 2
        """,
        start=start,
        end=end,
    )
