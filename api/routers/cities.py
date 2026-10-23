from datetime import date, timedelta

from fastapi import APIRouter, Query

from api.db import fetch_all, fetch_one
from shared.aqi import AQI_LABELS

# ------ PHASE 0: ROUTER --------

router = APIRouter(prefix="/cities", tags=["cities"])

ATTRIBUTION = "Weather and air pollution data: OpenWeather, via Kaggle (CC BY 4.0)"


# ------ PHASE 1: FIND CITIES --------


@router.get("")
def list_cities(
    q: str | None = None,
    country: str | None = None,
    limit: int = Query(50, le=600),
):
    return fetch_all(
        """
        select geonameid, city_name, country_code, country_name, continent, lat, lon, is_active
        from core.dim_city
        where (cast(:q as text) is null or city_name ilike '%' || :q || '%')
          and (cast(:country as text) is null or country_code = upper(:country))
        order by population desc
        limit :limit
        """,
        q=q,
        country=country,
        limit=limit,
    )


@router.get("/nearest")
def nearest_city(lat: float = Query(ge=-90, le=90), lon: float = Query(ge=-180, le=180)):
    """Closest active city by great-circle distance."""
    return fetch_one(
        """
        select geonameid, city_name, country_code, lat, lon,
               round((6371 * acos(least(1, cos(radians(:lat)) * cos(radians(lat))
                 * cos(radians(lon) - radians(:lon)) + sin(radians(:lat)) * sin(radians(lat)))
               ))::numeric, 1) as distance_km
        from core.dim_city
        where is_active
        order by distance_km
        limit 1
        """,
        lat=lat,
        lon=lon,
    )


@router.get("/{geonameid}")
def get_city(geonameid: int):
    return fetch_one("select * from core.dim_city where geonameid = :id", id=geonameid)


# ------ PHASE 2: READINGS --------


@router.get("/{geonameid}/current")
def current(geonameid: int):
    row = fetch_one(
        """
        select o.*, extract(epoch from now() - o.observed_at_utc) / 3600 as data_age_hours
        from core.fct_observation o
        where geonameid = :id
        order by observed_at_utc desc
        limit 1
        """,
        id=geonameid,
    )
    row["aqi_label"] = AQI_LABELS.get(row["aqi"])
    row["attribution"] = ATTRIBUTION
    return row


@router.get("/{geonameid}/history")
def history(geonameid: int, start: date | None = None, end: date | None = None):
    end = end or date.today()
    start = start or end - timedelta(days=30)
    return fetch_all(
        """
        select * from marts.city_daily
        where geonameid = :id and date_utc between :start and :end
        order by date_utc
        """,
        id=geonameid,
        start=start,
        end=end,
    )


# ------ PHASE 3: FORECASTS --------


@router.get("/{geonameid}/forecast")
def forecast(geonameid: int):

    # job 1: the city
    city = fetch_one(
        "select geonameid, city_name, country_code from core.dim_city where geonameid = :id",
        id=geonameid,
    )
    # job 2: latest daily-track forecast (PM2.5 + PM10)
    rows = fetch_all(
        """
        select horizon_h, target_time_utc, pm25_pred, pm25_p10, pm25_p90,
               pm10_pred, pm10_p10, pm10_p90, aqi_pred,
               model_version, pm10_model_version, issued_at_utc
        from ml.forecasts
        where geonameid = :id
          and issued_at_utc = (select max(issued_at_utc) from ml.forecasts where geonameid = :id)
        order by horizon_h
        """,
        id=geonameid,
    )
    # job 3: latest hourly-track forecast per pollutant
    hourly = fetch_all(
        """
        select distinct on (target) target, horizon_h, target_time_utc, pred, p10, p90,
               model_version, issued_at_utc
        from ml.forecasts_hourly
        where geonameid = :id
        order by target, issued_at_utc desc
        """,
        id=geonameid,
    )

    # job 4: shape the response
    def band(pred, lo, hi):
        return None if pred is None else {"pred": pred, "p10": lo, "p90": hi}

    return {
        "city": city,
        "issued_at": rows[0]["issued_at_utc"] if rows else None,
        "models": {
            "pm25": rows[0]["model_version"] if rows else None,
            "pm10": rows[0]["pm10_model_version"] if rows else None,
        },
        # Next-reading forecast from the daily models (about 24h ahead)
        "forecasts": [
            {
                "horizon_h": r["horizon_h"],
                "target_time": r["target_time_utc"],
                "pm25": band(r["pm25_pred"], r["pm25_p10"], r["pm25_p90"]),
                "pm10": band(r["pm10_pred"], r["pm10_p10"], r["pm10_p90"]),
                "aqi": {"level": r["aqi_pred"], "label": AQI_LABELS.get(r["aqi_pred"])},
            }
            for r in rows
        ],
        # 24h-ahead forecast from the hourly models (refreshed every hour)
        "hourly": {
            h["target"]: {
                "horizon_h": h["horizon_h"],
                "target_time": h["target_time_utc"],
                "issued_at": h["issued_at_utc"],
                "model_version": h["model_version"],
                **band(h["pred"], h["p10"], h["p90"]),
            }
            for h in hourly
        },
        "attribution": ATTRIBUTION,
    }
