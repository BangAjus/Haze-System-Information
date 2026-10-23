-- Hourly pollutant readings from every source, one row per city per UTC hour.
-- Sources: the OpenWeather history backfill, the hourly OpenWeather collection, and the
-- Kaggle daily file. All three come from the same OpenWeather Air Pollution model, so when
-- several give the same hour, the most recently ingested copy wins.

with readings as (
    select geonameid, to_timestamp(air_time_unix) as observed_at, aqi, pm2_5, pm10,
           co, "no", no2, o3, so2, nh3, _ingested_at, 'openweather_history' as source
    from {{ source('raw', 'openweather_air_history') }}

    union all

    select geonameid, to_timestamp(air_time_unix), aqi, pm2_5, pm10,
           co, "no", no2, o3, so2, nh3, _ingested_at, 'openweather'
    from {{ source('raw', 'openweather_snapshots') }}

    union all

    select geonameid, observed_at_utc, aqi, pm2_5, pm10,
           co, "no", no2, o3, so2, nh3, ingested_at, 'kaggle'
    from {{ ref('stg_kaggle_observations') }}
)

select distinct on (geonameid, date_trunc('hour', observed_at))
    geonameid::bigint                                as geonameid,
    date_trunc('hour', observed_at)                  as hour_utc,
    aqi::smallint                                    as aqi,
    pm2_5::double precision                          as pm2_5,
    pm10::double precision                           as pm10,
    co::double precision                             as co,
    "no"::double precision                           as no,
    no2::double precision                            as no2,
    o3::double precision                             as o3,
    so2::double precision                            as so2,
    nh3::double precision                            as nh3,
    _ingested_at                                     as ingested_at,
    source
from readings
order by geonameid, date_trunc('hour', observed_at), _ingested_at desc
