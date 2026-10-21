-- Daily summary per city. Feeds city pages, history charts and the world map.
-- In phase 1 there is usually one reading per city per day, so n_obs is 1 (sometimes 2).

with daily as (
    select
        geonameid,
        observed_at_utc::date        as date_utc,
        count(*)                     as n_obs,
        max(aqi)                     as aqi_max,
        avg(pm2_5)                   as pm25_mean,
        max(pm2_5)                   as pm25_max,
        avg(pm10)                    as pm10_mean,
        avg(temp_c)                  as temp_c_mean,
        avg(humidity_pct)            as humidity_mean,
        avg(wind_speed_ms)           as wind_ms_mean,
        max(observed_at_utc)         as last_observed_at
    from {{ ref('fct_observation') }}
    group by 1, 2
)

select
    *,
    pm25_mean - lag(pm25_mean) over (partition by geonameid order by date_utc) as pm25_delta_1d
from daily
