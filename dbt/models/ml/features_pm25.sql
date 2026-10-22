-- Feature table for the next-reading PM2.5 forecast.
-- One row per city per as-of time. Features use only data at or before as_of_utc.
-- The target is the next reading, about 24h later. Read by the Dagster asset ml/pm25_model.

with obs as (
    select f.*, c.lat, c.lon, c.continent, c.population
    from {{ ref('fct_observation') }} f
    join {{ ref('dim_city') }} c using (geonameid)
),

-- A few days have two collections: keep the latest reading per city per UTC day
daily as (
    select distinct on (geonameid, (observed_at_utc at time zone 'utc')::date) *
    from obs
    order by geonameid, (observed_at_utc at time zone 'utc')::date, observed_at_utc desc
)

select
    geonameid,
    observed_at_utc                                  as as_of_utc,

    -- PM2.5 history, including the reading at as_of
    pm2_5                                            as pm25_lag_0,
    lag(pm2_5, 1) over w                             as pm25_lag_1,
    lag(pm2_5, 2) over w                             as pm25_lag_2,
    lag(pm2_5, 3) over w                             as pm25_lag_3,
    lag(pm2_5, 4) over w                             as pm25_lag_4,
    lag(pm2_5, 5) over w                             as pm25_lag_5,
    lag(pm2_5, 6) over w                             as pm25_lag_6,
    lag(pm2_5, 7) over w                             as pm25_lag_7,

    -- Null until the window is full
    case when count(*) over w3 = 3 then avg(pm2_5) over w3 end          as pm25_roll_mean_3,
    case when count(*) over w7 = 7 then avg(pm2_5) over w7 end          as pm25_roll_mean_7,
    case when count(*) over w7 = 7 then stddev_samp(pm2_5) over w7 end  as pm25_roll_std_7,

    -- PM10 history (coarse particles: dust, sand, construction)
    pm10                                             as pm10_lag_0,
    lag(pm10, 1) over w                              as pm10_lag_1,
    case when count(*) over w3 = 3 then avg(pm10) over w3 end           as pm10_roll_mean_3,
    case when count(*) over w7 = 7 then avg(pm10) over w7 end           as pm10_roll_mean_7,
    -- Share of PM10 that is fine particles. High = combustion/smoke, low = dust
    pm2_5 / nullif(pm10, 0)                          as pm25_pm10_ratio,

    -- Weather at as_of
    temp_c,
    humidity_pct,
    pressure_hpa,
    wind_speed_ms,
    coalesce(wind_gust_ms, wind_speed_ms)            as wind_gust_ms,
    clouds_pct,
    rain_1h_mm,
    sin(radians(wind_deg))                           as wind_sin,
    cos(radians(wind_deg))                           as wind_cos,

    -- Calendar (ISODOW: Monday = 1 ... Sunday = 7)
    extract(isodow from observed_at_utc at time zone 'utc')::int                    as dow,
    sin(extract(month from observed_at_utc at time zone 'utc') * 2 * pi() / 12)      as month_sin,
    cos(extract(month from observed_at_utc at time zone 'utc') * 2 * pi() / 12)      as month_cos,

    -- Location
    lat,
    lon,
    continent,
    ln(1 + population)::double precision             as log_pop,

    -- Target: the next reading, only when it is roughly 24h later
    case
        when lead(observed_at_utc) over w - observed_at_utc
             between interval '18 hours' and interval '30 hours'
        then lead(pm2_5) over w
    end                                              as target_pm25,
    lead(observed_at_utc) over w                     as target_time_utc

from daily
window
    w  as (partition by geonameid order by observed_at_utc),
    w3 as (w rows between 2 preceding and current row),
    w7 as (w rows between 6 preceding and current row)
