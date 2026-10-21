-- Typed, renamed and deduplicated observations.
-- Profiled 2026-10-17: units_weather is always 'metric' (°C, m/s), the *_utc text columns
-- are null on 583 rows but weather_time_unix never is, and continent is null for every
-- North American country because 'NA' was read as missing upstream.

with source as (
    select * from {{ source('raw', 'weather_air_snapshots') }}
),

renamed as (
    select
        geonameid::bigint                                  as geonameid,
        city_name,
        country_code,
        country_name,
        coalesce(continent, 'NA')                          as continent,
        city_population::bigint                            as population,
        geo_lat::double precision                          as lat,
        geo_lon::double precision                          as lon,

        to_timestamp(weather_time_unix)                    as observed_at_utc,
        _ingested_at                                       as ingested_at,

        temp::double precision                             as temp_c,
        feels_like::double precision                       as feels_like_c,
        humidity::double precision                         as humidity_pct,
        pressure::double precision                         as pressure_hpa,
        wind_speed::double precision                       as wind_speed_ms,
        wind_deg::int                                      as wind_deg,
        wind_gust::double precision                        as wind_gust_ms,
        clouds_all::double precision                       as clouds_pct,
        visibility::double precision                       as visibility_m,
        coalesce(rain_1h, 0)::double precision             as rain_1h_mm,
        coalesce(snow_1h, 0)::double precision             as snow_1h_mm,
        is_daytime::int = 1                                as is_daytime,

        aqi::smallint                                      as aqi,
        pm2_5::double precision                            as pm2_5,
        pm10::double precision                             as pm10,
        co::double precision                               as co,
        "no"::double precision                             as no,
        no2::double precision                              as no2,
        o3::double precision                               as o3,
        so2::double precision                              as so2,
        nh3::double precision                              as nh3,

        _source                                            as source
        -- Dropped on purpose: target_* (same-time copies of aqi/pm2_5/pm10, a leakage trap),
        -- and the upstream derived columns (temp_category, comfort_proxy, *_interaction).
    from source
    where units_weather = 'metric'
)

-- One row per city per observation time, keeping the most recently ingested copy
select distinct on (geonameid, observed_at_utc) *
from renamed
order by geonameid, observed_at_utc, ingested_at desc
