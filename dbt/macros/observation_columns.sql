{#
  Renames and casts shared by every observation source. Both raw tables use the Kaggle
  file's column names, so the only difference between sources is which timestamp is the
  observation time. Used by stg_kaggle_observations and stg_openweather_observations.
#}
{% macro observation_columns(observed_at) %}
    geonameid::bigint                                  as geonameid,
    city_name,
    country_code,
    country_name,
    -- 'NA' (North America) was read as missing upstream in the Kaggle file
    coalesce(continent, 'NA')                          as continent,
    city_population::bigint                            as population,
    geo_lat::double precision                          as lat,
    geo_lon::double precision                          as lon,

    {{ observed_at }}                                  as observed_at_utc,
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
{% endmacro %}
