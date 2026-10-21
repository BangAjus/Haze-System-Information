{{ config(
    materialized='incremental',
    unique_key=['geonameid', 'observed_at_utc'],
    incremental_strategy='merge',
    on_schema_change='append_new_columns'
) }}

-- One row per city per observation. Incremental: each run only merges observations newer
-- than the latest one already loaded (minus a 2-day margin for late corrections).

select
    geonameid,
    observed_at_utc,
    ingested_at,
    temp_c,
    feels_like_c,
    humidity_pct,
    pressure_hpa,
    wind_speed_ms,
    wind_deg,
    wind_gust_ms,
    clouds_pct,
    visibility_m,
    rain_1h_mm,
    snow_1h_mm,
    is_daytime,
    aqi,
    pm2_5,
    pm10,
    co,
    no,
    no2,
    o3,
    so2,
    nh3,
    source
from {{ ref('stg_observations') }}

{% if is_incremental() %}
where observed_at_utc > (select max(observed_at_utc) - interval '2 days' from {{ this }})
{% endif %}
