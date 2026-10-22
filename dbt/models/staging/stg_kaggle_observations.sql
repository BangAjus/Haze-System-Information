-- Kaggle daily file, typed and renamed.
-- Profiled 2026-10-17: units_weather is always 'metric' (°C, m/s), and the *_utc text
-- columns are null on 583 rows but weather_time_unix never is.
-- Dropped on purpose: target_* (same-time copies of aqi/pm2_5/pm10, a leakage trap), and
-- the upstream derived columns (temp_category, comfort_proxy, *_interaction).

select
    {{ observation_columns('to_timestamp(weather_time_unix)') }}
from {{ source('raw', 'weather_air_snapshots') }}
where units_weather = 'metric'
