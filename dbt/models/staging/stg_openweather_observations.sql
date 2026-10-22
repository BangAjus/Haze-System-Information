-- Hourly OpenWeather collection, typed and renamed.
-- Observation time is the Air Pollution API's hourly timestamp: PM2.5 is the forecast target,
-- and the weather reading is at most minutes older.

select
    {{ observation_columns('to_timestamp(air_time_unix)') }}
from {{ source('raw', 'openweather_snapshots') }}
