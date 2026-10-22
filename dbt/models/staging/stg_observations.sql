-- All observation sources combined: the Kaggle daily file and the hourly OpenWeather
-- collection. One row per city per observation time, keeping the most recently ingested copy.

with combined as (
    select * from {{ ref('stg_kaggle_observations') }}
    union all
    select * from {{ ref('stg_openweather_observations') }}
)

select distinct on (geonameid, observed_at_utc) *
from combined
order by geonameid, observed_at_utc, ingested_at desc
