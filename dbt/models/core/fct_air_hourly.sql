{{ config(
    materialized='incremental',
    unique_key=['geonameid', 'hour_utc'],
    incremental_strategy='merge',
    indexes=[{'columns': ['geonameid', 'hour_utc'], 'unique': True}]
) }}

-- One row per city per UTC hour of pollutant readings. Read by the hourly forecast models
-- (LSTM, GRU, hourly gradient boosting).
-- Incremental by ingest time, not by hour: a backfill adds *old* hours, which a
-- "latest hours only" filter would skip. The 3h overlap covers loads that commit late.

select *
from {{ ref('stg_air_hourly') }}

{% if is_incremental() %}
where ingested_at > (select max(ingested_at) - interval '3 hours' from {{ this }})
{% endif %}
