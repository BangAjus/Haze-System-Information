-- One row per city, with attributes from its most recent observation.
-- A city counts as active if it appeared in the last 3 days of data.

with ranked as (
    select
        *,
        row_number() over (partition by geonameid order by observed_at_utc desc) as rn,
        min(observed_at_utc) over (partition by geonameid) as first_seen_at,
        max(observed_at_utc) over (partition by geonameid) as last_seen_at
    from {{ ref('stg_observations') }}
),

latest_data as (
    select max(observed_at_utc) as max_observed_at from {{ ref('stg_observations') }}
)

select
    geonameid,
    city_name,
    country_code,
    country_name,
    continent,
    population,
    lat,
    lon,
    first_seen_at,
    last_seen_at,
    last_seen_at >= latest_data.max_observed_at - interval '3 days' as is_active
from ranked
cross join latest_data
where rn = 1
