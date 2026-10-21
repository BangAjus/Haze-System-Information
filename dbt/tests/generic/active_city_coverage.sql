{# Fails when the latest collection day has readings for fewer than min_share of active cities. #}
{% test active_city_coverage(model, min_share=0.95) %}
with latest_day as (
    select max(observed_at_utc)::date as d from {{ model }}
),
seen as (
    select count(distinct geonameid) as n
    from {{ model }}, latest_day
    where observed_at_utc::date = latest_day.d
),
active as (
    select count(*) as n from {{ ref('dim_city') }} where is_active
)
select seen.n as cities_seen, active.n as active_cities
from seen, active
where seen.n < active.n * {{ min_share }}
{% endtest %}
