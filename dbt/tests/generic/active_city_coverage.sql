{#
  Fails when fewer than min_share of active cities have a reading in the 24 hours before the
  newest observation. A rolling window rather than the latest calendar day, because with
  hourly collection a new UTC day starts with only a few hours of data.
#}
{% test active_city_coverage(model, min_share=0.95) %}
with seen as (
    select count(distinct geonameid) as n
    from {{ model }}
    where observed_at_utc > (select max(observed_at_utc) - interval '24 hours' from {{ model }})
),
active as (
    select count(*) as n from {{ ref('dim_city') }} where is_active
)
select seen.n as cities_seen, active.n as active_cities
from seen, active
where seen.n < active.n * {{ min_share }}
{% endtest %}
