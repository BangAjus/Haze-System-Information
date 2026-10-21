{% test unique_observation(model) %}
select geonameid, observed_at_utc, count(*) as n
from {{ model }}
group by 1, 2
having count(*) > 1
{% endtest %}
