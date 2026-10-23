{#
  No-leakage rule for feature tables: every target must be the next reading, 18-30h after
  as_of_utc. Returns the rows that break it.
#}
{% test target_within_window(model) %}
select geonameid, as_of_utc, target_time_utc
from {{ model }}
where target is not null
  and not (target_time_utc - as_of_utc between interval '18 hours' and interval '30 hours')
{% endtest %}
