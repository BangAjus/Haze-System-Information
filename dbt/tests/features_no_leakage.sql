-- Singular test: returns rows whose target is not "the next reading, about 24h later".
-- Any returned row fails the test.

select geonameid, as_of_utc, target_time_utc
from {{ ref('features_pm25') }}
where target_pm25 is not null
  and not (target_time_utc - as_of_utc between interval '18 hours' and interval '30 hours')
