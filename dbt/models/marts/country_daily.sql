-- Daily summary per country, with a population-weighted PM2.5 and the worst city.

with city_day as (
    select d.*, c.country_code, c.population
    from {{ ref('city_daily') }} d
    join {{ ref('dim_city') }} c using (geonameid)
),

ranked as (
    select *,
        row_number() over (
            partition by country_code, date_utc order by pm25_mean desc
        ) as worst_rank
    from city_day
)

select
    country_code,
    date_utc,
    count(*)                                                    as n_cities,
    sum(pm25_mean * population) / nullif(sum(population), 0)    as pm25_pop_weighted,
    max(aqi_max)                                                as aqi_max,
    max(case when worst_rank = 1 then geonameid end)            as worst_geonameid
from ranked
group by 1, 2
