-- Next-reading PM10 forecast features. The query lives in macros/daily_features.sql.
-- Read by the Dagster asset ml/pm10_model.
{{ daily_features('pm10') }}
