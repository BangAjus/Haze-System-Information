-- Next-reading PM2.5 forecast features. The query lives in macros/daily_features.sql.
-- Read by the Dagster asset ml/pm25_model.
{{ daily_features('pm2_5') }}
