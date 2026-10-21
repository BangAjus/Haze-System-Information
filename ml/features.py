"""Feature columns the model trains on.

The feature table itself is built by dbt: dbt/models/ml/features_pm25.sql
(leakage rules are tested in dbt/tests/features_no_leakage.sql). Keep this list in sync
with that model's columns.
"""

WEATHER_COLS = [
    "temp_c",
    "humidity_pct",
    "pressure_hpa",
    "wind_speed_ms",
    "wind_gust_ms",
    "clouds_pct",
    "rain_1h_mm",
]

FEATURE_COLS = [
    *[f"pm25_lag_{i}" for i in range(8)],
    "pm25_roll_mean_3",
    "pm25_roll_mean_7",
    "pm25_roll_std_7",
    *WEATHER_COLS,
    "wind_sin",
    "wind_cos",
    "dow",
    "month_sin",
    "month_cos",
    "lat",
    "lon",
    "continent",
    "log_pop",
]
CATEGORICAL_COLS = ["continent"]
