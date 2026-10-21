"""OpenWeather AQI scale (1-5) and PM2.5 breakpoints.

Breakpoints follow OpenWeather's Air Pollution API docs; re-check them if the docs change.
OpenWeather's real AQI is the worst level across all pollutants, so a level derived from
PM2.5 alone is an approximation. Used for forecasts, where only PM2.5 is predicted.
"""

import polars as pl

AQI_LABELS = {1: "Good", 2: "Fair", 3: "Moderate", 4: "Poor", 5: "Very poor"}

# Lower bound (µg/m³) of each level, levels 2-5
PM25_BREAKPOINTS = [10.0, 25.0, 50.0, 75.0]


def pm25_to_aqi(pm25: float) -> int:
    return 1 + sum(pm25 >= b for b in PM25_BREAKPOINTS)


def pm25_to_aqi_expr(col: str | pl.Expr) -> pl.Expr:
    expr = pl.col(col) if isinstance(col, str) else col
    return (1 + pl.sum_horizontal([(expr >= b).cast(pl.Int8) for b in PM25_BREAKPOINTS])).cast(
        pl.Int16
    )
