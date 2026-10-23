"""OpenWeather AQI scale (1-5) and per-pollutant breakpoints.

Breakpoints follow OpenWeather's Air Pollution API docs; re-check them if the docs change.
OpenWeather's AQI is the worst level across all pollutants. Forecasts cover PM2.5 and PM10,
so the forecast AQI is the worse of those two levels (an approximation without the gases).
"""

import polars as pl

# ------ PHASE 1: SCALE AND BREAKPOINTS --------

AQI_LABELS = {1: "Good", 2: "Fair", 3: "Moderate", 4: "Poor", 5: "Very poor"}

# Lower bound (µg/m³) of levels 2-5, per pollutant
BREAKPOINTS = {
    "pm25": [10.0, 25.0, 50.0, 75.0],
    "pm10": [20.0, 50.0, 100.0, 200.0],
}
PM25_BREAKPOINTS = BREAKPOINTS["pm25"]


# ------ PHASE 2: CONVERSIONS --------


def to_aqi(value: float, pollutant: str = "pm25") -> int:
    return 1 + sum(value >= b for b in BREAKPOINTS[pollutant])


def to_aqi_expr(col: str | pl.Expr, pollutant: str = "pm25") -> pl.Expr:
    expr = pl.col(col) if isinstance(col, str) else col
    levels = [(expr >= b).cast(pl.Int8) for b in BREAKPOINTS[pollutant]]
    return (1 + pl.sum_horizontal(levels)).cast(pl.Int16)


def pm25_to_aqi(pm25: float) -> int:
    return to_aqi(pm25, "pm25")


def pm25_to_aqi_expr(col: str | pl.Expr) -> pl.Expr:
    return to_aqi_expr(col, "pm25")
