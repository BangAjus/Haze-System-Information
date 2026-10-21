import polars as pl

from shared.aqi import pm25_to_aqi, pm25_to_aqi_expr


def test_breakpoints():
    assert [pm25_to_aqi(v) for v in [0, 9.99, 10, 24.9, 25, 49.9, 50, 74.9, 75, 400]] == [
        1,
        1,
        2,
        2,
        3,
        3,
        4,
        4,
        5,
        5,
    ]


def test_expr_matches_scalar():
    values = [0.0, 10.0, 30.0, 60.0, 90.0]
    df = pl.DataFrame({"pm": values}).with_columns(pm25_to_aqi_expr("pm").alias("aqi"))
    assert df["aqi"].to_list() == [pm25_to_aqi(v) for v in values]
