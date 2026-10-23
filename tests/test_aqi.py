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


def test_pm10_breakpoints():
    from shared.aqi import to_aqi

    assert [to_aqi(v, "pm10") for v in [0, 19.9, 20, 50, 99.9, 100, 200, 500]] == [
        1,
        1,
        2,
        3,
        3,
        4,
        5,
        5,
    ]


def test_psi_flags_a_shifted_distribution():
    import numpy as np

    from ml.drift import psi, status

    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 5000)
    assert status(psi(ref, rng.normal(0, 1, 2000))) == "ok"
    assert status(psi(ref, rng.normal(1.5, 1, 2000))) == "drift"
