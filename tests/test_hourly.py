from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from ml import hourly as H


def make_hourly(n_cities: int = 3, n_hours: int = 24 * 20, gap_at: int | None = None):
    """Synthetic hourly series with a daily cycle, so there is something to learn."""
    rng = np.random.default_rng(0)
    start = datetime(2026, 6, 1, tzinfo=UTC)
    rows = []
    for city in range(n_cities):
        base = rng.uniform(5, 30)
        for h in range(n_hours):
            if gap_at is not None and city == 0 and gap_at <= h < gap_at + 10:
                continue  # a 10-hour gap: longer than MAX_FILL_GAP
            pm = base * (1 + 0.4 * np.sin(2 * np.pi * h / 24)) + rng.normal(0, 1)
            rows.append(
                dict(
                    geonameid=city,
                    hour_utc=start + timedelta(hours=h),
                    pm2_5=max(pm, 0.1),
                    pm10=max(pm * 2, 0.1),
                    no2=10.0,
                    o3=50.0,
                    co=200.0,
                    so2=3.0,
                    nh3=2.0,
                    no=0.5,
                    lat=1.0 + city,
                    lon=100.0,
                    continent="AS",
                    population=1_000_000,
                )
            )
    return pl.DataFrame(rows)


def test_targets_are_exactly_horizon_hours_ahead():
    data = H.build_hourly(make_hourly())
    rows = np.where(data.valid)[0]
    lead = (data.hour[rows + H.HORIZON] - data.hour[rows]).astype("timedelta64[h]").astype(int)
    assert (lead == H.HORIZON).all()
    assert (data.geonameid[rows + H.HORIZON] == data.geonameid[rows]).all()


def test_long_gaps_invalidate_windows_but_short_ones_are_filled():
    data = H.build_hourly(make_hourly(gap_at=200))
    city0 = data.geonameid == 0
    hours_since_start = np.arange(len(data.hour))[city0]
    # The gap is hours 200-209; the first MAX_FILL_GAP hours are forward-filled, so 203-209
    # stay missing. Anchors whose window or target touches those hours are invalid...
    first_missing, last_missing = 200 + H.MAX_FILL_GAP, 209
    h = hours_since_start
    target_in_gap = (h + H.HORIZON >= first_missing) & (h + H.HORIZON <= last_missing)
    window_in_gap = (h >= first_missing) & (h - H.WINDOW + 1 <= last_missing)
    assert not data.valid[city0][target_in_gap | window_in_gap].any()
    # A window wholly before the gap with a target after it is fine
    assert data.valid[city0][last_missing + 1 - H.HORIZON]
    # ...while an anchor whose target lands on a filled hour is still usable
    assert data.valid[city0][200 - H.HORIZON]
    assert data.valid[city0].any()


def test_windows_have_the_right_shape():
    data = H.build_hourly(make_hourly())
    rows = H.anchors(data, None, None, stride=6)[:10]
    assert H.windows(data, rows).shape == (10, H.WINDOW, data.seq.shape[1])


@pytest.mark.parametrize("name", list(H.HOURLY_CANDIDATES))
def test_every_hourly_candidate_fits_and_predicts(name, monkeypatch):
    if name in ("lstm", "gru"):
        monkeypatch.setitem(H.HOURLY_CANDIDATES, name, lambda n=name: H.RNNRegressor(n, epochs=1))
    data = H.build_hourly(make_hourly())
    rows = H.anchors(data, None, None, stride=3)
    model = H.fit_hourly(data, rows, name, "pm25")
    preds = model.predict(data, rows[:20])
    assert len(preds["pred"]) == 20
    assert (preds["p10"] <= preds["pred"]).all() and (preds["pred"] <= preds["p90"]).all()


def test_backtest_includes_baselines():
    data = H.build_hourly(make_hourly())
    res = H.backtest_hourly(data, "pm10", ["hgb_lags"], n_folds=1, test_days=3)
    assert {"persistence", "mean_24h", "hgb_lags"} == set(res["model"])
