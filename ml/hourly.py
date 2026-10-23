"""Hourly track: forecast a pollutant 24 hours ahead from the last 72 hours.

Input is core.fct_air_hourly (one row per city per hour, all 8 pollutants) joined with
core.dim_city. Weather isn't used here: the free tier has no weather history, and the
long backfill only covers pollutants.

Candidates:
  - hgb_lags: gradient boosting on hand-made lag/rolling features from the same window
  - lstm, gru: recurrent networks reading the raw 72-hour sequence (PyTorch, CPU)
Baselines:
  - persistence: the value now (24h before the target, i.e. same hour of day)
  - mean_24h: the average of the last 24 hours

Every model predicts log1p(value). The p10-p90 range uses the same split-conformal method
as the daily track: train on the older 90% of anchors, measure errors on the newest 10%.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
import polars as pl

from shared.aqi import to_aqi

# ------ PHASE 0: SETTINGS --------

WINDOW = 72  # hours of history the models see
HORIZON = 24  # hours ahead to predict
MAX_FILL_GAP = 3  # forward-fill gaps up to this many hours; longer gaps break windows
POLLUTANTS = ["pm2_5", "pm10", "no2", "o3", "co", "so2", "nh3", "no"]
TARGET_COLUMN = {"pm25": "pm2_5", "pm10": "pm10"}
CONTINENTS = ["AF", "AN", "AS", "EU", "NA", "OC", "SA"]
CALIBRATION_SHARE = 0.10


# ------ PHASE 1: BUILD THE HOURLY GRID --------


@dataclass
class HourlyData:
    """All cities' hourly series concatenated into one array, with per-row lookups.

    A window for anchor row g is rows g-WINDOW+1 .. g; its target is row g+HORIZON.
    `valid` marks anchors whose whole window and target exist within the same city.
    """

    seq: np.ndarray  # (rows, channels) float32: log1p pollutants + time-of-day/week
    static: np.ndarray  # (rows, static_channels) float32: location, continent, population
    raw: dict[str, np.ndarray]  # untransformed PM2.5/PM10 values, for targets and baselines
    hour: np.ndarray  # (rows,) datetime64[us] UTC hour
    geonameid: np.ndarray  # (rows,)
    valid: np.ndarray  # (rows,) bool, anchor usable for training/eval
    latest: np.ndarray  # (cities,) row index of each city's newest complete window


def build_hourly(df: pl.DataFrame) -> HourlyData:
    """df: fct_air_hourly joined with lat, lon, continent, population."""

    # job 1: complete hourly grid per city, so window positions mean real hours
    df = df.sort("geonameid", "hour_utc")
    grid = (
        df.select("geonameid", "hour_utc")
        .group_by("geonameid")
        .agg(pl.datetime_ranges(pl.col("hour_utc").min(), pl.col("hour_utc").max(), "1h"))
        .explode("hour_utc")
    )
    # job 2: join readings onto the grid and forward-fill gaps up to MAX_FILL_GAP hours
    static_cols = ["lat", "lon", "continent", "population"]
    statics = df.group_by("geonameid").agg(pl.col(static_cols).last())
    full = (
        grid.join(
            df.select("geonameid", "hour_utc", pl.col(POLLUTANTS).cast(pl.Float32)),
            on=["geonameid", "hour_utc"],
            how="left",
        )
        .join(statics, on="geonameid")
        .sort("geonameid", "hour_utc")
        .with_columns(
            pl.col(POLLUTANTS).fill_null(strategy="forward", limit=MAX_FILL_GAP).over("geonameid")
        )
    )

    del df, grid  # free the source tables as soon as the grid is joined

    # job 3: sequence channels (log1p pollutants + time of day/week) as float32
    n = full.height
    hours = full["hour_utc"]
    hod = hours.dt.hour().to_numpy().astype(np.float32) * np.float32(2 * np.pi / 24)
    dow = hours.dt.weekday().to_numpy().astype(np.float32) * np.float32(2 * np.pi / 7)
    # Built straight into float32 arrays: float64 intermediates doubled peak memory
    seq = np.empty((n, len(POLLUTANTS) + 4), dtype=np.float32)
    for i, p in enumerate(POLLUTANTS):
        seq[:, i] = np.log1p(np.clip(full[p].to_numpy().astype(np.float32), 0, None))
    seq[:, -4:] = np.column_stack([np.sin(hod), np.cos(hod), np.sin(dow), np.cos(dow)])
    logp = seq[:, : len(POLLUTANTS)]

    # job 4: static city features (location, population, continent)
    cont = full["continent"].to_numpy()
    static = np.empty((n, 3 + len(CONTINENTS)), dtype=np.float32)
    static[:, 0] = full["lat"].to_numpy() / 90
    static[:, 1] = full["lon"].to_numpy() / 180
    static[:, 2] = (np.log1p(full["population"].to_numpy().astype(np.float64)) - 13) / 3
    for i, c in enumerate(CONTINENTS):
        static[:, 3 + i] = cont == c

    raw = {p: full[p].to_numpy().astype(np.float64) for p in TARGET_COLUMN.values()}
    city = full["geonameid"].to_numpy()
    hours_np = hours.to_numpy()
    del full, hours, cont

    # job 5: window validity: WINDOW complete rows ending here, and a target HORIZON later,
    # all within the same city
    row_ok = ~np.isnan(logp).any(axis=1)
    new_city = np.r_[True, city[1:] != city[:-1]]
    pos = np.arange(len(city)) - np.maximum.accumulate(np.where(new_city, np.arange(len(city)), 0))
    ok_run = pl.Series(row_ok.astype(np.int32)).rolling_sum(WINDOW).fill_null(0).to_numpy()
    window_ok = (ok_run == WINDOW) & (pos >= WINDOW - 1)
    target_idx = np.arange(len(city)) + HORIZON
    same_city = np.zeros(len(city), bool)
    in_range = target_idx < len(city)
    same_city[in_range] = city[target_idx[in_range]] == city[in_range]
    target_ok = np.zeros(len(city), bool)
    target_ok[in_range] = row_ok[target_idx[in_range]]
    valid = window_ok & same_city & target_ok

    # job 6: newest complete window per city (for forecasting; no target needed)
    last_rows = np.r_[np.where(new_city)[0][1:] - 1, len(city) - 1]
    latest = np.array([r for r in last_rows if window_ok[r]], dtype=np.int64)

    return HourlyData(
        seq=seq,
        static=static,
        raw=raw,
        hour=hours_np,
        geonameid=city,
        valid=valid,
        latest=latest,
    )


def anchors(
    data: HourlyData, start: datetime | None, end: datetime | None, stride: int = 6
) -> np.ndarray:
    """Valid anchor rows with start <= hour < end, every `stride` hours (by hour of day)."""
    mask = data.valid.copy()
    hours = data.hour.astype("datetime64[h]").astype(np.int64)
    mask &= hours % stride == 0
    if start is not None:
        mask &= data.hour >= np.datetime64(start.replace(tzinfo=None), "us")
    if end is not None:
        mask &= data.hour < np.datetime64(end.replace(tzinfo=None), "us")
    return np.where(mask)[0]


def windows(data: HourlyData, rows: np.ndarray) -> np.ndarray:
    """(n, WINDOW, channels) sequences ending at each anchor row."""
    idx = rows[:, None] + np.arange(-WINDOW + 1, 1)[None, :]
    return data.seq[idx]


def targets(data: HourlyData, rows: np.ndarray, target: str) -> np.ndarray:
    return data.raw[TARGET_COLUMN[target]][rows + HORIZON]


# ------ PHASE 2: LAG FEATURES FOR GRADIENT BOOSTING --------


LAG_HOURS = [0, 1, 2, 3, 6, 12, 24, 48, 71]


def lag_features(data: HourlyData, rows: np.ndarray, chunk: int = 50_000) -> np.ndarray:
    """Tabular features from the same 72h window the RNNs see.

    Built in chunks: materializing every window at once would take gigabytes.
    """
    if len(rows) > chunk:
        return np.vstack(
            [lag_features(data, rows[i : i + chunk]) for i in range(0, len(rows), chunk)]
        )
    w = windows(data, rows)  # log1p pollutants in channels 0..7, time features after
    pm = w[:, :, :2]  # pm2_5, pm10
    parts = [pm[:, WINDOW - 1 - lag, :] for lag in LAG_HOURS]
    parts += [pm[:, -24:, :].mean(axis=1), pm[:, :, :].mean(axis=1), pm[:, -24:, :].std(axis=1)]
    parts += [w[:, -1, 2 : len(POLLUTANTS)]]  # other pollutants now
    parts += [w[:, -1, len(POLLUTANTS) :]]  # time of day / week now
    parts += [data.static[rows]]
    return np.column_stack(parts)


# ------ PHASE 3: MODELS (HGB, LSTM, GRU) --------


@dataclass
class HourlyModel:
    name: str
    target: str
    estimator: object  # sklearn regressor or a torch module wrapper
    resid_lo: float
    resid_hi: float
    meta: dict = field(default_factory=dict)
    feature_cols: str = "hourly-window-72h"  # marker checked by the registry

    def predict_log(self, data: HourlyData, rows: np.ndarray) -> np.ndarray:
        return self.estimator.predict(data, rows)

    def predict(self, data: HourlyData, rows: np.ndarray) -> dict[str, np.ndarray]:
        log_pred = self.predict_log(data, rows)
        return {
            "pred": np.expm1(log_pred).clip(min=0),
            "p10": np.expm1(log_pred + min(self.resid_lo, 0)).clip(min=0),
            "p90": np.expm1(log_pred + max(self.resid_hi, 0)).clip(min=0),
        }


class LagHGB:
    """HistGradientBoosting on lag features."""

    def __init__(self):
        from sklearn.ensemble import HistGradientBoostingRegressor

        self.model = HistGradientBoostingRegressor(
            loss="absolute_error", max_iter=300, learning_rate=0.08, random_state=0
        )

    def fit(self, data, rows, y_log, val_rows=None, val_y=None):
        self.model.fit(lag_features(data, rows), y_log)
        return self

    def predict(self, data, rows):
        return self.model.predict(lag_features(data, rows))


class RNNRegressor:
    """LSTM or GRU over the 72h sequence, combined with static city features.

    Trained with L1 loss on log1p(value) (a median forecast, like the other candidates),
    Adam, and early stopping on the validation anchors.
    """

    def __init__(
        self,
        cell: str,
        hidden: int = 64,
        epochs: int = 6,
        batch: int = 512,
        lr: float = 2e-3,
        max_train: int = 200_000,
        seed: int = 0,
    ):
        self.cell, self.hidden, self.epochs = cell, hidden, epochs
        self.batch, self.lr, self.max_train, self.seed = batch, lr, max_train, seed
        self.net = None

    def _build(self, n_seq: int, n_static: int):
        import torch
        from torch import nn

        rnn_cls = nn.LSTM if self.cell == "lstm" else nn.GRU

        class Net(nn.Module):
            def __init__(s):
                super().__init__()
                s.rnn = rnn_cls(n_seq, self.hidden, batch_first=True)
                s.head = nn.Sequential(
                    nn.Linear(self.hidden + n_static, 32), nn.ReLU(), nn.Linear(32, 1)
                )

            def forward(s, x, z):
                out, _ = s.rnn(x)
                return s.head(torch.cat([out[:, -1, :], z], dim=1)).squeeze(1)

        return Net()

    def _batches(self, data, rows, y=None, shuffle=False, rng=None):
        import torch

        order = rng.permutation(len(rows)) if shuffle else np.arange(len(rows))
        for i in range(0, len(rows), self.batch):
            idx = rows[order[i : i + self.batch]]
            x = (windows(data, idx) - self.mu) / self.sd
            z = data.static[idx]
            yy = None if y is None else torch.from_numpy(y[order[i : i + self.batch]]).float()
            yield torch.from_numpy(x).float(), torch.from_numpy(z).float(), yy

    def fit(self, data, rows, y_log, val_rows=None, val_y=None):
        import torch

        # job 1: seed, and cap the training anchors to keep CPU time reasonable
        torch.manual_seed(self.seed)
        torch.set_num_threads(4)
        rng = np.random.default_rng(self.seed)
        if len(rows) > self.max_train:
            keep = rng.choice(len(rows), self.max_train, replace=False)
            rows, y_log = rows[keep], y_log[keep]
        # job 2: standardize sequence channels with training statistics
        sample = windows(data, rows[: min(len(rows), 20_000)]).reshape(-1, data.seq.shape[1])
        self.mu, self.sd = sample.mean(0), sample.std(0) + 1e-6

        # job 3: train with L1 loss, early-stopping on the validation anchors
        self.net = self._build(data.seq.shape[1], data.static.shape[1])
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        loss_fn = torch.nn.L1Loss()
        best, best_state, patience = np.inf, None, 2
        for _ in range(self.epochs):
            self.net.train()
            for x, z, y in self._batches(data, rows, y_log, shuffle=True, rng=rng):
                opt.zero_grad()
                loss_fn(self.net(x, z), y).backward()
                opt.step()
            if val_rows is None or len(val_rows) == 0:
                continue
            val_loss = float(np.mean(np.abs(self.predict(data, val_rows) - val_y)))
            if val_loss < best - 1e-4:
                best, patience = val_loss, 2
                best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
            else:
                patience -= 1
                if patience == 0:
                    break
        # job 4: keep the best epoch's weights
        if best_state is not None:
            self.net.load_state_dict(best_state)
        return self

    def predict(self, data, rows):
        import torch

        self.net.eval()
        out = []
        with torch.no_grad():
            for x, z, _ in self._batches(data, rows):
                out.append(self.net(x, z).numpy())
        return np.concatenate(out) if out else np.array([])


HOURLY_CANDIDATES: dict[str, Callable[[], object]] = {
    "hgb_lags": LagHGB,
    "lstm": lambda: RNNRegressor("lstm"),
    "gru": lambda: RNNRegressor("gru"),
}


def fit_hourly(data: HourlyData, rows: np.ndarray, name: str, target: str) -> HourlyModel:
    """Fit on the older 90% of anchors; the newest 10% calibrate the range (and early-stop
    the RNNs). One fit per call keeps RNN training time reasonable on CPU."""

    # job 1: older 90% to fit, newest 10% to calibrate
    rows = np.sort(rows)
    n_cal = max(1, int(len(rows) * CALIBRATION_SHARE))
    fit_rows, cal_rows = rows[:-n_cal], rows[-n_cal:]
    y_fit = np.log1p(targets(data, fit_rows, target))
    y_cal = np.log1p(targets(data, cal_rows, target))
    # job 2: fit, then turn calibration errors into p10/p90 offsets
    est = HOURLY_CANDIDATES[name]().fit(data, fit_rows, y_fit, cal_rows, y_cal)
    resid = y_cal - est.predict(data, cal_rows)
    lo, hi = np.quantile(resid, [0.1, 0.9])
    return HourlyModel(name, target, est, float(lo), float(hi))


# ------ PHASE 4: EVALUATION --------


def score(actual: np.ndarray, pred: np.ndarray, target: str) -> dict[str, float]:
    aqi_a = np.array([to_aqi(v, target) for v in actual])
    aqi_p = np.array([to_aqi(v, target) for v in pred])
    return {
        "mae": float(np.mean(np.abs(actual - pred))),
        "rmse": float(np.sqrt(np.mean((actual - pred) ** 2))),
        "aqi_acc": float(np.mean(aqi_a == aqi_p)),
    }


def backtest_hourly(
    data: HourlyData,
    target: str,
    candidates: list[str],
    n_folds: int = 2,
    test_days: int = 7,
    train_stride: int = 6,
    test_stride: int = 3,
) -> pl.DataFrame:
    """Expanding-window backtest over the newest weeks. One row per fold per model."""
    # job 1: newest usable hour anchors the folds
    end = data.hour[data.valid].max().astype("datetime64[us]").item()
    col = TARGET_COLUMN[target]
    rows = []
    for k in range(n_folds, 0, -1):
        # job 2: fold k: train anchors whose targets land before the test window
        test_start = end - timedelta(days=test_days * k)
        test_end = test_start + timedelta(days=test_days)
        train = anchors(data, None, test_start - timedelta(hours=HORIZON), train_stride)
        test = anchors(data, test_start, test_end, test_stride)
        if len(train) == 0 or len(test) == 0:
            continue
        actual = targets(data, test, target)
        fold = dict(fold=k, test_start=test_start, n_test=len(test))

        # job 3: score the two baselines
        persistence = data.raw[col][test]
        ch = POLLUTANTS.index(col)
        mean_24h = np.expm1(
            np.concatenate(
                [
                    windows(data, test[i : i + 50_000])[:, -24:, ch].mean(axis=1)
                    for i in range(0, len(test), 50_000)
                ]
            )
        )
        rows.append({"model": "persistence", **fold, **score(actual, persistence, target)})
        rows.append({"model": "mean_24h", **fold, **score(actual, mean_24h, target)})

        # job 4: fit and score every candidate
        for name in candidates:
            preds = fit_hourly(data, train, name, target).predict(data, test)
            covered = (actual >= preds["p10"]) & (actual <= preds["p90"])
            rows.append(
                {
                    "model": name,
                    **fold,
                    **score(actual, preds["pred"], target),
                    "interval_coverage": float(covered.mean()),
                }
            )
    return pl.DataFrame(rows)
