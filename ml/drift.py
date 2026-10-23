"""Feature drift: how far recent inputs have moved from what the champion was trained on.

Uses the Population Stability Index (PSI) per feature, with bins taken from the training
(reference) distribution's deciles. Common reading: < 0.1 stable, 0.1-0.25 watch,
> 0.25 drifted (retrain or investigate the data).
"""

import numpy as np

PSI_WATCH = 0.10
PSI_DRIFT = 0.25


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    reference = reference[~np.isnan(reference)]
    current = current[~np.isnan(current)]
    if len(reference) < bins or len(current) < bins:
        return float("nan")
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:  # (nearly) constant feature
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref_share = np.histogram(reference, edges)[0] / len(reference)
    cur_share = np.histogram(current, edges)[0] / len(current)
    eps = 1e-4  # avoid log(0) for empty bins
    ref_share, cur_share = ref_share + eps, cur_share + eps
    return float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))


def status(value: float) -> str:
    if np.isnan(value):
        return "unknown"
    return "drift" if value >= PSI_DRIFT else "watch" if value >= PSI_WATCH else "ok"
