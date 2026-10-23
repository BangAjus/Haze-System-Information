"""The select-and-register step shared by every training asset (daily and hourly tracks).

Given a backtest summary, it logs every candidate to MLflow and ml.model_candidates, fits the
best passing candidate on all data, and promotes it if it beats the current champion.
"""

from collections.abc import Callable
from datetime import UTC, datetime

import polars as pl
from dagster import AssetExecutionContext, MaterializeResult

from ml.registry import (
    METRIC_KEYS,
    current_champion,
    load_model,
    log_candidates,
    record_run,
    save_model,
)
from ml.train import pick_best, should_promote
from shared.config import get_settings


def select_and_register(
    context: AssetExecutionContext,
    *,
    track: str,
    target: str,
    summary: pl.DataFrame,
    candidates: list[str],
    fit_winner: Callable[[str], tuple[object, object, object]],
    model_type_check: type,
) -> MaterializeResult:
    """fit_winner(name) -> (model, train_start, train_end), trained on all data."""
    import mlflow

    trained_at = datetime.now(UTC)
    stamp = trained_at.strftime("%Y%m%d%H%M")
    context.log.info(f"{track}/{target} backtest (mean of folds, best first):\n{summary}")
    best = pick_best(summary, candidates)

    mlflow.set_tracking_uri(get_settings().mlflow_tracking_uri)
    mlflow.set_experiment(f"{target}_{track}_forecast")
    with mlflow.start_run(run_name=f"train-{stamp}") as parent:
        # One child run per candidate, so MLflow shows them side by side
        for row in summary.iter_rows(named=True):
            with mlflow.start_run(run_name=row["model"], nested=True):
                mlflow.log_metrics({k: v for k, v in row.items() if isinstance(v, float)})
        log_candidates(
            summary, track, target, trained_at, parent.info.run_id, best and best["model"]
        )
        if best is None:
            mlflow.log_param("selected", "none")
            context.log.warning("No candidate beat persistence with an honest range.")
            return MaterializeResult(metadata={"selected": "none"})

        model, train_start, train_end = fit_winner(best["model"])
        version = f"{target}-{track}-{best['model'].replace('_', '')}-{stamp}"
        path = save_model(model, version)

        champion = current_champion(track, target)
        if champion and load_model(champion["model_version"], model_type_check) is None:
            context.log.warning(f"Champion {champion['model_version']} is unloadable; replacing.")
            champion = None
        promote = should_promote(best, champion)
        mlflow.log_params(
            {"selected": best["model"], "model_version": version, "promoted": promote}
        )
        mlflow.log_artifact(str(path))

    metrics = {k: best.get(k) for k in METRIC_KEYS}
    record_run(
        version=version,
        model_type=best["model"],
        track=track,
        target=target,
        mlflow_run_id=parent.info.run_id,
        trained_at=trained_at,
        train_start=train_start,
        train_end=train_end,
        metrics=metrics,
        promote=promote,
    )
    meta = {"model_version": version, "selected": best["model"], "promoted": promote}
    return MaterializeResult(metadata={**meta, **{k: v for k, v in metrics.items() if v}})
