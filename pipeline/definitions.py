import shutil
import sys
from pathlib import Path

from dagster import Definitions, load_asset_checks_from_modules, load_assets_from_modules
from dagster_dbt import DbtCliResource

from pipeline import jobs, sensors
from pipeline.assets import dbt, ingest, ml
from pipeline.resources import KaggleResource

# Prefer the dbt installed next to this Python (the .venv), even if the venv isn't activated
_venv_dbt = Path(sys.executable).parent / "dbt"
DBT_EXECUTABLE = str(_venv_dbt) if _venv_dbt.exists() else (shutil.which("dbt") or "dbt")

defs = Definitions(
    assets=load_assets_from_modules([ingest, dbt, ml]),
    asset_checks=load_asset_checks_from_modules([ingest, ml]),
    jobs=[jobs.daily_ingest, jobs.daily_ml, jobs.weekly_retrain],
    schedules=[jobs.daily_ingest_fallback, jobs.daily_ml_schedule, jobs.weekly_retrain_schedule],
    sensors=[sensors.kaggle_update_sensor, sensors.record_run_success, sensors.record_run_failure],
    resources={
        "kaggle": KaggleResource(),
        "dbt": DbtCliResource(project_dir=dbt.dbt_project, dbt_executable=DBT_EXECUTABLE),
    },
)
