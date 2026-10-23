import shutil
import sys
from pathlib import Path

from dagster import Definitions, load_asset_checks_from_modules, load_assets_from_modules
from dagster_dbt import DbtCliResource

from pipeline import jobs, sensors
from pipeline.assets import dbt, ingest, ml, ml_hourly, openweather
from pipeline.resources import KaggleResource, OpenWeatherResource

# Prefer the dbt installed next to this Python (the .venv), even if the venv isn't activated
_venv_dbt = Path(sys.executable).parent / "dbt"
DBT_EXECUTABLE = str(_venv_dbt) if _venv_dbt.exists() else (shutil.which("dbt") or "dbt")

defs = Definitions(
    assets=load_assets_from_modules([ingest, openweather, dbt, ml, ml_hourly]),
    asset_checks=load_asset_checks_from_modules([ingest, openweather, ml]),
    jobs=[
        jobs.daily_ingest,
        jobs.hourly_openweather,
        jobs.backfill_air_history,
        jobs.daily_ml,
        jobs.weekly_retrain,
    ],
    schedules=[
        jobs.daily_ingest_fallback,
        jobs.hourly_openweather_schedule,
        jobs.daily_ml_schedule,
        jobs.weekly_retrain_schedule,
    ],
    sensors=[sensors.kaggle_update_sensor, sensors.record_run_success, sensors.record_run_failure],
    resources={
        "kaggle": KaggleResource(),
        "openweather": OpenWeatherResource(),
        "dbt": DbtCliResource(project_dir=dbt.dbt_project, dbt_executable=DBT_EXECUTABLE),
    },
)
