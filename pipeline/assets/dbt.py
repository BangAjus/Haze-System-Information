from pathlib import Path

from dagster import AssetExecutionContext, AssetKey
from dagster_dbt import DagsterDbtTranslator, DbtCliResource, DbtProject, dbt_assets

# ------ PHASE 1: DBT PROJECT --------

DBT_DIR = Path(__file__).resolve().parents[2] / "dbt"

dbt_project = DbtProject(project_dir=DBT_DIR, profiles_dir=DBT_DIR)
# In `dagster dev` this runs `dbt parse` so the manifest is always fresh.
# In Docker the manifest is built at image build time.
dbt_project.prepare_if_dev()


# ------ PHASE 2: MAP DBT NODES TO DAGSTER ASSETS --------


class HazeTranslator(DagsterDbtTranslator):
    def get_asset_key(self, dbt_resource_props):
        # dbt source raw.weather_air_snapshots -> the Dagster asset raw/weather_air_snapshots
        if dbt_resource_props["resource_type"] == "source":
            return AssetKey([dbt_resource_props["source_name"], dbt_resource_props["name"]])
        return super().get_asset_key(dbt_resource_props)

    def get_group_name(self, dbt_resource_props):
        # Models under models/ml/ join the ml group so they run with the ML jobs
        fqn = dbt_resource_props.get("fqn", [])
        if len(fqn) > 1 and fqn[1] == "ml":
            return "ml"
        return "transform"


# ------ PHASE 3: TRANSFORM (DBT BUILD) --------


@dbt_assets(manifest=dbt_project.manifest_path, dagster_dbt_translator=HazeTranslator())
def haze_dbt_models(context: AssetExecutionContext, dbt: DbtCliResource):
    """staging -> core -> marts. dbt tests run as asset checks."""
    yield from dbt.cli(["build"], context=context).stream()
