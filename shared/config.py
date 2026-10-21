from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All runtime config, read from environment variables or a local .env file."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "haze"
    postgres_password: str = ""
    postgres_db: str = "haze"

    kaggle_dataset: str = "xjoannax88/global-weather-and-air-pollution-dataset"
    kaggle_file: str = "output/openweather_weather_airpollution_top3cities_per_country.parquet"

    haze_data_dir: Path = Path("./data")
    mlflow_tracking_uri: str = "http://localhost:5001"
    dagster_ui_url: str = "http://localhost:3000"
    api_admin_key: str = ""

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+psycopg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def raw_dir(self) -> Path:
        return self.haze_data_dir / "raw"


@lru_cache
def get_settings() -> Settings:
    return Settings()
