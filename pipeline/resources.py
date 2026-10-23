from pathlib import Path

from dagster import ConfigurableResource

from shared.config import get_settings


class KaggleResource(ConfigurableResource):
    """Downloads the dataset file and reports its current version fingerprint.

    Reads KAGGLE_USERNAME / KAGGLE_KEY from the environment (or ~/.kaggle/kaggle.json).
    """

    dataset: str = get_settings().kaggle_dataset
    file: str = get_settings().kaggle_file

    def _api(self):
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        return api

    def fingerprint(self) -> str:
        """Changes whenever the author publishes a new version of the file."""
        files = self._api().dataset_list_files(self.dataset).files
        match = next(f for f in files if f.name == self.file)
        created = getattr(match, "creation_date", None) or getattr(match, "creationDate", "")
        return f"{match.name}|{match.total_bytes}|{created}"

    def download(self, dest_dir: Path) -> Path:
        dest_dir.mkdir(parents=True, exist_ok=True)
        self._api().dataset_download_file(self.dataset, self.file, path=str(dest_dir), quiet=True)
        path = dest_dir / Path(self.file).name
        zipped = path.with_name(path.name + ".zip")
        if zipped.exists():
            import zipfile

            with zipfile.ZipFile(zipped) as zf:
                zf.extractall(dest_dir)
            zipped.unlink()
        return path


class OpenWeatherResource(ConfigurableResource):
    """Current Weather and Air Pollution APIs (both on OpenWeather's free tier).

    Paces requests to stay under the free tier's 60 calls/minute, and retries 429s and
    server errors with backoff. Reads OPENWEATHER_API_KEY from the environment.
    """

    api_key: str = get_settings().openweather_api_key
    calls_per_minute: int = 50  # free tier allows 60; keep a margin
    base_url: str = "https://api.openweathermap.org/data/2.5"

    def _get(self, client, path: str, lat: float, lon: float, **extra) -> dict:
        import time

        import httpx

        params = {"lat": lat, "lon": lon, "units": "metric", "appid": self.api_key, **extra}
        for attempt in range(4):
            started = time.monotonic()
            try:
                resp = client.get(f"{self.base_url}/{path}", params=params)
            except httpx.TransportError:
                resp = None
            # Pace every call, successful or not, so retries don't burst past the limit
            time.sleep(max(0.0, 60 / self.calls_per_minute - (time.monotonic() - started)))
            if resp is not None and resp.status_code == 200:
                return resp.json()
            if resp is not None and resp.status_code in (401, 403):
                raise PermissionError(f"OpenWeather rejected the API key (HTTP {resp.status_code})")
            time.sleep(2**attempt)
        raise RuntimeError(f"OpenWeather {path} failed for ({lat}, {lon}) after retries")

    def fetch(self, cities: list[dict], log=None) -> tuple[list[dict], list[dict]]:
        """Weather + air pollution for each city. Returns (results, failures)."""
        import httpx

        if not self.api_key:
            raise ValueError("OPENWEATHER_API_KEY is not set")
        results, failures = [], []
        with httpx.Client(timeout=20) as client:
            for i, city in enumerate(cities, 1):
                try:
                    weather = self._get(client, "weather", city["lat"], city["lon"])
                    air = self._get(client, "air_pollution", city["lat"], city["lon"])
                    results.append({"city": city, "weather": weather, "air": air})
                except PermissionError:
                    raise
                except Exception as exc:  # one bad city shouldn't stop the run
                    failures.append({"geonameid": city["geonameid"], "error": str(exc)})
                if log and i % 100 == 0:
                    log.info(f"Fetched {i}/{len(cities)} cities ({len(failures)} failed)")
        return results, failures

    def air_history(self, client, lat: float, lon: float, start: int, end: int) -> list[dict]:
        """Hourly air pollution between two unix times. One call covers a whole year."""
        if not self.api_key:
            raise ValueError("OPENWEATHER_API_KEY is not set")
        return self._get(client, "air_pollution/history", lat, lon, start=start, end=end)["list"]
