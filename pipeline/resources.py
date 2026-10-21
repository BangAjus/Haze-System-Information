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
