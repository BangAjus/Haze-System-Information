# MLflow server with the Postgres driver baked in, so startup doesn't depend on PyPI.
# Keep the version in step with mlflow-skinny in pyproject.toml.
FROM ghcr.io/mlflow/mlflow:v3.17.1
RUN pip install --no-cache-dir --default-timeout=120 "psycopg[binary]>=3.2"
