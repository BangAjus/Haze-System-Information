FROM python:3.12-slim

WORKDIR /opt/haze
ENV PYTHONUNBUFFERED=1 \
    DAGSTER_HOME=/opt/dagster/home

COPY pyproject.toml README.md ./
COPY shared ./shared
COPY pipeline ./pipeline
COPY ml ./ml
COPY api ./api
COPY dbt ./dbt
RUN pip install --no-cache-dir ".[pipeline,ml,api]"

# Build the dbt manifest that dagster-dbt reads at load time
RUN cd dbt && dbt parse --profiles-dir . --target-path target

RUN mkdir -p $DAGSTER_HOME
COPY infra/dagster/dagster.yaml infra/dagster/workspace.yaml $DAGSTER_HOME/
