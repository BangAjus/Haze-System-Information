FROM python:3.12-slim

WORKDIR /opt/haze
ENV PYTHONUNBUFFERED=1 \
    DAGSTER_HOME=/opt/dagster/home

# CPU-only PyTorch first (the default wheel pulls in ~2 GB of CUDA libraries we don't use).
# Its own layer, so code changes don't reinstall it.
RUN pip install --no-cache-dir --default-timeout=300 --retries 5 "torch>=2.4" \
    --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md ./
COPY shared ./shared
COPY pipeline ./pipeline
COPY ml ./ml
COPY api ./api
COPY dbt ./dbt
RUN pip install --no-cache-dir --default-timeout=300 --retries 5 ".[pipeline,ml,api]"

# Build the dbt manifest that dagster-dbt reads at load time
RUN cd dbt && dbt parse --profiles-dir . --target-path target

RUN mkdir -p $DAGSTER_HOME
COPY infra/dagster/dagster.yaml infra/dagster/workspace.yaml $DAGSTER_HOME/
