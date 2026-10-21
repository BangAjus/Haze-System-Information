.PHONY: setup up down db logs dev api dbt test lint

setup:            ## Create .venv and install everything
	python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"

up:               ## Start Postgres, MLflow, Dagster and the API in Docker
	docker compose --env-file .env -f infra/docker-compose.yml up -d --build

down:
	docker compose --env-file .env -f infra/docker-compose.yml down

logs:             ## Follow the Docker stack's logs
	docker compose --env-file .env -f infra/docker-compose.yml logs -f --tail=100

db:               ## Start only Postgres and MLflow (for local dev)
	docker compose --env-file .env -f infra/docker-compose.yml up -d postgres mlflow

dev:              ## Run Dagster locally against the Docker Postgres (UI on :3000)
	set -a && . ./.env && set +a && .venv/bin/dagster dev

api:              ## Run the API locally with reload (docs on :8000/docs)
	set -a && . ./.env && set +a && .venv/bin/uvicorn api.main:app --reload

dbt:              ## Run dbt build directly
	set -a && . ./.env && set +a && cd dbt && ../.venv/bin/dbt build --profiles-dir .

test:
	.venv/bin/pytest -q

lint:
	.venv/bin/ruff check . && .venv/bin/ruff format --check .
