from functools import lru_cache

import polars as pl
from sqlalchemy import Engine, create_engine

from shared.config import get_settings


@lru_cache
def get_engine() -> Engine:
    return create_engine(get_settings().database_url, pool_pre_ping=True)


def read_sql(query: str, **params) -> pl.DataFrame:
    """Run a query and return a Polars frame."""
    from sqlalchemy import text

    with get_engine().connect() as conn:
        result = conn.execute(text(query), params)
        rows = result.fetchall()
        df = pl.DataFrame(rows, schema=list(result.keys()), orient="row", infer_schema_length=None)
    # Postgres numeric arrives as Decimal; the rest of the code expects floats
    return df.with_columns(pl.col(pl.Decimal).cast(pl.Float64))


def write_table(df: pl.DataFrame, table: str, schema: str, if_exists: str = "replace") -> int:
    """Write a frame to Postgres. Returns the number of rows written."""
    df.write_database(
        f"{schema}.{table}",
        connection=get_engine(),
        if_table_exists=if_exists,
        engine="sqlalchemy",
    )
    return df.height


def replace_rows(df: pl.DataFrame, table: str, schema: str) -> int:
    """Swap a table's contents in one transaction, keeping the table itself.

    Unlike write_table(..., "replace"), this never drops the table, so views built on it
    (like dbt's staging views) keep working. Readers see the old rows until the commit.
    """
    from sqlalchemy import text

    name = f"{schema}.{table}"
    with get_engine().begin() as conn:
        exists = conn.execute(text("select to_regclass(:t)"), {"t": name}).scalar() is not None
        if exists:
            conn.execute(text(f"truncate table {name}"))
        df.write_database(
            name,
            connection=conn,
            if_table_exists="append" if exists else "replace",
            engine="sqlalchemy",
        )
    return df.height
