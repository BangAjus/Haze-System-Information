from typing import Any

from fastapi import HTTPException
from sqlalchemy import text

from shared.db import get_engine


def fetch_all(query: str, **params) -> list[dict[str, Any]]:
    with get_engine().connect() as conn:
        return [dict(r._mapping) for r in conn.execute(text(query), params)]


def fetch_one(query: str, **params) -> dict[str, Any]:
    rows = fetch_all(query, **params)
    if not rows:
        raise HTTPException(status_code=404, detail="Not found")
    return rows[0]
