from fastapi import APIRouter

from api.db import fetch_one

# ------ PHASE 1: HEALTH CHECK --------

router = APIRouter(tags=["health"])


@router.get("/health")
def health():
    fetch_one("select 1 as ok")
    return {"status": "ok"}
