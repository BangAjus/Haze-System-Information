from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import cities, health, ml, ops

# ------ PHASE 1: APP AND ROUTERS --------

app = FastAPI(
    title="Haze API",
    version="0.1.0",
    description="City air quality, forecasts and pipeline health. "
    "Data: OpenWeather and GeoNames via Kaggle (CC BY 4.0).",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3001"],  # Next.js dev server
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

for router in (health.router, cities.router, ops.router, ml.router):
    app.include_router(router, prefix="/v1")
