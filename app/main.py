from typing import Any

from fastapi import FastAPI

from app.api.routes import router as inference_router
from app.core.config import settings
from app.services.inference import inference_service

app = FastAPI(title=settings.app_name)
app.include_router(inference_router)


@app.get("/health", tags=["health"])
def healthcheck() -> dict[str, Any]:
    payload = {
        "status": "ok",
        "environment": settings.app_env,
        **inference_service.describe(),
    }
    return payload
