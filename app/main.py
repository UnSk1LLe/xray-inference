from fastapi import FastAPI

from app.api.routes import router as inference_router
from app.core.config import settings

app = FastAPI(title=settings.app_name)
app.include_router(inference_router)


@app.get("/health", tags=["health"])
def healthcheck() -> dict[str, str]:
    return {
        "status": "ok",
        "environment": settings.app_env,
        "model_name": settings.model_name,
    }

