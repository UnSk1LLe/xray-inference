from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class AnalysisResult(BaseModel):
    status: str
    confidence: int = Field(ge=0, le=100)
    findings: list[str]
    recommendations: list[str]
    ai_analysis: str
    raw: Any | None = None


class CreateJobRequest(BaseModel):
    report_id: int
    image_id: int
    object_key: str
    report_type: str = "chest_xray"


class CreateJobResponse(BaseModel):
    job_id: str
    status: str
    result: AnalysisResult | None = None
    message: str | None = None


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    created_at: datetime
    updated_at: datetime
    result: AnalysisResult | None = None
    message: str | None = None

