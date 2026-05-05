from fastapi import APIRouter, HTTPException

from app.schemas.inference import CreateJobRequest, CreateJobResponse, JobStatusResponse
from app.services.inference import inference_service

router = APIRouter(prefix="/v1/inference", tags=["inference"])


@router.post("/jobs", response_model=CreateJobResponse)
def create_inference_job(payload: CreateJobRequest) -> CreateJobResponse:
    return inference_service.create_job(payload)


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
def get_inference_job(job_id: str) -> JobStatusResponse:
    job = inference_service.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return job

