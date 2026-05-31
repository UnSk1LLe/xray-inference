# Xray Diagnosis Inference

FastAPI service for local chest X-ray inference.

Current behavior:

- exposes the endpoints already used by the Go backend
- loads a real PyTorch checkpoint from `xray_diagnosis`
- downloads uploaded X-ray images from MinIO using `object_key`
- returns a platform-compatible result payload with `status`, `confidence`, `findings`, `recommendations`, `ai_analysis`, and raw per-label scores plus model/image-quality metadata for deterministic report generation in the backend
- respects transform settings stored in the training checkpoint, including 640px letterbox preprocessing for newer DenseNet-121 runs

## Endpoints

- `GET /health`
- `POST /v1/inference/jobs`
- `GET /v1/inference/jobs/{job_id}`

## Required env

- `MODEL_CHECKPOINT_PATH`: path to `best_checkpoint.pt` or an exported inference bundle
- `MODEL_VERSION`: optional readable model/version identifier stored in generated reports
- `S3_ENDPOINT`, `S3_ACCESS_KEY`, `S3_SECRET_KEY`, `S3_BUCKET`, `S3_USE_SSL`: same storage values used by the Go backend
- `USE_MOCK=false` to enable real inference

The local `.env` in this workspace is already pointed at the latest April 19, 2026 stage-2 checkpoint.

## Run

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8010
```

## Local integration flow

1. Start MinIO, Postgres, and Redis from `xray-diagnosis-platform/docker-compose.yml`.
2. Run this FastAPI service on port `8010`.
3. In `xray-diagnosis-platform/backend/.env`, set `INFERENCE_USE_MOCK=false`.
4. Start the Go API and the Go worker.
5. Upload an X-ray from the mobile app and open the generated report once the worker completes.
