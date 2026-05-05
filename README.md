# Xray Diagnosis Inference

FastAPI scaffold for the future model-hosting service.

Current behavior:

- exposes the endpoints expected by the Go backend
- returns deterministic mock inference results
- keeps an in-memory job registry so the backend can already integrate against stable contracts

## Endpoints

- `GET /health`
- `POST /v1/inference/jobs`
- `GET /v1/inference/jobs/{job_id}`

## Run

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8010
```

