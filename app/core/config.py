from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv

load_dotenv()


def _as_bool(value: str, default: bool = True) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    app_name: str
    app_env: str
    app_host: str
    app_port: int
    model_name: str
    use_mock: bool
    model_checkpoint_path: Path | None
    inference_device: str
    top_k_findings: int
    s3_endpoint: str
    s3_access_key: str
    s3_secret_key: str
    s3_bucket: str
    s3_use_ssl: bool


def get_settings() -> Settings:
    checkpoint_path = os.getenv("MODEL_CHECKPOINT_PATH")
    return Settings(
        app_name=os.getenv("APP_NAME", "Xray Diagnosis Inference Service"),
        app_env=os.getenv("APP_ENV", "development"),
        app_host=os.getenv("APP_HOST", "0.0.0.0"),
        app_port=int(os.getenv("APP_PORT", "8010")),
        model_name=os.getenv("MODEL_NAME", "placeholder-cnn"),
        use_mock=_as_bool(os.getenv("USE_MOCK"), True),
        model_checkpoint_path=Path(checkpoint_path).expanduser() if checkpoint_path else None,
        inference_device=os.getenv("INFERENCE_DEVICE", "auto").strip().lower(),
        top_k_findings=max(1, int(os.getenv("TOP_K_FINDINGS", "5"))),
        s3_endpoint=os.getenv("S3_ENDPOINT", "localhost:9000"),
        s3_access_key=os.getenv("S3_ACCESS_KEY", "minioadmin"),
        s3_secret_key=os.getenv("S3_SECRET_KEY", "minioadmin"),
        s3_bucket=os.getenv("S3_BUCKET", "xray-images"),
        s3_use_ssl=_as_bool(os.getenv("S3_USE_SSL"), False),
    )


settings = get_settings()
