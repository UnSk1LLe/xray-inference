from dataclasses import dataclass
import os


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


def get_settings() -> Settings:
    return Settings(
        app_name=os.getenv("APP_NAME", "Xray Diagnosis Inference Service"),
        app_env=os.getenv("APP_ENV", "development"),
        app_host=os.getenv("APP_HOST", "0.0.0.0"),
        app_port=int(os.getenv("APP_PORT", "8010")),
        model_name=os.getenv("MODEL_NAME", "placeholder-cnn"),
        use_mock=_as_bool(os.getenv("USE_MOCK"), True),
    )


settings = get_settings()

