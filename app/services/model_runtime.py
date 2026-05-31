from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import torch
from minio import Minio
from minio.error import S3Error
from PIL import Image
from torch import nn
from torchvision import models, transforms
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode

from app.core.config import Settings

DEFAULT_MEAN = (0.485, 0.456, 0.406)
DEFAULT_STD = (0.229, 0.224, 0.225)
NIH_TARGET_LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Effusion",
    "Emphysema",
    "Fibrosis",
    "Hernia",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pleural_Thickening",
    "Pneumonia",
    "Pneumothorax",
]


@dataclass(slots=True)
class ModelMetadata:
    model_name: str
    checkpoint_path: str
    backbone: str
    device: str
    num_classes: int
    image_size: int
    resize_mode: str
    pad_fill: int
    label_names: list[str]
    thresholds: list[float]


class LetterboxSquare:
    def __init__(
        self,
        size: int,
        *,
        fill: int = 0,
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
    ) -> None:
        self.size = int(size)
        self.fill = int(fill)
        self.interpolation = interpolation

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        if width <= 0 or height <= 0:
            return TF.resize(
                image,
                [self.size, self.size],
                interpolation=self.interpolation,
                antialias=True,
            )

        scale = min(self.size / float(width), self.size / float(height))
        resized_width = max(1, int(round(width * scale)))
        resized_height = max(1, int(round(height * scale)))
        resized = TF.resize(
            image,
            [resized_height, resized_width],
            interpolation=self.interpolation,
            antialias=True,
        )

        background = Image.new(image.mode, (self.size, self.size), color=self.fill)
        offset_x = (self.size - resized_width) // 2
        offset_y = (self.size - resized_height) // 2
        background.paste(resized, (offset_x, offset_y))
        return background


class LocalModelRuntime:
    def __init__(self, settings: Settings) -> None:
        if settings.model_checkpoint_path is None:
            raise ValueError("MODEL_CHECKPOINT_PATH must be set when USE_MOCK=false")

        self._settings = settings
        self._checkpoint_path = settings.model_checkpoint_path
        self._device = self._resolve_device(settings.inference_device)
        self._storage = Minio(
            settings.s3_endpoint,
            access_key=settings.s3_access_key,
            secret_key=settings.s3_secret_key,
            secure=settings.s3_use_ssl,
        )

        checkpoint = torch.load(self._checkpoint_path, map_location="cpu")
        (
            model_name,
            backbone,
            num_classes,
            dropout,
            image_size,
            resize_mode,
            pad_fill,
            normalize,
            label_names,
            thresholds,
            state_dict,
        ) = (
            self._parse_checkpoint(checkpoint)
        )

        self._model = self._build_model(backbone, num_classes, dropout)
        self._model.load_state_dict(state_dict, strict=True)
        self._model = self._model.to(self._device).eval()
        if self._device.type == "cuda":
            self._model = self._model.to(memory_format=torch.channels_last)

        self._transform = self._build_eval_transform(
            image_size=image_size,
            resize_mode=resize_mode,
            pad_fill=pad_fill,
            normalize=normalize,
        )
        self.metadata = ModelMetadata(
            model_name=model_name,
            checkpoint_path=str(self._checkpoint_path.resolve()),
            backbone=backbone,
            device=str(self._device),
            num_classes=num_classes,
            image_size=image_size,
            resize_mode=resize_mode,
            pad_fill=pad_fill,
            label_names=label_names,
            thresholds=thresholds,
        )

    def predict(self, object_key: str) -> dict[str, Any]:
        image_bytes = self._download_image(object_key)
        source_image = Image.open(BytesIO(image_bytes))
        image_quality = self._assess_image_quality(source_image, object_key)
        image = source_image.convert("L")
        tensor = self._transform(image).unsqueeze(0).to(self._device)
        if self._device.type == "cuda":
            tensor = tensor.contiguous(memory_format=torch.channels_last)

        with torch.inference_mode():
            logits = self._model(tensor)
            probabilities = torch.sigmoid(logits).squeeze(0).float().cpu().numpy()

        return self._build_result(probabilities, image_quality)

    def _download_image(self, object_key: str) -> bytes:
        try:
            response = self._storage.get_object(self._settings.s3_bucket, object_key)
        except S3Error as exc:
            if exc.code == "NoSuchKey":
                raise FileNotFoundError(
                    f"object {object_key!r} was not found in bucket {self._settings.s3_bucket!r}"
                ) from exc
            raise RuntimeError(f"failed to fetch object {object_key!r} from MinIO: {exc}") from exc
        try:
            return response.read()
        finally:
            response.close()
            response.release_conn()

    def _build_result(self, probabilities: np.ndarray, image_quality: dict[str, Any]) -> dict[str, Any]:
        thresholds = np.asarray(self.metadata.thresholds, dtype=np.float32)
        positive_indices = [
            index
            for index, probability in enumerate(probabilities.tolist())
            if probability >= thresholds[index]
        ]
        positive_indices.sort(key=lambda index: float(probabilities[index]), reverse=True)

        if positive_indices:
            status = "Review Recommended"
            reference_score = float(probabilities[positive_indices[0]])
            findings = [
                (
                    f"The AI model detected image patterns that may be associated with {self.metadata.label_names[index]} "
                    f"({float(probabilities[index]) * 100:.1f}% model probability)."
                )
                for index in positive_indices[: self._settings.top_k_findings]
            ]
            recommendations = [
                "Recommend review by a clinician or radiologist.",
                "Interpret this AI screen together with symptoms, history, and formal clinical assessment.",
            ]
            ai_analysis = self._abnormal_summary(probabilities, positive_indices)
        else:
            status = "Normal"
            reference_score = float(1.0 - float(probabilities.max()) if probabilities.size else 1.0)
            findings = ["No supported finding exceeded the configured screening threshold in this AI analysis."]
            recommendations = [
                "Use this result as a screening aid and confirm with routine clinical review.",
                "Seek professional interpretation if symptoms, history, or image quality remain concerning.",
            ]
            ai_analysis = (
                "The uploaded chest X-ray did not trigger any supported finding above the model's configured thresholds. "
                "This should be treated as a screening output rather than a final diagnosis."
            )

        top_indices = np.argsort(probabilities)[::-1][: self._settings.top_k_findings]
        return {
            "status": status,
            "confidence": int(round(max(0.0, min(reference_score, 1.0)) * 100.0)),
            "findings": findings,
            "recommendations": recommendations,
            "ai_analysis": ai_analysis,
            "raw": {
                "model_name": self.metadata.model_name,
                "model_version": self._settings.model_version,
                "backbone": self.metadata.backbone,
                "checkpoint_path": self.metadata.checkpoint_path,
                "device": self.metadata.device,
                "label_order": list(self.metadata.label_names),
                "probabilities": {
                    label: float(probabilities[index])
                    for index, label in enumerate(self.metadata.label_names)
                },
                "thresholds": {
                    label: float(self.metadata.thresholds[index])
                    for index, label in enumerate(self.metadata.label_names)
                },
                "predicted_positive_labels": [
                    self.metadata.label_names[index] for index in positive_indices
                ],
                "top_labels": [
                    {
                        "label": self.metadata.label_names[index],
                        "probability": float(probabilities[index]),
                        "threshold": float(self.metadata.thresholds[index]),
                        "predicted_positive": bool(probabilities[index] >= self.metadata.thresholds[index]),
                    }
                    for index in top_indices
                ],
                "image_quality": image_quality,
                "explainability": {
                    "heatmap_url": None,
                    "explanation_text": None,
                },
            },
        }

    def _abnormal_summary(self, probabilities: np.ndarray, positive_indices: list[int]) -> str:
        fragments = []
        for index in positive_indices[: self._settings.top_k_findings]:
            fragments.append(
                f"{self.metadata.label_names[index]} ({float(probabilities[index]) * 100:.1f}%)"
            )
        joined = ", ".join(fragments)
        return (
            "The uploaded chest X-ray crossed configured screening thresholds for the following findings: "
            f"{joined}. This is a decision-support output and should be confirmed by a clinician or radiologist."
        )

    def _assess_image_quality(self, image: Image.Image, object_key: str) -> dict[str, Any]:
        supported_formats = {"jpeg", "jpg", "png", "webp"}
        image_format = (image.format or Path(object_key).suffix.lstrip(".") or "unknown").lower()
        width, height = image.size
        warnings: list[str] = []

        grayscale = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
        contrast = float(grayscale.std()) if grayscale.size else 0.0

        if image_format not in supported_formats:
            warnings.append("Image format is outside the preferred PNG/JPEG/WebP set used during routine testing.")
        if min(width, height) < 512:
            warnings.append("Image resolution is limited and may reduce model reliability.")
        if contrast < 0.12:
            warnings.append("Image contrast appears low and subtle findings may be less reliable.")

        quality_status = "ACCEPTABLE"
        if warnings:
            quality_status = "LOW_QUALITY"

        return {
            "image_type": image_format,
            "quality_status": quality_status,
            "warnings": warnings,
            "width": width,
            "height": height,
        }

    def _parse_checkpoint(
        self,
        checkpoint: dict[str, Any],
    ) -> tuple[
        str,
        str,
        int,
        float,
        int,
        str,
        int,
        dict[str, list[float]],
        list[str],
        list[float],
        dict[str, Any],
    ]:
        config = checkpoint.get("config") or {}

        if "model_state_dict" in checkpoint:
            model_config = config.get("model") or {}
            transform_config = config.get("transforms") or {}
            backbone = str(model_config.get("backbone") or "densenet121")
            num_classes = int(model_config.get("num_classes") or len(NIH_TARGET_LABELS))
            dropout = float(model_config.get("dropout") or 0.0)
            image_size = int(transform_config.get("image_size") or 224)
            resize_mode = str(transform_config.get("resize_mode") or "resize")
            pad_fill = int(transform_config.get("pad_fill") or 0)
            normalize = {
                "mean": list(transform_config.get("mean") or DEFAULT_MEAN),
                "std": list(transform_config.get("std") or DEFAULT_STD),
            }
            label_names = NIH_TARGET_LABELS[:num_classes]
            thresholds = checkpoint.get("thresholds") or [0.5] * num_classes
            state_dict = checkpoint["model_state_dict"]
            model_name = self._settings.model_name or f"{backbone}-checkpoint"
            return (
                model_name,
                backbone,
                num_classes,
                dropout,
                image_size,
                resize_mode,
                pad_fill,
                normalize,
                label_names,
                [float(value) for value in thresholds],
                state_dict,
            )

        if "state_dict" not in checkpoint:
            raise ValueError("Unsupported checkpoint format: missing model state")

        bundle_config = checkpoint.get("config") or {}
        backbone = str(checkpoint.get("backbone") or "densenet121")
        num_classes = int(bundle_config.get("num_classes") or len(checkpoint.get("label_names") or []))
        normalize = bundle_config.get("normalize") or {"mean": list(DEFAULT_MEAN), "std": list(DEFAULT_STD)}
        image_size = int((bundle_config.get("input_size") or [224, 224])[0])
        resize_mode = str(bundle_config.get("resize_mode") or "resize")
        pad_fill = int(bundle_config.get("pad_fill") or 0)
        label_names = list(checkpoint.get("label_names") or NIH_TARGET_LABELS[:num_classes])
        thresholds = checkpoint.get("thresholds") or [0.5] * len(label_names)
        model_name = self._settings.model_name or f"{backbone}-bundle"
        return (
            model_name,
            backbone,
            num_classes,
            0.0,
            image_size,
            resize_mode,
            pad_fill,
            {
                "mean": list(normalize.get("mean") or DEFAULT_MEAN),
                "std": list(normalize.get("std") or DEFAULT_STD),
            },
            label_names,
            [float(value) for value in thresholds],
            checkpoint["state_dict"],
        )

    def _build_model(self, backbone: str, num_classes: int, dropout: float) -> nn.Module:
        if backbone == "densenet121":
            model = models.densenet121(weights=None)
            in_features = model.classifier.in_features
            model.classifier = nn.Sequential(
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(in_features, num_classes),
            )
            return model
        if backbone == "densenet169":
            model = models.densenet169(weights=None)
            in_features = model.classifier.in_features
            model.classifier = nn.Sequential(
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(in_features, num_classes),
            )
            return model
        if backbone == "efficientnet_b0":
            model = models.efficientnet_b0(weights=None)
            in_features = model.classifier[-1].in_features
            model.classifier = nn.Sequential(
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(in_features, num_classes),
            )
            return model
        if backbone == "efficientnet_b2":
            model = models.efficientnet_b2(weights=None)
            in_features = model.classifier[-1].in_features
            model.classifier = nn.Sequential(
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(in_features, num_classes),
            )
            return model
        raise ValueError(f"Unsupported backbone for local inference: {backbone}")

    def _build_eval_transform(
        self,
        *,
        image_size: int,
        resize_mode: str,
        pad_fill: int,
        normalize: dict[str, list[float]],
    ) -> transforms.Compose:
        mean = tuple(float(value) for value in normalize.get("mean") or DEFAULT_MEAN)
        std = tuple(float(value) for value in normalize.get("std") or DEFAULT_STD)
        if resize_mode == "letterbox":
            resize = LetterboxSquare(
                image_size,
                fill=pad_fill,
                interpolation=InterpolationMode.BILINEAR,
            )
        else:
            resize = transforms.Resize(
                (image_size, image_size),
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            )
        return transforms.Compose(
            [
                resize,
                transforms.ToTensor(),
                transforms.Lambda(lambda tensor: tensor[:1, :, :] if tensor.shape[0] > 1 else tensor),
                transforms.Lambda(lambda tensor: tensor.repeat(3, 1, 1)),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    @staticmethod
    def _resolve_device(requested: str) -> torch.device:
        if requested == "cpu":
            return torch.device("cpu")
        if requested == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("INFERENCE_DEVICE=cuda was requested but CUDA is unavailable")
            return torch.device("cuda")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
