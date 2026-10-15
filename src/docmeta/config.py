from __future__ import annotations

import os
from dataclasses import dataclass

from .backends import AnthropicBackend, Backend, OpenAICompatBackend, TransformersBackend
from .examples import load_examples
from .pipeline import MetadataExtractor

PROVIDERS = ("openai", "transformers", "anthropic")


@dataclass
class Settings:
    provider: str = "openai"   # openai = OpenAI-compatible server | transformers = local HF model
    base_url: str | None = None
    model: str | None = None
    vision_model: str | None = None
    ocr: bool = False
    device_map: str = "auto"
    train_csv: str | None = None
    train_dir: str | None = None
    shots: int = 3
    max_upload_mb: int = 10

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            provider=os.getenv("DOCMETA_PROVIDER", "openai"),
            base_url=os.getenv("DOCMETA_BASE_URL"),
            model=os.getenv("DOCMETA_MODEL"),
            vision_model=os.getenv("DOCMETA_VISION_MODEL"),
            ocr=os.getenv("DOCMETA_OCR", "").lower() in {"1", "true", "yes"},
            device_map=os.getenv("DOCMETA_DEVICE_MAP", "auto"),
            train_csv=os.getenv("DOCMETA_TRAIN_CSV"),
            train_dir=os.getenv("DOCMETA_TRAIN_DIR"),
            shots=int(os.getenv("DOCMETA_SHOTS", "3")),
            max_upload_mb=int(os.getenv("DOCMETA_MAX_UPLOAD_MB", "10")),
        )


def build_backend(settings: Settings) -> Backend:
    if settings.provider == "anthropic":
        return AnthropicBackend(model=settings.model)
    if settings.provider == "transformers":
        return TransformersBackend(model=settings.model, device_map=settings.device_map)
    if settings.provider == "openai":
        return OpenAICompatBackend(base_url=settings.base_url, model=settings.model,
                                   vision_model=settings.vision_model)
    raise ValueError(f"Unknown provider '{settings.provider}'. Use one of: {PROVIDERS}")


def build_extractor(settings: Settings | None = None,
                    backend: Backend | None = None) -> MetadataExtractor:
    settings = settings or Settings.from_env()
    backend = backend or build_backend(settings)
    examples = []
    if settings.train_csv and settings.train_dir and settings.shots > 0:
        examples = load_examples(settings.train_csv, settings.train_dir, settings.shots)
    return MetadataExtractor(backend, examples, ocr=settings.ocr)
