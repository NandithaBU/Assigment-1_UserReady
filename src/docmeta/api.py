"""REST API. Run: uvicorn docmeta.api:app  (or: docmeta serve)

POST /extract  multipart file upload (.docx / .png / .jpg ...) -> JSON fields
GET  /health
"""
from __future__ import annotations

from typing import Callable

from .config import Settings, build_extractor
from .backends import ExtractionError
from .pipeline import MetadataExtractor
from .readers import SUPPORTED_EXTENSIONS, UnsupportedFormat
from . import __version__


def create_app(extractor_factory: Callable[[], MetadataExtractor] | None = None):
    from pathlib import Path

    from fastapi import FastAPI, File, HTTPException, UploadFile
    from starlette.concurrency import run_in_threadpool

    settings = Settings.from_env()
    factory = extractor_factory or (lambda: build_extractor(settings))
    state: dict[str, MetadataExtractor] = {}

    def get_extractor() -> MetadataExtractor:
        if "extractor" not in state:  # built on first request so import never needs a key
            state["extractor"] = factory()
        return state["extractor"]

    app = FastAPI(title="docmeta", version=__version__,
                  description="Extract agreement metadata from .docx files and scanned images.")

    @app.get("/health")
    def health():
        return {"status": "ok", "version": __version__}

    @app.post("/extract")
    async def extract(file: UploadFile = File(...)):
        filename = file.filename or "upload"
        if Path(filename).suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise HTTPException(415, f"Unsupported file type. Use one of: "
                                     f"{', '.join(sorted(SUPPORTED_EXTENSIONS))}")
        data = await file.read()
        if len(data) > settings.max_upload_mb * 1024 * 1024:
            raise HTTPException(413, f"File larger than {settings.max_upload_mb} MB.")
        try:
            fields = await run_in_threadpool(get_extractor().extract_bytes, filename, data)
        except UnsupportedFormat as exc:
            raise HTTPException(415, str(exc))
        except ExtractionError as exc:
            raise HTTPException(502, str(exc))
        return {"file_name": Path(filename).stem, "fields": fields}

    return app


app = create_app()
