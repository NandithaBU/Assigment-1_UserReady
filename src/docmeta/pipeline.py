from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from .backends import Backend
from .models import Example, LoadedDocument
from .normalize import clean_text, normalize_result
from .prompts import SYSTEM_PROMPT, build_user_text
from .readers import SUPPORTED_EXTENSIONS, load_bytes, load_document, ocr_images
from .schema import FIELDS, FILE_COLUMN, json_schema

log = logging.getLogger(__name__)


class MetadataExtractor:
    def __init__(self, backend: Backend, examples: Sequence[Example] = (),
                 ocr: bool = False):
        """ocr=True converts scans to text first, so a text-only model can be used."""
        self.backend = backend
        self.examples = list(examples)
        self.ocr = ocr

    def extract_document(self, doc: LoadedDocument) -> dict[str, Any]:
        if self.ocr and doc.images:
            doc = LoadedDocument(name=doc.name, text=ocr_images(doc.images))
        text = build_user_text(doc, self.examples)
        raw = self.backend.extract(SYSTEM_PROMPT, text, doc.images, json_schema())
        return normalize_result(raw)

    def extract_file(self, path: str | Path) -> dict[str, Any]:
        return self.extract_document(load_document(path))

    def extract_bytes(self, filename: str, data: bytes) -> dict[str, Any]:
        return self.extract_document(load_bytes(filename, data))


def _manifest_names(manifest_csv: str | Path) -> list[str]:
    df = pd.read_csv(manifest_csv, dtype=str)
    names = []
    for raw in df[FILE_COLUMN].tolist():
        name = clean_text(raw)
        if name and Path(name).suffix.lower() in SUPPORTED_EXTENSIONS:
            name = Path(name).stem
        names.append(name or "")
    return names


def find_files(input_dir: str | Path, manifest_csv: str | Path | None = None):
    """Return [(file_name_stem, path_or_None)]. With a manifest (e.g. test.csv),
    only listed files are used and the manifest's order is kept."""
    files = sorted(
        p for p in Path(input_dir).iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    by_stem: dict[str, Path] = {}
    for p in files:
        by_stem.setdefault(p.stem, p)
    if manifest_csv is None:
        return [(stem, p) for stem, p in by_stem.items()]
    return [(name, by_stem.get(name)) for name in _manifest_names(manifest_csv)]


def predict_directory(
    extractor: MetadataExtractor,
    input_dir: str | Path,
    output_csv: str | Path | None = None,
    manifest_csv: str | Path | None = None,
    workers: int = 4,
) -> pd.DataFrame:
    """Run extraction over a folder and return/write a predictions table.

    A failure on one file is logged and leaves that row blank rather than
    aborting the whole run.
    """
    targets = find_files(input_dir, manifest_csv)

    total, done, lock = len(targets), [0], threading.Lock()

    def progress(name, started, status):
        with lock:
            done[0] += 1
            log.info("[%d/%d] %s: %s (%.0fs)", done[0], total, name, status,
                     time.perf_counter() - started)

    def run(item):
        name, path = item
        started = time.perf_counter()
        if path is None:
            log.warning("No file found for '%s'", name)
            progress(name, started, "skipped, file not found")
            return name, {}
        try:
            result = extractor.extract_file(path)
            progress(name, started, "ok")
            return name, result
        except Exception as exc:
            log.error("Failed on %s: %s", path.name, exc)
            progress(name, started, "FAILED")
            return name, {}

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        results = list(pool.map(run, targets))

    rows = []
    for name, fields in results:
        row = {FILE_COLUMN: name}
        for f in FIELDS:
            value = fields.get(f.key)
            row[f.csv_column] = "" if value is None else str(value)
        rows.append(row)
    df = pd.DataFrame(rows, columns=[FILE_COLUMN] + [f.csv_column for f in FIELDS])
    if output_csv is not None:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
    return df
