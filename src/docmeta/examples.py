"""Build few-shot examples from a labelled training set (train.csv + train/)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .models import Example
from .normalize import clean_text, normalize_int
from .readers import DOCX_EXTENSIONS, docx_to_text
from .schema import FIELDS, FILE_COLUMN, column_candidates


def _get(row, field):
    for col in column_candidates(field):
        if col in row.index:
            return row[col]
    return None


def load_examples(train_csv: str | Path, train_dir: str | Path, k: int = 3,
                  max_chars: int = 6000) -> list[Example]:
    """Pick up to k .docx training documents, spread evenly through the CSV.

    Only .docx files are used (cheap text); scanned images are skipped.
    """
    if k <= 0:
        return []
    df = pd.read_csv(train_csv, dtype=str)
    docs = {p.stem: p for p in Path(train_dir).iterdir() if p.suffix.lower() in DOCX_EXTENSIONS}
    candidates = [r for _, r in df.iterrows() if clean_text(r[FILE_COLUMN]) in docs]
    if not candidates:
        return []
    k = min(k, len(candidates))
    step = len(candidates) / k
    chosen = [candidates[int(i * step)] for i in range(k)]

    examples = []
    for row in chosen:
        path = docs[clean_text(row[FILE_COLUMN])]
        with open(path, "rb") as fh:
            text = docx_to_text(fh)[:max_chars]
        labels = {}
        for f in FIELDS:
            value = _get(row, f)
            labels[f.key] = normalize_int(value) if f.kind == "int" else clean_text(value)
        examples.append(Example(text=text, labels=labels))
    return examples
