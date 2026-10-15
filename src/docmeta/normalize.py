"""Output normalisation.

This only *formats* values the LLM already extracted (whitespace, number and
date layout). It never searches the document for anything, so extraction
itself stays model-driven.

Dates are deliberately NOT checked against a calendar: real agreements (and
the training labels) contain dates such as 31.11.2009, which must survive
unchanged.
"""
from __future__ import annotations

import math
from typing import Any

from .schema import FIELDS


def clean_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = " ".join(str(value).split())
    return text or None


def normalize_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (int, float)):
        return int(round(value))
    text = clean_text(value)
    if text is None:
        return None
    text = text.replace(",", "").replace(" ", "")
    try:
        return int(round(float(text)))
    except ValueError:
        return None


def normalize_date(value: Any) -> str | None:
    """Return DD.MM.YYYY, or None if the value does not look like a date."""
    text = clean_text(value)
    if text is None:
        return None
    parts = text.replace("/", ".").replace("-", ".").split(".")
    if len(parts) != 3 or not all(p.isdigit() for p in parts):
        return None
    day, month, year = parts
    if len(year) != 4 or not (1 <= int(day) <= 31 and 1 <= int(month) <= 12):
        return None
    return f"{int(day):02d}.{int(month):02d}.{year}"


def normalize_result(raw: dict) -> dict:
    """Apply per-field normalisation to the raw LLM output."""
    out: dict[str, Any] = {}
    for f in FIELDS:
        value = raw.get(f.key)
        if f.kind == "int":
            out[f.key] = normalize_int(value)
        elif f.kind == "date":
            out[f.key] = normalize_date(value)
        else:
            out[f.key] = clean_text(value)
    return out
