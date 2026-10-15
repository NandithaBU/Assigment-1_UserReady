from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ImagePart:
    media_type: str
    data: bytes


@dataclass
class LoadedDocument:
    name: str
    text: str | None = None                                # set for .docx
    images: list[ImagePart] = field(default_factory=list)  # set for scans


@dataclass
class Example:
    """A labelled document used as a few-shot example."""
    text: str
    labels: dict[str, Any]
