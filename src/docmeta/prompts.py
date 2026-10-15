"""Prompt construction. Edit SYSTEM_PROMPT to adapt to new label conventions."""
from __future__ import annotations

import json
from typing import Sequence

from .models import Example, LoadedDocument

SYSTEM_PROMPT = """\
You extract structured metadata from rental / lease agreements. The input is \
either the text of a .docx or one or more scanned images. Scans and converted \
files often contain OCR-style noise (stray punctuation, garbled digits); read \
carefully and use context to recover the intended value.

Rules for each field:
- agreement_value: the monthly rent, as a plain integer. Not the security \
deposit, advance, or a yearly total unless no monthly rent is stated. When \
words and digits disagree or the digits are garbled, trust the amount written \
in words.
- agreement_start_date / agreement_end_date: formatted DD.MM.YYYY. Copy the \
dates exactly as the document states them. Do NOT "fix" dates that look \
impossible (for example 31.11.2009 stays 31.11.2009). If only a start date and \
a duration are given, derive the end date from them.
- renewal_notice_days: days of notice required to renew or terminate, as an \
integer (1 month = 30 days). null if the document does not say.
- party_one: the owner / landlord / lessor. party_two: the tenant / lessee / \
resident. Give the name only: no address, no "hereinafter referred to as". \
Keep initials and honorifics as written, but repair obvious OCR punctuation \
slips (for example a full stop where a comma belongs).

If a value is genuinely absent, return null. Never guess.
"""


def build_user_text(doc: LoadedDocument, examples: Sequence[Example] = ()) -> str:
    parts: list[str] = []
    if examples:
        parts.append("Worked examples (document text, then the correct metadata):")
        for i, ex in enumerate(examples, 1):
            parts.append(
                f'<example index="{i}">\n<document>\n{ex.text}\n</document>\n'
                f"<metadata>{json.dumps(ex.labels, ensure_ascii=False)}</metadata>\n</example>"
            )
        parts.append("Now the real document.")
    if doc.text is not None:
        parts.append(f"<document>\n{doc.text}\n</document>")
    else:
        parts.append("The document is attached as image(s). Read all of it.")
    parts.append("Return the extracted fields.")
    return "\n\n".join(parts)
