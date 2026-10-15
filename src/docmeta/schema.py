"""Field definitions: the single source of truth for what gets extracted.

To add or rename a field, edit FIELDS. The LLM tool schema, normalisation,
prediction CSV columns and evaluation all derive from it.
"""
from __future__ import annotations

from dataclasses import dataclass

FILE_COLUMN = "File Name"
TOOL_NAME = "record_metadata"


@dataclass(frozen=True)
class Field:
    key: str          # snake_case key used in Python / JSON
    csv_column: str   # column name used in train.csv / predictions
    kind: str         # "int" | "date" | "text"
    description: str  # shown to the LLM in the tool schema


FIELDS = (
    Field(
        "agreement_value", "Aggrement Value", "int",
        "Monthly rent as a plain integer: no currency symbol, no separators, "
        "no decimals. null if no amount is stated.",
    ),
    Field(
        "agreement_start_date", "Aggrement Start Date", "date",
        "Date the agreement/tenancy starts, formatted DD.MM.YYYY. null if absent.",
    ),
    Field(
        "agreement_end_date", "Aggrement End Date", "date",
        "Date the agreement/tenancy ends, formatted DD.MM.YYYY. null if absent.",
    ),
    Field(
        "renewal_notice_days", "Renewal Notice (Days)", "int",
        "Number of days of notice required for renewal/termination, as an "
        "integer (convert months to days at 30 per month). null if absent.",
    ),
    Field(
        "party_one", "Party One", "text",
        "Name of the first party: the owner / landlord / lessor. Name only, "
        "no address and no role description.",
    ),
    Field(
        "party_two", "Party Two", "text",
        "Name of the second party: the tenant / lessee / resident. Name only, "
        "no address and no role description.",
    ),
)
FIELD_KEYS = tuple(f.key for f in FIELDS)


def json_schema() -> dict:
    """JSON schema handed to the LLM as the structured-output tool."""
    props = {}
    for f in FIELDS:
        json_type = "integer" if f.kind == "int" else "string"
        props[f.key] = {"type": [json_type, "null"], "description": f.description}
    return {"type": "object", "properties": props, "required": list(FIELD_KEYS)}


def column_candidates(field: Field) -> list[str]:
    """Accept both the dataset's 'Aggrement' spelling and 'Agreement'."""
    names = [field.csv_column]
    if "Aggrement" in field.csv_column:
        names.append(field.csv_column.replace("Aggrement", "Agreement"))
    return names
