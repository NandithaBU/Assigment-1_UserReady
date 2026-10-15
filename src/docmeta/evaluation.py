"""Per-field recall, as defined by the assignment.

For each field: True  = predicted value exactly equals the labelled value,
                False = mismatch or nothing extracted,
                Recall = True / (True + False).

Documents whose label for a field is empty are not scored for that field
(there is nothing to recall). Whitespace is normalised on both sides by
default because the training labels contain stray trailing spaces; use
strict=True to compare raw strings instead.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .normalize import clean_text, normalize_int
from .schema import FIELDS, FILE_COLUMN, Field, column_candidates


@dataclass
class FieldScore:
    field: str
    correct: int
    total: int

    @property
    def recall(self) -> float:
        return self.correct / self.total if self.total else float("nan")


def _column(df: pd.DataFrame, field: Field) -> str | None:
    for name in column_candidates(field):
        if name in df.columns:
            return name
    return None


def _canon(value, kind: str, strict: bool, ignore_case: bool) -> str | None:
    if value is None or (isinstance(value, float) and value != value):
        return None
    if strict:
        text = str(value)
        return text if text != "" else None
    if kind == "int":
        number = normalize_int(value)
        return None if number is None else str(number)
    text = clean_text(value)
    if text is not None and ignore_case:
        text = text.casefold()
    return text


def evaluate(pred: pd.DataFrame, truth: pd.DataFrame, strict: bool = False,
             ignore_case: bool = False) -> tuple[list[FieldScore], pd.DataFrame]:
    """Return per-field scores and a table of every miss for error analysis."""
    pred = pred.copy()
    truth = truth.copy()
    pred["_key"] = pred[FILE_COLUMN].map(clean_text)
    truth["_key"] = truth[FILE_COLUMN].map(clean_text)
    pred = pred.drop_duplicates("_key").set_index("_key")

    scores: list[FieldScore] = []
    misses: list[dict] = []
    for f in FIELDS:
        t_col, p_col = _column(truth, f), _column(pred, f)
        correct = total = 0
        if t_col is None:
            scores.append(FieldScore(f.csv_column, 0, 0))
            continue
        for _, row in truth.iterrows():
            expected = _canon(row[t_col], f.kind, strict, ignore_case)
            if expected is None:
                continue
            total += 1
            got = None
            if p_col is not None and row["_key"] in pred.index:
                got = _canon(pred.at[row["_key"], p_col], f.kind, strict, ignore_case)
            if got == expected:
                correct += 1
            else:
                misses.append({FILE_COLUMN: row["_key"], "field": f.csv_column,
                               "expected": expected, "predicted": got})
        scores.append(FieldScore(f.csv_column, correct, total))
    return scores, pd.DataFrame(misses, columns=[FILE_COLUMN, "field", "expected", "predicted"])


def format_report(scores: list[FieldScore]) -> str:
    width = max(len(s.field) for s in scores)
    lines = [f"{'Field'.ljust(width)}  Correct/Total  Recall"]
    for s in scores:
        rec = "n/a" if s.total == 0 else f"{s.recall:.2%}"
        lines.append(f"{s.field.ljust(width)}  {s.correct:>3}/{s.total:<9} {rec}")
    scored = [s for s in scores if s.total]
    if scored:
        macro = sum(s.recall for s in scored) / len(scored)
        lines.append(f"{'Macro average'.ljust(width)}  {'':13}  {macro:.2%}")
    return "\n".join(lines)
