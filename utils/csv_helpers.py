"""CSV column detection and label parsing for batch evaluation."""

from __future__ import annotations

from .config import CFG


def resolve_prompt_column(fieldnames: list[str] | None) -> str | None:
    if not fieldnames:
        return None
    names = set(fieldnames)
    for col in (CFG.csv_prompt_column, "text", "prompts"):
        if col in names:
            return col
    return None


def parse_ground_truth(raw: str | None) -> int | None:
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    try:
        v = int(float(s))
        if v in (0, 1):
            return v
    except ValueError:
        pass
    return None
