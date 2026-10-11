"""Exclusive JSON report creation; overwrite and profile replacement stay caller-owned."""
from __future__ import annotations

import json
from pathlib import Path


def write_new_json(
    path: str | Path, payload: object, *,
    allow_nan: bool = False, trailing_newline: bool = False,
) -> str:
    """Create a new UTF-8 report and return its serialized text without a final newline."""
    serialized = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=allow_nan)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as file:
        file.write(serialized + ("\n" if trailing_newline else ""))
    return serialized
