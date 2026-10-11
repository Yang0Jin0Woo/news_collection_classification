"""Temporary-file JSON profile replacement; overwrite approval stays caller-owned."""
from __future__ import annotations

import json
from pathlib import Path


def replace_json_profile(
    path: str | Path, payload: object, *, exclusive_temporary: bool = False,
) -> None:
    """Replace a profile only after a complete strict UTF-8 JSON write.

    Confidence profiles reuse their temporary path; decision profiles require a
    new temporary file. Callers create the parent directory and retain their
    existing --force and error policies.
    """
    target = Path(path)
    temporary = target.with_suffix(target.suffix + ".tmp")
    if exclusive_temporary:
        with temporary.open("x", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2, allow_nan=False)
    else:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )
    temporary.replace(target)
