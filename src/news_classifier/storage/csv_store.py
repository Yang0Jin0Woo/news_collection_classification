from __future__ import annotations

import logging
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import pandas as pd

from news_classifier.models import CLASSIFIED_NEWS_COLUMNS, ClassifiedNews


logger = logging.getLogger(__name__)


class CsvNewsStore:
    def __init__(self, path: str):
        self.path = Path(path)

    def save(self, rows: list[ClassifiedNews]) -> None:
        data = [row.to_row() for row in rows]
        df = pd.DataFrame(data, columns=CLASSIFIED_NEWS_COLUMNS)
        if not df.empty and "published_at" in df.columns:
            df = df.sort_values(by="published_at", ascending=False, na_position="last")
        temporary_path = None
        try:
            # A unique file beside the target keeps replacement on one filesystem.
            with NamedTemporaryFile(
                mode="w", encoding="utf-8-sig", newline="",
                dir=self.path.parent, prefix=f".{self.path.name}.",
                suffix=".tmp", delete=False,
            ) as file:
                temporary_path = Path(file.name)
                df.to_csv(file, index=False)
                file.flush()
                os.fsync(file.fileno())
            # Close the file before replacement, including on Windows.
            temporary_path.replace(self.path)
        finally:
            if temporary_path is not None:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError as exc:
                    # Cleanup must not hide the original save/replacement error.
                    logger.warning("CSV 임시 파일 정리 실패: %s (%s)", temporary_path, exc)

    def load(self) -> pd.DataFrame:
        if not self.path.exists():
            return pd.DataFrame()
        return pd.read_csv(self.path)
