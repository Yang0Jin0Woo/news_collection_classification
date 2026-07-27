from __future__ import annotations

from pathlib import Path
import pandas as pd

from news_classifier.models import CLASSIFIED_NEWS_COLUMNS, ClassifiedNews


class CsvNewsStore:
    def __init__(self, path: str):
        self.path = Path(path)

    def save(self, rows: list[ClassifiedNews]) -> None:
        data = [row.to_row() for row in rows]
        df = pd.DataFrame(data, columns=CLASSIFIED_NEWS_COLUMNS)
        if not df.empty and "published_at" in df.columns:
            df = df.sort_values(by="published_at", ascending=False, na_position="last")
        df.to_csv(self.path, index=False, encoding="utf-8-sig")

    def load(self) -> pd.DataFrame:
        if not self.path.exists():
            return pd.DataFrame()
        return pd.read_csv(self.path)
