from __future__ import annotations

from pathlib import Path
import pandas as pd


class ExcelReportExporter:
    def export(self, df: pd.DataFrame, path: str) -> None:
        path_obj = Path(path)
        with pd.ExcelWriter(path_obj, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="news")
            if not df.empty and "final_category" in df.columns:
                summary = df.groupby("final_category").size().reset_index(name="count")
                summary.to_excel(writer, index=False, sheet_name="summary")
