"""CSV, TSV, and Excel export helpers."""
from __future__ import annotations
from io import BytesIO
import pandas as pd


def table_tsv(table: pd.DataFrame) -> str:
    """Return a headerless copy-ready TSV block."""
    return table.to_csv(sep="\t", index=False, header=False, na_rep="")


def table_csv(table: pd.DataFrame) -> bytes:
    """Return a headerless CSV suitable for Prism import."""
    return table.to_csv(index=False, header=False, na_rep="").encode("utf-8")


def workbook_bytes(tables: dict[str, pd.DataFrame]) -> bytes:
    """Create a clean, unmerged Excel workbook with plain values."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        for name, table in tables.items():
            safe = name[:31]
            table.to_excel(writer, sheet_name=safe, index=False, header=False, na_rep="")
            ws = writer.sheets[safe]
            ws.freeze_panes(1, 0)
            ws.set_column(0, max(0, len(table.columns) - 1), 14)
    return output.getvalue()
