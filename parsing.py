"""Generic grid detection and parsing for Promega NanoBRET workbooks."""
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any
import math
import re

import numpy as np
import pandas as pd
from openpyxl import load_workbook

MISSING_TEXT = {"", "NONE", "-", "NAN", "NA", "N/A"}

@dataclass
class GridBlock:
    """A rectangular plate grid detected from a worksheet."""
    label: str
    header_row: int
    first_col: int
    width: int
    height: int
    row_labels: list[str]
    values: list[list[Any]]

    @property
    def shape(self) -> tuple[int, int]:
        return self.height, self.width

    def frame(self, numeric: bool = False) -> pd.DataFrame:
        """Return the block as a row/column indexed DataFrame."""
        index = [x.upper() for x in self.row_labels]
        columns = list(range(1, self.width + 1))
        df = pd.DataFrame(self.values, index=index, columns=columns)
        if numeric:
            return df.apply(pd.to_numeric, errors="coerce")
        return df.apply(lambda col: col.map(normalize_text))

@dataclass
class ParsedPlate:
    """Parsed instrument and sample grids for one workbook."""
    result_blocks: list[GridBlock]
    sample_blocks: list[GridBlock]
    result_roles: dict[str, GridBlock]
    sample_roles: dict[str, GridBlock]
    warnings: list[str]


def normalize_text(value: Any) -> str | None:
    """Strip text and normalize common missing-value tokens."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value).strip()
    if text.upper() in MISSING_TEXT:
        return None
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return text


def _is_int(value: Any, expected: int) -> bool:
    if isinstance(value, bool) or value is None:
        return False
    try:
        return float(value) == expected and float(value).is_integer()
    except (TypeError, ValueError):
        return False


def _row_label(value: Any) -> bool:
    return isinstance(value, str) and bool(re.fullmatch(r"[A-Za-z]", value.strip()))


def _candidate_runs(ws: Any) -> list[tuple[int, int, int]]:
    """Find runs 1..N and return (header row, first column, width)."""
    candidates: list[tuple[int, int, int]] = []
    for r in range(1, ws.max_row + 1):
        c = 1
        while c <= ws.max_column:
            if _is_int(ws.cell(r, c).value, 1):
                start = c
                expected = 1
                while c <= ws.max_column and _is_int(ws.cell(r, c).value, expected):
                    expected += 1
                    c += 1
                width = expected - 1
                if width >= 2 and start > 1:
                    # Require at least two consecutive letter row labels below the run.
                    height = 0
                    rr = r + 1
                    while rr <= ws.max_row and _row_label(ws.cell(rr, start - 1).value):
                        height += 1
                        rr += 1
                    if height >= 2:
                        candidates.append((r, start, width))
            c += 1
    return candidates


def _context_label(ws: Any, header_row: int, first_col: int) -> str:
    """Collect the nearest meaningful labels preceding a detected grid."""
    parts: list[str] = []
    for r in range(max(1, header_row - 5), header_row + 1):
        for c in range(1, first_col):
            value = normalize_text(ws.cell(r, c).value)
            if value and not re.fullmatch(r"[A-Za-z]", value):
                parts.append(value)
    return " ".join(parts)


def _detect_blocks(ws: Any, sample_sheet: bool) -> list[GridBlock]:
    blocks: list[GridBlock] = []
    for header_row, first_col, width in _candidate_runs(ws):
        height = 0
        while header_row + 1 + height <= ws.max_row and _row_label(ws.cell(header_row + 1 + height, first_col - 1).value):
            height += 1
        row_labels = [str(ws.cell(header_row + 1 + i, first_col - 1).value).strip() for i in range(height)]
        values = [[ws.cell(header_row + 1 + i, first_col + j).value for j in range(width)] for i in range(height)]
        label = normalize_text(ws.cell(header_row, first_col - 1).value) or ""
        if sample_sheet:
            label = normalize_text(ws.cell(header_row, 2).value) or label
        else:
            label = _context_label(ws, header_row, first_col) or label
        blocks.append(GridBlock(label, header_row, first_col, width, height, row_labels, values))
    # De-duplicate any overlapping candidates, keeping the widest grid.
    unique: dict[tuple[int, int], GridBlock] = {}
    for block in blocks:
        key = (block.header_row, block.first_col)
        if key not in unique or block.width > unique[key].width:
            unique[key] = block
    return sorted(unique.values(), key=lambda x: x.header_row)


def _find_sheet(wb: Any, wanted: str) -> Any:
    for name in wb.sheetnames:
        if name.strip().lower() == wanted.lower():
            return wb[name]
    raise ValueError(f"Missing '{wanted}' sheet. Expected workbook sheets containing '{wanted}'.")


def _role_for_result(block: GridBlock, index: int) -> str:
    # The first Results grid is the instrument BRET grid; its nearby metadata
    # also mentions Donor and Acceptor, so position takes precedence there.
    if index == 0:
        return "bret"
    text = block.label.lower()
    if "donor" in text:
        return "donor"
    if "acceptor" in text:
        return "acceptor"
    if "bret" in text or "reading" in text:
        return "bret"
    return f"result_{index + 1}"


def _sample_role(label: str) -> str | None:
    text = label.lower().strip()
    if "mrna" in text:
        return "mrna"
    if ("guide" in text or "tracer" in text or "labelled" in text or "labeled" in text) and "conc" not in text:
        return "labelled_guide"
    if "competitor" in text and "conc" not in text:
        return "competitor"
    if "conc" in text or "concentration" in text:
        return "conc_nM"
    return None


def parse_workbook(source: str | Path | bytes) -> ParsedPlate:
    """Parse Results and Samples grids from a workbook without fixed cell addresses."""
    if isinstance(source, bytes):
        wb = load_workbook(BytesIO(source), data_only=True)
    else:
        wb = load_workbook(source, data_only=True)
    results_ws = _find_sheet(wb, "results")
    samples_ws = _find_sheet(wb, "samples")
    result_blocks = _detect_blocks(results_ws, sample_sheet=False)
    sample_blocks = _detect_blocks(samples_ws, sample_sheet=True)
    if len(result_blocks) < 2:
        raise ValueError("Unrecognised Results layout. Expected stacked plate grids with a 1..N header and letter row labels.")
    if not sample_blocks:
        raise ValueError("Unrecognised Samples layout. Expected labelled stacked plate grids with a 1..N header and letter row labels.")
    result_roles: dict[str, GridBlock] = {}
    for i, block in enumerate(result_blocks):
        role = _role_for_result(block, i)
        if role not in result_roles:
            result_roles[role] = block
    if "donor" not in result_roles or "acceptor" not in result_roles:
        raise ValueError("Results layout is missing Donor and/or Acceptor grids. Expected labels containing 'Donor:' and 'Acceptor:'.")
    sample_roles: dict[str, GridBlock] = {}
    for block in sample_blocks:
        role = _sample_role(block.label)
        if role and role not in sample_roles:
            sample_roles[role] = block
    warnings: list[str] = []
    for required in ("mrna", "labelled_guide", "competitor", "conc_nM"):
        if required not in sample_roles:
            warnings.append(f"No automatic Samples mapping found for {required}.")
    dims = [b.shape for b in result_roles.values() if b is not None] + [b.shape for b in sample_roles.values() if b is not None]
    if dims and any(d != dims[0] for d in dims):
        raise ValueError(f"Plate grid dimensions do not match. Expected every Results and Samples grid to have {dims[0][0]} rows x {dims[0][1]} columns.")
    return ParsedPlate(result_blocks, sample_blocks, result_roles, sample_roles, warnings)


def block_options(plate: ParsedPlate, kind: str) -> list[str]:
    """Return readable labels for UI role mapping."""
    blocks = plate.sample_blocks if kind == "samples" else plate.result_blocks
    return [f"{i}: {b.label or '(unlabelled)'} [{b.height}x{b.width}]" for i, b in enumerate(blocks)]
