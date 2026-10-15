"""NanoBRET calculations, grouping, and Prism table construction."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np
import pandas as pd

from parsing import GridBlock, ParsedPlate, normalize_text

TIDY_COLUMNS = [
    "well", "row", "col", "mRNA", "labelled_guide", "competitor", "conc_nM",
    "donor_RLU", "acceptor_RLU", "BRET", "mBU", "mBU_corrected",
    "flag_low_donor", "include", "source_file",
]

@dataclass
class CalculationResult:
    tidy: pd.DataFrame
    background: float | None
    background_wells: list[str]
    warnings: list[str]
    mismatch_count: int
    pseudo_concentration: float | None
    geometric_step: float | None
    single_controls: pd.DataFrame


def _grid_frame(block: GridBlock, numeric: bool = False) -> pd.DataFrame:
    return block.frame(numeric=numeric)


def _series_key(value: object) -> str | None:
    return normalize_text(value)


def build_tidy(
    plate: ParsedPlate,
    mapping: dict[str, GridBlock],
    source_file: str = "",
    background_mode: str = "auto-detect",
    manual_background_wells: Iterable[str] | None = None,
    low_donor_threshold: float = 0,
) -> CalculationResult:
    """Join mapped grids, calculate mBU, background correction, and QC flags."""
    warnings = list(plate.warnings)
    donor = _grid_frame(mapping["donor"], True)
    acceptor = _grid_frame(mapping["acceptor"], True)
    bret_block = mapping.get("bret")
    bret = _grid_frame(bret_block, True) if bret_block is not None else acceptor.divide(donor.replace(0, np.nan))
    if bret_block is None:
        warnings.append("BRET grid was not mapped; BRET was computed as Acceptor / Donor.")
    mRNA = _grid_frame(mapping["mrna"], False)
    guide = _grid_frame(mapping["labelled_guide"], False)
    competitor = _grid_frame(mapping["competitor"], False)
    conc = _grid_frame(mapping["conc_nM"], True)
    dims = {x.shape for x in (donor, acceptor, bret, mRNA, guide, competitor, conc)}
    if len(dims) != 1:
        raise ValueError(f"Results and Samples grids have inconsistent dimensions: {sorted(dims)}.")
    rows: list[dict] = []
    mismatch_count = 0
    for r in bret.index:
        for c in bret.columns:
            d = donor.loc[r, c]
            a = acceptor.loc[r, c]
            br = bret.loc[r, c]
            expected = a / d if pd.notna(a) and pd.notna(d) and d != 0 else np.nan
            if pd.notna(br) and pd.notna(expected) and not np.isclose(br, expected, rtol=0, atol=1e-6):
                mismatch_count += 1
            if pd.isna(br) and pd.notna(expected):
                br = expected
            well = f"{r}{int(c)}"
            mrna = _series_key(mRNA.loc[r, c])
            lab = _series_key(guide.loc[r, c])
            comp = _series_key(competitor.loc[r, c])
            cn = conc.loc[r, c]
            if pd.isna(cn):
                cn = np.nan
            else:
                cn = float(cn)
            rows.append({
                "well": well, "row": r, "col": int(c), "mRNA": mrna,
                "labelled_guide": lab, "competitor": comp, "conc_nM": cn,
                "donor_RLU": d, "acceptor_RLU": a, "BRET": br,
                "mBU": br * 1000 if pd.notna(br) else np.nan,
                "source_file": source_file,
            })
    df = pd.DataFrame(rows)
    donor_only = df["mRNA"].notna() & df["labelled_guide"].isna()
    if background_mode == "manual":
        selected = {str(x).upper() for x in (manual_background_wells or [])}
        bgmask = df["well"].str.upper().isin(selected)
        background_wells = df.loc[bgmask, "well"].tolist()
    elif background_mode == "none":
        bgmask = pd.Series(False, index=df.index)
        background_wells = []
    else:
        bgmask = donor_only
        background_wells = df.loc[bgmask, "well"].tolist()
    background = float(df.loc[bgmask, "mBU"].mean()) if bgmask.any() and df.loc[bgmask, "mBU"].notna().any() else None
    if background_mode != "none" and background is None:
        warnings.append("No valid donor-only background wells were found; no subtraction was applied.")
    df["mBU_corrected"] = df["mBU"] - background if background is not None else df["mBU"]
    df["flag_low_donor"] = (df["donor_RLU"] < low_donor_threshold) if low_donor_threshold and low_donor_threshold > 0 else False
    df["include"] = df["mRNA"].notna() & df["competitor"].notna() & df["conc_nM"].notna()
    if mismatch_count:
        warnings.append(f"BRET check failed for {mismatch_count} wells (absolute tolerance 1e-6).")
    return CalculationResult(df[TIDY_COLUMNS], background, background_wells, warnings, mismatch_count, None, None, pd.DataFrame())


def dilution_stats(tidy: pd.DataFrame) -> tuple[float | None, float | None]:
    """Return pseudo-zero concentration and geometric-mean dilution step."""
    vals = sorted({float(x) for x in tidy.loc[tidy["include"] & (tidy["conc_nM"] > 0), "conc_nM"].dropna()})
    if not vals:
        return None, None
    if len(vals) < 2:
        return vals[0] / 2, 2.0
    ratios = np.array(vals[1:]) / np.array(vals[:-1])
    ratios = ratios[ratios > 0]
    # Use the geometric mean of the consistent dilution steps.  A single
    # rounded/irregular concentration should not shift the pseudo-zero.
    median_ratio = float(np.median(ratios)) if len(ratios) else 2.0
    consistent = ratios[np.abs(ratios / median_ratio - 1) <= 0.02]
    if len(consistent) == 0:
        consistent = ratios
    step = float(np.exp(np.mean(np.log(consistent)))) if len(consistent) else 2.0
    return float(vals[0] / step), step


def _vehicle_set(vehicle_names: Iterable[str]) -> set[str]:
    return {str(x).strip().lower() for x in vehicle_names if str(x).strip()}


def _attach_series(tidy: pd.DataFrame, vehicle_names: Iterable[str], group_by: str) -> pd.DataFrame:
    """Assign a display series to every included well."""
    df = tidy.copy()
    vehicles = _vehicle_set(vehicle_names)
    df["series"] = None
    for row_name, row_idx in df.groupby("row", sort=False).groups.items():
        block = df.loc[row_idx]
        positive = block[(block["include"]) & (block["conc_nM"] > 0) & block["competitor"].notna()]
        if not positive.empty:
            name = str(positive["competitor"].mode().iloc[0])
            zero = block["conc_nM"].eq(0) & block["competitor"].notna()
            vehicle_zero = zero & block["competitor"].str.lower().isin(vehicles)
            # Only configured vehicle wells are attached to an existing positive series.
            attach = positive.index.union(block.index[vehicle_zero])
            df.loc[attach, "series"] = name
            # Non-vehicle zero wells retain their own competitor name and are not silently lost.
            for idx in block.index[zero & ~vehicle_zero]:
                if pd.isna(df.loc[idx, "series"]):
                    df.loc[idx, "series"] = df.loc[idx, "competitor"]
        else:
            valid = block[(block["include"]) & block["competitor"].notna()]
            if not valid.empty:
                name = str(valid["competitor"].mode().iloc[0])
                df.loc[valid.index, "series"] = name
    if group_by == "competitor name only":
        # Across files and rows, identical competitor names intentionally share a series.
        df["series"] = df["series"].where(df["series"].notna(), df["competitor"])
    return df


def build_prism_table(
    tidy: pd.DataFrame,
    value_column: str,
    vehicle_names: Iterable[str] = ("water", "vehicle", "DMSO", "buffer", "PBS"),
    group_by: str = "plate row",
    zero_handling: str = "Replace 0 with pseudo-concentration",
    add_log10: bool = False,
) -> tuple[pd.DataFrame, float | None, float | None]:
    """Build a Prism XY table with one side-by-side column per replicate."""
    df = _attach_series(tidy, vehicle_names, group_by)
    df = df[df["include"] & df["series"].notna()].copy()
    pseudo, step = dilution_stats(df)
    if zero_handling.startswith("Replace") and pseudo is not None:
        df["plot_conc"] = df["conc_nM"].replace(0, pseudo)
    elif zero_handling.startswith("Keep"):
        df["plot_conc"] = df["conc_nM"]
    else:
        df = df[df["conc_nM"] != 0].copy()
        df["plot_conc"] = df["conc_nM"]
    # Series order is first appearance, which follows plate/file order.
    series_order = list(dict.fromkeys(df["series"].tolist()))
    counts: dict[str, int] = {}
    for series in series_order:
        counts[series] = max(1, int(df.loc[df["series"] == series].groupby("conc_nM").size().max()))
    columns: list[str] = ["X"]
    if add_log10:
        columns.append("log10(X)")
    for series in series_order:
        columns.extend([f"{series}__rep{i + 1}" for i in range(counts[series])])
    rows: list[list[object]] = []
    header = ["Concentration (nM)"] + (["log10(conc)"] if add_log10 else [])
    for series in series_order:
        header.extend([series] + [""] * (counts[series] - 1))
    rows.append(header)
    concs = sorted(df["plot_conc"].dropna().unique())
    for x in concs:
        raw_x = x
        if zero_handling.startswith("Replace") and pseudo is not None and np.isclose(x, pseudo):
            raw_x = 0.0
        row: list[object] = [float(x)]
        if add_log10:
            row.append(float(np.log10(x)) if x > 0 else np.nan)
        for series in series_order:
            vals = df[(df["series"] == series) & (df["plot_conc"] == x)].sort_values(["source_file", "row", "well"])[value_column].dropna().tolist()
            vals = vals[:counts[series]]
            row.extend(vals + [np.nan] * (counts[series] - len(vals)))
        rows.append(row)
    return pd.DataFrame(rows, columns=columns), pseudo, step


def single_point_controls(
    tidy: pd.DataFrame,
    vehicle_names: Iterable[str] = ("water", "vehicle", "DMSO", "buffer", "PBS"),
    group_by: str = "plate row",
) -> pd.DataFrame:
    """Summarize included series with one distinct concentration."""
    df = _attach_series(tidy, vehicle_names, group_by)
    df = df[df["include"] & df["series"].notna()].copy()
    if df.empty:
        return pd.DataFrame(columns=["series", "conc_nM", "n", "mBU_corrected_mean", "mBU_corrected_SD", "mBU_corrected_%CV", "donor_RLU_mean", "donor_RLU_SD", "donor_RLU_%CV", "acceptor_RLU_mean", "acceptor_RLU_SD", "acceptor_RLU_%CV"])
    counts = df.groupby("series")["conc_nM"].nunique()
    keep = counts[counts == 1].index
    df = df[df["series"].isin(keep)]
    out: list[dict] = []
    def stats(s: pd.Series) -> tuple[int, float, float, float]:
        x = pd.to_numeric(s, errors="coerce").dropna()
        n = len(x); mean = float(x.mean()) if n else np.nan
        sd = float(x.std(ddof=1)) if n > 1 else np.nan
        cv = float(sd / mean * 100) if n > 1 and mean != 0 else np.nan
        return n, mean, sd, cv
    for series, g in df.groupby("series", sort=False):
        n1, m1, s1, c1 = stats(g["mBU_corrected"])
        n2, m2, s2, c2 = stats(g["donor_RLU"])
        n3, m3, s3, c3 = stats(g["acceptor_RLU"])
        out.append({"series": series, "conc_nM": float(g["conc_nM"].iloc[0]), "n": n1,
                    "mBU_corrected_mean": m1, "mBU_corrected_SD": s1, "mBU_corrected_%CV": c1,
                    "donor_RLU_mean": m2, "donor_RLU_SD": s2, "donor_RLU_%CV": c2,
                    "acceptor_RLU_mean": m3, "acceptor_RLU_SD": s3, "acceptor_RLU_%CV": c3})
    return pd.DataFrame(out)


def build_background_samples(
    results: Iterable[CalculationResult], background_mode: str
) -> pd.DataFrame:
    """Audit the finite-mBU wells actually used for each applied background."""
    labels = {
        "source_file": "Source file", "well": "Well", "mRNA": "mRNA",
        "labelled_guide": "Labelled guide", "competitor": "Competitor",
        "conc_nM": "Concentration (nM)", "donor_RLU": "Donor RLU",
        "acceptor_RLU": "Acceptor RLU", "BRET": "BRET", "mBU": "mBU",
    }
    columns = list(labels.values()) + [
        "Background mode", "Mean background applied (mBU)",
        "Background sample count", "Corrected mBU",
    ]
    frames: list[pd.DataFrame] = []
    for result in results:
        if result.background is None or background_mode == "none":
            continue
        # Include controls even if excluded from curves. The background selection,
        # not the curve Include checkbox, determines subtraction.
        used = result.tidy.loc[
            result.tidy["well"].isin(result.background_wells)
            & result.tidy["mBU"].notna()
        ].copy()
        if used.empty:
            continue
        audit = used[list(labels)].rename(columns=labels)
        audit["Background mode"] = background_mode
        audit["Mean background applied (mBU)"] = result.background
        audit["Background sample count"] = len(used)
        audit["Corrected mBU"] = used["mBU_corrected"].to_numpy()
        frames.append(audit)
    return pd.concat(frames, ignore_index=True)[columns] if frames else pd.DataFrame(columns=columns)
