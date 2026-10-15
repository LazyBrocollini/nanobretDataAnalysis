"""Streamlit UI for NanoBRET to Prism."""
from __future__ import annotations
from io import BytesIO
import hashlib
import re

import numpy as np
import pandas as pd
import streamlit as st

from parsing import parse_workbook, ParsedPlate, GridBlock
from calc import build_tidy, build_prism_table, single_point_controls
from export import table_tsv, table_csv, workbook_bytes

st.set_page_config(page_title="NanoBRET to Prism", layout="wide")

@st.cache_data(show_spinner=False)
def cached_parse(data: bytes) -> ParsedPlate:
    return parse_workbook(data)

@st.cache_data(show_spinner=False)
def cached_tidy(data: bytes, source_file: str, mapping_ids: tuple, background_mode: str, manual_wells: tuple, threshold: float) -> object:
    plate = cached_parse(data)
    # The UI stores selected grid positions as integers; index the detected
    # block lists directly rather than looking them up by string keys.
    sample_blocks = plate.sample_blocks
    result_blocks = plate.result_blocks
    mapping = {
        "mrna": sample_blocks[mapping_ids[0]], "labelled_guide": sample_blocks[mapping_ids[1]],
        "competitor": sample_blocks[mapping_ids[2]], "conc_nM": sample_blocks[mapping_ids[3]],
        "donor": result_blocks[mapping_ids[4]], "acceptor": result_blocks[mapping_ids[5]],
    }
    if mapping_ids[6] != -1:
        mapping["bret"] = result_blocks[mapping_ids[6]]
    return build_tidy(plate, mapping, source_file, background_mode, manual_wells, threshold)


def block_label(block: GridBlock, index: int) -> str:
    return f"{index}: {block.label or '(unlabelled)'} [{block.height}x{block.width}]"


def mapping_selector(label: str, blocks: list[GridBlock], default: int, key: str) -> int:
    options = list(range(len(blocks)))
    return st.sidebar.selectbox(label, options, index=min(default, max(0, len(options)-1)), format_func=lambda i: block_label(blocks[i], i), key=key)


def combine_files(uploaded: list) -> tuple[pd.DataFrame, list, list[str]]:
    all_tidy: list[pd.DataFrame] = []
    results = []
    for file in uploaded:
        data = file.getvalue()
        plate = cached_parse(data)
        results.append((file, data, plate))
    return pd.DataFrame(), results, []

st.title("NanoBRET to Prism")
st.caption("Local-only preparation of NanoBRET plate-reader output for GraphPad Prism. Curve fitting remains in Prism.")

recalculate_clicked = st.sidebar.button("Recalculate", type="primary", help="Recalculate background correction and all outputs using the current settings.")
if recalculate_clicked:
    # Force the calculation-state synchronization below while preserving Include edits.
    st.session_state.pop("tidy_calc_signature", None)
st.sidebar.caption("Use Recalculate after selecting manual background wells. Include edits are preserved.")

uploads = st.sidebar.file_uploader("NanoBRET workbook(s)", type=["xlsx"], accept_multiple_files=True)
if not uploads:
    st.info("Upload one or more Promega GloMax NanoBRET .xlsx workbooks to begin.")
    st.stop()

def filename_suffix(files: list) -> str:
    """Create a filesystem-safe suffix from the uploaded workbook name(s)."""
    if len(files) == 1:
        stem = re.sub(r"[^A-Za-z0-9._-]+", "_", files[0].name.rsplit(".", 1)[0]).strip("._-")
        return f"_{stem}" if stem else ""
    return "_combined"

source_suffix = filename_suffix(uploads)
custom_suffix_input = st.sidebar.text_input(
    "Custom download suffix (optional)",
    value="",
    help="Appended after the original workbook suffix, for example Day1 or replicate_A.",
)
custom_suffix_clean = re.sub(r"[^A-Za-z0-9._-]+", "_", custom_suffix_input).strip("._-")
custom_suffix = f"_{custom_suffix_clean}" if custom_suffix_clean else ""
download_suffix = source_suffix + custom_suffix

try:
    first_plate = cached_parse(uploads[0].getvalue())
except Exception as exc:
    st.error(str(exc)); st.stop()

st.sidebar.subheader("Role mapping")
sample_blocks = first_plate.sample_blocks
result_blocks = first_plate.result_blocks
sample_defaults = {"mrna": 0, "labelled_guide": 1, "competitor": 2, "conc_nM": 3}
for role, guessed in list(sample_defaults.items()):
    labels = [b.label.lower() for b in sample_blocks]
    for i, text in enumerate(labels):
        if (role == "mrna" and "mrna" in text) or (role == "labelled_guide" and any(x in text for x in ("guide", "labelled", "tracer"))) or (role == "competitor" and "competitor" in text and "conc" not in text) or (role == "conc_nM" and "conc" in text):
            sample_defaults[role] = i; break
mRNA_i = mapping_selector("mRNA grid", sample_blocks, sample_defaults["mrna"], "map_mrna")
guide_i = mapping_selector("Labelled guide grid", sample_blocks, sample_defaults["labelled_guide"], "map_guide")
comp_i = mapping_selector("Competitor grid", sample_blocks, sample_defaults["competitor"], "map_comp")
conc_i = mapping_selector("Concentration grid", sample_blocks, sample_defaults["conc_nM"], "map_conc")

result_indices = {"donor": 0, "acceptor": 0, "bret": -1}
for i, b in enumerate(result_blocks):
    tx = b.label.lower()
    if "donor" in tx: result_indices["donor"] = i
    if "acceptor" in tx: result_indices["acceptor"] = i
    if "bret" in tx or "reading" in tx: result_indices["bret"] = i
# dropdowns are safer than relying on guessed positions
r_donor = mapping_selector("Donor Results grid", result_blocks, result_indices["donor"], "map_donor")
r_acceptor = mapping_selector("Acceptor Results grid", result_blocks, result_indices["acceptor"], "map_acceptor")
r_bret_options = [-1] + list(range(len(result_blocks)))
r_bret = st.sidebar.selectbox("BRET Results grid", r_bret_options, index=(r_bret_options.index(result_indices["bret"]) if result_indices["bret"] in r_bret_options else 0), format_func=lambda i: "Compute from Acceptor / Donor" if i == -1 else block_label(result_blocks[i], i), key="map_bret")

st.sidebar.subheader("Calculation settings")
background_mode = st.sidebar.radio("Background", ["auto-detect", "manual", "none"], index=0)
manual_wells: list[str] = []
if background_mode == "manual":
    manual_rows = first_plate.sample_blocks[0].row_labels if first_plate.sample_blocks else list("ABCDEFGH")
    manual_width = first_plate.sample_blocks[0].width if first_plate.sample_blocks else 12
    manual_options = [f"{r}{c}" for r in manual_rows for c in range(1, manual_width + 1)]
    manual_wells = st.sidebar.multiselect("Background wells", manual_options, default=[])
vehicle_text = st.sidebar.text_input("Vehicle names (comma-separated)", "water, vehicle, DMSO, buffer, PBS")
vehicle_names = [x.strip() for x in vehicle_text.split(",") if x.strip()]
zero_handling = st.sidebar.radio("Zero handling", ["Replace 0 with pseudo-concentration", "Keep 0 as 0", "Drop 0 nM row"], index=0)
threshold = st.sidebar.number_input("Flag wells with donor RLU below", min_value=0.0, value=0.0, step=1000.0)
use_corrected = st.sidebar.toggle("Use corrected mBU", value=True)
group_by = st.sidebar.radio("Group series by", ["plate row", "competitor name only"], index=0)
add_log10 = st.sidebar.checkbox("Add optional log10(conc) column", value=False)

mapping_ids_base = (mRNA_i, guide_i, comp_i, conc_i, r_donor, r_acceptor, r_bret)
all_results = []
for file in uploads:
    try:
        result = cached_tidy(file.getvalue(), file.name, mapping_ids_base, background_mode, tuple(manual_wells), threshold)
        all_results.append(result)
    except Exception as exc:
        st.error(f"{file.name}: {exc}")
if not all_results:
    st.stop()

combined = pd.concat([x.tidy for x in all_results], ignore_index=True)
# Background is calculated independently per file, as each plate is an independent experiment.
background_values = [x.background for x in all_results if x.background is not None]
warnings = [w for x in all_results for w in x.warnings]
for warning in dict.fromkeys(warnings):
    st.warning(warning)
if background_values:
    bg_summary = pd.DataFrame({
        "source_file": [file.name for file in uploads],
        "background_wells": [", ".join(result.background_wells) for result in all_results],
        "background_mBU": [result.background if result.background is not None else np.nan for result in all_results],
    })
    st.info("Background correction: mBU_corrected = mBU - the mean donor-only background.\n\n" + "; ".join(f"{row.source_file}: {row.background_mBU:.3f} mBU from {row.background_wells}" for row in bg_summary.itertuples()))

# Synchronize recalculated values when settings change, while preserving manual Include edits.
file_signature = tuple((file.name, hashlib.sha1(file.getvalue()).hexdigest()) for file in uploads)
calc_signature = (file_signature, mapping_ids_base, background_mode, tuple(sorted(manual_wells)), float(threshold))
previous_signature = st.session_state.get("tidy_calc_signature")
if previous_signature != calc_signature or "editable_tidy" not in st.session_state:
    previous = st.session_state.get("editable_tidy")
    if previous is not None and "well" in previous and "include" in previous:
        include_by_well = previous.set_index("well")["include"].to_dict()
        combined["include"] = combined["well"].map(include_by_well).fillna(combined["include"]).astype(bool)
    st.session_state.editable_tidy = combined.copy()
    st.session_state.tidy_calc_signature = calc_signature
editable = st.session_state.editable_tidy.copy()

# Build outputs reactively from edited Include values.
value_col = "mBU_corrected" if use_corrected else "mBU"
prism_mbu, pseudo, step = build_prism_table(editable, value_col, vehicle_names, group_by, zero_handling, add_log10)
prism_donor, _, _ = build_prism_table(editable, "donor_RLU", vehicle_names, group_by, zero_handling, add_log10)
prism_acceptor, _, _ = build_prism_table(editable, "acceptor_RLU", vehicle_names, group_by, zero_handling, add_log10)
controls = single_point_controls(editable, vehicle_names, group_by)

if pseudo is not None and zero_handling.startswith("Replace"):
    st.info(f"Zero handling: replaced 0 nM with pseudo-concentration {pseudo:.6g} nM (geometric-mean dilution step {step:.6g}).")
else:
    st.info(f"Zero handling: {zero_handling}.")
negative_included = editable.loc[editable["include"] & editable["mBU_corrected"].lt(0)]
if negative_included.empty:
    st.success("Background check: no included wells have negative corrected mBU.")
else:
    st.warning(f"Background check: {len(negative_included)} included wells have negative corrected mBU; review the background selection or Include flags.")

plate_tab, prism_tab, tidy_tab, download_tab = st.tabs(["Plate view and QC", "Prism tables", "Tidy data", "Downloads"])
with plate_tab:
    st.subheader("Plate QC")
    for metric, col in [("BRET", "BRET"), ("Donor RLU", "donor_RLU"), ("Acceptor RLU", "acceptor_RLU")]:
        st.markdown(f"**{metric}**")
        view = editable.pivot(index="row", columns="col", values=col).sort_index()
        st.dataframe(view.style.format("{:.3f}" if metric == "BRET" else "{:.0f}"), use_container_width=True)
    flagged = editable[editable.flag_low_donor]
    if not flagged.empty:
        st.warning(f"{len(flagged)} wells flagged for low donor RLU: {', '.join(flagged.well.tolist())}")
    st.dataframe(editable[["well", "mRNA", "labelled_guide", "competitor", "conc_nM", "include", "flag_low_donor"]], use_container_width=True, hide_index=True)
with prism_tab:
    for title, table in [("NanoBRET dose-response (mBU)", prism_mbu), ("Donor raw (RLU)", prism_donor), ("Acceptor raw (RLU)", prism_acceptor)]:
        st.subheader(title)
        st.dataframe(table.style.format(precision=2), use_container_width=True, hide_index=True)
        st.code(table_tsv(table), language="tsv")
    st.subheader("Single-point controls")
    st.dataframe(controls.style.format(precision=2), use_container_width=True, hide_index=True)
    with st.expander("How to import into Prism"):
        st.markdown("1. New table → **XY**.\n2. Choose **Enter n replicate values in side-by-side subcolumns** and set *n* to the maximum replicate count.\n3. Paste the copy-ready block with column headers.\n4. Analyse with Nonlinear regression → **log(inhibitor) vs. response – variable slope (four parameters)**.\n5. Transform X to log10 if X is in nM.")
with tidy_tab:
    st.subheader("Editable tidy data")
    edited = st.data_editor(editable, use_container_width=True, hide_index=True, key="tidy_editor", column_config={"include": st.column_config.CheckboxColumn("Include")})
    st.session_state.editable_tidy = edited
with download_tab:
    st.subheader("Downloads")
    for name, table in [("Prism_DoseResponse_mBU", prism_mbu), ("Prism_Donor_RLU", prism_donor), ("Prism_Acceptor_RLU", prism_acceptor)]:
        st.download_button(f"Download {name}.csv", table_csv(table), file_name=f"{name}{download_suffix}.csv", mime="text/csv", key=f"dl_{name}")
    combined_xlsx = workbook_bytes({"Prism_DoseResponse_mBU": prism_mbu, "Prism_Donor_RLU": prism_donor, "Prism_Acceptor_RLU": prism_acceptor, "SingleControls": controls, "Tidy_data": editable})
    st.download_button("Download combined Excel workbook", combined_xlsx, file_name=f"NanoBRET_to_Prism{download_suffix}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
