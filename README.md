# NanoBRET to Prism

A local Streamlit app that parses Promega GloMax NanoBRET Excel exports, applies configurable donor-only background subtraction, builds tidy plate data, and produces Prism-ready XY tables for corrected mBU, donor RLU, and acceptor RLU. Curve fitting is intentionally left to Prism.

## Run

1. Create/activate a Python 3.10+ environment.
2. Install dependencies: `pip install -r requirements.txt`.
3. Run: `streamlit run app.py`.
4. Upload `nanobret example.xlsx` or one or more compatible `.xlsx` files.
5. Review the automatic role mapping, QC, editable Include column, Prism copy blocks, and downloads.

## Project files

- `app.py` — Streamlit UI and reactive controls.
- `parsing.py` — case-insensitive sheet matching and generic stacked-grid detection.
- `calc.py` — mBU/background/QC calculations, series grouping, controls, and Prism layouts.
- `export.py` — TSV, CSV, and unmerged Excel exports.
- `tests/test_reference.py` — reference workbook acceptance checks.

## Assumptions

- When multiple positive-concentration competitor names occur in one plate row, the modal competitor name defines that row's series.
- A zero-concentration well is attached to an existing positive series only when its competitor name matches the configured vehicle list; otherwise it remains with its own competitor series.
- Background is calculated independently per uploaded workbook, because each file is treated as an independent experiment.
- The `plate row` and `competitor name only` grouping modes produce the same display label when identical competitor names are used; the latter is intended for combining matching names across uploaded files.
- Plotly preview plots are represented by the table/QC views in this minimal delivery; no curve fitting is performed in the app.
