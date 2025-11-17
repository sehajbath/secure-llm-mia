# MIMIC-IV-Ext-BHC Schema

## Overview
Clinician-authored hospital course summaries for the BHC benchmark. Ideal for text-only fine-tuning slices.

## Access Path
- Upload `mimic-iv-bhc.csv` (~2.5 GB) to Google Drive: `/content/drive/MyDrive/mimic-iv-bhc/mimic-iv-bhc.csv`.
- The notebooks reference this path via `BHC_CSV_PATH`; update it in one place if your Drive layout differs.

## Key Fields
- `note_id`: unique discharge note/BHC pair identifier (string).
- `input`: discharge note text without the BHC section.
- `target`: cleaned Brief Hospital Course summary.
- `input_tokens`: GPT-4 token length of `input`.
- `target_tokens`: GPT-4 token length of `target`.

## Processing Notes
- `src/data/bhc.py` provides `load_bhc_dataframe` and `bhc_to_canonical` helpers.
- Canonical export combines `input`/`target` into a single instruction-formatted `text` column and derives `tokens_estimate = input_tokens + target_tokens`.
- Synthetic chronological timestamps are generated for quick testing. Replace them with real discharge times by joining to MIMIC-IV tables before final experiments.
- Run modes (`subset` vs `full`) are controlled via `SECURE_LLM_MIA_RUN_MODE`. Subset processing reads ≤2k rows for rapid debugging; full mode streams the entire CSV.

## Privacy Checklist
- Redact any residual identifiers using PhysioNet redaction utilities.
- Store intermediate artifacts inside mounted Drive with restricted permissions.
- Avoid exporting raw PHI outside the secure runtime.

## TODOs
- [ ] Replace synthetic discharge timestamps with real admission/discharge times.
- [ ] Choose note-to-document composition rules per slice.
- [ ] Validate language distribution across time to measure drift.
