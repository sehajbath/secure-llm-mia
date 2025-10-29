# MIMIC-IV-Ext-BHC Schema

## Overview
Clinician-authored hospital course summaries for the BHC benchmark. Ideal for text-only fine-tuning slices.

## Key Fields
- `subject_id`: unique patient identifier.
- `hadm_id`: admission identifier.
- `discharge_time`: timestamp used for chronological ordering.
- `note_text`: raw narrative notes.
- `summary_text`: optional target summaries (useful for evaluation but avoid leakage in label-free attacks).

## Processing Notes
- Concatenate multi-note sequences per patient within a slice window.
- Apply note cleaning (lowering noise, removing templates) but preserve clinical semantics.
- Track token estimates with the Llama tokenizer to enforce per-slice budgets.

## Privacy Checklist
- Redact any residual identifiers using PhysioNet redaction utilities.
- Store intermediate artifacts inside mounted Drive with restricted permissions.
- Avoid exporting raw PHI outside the secure runtime.

## TODOs
- [ ] Provide secure data load path (e.g., `'/content/drive/MyDrive/mimic/bhc/'`).
- [ ] Choose note-to-document composition rules per slice.
- [ ] Validate language distribution across time to measure drift.
