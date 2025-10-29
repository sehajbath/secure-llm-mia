# MIMIC-IV-Ext-22MCTS Schema

## Overview
Time-aligned ICU signals with relative timestamps per admission. Use these notes to build chronological slices for continual fine-tuning.

## Key Fields
- `subject_id`: patient identifier (use for member/non-member disjointness).
- `hadm_id`: admission identifier.
- `charttime`: relative timestamp for measurements.
- `eventtype`: structured event category.
- `valuenum`, `valueuom`: numeric values and units.

## Processing Notes
- Align measurements to absolute time using admission `intime` if available.
- Bucket events into fixed-length windows before tokenization; store aggregated summaries in the canonical Parquet format.
- Ensure PHI is never exported. Derived features only.

## Chronological Splits
- Sort by discharge time from the linked admissions table.
- Assign each patient to the earliest slice in which they appear.
- Track token counts per slice to respect the 25M token budget.

## TODOs
- [ ] Insert secure data extraction paths (PhysioNet credential required).
- [ ] Confirm timezone handling (UTC recommended).
- [ ] Define aggregation granularity (e.g., 1h bins) to match LLM context budget.
