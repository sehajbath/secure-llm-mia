# MIMIC-IV v3.1 (HOSP) Schema

## Overview
Structured admissions, patient demographics, and ICU stays. Serves as the backbone for chronological ordering and replay sampling.

## Key Tables
- `admissions.csv`: contains `hadm_id`, `admittime`, `dischtime`, `deathtime`.
- `patients.csv`: `subject_id`, `anchor_age`, `anchor_year` for de-identification.
- `transfers.csv`: transfer events for ICU and ward locations.

## Integration Notes
- Join with Ext datasets to align notes/time-series to the correct chronological slice using `subject_id` and `hadm_id`.
- Build slice metadata: assign `slice_id`, `split_tag`, `member_status` (train/val/test/global holdout).
- Persist ID lists (`artifacts/slice_t/ids/*.txt`) to guarantee disjointness across attacks.

## TODOs
- [ ] Document secure location of CSV files.
- [ ] Confirm all timestamps converted to UTC.
- [ ] Encode patient age bins for fairness analysis (optional).
