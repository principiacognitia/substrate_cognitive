# Redish LRA 2024 adapter

## Status

This adapter is the current preferred biological baseline for simple VTE at a left-right-alternate choice point.

The healthy control LRA subset is usable for biological VTE / outcome / deliberation comparisons.

DREADD / mPFC perturbation rows are retained separately and must not be mixed into the healthy baseline.

## Dataset role

Dataset name on disk:

- `2024 RedishLab: Recordings from medial prefrontal, dorsolateral striatum, and hippocampus on Left-Right-Alternate; DREADD disruption of mPFC`

Local root used during development:

- `E:\CRS-1\SUBSTRATE_COGNITIVE\LOGS\VTE_DATASETS\REDISH 2024`

Main cohorts:

- `lra`: healthy control LRA sessions;
- `mpfc_dreadds`: DREADD / vehicle perturbation sessions.

## Important methodological policy

### Healthy baseline

Use only:

- cohort: `lra`
- treatment: `control`
- native VTE source: `VTELap.ChoicePoint`

### Perturbation data

Do not mix with healthy baseline:

- cohort: `mpfc_dreadds`
- treatments: `DCZ`, `DCZ-Saline`, `VEH`, `VEH-Saline`

These rows are biological perturbation data, not healthy-control VTE labels.

### Choice direction

Raw action codes are preserved but not mapped to left/right.

Current policy:

- `native_action_code = code_1/code_2/code_3/code_4`
- `choice_direction_usable = false`
- `choice_direction_policy = raw_event_code_only_no_left_right_mapping`

Do not compare biological raw event codes directly against synthetic `left/right` labels.

## Main patches

### Patch 17A: probe and candidate extraction

Scripts:

- `probe_lra_dataset.py`
- `extract_lra_candidates.py`

Outputs include file inventories, candidate fields, and endpoint drafts.

### Patch 17B / 17B2: HDF5 / MATLAB v7.3 audit

Purpose:

- resolve MATLAB v7.3 / HDF5 structures;
- inspect `IdPhi`, `AvgIdPhi`, `pVTE`, `VTELap`;
- inspect `BEHAVIOR`, feeder events, error events, zone times.

Important finding:

- LRA control IdPhi file contains native `VTELap.ChoicePoint`;
- behavior file contains event-level `ChoicePointEntry`, `ChoicePointExit`, `FeedersFired`, `ErrorNotFired`.

### Patch 17C-17D: canonical choice endpoint

Patch 17D includes both healthy control and DREADD-derived rows.

Important output:

- `redish_lra_canonical_choice_endpoint_usable.csv`

Do not use this file directly as healthy baseline without policy split.

### Patch 17E: policy split

Preferred healthy-control baseline:

- `redish_lra17e_healthy_control_baseline.csv`

Perturbation endpoint:

- `redish_lra17e_dreadd_perturbation_continuous.csv`

Excluded rows:

- `redish_lra17e_excluded_from_healthy_baseline.csv`

Interpretation:

- healthy baseline uses native `VTELap.ChoicePoint`;
- DREADD rows are continuous perturbation data only;
- threshold-derived labels are excluded from healthy baseline.

### Patch 17F-17G: choice direction / same-trial audit

Purpose:

- audit raw feeder/error event codes;
- restrict events to same-trial events after `ChoicePointExit` and before next `ChoicePointEntry`;
- avoid left/right inference.

Patch 17G strict usable output:

- `redish_lra17g_choice_direction_same_trial_usable.csv`

### Patch 17H: healthy choice baseline

Preferred strict action-code subset:

- `redish_lra17h_healthy_choice_baseline.csv`

Use this when same-trial event action codes are required.

Use Patch 17E when full healthy VTE/outcome baseline is sufficient.

## Recommended files

Primary biological VTE baseline:

- `logs\vte\redish_lra_2024\patch17e_policy_split\redish_lra17e_healthy_control_baseline.csv`

Strict same-trial action-code subset:

- `logs\vte\redish_lra_2024\patch17h_healthy_choice_baseline\redish_lra17h_healthy_choice_baseline.csv`

Perturbation endpoint, not healthy baseline:

- `logs\vte\redish_lra_2024\patch17e_policy_split\redish_lra17e_dreadd_perturbation_continuous.csv`

## Comparability policy

Comparable to synthetic fork models:

- reward / outcome;
- VTE binary vs synthetic VTE proxy;
- `lab_idphi` / z-normalized deliberation proxy;
- dwell proxy;
- session-normalized fields.

Not comparable without additional decoding:

- biological `native_action_code`;
- synthetic `left/right`;
- DREADD treatment rows vs healthy baseline.

## Current best use

Use this adapter as the biological healthy VTE benchmark for balanced-fork model comparisons.
