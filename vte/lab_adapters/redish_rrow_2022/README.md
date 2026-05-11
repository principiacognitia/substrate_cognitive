# Redish RRow 2022 adapter

## Status

This adapter is retained as a Restaurant Row / two-stage decision dataset adapter.

It is not the preferred biological baseline for simple left-right VTE at a balanced fork.

## Dataset role

- biological Restaurant Row task;
- two decision stages:
  - offer_zone: accept / skip;
  - wait_zone: earn / quit;
- four restaurants / offers;
- delay and reward structure are task-defining variables;
- useful for decision-stage extraction and cost/delay analyses;
- not a clean left-right fork benchmark.

## Main patches

### Patch 16C-16D: decision endpoint

Canonicalized Restaurant Row rows into decision-stage records.

Important outputs:

- `Table_Redish_RRow_Decision_Endpoint.csv`
- `Table_Redish_RRow_Decision_Endpoint_Usable.csv`
- `Table_Redish_RRow_Decision_By_Stage.csv`
- `Table_Redish_RRow_Decision_By_Delay.csv`
- `Table_Redish_RRow_Decision_By_Session.csv`

### Patch 16E: canonical decision endpoint

Preferred RRow canonical endpoint.

Important output:

- `redish_rrow_canonical_decision_endpoint.csv`

Canonical fields include:

- `dataset_id`
- `trace_origin`
- `task_family`
- `subject_id`
- `session_id`
- `trial`
- `decision_stage`
- `restaurant_id`
- `chosen_action`
- `outcome`
- `reward`
- `cost`
- `dwell_proxy`
- `deliberation_proxy`
- `lab_idphi`
- z-normalized session/stage fields

### Patch 16F: biological-vs-synthetic comparability

Historical comparability attempt for RRow.

Use with caution. It is not the current target for simple balanced-fork VTE because RRow deliberation is confounded with:

- offer delay;
- restaurant identity;
- two-stage accept/skip/earn/quit structure;
- reward/cost policy.

## Current interpretation

Use this adapter for:

- Restaurant Row extraction;
- delay/cost decision analyses;
- testing generic biological table plumbing.

Do not use it as the primary healthy left-right VTE benchmark.

## Known limitations

- `chosen_action` labels are task-specific, not left/right.
- `cost` is meaningful and non-balanced.
- `reward` depends on task stage and offer/wait structure.
- Comparability to synthetic left-right fork behavior requires a separate semantic policy.

## Recommended current baseline

For simple healthy biological VTE at a choice point, use:

- `vte/lab_adapters/redish_lra_2024`
- Patch 17E healthy control baseline
- Patch 17H strict same-trial action-code subset
