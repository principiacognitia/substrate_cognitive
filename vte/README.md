# VTE Wrapper — Stage 3.2

This directory contains the Stage 3.2 VTE wrapper layer for `substrate_cognitive`.

The VTE module is intentionally separated from `stage3/`. It is not part of the toy cognitive model itself. It is a measurement and analysis layer that reads behavioral traces from `logs/vte/raw/` and writes derived VTE-style metrics to `logs/vte/analysis/` and `docs/results/vte/`.

## Boundary

The VTE wrapper must not import model internals from `stage3/`.

Allowed input:

- CSV or JSONL behavioral traces
- trial-level metadata
- geometry metadata, if supplied as data files

Disallowed input:

- internal gate state
- internal model configuration objects
- reward/threat configuration classes
- precomputed deliberation labels
- any variable that directly encodes the expected VTE conclusion

The wrapper treats model output as observed behavioral data.

## Input contract

A raw trace row describes an observed or reconstructed pose/action sample.

Required columns:

- `run_id`
- `seed`
- `trial`
- `tick`
- `x`
- `y`
- `heading`
- `choice_point_id`
- `at_choice_point`
- `action`
- `committed_path`
- `reward`
- `done`

Optional columns:

- `protocol`
- `condition`
- `ablation`
- `trial_phase`
- `pose_source`
- `event_type`
- `event_trial`
- `target_path`

Synthetic pose traces must set:

```text
pose_source = synthetic_from_model_trace
```

## Output contract

The wrapper produces trial-level and condition-level VTE metrics:

- `raw_idphi`
- `log_idphi`
- `z_idphi`
- `pause_ticks`
- `reorientation_count`
- `choice_point_duration`
- `vte_binary`

The wrapper may classify trials as VTE-like, but cognitive interpretation remains separate from measurement.

## Interpretation levels

Stage 3.2 separates three levels:

1. Trajectory measurement: IdPhi-like angular integration, pause duration, reorientation count.
2. Behavioral regime: VTE-like vs non-VTE-like trials.
3. Cognitive interpretation: deliberation, planning, procedural interruption.

Stage 3.2 primarily validates levels 1 and 2.

## Output locations

Raw logs:

```text
logs/vte/raw/
```

Analysis logs:

```text
logs/vte/analysis/
```

Publication artifacts:

```text
docs/results/vte/
```

## Design constraint

The wrapper is frozen before external biological datasets are inspected. External adapters may translate data into the same schema, but must not change the measurement core.
