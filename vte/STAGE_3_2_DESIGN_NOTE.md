# Stage 3.2 Design Note: Read-only VTE Wrapper

## Purpose

Stage 3.2 introduces a read-only VTE wrapper for converting behavioral trajectory logs into VTE-style metrics.

The wrapper is not a component of the agent. It is a measurement transform over already-produced behavioral traces. This separation is necessary to avoid fitting the VTE metric to the internal model structure.

## Scientific target

The target is not to claim biological equivalence directly. The target is narrower:

> If Stage 3 trajectories contain conflict-sensitive pause-and-reorient dynamics, a read-only VTE wrapper should detect IdPhi-like signatures under conflict, contingency shift, and procedural disruption.

This is compatible with the VTE literature, where VTE is operationalized as pausing and orienting at a decision point, often quantified through angular change or IdPhi-like trajectory measures (Redish, 2016).

## Measurement boundary

The wrapper reads only externalized trace data.

It must not read:

- gate values
- internal agent states
- model configuration objects
- reward/threat parameters
- causal labels such as `deliberation`
- precomputed acceptance checks

It may read:

- pose traces
- trial indices
- action labels
- committed path labels
- outcome labels
- event markers, if these are encoded as external trial metadata

## Data flow

```text
model / simulator
  -> logs/vte/raw/*.csv

vte wrapper
  -> logs/vte/analysis/*.csv
  -> docs/results/vte/tables/*.csv
  -> docs/results/vte/figures/*.png
  -> docs/results/vte/reports/*.md
  -> docs/results/vte/manifests/*.json
```

## Raw trace schema

Required columns:

| column | meaning |
|---|---|
| `run_id` | run identifier |
| `seed` | random seed |
| `trial` | trial number |
| `tick` | within-trial time step |
| `x` | x coordinate |
| `y` | y coordinate |
| `heading` | heading angle, radians unless otherwise specified |
| `choice_point_id` | decision point identifier |
| `at_choice_point` | boolean/int marker |
| `action` | observed action |
| `committed_path` | final selected path |
| `reward` | trial outcome or local reward |
| `done` | episode termination marker |

Optional metadata columns:

| column | meaning |
|---|---|
| `protocol` | e.g. baseline, shock, treat |
| `condition` | e.g. R1_T2 |
| `ablation` | e.g. full, novg, novp, nox |
| `trial_phase` | pre, event, post |
| `pose_source` | real, simulated, synthetic_from_model_trace |
| `event_type` | shock, treat, reversal, transition_violation |
| `event_trial` | event-aligned trial index |
| `target_path` | experimentally relevant path |

## Core metrics

The first implementation should compute:

| metric | definition |
|---|---|
| `raw_idphi` | sum of absolute heading changes within the choice-point window |
| `log_idphi` | log-transformed `raw_idphi` |
| `z_idphi` | seed/session-normalized `log_idphi` |
| `pause_ticks` | number of ticks at the choice point |
| `reorientation_count` | number of sign-reversing or thresholded heading changes |
| `choice_point_duration` | duration of choice-point occupancy |
| `vte_binary` | thresholded VTE-like trial label |

The thresholding rule must be fixed before external data are inspected.

## Non-goals

Stage 3.2 does not claim:

- direct neural homology;
- direct biological identity between the toy model and rodent behavior;
- that all VTE-like trajectories imply deliberation;
- that zIdPhi alone proves model-based planning.

## External validation rule

External datasets should be evaluated only after the wrapper specification is frozen.

Allowed changes after freezing:

- file-format adapters;
- column-name adapters;
- maze geometry adapters.

Disallowed changes after freezing:

- changing IdPhi definition per dataset;
- changing z-score procedure per dataset;
- changing VTE threshold per dataset;
- adding dataset-specific correction terms that improve agreement with one laboratory.

## Expected Stage 3.2 outputs

Tables:

- `Table_3_2_VTE_trial_metrics.csv`
- `Table_3_2_VTE_condition_summary.csv`
- `Table_3_2_VTE_event_window_summary.csv`

Figures:

- `Figure_3_2_VTE_zIdPhi_distribution.png`
- `Figure_3_2_VTE_by_condition.png`
- `Figure_3_2_VTE_event_aligned.png`
- `Figure_3_2_VTE_pause_vs_idphi.png`

Reports:

- `Stage3_2_VTE_Report.md`

Manifests:

- `stage3_2_vte_manifest.json`

## References

Redish, A. D. (2016). Vicarious trial and error. *Nature Reviews Neuroscience, 17*(3), 147–159. https://doi.org/10.1038/nrn.2015.30

Hasz, B. M., & Redish, A. D. (2018). Deliberation and procedural automation on a two-step task for rats. *Frontiers in Integrative Neuroscience, 12*, Article 30. https://doi.org/10.3389/fnint.2018.00030

Miller, K. J., Botvinick, M. M., & Brody, C. D. (2017). Dorsal hippocampus contributes to model-based planning. *Nature Neuroscience, 20*(9), 1269–1276. https://doi.org/10.1038/nn.4613

Akam, T., Rodrigues-Vaz, I., Marcelo, I., Zhang, X., Pereira, M., Oliveira, R. F., Dayan, P., & Costa, R. M. (2021). The anterior cingulate cortex predicts future states to mediate model-based action selection. *Neuron, 109*(1), 149–163. https://doi.org/10.1016/j.neuron.2020.10.013