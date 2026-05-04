# Stage 3.2 Design Note: Read-only VTE Wrapper

## Purpose

Stage 3.2 introduces a read-only VTE wrapper for converting behavioral trajectory logs into VTE-style metrics.

The wrapper is not a component of the agent. It is a measurement transform over already-produced behavioral traces. This separation is necessary to reduce the risk of fitting the VTE metric to the internal model structure.

---

## Current status

- **Stage 3.2A: VTE wrapper core — Complete.**
  - raw trace schema;
  - IdPhi-like angular integration;
  - pause and reorientation metrics;
  - seed/session-normalized `z_idphi`;
  - thresholded `vte_binary`;
  - wrapper CLI;
  - schema, metrics, and wrapper tests.

- **Stage 3.2B: Stage 3 log adapter + batch analysis — Final debug.**
  - Stage 3 step-log to VTE trace-schema adapter;
  - metadata propagation from Stage 3 logs;
  - analysis tables and figures;
  - batch workflow.

- **Stage 3.2C: biological-lab comparability layer — In development.**
  - geometry registry;
  - biological tracking adapters;
  - fixed threshold profiles;
  - cross-dataset comparison reports.

---

## Scientific target

The target is not direct biological equivalence. The target is narrower:

> If Stage 3 trajectories contain conflict-sensitive pause-and-reorient dynamics, a read-only VTE wrapper should detect IdPhi-like signatures under conflict, contingency shift, and procedural disruption.

This is compatible with the VTE literature, where VTE is operationalized as pausing and orienting at a decision point, often quantified through angular change or IdPhi-like trajectory measures (Redish, 2016).

---

## Measurement boundary

The wrapper reads only externalized trace data.

It must not read:

- gate values;
- internal agent states;
- model configuration objects;
- reward/threat parameter objects;
- causal labels such as `deliberation`;
- precomputed acceptance checks.

It may read:

- pose traces;
- trial indices;
- action labels;
- committed path labels;
- outcome labels;
- event markers, if these are encoded as external trial metadata;
- geometry metadata, if supplied as an external data file.

---

## Data flow

```text
model / simulator / lab adapter
  -> logs/vte/raw/*.csv

vte wrapper
  -> logs/vte/<run>/vte_trial_metrics.csv

vte analysis
  -> logs/vte/<analysis>/tables
  -> logs/vte/<analysis>/figures
  -> logs/vte/<analysis>/reports
  -> logs/vte/<analysis>/metadata
```

Publication-facing or reviewer-facing artifacts may later be copied into:

```text
docs/results/vte/
```

---

## Raw trace schema

Required columns:

| column | meaning |
|---|---|
| `run_id` | run identifier |
| `seed` | random seed or session identifier |
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
| `pose_source` | real, simulated, synthetic_from_stage3_steps |
| `event_type` | shock, treat, reversal, transition_violation |
| `event_trial` | event-aligned trial index |
| `target_path` | experimentally relevant path |
| `total_reward` | trial-level total reward, if available |
| `terminal_action` | terminal action code, if available |

---

## Pose-source labels

The `pose_source` column is required for interpretation.

Current Stage 3 adapter output:

```text
pose_source = synthetic_from_stage3_steps
```

Expected future biological adapter output:

```text
pose_source = biological_tracking
```

Possible simulator-native output:

```text
pose_source = simulated_pose
```

These labels must not affect metric definitions. They affect interpretation only.

---

## Core metrics

| metric | definition |
|---|---|
| `raw_idphi` | sum of absolute heading changes within the choice-point window |
| `log_idphi` | log-transformed `raw_idphi` |
| `z_idphi` | seed/session-normalized `log_idphi` |
| `pause_ticks` | number of ticks at the choice point |
| `reorientation_count` | number of thresholded heading-change reversals or alternations |
| `choice_point_duration` | duration of choice-point occupancy |
| `vte_binary` | thresholded VTE-like trial label |

The thresholding rule must be fixed before external biological datasets are inspected.

---

## Current CLIs

Translate Stage 3 step logs into VTE trace schema:

```bash
python -m vte.analysis.translate_stage3_steps_to_vte_trace \
  --input-csv <stage3_steps.csv> \
  --output-csv logs/vte/raw/<trace_name>.csv \
  --run-id <run_id>
```

Run wrapper:

```bash
python -m vte.analysis.run_stage3_2_vte \
  --input-csv logs/vte/raw/<trace_name>.csv \
  --output-dir logs/vte/<run_dir>
```

Analyze metrics:

```bash
python -m vte.analysis.analyze_stage3_2_vte \
  --metrics-csv logs/vte/<run_dir>/vte_trial_metrics.csv \
  --output-dir logs/vte/<analysis_dir>
```

Batch workflow:

```bash
python -m vte.analysis.run_stage3_2_vte_batch \
  --input-root <stage3_or_vte_input_root> \
  --output-root logs/vte/<batch_dir>
```

The batch CLI is part of Stage 3.2B final debug.

---

## Expected outputs

Wrapper output:

- `vte_trial_metrics.csv`
- `vte_wrapper_meta.json`

Analysis tables:

- `Table_3_2_VTE_Overall_Summary.csv`
- `Table_3_2_VTE_By_Seed.csv`
- `Table_3_2_VTE_By_Condition.csv`
- `Table_3_2_VTE_By_Committed_Path.csv`
- `Table_3_2_VTE_By_Condition_x_Path.csv`
- `Table_3_2_VTE_Distribution_By_Path.csv`

Analysis figures:

- `Figure_3_2_VTE_Rate_By_Path.png`
- `Figure_3_2_IdPhi_By_Path_Boxplot.png`
- `Figure_3_2_VTE_Rate_By_Seed.png`
- `Figure_3_2_IdPhi_vs_Pause.png`

Reports and metadata:

- `Stage3_2_VTE_Analysis_Report.md`
- `stage3_2_vte_analysis_meta.json`

---

## Non-goals

Stage 3.2 does not claim:

- direct neural homology;
- direct biological identity between the toy model and rodent behavior;
- that all VTE-like trajectories imply deliberation;
- that zIdPhi alone proves model-based planning;
- that synthetic Stage 3 poses are equivalent to biological tracking data.

---

## External validation rule

External datasets should be evaluated only after the wrapper specification is frozen.

Allowed changes after freezing:

- file-format adapters;
- column-name adapters;
- coordinate transforms;
- maze geometry adapters;
- metadata harmonization.

Disallowed changes after freezing:

- changing IdPhi definition per dataset;
- changing z-score procedure per dataset;
- changing VTE threshold per dataset;
- adding dataset-specific correction terms that improve agreement with one laboratory.

---

## Geometry registry requirement

Biological comparability requires a geometry registry.

The registry should define:

- maze/task identifier;
- coordinate system;
- choice-point zones;
- arm labels;
- commit zones;
- reward zones;
- route labels;
- event markers;
- inclusion/exclusion windows for IdPhi measurement.

This registry should also support static environment schematics for Stage 2 and Stage 3 so that external readers can understand the task topology without reading the simulator code.

## Stage 3.2C: biological-lab comparability layer

Status: In development.

The VTE wrapper is now separated from the Stage 3 simulator. The simulator writes
logs; the wrapper reads logs and emits VTE-compatible trial metrics. This boundary
is intentional: the wrapper must remain a measurement/translation layer, not a
model component.

The next layer is biological-lab comparability. Its purpose is to make outputs from
the model-side wrapper comparable with laboratory trajectory datasets without
rewriting the model or embedding lab-specific assumptions into the Stage 3 code.

### Geometry registry dependency

Biological-lab adapters must use an explicit environment-geometry registry.

A biological adapter must not hard-code choice points, route labels, maze arms,
or open/covered zones inside adapter logic. Instead, it must reference either:

1. `env_geometry/registries/builtin_env_geometries.json`, or
2. an external registry with the same conceptual schema.

The current built-in registry is schematic. Its coordinates are visualization
coordinates, not physical laboratory coordinates. It is sufficient for Stage 2 and
Stage 3 explanatory diagrams, but it is not a substitute for calibrated
laboratory maze coordinates.

Required conceptual fields for future lab adapters:

- `env_id`
- `task_family`
- `coordinate_system`
- `nodes`
- `edges`
- `zones`
- route labels
- choice-point zone definitions
- event-alignment rule

### Adapter boundary

A biological-lab adapter may transform raw trajectory data into the VTE trace
schema, but it must not change wrapper metrics after the fact.

Allowed adapter operations:

- map physical coordinates to registered zones;
- identify choice-point entry and exit windows;
- infer heading or turn-angle series;
- normalize trial/session identifiers;
- attach task metadata such as condition, animal/session ID, maze ID, and route.

Disallowed adapter operations:

- tune IdPhi thresholds to match a target paper;
- rewrite route outcomes after wrapper execution;
- define maze geometry implicitly in procedural code;
- use Stage 3 simulator internals;
- mix model-side and laboratory-side preprocessing rules without metadata.

### Comparability contract

The output of a biological adapter must be a valid VTE trace table accepted by
`vte.core.wrapper`.

At minimum, each trace row must expose:

- `run_id`
- `seed` or biological subject/session identifier
- `trial`
- `t`
- `x`
- `y`
- `heading`
- `choice_point_id`
- `at_choice_point`
- `done`

The wrapper may then compute the same core metrics:

- `raw_idphi`
- `log_idphi`
- `z_idphi`
- `vte_binary`
- `choice_point_duration`
- `pause_ticks`
- `reorientation_count`

This preserves the comparison boundary: biological data and model data may differ
in origin and geometry, but they enter the VTE measurement layer through the same
trace schema.

### Stage 3.2 status

- Stage 3.2A: VTE wrapper core — Complete.
- Stage 3.2B: Stage 3 log adapter + batch analysis — Final debug / closure.
- Stage 3.2C: biological-lab comparability layer — In development.

---

## References

Redish, A. D. (2016). Vicarious trial and error. *Nature Reviews Neuroscience, 17*(3), 147–159. https://doi.org/10.1038/nrn.2015.30

Hasz, B. M., & Redish, A. D. (2018). Deliberation and procedural automation on a two-step task for rats. *Frontiers in Integrative Neuroscience, 12*, Article 30. https://doi.org/10.3389/fnint.2018.00030

Miller, K. J., Botvinick, M. M., & Brody, C. D. (2017). Dorsal hippocampus contributes to model-based planning. *Nature Neuroscience, 20*(9), 1269–1276. https://doi.org/10.1038/nn.4613

Akam, T., Rodrigues-Vaz, I., Marcelo, I., Zhang, X., Pereira, M., Oliveira, R. F., Dayan, P., & Costa, R. M. (2021). The anterior cingulate cortex predicts future states to mediate model-based action selection. *Neuron, 109*(1), 149–163. https://doi.org/10.1016/j.neuron.2020.10.013