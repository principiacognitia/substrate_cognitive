# Stage 3.2: VTE Wrapper Service Layer

This directory contains the Stage 3.2 VTE wrapper layer for `substrate_cognitive`.

`vte/` is intentionally separated from `stage3/`. It is not part of the toy cognitive model and does not participate in action selection. It is a read-only measurement and analysis layer over externalized behavioral traces.

---

## Status

- **Stage 3.2A: VTE wrapper core — Complete.**  
  The wrapper reads externalized trace rows and writes trial-level VTE metrics. It does not call Stage 3 internals.

- **Stage 3.2B: Stage 3 log adapter + batch analysis — Final debug.**  
  The Stage 3 adapter converts Stage 3 step logs into the external VTE trace schema. Batch conversion and analysis are being stabilized.

- **Stage 3.2C: biological-lab comparability layer — In development.**  
  The next layer will add lab-tracking adapters, maze geometry registries, threshold profiles, and comparison reports.

---

## Boundary

The VTE wrapper must not import model internals from `stage3/`.

Allowed input:

- CSV or JSONL behavioral traces;
- trial-level metadata;
- geometry metadata, if supplied as data files.

Disallowed input:

- internal gate state;
- internal model configuration objects;
- reward/threat configuration classes;
- precomputed deliberation labels;
- any variable that directly encodes the expected VTE conclusion.

The wrapper treats model output as observed behavioral data.

---

## Input contract

A raw trace row describes an observed, simulated, or reconstructed pose/action sample.

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
- `total_reward`
- `terminal_action`

Stage 3 adapter traces currently set:

```text
pose_source = synthetic_from_stage3_steps
```

This label is important. It means the trace is reconstructed from Stage 3 step logs, not obtained from biological tracking data.

---

## Output contract

The wrapper produces trial-level VTE metrics:

- `raw_idphi`
- `log_idphi`
- `z_idphi`
- `pause_ticks`
- `reorientation_count`
- `choice_point_duration`
- `vte_binary`

The wrapper may classify trials as VTE-like, but cognitive interpretation remains separate from measurement.

---

## Current workflow

### 1. Translate Stage 3 step logs to VTE trace CSV

```bash
python -m vte.analysis.translate_stage3_steps_to_vte_trace \
  --input-csv logs/stage3/stage3_1_closure_raw/stage3_1b/<suite>/balanced/full/balanced_conflict_full_all_steps.csv \
  --output-csv logs/vte/raw/balanced_conflict_full_trace.csv \
  --run-id balanced_conflict_full
```

### 2. Run the VTE wrapper

```bash
python -m vte.analysis.run_stage3_2_vte \
  --input-csv logs/vte/raw/balanced_conflict_full_trace.csv \
  --output-dir logs/vte/stage3_2_smoke
```

### 3. Analyze VTE metrics

```bash
python -m vte.analysis.analyze_stage3_2_vte \
  --metrics-csv logs/vte/stage3_2_smoke/vte_trial_metrics.csv \
  --output-dir logs/vte/stage3_2_smoke_analysis
```

### 4. Batch workflow

```bash
python -m vte.analysis.run_stage3_2_vte_batch \
  --input-root logs/stage3/stage3_1_closure_raw/stage3_1b/<suite> \
  --output-root logs/vte/stage3_2_batch
```

The exact batch arguments may change while Stage 3.2B is in final debug.

---

## Output locations

Raw VTE traces:

```text
logs/vte/raw/
```

Wrapper outputs:

```text
logs/vte/
```

Analysis outputs:

```text
logs/vte/<analysis_dir>/
```

Publication-facing or reviewer-facing outputs, when explicitly curated:

```text
docs/results/vte/
```

Generated logs remain local unless explicitly copied into `docs/results/`.

---

## Interpretation levels

Stage 3.2 separates three levels:

1. **Trajectory measurement:** IdPhi-like angular integration, pause duration, reorientation count.
2. **Behavioral regime:** VTE-like vs non-VTE-like trials under a fixed thresholding rule.
3. **Cognitive interpretation:** deliberation, planning, procedural interruption.

Stage 3.2A validates levels 1 and 2 for the wrapper. Stage 3.2C is required before biological-lab comparability claims can be made.

---

## Biological-comparability boundary

Current Stage 3 traces use synthetic pose reconstruction. They are suitable for testing whether the wrapper is stable and whether Stage 3 logs contain VTE-like pause-and-reorient structure.

They are not yet biological tracking traces.

Biological comparison requires:

- a fixed trace schema;
- a geometry registry for choice points, arms, commit zones, and reward zones;
- lab-data adapters that translate tracking data into the same schema;
- a predeclared thresholding rule;
- comparison reports that do not change the measurement core per dataset.

Allowed future changes:

- file-format adapters;
- coordinate transforms;
- maze geometry adapters;
- metadata harmonization.

Disallowed future changes:

- changing IdPhi definition per dataset;
- changing z-score procedure per dataset;
- changing VTE threshold per dataset;
- adding dataset-specific correction terms that improve agreement with one laboratory.

---

## Tests

```bash
python -m pytest vte/tests
python -m pytest stage3/tests
```

The VTE tests cover schema validation, metric computation, wrapper behavior, Stage 3 adapter behavior, and analysis output generation.

---

## Design constraint

The wrapper core is frozen before external biological datasets are inspected. External adapters may translate data into the same schema, but must not change the measurement core.