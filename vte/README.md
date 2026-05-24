# Stage 3.2: VTE Measurement and Seed-Level Statistics Layer

This directory contains the Stage 3.2 VTE-style measurement layer for `substrate_cognitive`.

`vte/` is intentionally separated from `stage3/`. It is not part of the toy cognitive model and does not participate in action selection. It is a read-only measurement, adapter, analysis, and reporting layer over externalized behavioral traces.

Stage 3.2 is now closed as a measurement/statistical layer.

---

## Status

| Layer | Status | Meaning |
| :--- | :--- | :--- |
| **Stage 3.2A** | ✅ Complete | VTE wrapper core: fixed trace schema, IdPhi-like metrics, pause and reorientation metrics. |
| **Stage 3.2B** | ✅ Complete | Stage 3 step-log adapter, batch processing, reports, figures, reviewer-facing artifacts. |
| **Stage 3.2C** | ✅ Complete as decision-level biological comparability | Biological/lab adapters and comparability notes are constrained to decision-level and schema-level comparison. |
| **Patch 20B-20E** | ✅ Complete | Seed-level statistics, FDR correction, model-relevant vs wrapper-sanity separation, degenerate-ablation diagnostics. |
| **Patch 21 visualization** | ⬜ Deferred / optional | Visualization is useful for article figures but is not required for Stage 3.2 closure. |

---

## Boundary

The VTE layer must not import model internals from `stage3/`.

Allowed input:

- CSV or JSONL behavioral traces;
- trial-level metadata;
- geometry metadata supplied as data files;
- biological/lab data translated through an adapter into the same schema.

Disallowed input:

- internal Gate state;
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

This label is part of the interpretation boundary. It means the trace is reconstructed from Stage 3 step logs, not obtained from biological tracking data.

---

## Output contract

The wrapper produces trial-level VTE-style metrics:

- `raw_idphi`
- `log_idphi`
- `z_idphi`
- `pause_ticks`
- `reorientation_count`
- `choice_point_duration`
- `vte_binary`

The wrapper may classify trials as VTE-like under a fixed thresholding rule, but cognitive interpretation remains separate from measurement.

---

## Canonical Stage 3.2 outputs

Production-facing Stage 3.2 statistical outputs are stored in:

```text
docs/results/vte/stage3_2_seed_level_stats_analysis/
```

Main files:

```text
Stage3_2_Response_To_GLM_Stats_Critique.md
Stage3_2_Seed_Level_Stats_Analysis_Report.md
stage3_2_seed_level_stats_analysis_meta.json

Table_3_2_Seed_Level_Stats_By_Test_Role.csv
Table_3_2_Model_Relevant_Seed_Level_Tests.csv
Table_3_2_Wrapper_Sanity_Tests.csv
Table_3_2_Degenerate_Ablation_Diagnostics.csv

Table_3_2_Model_Relevant_Seed_Level_Tests.md
Table_3_2_Wrapper_Sanity_Tests.md
Table_3_2_Degenerate_Ablation_Diagnostics.md

Figure_3_2_Model_Relevant_Seed_Level_Effects.png
Figure_3_2_Degenerate_Ablation_Diagnostics.png
Figure_3_2_Seed_Level_VTE_Rate_By_Ablation.png
Figure_3_2_Seed_Level_Ablation_Effect_Sizes.png
Figure_3_2_Seed_Level_VTE_Delta_Effect_Sizes.png
```

---

## Statistical interpretation

Patch 20B performs seed-level statistical tests.

Patch 20D/20E are presentation and classification layers over Patch 20B. They do not recompute the underlying tests.

Patch 20E separates tests into:

1. **model_relevant_test**  
   Behavioral or ablation contrasts that can support model-level interpretation.

2. **wrapper_sanity_check**  
   Expected VTE-label separation on metrics used by or adjacent to the VTE measurement definition, such as IdPhi, pause, and reorientation.

3. **degenerate_ablation_diagnostic**  
   Extreme or collapsing ablation behavior, especially `novg`, which should not be treated as a clean localized effect.

4. **diagnostic**  
   Reserved for rows not classified by the above categories. In the current closed Stage 3.2 package, this category is expected to be empty.

---

## Biological-comparability boundary

Stage 3.2 includes biological/lab adapter work only under a restricted interpretation.

Allowed claims:

- the same schema can represent synthetic Stage 3 decision traces and selected biological decision-level records;
- decision-level biological comparators can be used to test whether the measurement vocabulary is plausible;
- biological adapters can support future work by fixing data contracts and geometry metadata.

Disallowed claims:

- rodent-level VTE equivalence;
- biological trajectory replay equivalence;
- neural mechanism identity;
- allocentric spatial cognition;
- full W-maze or RROW task equivalence;
- dataset-specific retuning of IdPhi or VTE thresholds to improve agreement.

Decision-level comparability is not movement-level comparability.

---

## Workflow

Translate Stage 3 step logs to VTE trace CSV:

```bash
python -m vte.analysis.translate_stage3_steps_to_vte_trace \
  --input-csv logs/stage3/stage3_1_closure_raw/stage3_1b/<suite>/balanced/full/balanced_conflict_full_all_steps.csv \
  --output-csv logs/vte/raw/balanced_conflict_full_trace.csv \
  --run-id balanced_conflict_full
```

Run the wrapper:

```bash
python -m vte.analysis.run_stage3_2_vte \
  --input-csv logs/vte/raw/balanced_conflict_full_trace.csv \
  --output-dir logs/vte/stage3_2_smoke
```

Analyze VTE metrics:

```bash
python -m vte.analysis.analyze_stage3_2_vte \
  --metrics-csv logs/vte/stage3_2_smoke/vte_trial_metrics.csv \
  --output-dir logs/vte/stage3_2_smoke_analysis
```

Run seed-level statistics:

```bash
python -m vte.analysis.run_stage3_2_seed_level_stats \
  --input-dir logs/vte/stage3_2_batch \
  --output-dir logs/vte/stage3_2_seed_level_stats
```

Analyze and package seed-level statistics:

```bash
python -m vte.analysis.analyze_stage3_2_seed_level_stats \
  --stats-dir logs/vte/stage3_2_seed_level_stats \
  --output-dir docs/results/vte/stage3_2_seed_level_stats_analysis \
  --top-n 25
```

Build reviewer package:

```bash
python -m stage3.analysis.build_stage3_reviewer_package \
  --preset stage3_1_3_2 \
  --profile llm5 \
  --results-root docs/results \
  --output-dir docs/reviewer_packages/stage3_1_3_2 \
  --clean
```

---

## Tests

```bash
python -m pytest vte/tests
python -m pytest stage3/tests
```

The VTE tests cover schema validation, metric computation, wrapper behavior, Stage 3 adapter behavior, biological adapter contracts, seed-level statistics, analysis output generation, and reviewer package integration.

---

## Deferred work

Deferred Stage 3.2 work is tracked in:

```text
docs/stage3_2_TBD.md
```

Deferred items include VTE-like trail visualization, animation, side-by-side visual comparison, W-maze configuration, maze-builder utilities, and possible neural-data comparison of Gate dynamics at choice points.

---

## Closure

Stage 3.2 is closed as a read-only measurement and statistical analysis layer.

The next recommended project step is article writing, using:

```text
docs/STAGE3_2_CLOSURE.md
docs/article_handoff/Stage3_Followup_Article_Outline.md
```