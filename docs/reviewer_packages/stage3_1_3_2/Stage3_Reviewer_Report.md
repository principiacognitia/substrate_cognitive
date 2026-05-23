# Stage 3.1/3.2 Consolidated Reviewer Package

This package combines Stage 3.1A/B closure artifacts with Stage 3.2 seed-level VTE statistics.

This is a compact reviewer-facing transport package.
It does not rerun experiments, does not recompute statistics, and does not modify source artifacts.

## Source

- Preset: `stage3_1_3_2`
- Profile: `llm5`
- Results root: `E:\CRS-1\substrate_cognitive\docs\results`
- Source commit: `not provided`
- Stage 3.1 manifest mode: `full`
- Stage 3.1 manifest run id: `20260502_172750`
- Stage 3.2 patch: `20E`
- Stage 3.2 tests read: `100`

## Included files

- `Stage3_Reviewer_Report.md`
- `Stage3_Key_Tables.md`
- `Figure_Stage3_Reviewer_Page_Main.png`
- `Figure_Stage3_Reviewer_Page_Diagnostics.png`
- `reviewer_package_registry.json`

## Interpretation boundary

This package combines Stage 3.1A/B closure with Stage 3.2 seed-level VTE statistics.
Stage 3.2 statistics are role-aware: wrapper-sanity effects are separated from model-relevant effects and degenerate-ablation diagnostics.
It does not claim rodent-level VTE equivalence, absence inference, allocentric spatial cognition, or self-model-based visibility reasoning.

## Stage 3.2 statistical critique response

This note separates the Patch 20B seed-level statistics into interpretive roles.
Patch 20E does not recompute the Patch 20B tests. It classifies their inferential role and regenerates presentation tables and figures.

## Main distinction

Patch 20B contains two different kinds of statistically significant effects:

1. **Wrapper sanity checks**: expected separation between VTE and non-VTE rows on metrics that define or closely track the VTE measurement itself, such as IdPhi, z-IdPhi, pause, and reorientation.
2. **Model-relevant tests**: behavioral or ablation contrasts that can support interpretation of the model beyond the mechanical definition of the VTE label.

These categories must not be conflated.

## Counts

- Total Patch 20B tests read: 100
- Model-relevant tested rows: 45
- Wrapper-sanity tested rows: 48
- Degenerate-ablation diagnostic rows: 7
- Model-relevant q<0.05 rows: 17
- Wrapper-sanity q<0.05 rows: 48
- Degenerate-ablation q<0.05 rows: 6

## Interpretation

The largest VTE-minus-non-VTE effects on IdPhi-like metrics are expected and should be interpreted as wrapper sanity checks, not as independent validation of the cognitive model.

The model-relevant evidence should instead be read from seed-level behavioral and ablation contrasts, especially contrasts that remain meaningful after circular VTE-definition metrics are separated.

The `novg` ablation is treated separately as a degenerate-ablation diagnostic. Its large effects indicate architectural collapse or extreme regime shift, not a clean localized component effect.

## Files produced by Patch 20E

- `Table_3_2_Model_Relevant_Seed_Level_Tests.csv`
- `Table_3_2_Wrapper_Sanity_Tests.csv`
- `Table_3_2_Degenerate_Ablation_Diagnostics.csv`
- `Table_3_2_Seed_Level_Stats_By_Test_Role.csv`
- `Figure_3_2_Model_Relevant_Seed_Level_Effects.png`
- `Figure_3_2_Degenerate_Ablation_Diagnostics.png`

## Boundary

Patch 20E is a classification and presentation layer over Patch 20B. It should be cited as a response to statistical interpretation concerns, not as a new experiment.

## Role summary

- degenerate_ablation_diagnostic: n=7, q<0.05=6, median |dz|=3.38078
- model_relevant_test: n=45, q<0.05=17, median |dz|=0.149583
- wrapper_sanity_check: n=48, q<0.05=48, median |dz|=15.4531
