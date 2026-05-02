# Stage 3.1B Closure Results

This folder contains curated publication-facing outputs for Stage 3.1B closure.

Stage 3.1B closes the valence/exposure kernel. It tests tradeoff-sensitive
path choice, persistent one-shot deformation, and ablation-localized carrier
effects.

## Artifact layers

1. `matrix`: 3x3 reward x threat conflict surface.
2. `balanced ablation`: balanced-conflict ablation summary.
3. `one-shot shock`: event-aligned negative one-shot carryover.
4. `one-shot treat`: event-aligned positive one-shot carryover.
5. `diagnostics`: placebo-window, carrier, and ablation-localization checks.

## Source

- Raw suite dir: `E:\CRS-1\substrate_cognitive\logs\stage3\stage3_1_closure_raw\stage3_1b\stage3_1b_ablation_suite_20260502_172810`
- Raw analysis dir: `E:\CRS-1\substrate_cognitive\logs\stage3\stage3_1_closure_raw\stage3_1b\stage3_1b_ablation_suite_20260502_172810\analysis`
- Raw one-shot publication analysis dir: `E:\CRS-1\substrate_cognitive\logs\stage3\stage3_1_closure_raw\stage3_1b\stage3_1b_ablation_suite_20260502_172810\analysis_publication`
- Raw matrix run dir: `E:\CRS-1\substrate_cognitive\logs\stage3\stage3_1_closure_raw\stage3_1b_matrix\grid_full_20260502_173228`
- Raw matrix analysis dir: `E:\CRS-1\substrate_cognitive\logs\stage3\stage3_1_closure_raw\stage3_1b_matrix\grid_full_20260502_173228\analysis_matrix`
- Ablations: all
- Balanced seeds: 50
- Balanced trials per seed: 100
- One-shot seeds: 50
- One-shot trials per seed: 100
- Matrix seeds: 50
- Matrix trials per seed: 100

## Curated folders

- `figures/`: publication and diagnostic figures
- `tables/`: publication and diagnostic tables
- `stats/`: JSON summaries and metadata
- `reports/`: generated reports

## Interpretation boundary

This package supports the Stage 3.1B valence/exposure kernel claim only. It does
not claim absence inference, allocentric spatial cognition, self-model-based
visibility reasoning, or rodent-level VTE equivalence.
