# Artifact Registry: 3.1B

- Package ID: `stage3_1b_closure_20260429_001747`
- Schema: `stage3_1_closure_artifact_registry_v1`
- Created: `2026-04-29T00:18:05.867147`
- Artifact count: `17`

## Git

- Branch: `stage3_1b_closure`
- Commit: `e543c8c6c8b8cc2aceedd5c6e74d57118a3d25ef`
- Working tree clean: `False`

## Source

- suite_dir: `E:\CRS-1\substrate_cognitive\logs\stage3\stage3_1_closure_raw\stage3_1b\stage3_1b_ablation_suite_20260429_001757`
- analysis_dir: `E:\CRS-1\substrate_cognitive\logs\stage3\stage3_1_closure_raw\stage3_1b\stage3_1b_ablation_suite_20260429_001757\analysis`
- curated_dir: `E:\CRS-1\substrate_cognitive\docs\results\stage3_1b_closure`
- suite_manifest: `E:\CRS-1\substrate_cognitive\logs\stage3\stage3_1_closure_raw\stage3_1b\stage3_1b_ablation_suite_20260429_001757\manifest.json`
- source_scripts: `['stage3.analysis.run_stage3_1b_ablation_suite', 'stage3.analysis.analyze_stage3_1b_ablation_suite', 'stage3.analysis.run_stage3_1_closure_package']`

## Artifacts

| File | Type | Condition | Protocol | Branch semantics | Primary variables | Paper role |
|---|---:|---:|---:|---:|---|---|
| `figures/Figure_3_1B_ablation_balanced_commit_latency.png` | figure | balanced | balanced_conflict_matrix | mixed | commit_latency | Stage 3.1B balanced-conflict / matrix artifact |
| `figures/Figure_3_1B_ablation_balanced_p_open.png` | figure | balanced | balanced_conflict_matrix | mixed | p_open | Stage 3.1B balanced-conflict / matrix artifact |
| `figures/Figure_3_1B_ablation_balanced_p_timeout.png` | figure | balanced | balanced_conflict_matrix | mixed | timeout | Stage 3.1B balanced-conflict / matrix artifact |
| `figures/Figure_3_1B_ablation_shock_delta.png` | figure | shock | one_shot_shock | negative | delta_post_minus_pre | Stage 3.1B negative one-shot shock artifact |
| `figures/Figure_3_1B_ablation_shock_post_timeout.png` | figure | shock | one_shot_shock | negative | timeout | Stage 3.1B negative one-shot shock artifact |
| `figures/Figure_3_1B_ablation_treat_delta.png` | figure | treat | one_shot_treat | positive | delta_post_minus_pre | Stage 3.1B positive one-shot treat artifact |
| `figures/Figure_3_1B_ablation_treat_first_target_choice.png` | figure | treat | one_shot_treat | positive | target_choice | Stage 3.1B positive one-shot treat artifact |
| `figures/Figure_3_1B_ablation_treat_first_target_lb.png` | figure | treat | one_shot_treat | positive | local_bonus | Stage 3.1B positive one-shot treat artifact |
| `figures/Figure_3_1B_ablation_treat_first_target_prob.png` | figure | treat | one_shot_treat | positive | target_prob | Stage 3.1B positive one-shot treat artifact |
| `figures/Figure_3_1B_ablation_treat_post_timeout.png` | figure | treat | one_shot_treat | positive | timeout | Stage 3.1B positive one-shot treat artifact |
| `README.md` | report | baseline | stage3_1b_closure | none |  | Curated Stage 3.1 artifact |
| `reports/STAGE3_1B_CLOSURE_REPORT.md` | report | baseline | stage3_1b_closure | none |  | Curated Stage 3.1 artifact |
| `stats/analysis_meta.json` | stats | baseline | stage3_1b_closure | none |  | Curated Stage 3.1 artifact |
| `stats/stage3_1b_suite_manifest.json` | stats | baseline | stage3_1b_closure | none |  | Curated Stage 3.1 artifact |
| `tables/Table_3_1B_ablation_seed_metrics.csv` | table | ablation_suite | ablation_suite | mixed | seed | Stage 3.1B ablation-localization artifact |
| `tables/Table_3_1B_ablation_summary_long.csv` | table | ablation_suite | ablation_suite | mixed |  | Stage 3.1B ablation-localization artifact |
| `tables/Table_3_1B_ablation_summary_wide.csv` | table | ablation_suite | ablation_suite | mixed |  | Stage 3.1B ablation-localization artifact |
