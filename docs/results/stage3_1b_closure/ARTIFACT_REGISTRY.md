# Artifact Registry: 3.1B

- Package ID: `stage3_1b_closure_20260429_231006`
- Schema: `stage3_1_closure_artifact_registry_v1`
- Created: `2026-04-30T01:30:57.815625`
- Artifact count: `39`

## Git

- Branch: `stage3_1b_closure`
- Commit: `55852b25e8e09b2319f5eceede6fe22755c25d90`
- Working tree clean: `True`

## Source

- suite_dir: `E:\CRS-1\substrate_cognitive\logs\stage3\stage3_1_closure_raw\stage3_1b\stage3_1b_ablation_suite_20260429_231025`
- analysis_dir: `E:\CRS-1\substrate_cognitive\logs\stage3\stage3_1_closure_raw\stage3_1b\stage3_1b_ablation_suite_20260429_231025\analysis`
- publication_analysis_dir: `E:\CRS-1\substrate_cognitive\logs\stage3\stage3_1_closure_raw\stage3_1b\stage3_1b_ablation_suite_20260429_231025\analysis_publication`
- curated_dir: `E:\CRS-1\substrate_cognitive\docs\results\stage3_1b_closure`
- suite_manifest: `E:\CRS-1\substrate_cognitive\logs\stage3\stage3_1_closure_raw\stage3_1b\stage3_1b_ablation_suite_20260429_231025\manifest.json`
- source_scripts: `['stage3.analysis.run_stage3_1b_ablation_suite', 'stage3.analysis.analyze_stage3_1b_ablation_suite', 'stage3.analysis.analyze_stage3_1b_one_shot_publication', 'stage3.analysis.run_stage3_1_closure_package']`

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
| `figures/Figure_3_1B_OneShot_Ablation_Localization.png` | figure | ablation_suite | ablation_suite | mixed |  | Stage 3.1B ablation-localization artifact |
| `figures/Figure_3_1B_OneShot_Carrier_Effect_By_Window.png` | figure | baseline | stage3_1b_closure | none |  | Curated Stage 3.1 artifact |
| `figures/Figure_3_1B_OneShot_Carryover_Decay_By_Window.png` | figure | baseline | stage3_1b_closure | none |  | Curated Stage 3.1 artifact |
| `figures/Figure_3_1B_OneShot_Effect_By_Ablation.png` | figure | ablation_suite | ablation_suite | mixed |  | Stage 3.1B ablation-localization artifact |
| `figures/Figure_3_1B_OneShot_Placebo_Window_Null.png` | figure | baseline | stage3_1b_closure | none |  | Curated Stage 3.1 artifact |
| `figures/Figure_3_1B_Shock_Risk_Carrier_Around_Event_Zoom.png` | figure | shock | one_shot_shock | negative | h_risk | Stage 3.1B negative one-shot shock artifact |
| `figures/Figure_3_1B_Shock_Target_Choice_Around_Event_Zoom.png` | figure | shock | one_shot_shock | negative | target_choice | Stage 3.1B negative one-shot shock artifact |
| `figures/Figure_3_1B_Treat_Opportunity_Carrier_Around_Event_Zoom.png` | figure | treat | one_shot_treat | positive | h_opp | Stage 3.1B positive one-shot treat artifact |
| `figures/Figure_3_1B_Treat_Target_Choice_Around_Event_Zoom.png` | figure | treat | one_shot_treat | positive | target_choice | Stage 3.1B positive one-shot treat artifact |
| `README.md` | report | baseline | stage3_1b_closure | none |  | Curated Stage 3.1 artifact |
| `reports/STAGE3_1B_CLOSURE_REPORT.md` | report | baseline | stage3_1b_closure | none |  | Curated Stage 3.1 artifact |
| `reports/Stage3_1B_OneShot_Publication_Report.md` | report | baseline | stage3_1b_closure | none |  | Curated Stage 3.1 artifact |
| `stats/analysis_meta.json` | stats | baseline | stage3_1b_closure | none |  | Curated Stage 3.1 artifact |
| `stats/one_shot_acceptance_summary.json` | stats | one_shot | one_shot | mixed |  | Curated Stage 3.1 artifact |
| `stats/one_shot_publication_analysis_meta.json` | stats | one_shot | one_shot | mixed |  | Curated Stage 3.1 artifact |
| `stats/stage3_1b_suite_manifest.json` | stats | baseline | stage3_1b_closure | none |  | Curated Stage 3.1 artifact |
| `tables/Table_3_1B_ablation_seed_metrics.csv` | table | ablation_suite | ablation_suite | mixed | seed | Stage 3.1B ablation-localization artifact |
| `tables/Table_3_1B_ablation_summary_long.csv` | table | ablation_suite | ablation_suite | mixed |  | Stage 3.1B ablation-localization artifact |
| `tables/Table_3_1B_ablation_summary_wide.csv` | table | ablation_suite | ablation_suite | mixed |  | Stage 3.1B ablation-localization artifact |
| `tables/Table_3_1B_one_shot_ablation_localization.csv` | table | one_shot | one_shot | mixed |  | Curated Stage 3.1 artifact |
| `tables/Table_3_1B_one_shot_acceptance_summary.csv` | table | one_shot | one_shot | mixed |  | Curated Stage 3.1 artifact |
| `tables/Table_3_1B_one_shot_carrier_effect_stats.csv` | table | one_shot | one_shot | mixed |  | Curated Stage 3.1 artifact |
| `tables/Table_3_1B_one_shot_effect_stats.csv` | table | one_shot | one_shot | mixed |  | Curated Stage 3.1 artifact |
| `tables/Table_3_1B_one_shot_first_post_junction.csv` | table | one_shot | one_shot | mixed |  | Curated Stage 3.1 artifact |
| `tables/Table_3_1B_one_shot_placebo_window_stats.csv` | table | one_shot | one_shot | mixed |  | Curated Stage 3.1 artifact |
| `tables/Table_3_1B_one_shot_schema_validation.csv` | table | one_shot | one_shot | mixed |  | Curated Stage 3.1 artifact |
| `tables/Table_3_1B_one_shot_trial_series.csv` | table | one_shot | one_shot | mixed |  | Curated Stage 3.1 artifact |
| `tables/Table_3_1B_one_shot_window_seed_metrics.csv` | table | one_shot | one_shot | mixed | seed | Curated Stage 3.1 artifact |
| `tables/Table_3_1B_one_shot_window_summary.csv` | table | one_shot | one_shot | mixed |  | Curated Stage 3.1 artifact |
