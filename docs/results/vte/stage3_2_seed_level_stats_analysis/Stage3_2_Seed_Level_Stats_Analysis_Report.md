# Patch 20E Seed-Level Statistics Presentation Update

Patch 20E reads Patch 20B outputs through the Patch 20C analyzer path.
It changes presentation only: context columns, figure readability, and production output packaging.
It does not recompute or alter Patch 20B statistical tests.

## Inputs

- Stats directory: `logs\vte\stage3_2_seed_level_stats`

## Outputs

- `docs\results\vte\stage3_2_seed_level_stats_analysis\Table_3_2_Seed_Level_Stats_Top_Findings.csv`
- `docs\results\vte\stage3_2_seed_level_stats_analysis\Table_3_2_Seed_Level_Stats_By_Test_Family.csv`
- `docs\results\vte\stage3_2_seed_level_stats_analysis\Table_3_2_Seed_Level_Stats_By_Metric_Direction.csv`
- `docs\results\vte\stage3_2_seed_level_stats_analysis\Table_3_2_Seed_Level_Stats_By_Test_Role.csv`
- `docs\results\vte\stage3_2_seed_level_stats_analysis\Table_3_2_Model_Relevant_Seed_Level_Tests.csv`
- `docs\results\vte\stage3_2_seed_level_stats_analysis\Table_3_2_Wrapper_Sanity_Tests.csv`
- `docs\results\vte\stage3_2_seed_level_stats_analysis\Table_3_2_Degenerate_Ablation_Diagnostics.csv`
- `docs\results\vte\stage3_2_seed_level_stats_analysis\Table_3_2_Diagnostic_Seed_Level_Tests.csv`
- `docs\results\vte\stage3_2_seed_level_stats_analysis\Stage3_2_Response_To_GLM_Stats_Critique.md`
- `docs\results\vte\stage3_2_seed_level_stats_analysis\Table_3_2_Model_Relevant_Seed_Level_Tests.md`
- `docs\results\vte\stage3_2_seed_level_stats_analysis\Table_3_2_Wrapper_Sanity_Tests.md`
- `docs\results\vte\stage3_2_seed_level_stats_analysis\Table_3_2_Degenerate_Ablation_Diagnostics.md`

## Figures

- `Figure_3_2_Seed_Level_VTE_Delta_Effect_Sizes.png`
- `Figure_3_2_Seed_Level_Ablation_Effect_Sizes.png`
- `Figure_3_2_Seed_Level_VTE_Rate_By_Ablation.png`
- `Figure_3_2_Model_Relevant_Seed_Level_Effects.png`
- `Figure_3_2_Degenerate_Ablation_Diagnostics.png`

## Test-family summary

- ablation_vs_reference_within_seed: n_tests=28, n_tested=28, fdr<0.05=13
- vte_binary_within_seed: n_tests=72, n_tested=72, fdr<0.05=58

## Top findings with context

- ablation_vs_reference_within_seed | novg minus full | protocol=stage3_steps | condition=R1_T2 | reference_ablation=full | candidate_ablation=novg | pause_ticks_mean: effect=-31.9559, p_fdr=8.19631e-05
- vte_binary_within_seed | vte_minus_nonvte | protocol=stage3_steps | condition=R1_T2 | ablation=full | run_id=shock_full_one_shot_f… | z_idphi: effect=24.4763, p_fdr=8.19631e-05
- vte_binary_within_seed | vte_minus_nonvte | protocol=stage3_steps | condition=R1_T2 | ablation=novp | run_id=shock_novp_one_shot_n… | z_idphi: effect=24.4763, p_fdr=8.19631e-05
- vte_binary_within_seed | vte_minus_nonvte | protocol=stage3_steps | condition=R1_T2 | ablation=nox | run_id=shock_nox_one_shot_no… | z_idphi: effect=23.4746, p_fdr=8.19631e-05
- vte_binary_within_seed | vte_minus_nonvte | protocol=stage3_steps | condition=R1_T2 | ablation=novp | run_id=treat_novp_one_shot_n… | z_idphi: effect=22.8667, p_fdr=8.19631e-05
- vte_binary_within_seed | vte_minus_nonvte | protocol=stage3_steps | condition=R1_T2 | ablation=full | run_id=balanced_full_balance… | z_idphi: effect=21.5202, p_fdr=8.19631e-05
- vte_binary_within_seed | vte_minus_nonvte | protocol=stage3_steps | condition=R1_T2 | ablation=novp | run_id=balanced_novp_balance… | z_idphi: effect=21.5202, p_fdr=8.19631e-05
- vte_binary_within_seed | vte_minus_nonvte | protocol=stage3_steps | condition=R1_T2 | ablation=nox | run_id=balanced_nox_balanced… | z_idphi: effect=21.5202, p_fdr=8.19631e-05
- vte_binary_within_seed | vte_minus_nonvte | protocol=stage3_steps | condition=R1_T2 | ablation=one_shot_off | run_id=balanced_one_shot_off… | z_idphi: effect=21.5202, p_fdr=8.19631e-05
- vte_binary_within_seed | vte_minus_nonvte | protocol=stage3_steps | condition=R1_T2 | ablation=one_shot_off | run_id=shock_one_shot_off_on… | z_idphi: effect=21.5202, p_fdr=8.19631e-05
- vte_binary_within_seed | vte_minus_nonvte | protocol=stage3_steps | condition=R1_T2 | ablation=one_shot_off | run_id=treat_one_shot_off_on… | z_idphi: effect=21.5202, p_fdr=8.19631e-05
- vte_binary_within_seed | vte_minus_nonvte | protocol=stage3_steps | condition=R1_T2 | ablation=nox | run_id=shock_nox_one_shot_no… | raw_idphi: effect=17.8281, p_fdr=8.19631e-05

## Interpretation boundary

Patch 20E is a presentation-layer patch. It should not be cited as a new inferential analysis.
Wrapper sanity effects and model-relevant effects remain to be separated in Patch 20E.
