# Stage 3.2 Key Tables

Compact Markdown aggregation for LLM reviewers that cannot reliably read CSV/XLSX files.

## Test-role summary

Role-level counts and effect-size summaries.

| test_role | n_tests | n_fdr_lt_05 | n_fdr_lt_10 | min_p_fdr_bh | median_abs_effect_size | max_abs_effect_size |
| --- | --- | --- | --- | --- | --- | --- |
| degenerate_ablation_diagnostic | 7 | 6 | 6 | 8.196311495900614e-05 | 3.380775208826424 | 31.95593627977533 |
| model_relevant_test | 45 | 17 | 18 | 8.196311495900614e-05 | 0.149582802068128 | 0.7069109963398387 |
| wrapper_sanity_check | 48 | 48 | 48 | 8.196311495900614e-05 | 15.45307959718464 | 24.476265264262945 |

Rows shown: 3 of 3.

## Model-relevant seed-level tests

Behavioral or ablation contrasts that can support model-level interpretation.

# Table 3.2 Model-Relevant Seed-Level Tests

| test_role           | test_family                       | contrast         | protocol     | condition   | ablation   | candidate_ablation   | metric                   | group_a   | group_b   |   n_seed_pairs |   mean_delta |   effect_size |     ci95_low |   ci95_high |    p_fdr_bh | direction   | status   |
|:--------------------|:----------------------------------|:-----------------|:-------------|:------------|:-----------|:---------------------|:-------------------------|:----------|:----------|---------------:|-------------:|--------------:|-------------:|------------:|------------:|:------------|:---------|
| model_relevant_test | ablation_vs_reference_within_seed | nox minus full   | stage3_steps | R1_T2       | nan        | nox                  | pause_ticks_mean         | full      | nox       |            150 |  -0.479133   |     -0.706911 | -0.5874      | -0.3698     | 8.19631e-05 | negative    | tested   |
| model_relevant_test | ablation_vs_reference_within_seed | nox minus full   | stage3_steps | R1_T2       | nan        | nox                  | raw_idphi_mean           | full      | nox       |            150 |  -0.318034   |     -0.701249 | -0.394689    | -0.247348   | 8.19631e-05 | negative    | tested   |
| model_relevant_test | ablation_vs_reference_within_seed | nox minus full   | stage3_steps | R1_T2       | nan        | nox                  | reorientation_count_mean | full      | nox       |            150 |  -0.1944     |     -0.693553 | -0.240668    | -0.149995   | 8.19631e-05 | negative    | tested   |
| model_relevant_test | ablation_vs_reference_within_seed | novp minus full  | stage3_steps | R1_T2       | nan        | novp                 | pause_ticks_mean         | full      | novp      |            150 |  -0.167467   |     -0.691536 | -0.206135    | -0.129398   | 8.19631e-05 | negative    | tested   |
| model_relevant_test | ablation_vs_reference_within_seed | novp minus full  | stage3_steps | R1_T2       | nan        | novp                 | raw_idphi_mean           | full      | novp      |            150 |  -0.147236   |     -0.62586  | -0.185563    | -0.11027    | 8.19631e-05 | negative    | tested   |
| model_relevant_test | ablation_vs_reference_within_seed | novp minus full  | stage3_steps | R1_T2       | nan        | novp                 | reorientation_count_mean | full      | novp      |            150 |  -0.058      |     -0.588311 | -0.0738017   | -0.0425333  | 8.19631e-05 | negative    | tested   |
| model_relevant_test | ablation_vs_reference_within_seed | nox minus full   | stage3_steps | R1_T2       | nan        | nox                  | vte_rate                 | full      | nox       |            150 |  -0.0105333  |     -0.3279   | -0.0158683   | -0.00573333 | 8.19631e-05 | negative    | tested   |
| model_relevant_test | vte_binary_within_seed            | vte_minus_nonvte | stage3_steps | R1_T2       | nox        | nan                  | reward                   | nonvte    | vte       |             50 |   0.0840639  |      0.524831 |  0.0390292   |  0.128647   | 0.000403206 | positive    | tested   |
| model_relevant_test | vte_binary_within_seed            | vte_minus_nonvte | stage3_steps | R1_T2       | nox        | nan                  | total_reward             | nonvte    | vte       |             50 |   0.0840639  |      0.524831 |  0.0396067   |  0.128212   | 0.000793611 | positive    | tested   |
| model_relevant_test | vte_binary_within_seed            | vte_minus_nonvte | stage3_steps | R1_T2       | nox        | nan                  | total_reward             | nonvte    | vte       |             50 |  -0.0799481  |     -0.499632 | -0.124944    | -0.0363111  | 0.0010937   | negative    | tested   |
| model_relevant_test | vte_binary_within_seed            | vte_minus_nonvte | stage3_steps | R1_T2       | nox        | nan                  | reward                   | nonvte    | vte       |             50 |  -0.0799481  |     -0.499632 | -0.1239      | -0.0376166  | 0.00115379  | negative    | tested   |
| model_relevant_test | vte_binary_within_seed            | vte_minus_nonvte | stage3_steps | R1_T2       | full       | nan                  | reward                   | nonvte    | vte       |             50 |  -0.0799481  |     -0.499632 | -0.125249    | -0.0364112  | 0.00119397  | negative    | tested   |
| model_relevant_test | vte_binary_within_seed            | vte_minus_nonvte | stage3_steps | R1_T2       | full       | nan                  | total_reward             | nonvte    | vte       |             50 |  -0.0799481  |     -0.499632 | -0.125928    | -0.0369333  | 0.00119397  | negative    | tested   |
| model_relevant_test | vte_binary_within_seed            | vte_minus_nonvte | stage3_steps | R1_T2       | full       | nan                  | reward                   | nonvte    | vte       |             50 |   0.067579   |      0.399129 |  0.0216395   |  0.114132   | 0.00876768  | positive    | tested   |
| model_relevant_test | vte_binary_within_seed            | vte_minus_nonvte | stage3_steps | R1_T2       | novp       | nan                  | reward                   | nonvte    | vte       |             50 |   0.067579   |      0.399129 |  0.0218013   |  0.113614   | 0.00876768  | positive    | tested   |
| model_relevant_test | vte_binary_within_seed            | vte_minus_nonvte | stage3_steps | R1_T2       | novp       | nan                  | total_reward             | nonvte    | vte       |             50 |   0.067579   |      0.399129 |  0.0205414   |  0.112856   | 0.0091424   | positive    | tested   |
| model_relevant_test | vte_binary_within_seed            | vte_minus_nonvte | stage3_steps | R1_T2       | full       | nan                  | total_reward             | nonvte    | vte       |             50 |   0.067579   |      0.399129 |  0.0192351   |  0.113363   | 0.0113375   | positive    | tested   |
| model_relevant_test | ablation_vs_reference_within_seed | novp minus full  | stage3_steps | R1_T2       | nan        | novp                 | vte_rate                 | full      | novp      |            150 |   0.00846667 |      0.164953 |  0.000466667 |  0.0168017  | 0.0602748   | positive    | tested   |
| model_relevant_test | vte_binary_within_seed            | vte_minus_nonvte | stage3_steps | R1_T2       | novp       | nan                  | reward                   | nonvte    | vte       |             50 |  -0.0361709  |     -0.251732 | -0.0774173   |  0.00197017 | 0.108035    | negative    | tested   |
| model_relevant_test | vte_binary_within_seed            | vte_minus_nonvte | stage3_steps | R1_T2       | novp       | nan                  | total_reward             | nonvte    | vte       |             50 |  -0.0361709  |     -0.251732 | -0.075885    |  0.00233889 | 0.108035    | negative    | tested   |
| model_relevant_test | vte_binary_within_seed            | vte_minus_nonvte | stage3_steps | R1_T2       | full       | nan                  | reward                   | nonvte    | vte       |             50 |   0.0249161  |      0.149583 | -0.0202593   |  0.0704546  | 0.350389    | positive    | tested   |
| model_relevant_test | vte_binary_within_seed            | vte_minus_nonvte | stage3_steps | R1_T2       | full       | nan                  | total_reward             | nonvte    | vte       |             50 |   0.0249161  |      0.149583 | -0.0195269   |  0.070735   | 0.350389    | positive    | tested   |
| model_relevant_test | vte_binary_within_seed            | vte_minus_nonvte | stage3_steps | R1_T2       | novp       | nan                  | reward                   | nonvte    | vte       |             50 |   0.0249161  |      0.149583 | -0.020817    |  0.0701627  | 0.350389    | positive    | tested   |
| model_relevant_test | vte_binary_within_seed            | vte_minus_nonvte | stage3_steps | R1_T2       | novp       | nan                  | total_reward             | nonvte    | vte       |             50 |   0.0249161  |      0.149583 | -0.0214852   |  0.0701155  | 0.350389    | positive    | tested   |
| model_relevant_test | vte_binary_within_seed            | vte_minus_nonvte | stage3_steps | R1_T2       | nox        | nan                  | reward                   | nonvte    | vte       |             50 |   0.0249161  |      0.149583 | -0.0201361   |  0.0696876  | 0.350389    | positive    | tested   |

Rows shown: 25 of 45.

## Degenerate ablation diagnostics

Rows separated from clean localized model effects because they indicate architectural collapse or extreme regime shift.

# Table 3.2 Degenerate Ablation Diagnostics

| test_role                      | test_family                       | contrast        | protocol     | condition   |   ablation | candidate_ablation   | metric                   | group_a   | group_b   |   n_seed_pairs |   mean_delta |   effect_size |     ci95_low |    ci95_high |    p_fdr_bh | direction   | status   |
|:-------------------------------|:----------------------------------|:----------------|:-------------|:------------|-----------:|:---------------------|:-------------------------|:----------|:----------|---------------:|-------------:|--------------:|-------------:|-------------:|------------:|:------------|:---------|
| degenerate_ablation_diagnostic | ablation_vs_reference_within_seed | novg minus full | stage3_steps | R1_T2       |        nan | novg                 | pause_ticks_mean         | full      | novg      |            150 |  -3.4748     |    -31.9559   | -3.4916      | -3.45693     | 8.19631e-05 | negative    | tested   |
| degenerate_ablation_diagnostic | ablation_vs_reference_within_seed | novg minus full | stage3_steps | R1_T2       |        nan | novg                 | raw_idphi_mean           | full      | novg      |            150 |  -3.04986    |    -16.9244   | -3.07793     | -3.02148     | 8.19631e-05 | negative    | tested   |
| degenerate_ablation_diagnostic | ablation_vs_reference_within_seed | novg minus full | stage3_steps | R1_T2       |        nan | novg                 | reorientation_count_mean | full      | novg      |            150 |  -1.0476     |    -10.0273   | -1.06387     | -1.0306      | 8.19631e-05 | negative    | tested   |
| degenerate_ablation_diagnostic | ablation_vs_reference_within_seed | novg minus full | stage3_steps | R1_T2       |        nan | novg                 | vte_rate                 | full      | novg      |            150 |  -0.1204     |     -3.38078  | -0.126267    | -0.115       | 8.19631e-05 | negative    | tested   |
| degenerate_ablation_diagnostic | ablation_vs_reference_within_seed | novg minus full | stage3_steps | R1_T2       |        nan | novg                 | reward_mean              | full      | novg      |            150 |  -0.0469533  |     -0.691867 | -0.0576603   | -0.0357463   | 8.19631e-05 | negative    | tested   |
| degenerate_ablation_diagnostic | ablation_vs_reference_within_seed | novg minus full | stage3_steps | R1_T2       |        nan | novg                 | total_reward_mean        | full      | novg      |            150 |  -0.0469533  |     -0.691867 | -0.0575873   | -0.0362063   | 8.19631e-05 | negative    | tested   |
| degenerate_ablation_diagnostic | ablation_vs_reference_within_seed | novg minus full | stage3_steps | R1_T2       |        nan | novg                 | z_idphi_mean             | full      | novg      |            150 |  -2.0639e-17 |     -0.125122 | -4.66936e-17 |  5.76122e-18 | 1           | negative    | tested   |

Rows shown: 7 of 7.

## Wrapper sanity checks

Expected VTE-label separation on IdPhi-like, pause, or reorientation metrics.

# Table 3.2 Wrapper-Sanity Tests

| test_role            | test_family            | contrast         | protocol     | condition   | ablation     |   candidate_ablation | metric              | group_a   | group_b   |   n_seed_pairs |   mean_delta |   effect_size |   ci95_low |   ci95_high |    p_fdr_bh | direction   | status   |
|:---------------------|:-----------------------|:-----------------|:-------------|:------------|:-------------|---------------------:|:--------------------|:----------|:----------|---------------:|-------------:|--------------:|-----------:|------------:|------------:|:------------|:---------|
| wrapper_sanity_check | vte_binary_within_seed | vte_minus_nonvte | stage3_steps | R1_T2       | full         |                  nan | z_idphi             | nonvte    | vte       |             50 |      1.54531 |       24.4763 |    1.52748 |     1.56266 | 8.19631e-05 | positive    | tested   |
| wrapper_sanity_check | vte_binary_within_seed | vte_minus_nonvte | stage3_steps | R1_T2       | novp         |                  nan | z_idphi             | nonvte    | vte       |             50 |      1.54531 |       24.4763 |    1.52765 |     1.56245 | 8.19631e-05 | positive    | tested   |
| wrapper_sanity_check | vte_binary_within_seed | vte_minus_nonvte | stage3_steps | R1_T2       | nox          |                  nan | z_idphi             | nonvte    | vte       |             50 |      1.53796 |       23.4746 |    1.52042 |     1.55674 | 8.19631e-05 | positive    | tested   |
| wrapper_sanity_check | vte_binary_within_seed | vte_minus_nonvte | stage3_steps | R1_T2       | novp         |                  nan | z_idphi             | nonvte    | vte       |             50 |      1.55521 |       22.8667 |    1.53691 |     1.57385 | 8.19631e-05 | positive    | tested   |
| wrapper_sanity_check | vte_binary_within_seed | vte_minus_nonvte | stage3_steps | R1_T2       | full         |                  nan | z_idphi             | nonvte    | vte       |             50 |      1.54785 |       21.5202 |    1.52823 |     1.56714 | 8.19631e-05 | positive    | tested   |
| wrapper_sanity_check | vte_binary_within_seed | vte_minus_nonvte | stage3_steps | R1_T2       | novp         |                  nan | z_idphi             | nonvte    | vte       |             50 |      1.54785 |       21.5202 |    1.52769 |     1.56704 | 8.19631e-05 | positive    | tested   |
| wrapper_sanity_check | vte_binary_within_seed | vte_minus_nonvte | stage3_steps | R1_T2       | nox          |                  nan | z_idphi             | nonvte    | vte       |             50 |      1.54785 |       21.5202 |    1.52861 |     1.56781 | 8.19631e-05 | positive    | tested   |
| wrapper_sanity_check | vte_binary_within_seed | vte_minus_nonvte | stage3_steps | R1_T2       | one_shot_off |                  nan | z_idphi             | nonvte    | vte       |             50 |      1.54785 |       21.5202 |    1.5278  |     1.56727 | 8.19631e-05 | positive    | tested   |
| wrapper_sanity_check | vte_binary_within_seed | vte_minus_nonvte | stage3_steps | R1_T2       | one_shot_off |                  nan | z_idphi             | nonvte    | vte       |             50 |      1.54785 |       21.5202 |    1.52826 |     1.56746 | 8.19631e-05 | positive    | tested   |
| wrapper_sanity_check | vte_binary_within_seed | vte_minus_nonvte | stage3_steps | R1_T2       | one_shot_off |                  nan | z_idphi             | nonvte    | vte       |             50 |      1.54785 |       21.5202 |    1.52813 |     1.56761 | 8.19631e-05 | positive    | tested   |
| wrapper_sanity_check | vte_binary_within_seed | vte_minus_nonvte | stage3_steps | R1_T2       | nox          |                  nan | raw_idphi           | nonvte    | vte       |             50 |      4.12673 |       17.8281 |    4.0664  |     4.1902  | 8.19631e-05 | positive    | tested   |
| wrapper_sanity_check | vte_binary_within_seed | vte_minus_nonvte | stage3_steps | R1_T2       | novp         |                  nan | reorientation_count | nonvte    | vte       |             50 |      2.49924 |       17.8156 |    2.46069 |     2.53832 | 8.19631e-05 | positive    | tested   |
| wrapper_sanity_check | vte_binary_within_seed | vte_minus_nonvte | stage3_steps | R1_T2       | nox          |                  nan | reorientation_count | nonvte    | vte       |             50 |      2.49924 |       17.8156 |    2.46013 |     2.53857 | 8.19631e-05 | positive    | tested   |
| wrapper_sanity_check | vte_binary_within_seed | vte_minus_nonvte | stage3_steps | R1_T2       | novp         |                  nan | raw_idphi           | nonvte    | vte       |             50 |      4.10891 |       17.7582 |    4.04702 |     4.1751  | 8.19631e-05 | positive    | tested   |
| wrapper_sanity_check | vte_binary_within_seed | vte_minus_nonvte | stage3_steps | R1_T2       | full         |                  nan | z_idphi             | nonvte    | vte       |             50 |      1.88322 |       16.3529 |    1.85181 |     1.91584 | 8.19631e-05 | positive    | tested   |
| wrapper_sanity_check | vte_binary_within_seed | vte_minus_nonvte | stage3_steps | R1_T2       | nox          |                  nan | z_idphi             | nonvte    | vte       |             50 |      1.88322 |       16.3529 |    1.85272 |     1.91495 | 8.19631e-05 | positive    | tested   |
| wrapper_sanity_check | vte_binary_within_seed | vte_minus_nonvte | stage3_steps | R1_T2       | full         |                  nan | reorientation_count | nonvte    | vte       |             50 |      2.48986 |       15.4903 |    2.44587 |     2.53278 | 8.19631e-05 | positive    | tested   |
| wrapper_sanity_check | vte_binary_within_seed | vte_minus_nonvte | stage3_steps | R1_T2       | novp         |                  nan | reorientation_count | nonvte    | vte       |             50 |      2.48986 |       15.4903 |    2.44718 |     2.53352 | 8.19631e-05 | positive    | tested   |
| wrapper_sanity_check | vte_binary_within_seed | vte_minus_nonvte | stage3_steps | R1_T2       | nox          |                  nan | reorientation_count | nonvte    | vte       |             50 |      2.48986 |       15.4903 |    2.44404 |     2.53392 | 8.19631e-05 | positive    | tested   |
| wrapper_sanity_check | vte_binary_within_seed | vte_minus_nonvte | stage3_steps | R1_T2       | one_shot_off |                  nan | reorientation_count | nonvte    | vte       |             50 |      2.48986 |       15.4903 |    2.44476 |     2.53235 | 8.19631e-05 | positive    | tested   |
| wrapper_sanity_check | vte_binary_within_seed | vte_minus_nonvte | stage3_steps | R1_T2       | one_shot_off |                  nan | reorientation_count | nonvte    | vte       |             50 |      2.48986 |       15.4903 |    2.44582 |     2.53244 | 8.19631e-05 | positive    | tested   |
| wrapper_sanity_check | vte_binary_within_seed | vte_minus_nonvte | stage3_steps | R1_T2       | one_shot_off |                  nan | reorientation_count | nonvte    | vte       |             50 |      2.48986 |       15.4903 |    2.44472 |     2.53177 | 8.19631e-05 | positive    | tested   |
| wrapper_sanity_check | vte_binary_within_seed | vte_minus_nonvte | stage3_steps | R1_T2       | full         |                  nan | raw_idphi           | nonvte    | vte       |             50 |      4.10006 |       15.4531 |    4.03009 |     4.17178 | 8.19631e-05 | positive    | tested   |
| wrapper_sanity_check | vte_binary_within_seed | vte_minus_nonvte | stage3_steps | R1_T2       | novp         |                  nan | raw_idphi           | nonvte    | vte       |             50 |      4.10006 |       15.4531 |    4.02774 |     4.17046 | 8.19631e-05 | positive    | tested   |
| wrapper_sanity_check | vte_binary_within_seed | vte_minus_nonvte | stage3_steps | R1_T2       | nox          |                  nan | raw_idphi           | nonvte    | vte       |             50 |      4.10006 |       15.4531 |    4.02554 |     4.16937 | 8.19631e-05 | positive    | tested   |

Rows shown: 25 of 48.
