# Stage 3.1B One-Shot Publication Report

## Scope

This report summarizes protocol-aware one-shot analysis for Stage 3.1B closure. Shock is treated as the negative branch (`h_risk/q_neg`), while treat is treated as the positive branch (`h_opp/q_pos`).

- Source manifest: `E:\CRS-1\substrate_cognitive\logs\stage3\stage3_1_closure_raw\stage3_1b\stage3_1b_ablation_suite_20260429_003116\manifest.json`

## Schema validation

```
protocol ablation                                                                                                                     run_dir                check   ok missing_columns
   shock     full E:\CRS-1\substrate_cognitive\logs\stage3\stage3_1_closure_raw\stage3_1b\stage3_1b_ablation_suite_20260429_003116\shock\full               trials True                
   shock     full E:\CRS-1\substrate_cognitive\logs\stage3\stage3_1_closure_raw\stage3_1b\stage3_1b_ablation_suite_20260429_003116\shock\full         steps_common True                
   shock     full E:\CRS-1\substrate_cognitive\logs\stage3\stage3_1_closure_raw\stage3_1b\stage3_1b_ablation_suite_20260429_003116\shock\full steps_shock_semantic True                
   treat     full E:\CRS-1\substrate_cognitive\logs\stage3\stage3_1_closure_raw\stage3_1b\stage3_1b_ablation_suite_20260429_003116\treat\full               trials True                
   treat     full E:\CRS-1\substrate_cognitive\logs\stage3\stage3_1_closure_raw\stage3_1b\stage3_1b_ablation_suite_20260429_003116\treat\full         steps_common True                
   treat     full E:\CRS-1\substrate_cognitive\logs\stage3\stage3_1_closure_raw\stage3_1b\stage3_1b_ablation_suite_20260429_003116\treat\full steps_treat_semantic True                
```

## Directional effect statistics

```
protocol ablation  post_window  n_seeds  delta_post_minus_pre_mean  delta_ci_low  delta_ci_high  directional_sign_consistency  directional_sign_flip_p
   shock     full     post_1_3        3                  -0.337165     -0.551724      -0.045977                      1.000000                 0.222222
   shock     full    post_4_10        3                  -0.448276     -0.551724      -0.379310                      1.000000                 0.222222
   shock     full   post_11_30        3                  -0.248276     -0.301724      -0.213793                      1.000000                 0.222222
   shock     full post_31_plus        3                   0.010057     -0.176724       0.170690                      0.333333                 0.666667
   shock     full     post_all        3                  -0.124466     -0.266010      -0.007882                      1.000000                 0.222222
   treat     full     post_1_3        3                   0.114943     -0.252874       0.379310                      0.666667                 0.444444
   treat     full    post_4_10        3                   0.114943     -0.049261       0.270936                      0.666667                 0.333333
   treat     full   post_11_30        3                   0.148276      0.029310       0.301724                      1.000000                 0.222222
   treat     full post_31_plus        3                   0.131609      0.029310       0.301724                      1.000000                 0.222222
   treat     full     post_all        3                   0.133990      0.036453       0.280296                      1.000000                 0.222222
```

## First post-event junction

```
protocol ablation  seed target_path  first_post_trial  target_prob  target_prob_wins  target_choice  target_local_bonus    mode                            gate_trigger
   shock     full    42        open                31     0.466301             False           True            0.000000 explore standard_arbitration (high uncertainty)
   shock     full    43        open                31     0.466301             False          False            0.000000 explore standard_arbitration (high uncertainty)
   shock     full    44        open                31     0.466301             False           True            0.000000 explore standard_arbitration (high uncertainty)
   treat     full    42     covered                31     0.656892              True          False            0.514473 explore standard_arbitration (high uncertainty)
   treat     full    43     covered                31     0.656892              True           True            0.514473 explore standard_arbitration (high uncertainty)
   treat     full    44     covered                31     0.656892              True           True            0.514473 explore standard_arbitration (high uncertainty)
```
