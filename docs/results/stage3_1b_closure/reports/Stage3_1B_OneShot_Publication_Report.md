# Stage 3.1B One-Shot Publication Report

## Scope

This report summarizes protocol-aware one-shot analysis for Stage 3.1B closure. Shock is treated as the negative branch (`h_risk/q_neg`), while treat is treated as the positive branch (`h_opp/q_pos`).

- Source manifest: `E:\CRS-1\substrate_cognitive\logs\stage3\stage3_1_closure_raw\stage3_1b\stage3_1b_ablation_suite_20260429_030855\manifest.json`

## Schema validation

```
protocol ablation                                                                                                                     run_dir                check   ok missing_columns
   shock     full E:\CRS-1\substrate_cognitive\logs\stage3\stage3_1_closure_raw\stage3_1b\stage3_1b_ablation_suite_20260429_030855\shock\full               trials True                
   shock     full E:\CRS-1\substrate_cognitive\logs\stage3\stage3_1_closure_raw\stage3_1b\stage3_1b_ablation_suite_20260429_030855\shock\full         steps_common True                
   shock     full E:\CRS-1\substrate_cognitive\logs\stage3\stage3_1_closure_raw\stage3_1b\stage3_1b_ablation_suite_20260429_030855\shock\full steps_shock_semantic True                
   treat     full E:\CRS-1\substrate_cognitive\logs\stage3\stage3_1_closure_raw\stage3_1b\stage3_1b_ablation_suite_20260429_030855\treat\full               trials True                
   treat     full E:\CRS-1\substrate_cognitive\logs\stage3\stage3_1_closure_raw\stage3_1b\stage3_1b_ablation_suite_20260429_030855\treat\full         steps_common True                
   treat     full E:\CRS-1\substrate_cognitive\logs\stage3\stage3_1_closure_raw\stage3_1b\stage3_1b_ablation_suite_20260429_030855\treat\full steps_treat_semantic True                
```

## Compact acceptance summary

```
protocol ablation                          check       status     value                                                                                            threshold_or_expectation                                                                   note
   shock     full        schema_required_columns         pass  1.000000                                                                      all required protocol-specific columns present                                                                       
   treat     full        schema_required_columns         pass  1.000000                                                                      all required protocol-specific columns present                                                                       
   shock     full      p_target_delta_post_11_30 direction_ok -0.248276 delta < 0 for shock/negative; delta > 0 for treat/positive; pass requires directional p<=0.05 and CI excluding zero ci=[-0.3017, -0.2138], directional_p=0.222222, sign_consistency=1.0000
   shock     full    p_target_delta_post_31_plus         fail  0.010057 delta < 0 for shock/negative; delta > 0 for treat/positive; pass requires directional p<=0.05 and CI excluding zero  ci=[-0.1767, 0.1707], directional_p=0.666667, sign_consistency=0.3333
   shock     full        p_target_delta_post_all direction_ok -0.124466 delta < 0 for shock/negative; delta > 0 for treat/positive; pass requires directional p<=0.05 and CI excluding zero ci=[-0.2660, -0.0079], directional_p=0.222222, sign_consistency=1.0000
   treat     full      p_target_delta_post_11_30 direction_ok  0.148276 delta < 0 for shock/negative; delta > 0 for treat/positive; pass requires directional p<=0.05 and CI excluding zero   ci=[0.0293, 0.3017], directional_p=0.222222, sign_consistency=1.0000
   treat     full    p_target_delta_post_31_plus direction_ok  0.131609 delta < 0 for shock/negative; delta > 0 for treat/positive; pass requires directional p<=0.05 and CI excluding zero   ci=[0.0293, 0.3017], directional_p=0.222222, sign_consistency=1.0000
   treat     full        p_target_delta_post_all direction_ok  0.133990 delta < 0 for shock/negative; delta > 0 for treat/positive; pass requires directional p<=0.05 and CI excluding zero   ci=[0.0365, 0.2803], directional_p=0.222222, sign_consistency=1.0000
   shock     full  first_post_target_probability direction_ok  0.466301                                  shock/negative expects target_prob < 0.5; treat/positive expects target_prob > 0.5                target_choice_rate=0.6667, target_prob_wins_rate=0.0000
   treat     full  first_post_target_probability direction_ok  0.656892                                  shock/negative expects target_prob < 0.5; treat/positive expects target_prob > 0.5                target_choice_rate=0.6667, target_prob_wins_rate=1.0000
   shock     full placebo_window_null_post_11_30         pass  0.045791       directional placebo p<=0.05 indicates real event-aligned effect is stronger than random fake event boundaries   observed_delta=-0.2483, null_mean=-0.0165, null_ci=[-0.2804, 0.1794]
   shock     full   placebo_window_null_post_all         pass  0.003599       directional placebo p<=0.05 indicates real event-aligned effect is stronger than random fake event boundaries    observed_delta=-0.1245, null_mean=0.0617, null_ci=[-0.0778, 0.1653]
   treat     full placebo_window_null_post_11_30   diagnostic  0.135973       directional placebo p<=0.05 indicates real event-aligned effect is stronger than random fake event boundaries     observed_delta=0.1483, null_mean=0.0790, null_ci=[-0.0464, 0.1958]
   treat     full   placebo_window_null_post_all   diagnostic  0.245751       directional placebo p<=0.05 indicates real event-aligned effect is stronger than random fake event boundaries      observed_delta=0.1340, null_mean=0.0993, null_ci=[0.0093, 0.2079]
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
