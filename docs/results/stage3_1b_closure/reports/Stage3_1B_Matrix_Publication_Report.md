# Stage 3.1B Matrix Publication Report

## Scope

This report covers the 3x3 reward x threat matrix layer of Stage 3.1B.
It is separate from one-shot shock/treat event-aligned analyses.

## Sources

- Condition summary: `E:\CRS-1\substrate_cognitive\logs\stage3\stage3_1_closure_raw\stage3_1b_matrix\grid_full_20260502_173228\grid_3x3_full_condition_summary.csv`
- Trial table: `E:\CRS-1\substrate_cognitive\logs\stage3\stage3_1_closure_raw\stage3_1b_matrix\grid_full_20260502_173228\grid_3x3_full_all_trials.csv`

## Matrix cell summary

```
condition_id  n_seeds  n_trials_total  p_open  p_covered  mean_junction_pause_duration  mean_commit_latency  mean_reorientation_count  mean_junction_deliberation_proxy  p_commit_bound  p_commit_timeout                                  mode_at_junction_distribution  reward_idx  threat_idx  explore_rate  exploit_rate  exploit_safe_rate  absence_check_rate dominant_mode
       R0_T1       50            5000  0.3182     0.6818                        4.1816               3.1816                    1.0604                          4.058910          1.0000            0.0000                                               {'explore': 1.0}           0           1        1.0000        0.0000             0.0000                 0.0       explore
       R1_T1       50            5000  0.4820     0.5180                        5.4836               4.4836                    1.7324                          5.260536          0.8196            0.1804                                               {'explore': 1.0}           1           1        1.0000        0.0000             0.0000                 0.0       explore
       R2_T1       50            5000  0.6796     0.3204                        4.1758               3.1758                    1.0616                          4.057185          1.0000            0.0000                                               {'explore': 1.0}           2           1        1.0000        0.0000             0.0000                 0.0       explore
       R0_T2       50            5000  0.2324     0.7676                        2.8724               1.8724                    0.4130                          2.758593          1.0000            0.0000                         {'explore': 0.5934, 'exploit': 0.4066}           0           2        0.5934        0.4066             0.0000                 0.0       explore
       R1_T2       50            5000  0.4422     0.5578                        5.4748               4.4748                    1.7276                          5.253475          0.8198            0.1802                                               {'explore': 1.0}           1           2        1.0000        0.0000             0.0000                 0.0       explore
       R2_T2       50            5000  0.6486     0.3514                        4.4732               3.4732                    1.2220                          4.312318          0.8828            0.1172                                               {'explore': 1.0}           2           2        1.0000        0.0000             0.0000                 0.0       explore
       R0_T3       50            5000  0.0080     0.9920                        2.0000               1.0000                    0.0000                          1.791759          1.0000            0.0000 {'exploit_safe': 0.9458, 'exploit': 0.0406, 'explore': 0.0136}           0           3        0.0136        0.0406             0.9458                 0.0  exploit_safe
       R1_T3       50            5000  0.0080     0.9920                        2.0542               1.0542                    0.0266                          1.848633          0.9982            0.0018                    {'exploit_safe': 0.9726, 'explore': 0.0274}           1           3        0.0274        0.0000             0.9726                 0.0  exploit_safe
       R2_T3       50            5000  0.0132     0.9868                        2.0630               1.0630                    0.0318                          1.855981          0.9978            0.0022                        {'exploit_safe': 0.98, 'explore': 0.02}           2           3        0.0200        0.0000             0.9800                 0.0  exploit_safe
```

## Acceptance checks

```json
{
  "reward_monotonicity_per_threat": {
    "T1": true,
    "T2": true,
    "T3": true
  },
  "threat_monotonicity_per_reward": {
    "R0": true,
    "R1": true,
    "R2": true
  },
  "reward_axis_order_ok": true,
  "threat_axis_order_ok": true,
  "center_pause_peak": false,
  "center_latency_peak": false,
  "center_reorientation_peak": false,
  "center_proxy_peak": false,
  "balanced_conflict_peak_ok": false,
  "n_matrix_cells": 9,
  "matrix_complete_3x3": true
}
```

## Interpretation boundary

Matrix heatmaps describe the conflict surface across reward and threat levels. They do not by themselves demonstrate one-shot carryover; that claim belongs to the one-shot publication tables and figures.