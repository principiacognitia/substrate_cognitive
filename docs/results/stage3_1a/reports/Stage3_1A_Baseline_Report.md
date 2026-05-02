# Stage 3.1A Baseline Report

## Run metadata

- Stage: 3.1A
- Seeds: 50
- Trials per seed: 100
- Total trials: 5000

## Core summary

- P(covered): 0.6670
- Mean commit latency: 3.3086
- Mean junction pause duration: 4.3086
- Mean reorientation count: 1.1454
- P(commit by bound): 0.8854
- P(commit by timeout): 0.1146
- P(explore at junction): 0.9514

## Seed-level spread

- Mean covered-rate across seeds: 0.6670
- Std covered-rate across seeds: 0.0442
- Best seed: 74 with P(covered)=0.8100
- Worst seed: 60 with P(covered)=0.5700

## Commit reason × path choice

| commit_reason   | path_choice   |   n_trials |
|:----------------|:--------------|-----------:|
| bound           | covered       |       2987 |
| bound           | open          |       1440 |
| timeout         | covered       |        348 |
| timeout         | open          |        225 |

## Path choice × deliberation metrics

| path_choice   |   n_trials |   mean_commit_latency |   median_commit_latency |   mean_junction_pause_duration |   median_junction_pause_duration |   mean_reorientation_count |   p_commit_bound |   p_commit_timeout |
|:--------------|-----------:|----------------------:|------------------------:|-------------------------------:|---------------------------------:|---------------------------:|-----------------:|-------------------:|
| covered       |       3335 |               3.25757 |                       2 |                        4.25757 |                                3 |                    1.10045 |         0.895652 |           0.104348 |
| open          |       1665 |               3.41081 |                       2 |                        4.41081 |                                3 |                    1.23544 |         0.864865 |           0.135135 |

## Block dynamics

| trial_block_label   |   n_trials |   p_covered |   mean_commit_latency |   mean_junction_pause_duration |   mean_reorientation_count |   p_commit_bound |   p_commit_timeout |
|:--------------------|-----------:|------------:|----------------------:|-------------------------------:|---------------------------:|-----------------:|-------------------:|
| 1-25                |       1250 |      0.6664 |                3.3352 |                         4.3352 |                     1.156  |           0.8896 |             0.1104 |
| 26-50               |       1250 |      0.6704 |                3.2536 |                         4.2536 |                     1.1192 |           0.888  |             0.112  |
| 51-75               |       1250 |      0.6752 |                3.3544 |                         4.3544 |                     1.1784 |           0.8792 |             0.1208 |
| 76-100              |       1250 |      0.656  |                3.2912 |                         4.2912 |                     1.128  |           0.8848 |             0.1152 |
