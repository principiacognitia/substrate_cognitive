# Stage 3.1A Baseline Report

## Run metadata

- Stage: 3.1A
- Seeds: 50
- Trials per seed: 100
- Total trials: 5000

## Core summary

- P(covered): 0.7076
- Mean commit latency: 3.4232
- Mean junction pause duration: 4.4232
- Mean reorientation count: 1.1952
- P(commit by bound): 0.8790
- P(commit by timeout): 0.1210
- P(explore at junction): 1.0000

## Seed-level spread

- Mean covered-rate across seeds: 0.7076
- Std covered-rate across seeds: 0.0423
- Best seed: 74 with P(covered)=0.8400
- Worst seed: 73 with P(covered)=0.5800

## Commit reason × path choice

| commit_reason   | path_choice   |   n_trials |
|:----------------|:--------------|-----------:|
| bound           | covered       |       2933 |
| bound           | open          |       1462 |
| timeout         | covered       |        605 |

## Path choice × deliberation metrics

| path_choice   |   n_trials |   mean_commit_latency |   median_commit_latency |   mean_junction_pause_duration |   median_junction_pause_duration |   mean_reorientation_count |   p_commit_bound |   p_commit_timeout |
|:--------------|-----------:|----------------------:|------------------------:|-------------------------------:|---------------------------------:|---------------------------:|-----------------:|-------------------:|
| covered       |       3538 |               3.56303 |                       4 |                        4.56303 |                                5 |                   1.3502   |         0.828999 |           0.171001 |
| open          |       1462 |               3.08482 |                       2 |                        4.08482 |                                3 |                   0.820109 |         1        |           0        |

## Block dynamics

| trial_block_label   |   n_trials |   p_covered |   mean_commit_latency |   mean_junction_pause_duration |   mean_reorientation_count |   p_commit_bound |   p_commit_timeout |
|:--------------------|-----------:|------------:|----------------------:|-------------------------------:|---------------------------:|-----------------:|-------------------:|
| 1-25                |       1250 |      0.7024 |                3.4096 |                         4.4096 |                     1.1784 |           0.8904 |             0.1096 |
| 26-50               |       1250 |      0.7136 |                3.4256 |                         4.4256 |                     1.2248 |           0.8688 |             0.1312 |
| 51-75               |       1250 |      0.7224 |                3.4656 |                         4.4656 |                     1.224  |           0.8768 |             0.1232 |
| 76-100              |       1250 |      0.692  |                3.392  |                         4.392  |                     1.1536 |           0.88   |             0.12   |
