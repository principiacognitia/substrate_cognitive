# Stage 3.1A Baseline Report

## Run metadata

- Stage: 3.1A
- Seeds: 50
- Trials per seed: 100
- Total trials: 5000

## Core summary

- P(covered): 0.6806
- Mean commit latency: 3.4136
- Mean junction pause duration: 4.4136
- Mean reorientation count: 1.1832
- P(commit by bound): 0.8826
- P(commit by timeout): 0.1174
- P(explore at junction): 1.0000

## Seed-level spread

- Mean covered-rate across seeds: 0.6806
- Std covered-rate across seeds: 0.0481
- Best seed: 74 with P(covered)=0.8100
- Worst seed: 73 with P(covered)=0.5500

## Commit reason × path choice

| commit_reason   | path_choice   |   n_trials |
|:----------------|:--------------|-----------:|
| bound           | covered       |       3056 |
| bound           | open          |       1357 |
| timeout         | covered       |        347 |
| timeout         | open          |        240 |

## Path choice × deliberation metrics

| path_choice   |   n_trials |   mean_commit_latency |   median_commit_latency |   mean_junction_pause_duration |   median_junction_pause_duration |   mean_reorientation_count |   p_commit_bound |   p_commit_timeout |
|:--------------|-----------:|----------------------:|------------------------:|-------------------------------:|---------------------------------:|---------------------------:|-----------------:|-------------------:|
| covered       |       3403 |               3.37702 |                       2 |                        4.37702 |                                3 |                    1.13929 |         0.898031 |           0.101969 |
| open          |       1597 |               3.49155 |                       2 |                        4.49155 |                                3 |                    1.27677 |         0.849718 |           0.150282 |

## Block dynamics

| trial_block_label   |   n_trials |   p_covered |   mean_commit_latency |   mean_junction_pause_duration |   mean_reorientation_count |   p_commit_bound |   p_commit_timeout |
|:--------------------|-----------:|------------:|----------------------:|-------------------------------:|---------------------------:|-----------------:|-------------------:|
| 1-25                |       1250 |      0.6744 |                3.4224 |                         4.4224 |                     1.2016 |           0.884  |             0.116  |
| 26-50               |       1250 |      0.688  |                3.3888 |                         4.3888 |                     1.1704 |           0.8816 |             0.1184 |
| 51-75               |       1250 |      0.6976 |                3.424  |                         4.424  |                     1.192  |           0.8768 |             0.1232 |
| 76-100              |       1250 |      0.6624 |                3.4192 |                         4.4192 |                     1.1688 |           0.888  |             0.112  |
