# Stage 3.1A Baseline Report

## Run metadata

- Stage: 3.1A
- Seeds: 50
- Trials per seed: 100
- Total trials: 5000

## Core summary

- P(covered): 0.6190
- Mean commit latency: 4.4764
- Mean junction pause duration: 5.4764
- Mean reorientation count: 1.7342
- P(commit by bound): 0.8180
- P(commit by timeout): 0.1820
- P(explore at junction): 1.0000

## Seed-level spread

- Mean covered-rate across seeds: 0.6190
- Std covered-rate across seeds: 0.0383
- Best seed: 74 with P(covered)=0.7200
- Worst seed: 89 with P(covered)=0.5500

## Commit reason × path choice

| commit_reason   | path_choice   |   n_trials |
|:----------------|:--------------|-----------:|
| bound           | covered       |       2592 |
| bound           | open          |       1498 |
| timeout         | covered       |        503 |
| timeout         | open          |        407 |

## Path choice × deliberation metrics

| path_choice   |   n_trials |   mean_commit_latency |   median_commit_latency |   mean_junction_pause_duration |   median_junction_pause_duration |   mean_reorientation_count |   p_commit_bound |   p_commit_timeout |
|:--------------|-----------:|----------------------:|------------------------:|-------------------------------:|---------------------------------:|---------------------------:|-----------------:|-------------------:|
| covered       |       3095 |               4.40258 |                       4 |                        5.40258 |                                5 |                    1.66559 |         0.83748  |           0.16252  |
| open          |       1905 |               4.59633 |                       4 |                        5.59633 |                                5 |                    1.84567 |         0.786352 |           0.213648 |

## Block dynamics

| trial_block_label   |   n_trials |   p_covered |   mean_commit_latency |   mean_junction_pause_duration |   mean_reorientation_count |   p_commit_bound |   p_commit_timeout |
|:--------------------|-----------:|------------:|----------------------:|-------------------------------:|---------------------------:|-----------------:|-------------------:|
| 1-25                |       1250 |      0.6264 |                4.4832 |                         5.4832 |                     1.7544 |           0.8184 |             0.1816 |
| 26-50               |       1250 |      0.608  |                4.5024 |                         5.5024 |                     1.748  |           0.8232 |             0.1768 |
| 51-75               |       1250 |      0.6336 |                4.4608 |                         5.4608 |                     1.7224 |           0.8064 |             0.1936 |
| 76-100              |       1250 |      0.608  |                4.4592 |                         5.4592 |                     1.712  |           0.824  |             0.176  |
