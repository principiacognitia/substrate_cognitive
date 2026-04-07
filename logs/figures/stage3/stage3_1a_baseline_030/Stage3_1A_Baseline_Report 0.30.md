# Stage 3.1A Baseline Report

## Run metadata

- Stage: 3.1A
- Seeds: 50
- Trials per seed: 100
- Total trials: 5000

## Core summary

- P(covered): 0.6362
- Mean commit latency: 3.4252
- Mean junction pause duration: 4.4252
- Mean reorientation count: 1.1964
- P(commit by bound): 0.8794
- P(commit by timeout): 0.1206
- P(explore at junction): 1.0000

## Seed-level spread

- Mean covered-rate across seeds: 0.6362
- Std covered-rate across seeds: 0.0492
- Best seed: 74 with P(covered)=0.7700
- Worst seed: 73 with P(covered)=0.5100

## Commit reason × path choice

| commit_reason   | path_choice   |   n_trials |
|:----------------|:--------------|-----------:|
| bound           | covered       |       2820 |
| bound           | open          |       1577 |
| timeout         | covered       |        361 |
| timeout         | open          |        242 |

## Path choice × deliberation metrics

| path_choice   |   n_trials |   mean_commit_latency |   median_commit_latency |   mean_junction_pause_duration |   median_junction_pause_duration |   mean_reorientation_count |   p_commit_bound |   p_commit_timeout |
|:--------------|-----------:|----------------------:|------------------------:|-------------------------------:|---------------------------------:|---------------------------:|-----------------:|-------------------:|
| covered       |       3181 |               3.39202 |                       2 |                        4.39202 |                                3 |                    1.16599 |         0.886514 |           0.113486 |
| open          |       1819 |               3.48323 |                       2 |                        4.48323 |                                3 |                    1.24959 |         0.86696  |           0.13304  |

## Block dynamics

| trial_block_label   |   n_trials |   p_covered |   mean_commit_latency |   mean_junction_pause_duration |   mean_reorientation_count |   p_commit_bound |   p_commit_timeout |
|:--------------------|-----------:|------------:|----------------------:|-------------------------------:|---------------------------:|-----------------:|-------------------:|
| 1-25                |       1250 |      0.6456 |                3.4496 |                         4.4496 |                     1.204  |           0.8808 |             0.1192 |
| 26-50               |       1250 |      0.64   |                3.3856 |                         4.3856 |                     1.1912 |           0.8792 |             0.1208 |
| 51-75               |       1250 |      0.6472 |                3.472  |                         4.472  |                     1.236  |           0.8736 |             0.1264 |
| 76-100              |       1250 |      0.612  |                3.3936 |                         4.3936 |                     1.1544 |           0.884  |             0.116  |
