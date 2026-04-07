# Stage 3.1A Baseline Report

## Run metadata

- Stage: 3.1A
- Seeds: 50
- Trials per seed: 100
- Total trials: 5000

## Core summary

- P(covered): 0.6596
- Mean commit latency: 3.4232
- Mean junction pause duration: 4.4232
- Mean reorientation count: 1.1952
- P(commit by bound): 0.8790
- P(commit by timeout): 0.1210
- P(explore at junction): 1.0000

## Seed-level spread

- Mean covered-rate across seeds: 0.6596
- Std covered-rate across seeds: 0.0504
- Best seed: 74 with P(covered)=0.8100
- Worst seed: 73 with P(covered)=0.5100

## Commit reason × path choice

| commit_reason   | path_choice   |   n_trials |
|:----------------|:--------------|-----------:|
| bound           | covered       |       2933 |
| bound           | open          |       1462 |
| timeout         | covered       |        365 |
| timeout         | open          |        240 |

## Path choice × deliberation metrics

| path_choice   |   n_trials |   mean_commit_latency |   median_commit_latency |   mean_junction_pause_duration |   median_junction_pause_duration |   mean_reorientation_count |   p_commit_bound |   p_commit_timeout |
|:--------------|-----------:|----------------------:|------------------------:|-------------------------------:|---------------------------------:|---------------------------:|-----------------:|-------------------:|
| covered       |       3298 |               3.38569 |                       2 |                        4.38569 |                                3 |                    1.15434 |         0.889327 |           0.110673 |
| open          |       1702 |               3.49589 |                       2 |                        4.49589 |                                3 |                    1.27438 |         0.858989 |           0.141011 |

## Block dynamics

| trial_block_label   |   n_trials |   p_covered |   mean_commit_latency |   mean_junction_pause_duration |   mean_reorientation_count |   p_commit_bound |   p_commit_timeout |
|:--------------------|-----------:|------------:|----------------------:|-------------------------------:|---------------------------:|-----------------:|-------------------:|
| 1-25                |       1250 |      0.66   |                3.4096 |                         4.4096 |                     1.1784 |           0.8904 |             0.1096 |
| 26-50               |       1250 |      0.6728 |                3.4256 |                         4.4256 |                     1.2248 |           0.8688 |             0.1312 |
| 51-75               |       1250 |      0.6672 |                3.4656 |                         4.4656 |                     1.224  |           0.8768 |             0.1232 |
| 76-100              |       1250 |      0.6384 |                3.392  |                         4.392  |                     1.1536 |           0.88   |             0.12   |
