# Stage 3.1A Baseline Report

## Run metadata

- Stage: 3.1A
- Seeds: 3
- Trials per seed: 20
- Total trials: 60

## Core summary

- P(covered): 0.7000
- Mean commit latency: 3.0833
- Mean junction pause duration: 4.0833
- Mean reorientation count: 0.8833
- P(commit by bound): 0.9333
- P(commit by timeout): 0.0667
- P(explore at junction): 0.9833

## Seed-level spread

- Mean covered-rate across seeds: 0.7000
- Std covered-rate across seeds: 0.0816
- Best seed: 44 with P(covered)=0.8000
- Worst seed: 43 with P(covered)=0.6000

## Commit reason × path choice

| commit_reason   | path_choice   |   n_trials |
|:----------------|:--------------|-----------:|
| bound           | covered       |         39 |
| bound           | open          |         17 |
| timeout         | covered       |          3 |
| timeout         | open          |          1 |

## Path choice × deliberation metrics

| path_choice   |   n_trials |   mean_commit_latency |   median_commit_latency |   mean_junction_pause_duration |   median_junction_pause_duration |   mean_reorientation_count |   p_commit_bound |   p_commit_timeout |
|:--------------|-----------:|----------------------:|------------------------:|-------------------------------:|---------------------------------:|---------------------------:|-----------------:|-------------------:|
| covered       |         42 |               2.97619 |                       2 |                        3.97619 |                                3 |                   0.761905 |         0.928571 |          0.0714286 |
| open          |         18 |               3.33333 |                       2 |                        4.33333 |                                3 |                   1.16667  |         0.944444 |          0.0555556 |

## Block dynamics

| trial_block_label   |   n_trials |   p_covered |   mean_commit_latency |   mean_junction_pause_duration |   mean_reorientation_count |   p_commit_bound |   p_commit_timeout |
|:--------------------|-----------:|------------:|----------------------:|-------------------------------:|---------------------------:|-----------------:|-------------------:|
| 1-25                |         60 |         0.7 |               3.08333 |                        4.08333 |                   0.883333 |         0.933333 |          0.0666667 |
