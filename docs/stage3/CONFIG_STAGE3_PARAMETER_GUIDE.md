# Stage 3 Configuration Parameter Guide

Patch: 20A  
Scope: documentation only  
Target modules: `stage3/configs/config_stage3_1a.py`, `stage3/configs/config_stage3_1b.py`  
Status: external-review hardening document

## 1. Purpose

This document explains the Stage 3.1A and Stage 3.1B configuration files as experimental protocols.

The configs should be read as a reproducibility contract, not only as Python parameter files.

- Stage 3.1A defines the frozen open/covered baseline.
- Stage 3.1B extends that baseline with a reward x threat matrix and one-shot protocols.
- Stage 3.1B keeps the core architecture fixed and varies task conditions and ablations.

## 2. Stage 3.1A: open/covered baseline

Stage 3.1A is the baseline spatial task. It contains no explicit reward/threat conflict. Open and covered paths have equal nominal reward and equal length, but different exposure profiles.

The goal is to establish a stable covered-path bias and deliberation/VTE proxy logging under a frozen agent configuration.

### 2.1 Environment parameters

| Parameter | Value | Role |
|---|---:|---|
| `maze_type` | `open_covered_choice` | Spatial task family |
| `n_trials` | 100 | Trials per session |
| `n_sessions` | 30 | Sessions / seeds in default config |
| `seed_range` | `(42, 71)` | Default seed interval |
| `node_ids` | `start`, `junction`, `open_mid`, `covered_mid`, `goal` | Minimal five-node topology |
| `start_node` | `start` | Trial start |
| `junction_node` | `junction` | Choice point |
| `goal_node` | `goal` | Goal node |

### 2.2 Path parameters

| Path | Length | Base reward | Reward probability | `X_risk` | `X_opp` | `D_est` | Meaning |
|---|---:|---:|---:|---:|---:|---:|---|
| open | 3 | 1.0 | 0.70 | 0.70 | 0.50 | 0.90 | Higher exposure / visibility path |
| covered | 3 | 1.0 | 0.70 | 0.20 | 0.50 | 0.30 | Lower exposure / visibility path |

The reward distribution is Bernoulli. Equal reward and equal path length isolate exposure sensitivity.

### 2.3 Zone parameters

| Zone | `X_risk` | `X_opp` | `D_est` | `nu` | `stakes` | Notes |
|---|---:|---:|---:|---:|---:|---|
| start | 0.10 | 0.00 | 0.50 | 0.0 | 1.0 | Low-risk starting zone |
| junction | 0.30 | 0.30 | 0.80 | 0.0 | 1.0 | Choice point; `is_choice_point=True` |
| goal | 0.10 | 1.00 | 0.50 | 1.0 | 1.0 | Opportunity/reward zone |

### 2.4 Deliberation parameters

| Parameter | Stage 3.1A value | Role |
|---|---:|---|
| `max_deliberation_ticks` | 6 | Maximum deliberation ticks in the environment-level junction process |
| `junction_q_values` | `[0.5, 0.5]` | Initial symmetric option values at junction |
| `exposure_q_bias` | 0.35 | Frozen exposure bias calibrated in Stage 3.1A |
| `eps` | 0.05 | Small stochasticity/noise term |
| `evidence_bound_base` | 0.35 | Base evidence threshold for commit |
| `evidence_bound_min` | 0.20 | Lower bound on urgency-reduced threshold |
| `urgency_slope` | 0.04 | Evidence-bound reduction with time |
| `min_evidence_step` | 0.10 | Minimum evidence accumulation increment |

`exposure_q_bias = 0.35` and `eps = 0.05` are treated as frozen baseline values for Stage 3.1B.

## 3. Shared Stage 3 agent parameters

### 3.1 Stage 2 legacy block

| Parameter | Value | Role |
|---|---:|---|
| `alpha` | 0.35 | Learning-rate compatibility parameter |
| `beta` | 4.0 | Legacy inverse temperature fallback |
| `k_use` | 0.08 | Legacy action-viscosity hardening |
| `k_melt` | 0.20 | Legacy action-viscosity melting |
| `lambda_decay` | 0.01 | Legacy decay parameter |
| `tau_vol` | 0.50 | Legacy volatility threshold |

These values are retained for compatibility and for Stage 2-related comparison. Stage 3.1B-specific action selection uses `action_policy`.

### 3.2 Action policy

| Parameter | Stage 3.1A | Stage 3.1B | Role |
|---|---:|---:|---|
| `beta_exploit` | 4.0 | 4.0 | Softmax inverse temperature for `EXPLOIT` |
| `beta_explore` | 1.0 | 1.0 | Softmax inverse temperature for `EXPLORE` and `ABSENCE_CHECK` |
| `beta_safe` | 5.0 | 5.0 | Softmax inverse temperature for `EXPLOIT_SAFE` |
| `lambda_risk` | 2.0 | 2.0 | Risk penalty weight in `EXPLOIT_SAFE` |
| `epsilon_explore` | 0.0 | 0.0 | Optional random exploration; disabled by default |
| `commit_confidence` | 0.70 | 0.70 | Policy-level confidence threshold |
| `max_deliberation_ticks` | 10 | 10 | Agent-level deliberation cap |
| `local_opp_bonus_weight` | not used | 1.0 | Source-local appetitive carryover weight in post-Gate valuation |

`local_opp_bonus_weight` is policy-side. It does not enter Gate routing.

### 3.3 Gate thresholds

| Parameter | Stage 3.1A | Stage 3.1B | Role |
|---|---:|---:|---|
| `critical_risk_threshold` | 0.70 | 0.37 | EXPLOIT_SAFE threshold |
| `suspicion_threshold` | 0.50 | 0.50 | ABSENCE_CHECK h_risk threshold |
| `visibility_threshold` | 0.30 | 0.30 | ABSENCE_CHECK visibility threshold |
| `safe_window_threshold` | 50 | 50 | ABSENCE_CHECK time-window threshold |
| `theta_mb` | 0.30 | 0.30 | EXPLORE gate-output threshold |
| `theta_u` | 1.50 | 1.50 | uncertainty baseline |
| `safe_drive_weight_current` | default | 0.60 | weight on current `X_risk` in safe drive |
| `safe_drive_weight_temporal` | default | 0.40 | weight on accumulated `h_risk` in safe drive |
| `w_volatility` | default | 1.0 | uncertainty weight on volatility |
| `w_entropy` | default | 1.0 | uncertainty weight on entropy |
| `v_g_weight_hrisk` | default | 0.70 | accumulated-risk component of gate-viscosity approximation |
| `v_g_weight_xrisk` | default | 0.30 | current-risk component of gate-viscosity approximation |

Stage 3.1B lowers `critical_risk_threshold` and explicitly weights current and accumulated risk because it is the conflict and one-shot phase.

### 3.4 Temporal state

| Parameter | Stage 3.1A | Stage 3.1B | Role |
|---|---:|---:|---|
| `lambda_risk` | 0.10 | 0.10 | base update rate for `h_risk` |
| `lambda_opp` | 0.10 | 0.10 | base update rate for `h_opp` |
| `lambda_input_risk` | 0.10 | 0.10 | retained config field for risk input gain |
| `lambda_input_opp` | 0.10 | 0.10 | retained config field for opportunity input gain |
| `rho_neg` | 0.98 | 0.98 | retention of `q_neg` |
| `rho_pos` | 0.95 | 0.95 | retention of `q_pos` |
| `k_neg` | 1.0 | 1.0 | gain for negative importance input |
| `k_pos` | 0.7 | 0.7 | gain for positive importance input |
| `theta_baseline` | 0.25 | 0.25 | baseline field threshold for importance input |
| `theta_shot` | 5.0 | 5.0 | one-shot classification threshold |
| `w_neg_to_risk` | 2.0 | 2.0 | coupling from `q_neg` to risk trace time constant |
| `w_pos_to_opp` | 1.0 | 1.0 | coupling from `q_pos` to opportunity trace time constant |
| `w_qneg_input` | 1.0 | 1.0 | coupling from `q_neg` to effective risk input |
| `w_qpos_input` | 0.0 | 0.0 | positive input-gain coupling; disabled in current closure config |
| `local_opp_immediate_seed_weight` | 0.75 | 0.75 | immediate source-local positive seed strength |
| `salience_threshold` | 0.50 | 0.50 | h_time reset threshold |
| `q_clip` | 10.0 | 10.0 | safety clip for q traces |

Important semantic note: `lambda_risk` and `lambda_opp` are update rates, not retention coefficients.

## 4. Stage 3.1B: reward x threat conflict

Stage 3.1B extends Stage 3.1A by introducing unequal reward and unequal threat/exposure while preserving the frozen baseline calibration.

It tests whether the same architecture can produce tradeoff-sensitive path choice, one-shot persistence, and interpretable ablation localization.

### 4.1 Fixed covered path

| Parameter | Value |
|---|---:|
| length | 3 |
| base_reward | 1.0 |
| reward_prob | 0.70 |
| reward_bonus | 0.0 |
| threat_penalty | 0.0 |
| threat_prob | 0.0 |
| `X_risk` | 0.20 |
| `X_opp` | 0.50 |
| `D_est` | 0.30 |

The covered path remains the low-risk baseline.

### 4.2 Open reward levels

| Level | `reward_prob` | `reward_bonus` | Meaning |
|---|---:|---:|---|
| R0 | 0.70 | 0.0 | no reward premium |
| R1 | 0.80 | 0.2 | moderate reward premium |
| R2 | 0.90 | 0.4 | high reward premium |

### 4.3 Open threat levels

| Level | `X_risk` | `threat_penalty` | `threat_prob` | Meaning |
|---|---:|---:|---:|---|
| T1 | 0.45 | 0.0 | 0.0 | mild threat |
| T2 | 0.60 | 0.5 | 0.3 | balanced threat |
| T3 | 0.75 | 1.0 | 0.5 | strong threat |

### 4.4 Full condition grid

The grid is generated by all reward x threat combinations:

```text
R0_T1  R1_T1  R2_T1
R0_T2  R1_T2  R2_T2
R0_T3  R1_T3  R2_T3
```

### 4.5 Canonical conditions

| Name | Condition ID | Description |
|---|---|---|
| `reward_dominant` | `R2_T1` | high open reward premium + mild threat |
| `balanced_conflict` | `R1_T2` | moderate open reward premium + balanced threat |
| `threat_dominant` | `R0_T3` | no open reward premium + strong threat |

These canonical conditions are used for smoke checks, ablations, and acceptance protocol.

### 4.6 Conflict variables

`compute_conflict_variables()` derives:

| Variable | Computation | Meaning |
|---|---|---|
| `reward_gap` | expected open reward - expected covered reward | reward advantage of open path |
| `risk_gap` | open `X_risk` - covered `X_risk` | exposure-risk disadvantage of open path |
| `threat_gap` | expected open threat cost - expected covered threat cost | threat-cost disadvantage of open path |
| `preferred_by_reward` | open if `reward_gap > 0`, else covered | reward-preferred path |
| `preferred_by_threat` | covered if `risk_gap > 0`, else open | threat-preferred path |

Canonical condition labels are assigned by condition id, not by heuristic thresholds.

## 5. One-shot protocol

One-shot is off by default in normal grid runs. It is enabled only by a separate one-shot runner mode.

Default `get_one_shot_protocol()` values:

| Field | Default |
|---|---|
| `shock_trial` | 30 |
| `condition_name` | `balanced_conflict` |
| `condition_id` | `R1_T2` |
| `path` | `open` |
| `salience` | 0.9 |
| `stakes` | 10.0 |
| `total_trials` | 100 |
| `pre_block_trials` | 30 |
| `post_block_trials` | 69 |

| Kind | Default reward | Source override mode | Interpretation |
|---|---:|---|---|
| `shock` | -5.0 | `path_negative` | aversive negative event |
| `treat` | +5.0 | `positive_reward` | positive jackpot/appetitive event |

One-shot persistence is implemented through `TemporalState`, not through a separate trauma or reinforcement module.

## 6. Logging contracts

### 6.1 Shared trial-level VTE proxies

| Field | Meaning |
|---|---|
| `junction_pause_duration` | ticks at junction before choice |
| `reorientation_count` | candidate path switches at junction |
| `retreat_return_count` | return-to-start events after junction |
| `commit_latency` | ticks to final commit |
| `junction_deliberation_proxy` | composite plotting proxy; not primary statistical metric |

### 6.2 Stage 3.1B extensions

| Field | Meaning |
|---|---|
| `condition_id` | matrix cell ID |
| `reward_level` | R0/R1/R2 |
| `threat_level` | T1/T2/T3 |
| `reward_gap` | derived expected reward difference |
| `risk_gap` | derived exposure-risk difference |
| `threat_gap` | derived threat-cost difference |
| `conflict_condition` | canonical or mixed condition label |
| `one_shot_active` | whether one-shot protocol is active |
| `one_shot_trial` | one-shot event trial |
| `path_choice_preferred_by_reward` | reward-preferred path |
| `path_choice_preferred_by_threat` | threat-preferred path |

## 7. Statistical and acceptance configuration

The config contains two conceptually distinct layers.

### 7.1 Acceptance criteria

Acceptance criteria are engineering/model-readiness checks. Examples:

| Criterion | Meaning |
|---|---|
| `16.1_path_tradeoff` | path choice varies across reward x threat matrix |
| `16.2_balanced_conflict_max_deliberation` | balanced conflict has elevated deliberation proxies |
| `16.3_gate_not_constant` | Gate mode at junction is not constant |
| `16.4_one_shot_persistence` | post-shock choice and risk trace shift |
| `16.5_no_architecture_regression` | architecture invariants hold |

Acceptance criteria are not a substitute for inferential statistics.

### 7.2 Statistical tests

The Stage 3 config includes statistical comparison policy, including Bonferroni correction and effect-size names. Publication-grade reporting should still expose seed-level inferential tests in reviewer tables.

## 8. Parameter-freeze policy

| Parameter family | Status |
|---|---|
| Stage 3.1A baseline `exposure_q_bias=0.35` | frozen for Stage 3.1B |
| Stage 3.1A `eps=0.05` | frozen for Stage 3.1B |
| reward x threat matrix values | experimental manipulation |
| one-shot kind/trial/path/reward/salience/stakes | one-shot protocol manipulation |
| Gate thresholds in Stage 3.1B | closure configuration, not swept inside the reviewer package |
| temporal-state coupling values | closure configuration, not swept inside the reviewer package |
| ablation modifications | diagnostic interventions |

## 9. Methods checklist

A Methods section should cite:

1. the five-node open/covered topology;
2. the frozen Stage 3.1A baseline parameters;
3. the 3 x 3 reward x threat matrix;
4. the canonical conditions;
5. the one-shot protocol;
6. the five ablation conditions;
7. the logging fields used for outcome and VTE-proxy analysis;
8. the separation between acceptance checks and inferential tests.
