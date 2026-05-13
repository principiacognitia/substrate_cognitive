# Stage 3 Agent Architecture

Patch: 20A  
Scope: documentation only  
Target modules: `stage3/core/agent_stage3.py`, `stage3/core/gate_inputs.py`, `stage3/core/temporal_state.py`, `stage3/core/gate_stage3.py`  
Status: external-review hardening document

## 1. Purpose

`AgentStage3` is the orchestration layer for Stage 3. It connects the spatial environment, exposure aggregation, temporal traces, threshold-cascade Gate routing, and mode-specific stochastic action selection.

The agent is not a classifier over semantic object labels. It receives numeric diagnostics, numeric exposure aggregates, and numeric temporal traces. These are then used by a single Gate to select one of the control modes.

High-level step architecture:

```text
Environment observation
    -> InstantDiagnostics
    -> ExposureAggregates
    -> TemporalState update
    -> GateInput
    -> GateStage3 threshold cascade
    -> mode-specific stochastic policy
    -> action + metadata/logging
```

## 2. Architectural constraints

| Constraint | Meaning in implementation |
|---|---|
| One Gate only | All modes are routed by `GateStage3`; no separate trauma, safety, or one-shot gate is introduced. |
| No argmax arbitration | Gate routing is a priority cascade of threshold checks. Diagnostic scores are logged, but the selected mode is not chosen by global argmax. |
| No ready semions at port | Gate input dataclasses contain numeric fields, not object labels such as `food`, `snake`, or `shelter`. |
| Ontology != engineering interface | Analytic variables such as valence, observability, and exposure may be used for tracing, but the Gate receives only aggregate numeric fields. |
| One-shot is an update regime | High-amplitude events alter continuous temporal traces; they are not stored as symbolic event memories or fixed post-event windows. |
| Source-local carryover is policy-side | Source-local appetitive traces may bias option valuation, but source identity is not a Gate-routing input. |

## 3. Core modules

| Module | Role |
|---|---|
| `gate_inputs.py` | Defines the three-layer Gate input contract: `InstantDiagnostics`, `ExposureAggregates`, `TemporalState`, and combined `GateInput`. |
| `temporal_state.py` | Updates global temporal traces and source-local appetitive traces. |
| `gate_stage3.py` | Selects Gate mode through threshold cascade. |
| `agent_stage3.py` | Orchestrates diagnostics, exposure, temporal update, Gate selection, action policy, and logging. |

## 4. Gate input layers

### 4.1 InstantDiagnostics

`InstantDiagnostics` contains Stage 2-compatible immediate diagnostic signals:

| Field | Meaning |
|---|---|
| `u_delta` | unsigned prediction error or surprise proxy |
| `u_entropy` | policy entropy |
| `u_volatility` | EMA-style volatility proxy derived from prediction error |
| `trial` | metadata for logging |

These variables are numeric. They do not carry semantic labels.

### 4.2 ExposureAggregates

`ExposureAggregates` contains current exposure-field aggregates:

| Field | Meaning |
|---|---|
| `X_risk` | current risky exposure aggregate |
| `X_opp` | current opportunity exposure aggregate |
| `D_est` | current detectability / visibility estimate |
| `region_id` | optional logging metadata, not used as routing semantics |
| `trial` | metadata for logging |

`ExposureAggregates.zeros()` is used for compatibility and for the `nox` ablation path.

### 4.3 TemporalState

`TemporalState` stores compressed temporal history:

| Field | Gate-visible? | Meaning |
|---|---:|---|
| `h_risk` | yes | smoothed risk trace |
| `h_opp` | yes | smoothed opportunity trace |
| `h_time` | yes | time since last high-salience event |
| `q_neg` | yes, through temporal dynamics and logging | negative-event importance trace |
| `q_pos` | yes, through temporal dynamics and logging | positive-event importance trace |
| `q_pos_local` | no Gate routing | source-local positive importance trace |
| `h_opp_local` | no Gate routing | source-local opportunity carryover used by policy valuation |
| `one_shot_pending` | logging/debug | whether current update exceeded one-shot threshold |
| `one_shot_amplitude` | logging/debug | salience times stakes |
| `one_shot_type` | logging/debug | `none`, `negative`, or `positive` |

The local dictionaries are intentionally separated from Gate routing. They may affect post-Gate option valuation through the action policy, but they do not introduce source-id routing into the Gate.

## 5. One agent step

`AgentStage3.step()` executes the following sequence.

### 5.1 Compute instant diagnostics

The agent computes:

```text
u_delta      = abs(reward - expected_reward) if reward is available
u_entropy    = observation["policy_entropy"] or 0
u_volatility = EMA(u_delta)
```

The diagnostics are stored in `InstantDiagnostics`.

### 5.2 Compute exposure aggregates

If the observation already contains `X_risk`, `X_opp`, and `D_est`, these values are used directly. Otherwise, `ExposureField.compute_exposure()` is called.

For the `nox` ablation, `exposure_zero_output=True` forces the aggregate output to zeros.

At a junction, Gate exposure may use the mean of option-level risk values when available:

```text
gate_exposure.X_risk = mean(option_risk_values)
gate_exposure.D_est  = mean(option_visibility_values), if available
```

The metadata field `gate_exposure_source` records whether Gate exposure came from node exposure, junction option risk, or ablation-zero exposure.

### 5.3 Update temporal state

Temporal state is updated before Gate routing.

Important event pathway fields may be read from observation:

| Observation field | Role |
|---|---|
| `one_shot_fired` | marks that the environment emitted a high-amplitude event |
| `one_shot_salience` | event salience override |
| `one_shot_stakes` | event stakes override |
| `one_shot_source_X_risk` | source risk override for negative event pathway |
| `one_shot_source_X_opp` | source opportunity override for positive event pathway |
| `one_shot_source_id` | source id used only for source-local positive traces |
| `option_source_ids` | option identifiers for policy-side local carryover |
| `option_reward_values` | numeric option affordance values for local appetitive input |

The one-shot pathway modifies continuous traces; it does not create a discrete memory module.

### 5.4 Build GateInput

If compatibility mode is off, the agent constructs:

```python
GateInput(
    instant=instant_diagnostics,
    exposure=gate_exposure_aggregates,
    temporal=current_temporal_state,
)
```

If compatibility mode is on, the Stage 2 compatibility shim creates a reduced input.

### 5.5 Select Gate mode

`GateStage3.select_mode()` routes the input through threshold cascade. The returned metadata includes:

| Metadata field | Meaning |
|---|---|
| `winning_constraint` | the threshold condition that selected the mode |
| `mode_scores` | diagnostic scores for all modes, used for logging only |
| `gate_state_snapshot` | numeric snapshot of Gate-relevant signals |
| `safe_drive` | current plus accumulated threat drive |
| `uncertainty_signal` | sigmoid-transformed uncertainty |
| `v_g_approx` | approximate gate viscosity from current and accumulated risk |
| `explore_gate_output` | uncertainty signal attenuated by gate viscosity |

### 5.6 Select action

The selected mode determines the stochastic policy.

| Gate mode | Policy |
|---|---|
| `EXPLOIT` | softmax over Q-values with `beta_exploit` |
| `EXPLORE` | softmax over Q-values with `beta_explore`, optionally epsilon-random if configured |
| `EXPLOIT_SAFE` | softmax over risk-penalized values with `beta_safe` |
| `ABSENCE_CHECK` | softmax over Q-values with exploratory temperature |

The action policy is stochastic. It uses softmax, not argmax.

For `EXPLOIT_SAFE`:

```text
q_safe = q_values_with_local_bonus - lambda_risk * risk_values
```

Source-local appetitive carryover is applied only at policy valuation:

```text
q_values_with_local_bonus[action_i] =
    q_values[action_i] + local_opp_bonus_weight * h_opp_local[source_i]
```

This is not Gate routing.

### 5.7 Log and return metadata

The agent returns `(action, metadata)`. The metadata contains selected mode, action, node exposure, Gate exposure, temporal state, instant diagnostics, Gate constraint, action probabilities, raw and local-bonus Q-values, risk values, source-local trace maps, Gate snapshot, and one-shot diagnostic fields.

## 6. Gate routing

The Gate priority order is:

```text
1. ABSENCE_CHECK
2. EXPLOIT_SAFE
3. EXPLORE
4. EXPLOIT
```

### 6.1 ABSENCE_CHECK

Condition:

```text
h_risk > suspicion_threshold
AND D_est < visibility_threshold
AND h_time > safe_window_threshold
```

This mode represents a high-suspicion, low-visibility, safe-window check.

### 6.2 EXPLOIT_SAFE

Safe drive:

```text
safe_drive =
    safe_drive_weight_current  * X_risk
  + safe_drive_weight_temporal * h_risk
```

Condition:

```text
safe_drive > critical_risk_threshold
```

This override is above normal exploration. Critical threat exposure can bypass the standard explore barrier.

### 6.3 EXPLORE

Uncertainty signal:

```text
uncertainty_signal =
    sigmoid(w_volatility * u_volatility + w_entropy * u_entropy - theta_u)
```

Approximate gate viscosity:

```text
v_g_approx =
    clip(v_g_weight_hrisk * h_risk + v_g_weight_xrisk * X_risk, 0, 1)
```

Explore output:

```text
explore_gate_output = uncertainty_signal * (1 - v_g_approx)
```

Condition:

```text
explore_gate_output > theta_mb
```

### 6.4 EXPLOIT

Default mode when none of the higher-priority thresholds fire.

## 7. Temporal-state update

The rebuilt Stage 3.1B temporal updater treats one-shot as a continuous update regime.

### 7.1 Event amplitude

```text
surprise_amplitude = max(0, salience) * max(0, stakes)
surprise_excess    = max(0, surprise_amplitude - theta_shot)
is_one_shot        = surprise_excess > 0
```

### 7.2 Event source

If event-specific source fields are supplied, they are used:

```text
source_X_risk = event_X_risk if supplied else X_risk
source_X_opp  = event_X_opp  if supplied else X_opp
```

This prevents false positive positive traces during negative events and false negative negative traces during positive events.

### 7.3 Importance input

```text
risk_excess = max(0, source_X_risk - theta_baseline)
opp_excess  = max(0, source_X_opp  - theta_baseline)

shot_neg = risk_excess * surprise_excess
shot_pos = opp_excess  * surprise_excess
```

### 7.4 Global importance traces

```text
q_neg(t+1) = clip(rho_neg * q_neg(t) + k_neg * shot_neg, 0, q_clip)
q_pos(t+1) = clip(rho_pos * q_pos(t) + k_pos * shot_pos, 0, q_clip)
```

### 7.5 Effective update rates

The effective update rate is reduced by prior importance, which slows relaxation of the corresponding trace:

```text
lambda_risk_eff = lambda_risk / (1 + w_neg_to_risk * q_neg(t))
lambda_opp_eff  = lambda_opp  / (1 + w_pos_to_opp  * q_pos(t))
```

### 7.6 Effective inputs

Prior importance can also amplify incoming exposure input:

```text
risk_gain = 1 + w_qneg_input * q_neg(t) / (1 + q_neg(t))
opp_gain  = 1 + w_qpos_input * q_pos(t) / (1 + q_pos(t))

X_risk_eff = clip(X_risk * risk_gain, 0, 1)
X_opp_eff  = clip(X_opp  * opp_gain, 0, 1)
```

### 7.7 Global trace update

```text
h_risk(t+1) =
    (1 - lambda_risk_eff) * h_risk(t)
  + lambda_risk_eff       * X_risk_eff

h_opp(t+1) =
    (1 - lambda_opp_eff) * h_opp(t)
  + lambda_opp_eff       * X_opp_eff
```

### 7.8 h_time

```text
h_time(t+1) = 0             if salience > salience_threshold
h_time(t+1) = h_time(t) + 1 otherwise
```

### 7.9 Source-local positive carryover

Positive one-shot events can create source-local traces:

```text
q_pos_local[source_id]
h_opp_local[source_id]
```

Only event-tagged positive sources receive ongoing local appetitive input. Ordinary visible options do not automatically seed local opportunity traces.

The source-local trace may immediately seed `h_opp_local` on a positive one-shot. This supports immediate post-event branch-specific bias without adding a source-id term to Gate routing.

## 8. Logging contract

`AgentStage3` logs both routing-level and policy-level diagnostics. Key fields include:

| Field group | Examples |
|---|---|
| trial/time | `seed`, `trial`, `tick`, `node_id`, `at_junction` |
| Gate | `mode_after`, `gate_trigger`, `safe_drive`, `uncertainty_signal`, `v_g_approx` |
| diagnostics | `u_delta`, `u_entropy`, `u_volatility` |
| exposure | `X_risk`, `X_opp`, `D_est` |
| temporal | `h_risk`, `h_opp`, `h_time`, `q_neg`, `q_pos` |
| policy | `q_values`, `q_values_with_local_bonus`, `risk_values`, `action_probs`, `sampled_action` |
| one-shot | `one_shot_fired`, `one_shot_type`, `one_shot_amplitude` |

## 9. What this architecture does not claim

The Stage 3 agent does not implement:

- symbolic episodic memory;
- object permanence;
- allocentric spatial cognition;
- a separate trauma module;
- a semantic object classifier;
- direct biological equivalence of synthetic trajectories.

It implements a numeric control architecture in which exposure aggregates and continuous temporal traces modulate Gate routing and post-Gate stochastic policy.
