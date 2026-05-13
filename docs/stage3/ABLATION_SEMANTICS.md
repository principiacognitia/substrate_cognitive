# Stage 3.1B Ablation Semantics

Patch: 20A  
Scope: documentation only  
Target module: `stage3/configs/config_stage3_1b.py`  
Status: external-review hardening document

## 1. Purpose

This document defines what each Stage 3.1B ablation changes, what it leaves intact, and what claim it can or cannot support.

The ablations are not generic lesions of psychological constructs. They are specific parameter-level interventions in the Stage 3.1B implementation.

## 2. Scope

Primary scope: Stage 3.1B closure configuration.

Stage 3.1A contains earlier ablation descriptions that were useful during baseline development. Stage 3.1B updates the operational meaning of several ablations, especially `novg` and `novp`. Publication and reviewer-package interpretation should use the Stage 3.1B definitions below.

## 3. Ablation overview

| Ablation | Short name | Operational target | Main diagnostic question |
|---|---|---|---|
| `full` | Full | no modification | Does the intact closure model show the expected behavior? |
| `novg` | NoVG | Gate-level temporal rigidity removed | Are effects dependent on accumulated temporal risk in Gate routing? |
| `novp` | NoVp | policy-level source-local branch bias removed | Are positive/appetitive effects dependent on source-local policy carryover? |
| `nox` | NoX-to-Gate | exposure aggregate output blocked | Are effects dependent on exposure-field input to the Gate route? |
| `one_shot_off` | One-Shot-Off | one-shot formation and downstream influence disabled | Are effects dependent on the one-shot update regime itself? |

## 4. `full`

### 4.1 Modification

No modification.

```python
'modifications': {}
```

### 4.2 What remains active

Everything remains active:

- exposure aggregates;
- temporal state update;
- global `q_neg` and `q_pos`;
- source-local positive traces;
- Gate threshold cascade;
- stochastic policy;
- risk-penalized `EXPLOIT_SAFE`;
- local opportunity bonus in policy valuation;
- one-shot event pathway.

### 4.3 Interpretation

`full` is the reference model. All ablation comparisons are interpreted relative to it.

## 5. `novg`: Gate-level temporal rigidity removed

### 5.1 Modification

```python
'gate_thresholds': {
    'safe_drive_weight_temporal': 0.0,
    'safe_drive_weight_current': 1.0,
    'v_g_weight_hrisk': 0.0,
    'v_g_weight_xrisk': 1.0,
}
```

### 5.2 What is disabled

`novg` disables the accumulated temporal-risk contribution to Gate routing.

Specifically:

- `h_risk` no longer contributes to `safe_drive`;
- `h_risk` no longer contributes to `v_g_approx`;
- Gate behavior is driven by current exposure/risk more than accumulated temporal-risk trace.

### 5.3 What remains active

The following remain active:

- current `X_risk`;
- exposure aggregates;
- temporal updater;
- `q_neg` and `q_pos` trace formation;
- policy-level stochastic action selection;
- source-local positive traces, unless disabled by other ablations.

### 5.4 What it is not

`novg` is not a literal removal of all risk sensitivity.

It does not set all risk variables to zero. Current exposure risk can still influence the Gate because `safe_drive_weight_current=1.0` and `v_g_weight_xrisk=1.0`.

### 5.5 Claim tested

`novg` tests whether Stage 3.1B effects require accumulated temporal-risk contribution to Gate routing.

If an effect disappears in `novg`, the most conservative interpretation is:

> The effect depends on temporal-risk accumulation entering Gate routing.

A stronger interpretation such as "the psychological construct of gate viscosity is absent" should be avoided unless supported by additional analysis.

### 5.6 Expected reviewer concern

Because `novg` can strongly alter overall Gate routing, it may be a broad ablation. It should be described as a gate-temporal ablation, not as a clean isolated removal of a single scalar.

## 6. `novp`: policy-level branch bias removed

### 6.1 Modification

```python
'action_policy': {
    'local_opp_bonus_weight': 0.0
}
```

### 6.2 What is disabled

`novp` disables the source-local opportunity bonus in post-Gate option valuation:

```text
q_values_with_local_bonus =
    q_values + local_opp_bonus_weight * h_opp_local[source_id]
```

With `local_opp_bonus_weight=0.0`, source-local positive carryover no longer affects action values.

### 6.3 What remains active

The following remain active:

- Gate routing;
- exposure aggregates;
- temporal updater;
- global `h_risk`, `h_opp`, `q_neg`, `q_pos`;
- source-local trace formation in `TemporalState`;
- risk-penalized safe policy.

Only the policy-side use of local opportunity carryover is removed.

### 6.4 What it is not

`novp` does not remove all policy stochasticity.

It also does not remove global temporal-state dynamics. It specifically removes branch-specific appetitive carryover at the action-value level.

### 6.5 Claim tested

`novp` tests whether positive source-local persistence depends on policy-level branch bias.

If treat effects disappear in `novp` while shock effects remain, the conservative interpretation is:

> The positive source-local effect is policy-side; the negative effect is not primarily dependent on local appetitive branch bias.

## 7. `nox`: exposure field output blocked

### 7.1 Modification

```python
'exposure_field': {
    'zero_output': True
}
```

### 7.2 What is disabled

`nox` forces exposure aggregate output to zeros in the agent exposure pathway.

The intended disabled route is:

```text
ExposureField / exposure aggregates -> GateInput.exposure
```

### 7.3 What remains active

The following remain active:

- instant diagnostics;
- temporal state updater;
- stochastic policy;
- one-shot machinery unless separately disabled;
- any raw observation fields not overwritten by the exposure-zero pathway.

### 7.4 Important implementation note

`nox` blocks exposure aggregate output. It should not be described as "the environment has no risk" unless the runner also removes all raw risk-related observation fields.

The conservative interpretation is:

> The Gate's aggregate exposure channel is blocked.

### 7.5 Claim tested

`nox` tests whether effects depend on exposure aggregates entering the Gate/input pathway.

If a shock effect disappears in `nox`, the conservative interpretation is:

> The negative one-shot effect depends on exposure-derived risk entering the Stage 3 routing/update pathway.

## 8. `one_shot_off`: one-shot formation and downstream influence disabled

### 8.1 Modification

```python
'temporal_state': {
    'theta_shot': 999.0,
    'k_neg': 0.0,
    'k_pos': 0.0,
    'w_neg_to_risk': 0.0,
    'w_pos_to_opp': 0.0,
    'w_qneg_input': 0.0,
    'w_qpos_input': 0.0,
    'local_opp_immediate_seed_weight': 0.0,
}
```

### 8.2 What is disabled

This disables both one-shot trace formation and downstream influence:

| Parameter | Effect |
|---|---|
| `theta_shot=999.0` | one-shot threshold is effectively unreachable |
| `k_neg=0.0` | no new negative importance trace |
| `k_pos=0.0` | no new positive importance trace |
| `w_neg_to_risk=0.0` | `q_neg` cannot slow risk-trace update |
| `w_pos_to_opp=0.0` | `q_pos` cannot slow opportunity-trace update |
| `w_qneg_input=0.0` | `q_neg` cannot amplify effective risk input |
| `w_qpos_input=0.0` | `q_pos` cannot amplify effective opportunity input |
| `local_opp_immediate_seed_weight=0.0` | no immediate source-local appetitive seed |

### 8.3 What remains active

The following remain active:

- ordinary exposure and temporal updates below one-shot threshold;
- Gate threshold cascade;
- policy softmax;
- ordinary reward/threat matrix structure.

### 8.4 Claim tested

`one_shot_off` is the cleanest ablation for the central one-shot claim.

If shock and treat effects disappear in `one_shot_off` while baseline matrix behavior remains interpretable, the conservative interpretation is:

> The observed post-event behavioral deformation depends on the amplitude-dependent one-shot update regime.

## 9. How to interpret common patterns

### 9.1 Full effect present, one_shot_off absent

This supports the one-shot mechanism claim.

```text
full effect != 0
one_shot_off effect ~ 0
```

Interpretation:

> The effect is not merely a consequence of ordinary stochastic policy or reward/threat matrix structure.

### 9.2 Shock preserved in `novp`, treat removed in `novp`

Interpretation:

> Negative one-shot effects are not primarily carried by source-local appetitive policy bias, while positive/treat effects depend on that policy-side branch bias.

### 9.3 Shock removed in `nox`

Interpretation:

> Negative one-shot effects depend on exposure-derived risk input.

Avoid saying:

> The model has no risk processing in `nox`.

The exposure aggregate channel is blocked; other channels may remain depending on runner observations.

### 9.4 Effects removed in `novg`

Interpretation:

> Accumulated temporal-risk contribution to Gate routing is necessary for the observed effect under this protocol.

Avoid overclaiming:

> `V_G` as a psychological construct has been completely removed.

The ablation removes specific Gate-level temporal-risk weights.

## 10. Publication wording

Recommended wording:

> We used five diagnostic ablations. `NoVG` removed accumulated temporal-risk contribution to Gate routing while preserving current exposure input. `NoVp` removed source-local appetitive carryover at the policy valuation layer. `NoX-to-Gate` blocked exposure aggregate output. `One-Shot-Off` disabled formation and downstream coupling of one-shot importance traces. These ablations were interpreted as implementation-level causal probes rather than as exhaustive removals of psychological constructs.

Avoid wording:

- "NoVG removes all risk."
- "NoVp removes all action selection."
- "NoX removes the environment's threat."
- "One-Shot-Off removes memory."
- "The ablation proves biological equivalence."

## 11. Relation to Stage 3.1A legacy ablations

Stage 3.1A used earlier ablation descriptions:

| Ablation | Stage 3.1A legacy meaning | Stage 3.1B closure meaning |
|---|---|---|
| `novg` | forced temporal traces toward zero | Gate-level temporal-risk weights removed |
| `novp` | legacy viscosity parameters disabled | source-local policy branch bias removed |
| `nox` | exposure aggregates zeroed | exposure aggregate output blocked |
| `one_shot_off` | one-shot threshold/boost disabled | one-shot trace formation and downstream influence disabled |

For current reviewer-package interpretation, use the Stage 3.1B closure definitions.
