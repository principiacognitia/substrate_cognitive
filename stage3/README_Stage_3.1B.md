# README_Stage_3.1B (Rebuild Specification)

## Status

This document is the **approved rebuild specification** for Stage 3.1B.
It preserves the already accepted methodological constraints and rewrites the one-shot protocol around a stricter substrate-agnostic principle:

> a strong event must not be represented by a special symbolic label or by a discrete persistence schedule;
> it must instead alter the agent's own continuous ranking of importance and thereby modulate the dynamics of already existing traces and rheological carriers.

Stage 3.1A remains the calibrated baseline.
Stage 3.1B remains split into two protocols:

- **Stage 3.1B-1: Conflict protocol**
- **Stage 3.1B-2: Strong one-shot persistence protocol**

---

## 1. What Stage 3.1A already established

Stage 3.1A established only the following:

1. path choice no longer collapses into deterministic behavior;
2. covered-bias is stable;
3. junction deliberation is alive and non-degenerate;
4. timeout is no longer a hidden default.

This is sufficient to treat Stage 3.1A as a calibrated baseline, but not sufficient to test:

- reward-threat conflict;
- conflict-sensitive gate arbitration;
- persistent carryover from strong events.

---

## 2. What Stage 3.1B must prove

Stage 3.1B must test four stronger claims.

### C1. Tradeoff-sensitive choice

The agent must systematically shift between `open` and `covered` as reward premium and threat magnitude change.

### C2. Conflict-sensitive deliberation

VTE-like deliberation must peak when reward and threat are balanced in effective field magnitude, not simply when threat is high.

### C3. Gate relevance under conflict

The gate must stop being a decorative constant and show condition-sensitive variation in mode selection and commitment dynamics.

### C4. Persistent deformation by strong events

A single very strong event must be able to produce a carryover comparable in scale to many ordinary trials, by modulating already existing agent dynamics.

---

## 3. Hard constraints

### 3.1 Do not change the environment-agent boundary

The environment provides field potentials and path contingencies.
The agent performs inference, arbitration, temporal accumulation, and action selection.

The environment does **not** evaluate importance for the agent.

### 3.2 Do not introduce object-like or scheduler-like special cases

The following are **not allowed**:

- explicit shock counters;
- step-indexed persistence windows;
- cooldown timers;
- symbolic event labels that directly force a mode;
- a separate memory store for special events.

### 3.3 Stage 3.1B may refine the agent architecture

This is an explicit clarification.

Stage 3.1B **may** refine the coarse Stage 2 / 3.1A architecture, provided that the refinement:

- remains substrate-agnostic;
- remains inside the S-O-R + Gate logic;
- adds only continuous state variables with Markovian dynamics;
- uses those variables only as modulators of already existing update rules.

### 3.4 Allowed form of extension

The allowed extension is:

> add continuous variables that encode the current ranking of event importance by valence, and let those variables modulate the time constants or learning rates of already existing temporal and rheological dynamics.

No hidden scheduler is allowed.
Persistence must emerge from the coupled dynamics themselves.

---

## 4. Conceptual revision: what one-shot means in Stage 3.1B

The previous weak version of one-shot behaved mostly like a transient amplitude boost.
That was sufficient to test signal plumbing, but insufficient to satisfy the scientific target.

The revised interpretation is:

> a strong event is not merely a larger reward or punishment; it is an event whose valence is high enough to reweight the agent's own ranking of importance, thereby changing how already existing traces are retained, consolidated, and allowed to decay.

This is **not** a biological copy.
It is a substrate-agnostic computational analogue of the general fact that some events are treated by adaptive systems as more important than others.

The function of the new mechanism is therefore:

- not object recognition,
- not symbolic analysis,
- not explicit episodic labeling,
- but **priority assignment through valence**.

This gives Stage 3.1B a broader significance:

> the agent begins to solve the machine-learning problem of event ranking through field valence, not through object annotation or external importance labels.

---

## 5. Architectural update

### 5.1 Existing state classes remain

The architecture already contains:

- short-lived traces (`h_risk`, `h_opp`, `h_time`);
- gate rheology (`V_G`);
- pattern rheology (`V_p`).

These remain in place.

### 5.2 New state family: importance traces

Stage 3.1B introduces two additional continuous variables inside temporal dynamics:

- `q_neg` — persistent importance trace induced by strongly negative valence
- `q_pos` — persistent importance trace induced by strongly positive valence

These are **not** memories of objects or episodes.
They are slowly decaying state variables representing the current **importance weighting field** induced by past events.

### 5.3 Functional role of `q_neg` and `q_pos`

They do **not** store the event directly.
They modulate the already existing dynamics.

Specifically, they may influence:

- retention of `h_risk` and `h_opp`;
- effective salience passed into rheological updates;
- effective responsiveness of gate and pattern inertia to a strong event.

The intended principle is:

- ordinary trials shape traces gradually;
- strong events transiently raise the importance weighting field;
- the raised importance weighting field makes ordinary consolidation and decay dynamics temporarily asymmetric.

### 5.4 Minimal substrate-agnostic equations

The exact implementation may vary, but the design target is of this form.

#### Importance traces

```text
q_neg(t) = rho_neg * q_neg(t-1) + k_neg * max(0, X_risk(t) - theta_baseline)
q_pos(t) = rho_pos * q_pos(t-1) + k_pos * max(0, X_opp(t) - theta_baseline)
```

Where:

- `rho_neg`, `rho_pos` are retention coefficients of the importance traces;
- `k_neg`, `k_pos` control how strongly extreme events are injected;
- `theta_baseline` is the minimum effective field level at which an event begins to count as importance-relevant.

This is only the minimal form.
The actual injection term may also depend on salience and stakes, provided it remains continuous and Markovian.

#### Modulation of temporal traces

The effective decay or retention of already existing traces becomes a function of `q_neg` / `q_pos`.
A minimal target form is:

```text
lambda_eff_risk(t) = lambda_base / (1 + w_neg * q_neg(t))
lambda_eff_opp(t)  = lambda_base / (1 + w_pos * q_pos(t))
```

with temporal updates of the form:

```text
h_risk(t) = (1 - lambda_eff_risk(t)) * h_risk(t-1) + lambda_in_risk * X_risk(t)
h_opp(t)  = (1 - lambda_eff_opp(t))  * h_opp(t-1)  + lambda_in_opp  * X_opp(t)
```

Interpretation:

- when `q_neg` is low, `lambda_eff_risk` remains near the baseline decay regime;
- when `q_neg` is high, effective decay is suppressed and `h_risk` becomes more persistent;
- an analogous logic may hold for `q_pos` and `h_opp`.

#### Modulation of salience-sensitive rheology

The same importance traces may modulate the effective salience entering already existing rheological updates.
A schematic form is:

```text
S_eff = S * (1 + b_neg * q_neg + b_pos * q_pos)
```

This is not a new rheology.
It is a continuous reweighting of the already existing one.

### 5.5 Design commitment

The scientific commitment is **not** the exact formula above.
The scientific commitment is the architectural principle:

> strong events alter the persistence and impact of already existing traces by means of continuous importance states, without explicit clocks or symbolic event memory.

---

## 6. Stage 3.1B-1 — Conflict protocol

### 6.1 Purpose

Test whether the already calibrated agent expresses:

- reward-threat tradeoff;
- peak deliberation under balanced conflict;
- gate relevance.

### 6.2 Environment

Keep the same Stage 3 topology.
Do not change the junction state machine.

### 6.3 Value structure

Open path:

- higher reward potential;
- higher threat / exposure.

Covered path:

- lower reward potential;
- lower threat / exposure.

### 6.4 Conditions

Use a minimal 3 x 3 matrix:

- 3 reward premium levels;
- 3 open-threat levels.

Interpret conditions in **effective field space**, not only in raw environmental parameters.

### 6.5 Main hypotheses

- `P(open)` tracks reward-threat tradeoff;
- deliberation metrics peak in balanced conflict;
- gate metrics vary by condition.

---

## 7. Stage 3.1B-2 — Strong one-shot persistence protocol

### 7.1 Purpose

Test whether a single strong event can produce a carryover comparable in scale to many ordinary trials.

### 7.2 Split protocol

Run separately:

- **negative one-shot**
- **positive one-shot**

The system is **not** required to be symmetric.
A stronger and longer negative carryover is theoretically acceptable.

### 7.3 Required logic

One-shot must be implemented through continuous importance dynamics.
It must **not** be implemented through:

- forced post-shock windows;
- mode forcing;
- event counters;
- scheduler-based persistence.

### 7.4 Main hypotheses

- a negative one-shot increases `q_neg`, thereby extending the persistence of `h_risk` and biasing subsequent behavior toward safer modes and safer paths;
- a positive one-shot increases `q_pos`, thereby extending the persistence of opportunity-related traces and biasing subsequent behavior toward reward-seeking when conflict allows it;
- carryover must be localizable by ablation to `V_G`, `V_p`, or their interaction.

---

## 8. Primary metrics

### 8.1 Choice metrics

- `P(open)`
- `P(covered)`

### 8.2 Conflict metrics

- `junction_pause_duration`
- `commit_latency`
- `reorientation_count`
- `junction_deliberation_proxy`

### 8.3 Gate metrics

- `mode_before_junction`
- `mode_at_junction`
- `final_mode`
- `commit_reason`

### 8.4 Temporal metrics

- `h_risk`
- `h_opp`
- `h_time`
- `q_neg`
- `q_pos`
- `one_shot_fired`
- `one_shot_type`

---

## 9. Acceptance and falsification

### A1. Conflict success

Conflict matrix is accepted only if:

- path choice tracks tradeoff;
- deliberation peaks in balanced conflict;
- gate metrics are not constant decoration.

### A2. Strong one-shot success

Strong one-shot is accepted only if:

- carryover lasts on the order of tens of trials rather than 1–3 trials;
- the effect is stronger in `fired` than in `non_fired`;
- `q_neg` / `q_pos` predict behavior rather than merely changing internally.

### A3. Localization success

One-shot is accepted only if ablations help localize the carrier:

- `NoVG`
- `NoVp`
- optionally both.

### Falsification examples

The rebuild fails if:

- persistence still depends on hidden timing logic;
- strong events do not outlast ordinary traces;
- conflict matrix collapses after the extension;
- one-shot changes internal traces but leaves behavior effectively unchanged.

---

## 10. Final principle

Stage 3.1B is successful only if the agent remains intelligible as the same S-O-R + Gate + Rheology system, while strong events become capable of reweighting the persistence of already existing dynamics through continuous importance-state modulation.
