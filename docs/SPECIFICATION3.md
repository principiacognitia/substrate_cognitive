# Stage 3.0 — Technical Specification

## 1. Purpose

Stage 3.0 is an architectural refactor of the Stage 2 agent.

Its purpose is to preserve the single-Gate logic of S-O-R+Gate while extending the input interface from a purely instantaneous diagnostic vector to a two-layer control interface:

- instantaneous diagnostics
- compressed temporal state

Stage 3.0 does **not** implement the full Shelter Cross task, full absence inference, or a richer rheological theory.
It prepares the architecture for Stage 3.1 and Stage 3.2.

---

## 2. Core Design Constraints

### 2.1 One Gate only
There is exactly one Gate.
No parallel arbitration modules for threat, reward, absence, or visibility.

### 2.2 No `argmax` arbitration
Gate routing must not be implemented as:

- global scoring over all modes
- utility ranking
- `argmax(mode_score)`

Mode selection must be implemented as a **cascade of thresholds / interrupts**, where stronger signals override weaker ones.

### 2.3 No ready semions at port
Sensory preprocessing must not deliver pre-classified object labels such as `"snake"`, `"stick"`, `"food"`, or `"predator"`.

The sensory interface may only return:

- valence-/exposure-weighted embeddings
- aggregated field quantities
- compressed traces

A semion is not an object label at the input.
A semion is a downstream stabilization/compression in the S-O-R cycle.

### 2.4 Ontology ≠ engineering decomposition
For the agent, valence and observability are one prospective field of expected future entropy change.

For implementation, logging, ablation, and tests, the field may be analytically decomposed as:

\[
X = O \cdot \nu
\]

where:

- \(\nu\) = valence-weighted signal profile
- \(O\) = detectability / observability
- \(X\) = integrated exposure

This decomposition is engineering-facing, not an ontological claim about separate primitives inside the agent.

### 2.5 One-shot is not a separate module
One-shot learning is not a new memory subsystem.

It is an **amplitude-dependent update regime** of the same temporal state:
- either many weak events
- or one strong event

must be able to shift the same traces.

---

## 3. Gate Input Architecture

The Gate reads three typed layers.

### 3.1 `InstantDiagnostics`
Minimal instantaneous diagnostics inherited from Stage 2:

- `u_delta`: unsigned prediction error
- `u_entropy`: policy entropy
- `u_volatility`: volatility proxy

These are scalar, non-categorical quantities.

### 3.2 `ExposureAggregates`
Aggregated field quantities produced by `exposure_field.py`:

- `X_risk`: current risky exposure
- `X_opp`: current opportunity exposure
- `D_est`: current diagnosticity / visibility estimate

These are the only exposure quantities directly visible to the Gate.

The Gate must not receive raw object labels, raw taxonomies, or semantically named classes.

### 3.3 `TemporalState`
Compressed temporal history:

- `h_risk`: exponentially smoothed risky exposure trace
- `h_opp`: exponentially smoothed opportunity trace
- `h_time`: time trace since last high-salience event

This is the minimal temporal state for Stage 3.0.

No additional temporal variables are part of the core Stage 3.0 contract.

---

## 4. Temporal Update Rules

Stage 3.0 must implement the temporal layer through simple trace dynamics.

### 4.1 Risk trace
\[
h_{risk}(t+1)=\lambda_r h_{risk}(t)+(1-\lambda_r)X_{risk}(t)
\]

### 4.2 Opportunity trace
\[
h_{opp}(t+1)=\lambda_o h_{opp}(t)+(1-\lambda_o)X_{opp}(t)
\]

### 4.3 Time trace
\[
h_{time}(t+1)=
\begin{cases}
0, & \text{if salience} > \tau_{sal} \\
h_{time}(t)+1, & \text{otherwise}
\end{cases}
\]

### 4.4 One-shot regime
One-shot occurs when event amplitude is high enough to produce a large instantaneous shift in `h_risk` or `h_opp`.

This is not a separate module and must not introduce a second memory architecture.

---

## 5. Gate Routing Logic

Mode routing must be implemented as a threshold cascade.

Available modes:

- `EXPLOIT`
- `EXPLORE`
- `EXPLOIT_SAFE`
- `ABSENCE_CHECK`

### 5.1 Threat override
If risky temporal pressure is critical, the Gate must immediately route to `EXPLOIT_SAFE`.

Conceptually:
```python
if h_risk > THRESHOLD_CRITICAL_RISK:
    return EXPLOIT_SAFE
````

This route can bypass the standard explore barrier.

### 5.2 Absence trigger

If suspicion remains elevated, current visibility is poor, and the temporal state indicates that the system can afford a costly check, the Gate may enter `ABSENCE_CHECK`.

Conceptually:

```
if h_risk > THRESHOLD_SUSPICION and D_est < THRESHOLD_VISIBILITY and h_time > THRESHOLD_WAIT:
    return ABSENCE_CHECK
```

### 5.3 Standard arbitration

If no critical override is active, Stage 2 logic is preserved:

```
pressure = compute_pressure(u_t)
if pressure * (1 - V_G) > THETA_MB:
    return EXPLORE
return EXPLOIT
```

No global comparison over modes is allowed.

* * *

6\. Backward Compatibility
--------------------------

Stage 3.0 must degrade to Stage 2 behavior when:

*   `X_risk = 0`
*   `X_opp = 0`
*   `D_est = 0`
*   `h_risk = 0`
*   `h_opp = 0`
*   `h_time = 0`
*   only `EXPLOIT` and `EXPLORE` are reachable

Under these conditions, Stage 3.0 must reproduce Stage 2A/2B behavior up to numerical tolerance.

* * *

7\. Repository Responsibilities
-------------------------------

### 7.1 `stage3/core/gate_modes.py`

Contains:

*   mode enum / constants only

No gate logic.

### 7.2 `stage3/core/gate_inputs.py`

Contains:

*   dataclass definitions only:
    *   `InstantDiagnostics`
    *   `ExposureAggregates`
    *   `TemporalState`
    *   optional `GateInput`

No update logic.  
No field computation.  
No unit tests.  
No scoring/vectorization utilities required by `argmax`.

### 7.3 `stage3/core/exposure_field.py`

Responsible for:

*   computing internal  $\nu$ ,  $O$ ,  $X$ 
*   exporting only:
    *   `X_risk`
    *   `X_opp`
    *   `D_est`

### 7.4 `stage3/core/temporal_state.py`

Responsible for:

*   updating `h_risk`
*   updating `h_opp`
*   updating `h_time`
*   implementing one-shot as amplitude-dependent update regime

### 7.5 `stage3/core/gate_stage3.py`

Responsible for:

*   threshold cascade
*   mode routing
*   no global optimizer
*   no `argmax`

### 7.6 `stage3/core/compatibility.py`

Responsible for:

*   Stage 2 compatibility shims
*   zero-exposure degradation mode

* * *

8\. Acceptance Criteria
-----------------------

Stage 3.0 is complete only if all of the following hold.

### 8.1 Backward compatibility

With zero exposure and zero temporal traces, Stage 3.0 reproduces Stage 2 behavior.

### 8.2 No ready semions

The sensory interface does not expose categorical object labels to the Gate.

### 8.3 One-shot as update regime

A single high-amplitude event can shift the same temporal traces that many weak events shift gradually.

### 8.4 Exposure-aware routing

High risky exposure can force `EXPLOIT_SAFE` without global mode comparison.

### 8.5 Action-basin equivalence

Two perceptually different stimuli that induce similar exposure profiles and require the same response regime must lead to similar Gate routing.

* * *

9\. Required Tests
------------------

### 9.1 `test_no_ready_semions.py`

Must verify that the Gate-facing input contains no object labels or semantic class names.

### 9.2 `test_temporal_state.py`

Must verify correct updates of:

*   `h_risk`
*   `h_opp`
*   `h_time`
*   one-shot amplitude regime

### 9.3 `test_gate_stage3.py`

Must verify:

*   threat override
*   absence trigger
*   standard Stage 2 fallback

### 9.4 `test_backward_compatibility.py`

Must verify Stage 2-equivalent routing under zeroed Stage 3 fields.

### 9.5 `test_action_basin_equivalence.py`

Must verify that perceptually different but actionally equivalent stimuli converge to the same mode basin.

* * *

10\. Non-Goals of Stage 3.0
---------------------------

Stage 3.0 does not implement:

*   a full absence-check task
*   a full Shelter Cross environment
*   full spatial VTE matching
*   a richer thixotropic rheology
*   a unified field ontology engine inside the repo

Stage 3.0 is an interface refactor, not a full new experimental paper.

