# Stage 3.0: Architectural Refactor

**Status:** 🟡 **In Development**

**Purpose:** Refactor the Gate interface from instantaneous diagnostics to two-layer control (instantaneous + temporal).

**Full Specification:** [docs/SPECIFICATION3.md](../docs/SPECIFICATION3.md)

---

## 📖 Description

Stage 3.0 is an **architectural refactor** of the Stage 2 agent, which extends the Gate input interface from a purely instantaneous diagnostic vector to a **two-layer control system**:

1. **Instantaneous Diagnostics ($u_t$)** — instantaneous diagnostic variables (inherited from Stage 2)
2. **Temporal State ($h_t$)** — compressed temporal history
3. **Exposure Aggregates** — aggregated curvatures of the exposure field

### Key Principle

> **No Ready Semions at Port.** The sensory input does not contain pre-classified object labels ("snake," "stick"). Instead, it consists of valence/exposure-weighted embeddings. ---

## 🗂️ Structure

```
stage3/
├── core/                       # Stage 3.0 Core
│   ├── gate_modes.py           # Gate Modes (enum only)
│   ├── gate_inputs.py          # Dataclass Definitions (3 layers)
│   ├── exposure_field.py       # Calculation of ν, O, X → aggregates
│   ├── temporal_state.py       # Updating h_risk, h_opp, h_time
│   ├── gate_stage3.py          # Threshold Cascade Routing
│   ├── agent_stage3.py         # Orchestration (env → gate → action)
│   └── compatibility.py        # Stage 2 Compatibility Shims
├── configs/                    # Configurations
│   ├── config_stage3_base.py
│   └── config_stage3_debug.py
├── envs/                       # Test Environments
│   ├── dummy_temporal_env.py
│   └── open_covered_choice_env.py
├── tests/                      # Integration Tests
│   ├── test_no_ready_semions.py
│   ├── test_temporal_state.py
│   ├── test_gate_stage3.py
│   ├── test_backward_compatibility.py
│   └── test_action_basin_equivalence.py
└── README.md                   # This file
```

---

## 🚀 Quick Start

### Installation

```bash
# From the repository root
cd substrate_cognitive

# Install dependencies (if not already installed)
pip install -r requirements.txt
```

### Running Tests

```bash
# Unit tests for each module
python -m stage3.core.gate_inputs
python -m stage3.core.gate_modes
python -m stage3.core.exposure_field
python -m stage3.core.temporal_state
python -m stage3.core.gate_stage3
python -m stage3.core.agent_stage3
python -m stage3.core.compatibility
```

###  Creating an Agent

```python
from stage3.core.agent_stage3 import AgentStage3, AgentStage3Config

# Configuration
config = AgentStage3Config( 
compatibility_mode=False, # Stage 3 mode (not Stage 2 emulation) 
log_level=1
)

# Creating an agent
agent = AgentStage3(config)

# One step
observation = { 
'prediction_error': 0.5, 
'policy_entropy': 0.3, 
'q_values': [0.6, 0.4]
}

action, metadata = agent.step( 
observation=observation, 
reward=0.8, 
action=0
)

print(f"Selected mode: {metadata['mode']}")
print(f"Gate constraint: {metadata['gate_constraint']}")
```

### Backward Compatibility (Stage 2 emulation)

```python
from stage3.core.agent_stage3 import AgentStage3, AgentStage3Config

# Enable Stage 2 compatibility mode
config = AgentStage3Config(
compatibility_mode=True, # Zero exposure and temporal
log_level=1
)

agent = AgentStage3(config)

# Behavior must match Stage 2 (within 5% tolerance)
```

---

## 🎯 Design Constraints

### 1. One Gate only

No parallel arbitration modules for threat, reward, absence, or visibility. Gate **one**.

### 2. No `argmax` arbitration

Mode selection is implemented as a **cascade of thresholds/interrupts**, where strong signals override weaker ones:

```
Priority (highest → lowest):
1. ABSENCE_CHECK — high stakes + poor visibility + safe window
2. EXPLOIT_SAFE — critical threat exposure
3. EXPLORE — high uncertainty + low threat
4. EXPLOIT — default
```

### 3. No ready-made semions at the port

The sensory input does not contain string labels:

```python
# ❌ INCORRECT:
observation = {'object': 'snake', 'threat_level': 'high'}

# ✅ CORRECT:
observation = {
'prediction_error': 0.8,
'policy_entropy': 0.3,
'q_values': [0.2, 0.8]
}
```

### 4. Ontology ≠ Engineering Interface

For the agent: a unified *prospective field* of expected entropy change.
For the researcher: an analytical decomposition into ν, O, and X for tracing purposes.

### 5. One-shot learning is not a separate module

One-shot learning is an **amplitude-dependent update regime** applied to the *same* temporal state, not a separate memory module.

---

## 📊 Acceptance Criteria

Stage 3.0 is considered complete only if all the following conditions are met:

| # | Criterion | How to Test |
| :--- | :--- | :--- |
| **8.1** | Backward compatibility | With zero exposure/temporal values ​​→ Stage 2 behavior |
| **8.2** | No ready-made semions | The sensory interface contains no object labels |
| **8.3** | One-shot as an update regime | A single high-amplitude event shifts the same traces |
| **8.4** | Exposure-aware routing | High X_risk forces the EXPLOIT_SAFE mode |
| **8.5** | Action-basin equivalence | perceptually distinct → same mode basin |

---

## 🧪 Tests

### Running All Tests

```bash
# From the repository root
python -m stage3.tests.test_no_ready_semions
python -m stage3.tests.test_temporal_state
python -m stage3.tests.test_gate_stage3
python -m stage3.tests.test_backward_compatibility
python -m stage3.tests.test_action_basin_equivalence
```

### Example: test_backward_compatibility.py

```python
from stage3.core.agent_stage3 import AgentStage3, AgentStage3Config
from stage3.core.compatibility import verify_backward_compatibility

# Stage 3 in compatibility mode
config = AgentStage3Config(compatibility_mode=True)
agent = AgentStage3(config)

# Run an episode
for trial in range(100):
action, metadata = agent.step(observation, reward)

# Verify Stage 2 equivalence
stage3_results = {...}  # metrics from Stage 3
stage2_results = {...}  # metrics from Stage 2

passed, message = verify_backward_compatibility(
stage3_results,
stage2_results,
tolerance=0.05
)

assert passed, message
```

---

## 📈 Gate Modes

| Mode | Activation Condition | Priority |
| :--- | :--- | :--- |
| **EXPLOIT** | Default (low uncertainty) | 4 (lowest) |
| **EXPLORE** | High uncertainty + low threat | 3 |
| **EXPLOIT_SAFE** | Critical threat exposure | 2 |
| **ABSENCE_CHECK** | High stakes + poor visibility + safe window | 1 (highest) | ### Threshold Cascade Logic

```python
# Simplified logic (full: gate_stage3.py)

if h_risk > THRESHOLD_SUSPICION and D_est < THRESHOLD_VISIBILITY and h_time > THRESHOLD_WAIT: 
return ABSENCE_CHECK

if X_risk > THRESHOLD_CRITICAL_RISK: 
return EXPLOIT_SAFE

if pressure * (1 - V_G) > THETA_MB: 
return EXPLORE

return EXPLOIT
```

---

## 🔧 Configuration

###AgentStage3Config

```python
from stage3.core.agent_stage3 import AgentStage3Config

config = AgentStage3Config( 
compatibility_mode=False, # Stage 3 mode 
exposure_field_config={ # exposure_field.py parameters 
'valence_scale': 1.0, 
'observability_scale': 1.0, 
'risk_threshold': 0.5 
}, 
temporal_state_config={ # temporal_state.py options 
'lambda_risk': 0.9, 
'lambda_opp': 0.9, 
'salience_threshold': 0.5, 
'one_shot_threshold': 5.0, 
'one_shot_boost': 2.0 
}, 
gate_thresholds={ # gate_stage3.py thresholds 
'critical_risk_threshold': 0.7, 
'suspicion_threshold': 0.5, 
'visibility_threshold': 0.3, 
'safe_window_threshold': 50, 
'theta_mb': 0.30, 
'theta_u': 1.5 
}, 
log_level=1 # 0=none, 1=summary, 2=full
)
```

---

## 📄 Documents

| Document | Description |
| :--- | :--- |
| [Full Specification](../docs/SPECIFICATION3.md) | Full technical specification Stage 3.0 |
| [Stage 2 README](../stage2/README.md) | Previous Version (for comparison) |
| [Root README](../README.md) | Project Overview |

---

## 🐛 Troubleshooting

### Error: "NameError: temporal is not defined"

Ensure that `gate_stage3.py` receives all three layers:

```python
# ✅ CORRECT:
instant = gate_input.instant
exposure = gate_input.exposure
temporal = gate_input.temporal

# ❌ INCORRECT:
if self._should_trigger_explore(instant, exposure):  # missing temporal
```

### Error: "No Ready Semions constraint violated"

The sensory input contains string labels:

```python
# ❌ INCORRECT:
observation = {'object_label': 'snake'}

# ✅ CORRECT:
observation = {'prediction_error': 0.5, 'policy_entropy': 0.3}
```

---

## 📬 Contact

**Author:** Alex Snow (Aleksey L. Snigirov)
**Email:** alex2saaba@gmail.com
**ORCID:** 0009-0001-3713-055X

---

**Last updated:** March 2026