# VTE Lab Adapters

This directory contains biological data adapters used by the Stage 3.2C biological comparability layer.

The purpose of this layer is to translate external laboratory datasets into explicit, auditable comparison endpoints for the frozen Stage 3.2 VTE wrapper.

The adapters are not intended to prove that a synthetic animal visually resembles a biological animal. They are intended to test whether biological behavioral and neural data can be mapped into choice-centered measurements comparable to the model outputs.

## Core principle

The primary Stage 3 claim is not VTE itself.

The primary claim is gate-viscosity dynamics:

```text
viscous gate dynamics
  -> boundary / ambiguous conditions:
       delayed commitment, instability, deliberation-like behavior
  -> obvious conditions:
       stable and confident commitment
  -> strong one-shot events:
       rapid update of future choice
  -> stimulus removal:
       carryover / hysteresis
````

VTE-like behavior is treated as a secondary behavioral readout of that dynamic.

Therefore, biological VTE-like metrics are useful only when they are linked to recoverable choice episodes.

## Biological comparability hierarchy

### Primary endpoints

The strongest biological comparison targets are choice and outcome variables:

* chosen arm / route / path;
* available alternatives;
* reward / outcome;
* correctness or error when recoverable;
* switch / stay;
* perseveration;
* condition-wise shift;
* reversal, perturbation, risk, conflict, or carryover when recoverable.

### Secondary behavioral endpoints

VTE-like trajectory measures are secondary endpoints:

* choice-point dwell;
* pause duration;
* reorientation count;
* IdPhi-like angular displacement;
* trajectory tortuosity;
* head-scanning / hesitation proxies.

These metrics should not be interpreted outside a choice context.

### Neural proxy endpoints

Neural data can be used as a deliberation proxy only when aligned to choice episodes and interpretable alternatives.

Potential neural proxies:

* decoded candidate-arm or candidate-route representations;
* replay content around a choice point;
* alternating representation of competing options;
* dominance margin between alternatives;
* representation entropy;
* replay event count;
* LFP or spike events with explicit relation to choice state.

A spike, replay event, or LFP event is not sufficient by itself. It becomes relevant only when linked to task alternatives and choice/outcome structure.

## Adapter contract

A biological adapter should preserve the following identifiers whenever available:

```text
dataset_id
trace_origin
subject_id
animal_id
session_id
run_id
trial or choice_episode_id
choice_point_id
committed_path / chosen_arm / chosen_route
event_time
reward / outcome
condition / task state
```

Adapters must write metadata documenting:

```text
source files
subject/session extraction rule
position source
event source
time-alignment method
choice-event definition
reward-event definition
window definition
heading derivation
sample-rate estimate
smoothing parameters
filters / refractory periods
known limitations
```

## Frozen wrapper rule

Lab adapters must transform biological data into the existing canonical VTE trace schema.

They must not modify the frozen Stage 3.2A/B VTE wrapper to make a biological dataset fit.

Allowed:

* adapter-side event filtering;
* adapter-side canonical trace export;
* adapter-side heading derivation;
* adapter-side reward/event attachment;
* post-wrapper normalization;
* post-wrapper eligibility and comparison reports.

Not allowed:

* changing wrapper logic for a specific biological dataset;
* changing IdPhi computation inside the wrapper for lab data;
* tuning VTE thresholds retrospectively from biological outcomes;
* hiding source-specific assumptions.

## Current adapters

### `crcns_wtrack`

Exploratory CRCNS W-track ingestion and segmentation diagnostics.

Implemented scope:

* MATLAB 5.0 file loading;
* CRCNS file inventory;
* position probe;
* task probe;
* canonical trace export at epoch level;
* inferred W-track geometry diagnostics;
* trial segmentation diagnostics;
* segmentation QA tables and figures.

Current role:

```text
exploratory / reverse-proof candidate
```

The CRCNS W-track data are useful for future W-maze geometry reconstruction and out-of-domain testing. They are not yet treated as a direct Stage 3.2C gate-viscosity evidence dataset.

### `dandi_000115`

DANDI 000115 behavioral NWB adapter.

Implemented scope:

* NWB behavior probe;
* position series probe;
* behavioral event channel probe;
* StateScript attribute extraction;
* StateScript event parsing;
* StateScript-to-position alignment diagnostics;
* filtered arm-choice canonical trace export;
* frozen VTE wrapper compatibility.

Current role:

```text
event-centered biological proxy bridge
```

Patch 15A probes the NWB behavior layer.

Patch 15B exports arm-choice event-centered traces accepted by the frozen VTE wrapper.

DANDI 000115 is not treated as an explicit VTE-label benchmark. It currently supports biological event-centered VTE-proxy extraction, but not yet full gate-viscosity validation, because ambiguity, correctness, one-shot perturbation, and carryover semantics have not been fully reconstructed.

## Current dataset interpretation

### DANDI 000115

Supported:

* raw position;
* arm-choice beam events;
* reward pump events;
* StateScript task logs;
* event-centered canonical trace export;
* frozen wrapper metrics.

Not yet supported:

* explicit VTE labels;
* full task geometry equivalence;
* correctness/error semantics;
* ambiguity vs obvious condition labeling;
* one-shot perturbation;
* carryover after stimulus removal.

Classification:

```text
B/C technical bridge:
  choice + reward events are available;
  behavioral VTE-proxy extraction is possible;
  full gate-viscosity comparison is not established yet.
```

### CRCNS hc-6 / hc-28

Supported:

* position/task file ingestion;
* W-track trajectory diagnostics;
* exploratory segmentation.

Not yet supported:

* stable full W-maze choice semantics;
* direct Stage 3.2C gate-viscosity comparison;
* one-shot/carryover comparison.

Classification:

```text
C / reverse-proof candidate:
  useful for future W-maze reconstruction and model environment replication.
```

### Miles et al. 2021 / Mizumori VTE dataset

Status:

```text
access pending / data link recovery needed
```

Potential role:

```text
A/B candidate if data access is restored
```

This remains a high-priority target because it is closer to explicit VTE detection and choice behavior.

### Redish Lab OSF datasets

Status:

```text
inventory needed
```

Potential role:

```text
A/B/C depending on available files
```

These datasets should be inspected before extending the biological comparison layer with more code.

## Stage 3.2C stopping rule

Do not add new biological adapters unless the next dataset or patch improves at least one of the following:

* choice semantics;
* outcome semantics;
* uncertainty / ambiguity semantics;
* perturbation / reversal / carryover semantics;
* neural representation of competing alternatives;
* multi-subject robustness after endpoint eligibility is established.

Do not add converters only to increase dataset count.

## Related documents

* `vte/lab_adapters/BIOLOGICAL_COMPARABILITY.md`
* `docs/stage3_2c_biological_comparability_note.md`

