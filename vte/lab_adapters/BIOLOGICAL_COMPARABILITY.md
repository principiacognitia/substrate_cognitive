# Biological Comparability Contract

## Purpose

The lab adapter layer translates biological datasets into auditable comparison endpoints for the Stage 3 model.

It is not a visual similarity layer. It is not designed to prove that a synthetic trace and a biological trace have identical movement morphology.

The relevant model claim is gate-viscosity dynamics:

```text
gate viscosity
  -> delayed or unstable commitment under boundary conditions
  -> stable commitment under obvious conditions
  -> one-shot learning after strong events
  -> carryover after stimulus removal
```

VTE-like behavior is a secondary readout of this dynamic.

## Core rule

Do not compare VTE-like metrics unless the biological data contain a recoverable choice context.

A usable biological trace must preserve:

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
reward / outcome if available
condition / task state if available
```

## Endpoint classes

### Class A: Direct gate-viscosity comparison

A dataset is Class A if it contains:

- recoverable choice;
- available alternatives;
- reward, outcome, correctness, or error;
- uncertainty, conflict, reversal, perturbation, risk, or carryover condition;
- subject/session/trial structure.

Class A supports direct Stage 3.2C claims.

### Class B: Choice + outcome comparison

A dataset is Class B if it contains:

- recoverable choice;
- reward, outcome, correctness, or error;
- subject/session/trial structure;
- no clear ambiguity, perturbation, one-shot, or carryover manipulation.

Class B supports choice/outcome comparison, but not the full gate-viscosity claim.

### Class C: Choice-centered behavioral proxy

A dataset is Class C if it contains:

- recoverable choice events;
- tracking / position / pose samples around those events;
- enough event or geometry structure to define event-centered windows.

Class C supports VTE-proxy extraction only.

### Class D: Choice-linked neural proxy

A dataset is Class D if it contains:

- neural events or decoded representations;
- alignment to choice episodes;
- interpretable candidate alternatives;
- optional reward/outcome linkage.

Class D can support the hypothesis if neural activity reflects representational competition between alternatives.

### Reject

Reject for Stage 3.2C evidence if:

- no recoverable choice exists;
- no alternatives are recoverable;
- only rest/sleep replay is available;
- only fixed-route navigation is available;
- tracking cannot be aligned to event or trial structure;
- only summary figures or aggregate values are available;
- the adapter would need to change the frozen VTE wrapper to make the dataset fit.

## Allowed proxy metrics

### Behavioral trajectory proxy

Allowed only around choice episodes:

```text
raw_idphi
log_idphi
z_idphi
vte_binary
choice dwell
pause duration
reorientation count
trajectory tortuosity
switch/stay behavior
```

Absolute values are not necessarily cross-source comparable. Sample rate, window duration, smoothing, and coordinate source must be recorded.

### Choice/outcome proxy

Primary comparison variables:

```text
committed_path
reward
outcome
correct/error
switch/stay
perseveration
choice probability by condition
post-event choice shift
carryover after stimulus removal
```

### Neural proxy

Allowed if choice-linked:

```text
decoded candidate representation
candidate intensity
dominance margin
representation entropy
alternation count
replay event count
LFP or spike event marker
```

A spike, replay event, or LFP event is not a deliberation proxy by itself. It becomes useful only if aligned to choice alternatives and task context.

## Required metadata for any biological trace

Every adapter must write metadata documenting:

```text
dataset_id
source files
subject_id extraction rule
session_id extraction rule
position source
event source
alignment method
window definition
choice-event definition
reward-event definition
heading derivation
sample rate estimate
smoothing parameters
filters / refractory periods
known limitations
```

## Frozen wrapper rule

Adapters must transform biological data into the existing canonical trace schema.

They must not change the frozen Stage 3.2A/B VTE wrapper to improve biological fit.

Allowed:

- adapter-side filtering;
- adapter-side event selection;
- adapter-side heading derivation;
- adapter-side metadata and normalization;
- post-wrapper normalized analysis.

Not allowed:

- changing wrapper logic for a specific biological dataset;
- redefining IdPhi inside the wrapper for lab data;
- using biological outcomes to tune VTE thresholds retrospectively;
- hiding source-specific assumptions.

## Current adapter statuses

### DANDI 000115

Status:

```text
Patch 15A:
  behavioral NWB probe complete

Patch 15B:
  event-centered arm-choice canonical trace export complete
```

Evidence supported:

- biological trace can be exported into wrapper-readable form;
- arm-choice events can be filtered and windowed;
- reward pump events can be attached;
- frozen wrapper runs on the exported trace.

Evidence not yet supported:

- explicit VTE labels;
- full task geometry equivalence;
- ambiguity / obvious-condition contrast;
- one-shot perturbation / carryover semantics;
- direct proof of gate-viscosity dynamics.

Classification:

```text
B/C technical bridge
```

### CRCNS hc-6 / hc-28 W-track

Status:

```text
Patch 14:
  inventory, position probe, task probe, canonical trace, segmentation diagnostics
```

Evidence supported:

- exploratory trajectory and task-file ingestion;
- future geometry reverse-proof potential.

Evidence not yet supported:

- direct Stage 3.2C gate-viscosity comparison;
- stable full W-maze choice semantics;
- one-shot/carryover comparison.

Classification:

```text
C / reverse-proof candidate
```

### Miles et al. 2021

Status:

```text
access pending / link recovery needed
```

Potential:

```text
A/B candidate if raw data access is restored
```

Reason:

- explicit VTE detection context;
- likely closer to direct choice-deliberation comparison than DANDI 000115.

### Redish Lab OSF datasets

Status:

```text
inventory needed
```

Potential:

```text
A/B/C depending on available files
```

## Minimum eligibility table

Future dataset audits should produce:

```text
Table_BioDataset_Endpoint_Eligibility.csv
```

Suggested columns:

```text
dataset_id
paper
repository
access_status
file_formats
species
task_type
has_choice
has_alternatives
has_trial_structure
has_tracking
has_heading_or_orientation
has_reward
has_correct_error
has_condition_shift
has_perturbation
has_one_shot_or_carryover
has_neural_proxy
neural_proxy_type
geometry_reconstructable
wrapper_compatible
endpoint_class
recommended_priority
limitations
```

## Interpretation rules

Valid statement:

```text
This dataset supports event-centered biological VTE-proxy extraction around recoverable choice episodes.
```

Invalid statement:

```text
This dataset proves that the model animal performs biological VTE.
```

Valid statement:

```text
The adapter establishes a measurement bridge from biological choice events to frozen wrapper metrics.
```

Invalid statement:

```text
The model is biologically validated because raw IdPhi values resemble animal trajectories.
```

Valid statement:

```text
A neural replay or decoded representation can be used as a deliberation proxy if it is linked to choice alternatives and outcome.
```

Invalid statement:

```text
Any spike or replay event is evidence of deliberation.
```

## Stage 3.2C stopping rule

Do not extend biological code unless the next dataset or patch improves one of the following:

```text
choice semantics
outcome semantics
uncertainty / perturbation / carryover semantics
neural alternative-representation proxy
multi-subject robustness after endpoint eligibility is established
```

Do not add more converters only to increase the number of datasets.

## Next recommended actions

1. Close Patch 15B as a technical biological bridge.
2. Prioritize access recovery for Miles et al. 2021.
3. Inventory Redish OSF datasets.
4. Use DANDI 000115 for proxy and infrastructure evidence, not final gate-viscosity validation.
5. Return to Stage 3 manuscript text while stronger biological endpoint data are being located.