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

---

## Current claim boundary

The current Stage 3.2 biological layer supports:

```text
approximation of the theoretical VTE description
```

It does not yet support:

```text
direct synthetic behavior vs biological behavior equivalence
```

This distinction is required.

Stage 3.2 can say:

```text
The model produces VTE-like behavior in the theoretical sense used in the VTE literature:
pause / reorientation / delayed commitment under ambiguous choice.
```

Stage 3.2 cannot yet say:

```text
The synthetic agent reproduces rat VTE trajectories.
```

---

## Core rule

Do not compare VTE-like metrics unless the biological data contain a recoverable choice context.

A usable biological trace or decision-level endpoint must preserve:

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

---

## Endpoint classes

### Class A: Direct gate-viscosity comparison

A dataset is Class A if it contains:

- recoverable choice;
- available alternatives;
- reward, outcome, correctness, or error;
- uncertainty, conflict, reversal, perturbation, risk, or carryover condition;
- subject/session/trial structure;
- enough event alignment to compare behavior across condition changes.

Class A supports direct Stage 3.2C gate-viscosity claims.

Current status:

```text
No closed Stage 3.2 adapter fully satisfies Class A.
```

---

### Class B: Choice + outcome comparison

A dataset is Class B if it contains:

- recoverable choice;
- reward, outcome, correctness, or error;
- subject/session/trial structure;
- no clear ambiguity, perturbation, one-shot, or carryover manipulation.

Class B supports choice/outcome comparison, but not the full gate-viscosity claim.

Current examples:

```text
dandi_000115
selected redish_lra_2024 endpoints, depending on field availability
```

---

### Class C: Choice-centered behavioral proxy

A dataset is Class C if it contains:

- recoverable choice events;
- tracking / position / pose samples around those events;
- enough event or geometry structure to define event-centered windows.

Class C supports VTE-proxy extraction only.

Current examples:

```text
crcns_wtrack
future Miles/Mizumori trajectory extraction
future Stout trajectory reconstruction, if valid
```

---

### Class D: Choice-linked neural proxy

A dataset is Class D if it contains:

- neural events or decoded representations;
- alignment to choice episodes;
- interpretable candidate alternatives;
- optional reward/outcome linkage.

Class D can support the hypothesis if neural activity reflects representational competition between alternatives.

Current examples:

```text
future Redish / hippocampal sequence / mPFC-related analyses
future W-track neural-alternative representation analyses
```

A neural event is not a deliberation proxy by itself. It becomes useful only if aligned to choice alternatives and task context.

---

### Decision-level comparator

A decision-level comparator is not a full Class A biological trace.

It is a dataset endpoint that preserves:

- subject/session/trial structure;
- decision label or native action code;
- VTE-related variable or choice-point measure;
- condition label;
- enough metadata to prevent invalid label mapping.

Current example:

```text
redish_lra_2024 Patch 18C
```

This class is useful for article figures and theory-level comparison, but it must be labeled as decision-level.

---

### Reject

Reject for Stage 3.2 evidence if:

- no recoverable choice exists;
- no alternatives are recoverable;
- only rest/sleep replay is available;
- only fixed-route navigation is available;
- tracking cannot be aligned to event or trial structure;
- only summary figures or aggregate values are available;
- the adapter would need to change the frozen VTE wrapper to make the dataset fit;
- biological action labels would need to be silently renamed into synthetic labels;
- the source does not allow subject/session/trial provenance to be preserved.

---

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

Absolute values are not necessarily cross-source comparable. Sample rate, window duration, smoothing, coordinate source, and heading derivation must be recorded.

### Choice/outcome proxy

Primary comparison variables:

```text
committed_path
chosen_arm
chosen_route
native_action_code
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

---

## Required metadata for any biological endpoint

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
license / access status
citation requirements
author correspondence status if applicable
```

---

## Frozen wrapper rule

Adapters must transform biological data into the existing canonical trace schema or into an explicitly labeled decision-level endpoint.

They must not change the frozen Stage 3.2A/B VTE wrapper to improve biological fit.

Allowed:

- adapter-side filtering;
- adapter-side event selection;
- adapter-side heading derivation;
- adapter-side metadata and normalization;
- post-wrapper normalized analysis;
- decision-level endpoint export when trajectory replay is invalid.

Not allowed:

- changing wrapper logic for a specific biological dataset;
- redefining IdPhi inside the wrapper for lab data;
- using biological outcomes to tune VTE thresholds retrospectively;
- hiding source-specific assumptions;
- mixing healthy and perturbed conditions without labels;
- silently mapping biological action codes to synthetic left/right labels;
- presenting decision-level rows as biological movement traces.

---

## Current adapter statuses

### CRCNS W-track / hc-6 / hc-28

Status:

```text
Patch 14 family:
  inventory, position probe, task probe, canonical trace, segmentation diagnostics
```

Evidence supported:

- exploratory trajectory and task-file ingestion;
- W-track geometry reverse-proof potential;
- future W-maze environment reconstruction.

Evidence not yet supported:

- direct Stage 3.2 gate-viscosity comparison;
- stable full W-maze choice semantics;
- one-shot/carryover comparison;
- direct synthetic-left / biological-right panels.

Classification:

```text
Class C / reverse-proof candidate
```

---

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
Class B/C technical bridge
```

---

### Redish LRA 2024

Status:

```text
Patch 17A-17H:
  inventory, HDF5 reference resolution, canonical choice extraction, policy split

Patch 18A:
  biosynthetic comparability construction

Patch 18B:
  validation

Patch 18C:
  decision-level visualization bridge
```

Evidence supported:

- healthy-control baseline extraction;
- strict same-trial action-code subset;
- separation of healthy-control and mPFC-DREADD conditions;
- validated comparable decision rows;
- decision-level visualization endpoint;
- preservation of biological native action codes.

Evidence not supported:

- biological movement replay;
- native action-code equivalence to synthetic left/right;
- healthy + DREADD pooling without explicit condition labels;
- rodent trajectory equivalence.

Classification:

```text
Decision-level comparator
Primary closed Stage 3.2 biological endpoint
```

Allowed use:

```text
theory-level and decision-level comparison
```

Disallowed use:

```text
direct synthetic trajectory vs biological trajectory equivalence
```

---

### Redish RROW 2022

Status:

```text
Patch 16/18 family:
  inventory / negative endpoint
```

Evidence supported:

- task relevance;
- future value/conflict comparison potential.

Evidence not yet supported:

- simple fork/open-closed comparability;
- current Stage 3 geometry equivalence;
- current Stage 3 one-shot protocol equivalence.

Classification:

```text
future task-family candidate
not current direct comparator
```

Reason:

RROW-like tasks involve multiple simultaneous factors: offer value, route structure, alternation or sequence demands, reward history, procedural learning, and deliberation. A synthetic RROW analogue should be a new task class, not a small Stage 3.2 extension.

---

### Stout 2022 / Griffin Lab

Status:

```text
Patch 19 family:
  source inspection, variable inventory, trajectory recovery attempt, endpoint classification
```

Evidence supported:

- VTE-related trial-level variables;
- published analysis scripts;
- candidate position fields;
- possible future trace reconstruction.

Evidence not yet supported:

- reliable full trial-level trajectory reconstruction;
- direct movement comparator;
- current primary biological baseline status.

Classification:

```text
diagnostic VTE-variable endpoint
```

---

### Miles et al. 2021 / Mizumori

Status:

```text
author access response received
manual VTE annotation benchmark candidate
```

Evidence supported:

- explicit VTE-detection context;
- manually annotated behavioral material;
- future trajectory-shape or classifier benchmark potential.

Evidence not yet supported in the closed Stage 3.2 pass:

- direct task equivalence;
- current open/covered or LRA decision endpoint equivalence;
- integrated adapter.

Classification:

```text
future Class A/B/C candidate pending manual audit
```

---

## Permission and correspondence policy

Author correspondence may establish practical access permission, but it does not replace formal dataset licenses, repository terms, or publication citation requirements.

Every use of correspondence-enabled data must record:

```text
contacted_author
response_date
access_method
allowed_use_summary
citation_requested
license_or_repository_terms
```

Known correspondence outcomes:

```text
Mizumori / Miles 2021:
  access restored or provided;
  use requires citation of Miles et al. 2021 and acknowledgement of data source.

Griffin / Stout 2022:
  instructions/source access provided;
  use requires citation of Stout et al. 2022 and acknowledgement of repository/source data.

Redish Lab:
  public OSF source and related publications must be cited;
  future requests should be specific to trial-level or trajectory-level fork/open-closed VTE data.
```

---

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
exclusion_reason
next_action
```

---

## Synthetic-left / biological-right readiness

A dataset is ready for side-by-side comparison only if it supports one of the following:

### Decision-level panel

Required:

- trial-level biological endpoint;
- VTE or VTE-like variable;
- choice/outcome labels;
- subject/session identifiers;
- condition labels;
- explicit statement that no trajectory replay is implied.

Example:

```text
Patch 18C Redish LRA decision-level panel
```

### Behavioral trajectory panel

Required:

- synchronized x/y/t samples around choice point;
- recoverable heading or derivable heading;
- recoverable choice-point window;
- committed path or chosen arm;
- trial/session/subject identifiers;
- stable geometry registry;
- post-wrapper metrics computed without modifying wrapper.

No current closed biological adapter fully satisfies this class.

### Neural comparison panel

Required:

- neural data aligned to choice point;
- interpretable alternatives;
- trial-level choice/outcome structure;
- clear variable mapping, such as decoded candidate representation or sequence content.

No current closed Stage 3.2 adapter fully satisfies this class.

---

## Stage 3.2 biological conclusion

The biological adapter layer has reached a useful but limited closure state.

It supports:

```text
The model reproduces a theoretical VTE-like behavioral profile under ambiguous choice.
```

It does not yet support:

```text
The synthetic agent reproduces biological rat behavior trajectory-by-trajectory.
```

Future work should therefore aim at directly synchronized comparability data:

```text
synthetic left panel / biological behavioral right panel
synthetic left panel / neurobiological right panel
```

That work belongs in:

```text
docs/stage3_2_biological_dataset_shortlist.md
docs/stage3_2_biological_dataset_TBD.md
```

not in the current Stage 3.2 closure claim.

---

## References

Gillespie, A. K., Astudillo Maya, D. A., Denovellis, E. L., Liu, D. F., Kastner, D. B., Coulter, M. E., Roumis, D. K., Eden, U. T., & Frank, L. M. (2021). Hippocampal replay reflects specific past experiences rather than a plan for subsequent choice. *Neuron, 109*(19), 3149-3163.e6. https://doi.org/10.1016/j.neuron.2021.07.029

Jadhav, S. P., & Frank, L. M. (2020). *Simultaneous extracellular recordings from hippocampal area CA1 and medial prefrontal cortex from rats performing a W-track alternation task*. CRCNS.org. https://doi.org/10.6080/K02N50G9

Miles, J. T., Kidder, K. S., Wang, Z., Zhu, Y., Gire, D. H., & Mizumori, S. J. Y. (2021). A machine learning approach for detecting vicarious trial and error behaviors. *Frontiers in Neuroscience, 15*, Article 676779. https://doi.org/10.3389/fnins.2021.676779

Mugan, U., Amemiya, S., Regier, P. S., & Redish, A. D. (2023). *Navigation through the complex world: The neurophysiology of decision-making processes*. arXiv. https://doi.org/10.48550/arXiv.2306.03162

Redish, A. D. (2016). Vicarious trial and error. *Nature Reviews Neuroscience, 17*(3), 147-159. https://doi.org/10.1038/nrn.2015.30

Redish, A. D. (2024). *2024 RedishLab: Recordings from medial prefrontal, dorsolateral striatum, and hippocampus on Left-Right-Alternate; DREADD disruption of mPFC*. OSF Project c4fjm.

Schmidt, B., Duin, A. A., & Redish, A. D. (2019). Disrupting the medial prefrontal cortex alters hippocampal sequences during deliberative decision making. *Journal of Neurophysiology, 121*(5), 1981-2000. https://doi.org/10.1152/jn.00793.2018

Schmidt, B., & Redish, A. D. (2021). Disrupting the medial prefrontal cortex with designer receptors exclusively activated by designer drug alters hippocampal sharp-wave ripples and their associated cognitive processes. *Hippocampus, 31*(11), 1219-1235. https://doi.org/10.1002/hipo.23367

Stout, J. J., Hallock, H. L., George, A. E., Adiraju, S. S., & Griffin, A. L. (2022). The ventral midline thalamus coordinates prefrontal-hippocampal neural synchrony during vicarious trial and error. *Scientific Reports, 12*, Article 10940. https://doi.org/10.1038/s41598-022-14707-8