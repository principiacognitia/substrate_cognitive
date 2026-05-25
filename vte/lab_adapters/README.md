# VTE Lab Adapters

This directory contains biological-data adapters used by the Stage 3.2 biological comparability layer.

The adapter layer is not part of the Stage 3 cognitive model. It does not change the Gate, the Stage 3 environment, the VTE wrapper, or the Stage 3.2 statistical tests.

Its role is narrower:

1. inspect external laboratory datasets;
2. translate usable biological records into auditable comparison endpoints;
3. document what kind of comparison is valid;
4. reject datasets or fields that would force the frozen wrapper to change.

---

## Closed Stage 3.2 interpretation boundary

Stage 3.2 supports approximation of the **theoretical description of VTE-like deliberation**, not direct equivalence between synthetic and biological behavior.

Supported claim:

```text
A viscous Gate can generate behavior that matches the theoretical VTE profile:
  - delayed or unstable commitment under ambiguous choice;
  - stable commitment under obvious choice;
  - pause / reorientation / IdPhi-like signatures at choice points;
  - one-shot valence deformation;
  - carryover after stimulus removal.
```

Unsupported claim:

```text
Synthetic Stage 3 trajectories are not yet directly matched to biological rat trajectories.
```

The current biological layer therefore supports **decision-level**, **schema-level**, and **theory-level** comparison. It does not support rodent-level trajectory equivalence, allocentric spatial cognition, neural mechanism identity, or direct movement replay.

---

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
```

VTE-like behavior is treated as a secondary behavioral readout of this dynamic.

Biological VTE-like metrics are useful only when they are linked to recoverable choice episodes.

---

## Biological comparability hierarchy

### Primary endpoints

The strongest biological comparison targets are choice and outcome variables:

- chosen arm / route / path;
- available alternatives;
- reward / outcome;
- correctness or error when recoverable;
- switch / stay;
- perseveration;
- condition-wise shift;
- reversal, perturbation, risk, conflict, or carryover when recoverable.

### Secondary behavioral endpoints

VTE-like trajectory measures are secondary endpoints:

- choice-point dwell;
- pause duration;
- reorientation count;
- IdPhi-like angular displacement;
- trajectory tortuosity;
- head-scanning / hesitation proxies.

These metrics should not be interpreted outside a choice context.

### Neural proxy endpoints

Neural data can be used as a deliberation proxy only when aligned to choice episodes and interpretable alternatives.

Potential neural proxies:

- decoded candidate-arm or candidate-route representations;
- replay content around a choice point;
- alternating representation of competing options;
- dominance margin between alternatives;
- representation entropy;
- replay event count;
- LFP or spike events with explicit relation to choice state.

A spike, replay event, or LFP event is not sufficient by itself. It becomes relevant only when linked to task alternatives and choice/outcome structure.

---

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

---

## Frozen wrapper rule

Lab adapters must transform biological data into the existing canonical VTE trace schema or into a clearly labeled decision-level endpoint.

They must not modify the frozen Stage 3.2A/B VTE wrapper to make a biological dataset fit.

Allowed:

- adapter-side event filtering;
- adapter-side canonical trace export;
- adapter-side heading derivation;
- adapter-side reward/event attachment;
- post-wrapper normalization;
- post-wrapper eligibility and comparison reports;
- decision-level endpoint construction when trajectory replay is not valid.

Not allowed:

- changing wrapper logic for a specific biological dataset;
- changing IdPhi computation inside the wrapper for lab data;
- tuning VTE thresholds retrospectively from biological outcomes;
- silently mapping biological native action codes to synthetic left/right labels;
- treating decision-level endpoints as movement traces;
- hiding source-specific assumptions.

---

## Current adapters and closure status

### `crcns_wtrack`

Exploratory CRCNS W-track ingestion and segmentation diagnostics.

Patch status:

```text
Patch 14 family: closed exploratory endpoint
```

Implemented scope:

- MATLAB 5.0 file loading;
- CRCNS file inventory;
- position probe;
- task probe;
- canonical trace export at epoch level;
- inferred W-track geometry diagnostics;
- trial segmentation diagnostics;
- segmentation QA tables and figures.

Current role:

```text
Class C / reverse-proof candidate
```

Interpretation:

CRCNS W-track data are useful for future W-maze geometry reconstruction and out-of-domain testing. They are not treated as direct Stage 3.2 gate-viscosity evidence because stable full W-maze choice semantics, one-shot/carryover semantics, and direct Stage 3 task equivalence have not been established.

---

### `dandi_000115`

DANDI 000115 behavioral NWB adapter.

Patch status:

```text
Patch 15A: NWB behavior probe complete
Patch 15B: event-centered arm-choice canonical trace export complete
```

Implemented scope:

- NWB behavior probe;
- position series probe;
- behavioral event channel probe;
- StateScript attribute extraction;
- StateScript event parsing;
- StateScript-to-position alignment diagnostics;
- filtered arm-choice canonical trace export;
- frozen VTE wrapper compatibility.

Current role:

```text
Class B/C technical bridge
```

Interpretation:

DANDI 000115 shows that biological event-centered arm-choice data can be translated into a wrapper-readable trace. It is not treated as an explicit VTE-label benchmark and does not yet support full gate-viscosity validation because ambiguity, correctness, one-shot perturbation, and carryover semantics have not been reconstructed.

---

### `redish_rrow_2022`

Redish Lab Restaurant Row / RROW-related adapter family.

Patch status:

```text
Patch 16/18 family: closed inventory / negative endpoint
```

Implemented scope:

- dataset inventory;
- task-structure inspection;
- candidate endpoint search;
- biological comparability assessment;
- exclusion from the current direct comparison set.

Current role:

```text
Out-of-scope for current Stage 3.2 direct comparison
```

Interpretation:

RROW is conceptually relevant because it involves competing values, deliberation, and route/offer decisions. It is not a small extension of the current Stage 3 configuration. A faithful synthetic analogue would require richer task geometry, offer structure, reward history, and sequence policy. RROW remains a future candidate, not a current direct comparator.

---

### `redish_lra_2024`

Redish Lab Left-Right-Alternate / DREADD mPFC dataset adapter.

Patch status:

```text
Patch 17A-17H: inventory, extraction, policy split, strict endpoint filtering
Patch 18A: BioSynth comparability table
Patch 18B: validation
Patch 18C: decision-level visualization bridge
Closed as primary Stage 3.2 biological decision-level endpoint
```

Implemented scope:

- OSF dataset inventory;
- HDF5 reference resolution;
- canonical choice endpoint extraction;
- healthy-control vs mPFC-DREADD policy split;
- same-trial action-code filtering;
- BioSynth comparability table construction;
- validation of comparable rows;
- decision-level visualization endpoint export.

Recommended outputs:

```text
redish_lra17e_healthy_control_baseline.csv
redish_lra17h_healthy_choice_baseline.csv
Table_LRA_BioSynth18B_Comparable_Rows_Validated.csv
Table_LRA_BioSynth18C_Visualization_Decision_Endpoint.csv
```

Current role:

```text
Primary biological decision-level comparator
```

Interpretation boundary:

- healthy-control LRA rows may be used as the baseline comparator;
- mPFC-DREADD rows must not be mixed into the healthy baseline;
- native biological action codes must not be silently renamed into synthetic left/right labels;
- Patch 18C is a decision-level visualization bridge, not a biological trajectory replay layer;
- `movement_trace_available = false` means that Patch 18C outputs cannot be used as animation input.

This adapter supports theory-level and decision-level comparison with VTE-like behavior. It does not support direct synthetic-vs-biological movement equivalence.

---

### `stout_2022`

Griffin Lab / Stout et al. VTE-related dataset adapter.

Patch status:

```text
Patch 19 family: closed diagnostic endpoint
```

Implemented scope:

- repository/source inspection;
- MATLAB data probe;
- VTE-related variable inventory;
- trial-level endpoint assessment;
- trajectory-field recovery attempt;
- comparability classification.

Relevant variables inspected include:

```text
zIdPhi
IdPhi
oopsTrials
accuracy
turnDirection
timeSpent_CP
tsPosOG
xPosOG
yPosOG
```

Current role:

```text
Diagnostic VTE-related endpoint
```

Interpretation:

The dataset contains useful VTE-related trial-level variables and published analysis scripts. However, full trial-level trajectory reconstruction from `tsPosOG`, `xPosOG`, and `yPosOG` was not established as a reliable direct movement comparator in the closed Stage 3.2 pass. It remains useful for VTE-variable diagnostics and future trace-reconstruction work, but it is not used as the primary direct behavioral trajectory comparator.

---

## Author correspondence and permissions

Some dataset access paths were clarified through direct author correspondence. These permissions and responses are project correspondence, not replacements for dataset licenses or repository terms.

### Mizumori / Miles et al. 2021

Professor Sheri Mizumori responded to the access request and restored or provided access to the VTE-related behavioral data and accompanying notes.

Current interpretation:

```text
manually annotated VTE benchmark candidate
```

Use boundary:

- useful for future trajectory-based validation or VTE-classifier checks;
- not the current primary comparator because the task format is not directly matched to the Stage 3 open/covered or LRA decision setting;
- if used, cite Miles et al. (2021) and acknowledge the data source.

### Griffin / Stout et al. 2022

Professor Amy Griffin responded to the access request and forwarded instructions pointing to the source data and scripts associated with Stout et al. (2022).

Current interpretation:

```text
diagnostic VTE-variable endpoint
```

Use boundary:

- useful for VTE-related variables and possible future trace extraction;
- not yet a direct trajectory comparator;
- if used, cite Stout et al. (2022) and acknowledge the source repository/data.

### Redish Lab

Redish Lab sources are treated under public dataset/source availability and formal citation boundaries.

Current interpretation:

```text
primary closed decision-level biological endpoint: redish_lra_2024
future request target: fork / open-closed / lifted-maze VTE trajectories, if available
```

Use boundary:

- cite the OSF dataset record and related Redish Lab publications;
- do not imply direct rodent trajectory equivalence from decision-level endpoints;
- do not mix healthy-control and mPFC-DREADD rows without explicit condition labeling.

---

## Closed Stage 3.2 adapter conclusion

The current adapter layer establishes that biological data can be mapped into auditable comparison endpoints, but the strongest current Stage 3.2 claim is:

```text
The model approximates the theoretical behavioral profile attributed to biological agents under ambiguous choice and VTE-like deliberation.
```

The current adapter layer does not yet establish:

```text
direct synthetic trajectory ↔ biological trajectory equivalence
```

That future comparison requires datasets with synchronized position traces, recoverable choice-point windows, heading or derivable heading, choice labels, and outcome/condition metadata.

---

## Future biological dataset work

Future work should be tracked outside this adapter README in:

```text
docs/stage3_2_biological_dataset_shortlist.md
docs/stage3_2_biological_dataset_TBD.md
```

The next dataset pass should prioritize fit to the question:

```text
VTE traces at a recoverable choice point
```

not merely public availability.

Request-access datasets should remain eligible. Stage 3.2 correspondence showed that author contact is a viable access path when the request is specific and bounded.

---

## Stage 3.2C stopping rule

Do not add new biological adapters unless the next dataset or patch improves at least one of the following:

- choice semantics;
- outcome semantics;
- uncertainty / ambiguity semantics;
- perturbation / reversal / carryover semantics;
- neural representation of competing alternatives;
- multi-subject robustness after endpoint eligibility is established;
- direct synthetic-left / biological-right comparison readiness.

Do not add converters only to increase dataset count.

---

## Related documents

- `vte/lab_adapters/BIOLOGICAL_COMPARABILITY.md`
- `docs/STAGE3_2_CLOSURE.md`
- `docs/stage3_2_TBD.md`
- `docs/stage3_2_biological_dataset_shortlist.md`
- `docs/stage3_2_biological_dataset_TBD.md`

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