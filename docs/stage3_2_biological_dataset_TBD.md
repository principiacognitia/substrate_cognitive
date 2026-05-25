# Stage 3.2 Biological Dataset TBD

## Purpose

This file records manual-audit notes for biological datasets considered after Stage 3.2 closure.

The closed Stage 3.2 result is not direct biological behavior matching. It is:

```text
approximation of the theoretical VTE description by a viscous Gate under ambiguous choice
```

The datasets below are candidates for future work toward direct comparability:

```text
synthetic behavioral left / biological behavioral right
synthetic gate-dynamics left / biological neural right
```

No dataset in this file is required for the current Stage 3.2 closure or for the first Stage 3 article draft.

---

## Minimum eligibility table for future audits

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

## Common first-pass audit rule

For large electrophysiology datasets, do not start with spikes or LFP.

Start with:

```text
task files
position files
event files
metadata / XML / schema files
```

Only after geometry, event structure, and choice windows are recoverable should neural data be loaded.

---

## Redish Lab LRA 2024 OSF

### Status

Closed Stage 3.2 primary biological decision-level endpoint.

### Dataset fit

Redish Lab LRA provides the strongest closed Stage 3.2 biological endpoint so far. It supports decision-level and schema-level comparability, especially through healthy-control LRA rows.

### Current role

```text
Primary biological decision-level comparator
```

### Boundary

Allowed:

- healthy-control baseline;
- decision-level biological endpoint;
- preservation of native biological action codes;
- explicit condition separation.

Not allowed:

- movement replay;
- rodent trajectory equivalence;
- pooling healthy-control and mPFC-DREADD rows without labels;
- silently renaming biological action codes into synthetic left/right labels.

### Next action

Keep as canonical closed endpoint. Reopen only if raw trajectory-level biological data can be obtained or extracted.

---

## Miles et al. 2021 / Mizumori VTE dataset

### Status

Author access response received. Future high-priority audit.

### Dataset fit

This is one of the best candidates because it is explicitly VTE-oriented and includes manually annotated behavioral material.

### Current classification

```text
A/B/C pending manual audit
```

### Possible endpoints

- manual VTE labels;
- task episodes;
- subject/session/trial metadata;
- possible trajectory material;
- VTE-classifier benchmark variables.

### Reason not yet closed as Stage 3.2 comparator

Task equivalence, trajectory availability, geometry, and trial schema still require manual audit.

### Next audit questions

1. What files are available after restored/provided access?
2. Are VTE labels trial-level, frame-level, or episode-level?
3. Are x/y trajectories available?
4. Is heading available or derivable?
5. Are choice points and alternatives recoverable?
6. Are outcomes available?
7. Can the data support a side-by-side synthetic/biological behavioral panel?
8. What citation and acknowledgement language is requested?

---

## Stout et al. 2022 / Griffin Lab

### Status

Closed diagnostic endpoint in Patch 19; future re-audit possible.

### Dataset fit

The dataset contains useful VTE-related trial-level variables and published source code. It is not currently a reliable direct trajectory comparator.

### Relevant variables inspected

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

### Current classification

```text
diagnostic VTE-variable endpoint
```

### Reason not promoted to primary comparator

Full trial-level trajectory reconstruction from the candidate position fields was not established in the closed Stage 3.2 pass.

### Next audit questions

1. Can `tsPosOG`, `xPosOG`, and `yPosOG` be mapped to trial windows without ambiguity?
2. Are choice-point windows explicit?
3. Are IdPhi values already computed from the same raw traces?
4. Are manual or algorithmic VTE labels included?
5. Can a subset be used without changing the frozen wrapper?
6. What repository and paper citations are required?

---

## CRCNS pfc-8 — Oberto / Wiener / Zugaro T-maze rule switching

### Status

Manual audit started. Strong B-list candidate.

### Dataset fit

CRCNS pfc-8 contains simultaneous single-unit and LFP recordings from medial prefrontal cortex, ventral striatum, and dorsomedial striatum in rats performing flexible rule switching on the same T-maze.

The task structure is:

```text
visual discrimination
  -> spatial discrimination
  -> visual discrimination again
```

This makes the dataset especially relevant for gate-mode arbitration and flexible-control comparison.

### Available endpoints

Potentially available:

- T-maze position traces;
- head-mounted LED coordinates;
- task events;
- visual cue events;
- left/right spatial-discrimination rule;
- visual-discrimination rule;
- rule-switching epochs;
- sorted spikes;
- LFP;
- mPFC units;
- vSTR units;
- dmSTR units;
- pre-task and post-task sleep sessions.

### Current classification

```text
Class B/C/D
Priority B1
```

### Reason not promoted to A-list

The dataset does not provide explicit VTE labels, IdPhi annotations, or published head-sweep/VTE endpoints.

### Access / permission note

CRCNS access is available after registration, but publication use requires contacting Sidney Wiener or Michaël Zugaro to discuss potential authorship.

### Next audit questions

1. What event codes are present in `.all.evt`, `.cue.evt`, and `.cat.evt`?
2. Do event files identify trial start, cue onset, rule block, arm entry, reward, and error?
3. Can the `.pos` file reconstruct the T-maze geometry reliably?
4. Is there a stable choice-point zone?
5. Can left/right choice be inferred from position after the junction?
6. Can visual-rule and spatial-rule trials be separated cleanly?
7. Are error trials available and usable as conflict / uncertainty proxies?
8. Can dwell/pause before commitment be measured at the junction?
9. Can turn instability or head-direction changes be derived from LED coordinates?
10. Can mPFC/vSTR/dmSTR spikes be aligned to the same choice windows?
11. Should pre/post sleep be excluded from current behavioral comparison?
12. What exact contact requirements apply before public reuse?

### Possible future use

Behavioral comparison:

```text
synthetic choice-point dwell / delayed commitment
vs
biological T-maze dwell / pause / commitment under visual-spatial rule switching
```

Neural comparison:

```text
synthetic gate-state switching
vs
mPFC-striatal assembly dynamics around rule-guided choice
```

---

## CRCNS hc-13 — Yu / Frank CA1–mPFC foraging alternate-choice dataset

### Status

Manual audit started. Strong B-list candidate, not yet A-list.

### Dataset fit

CRCNS hc-13 contains simultaneous extracellular recordings from dorsal CA1 and dorsal medial prefrontal cortex in rats performing a foraging task. The main behavioral structure appears to involve alternate-choice behavior in a foraging context.

### Available files

Per animal:

```text
<animal>rawpos.mat
<animal>task.mat
<animal>spikes.mat
<animal>eeg.mat
<animal>cellinfo.mat
<animal>tetinfo.mat
```

Animals observed in downloaded archive:

```text
T1
S2
Q1
R1
N2
```

### Potential endpoints

Behavioral:

- x/y position;
- epoch-level task metadata;
- inferred choice zones;
- inferred foraging choice episodes;
- dwell / pause around candidate choice zone;
- speed and immobility if derivable from raw position;
- committed route / arm if geometry supports zone classification.

Neural:

- CA1 spikes;
- dorsal mPFC spikes;
- LFP/eeg;
- movement vs immobility neural states;
- possible choice-aligned neural activity;
- possible candidate replay / sequence proxy if later analysis supports it.

### Current classification

```text
Class C/D, potentially B after audit
Priority B2
```

### Reason not promoted to A-list

The dataset does not currently expose explicit VTE labels, ready-made choice-point windows, or turnkey trial-level choice/outcome tables.

### Next audit questions

1. What fields are present in each `<animal>task.mat`?
2. Does `task{day}{epoch}` contain arm choice, reward, correctness, or event definitions?
3. What are the columns in `rawpos{day}{epoch}.data`?
4. Can the foraging maze geometry be reconstructed from x/y plots?
5. Is there a stable task-defined junction or choice zone?
6. Can dwell/pause episodes be detected before committed movement into a route?
7. Can choices be inferred from first stable zone/arm occupancy after leaving the candidate choice zone?
8. Is reward/outcome available directly, or must it be inferred from task rule?
9. Are CA1/mPFC spikes alignable to the same choice windows?
10. Is the position sampling sufficient for IdPhi-like or reorientation metrics?

### Possible future use

```text
synthetic Stage 3 choice-point trace
vs
biological free-moving rat foraging choice trace
```

and separately:

```text
synthetic gate-state instability / dwell proxy
vs
CA1–mPFC activity around biological choice / immobility episodes
```

---

## CRCNS hc-5 — Buzsáki / Pastalkova / Mizuseki / Wang left-right alternation dataset

### Status

Manual audit started. Not a primary Stage 3.2 VTE-trace comparator.

### Dataset fit

CRCNS hc-5 contains simultaneous extracellular recordings from left and right hippocampal CA1 and right entorhinal cortex in one rat performing left/right alternation, wheel running, and platform exploration tasks.

### Available endpoints

Potentially available:

- left/right alternation sessions;
- position traces;
- linearized track position;
- maze section labels;
- lap IDs;
- left/right direction choice;
- correct-choice labels;
- head direction;
- speed and acceleration;
- spike timing;
- LFP;
- theta phase;
- hippocampal and entorhinal unit metadata.

### Current classification

```text
Class C/D
Priority B3 or C1
```

### Reason not promoted to primary VTE shortlist

The dataset was not designed as a VTE-label or ambiguous-choice VTE dataset. It contains one animal and several task types, and its primary publications focus on hippocampal/entorhinal sequences and theta organization.

### Possible future use

Useful for alternation / sequence-memory comparison:

```text
synthetic gate/choice dynamics around alternation point
vs
hippocampal/entorhinal sequence dynamics around biological alternation events
```

### Next audit questions

1. Which sessions contain clean left/right alternation?
2. Can `mazeSect`, `lapID`, `dirChoice`, and `corrChoice` reconstruct trial-level choice episodes?
3. Is there a stable choice-point region in x/y or linearized coordinates?
4. Can heading and speed be extracted around the decision window?
5. Are error trials usable as ambiguity or conflict proxies?
6. Can neural sequence variables be aligned to the same choice window?
7. Does the access path allow reuse in a public comparison artifact?

---

## DANDI:001371 — Prince / Singer Lab update-task VR Y-maze

### Status

Manual audit started. Not a primary Stage 3.2 VTE-trace comparator.

### Dataset fit

The dataset contains head-fixed mouse VR Y-maze behavior with CA1 and mPFC electrophysiology. The task is a memory-based update task: animals first receive an original cue, maintain the goal arm, and on update trials receive a second cue requiring switch or stay behavior.

### Available endpoints

- trial-level choice;
- correctness;
- turn type;
- update type;
- update and choice times;
- VR position;
- view angle;
- translational and rotational velocity;
- licks and rewards;
- CA1/mPFC LFP, units, and event intervals.

### Current classification

```text
Class B/D
Optional C-lite if view-angle windows prove usable
```

### Reason not promoted to primary VTE shortlist

The dataset does not provide explicit VTE labels, IdPhi/head-sweep annotations, or natural choice-point VTE behavior. It is closer to memory update and prospective coding than to Stage 3.2 VTE-like trail comparison.

### Possible future use

Useful for future Gate-update / prospective-code comparison:

```text
synthetic gate state around new information
vs
biological CA1/mPFC prospective-code adaptation around update cue
```

---

## DANDI 000115

### Status

Closed technical bridge.

### Dataset fit

DANDI 000115 shows that biological NWB event-centered arm-choice data can be translated into a wrapper-readable trace.

### Current classification

```text
Class B/C technical bridge
```

### Reason not primary

The dataset is not an explicit VTE-label benchmark and does not yet provide ambiguity, one-shot perturbation, or carryover semantics.

### Future use

Keep as proof that NWB event-centered behavior can be brought into the Stage 3.2 schema.

---

## CRCNS hc-6 / hc-28 W-track family

### Status

Future W-maze / W-track audit candidate.

### Dataset fit

Potentially relevant because W-track alternation can provide repeated choice points, route structure, and hippocampal/prefrontal neural data.

### Current classification

```text
Class B/C/D pending audit
```

### Next audit questions

1. Are choice points and route alternatives recoverable?
2. Are trial starts/stops and rewards explicit?
3. Is there a stable geometry registry?
4. Can alternation correctness be reconstructed?
5. Is there sufficient head direction or derivable heading?
6. Can dwell/pause/reorientation windows be measured?
7. Can neural events be aligned to choice windows?

---

## Figshare 10248158 — Rtrack water maze tracking data

### Status

Manual audit started. Not a Stage 3.2 VTE-trace comparator.

### Dataset fit

The dataset contains Morris water maze tracking data with training and reversal structure. It is useful for trajectory reconstruction, path tortuosity, search dynamics, and reversal adaptation.

### Current classification

```text
Class C
```

### Reason not promoted to VTE shortlist

The Morris water maze does not provide a discrete fork, T-maze, W-maze, or route-choice point. Deliberation, if present, is distributed across search behavior rather than localized at a recoverable junction.

### Possible future use

```text
synthetic spatial search / reversal trajectory
vs
biological water maze reversal trajectory
```

---

## Redish RROW / Restaurant Row family

### Status

Future task-family candidate, not current direct comparator.

### Dataset fit

RROW is conceptually relevant because it involves competing values, deliberation, reward history, and route/offer decisions.

### Reason not current Stage 3.2 work

A faithful synthetic RROW analogue would require richer task geometry, offer structure, reward history, sequence policy, and procedural constraints.

### Recommendation

Do not treat as a small adapter patch. Treat as a future task family.

---

## Final Stage 3.2 biological conclusion

The current closed biological layer supports:

```text
The model approximates a theoretical VTE-like behavioral profile under ambiguous choice.
```

It does not yet support:

```text
The synthetic agent reproduces biological rat behavior trajectory-by-trajectory.
```

---

## References

Gillespie, A. K., Astudillo Maya, D. A., Denovellis, E. L., Liu, D. F., Kastner, D. B., Coulter, M. E., Roumis, D. K., Eden, U. T., & Frank, L. M. (2021). Hippocampal replay reflects specific past experiences rather than a plan for subsequent choice. *Neuron, 109*(19), 3149–3163.e6. https://doi.org/10.1016/j.neuron.2021.07.029

Miles, J. T., Kidder, K. S., Wang, Z., Zhu, Y., Gire, D. H., & Mizumori, S. J. Y. (2021). A machine learning approach for detecting vicarious trial and error behaviors. *Frontiers in Neuroscience, 15*, Article 676779. https://doi.org/10.3389/fnins.2021.676779

Oberto, V. J., Gao, H. Y., & Wiener, S. I. (2021). *Single unit and LFP recordings of medial prefrontal cortex and ventral and medial striatum of rats alternating between visual and spatial discrimination tasks in a T-maze* [Data set]. CRCNS.org. https://doi.org/10.6080/K0R20ZKP

Overall, R., & Kempermann, G. (2020). *Water maze tracking data* [Data set]. Figshare. https://doi.org/10.6084/m9.figshare.10248158

Pastalkova, E., Wang, Y., Mizuseki, K., & Buzsáki, G. (2015). *Simultaneous extracellular recordings from left and right hippocampal areas CA1 and right entorhinal cortex from a rat performing a left / right alternation task and other behaviors* [Data set]. CRCNS.org. https://doi.org/10.6080/K0KS6PHF

Redish, A. D. (2024). *2024 RedishLab: Recordings from medial prefrontal, dorsolateral striatum, and hippocampus on Left-Right-Alternate; DREADD disruption of mPFC*. OSF Project c4fjm.

Stout, J. J., Hallock, H. L., George, A. E., Adiraju, S. S., & Griffin, A. L. (2022). The ventral midline thalamus coordinates prefrontal–hippocampal neural synchrony during vicarious trial and error. *Scientific Reports, 12*, Article 10940. https://doi.org/10.1038/s41598-022-14707-8

Yu, J. Y., Liu, D., Grossrubatscher, I., & Frank, L. M. (2017). *Simultaneous extracellular recordings from hippocampal area CA1 and dorsal medial prefrontal cortex from rats performing a foraging task* [Data set]. CRCNS.org. https://doi.org/10.6080/K0H41PK8
