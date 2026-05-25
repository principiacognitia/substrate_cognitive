# Stage 3.2 Biological Dataset Shortlist

## Purpose

This file records the biological-dataset shortlist for future Stage 3.2 biological comparability work.

Stage 3.2 is closed as a read-only VTE-style measurement and statistical layer. The current biological conclusion is deliberately limited:

```text
The Stage 3 Gate-with-viscosity model approximates the theoretical behavioral profile attributed to biological VTE-like deliberation under ambiguous choice.
```

It does **not** yet establish direct synthetic-vs-biological behavioral trajectory equivalence.

---

## Selection rule after Patch 20I

Public direct download is no longer the primary filter. The next pass should prioritize fit to the scientific question:

```text
VTE-like traces at a recoverable choice point
```

Priority criteria:

1. recoverable choice point;
2. raw or reconstructable x/y trajectory around the choice point;
3. heading, head direction, or derivable heading;
4. VTE, head scanning, IdPhi, dwell, pause, or reorientation labels/proxies;
5. trial-level choice and outcome;
6. perturbation, reversal, alternation, risk, conflict, ambiguity, or learning condition;
7. neural recordings aligned to the same choice episodes;
8. reusable access path through public repository or author correspondence.

---

## Dataset classes

| Class | Meaning |
|---|---|
| **A** | Best candidate for direct future VTE-like behavioral comparison. |
| **B** | Strong candidate, but requires manual geometry, event, or trial reconstruction. |
| **C** | Useful control, diagnostic, or non-primary comparator. |
| **D** | Neural-comparator candidate; useful only if aligned to interpretable choice episodes. |
| **S2/S4** | Moved to Stage 2 / Stage 4 biological-comparator backlog, not Stage 3.2 VTE shortlist. |

---

## A-list candidates

| Priority | Dataset | Class | Reason | Current action |
|---|---|---:|---|---|
| A1 | Miles et al. 2021 / Mizumori VTE dataset | A/B/C pending audit | Explicit VTE-detection context and author-assisted access path. Potentially the closest manually annotated VTE benchmark if usable traces and labels are confirmed. | Audit downloaded material; identify trajectory, labels, task geometry, subject/session/trial structure. |
| A2 | Redish classic decision-point / multiple-T VTE data | A pending access | Strongest conceptual match to classic VTE at a decision point. Likely requires direct inquiry to Redish Lab or related archive search. | Locate accessible source or request trajectory/choice-point records. |
| A3 | Redish Lab LRA 2024 OSF | B decision-level endpoint | Closed Stage 3.2 primary biological endpoint. Good decision-level comparator, but Patch 18C does not provide biological movement replay. | Keep as canonical closed endpoint; do not upgrade to movement comparator without new trace data. |
| A4 | CRCNS hc-6 / hc-28 W-track family | B/C pending audit | W-track geometry may support recoverable choice points and route alternation. Potentially useful for future W-maze-like Stage 3.x work. | Audit task files, geometry, epoch segmentation, choice zones, and position quality. |

---

## B-list candidates

| Priority | Dataset | Class | Reason | Current action |
|---|---|---:|---|---|
| B1 | CRCNS pfc-8 / Oberto-Wiener-Zugaro T-maze rule switching | B/C/D | Strong free-moving T-maze dataset with visual/spatial rule switching, left/right choice structure, position, mPFC/vSTR/dmSTR spikes/LFP. Not an explicit VTE-label dataset, but highly relevant for future gate-mode arbitration and choice-point dwell/pause comparison. | Audit `.xml`, `.pos`, `.all.evt`, `.cue.evt`, `.cat.evt`; contact authors before publication use. |
| B2 | CRCNS hc-13 / Yu-Frank CA1-mPFC foraging alternate-choice | C/D, potentially B | Free-moving rats performing foraging / alternate-choice behavior; includes raw position, task files, CA1 and dorsal mPFC spikes/LFP. Promising if choice episodes and geometry zones can be reconstructed. | Audit `*task.mat` and `*rawpos.mat` before spikes/LFP. |
| B3 | CRCNS hc-5 / Buzsáki left-right alternation | C/D | Left/right alternation, hippocampal/entorhinal recordings, position and head direction fields. Useful for alternation/sequence-memory comparison, but one rat and not designed as VTE dataset. | Audit maze sessions and choice-window recoverability. |
| B4 | Stout et al. 2022 / Griffin Lab | B/C diagnostic | Contains VTE-related trial variables and published scripts. Trajectory reconstruction was not established in the closed Stage 3.2 pass. | Keep as diagnostic VTE-variable endpoint; re-audit trajectory fields only if needed. |
| B5 | DANDI 000115 / Gillespie et al. 2021 | B/C technical bridge | Strong NWB event-centered arm-choice bridge; not explicit VTE-label benchmark. | Keep as technical bridge, not primary VTE evidence. |
| B6 | DANDI:001371 / Prince-Singer update-task VR Y-maze | B/D, optional C-lite | Strong update-task and CA1/mPFC prospective-code comparator; not primary VTE-trace dataset. Head-fixed VR reduces route-trajectory comparability. | Move primarily to Stage 4/prospective-code backlog; optional audit for event-centered view-angle instability. |

---

## C-list / not primary Stage 3.2 VTE candidates

| Priority | Dataset | Class | Reason | Current action |
|---|---|---:|---|---|
| C1 | Redish RROW / Restaurant Row-related data | future task-family candidate | Conceptually relevant for value conflict and deliberation, but task complexity exceeds current Stage 3 configuration. | Do not treat as small Stage 3.2 extension. |
| C2 | Figshare 10248158 / Rtrack water maze tracking data | C | Open trajectory dataset with reversal structure and reconstructable geometry; useful for path/tortuosity/reversal tools, but not a discrete choice-point or VTE dataset. | Keep for trajectory/reversal/environment-builder backlog. |
| C3 | Wirth macaque virtual-maze datasets | C/D out-of-domain | Potentially useful for gaze/virtual-navigation controls, but not rodent VTE comparator. | Low priority unless article needs out-of-domain visual comparison. |

---

## Moved out of Stage 3.2 VTE shortlist

| Dataset | Destination | Reason |
|---|---|---|
| OSF Two-step ACC / Akam et al. | `docs/stages_2_and_4_biological_dataset_TBD.md` | Best biological comparator for Stage 2 Gate-Rheology / model-based control; not spatial VTE. |
| Figshare 20449140 / Miller-Botvinick-Brody rodent two-step | `docs/stages_2_and_4_biological_dataset_TBD.md` | Strong Stage 2 two-step learning-vs-choice comparator; no x/y/heading/VTE trace. |
| IBL standardized behavior / Brain Wide Map | `docs/stages_2_and_4_biological_dataset_TBD.md` | Head-fixed visual 2AFC with hidden block priors; requires new environment builder. |
| DANDI:001371 | cross-listed | Update-task / prospective-code comparator; not primary VTE. |

---

## Current recommendation

For a future direct biological-comparison cycle, the practical order should be:

1. Miles/Mizumori if manual VTE labels and traces are straightforward.
2. Redish classic / Redish Lab direct request if decision-point VTE traces are available.
3. CRCNS pfc-8 if T-maze geometry and event codes reconstruct cleanly.
4. CRCNS hc-13 if choice zones and foraging choice episodes reconstruct cleanly.
5. CRCNS hc-6/hc-28 W-track if W-track choice-point windows are clean.
6. Stout/Griffin only if trial-level trajectory reconstruction becomes reliable.

The target artifact for that future cycle is:

```text
synthetic behavioral left panel / biological behavioral right panel
```

and, for datasets with aligned neural data:

```text
synthetic gate-dynamics left panel / biological neural right panel
```

---

## References

Akam, T., Rodrigues-Vaz, I., Marcelo, I., Zhang, X., Pereira, M., Oliveira, R. F., Dayan, P., & Costa, R. M. (2021). The anterior cingulate cortex predicts future states to mediate model-based action selection. *Neuron, 109*(1), 149–163. https://doi.org/10.1016/j.neuron.2020.10.013

Gillespie, A. K., Astudillo Maya, D. A., Denovellis, E. L., Liu, D. F., Kastner, D. B., Coulter, M. E., Roumis, D. K., Eden, U. T., & Frank, L. M. (2021). Hippocampal replay reflects specific past experiences rather than a plan for subsequent choice. *Neuron, 109*(19), 3149–3163.e6. https://doi.org/10.1016/j.neuron.2021.07.029

Johnson, A., & Redish, A. D. (2007). Neural ensembles in CA3 transiently encode paths forward of the animal at a decision point. *The Journal of Neuroscience, 27*(45), 12176–12189. https://doi.org/10.1523/JNEUROSCI.3761-07.2007

Miles, J. T., Kidder, K. S., Wang, Z., Zhu, Y., Gire, D. H., & Mizumori, S. J. Y. (2021). A machine learning approach for detecting vicarious trial and error behaviors. *Frontiers in Neuroscience, 15*, Article 676779. https://doi.org/10.3389/fnins.2021.676779

Oberto, V. J., Gao, H. Y., & Wiener, S. I. (2021). *Single unit and LFP recordings of medial prefrontal cortex and ventral and medial striatum of rats alternating between visual and spatial discrimination tasks in a T-maze* [Data set]. CRCNS.org. https://doi.org/10.6080/K0R20ZKP

Pastalkova, E., Wang, Y., Mizuseki, K., & Buzsáki, G. (2015). *Simultaneous extracellular recordings from left and right hippocampal areas CA1 and right entorhinal cortex from a rat performing a left / right alternation task and other behaviors* [Data set]. CRCNS.org. https://doi.org/10.6080/K0KS6PHF

Redish, A. D. (2024). *2024 RedishLab: Recordings from medial prefrontal, dorsolateral striatum, and hippocampus on Left-Right-Alternate; DREADD disruption of mPFC*. OSF Project c4fjm.

Stout, J. J., Hallock, H. L., George, A. E., Adiraju, S. S., & Griffin, A. L. (2022). The ventral midline thalamus coordinates prefrontal–hippocampal neural synchrony during vicarious trial and error. *Scientific Reports, 12*, Article 10940. https://doi.org/10.1038/s41598-022-14707-8

Yu, J. Y., Liu, D., Grossrubatscher, I., & Frank, L. M. (2017). *Simultaneous extracellular recordings from hippocampal area CA1 and dorsal medial prefrontal cortex from rats performing a foraging task* [Data set]. CRCNS.org. https://doi.org/10.6080/K0H41PK8
