# Stage 3.2C Biological Comparability Note

## Status

Stage 3.2A/B established a frozen VTE wrapper and a synthetic trace path from Stage 3 model logs to trial-level VTE-like metrics.

Patch 14 established exploratory CRCNS W-track ingestion and segmentation diagnostics.

Patch 15A/B established a DANDI 000115 behavioral bridge:

```text
DANDI NWB
  -> behavioral layer probe
  -> filtered arm-choice canonical trace
  -> frozen VTE wrapper
  -> biological VTE-proxy metrics
```

This proves technical interoperability. It does not yet prove the central Stage 3 biological claim.

## Main hypothesis

The biological comparison layer is not designed to show that a synthetic animal and a biological animal move identically at a choice point.

The target claim is gate-viscosity dynamics:

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

VTE-like behavior is therefore a secondary readout. It is useful only when linked to interpretable choice episodes.

## Endpoint hierarchy

### Primary endpoints

Primary biological comparability requires recoverable choice semantics:

- chosen arm / route / path;
- available alternatives;
- reward, outcome, correctness, or error;
- trial, session, and subject identity;
- condition, perturbation, reversal, risk, conflict, or carryover structure when available.

These endpoints are closest to the Stage 3 model claim.

### Secondary behavioral endpoints

Secondary behavioral proxies are useful only around choice episodes:

- dwell time at a choice point;
- reorientation count;
- IdPhi-like angular displacement;
- trajectory tortuosity;
- pause / hesitation;
- switch / stay behavior;
- choice repetition or perseveration.

These metrics should not be interpreted outside a recoverable choice context.

### Neural proxy endpoints

Neural data can support comparison if it is aligned to choice episodes and alternatives.

Useful neural proxies include:

- decoded replay content around a choice point;
- place-cell or ensemble representation of candidate arms;
- alternation between candidate representations;
- representation entropy;
- dominance margin between alternatives;
- replay event count;
- LFP or spike event markers with interpretable relation to choice.

A neural event is not sufficient by itself. It becomes relevant only if it is linked to a choice episode and to alternatives, outcome, or task state.

## What counts as usable biological evidence

A dataset is useful for Stage 3.2C if it supports at least one of the following:

### A. Direct gate-viscosity comparison

Requirements:

- recoverable choice;
- alternatives known;
- outcome or correctness known;
- uncertainty, reversal, perturbation, risk, conflict, or carryover condition known;
- optional behavioral or neural deliberation readout.

This is the strongest class.

### B. Choice + outcome comparison

Requirements:

- recoverable choice;
- reward, outcome, correctness, or error recoverable;
- session and subject identity preserved.

This supports choice/outcome comparison but may not test ambiguity or carryover directly.

### C. Choice-centered deliberation proxy

Requirements:

- recoverable choice events;
- tracking or pose data around choice;
- enough geometry or event structure to define choice-centered windows.

This supports VTE-proxy extraction but not the full gate-viscosity claim by itself.

### D. Neural deliberation proxy

Requirements:

- neural event or decoded representation aligned to choice episodes;
- candidate alternatives recoverable;
- event timing linked to choice and outcome.

This can support the model if it reflects representational competition between alternatives.

### Reject / not eligible

A dataset should not be used for Stage 3.2C evidence if:

- no recoverable choice exists;
- only rest/sleep replay is available;
- only navigation without alternatives is available;
- only summary figures are available;
- behavioral events cannot be aligned to position or trial structure;
- task semantics cannot be reconstructed sufficiently for any comparison endpoint.

## Current dataset classification

### DANDI 000115

Associated paper: Gillespie et al. (2021).

Current local probe status:

- raw position: available;
- arm/home/R/W behavioral events: available;
- pump / reward events: available;
- StateScript: available;
- StateScript-to-position alignment: high-overlap alignment found;
- filtered arm-choice trace: exported;
- frozen VTE wrapper: accepts trace and writes metrics;
- explicit VTE labels: not found;
- full task correctness / ambiguity / carryover semantics: not yet reconstructed.

Classification:

```text
B/C:
  choice events and reward events are available;
  behavioral VTE-proxy extraction is technically supported;
  full gate-viscosity comparison is not established yet.
```

Use:

- technical biological bridge;
- event-centered trajectory proxy;
- reward-linked arm-choice summary;
- possible future neural proxy if replay/representation content can be aligned to choice events.

Do not claim:

- direct VTE-label validation;
- full maze equivalence;
- proof that biological and synthetic animals share the same head-scanning behavior;
- proof of gate viscosity without recoverable choice-condition semantics.

### CRCNS hc-6 / hc-28 W-track

Current status:

- MATLAB position/task files can be loaded;
- W-track position probes and segmentation diagnostics are available;
- route/event segmentation remains exploratory;
- useful for future W-maze reverse-proof work.

Classification:

```text
C / reverse-proof candidate:
  useful for trajectory and task-geometry work;
  not yet a direct Stage 3.2C evidence dataset.
```

Use:

- future W-maze environment reconstruction;
- out-of-domain geometry checkpoint;
- later model environment replication.

### Miles et al. 2021 / Mizumori VTE dataset

Current status:

- high-priority target;
- reported data link currently inaccessible;
- access request should be pursued.

Potential classification:

```text
A/B if data access is restored:
  likely close to explicit VTE detection and choice behavior;
  priority candidate for direct biological VTE comparison.
```

### Redish Lab OSF datasets

Current status:

- datasets appear available;
- internal structure not yet inventoried.

Potential classification:

```text
A/B/C depending on file contents:
  high priority for targeted inventory.
```

## Required outputs for future Stage 3.2C evidence

A complete biological comparison package should produce:

```text
canonical_biological_choice_trace.csv
biological_vte_trial_metrics.csv
Table_BioDataset_Choice_Semantics.csv
Table_BioDataset_Endpoint_Eligibility.csv
Table_BioDataset_Choice_Reward_Summary.csv
Table_BioDataset_VTE_Proxy_By_Condition.csv
Stage3_2C_Biological_Comparability_Report.md
```

The report must distinguish:

```text
primary comparison:
  choice and outcome structure

secondary comparison:
  VTE-like behavioral proxies

optional neural comparison:
  decoded or event-aligned representational competition

not claimed:
  biological identity
  full maze equivalence
  direct proof from VTE shape alone
```

## Current recommendation

Do not implement a full model-vs-biology comparison layer until a stronger biological endpoint is available.

Patch 15B should be closed as a technical bridge:

```text
Patch 15B:
  DANDI 000115 event-centered arm-choice trace export — complete.
```

The next code-heavy biological patch should wait for one of the following:

1. restored access to Miles et al. 2021 data;
2. Redish OSF inventory showing usable choice/VTE files;
3. DANDI task semantics reconstructed enough to classify correctness, ambiguity, or condition;
4. multi-subject DANDI batch export after endpoint eligibility is clarified.

Meanwhile, Stage 3 manuscript work can use the biological adapter results to state that the project has a working two-way measurement bridge, while keeping biological validation claims explicitly limited.

## References

Gillespie, A. K., Astudillo Maya, D. A., Denovellis, E. L., Liu, D. F., Kastner, D. B., Coulter, M. E., Roumis, D. K., Eden, U. T., & Frank, L. M. (2021). Hippocampal replay reflects specific past experiences rather than a plan for subsequent choice. *Neuron, 109*(19), 3149–3163.e6. https://doi.org/10.1016/j.neuron.2021.07.029

Johnson, A., & Redish, A. D. (2007). Neural ensembles in CA3 transiently encode paths forward of the animal at a decision point. *The Journal of Neuroscience, 27*(45), 12176–12189. https://doi.org/10.1523/JNEUROSCI.3761-07.2007

Miles, J. T., Kidder, K. S., Wang, Z., Zhu, Y., Gire, D. H., & Mizumori, S. J. Y. (2021). A machine learning approach for detecting vicarious trial and error behaviors. *Frontiers in Neuroscience, 15*, Article 676779. https://doi.org/10.3389/fnins.2021.676779