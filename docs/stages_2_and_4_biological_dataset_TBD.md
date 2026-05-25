# Stages 2 and 4 Biological Dataset TBD

## Purpose

This file records biological datasets that are useful for Stage 2 Gate-Rheology or future Stage 4 work, but are **not** primary Stage 3.2 VTE-choice-point datasets.

The common reason for exclusion from Stage 3.2 is:

```text
no free-moving VTE-like trajectory at a recoverable spatial choice point
```

These datasets may still be scientifically important because they test adjacent claims:

- model-based vs model-free control;
- transition learning;
- reversal learning;
- hidden priors;
- perceptual evidence ambiguity;
- future-state prediction;
- update under new information;
- broad neural correlates of decision variables.

---

## Stage 2 biological-comparator candidates

### OSF Two-step ACC — Akam / Costa mouse model-based control dataset

#### Status

High-priority candidate for a Stage 2 appendix or follow-up validation note.

#### Dataset fit

The OSF Two-step ACC project contains data for a mouse two-step task designed to dissociate model-based and model-free reinforcement learning. The task uses four nose-poke ports. At the first step, mice choose between top and bottom center ports. This choice probabilistically leads to a left-active or right-active second-step state, where mice can obtain probabilistic water reward.

The key experimental feature is that both reward probabilities and action-state transition probabilities can reverse over blocks.

#### Why it fits Stage 2

Stage 2 Gate-Rheology already implemented Two-Step / Reversal-style designs inspired by this task family. Akam et al. are therefore a near-ideal biological comparator for the existing Stage 2 article.

The best interpretation is not:

```text
ACC is the Gate.
```

The safer interpretation is:

```text
ACC provides a biological comparator for one functional role assigned to the Gate:
transition-sensitive, future-state-mediated control over action selection.
```

#### Available endpoints

Behavioral:

- mouse/session/date identifiers;
- first-step choice;
- second-step state;
- common vs rare transition;
- reward / omission;
- reward-probability block;
- transition-probability block;
- stay/switch behavior;
- reaction time from first-step choice to second-step response;
- reversal-learning task logs.

Neural / causal:

- ACC calcium imaging subset;
- ACC optogenetic inhibition subset;
- reversal-learning optogenetic control subset;
- published model-based and model-free regression analyses.

#### Possible Stage 2 appendix

```text
Appendix B. Biological Task Comparator: Akam Two-Step ACC Dataset
```

Candidate analyses:

- outcome predictor;
- transition predictor;
- transition × outcome predictor;
- stay/switch probability;
- lagged reward and transition effects;
- transition-reversal adaptation;
- viscosity-like persistence estimate;
- synthetic Stage 2 vs mouse two-step behavioral predictors.

#### Next audit questions

1. What exact fields are present in the `.txt` behavioral logs?
2. Can logs be parsed directly using the `Two-step_ACC` code?
3. Are transition-probability block labels explicit or reconstructed?
4. Are reward-probability block labels explicit or reconstructed?
5. Are optogenetic stimulation trials explicitly marked?
6. Are imaging sessions and behavioral sessions aligned through common IDs?
7. Is there a compact behavioral-only subset sufficient for Stage 2 comparison?
8. Should this become a Stage 2 appendix before or after the Stage 3 article draft?

#### Recommendation

Use after Stage 3 article handoff as the first biological-comparator appendix for the Stage 2 Gate-Rheology paper.

---

### Figshare 20449140 — Miller / Botvinick / Brody rodent two-step task

#### Status

Stage 2 biological-comparator candidate.

#### Dataset fit

The dataset is associated with a rodent two-step task separating learning from choice. Rats choose between two upper choice ports, transition probabilistically to one of two lower outcome ports, and receive reward or omission according to block-wise reward probabilities.

#### Available endpoints

Likely useful:

- trial-level choice;
- transition type;
- outcome port;
- reward / omission;
- reward-probability block;
- session / subject IDs;
- event timing, if included.

#### Current classification

```text
Class B for Stage 2
```

Strong choice/outcome comparator, not a spatial trajectory comparator.

#### Reason not Stage 3.2

The task is performed in operant chambers with nose ports. It does not provide free-moving route trajectories, heading, choice-point dwell, or VTE/head-sweep behavior.

#### Possible future use

```text
synthetic two-step Gate-Rheology agent
vs
rodent two-step learning and choice behavior
```

#### Difference from Akam

```text
Miller/Brody:
  stronger orbitofrontal value / learning-not-choice comparator

Akam/Costa:
  stronger ACC future-state / transition-model comparator
```

Both are useful for Stage 2, but Akam is the stronger first target for Gate-as-control-arbitration comparison.

---

## Stage 4 or Stage 4-adjacent candidates

### IBL standardized behavior / Brain Wide Map

#### Status

Manual audit corrected. Not a VTE-trace or maze-choice dataset.

#### Dataset fit

The IBL task is a head-fixed mouse visual decision-making task. On each trial, a visual grating appears on the left or right side of a screen, and the mouse reports the perceived side by turning a wheel. Stimulus contrast varies across trials, including 0% contrast trials. The probability of stimulus side changes across uncued blocks, typically 20:80 or 80:20.

#### Available endpoints

Behavioral:

- subject/session/trial IDs;
- stimulus side;
- stimulus contrast;
- probabilityLeft / block prior;
- choice;
- reaction or response time;
- feedback type;
- reward volume;
- trial timing;
- wheel movement.

Brain Wide Map additionally supports:

- Neuropixels recordings across many brain regions;
- video / pose information;
- DeepLabCut-derived body-part coordinates;
- broad neural correlates of stimulus, choice, prior, action, and outcome.

#### Current classification

```text
Class B/D for Stage 4-adjacent work
```

- B: choice/outcome and prior-sensitive behavioral comparator;
- D: neural comparator for prior, stimulus, choice, action, and reward encoding.

#### Reason not Stage 3.2 VTE

The task is not a T-maze, W-maze, fork, route-choice, or free-moving spatial navigation task. It does not provide VTE labels, choice-point dwell, biological route trajectories, or head-scanning behavior at a maze junction.

#### Possible future use

Useful for a future IBL-like environment builder:

```text
synthetic prior-sensitive Gate / evidence-accumulation task
vs
IBL mouse visual 2AFC behavior and neural activity
```

#### Required new model work

```text
visual 2AFC stimulus
contrast-dependent sensory evidence
hidden block prior
wheel-like left/right response
reward / timeout feedback
quiescence period
history-sensitive prior inference
```

This belongs to a future Stage 3.x or Stage 4-adjacent branch, not to the closed Stage 3.2 VTE wrapper.

---

### DANDI:001371 — Prince / Singer Lab update-task VR Y-maze

#### Status

Cross-listed with Stage 3.2 TBD, but stronger as Stage 4/prospective-code material.

#### Dataset fit

The dataset contains head-fixed mouse VR Y-maze behavior with CA1 and mPFC electrophysiology. The task is a memory-based update task: animals first receive an original cue, maintain the goal arm, and on update trials receive a second cue requiring switch or stay behavior.

#### Available endpoints

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

#### Current classification

```text
Class B/D for Stage 4 bridge
```

#### Reason not Stage 3.2 VTE

No explicit VTE labels, no natural free-moving choice-point VTE behavior, and head-fixed VR limits route-trajectory comparability.

#### Possible future use

```text
synthetic gate state around new information
vs
biological CA1/mPFC prospective-code adaptation around update cue
```

This is closer to update/prospective coding than to Stage 3.2 VTE-like trail comparison.

---

### Figshare 10248158 — Rtrack water maze tracking data

#### Status

Trajectory/reversal/environment-builder candidate.

#### Dataset fit

The dataset contains Morris water maze tracking data with training and reversal structure. It is useful for trajectory reconstruction, path tortuosity, search dynamics, and reversal adaptation.

#### Available endpoints

Potentially useful:

- x/y trajectory;
- trial/day/session structure;
- training vs reversal condition;
- arena geometry;
- platform location;
- path length;
- latency to platform;
- tortuosity;
- old-platform perseveration after reversal.

#### Current classification

```text
Class C for spatial/reversal tooling
```

#### Reason not Stage 3.2 VTE

The Morris water maze does not provide a discrete fork, T-maze, W-maze, or route-choice point. Deliberation, if present, is distributed across search behavior rather than localized at a recoverable junction.

#### Possible future use

```text
synthetic spatial search / reversal trajectory
vs
biological water maze reversal trajectory
```

This may be useful for environment-builder work, not for VTE choice-point comparison.

---

## Cross-stage bridge candidates

### CRCNS pfc-8

Primary listing is in `docs/stage3_2_biological_dataset_TBD.md`.

It is also relevant here because T-maze rule switching may support a future Stage 2/4 bridge:

```text
synthetic gate-mode switching
vs
mPFC-striatal rule-guided choice and assembly dynamics
```

It should remain a high-priority biological-comparator candidate, but not a current Stage 3.2 VTE benchmark.

### CRCNS hc-13

Primary listing is in `docs/stage3_2_biological_dataset_TBD.md`.

It is also relevant to future Stage 4 or Stage 3.x work if movement/immobility states can be aligned to CA1-mPFC neural state transitions.

---

## Priority order for non-VTE work

1. **Akam Two-step ACC** — best Stage 2 biological appendix candidate.
2. **Miller/Brody OFC two-step** — strong Stage 2 learning-vs-choice comparator.
3. **IBL standardized behavior / Brain Wide Map** — strong future prior-sensitive decision environment.
4. **DANDI:001371** — strong update/prospective-code candidate.
5. **Rtrack water maze** — useful trajectory/reversal environment-builder checkpoint.
6. **pfc-8** — strong cross-stage gate-mode arbitration candidate, but requires event/geometry audit.

---

## Recommended immediate action

Do not start these before the Stage 3 article draft.

After article handoff, the first practical biological-comparator task should be:

```text
Stage 2 Appendix Patch:
  parse Akam behavioral logs
  reproduce model-based regression family
  compare Stage 2 synthetic traces to Akam behavioral predictors
  write biological-comparator appendix
```

This is the most direct biological follow-up to the already completed Stage 2 paper.

---

## References

Akam, T., Rodrigues-Vaz, I., Marcelo, I., Zhang, X., Pereira, M., Oliveira, R. F., Dayan, P., & Costa, R. M. (2021). The anterior cingulate cortex predicts future states to mediate model-based action selection. *Neuron, 109*(1), 149–163. https://doi.org/10.1016/j.neuron.2020.10.013

Akam, T., Rodrigues-Vaz, I., Marcelo, I., Zhang, X., Pereira, M., Oliveira, R. F., Dayan, P., & Costa, R. M. (2020). *The anterior cingulate cortex predicts future states to mediate model-based action selection* [Data set]. OSF. https://doi.org/10.17605/OSF.IO/8JWHM

International Brain Laboratory, Aguillon-Rodriguez, V., Angelaki, D., Bayer, H., Bonacchi, N., Carandini, M., Cazettes, F., Chapuis, G. A., Churchland, A. K., Dan, Y., DeWitt, E., Faulkner, M., Forrest, H., Haetzel, L., Häusser, M., Hofer, S. B., Hu, F., Khanal, A., Krasniak, C., … Zador, A. M. (2021). Standardized and reproducible measurement of decision-making in mice. *eLife, 10*, e63711. https://doi.org/10.7554/eLife.63711

Meshulam, L., Angelaki, D., Benson, B., Benson, J., Birman, D., Arlandis, J., Bonacchi, N., Bougrova, K., Bruijns, S. A., Carandini, M., Catarino, J. A., Chapuis, G. A., Churchland, A. K., Dan, Y., Davatolhagh, F., Dayan, P., DeWitt, E. E., Engel, T. A., Fabbri, M., … Witten, I. B. (2025). A brain-wide map of neural activity during complex behaviour. *Nature, 645*(8079), 177–191. https://doi.org/10.1038/s41586-025-09235-0

Miller, K. J., Botvinick, M. M., & Brody, C. D. (2022). Value representations in the rodent orbitofrontal cortex drive learning, not choice. *eLife, 11*, e64575. https://doi.org/10.7554/eLife.64575

Miller, K. J., Botvinick, M. M., & Brody, C. D. (2022). *Value representations in the rodent orbitofrontal cortex drive learning, not choice* [Data set]. Figshare. https://doi.org/10.6084/m9.figshare.20449140

Overall, R., & Kempermann, G. (2020). *Water maze tracking data* [Data set]. Figshare. https://doi.org/10.6084/m9.figshare.10248158
