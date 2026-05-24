# Stage 3 Follow-up Article Handoff

## Working title

**Gate-Rheology and Deliberation: Viscous Control as a Source of VTE-like Behavior under Ambiguous Choice and One-Shot Valence Deformation**

---

## Article type

Follow-up article to the Stage 2 Gate-Rheology paper.

This should not be written as a general theory paper and should not absorb Stage 4. The article should stay close to the Stage 3 empirical/computational package.

---

## Core thesis

VTE-like deliberation does not require a special deliberation module.

In the Stage 3 model, deliberation-like behavior appears as a measurable behavioral regime when a viscous Gate arbitrates under:

1. ambiguous or weakly dominated choice;
2. conflicting exposure/reward fields;
3. persistent one-shot valence traces;
4. positive and negative carrier separation.

---

## Relation to the Stage 2 paper

Stage 2 established Gate-Rheology in abstract sequential decision tasks:

- Two-Step task;
- Reversal task;
- dissociation between control-mode viscosity `V_G` and action-level viscosity `V_p`;
- heavy-tailed switching latency;
- cross-task parameter reuse.

Stage 3 extends the same architecture into a spatial/exposure-like choice setting:

- Open/Covered baseline;
- reward-threat matrix;
- balanced-conflict ablations;
- one-shot shock/treat carryover;
- VTE-like trace measurement;
- seed-level statistical interpretation.

---

## Candidate abstract

Gate-Rheology proposes that arbitration between cognitive control modes has intrinsic inertia. Prior work tested this idea in abstract sequential decision tasks. Here we extend the framework to spatial/exposure-like choice and one-shot valence deformation. We show that a fixed Stage 3 S-O-R+Gate architecture produces persistent shock/treat carryover, separates positive and negative carrier dynamics, and yields VTE-like choice-point regimes under ambiguous or weakly dominated choice. A read-only VTE measurement layer translates externalized Stage 3 traces into a fixed trajectory schema, computes IdPhi-like, pause, and reorientation metrics, and evaluates effects at the seed level. We distinguish wrapper-sanity effects from model-relevant statistics and degenerate-ablation diagnostics. The results support an interpretation of deliberation-like behavior as a normal regime of viscous control under conflict, rather than as a special-purpose deliberation module. Biological comparison is limited to decision-level schema comparability and does not claim rodent-level VTE equivalence.

---

## Main claims

### Claim 1: Stage 3.1B closes the valence/exposure kernel

The model supports:

- balanced reward-threat conflict;
- one-shot shock/treat deformation;
- positive and negative carrier separation;
- ablation-localized effects;
- placebo-window controls.

### Claim 2: Deliberation-like behavior is measurable without model internals

The VTE layer reads externalized traces only.

It does not use:

- Gate state;
- Stage 3 config objects;
- internal reward/threat fields;
- precomputed deliberation labels.

### Claim 3: VTE-like effects must be interpreted by role

IdPhi/pause/reorientation separation between VTE and non-VTE rows is expected and should be treated as wrapper sanity.

Model-relevant evidence should be read from behavioral and ablation contrasts that remain meaningful after circular metrics are separated.

### Claim 4: Degenerate ablations are diagnostics, not clean localized effects

`novg` collapse should be interpreted as architectural collapse or extreme regime shift, not as a clean isolated component effect.

### Claim 5: Biological comparison remains decision-level

Stage 3.2 can support biological decision-level comparability, but not rodent trajectory equivalence.

---

## Suggested paper structure

### 1. Introduction

Problem:

- deliberation is often treated as a separate cognitive faculty or planning module;
- VTE in rodents suggests observable pause/reorient regimes at choice points;
- Gate-Rheology suggests an alternative framing: deliberation-like behavior may arise from viscous control arbitration under ambiguity.

Bridge from Stage 2:

- Stage 2 established viscosity of control-mode arbitration;
- Stage 3 asks whether the same architecture produces measurable deliberation-like regimes in spatial/exposure-like choice.

### 2. Model

Describe:

- S-O-R+Gate architecture;
- exposure field;
- temporal state;
- Gate cascade;
- one-shot shock/treat protocols;
- positive and negative carrier separation;
- ablations.

Boundary:

- no absence inference;
- no biological spatial cognition claim;
- no rodent equivalence claim.

### 3. Stage 3.1B valence/exposure closure

Use figures/tables from:

```text
docs/results/stage3_1b_closure/
```

Core results:

- reward-threat matrix;
- balanced conflict;
- one-shot shock/treat carryover;
- carrier-level diagnostics;
- ablation localization.

### 4. Stage 3.2 VTE measurement layer

Describe:

- read-only trace schema;
- Stage 3 log adapter;
- IdPhi-like metric;
- pause and reorientation metrics;
- thresholding;
- why wrapper does not import Stage 3 internals.

### 5. Seed-level statistics

Use outputs from:

```text
docs/results/vte/stage3_2_seed_level_stats_analysis/
```

Required distinction:

- wrapper sanity checks;
- model-relevant tests;
- degenerate ablation diagnostics.

Avoid presenting IdPhi self-separation as independent validation.

### 6. Biological comparability

Use cautious language:

- biological data are used as decision-level comparators;
- comparability is schema-level and decision-level;
- movement-level replay and rodent equivalence are deferred.

### 7. Discussion

Main interpretive line:

- deliberation-like behavior can be understood as a regime of viscous arbitration;
- one-shot valence deformation shifts the control field over time;
- VTE-like behavior appears where choice criteria are weakly dominant or conflicted;
- Stage 3 does not need a separate deliberation module.

### 8. Limitations

Must include:

- synthetic pose reconstruction;
- no rodent-level VTE equivalence;
- limited biological comparison;
- no allocentric spatial model;
- no Stage 4 absence inference;
- no full W-maze/RROW replication.

### 9. Future work

Use:

```text
docs/stage3_2_TBD.md
```

---

## Candidate figures

### Figure 1: Stage 3 architecture and closure boundary

S-O-R+Gate v3, exposure field, temporal state, one-shot shock/treat, VTE wrapper as read-only layer.

### Figure 2: Stage 3.1B valence/exposure closure

Reward-threat matrix + one-shot shock/treat carryover.

### Figure 3: VTE-style measurement layer

Trace schema, choice point, IdPhi-like angular integration, pause, reorientation count.

### Figure 4: Seed-level statistics

Model-relevant effects, VTE rate by ablation, degenerate-ablation diagnostics.

### Figure 5: Biological decision-level comparability boundary

Synthetic Stage 3 decision trace vs biological decision-level comparator; explicitly not trajectory equivalence.

---

## Reviewer package

Use:

```bash
python -m stage3.analysis.build_stage3_reviewer_package \
  --preset stage3_1_3_2 \
  --profile llm5 \
  --results-root docs/results \
  --output-dir docs/reviewer_packages/stage3_1_3_2 \
  --clean
```

The reviewer package is not the article. It is a compact evidence bundle for model-assisted review.

---

## Writing rules for the paper

Use restrained claims.

Do not write:

- “rodent VTE reproduced”;
- “biological validation”;
- “allocentric deliberation”;
- “planning module discovered”;
- “absence inference”.

Prefer:

- “VTE-like measurement”;
- “decision-level comparability”;
- “behavioral regime”;
- “read-only trace wrapper”;
- “schema-level biological comparator”;
- “deliberation-like behavior under viscous arbitration”.

---

## Article readiness

The article can be drafted after:

1. Stage 3.2 closure note is committed.
2. `README.md` and `vte/README.md` are synchronized.
3. reviewer package builds cleanly.
4. `stage3/tests` and `vte/tests` pass.