# Paper 2 Plan: Gate-Rheology and Deliberation

## Working title

**Gate-Rheology and Deliberation: Viscous Control as a Source of VTE-like Behavior under Ambiguous Choice and One-Shot Valence Deformation**

---

## Status

Planning document for the Stage 3 follow-up paper.

This file is not the manuscript. It is an article-construction map: claims, non-claims, evidence blocks, figure plan, table plan, terminology rules, and section checklist.

---

## Core thesis

VTE-like deliberation need not be implemented as a special deliberation module.

In the Stage 3 model, it emerges as a measurable behavioral regime when a viscous Gate arbitrates under:

1. weakly dominant choice criteria;
2. conflicting exposure/reward gradients;
3. persistent one-shot valence traces;
4. positive and negative carrier asymmetry.

The article should argue that deliberation-like behavior can be treated as a regime of viscous control arbitration in a valence-affordance field, not as evidence for a hidden symbolic planner or homuncular evaluator.

---

## Central conceptual move

Low-level cognition does not begin with objects.

A finite agent does not encounter the world as a neutral array of already-formed objects with ready-made valences. It operates over selectively sampled gradients: risk, opportunity, exposure, cost, effort, access, reward, and future controllability.

Object-like cognition appears later as a stabilized compression over recurrent affordance basins. At the level modeled in Stage 3, object representation is not required.

The sensory system is not a passive recorder. It already filters and weights the incoming stream. This “coloring” of initially neutral variation does not require a homunculus. It can arise from evolved or engineered coefficients, feedback loops, temporal traces, and action-linked consequences.

The Gate does not inspect objects and does not assign semantic meanings. It receives compressed diagnostics and arbitrates control under viscosity.

---

## Content checklist

| Status | Section | Function | Source artifacts / inputs | Notes |
| :--- | :--- | :--- | :--- | :--- |
| ⬜ | Abstract | State the paper claim in 200–250 words | Stage 3.1B closure, Stage 3.2 closure, reviewer package | Must avoid biological-overclaim language |
| ⬜ | 1. Introduction | Frame deliberation without a deliberation module | Paper 1, Redish VTE literature, Stage 3 results | Introduce VTE-like behavior as behavioral regime |
| ⬜ | 2. From Objects to Valence-Affordance Fields | Theoretical bridge from PC/S-O-R to Stage 3 | PC v3, Stage 3 architecture | Crucial section; must avoid metaphysical overreach |
| ⬜ | 3. S-O-R+Gate Architecture | Explain model components | Paper 1, Stage 3 README, SPECIFICATION3 | Keep concise; do not rewrite Paper 1 |
| ⬜ | 4. Valence/Exposure Field | Explain open/covered choice, reward/exposure gradients | Stage 3.1A/B docs and configs | Emphasize field-level gradients, not object labels |
| ⬜ | 5. One-Shot Valence Deformation | Present shock/treat carryover mechanism | Stage 3.1B figures/tables | Need negative and positive branches |
| ⬜ | 6. VTE-like Measurement Layer | Explain read-only wrapper and metrics | `vte/README.md`, Stage 3.2 outputs | Must state wrapper does not alter agent |
| ⬜ | 7. Seed-Level Statistics | Present model-relevant vs wrapper-sanity split | Patch 20E tables/figures | Avoid treating IdPhi self-separation as independent validation |
| ⬜ | 8. Biological Decision-Level Comparability | Cautious biological comparison | Lab adapter docs, Redish/Griffin/Mizumori notes | Decision-level only, not trajectory equivalence |
| ⬜ | 9. Discussion | Interpret deliberation-like behavior as viscous arbitration | All results | Main theoretical synthesis |
| ⬜ | 10. Limitations | Explicitly list non-claims | Stage 3.2 closure note | Important for reviewer trust |
| ⬜ | 11. Future Work | Visualization, W-maze, Stage 4 | `docs/stage3_2_TBD.md` | Keep short; do not open new experiment inside paper |
| ⬜ | References | Formal APA references | Paper 1 refs + Redish/Gibson/Shenhav | Verify all biological dataset references before submission |

Legend:

```text
⬜ planned
🟡 drafting / partial
✅ ready for manuscript insertion
⚠️ needs verification
```

---

## Proposed abstract

Gate-Rheology proposes that arbitration between control modes has intrinsic inertia. Prior work showed that this inertia explains delayed mode switching in abstract sequential decision tasks. Here we extend the framework to spatial/exposure-like choice and one-shot valence deformation. We argue that low-level cognition need not begin with object representations: a finite agent first operates in a valence-affordance field, where sensory differences acquire asymmetric significance for action, risk, opportunity, exposure, and cost. In this setting, a fixed S-O-R+Gate architecture produces tradeoff-sensitive path choice, persistent shock/treat carryover, and VTE-like dwell-reorientation regimes under ambiguous choice. A read-only VTE measurement layer translates externalized traces into fixed trajectory metrics and evaluates effects at the seed level. We distinguish wrapper-sanity effects from model-relevant statistics and degenerate-ablation diagnostics. The results support a mechanistic interpretation of deliberation-like behavior as a regime of viscous control arbitration, not as a special-purpose deliberation module or a homuncular object evaluator.

---

## Main claims

### Claim 1: Low-level cognition is field-based, not object-first

The model does not require object-level sensory labels.

The agent acts in a valence-affordance field: a structured set of gradients over exposure, reward, opportunity, threat, cost, and action availability.

Object-like structure is treated as a later compression over recurrent affordance basins, not as the primitive unit of Stage 3 cognition.

### Claim 2: Gate arbitration does not require a homunculus

The Gate does not inspect objects, assign meanings, or internally deliberate.

It receives compressed field-derived diagnostics and applies fixed arbitration dynamics with viscosity.

### Claim 3: One-shot learning can be modeled as field deformation

Shock and treat events do not need to become symbolic episodic objects.

They deform future choice through temporal traces and positive/negative carrier dynamics.

### Claim 4: VTE-like behavior can emerge as a regime of viscous arbitration

When choice criteria are weakly dominant or conflicted, the model produces dwell/reorientation regimes measurable by VTE-like metrics.

This is not implemented as a special deliberation module.

### Claim 5: VTE measurement is external and read-only

The VTE wrapper reads externalized traces.

It does not import Gate state, reward/threat config objects, or precomputed deliberation labels.

### Claim 6: Statistical interpretation must separate roles

Large VTE-minus-non-VTE effects on IdPhi/pause/reorientation metrics are wrapper-sanity effects.

Model-relevant evidence must be read from behavioral and ablation contrasts after separating circular measurement effects.

### Claim 7: Biological comparison remains decision-level

Biological datasets can constrain the schema and provide decision-level comparators.

They do not establish rodent-level VTE equivalence or movement-level trajectory replay.

---

## Non-claims

The paper must not claim:

- rodent-level VTE equivalence;
- biological trajectory replay;
- neural mechanism identity;
- allocentric spatial cognition;
- object permanence;
- absence inference;
- self-model-based visibility reasoning;
- symbolic episodic event memory;
- general-purpose planning;
- full W-maze or RROW task equivalence.

Preferred language:

```text
VTE-like measurement
deliberation-like behavior
decision-level biological comparator
schema-level biological bridge
field deformation
viscous arbitration
read-only trace wrapper
```

Avoid:

```text
biological validation
rat VTE reproduced
rodent equivalence
planning module
object understanding
absence reasoning
neural mechanism confirmed
```

---

## Section outline

## 1. Introduction: Deliberation Without a Deliberation Module

Purpose:

- introduce VTE as an observable behavioral regime;
- explain why VTE is often interpreted as deliberation;
- state the alternative: VTE-like behavior may arise from control arbitration under conflict;
- connect to Gate-Rheology Paper 1.

Key paragraph target:

```text
The question is not whether the model has an internal deliberation faculty. The question is whether a fixed viscous Gate, acting over field-derived diagnostics, produces the measurable behavioral signatures normally associated with deliberation under ambiguous choice.
```

---

## 2. From Objects to Valence-Affordance Fields

Purpose:

- present the core theoretical bridge;
- explain why object-first cognition is not required;
- define valence-affordance field;
- explain sensory “coloring” without homunculus.

Required points:

- raw physical variation is not yet cognition;
- finite systems selectively sample differences;
- selected differences acquire asymmetric significance through action and viability loops;
- valence is not narrow emotion;
- affordance is relational, not an isolated object property;
- object-like cognition is a later compression.

Possible schematic:

```text
physical variation
  -> selective sensory uptake
  -> weighted gradients
  -> valence-affordance field
  -> compressed diagnostics
  -> Gate arbitration
  -> action / dwell / reorientation
```

---

## 3. S-O-R+Gate Architecture

Purpose:

- summarize the model without rewriting Paper 1.

Include:

- States;
- Operations;
- Relations;
- Gate;
- `V_G`;
- `V_p`;
- diagnostic vector;
- threshold cascade;
- temporal traces.

Boundary:

```text
The Gate is not a semantic evaluator. It is a control-allocation mechanism over compressed diagnostics.
```

---

## 4. Stage 3 Valence/Exposure Task

Purpose:

- explain the task environment.

Include:

- open/covered choice;
- exposure gradient;
- reward gradient;
- ambiguous / weakly dominant choice;
- balanced conflict;
- ablation conditions.

Interpretive point:

```text
The task does not require the agent to represent “open path” or “covered path” as object-like entities. It requires the system to respond to structured gradients of reward and exposure.
```

---

## 5. One-Shot Valence Deformation

Purpose:

- show that a single event changes future choice.

Subsections:

### 5.1 Shock branch

Use:

- `h_risk`;
- `q_neg`;
- post-shock shift;
- persistence across post-event windows.

### 5.2 Treat branch

Use:

- `h_opp`;
- `q_pos`;
- target shift;
- positive carrier dynamics.

### 5.3 Ablation localization

Show:

- full vs no-vg vs no-vp vs other ablations;
- carrier-level asymmetry;
- degenerate ablation diagnostics where relevant.

Core sentence:

```text
The event does not need to become a symbolic memory item. It changes the subsequent field through persistent traces and feedback.
```

---

## 6. VTE-like Measurement Layer

Purpose:

- explain the wrapper.

Include:

- externalized trace schema;
- synthetic pose reconstruction label;
- IdPhi-like metric;
- pause/dwell;
- reorientation count;
- `vte_binary`;
- fixed thresholding;
- no imports from `stage3/`.

Boundary:

```text
The wrapper measures traces; it does not generate behavior.
```

---

## 7. Seed-Level Statistics and Role-Aware Interpretation

Purpose:

- present Patch 20E statistical interpretation.

Include:

- seed as inferential unit;
- trial rows as measurement observations;
- FDR correction;
- model-relevant tests;
- wrapper-sanity checks;
- degenerate-ablation diagnostics.

Important distinction:

```text
VTE-minus-non-VTE separation on IdPhi-like metrics is expected because those metrics define or closely track the VTE label. These results are sanity checks, not independent validation.
```

Model-relevant evidence should focus on:

- behavioral outcome;
- reward;
- VTE rate by ablation;
- ablation contrasts;
- non-circular effects.

---

## 8. Biological Decision-Level Comparability

Purpose:

- place biological data without overclaiming.

Include:

- Redish LRA as decision-level comparator;
- Stout / Griffin dataset as diagnostic VTE endpoint;
- Mizumori / Miles as deferred VTE-label benchmark;
- DANDI and CRCNS as technical bridges / future geometry candidates.

Boundary:

```text
Decision-level comparability is not movement-level comparability.
```

---

## 9. Discussion: Viscous Control as Deliberation-like Behavior

Purpose:

- synthesize the argument.

Points:

- deliberation-like behavior can arise from arbitration dynamics;
- ambiguity creates dwell/reorientation regimes;
- one-shot traces deform the field;
- Gate viscosity explains persistence and delayed commitment;
- object-level representation is not needed for the tested behavior.

---

## 10. Limitations

Required limitations:

- toy-model environment;
- synthetic pose reconstruction;
- no rodent-level VTE equivalence;
- no neural mechanism identity;
- no object permanence;
- no absence inference;
- no W-maze/RROW equivalence;
- biological comparison currently decision-level;
- parameters are effective model parameters, not directly fitted biological constants.

---

## 11. Future Work

Use `docs/stage3_2_TBD.md`.

Possible items:

- VTE-like trail visualization;
- VTE-like trail animation;
- side-by-side decision-level comparison;
- W-maze-like config;
- deterministic maze builder;
- Gate dynamics comparison with neural data;
- Stage 4 absence/visibility/self-model extension.

---

## Figure plan

| Figure | Working title | Purpose | Source | Status |
| :--- | :--- | :--- | :--- | :--- |
| Figure 1 | From Field to Gate | Explain non-object-first cognition and Gate arbitration | New schematic | ⬜ |
| Figure 2 | Stage 3.1B Valence/Exposure Closure | Show matrix + one-shot behavioral effects | `docs/results/stage3_1b_closure/` | ⬜ |
| Figure 3 | One-Shot Valence Deformation | Show shock/treat traces and carrier asymmetry | Stage 3.1B outputs | ⬜ |
| Figure 4 | VTE-like Measurement Layer | Explain trace wrapper, IdPhi, dwell, reorientation | `vte/` + new schematic | ⬜ |
| Figure 5 | Seed-Level Statistics | Show model-relevant effects and degenerate diagnostics | Patch 20E figures | ⬜ |
| Figure 6 | Biological Decision-Level Boundary | Show synthetic vs biological decision endpoint boundary | lab adapter docs | ⬜ |

---

## Table plan

| Table | Working title | Purpose | Source | Status |
| :--- | :--- | :--- | :--- | :--- |
| Table 1 | Stage 3 Claim / Evidence / Non-claim Map | Prevent overinterpretation | Closure docs | ⬜ |
| Table 2 | One-Shot Effects by Branch and Ablation | Summarize shock/treat deformation | Stage 3.1B tables | ⬜ |
| Table 3 | VTE Test Role Summary | Separate model-relevant and wrapper-sanity tests | Patch 20E | ⬜ |
| Table 4 | Biological Dataset Endpoint Eligibility | Clarify biological comparability boundary | lab adapters | ⬜ |
| Table 5 | Terminology Rules | Standardize cautious language | This document | ⬜ |

---

## Reviewer risk register

| Risk | Likely objection | Response strategy | Status |
| :--- | :--- | :--- | :--- |
| Hidden homunculus in Gate | “The Gate decides what things mean.” | Gate receives compressed diagnostics; it does not assign semantic meaning. | ⬜ |
| Object labels smuggled into sensors | “The model already knows objects.” | Inputs are gradients and traces, not object tokens. | ⬜ |
| VTE circularity | “VTE metrics validate themselves.” | Separate wrapper-sanity from model-relevant tests. | ⬜ |
| Biological overclaim | “This is not rodent VTE.” | State decision-level comparator boundary. | ⬜ |
| Hand-tuned toy model | “The coefficients were chosen to get the result.” | Emphasize ablations, fixed wrapper, seed-level tests, and non-claims. | ⬜ |
| Scope drift into Stage 4 | “The paper implies absence or object permanence.” | Explicitly defer absence/visibility/self-model reasoning. | ⬜ |

---

## Terminology rules

Use:

```text
VTE-like behavior
deliberation-like regime
valence-affordance field
field-derived diagnostics
viscous arbitration
one-shot valence deformation
decision-level biological comparator
schema-level bridge
read-only measurement layer
```

Avoid or qualify:

```text
VTE
deliberation
planning
biological validation
object representation
memory item
rodent equivalence
neural mechanism
```

When using stronger terms, immediately specify the operational meaning.

Example:

```text
“Deliberation-like” refers to measurable dwell/reorientation regimes at choice points, not to an asserted symbolic planning process.
```

---

## Minimal manuscript claim boundary

The paper is ready only if the final draft preserves this boundary:

```text
Stage 3 shows that a fixed S-O-R+Gate architecture can produce one-shot valence deformation and VTE-like measurement signatures under ambiguous choice.

It does not show that the model has object cognition, rodent-equivalent VTE, neural deliberation, absence inference, or symbolic planning.
```

---

## Immediate next tasks

| Status | Task | File / target |
| :--- | :--- | :--- |
| ⬜ | Create manuscript skeleton | `docs/papers/paper2_gate_rheology_deliberation.md` |
| ⬜ | Draft Section 2 first | `From Objects to Valence-Affordance Fields` |
| ⬜ | Extract Figure 2 candidates | `docs/results/stage3_1b_closure/` |
| ⬜ | Extract Figure 5 candidates | `docs/results/vte/stage3_2_seed_level_stats_analysis/` |
| ⬜ | Build article-specific figure registry | `docs/article_handoff/Paper2_Figure_Registry.md` |
| ⬜ | Verify biological references | Redish, Stout, Miles/Mizumori |
| ⬜ | Build reviewer package | `stage3_1_3_2`, `llm5` profile |

---

## References to verify before manuscript submission

Gibson, J. J. (1979). *The ecological approach to visual perception*. Houghton Mifflin.

Redish, A. D. (2016). Vicarious trial and error. *Nature Reviews Neuroscience, 17*(3), 147–159. https://doi.org/10.1038/nrn.2015.30

Shenhav, A., Botvinick, M. M., & Cohen, J. D. (2013). The expected value of control: An integrative theory of anterior cingulate cortex function. *Neuron, 79*(2), 217–240. https://doi.org/10.1016/j.neuron.2013.07.007