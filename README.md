# Principia Cognitia: Substrate-Independent Cognitive Architecture

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Status: Stage 2 Complete](https://img.shields.io/badge/status-stage--2--complete-green)](stage2/)
[![Status: Stage 3.1 Complete](https://img.shields.io/badge/status-stage--3.1--complete-green)](stage3/)
[![Status: Stage 3.2 Closed](https://img.shields.io/badge/status-stage--3.2--closed-green)](vte/)

**Author:** Alex Snow (Aleksey L. Snigirov)
**Email:** alex2saaba@gmail.com
**ORCID:** 0009-0001-3713-055X
**GitHub:** https://github.com/principiacognitia/substrate_cognitive

---

## 📖 Description

**Principia Cognitia** is a research framework for modeling cognitive systems capable of functioning across various substrates (both biological and artificial). The architecture is grounded in the principles of **Gate-Rheology**—a mechanistic model for arbitrating between computational modes, each possessing its own inherent inertia.

### Key Idea

Cognitive rigidity does not stem from the content of representations, but rather from the **dynamics of control mode selection**. The arbitration between modes (exploit vs. explore) possesses its own intrinsic viscosity ($V_G$), which accumulates over time and exhibits hysteresis.
The repository develops a sequence of toy-model experiments around a central architectural claim: cognitive rigidity and deliberation-like behavior can be modeled as consequences of dynamic control-mode arbitration rather than as fixed value representations or special-purpose deliberation modules.

The current stable line is:

- **Stage 2:** Gate-Rheology in Two-Step and Reversal tasks.
- **Stage 3.1:** valence/exposure kernel closure, including balanced conflict and one-shot shock/treat carryover.
- **Stage 3.2:** read-only VTE-style measurement and seed-level statistics over externalized Stage 3 traces.

Stage 3.2 is now closed as a measurement/statistical layer. Visualization, W-maze variants, biological neural-data comparison, and Stage 4 mechanisms are explicitly deferred.

---

## 📊 Project Status

| Stage | Task | Status | Documentation |
| :--- | :--- | :--- | :--- |
| **MVP** | Initial T-maze / gate-rheology prototype | **Historical** | [README.md](mvp/README.md) |
| **Stage 2** | Two-Step and Reversal validation | ✅ **Complete  / historical baseline** | [Preprint](docs/Gate-Rheology%20-%20Inertia%20of%20Cognitive%20Control%20Explains%20Meta-Rigidity%20in%20Sequential%20Decision%20Making%20and%20Reversal%20Learning.pdf), [stage2/README.md](stage2/README.md) |
| **Stage 3.0** | Gate v3 architectural refactor | ✅ **Implemented as Stage 3 substrate** | [SPECIFICATION3.md](docs/SPECIFICATION3.md), [stage3/README.md](stage3/README.md) |
| **Stage 3.1A** | Open/Covered baseline compatibility | ✅ **Compatibility layer** | [stage3/README_Stage_3.1A.md](stage3/README_Stage_3.1A.md)  |
| **Stage 3.1B** | Valence/exposure closure package | ✅ **Complete / closure package** | [stage3/README.md](stage3/README.md), [closure package](docs/results/stage3_1b_closure/README.md) |
| **Stage 3.2A** | VTE wrapper core | ✅ **Complete** | [vte/README.md](vte/README.md), [design note](vte/STAGE_3_2_DESIGN_NOTE.md) |
| **Stage 3.2B** | Stage 3 trace adapter, batch analysis, biological decision-level comparability, seed-level statistics | ✅ **Complete / closed** | [docs/STAGE3_2_CLOSURE.md](docs/STAGE3_2_CLOSURE.md) |
| **Stage 3.2C** |  VTE-like trail visualization, animation, side-by-side visual comparison | ⬜ **Deferred** | [docs/stage3_2_TBD.md](docs/stage3_2_TBD.md) |
| **Stage 4** | Absence / visibility / self-model extension | ⬜ **Not started** | Deferred |

Stage 3.1B is not a new environment family. It closes a specific kernel: reward-threat matrix behavior, balanced-conflict ablation behavior, one-shot shock/treat carryover, carrier-level diagnostics, and placebo-window controls.

Stage 3.2 is implemented as a separate top-level `vte/` service module. It does not modify the Stage 3 agent. It reads externalized traces and produces VTE-style trajectory metrics, adapter outputs, batch summaries, figures, reports, and metadata.

---
## 🗂️ Repository Structure

```
substrate_cognitive/
├── mvp/                        # ✅ Completed MVP (T-maze, v5.0)
├── stage2/                     # ✅ Stage 2 validation experiments
│   ├── core/                   # ✅ Core modules (gate, rheology, baselines)
│   ├── twostep/                # ✅ Two-Step Task (Daw et al., 2011)
│   ├── reversal/               # ✅ Block-Reversal Task (Le et al., 2023)
│   └── analysis/               # ✅ Analysis and visualization scripts
├── stage3/                     # ✅ Stage 3 architecture, environments, tests, analysis
│   ├── core/                   # ✅ Gate v3, exposure field, temporal state
│   ├── envs/                   # ✅ Stage 3 environments
│   ├── tests/                  # ✅ Stage 3 regression and integration tests
│   └── analysis/               # ✅ Stage 3 runners, analyzers, validators
├── vte/                        # ✅ Stage 3.2 VTE service layer
│   ├── core/                   # ✅ Read-only VTE wrapper and metrics
│   ├── adapters/               # ✅ Stage 3 and biological-data adapters
│   ├── lab_adapters/           # ✅ Biological/lab comparability adapters
│   ├── analysis/               # ✅ VTE analysis, statistics, reporting
│   ├── tests/                  # ✅ VTE schema, wrapper, adapter, and analysis tests
│   └── visualization/          # 🟡 Optional visualization layer
├── docs/                       # Specifications, notes, curated result packages
│   ├── results/                # Publication-facing curated outputs
│   ├── reviewer_packages/       # compact LLM/human reviewer packages
│   ├── STAGE3_2_CLOSURE.md
│   ├── stage3_2_TBD.md
│   └── article_handoff/
├── logs/                       # Raw/generated local outputs; ignored by git
├── tests/                      # 🟡 Automated debug tests
└── README.md                   # This file
```

---

## 🚀 Quick Start

### Installation

```bash
# Cloning the repository
git clone https://github.com/principiacognitia/substrate_cognitive.git
cd substrate_cognitive

# Installing dependencies
pip install -r requirements.txt

# Run tests:
python -m pytest stage3/tests
python -m pytest vte/tests
```


## Stage 2: Reproducing Results

```bash
# Generate all figures for the preprint
python -m stage2.analysis.run_all

# Run a specific experiment
python -m stage2.analysis.run_all --experiment-id twostep_ablation_20260310_195529

# Figure 3 only (V_G dynamics)
python -m stage2.analysis.run_all --figure 3
```

## Stage 3.1 closure workflow

### Smoke run

```bash
python -m stage3.analysis.run_stage3_1_closure_package --mode smoke
python -m stage3.analysis.validate_stage3_1_artifacts --root docs/results
python -m stage3.analysis.validate_stage3_1b_acceptance --results-root docs/results --mode smoke
```

### Full production run

Run only after code and documentation are committed.

```bash
python -m stage3.analysis.run_stage3_1_closure_package --mode full
python -m stage3.analysis.validate_stage3_1_artifacts --root docs/results
python -m stage3.analysis.validate_stage3_1b_acceptance --results-root docs/results --mode full
```

The full profile uses 50 seeds and 100 trials per seed for the Stage 3.1A
compatibility rerun, Stage 3.1B balanced ablation suite, one-shot protocols, and
the Stage 3.1B 3x3 matrix layer.

---

## Stage 3.1B artifact layers

The curated Stage 3.1B package is organized into four layers:

1. `matrix`: 3x3 reward x threat conflict surface.
2. `balanced ablation`: balanced-conflict ablation metrics.
3. `one-shot shock`: event-aligned negative carryover.
4. `one-shot treat`: event-aligned positive carryover.
5. `diagnostics`: placebo-window, carrier, and ablation-localization checks.

Publication-facing outputs are written to:

```text
docs/results/stage3_1b_closure/
├── figures/
├── tables/
├── stats/
├── reports/
├── artifact_registry.json
└── ARTIFACT_REGISTRY.md
```

Raw run outputs remain under:

```text
logs/stage3/stage3_1_closure_raw/
```

---

## Stage 3.2 closure workflow

Stage 3.2 is a read-only measurement and analysis layer over externalized Stage 3 behavioral traces.

Canonical Stage 3.2 outputs:

```text
docs/results/vte/stage3_2_seed_level_stats_analysis/
```

Main files:

```text
Stage3_2_Response_To_GLM_Stats_Critique.md
Stage3_2_Seed_Level_Stats_Analysis_Report.md
Table_3_2_Seed_Level_Stats_By_Test_Role.csv
Table_3_2_Model_Relevant_Seed_Level_Tests.csv
Table_3_2_Wrapper_Sanity_Tests.csv
Table_3_2_Degenerate_Ablation_Diagnostics.csv
Figure_3_2_Model_Relevant_Seed_Level_Effects.png
Figure_3_2_Degenerate_Ablation_Diagnostics.png
Figure_3_2_Seed_Level_VTE_Rate_By_Ablation.png
```

Reviewer package:

```bash
python -m stage3.analysis.build_stage3_reviewer_package \
  --preset stage3_1_3_2 \
  --profile llm5 \
  --results-root docs/results \
  --output-dir docs/reviewer_packages/stage3_1_3_2 \
  --clean
```

---

## Stage 3.2 interpretation boundary

Stage 3.2 supports the following claims:

1. Stage 3 step logs can be translated into a fixed external VTE trace schema.
2. VTE-style metrics can be computed without importing Stage 3 model internals.
3. Seed-level statistics separate wrapper-sanity effects from model-relevant effects.
4. Degenerate ablations are marked separately and not treated as clean localized model effects.
5. Biological comparison is currently decision-level and schema-level, not rodent trajectory equivalence.

Stage 3.2 does not claim:

- rodent-level VTE equivalence;
- allocentric spatial cognition;
- biological neural mechanism identity;
- absence inference;
- self-model-based visibility reasoning;
- full W-maze or RROW task equivalence.

---





## Documentation map

| Path | Role |
| :--- | :--- |
| [docs/README.md](docs/README.md) | Documentation and curated result package map |
| [stage3/README.md](stage3/README.md) | Stage 3 architecture and Stage 3.1 closure commands |
| [stage3_1b_closure/README.md](docs/results/stage3_1b_closure/README.md) | Generated closure package README |
| [STAGE3_1B_CLOSURE_REPORT.md](docs/results/stage3_1b_closure/reports/STAGE3_1B_CLOSURE_REPORT.md) | Generated closure report |
| [ARTIFACT_REGISTRY.md](docs/results/stage3_1b_closure/ARTIFACT_REGISTRY.md) | Generated artifact registry |
| [docs/STAGE3_2_CLOSURE.md](docs/STAGE3_2_CLOSURE.md) | Stage 3.2 closure note |
| [docs/stage3_2_TBD.md](docs/stage3_2_TBD.md) | Deferred Stage 3 extensions |
| [vte/README.md](vte/README.md) | Stage 3.2 VTE measurement layer |
| [vte/STAGE_3_2_DESIGN_NOTE.md](vte/STAGE_3_2_DESIGN_NOTE.md) | Stage 3.2 design note |
| [docs/results/vte/stage3_2_seed_level_stats_analysis/](docs/results/vte/stage3_2_seed_level_stats_analysis/) | Canonical Stage 3.2 statistical outputs |
| [docs/reviewer_packages/stage3_1_3_2/](docs/reviewer_packages/stage3_1_3_2/) | Compact reviewer package |
| [docs/article_handoff/Stage3_Followup_Article_Outline.md](docs/article_handoff/Stage3_Followup_Article_Outline.md) | Article handoff outline |




---


## 🔬 Publications

### Preprint (2026)
**Gate-Rheology: Inertia of Cognitive Control Explains Meta-Rigidity in Sequential Decision Making and Reversal Learning**

*Abstract:* We introduce Gate-Rheology, a mechanistic framework in which arbitration between computational modes possesses intrinsic inertia. Across 30 seeds, we demonstrate dissociable double dissociation between control-mode inertia ($V_G$) and action perseveration ($V_p$).

#### 📈 Key Results (Stage 2)

| Metric | Result | Status |
| :--- | :--- | :--- |
| **MB/MF Signatures** | interaction coef = 0.312 ± 0.118, p = 0.008 | ✅ PASS |
| **V_G Hysteresis** | Latency = 35 trials (median), max = 699 | ✅ PASS |
| **V_G Ablation** | 35× difference in latency, p = 3.12×10⁻¹⁰ | ✅ PASS |
| **V_p Ablation** | Perseveration: 3 vs. 8, p = 5.85×10⁻⁵ | ✅ PASS |
| **Cross-task Generalization** | Identical parameters for Two-Step + Reversal | ✅ PASS |
| **Parameter Sensitivity** | 25 combinations × 30 seeds = 2,250 runs | ✅ PASS |

**Status:** ✅ Ready for submission
**Preprint:** Snigirov, Aleksey, Gate-Rheology: Inertia of Cognitive Control Explains Meta-Rigidity in Sequential Decision Making and Reversal Learning (March 19, 2026). Available at SSRN: https://ssrn.com/abstract=6442142 or http://dx.doi.org/10.2139/ssrn.6442142

---

#### 🧪 Architectural components

| Component | Description | Status |
| :--- | :--- | :--- |
| **S-O-R Primitive** | Basic Units: States, Operations, Relations | ✅ Completed |
| **Gate (v2)** | Arbitration between MF/MB modes (Stage 2) | ✅ Completed |
| **Gate (v3)** | Threshold cascade + exposure field (Stage 3) | ✅ Completed |
| **Rheology ($V_G$, $V_p$)** | Viscosity of control and action | ✅ Completed |
| **Exposure Field** | Valence/observability as a unified field | ✅ Completed |
| **Temporal State** | Compressed temporal history ($h_t$) | ✅ Completed |
| **VTE Wrapper** | Read-only trajectory measurement layer over externalized traces | ✅ Stage 3.2A complete |
| **Stage 3 VTE Adapter** | Stage 3 step-log to VTE trace-schema translator | ✅ Stage 3.2B complete |
| **Biological Comparability Layer** | Lab-trace adapters, geometry registry, and fixed-threshold comparison reports | 🟡 Stage 3.2C in development |

---

### Stage 3 follow-up paper

Working title: **Gate-Rheology and Deliberation: Viscous Control as a Source of VTE-like Behavior under Ambiguous Choice and One-Shot Valence Deformation**

Status: article handoff prepared. See: [docs/article_handoff/Stage3_Followup_Article_Outline.md](docs/article_handoff/Stage3_Followup_Article_Outline.md)


## 🤝 Contribution

This repository is part of the **Principia Cognitia** dissertation project. For collaboration inquiries, please contact the author.

### How to Contribute

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit the changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📝 License

MIT License — see [LICENSE](LICENSE) file.

---

## 📚 Key References

1. Daw, N. D., et al. (2011). Model-based influences on humans' choices and striatal prediction errors. *Neuron, 69*(6), 1204–1215.
2. Le, N. M., et al. (2023). Mixtures of strategies underlie rodent behavior during reversal learning. *PLOS Computational Biology, 19*(9), e1011430.
3. Hasz, B. M., & Redish, A. D. (2018). Deliberation and procedural automation on a two-step task for rats. *Frontiers in Integrative Neuroscience, 12*, 30.
4. Lee, S. W., Shimojo, S., & O'Doherty, J. P. (2014). Neural computations underlying arbitration between model-based and model-free learning. *Neuron, 81*(3), 687–699.
5. Redish, A. D. (2016). Vicarious trial and error. *Nature Reviews Neuroscience, 17*(3), 147-159.
6. Wilson, R. C., & Collins, A. G. E. (2019). Ten simple rules for the computational modeling of behavioral data. *eLife, 8*, e49547.

---

**Last updated:** May 2026
