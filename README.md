# Principia Cognitia: Substrate-Independent Cognitive Architecture

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Status: Stage 2 Complete](https://img.shields.io/badge/status-stage--2--complete-green)](https://github.com/principiacognitia/substrate_cognitive)
[![Status: Stage 3.1 Complete](https://img.shields.io/badge/status-stage--3.1--complete-green)](https://github.com/principiacognitia/substrate_cognitive)
[![Stage 3.2: VTE](https://img.shields.io/badge/stage--3.2-VTE--in--development-orange)](vte/)

**Author:** Alex Snow (Aleksey L. Snigirov)
**Email:** alex2saaba@gmail.com
**ORCID:** 0009-0001-3713-055X
**GitHub:** https://github.com/principiacognitia/substrate_cognitive

---

## 📖 Description

**Principia Cognitia** is a research framework for modeling cognitive systems capable of functioning across various substrates (both biological and artificial). The architecture is grounded in the principles of **Gate-Rheology**—a mechanistic model for arbitrating between computational modes, each possessing its own inherent inertia.

### Key Idea

Cognitive rigidity does not stem from the content of representations, but rather from the **dynamics of control mode selection**. The arbitration between modes (exploit vs. explore) possesses its own intrinsic viscosity ($V_G$), which accumulates over time and exhibits hysteresis.

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
| **Stage 3.2B** | Stage 3 log adapter + batch analysis | 🟡 **Final debug** | [vte/README.md](vte/README.md) |
| **Stage 3.2C** | Biological-lab comparability layer | 🟡 **In development** | [vte/STAGE_3_2_DESIGN_NOTE.md](vte/STAGE_3_2_DESIGN_NOTE.md) |

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
├── vte/                        # 🟡 Stage 3.2 VTE service layer
│   ├── core/                   # ✅ Read-only VTE wrapper and metrics
│   ├── adapters/               # 🟡 Stage 3 trace adapter and future lab adapters
│   ├── configs/                # 🟡 Wrapper and analysis defaults
│   ├── tests/                  # ✅ VTE schema, wrapper, adapter, and analysis tests
│   └── analysis/               # 🟡 CLI runners, analyzers, and batch workflows
├── docs/                       # Specifications, notes, curated result packages
│   └── results/                # Publication-facing curated outputs
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

## Interpretation boundary

Stage 3.1B supports the valence/exposure kernel claim. It does not by itself claim absence inference, allocentric spatial cognition, self-model-based visibility reasoning, or rodent-level VTE equivalence.

Those claims require later stages or separate protocols.

Stage 3.2 currently supports trajectory-level VTE-style measurement only. The current Stage 3 adapter uses synthetic poses reconstructed from Stage 3 step logs. Biological comparison requires a separate lab-data adapter, fixed geometry registry, and predeclared thresholding policy.

---

## Tests

```bash
python -m pytest stage3/tests
```

The Stage 3 test suite currently covers compatibility, gate routing, temporal
state, no-ready-semions constraints, action-basin equivalence, Stage 3.1B config,
and one-shot protocol behavior.

---

## Documentation map

| Path | Role |
| :--- | :--- |
| [docs/README.md](docs/README.md) | Documentation and curated result package map |
| [stage3/README.md](stage3/README.md) | Stage 3 architecture and Stage 3.1 closure commands |
| [stage3_1b_closure/README.md](docs/results/stage3_1b_closure/README.md) | Generated closure package README |
| [STAGE3_1B_CLOSURE_REPORT.md](docs/results/stage3_1b_closure/reports/STAGE3_1B_CLOSURE_REPORT.md) | Generated closure report |
| [ARTIFACT_REGISTRY.md](docs/results/stage3_1b_closure/ARTIFACT_REGISTRY.md) | Generated artifact registry |
| [vte/README.md](vte/README.md) | Stage 3.2 VTE wrapper service layer |
| [vte/STAGE_3_2_DESIGN_NOTE.md](vte/STAGE_3_2_DESIGN_NOTE.md) | Stage 3.2 design note and biological-comparability boundary |


---

---

## Stage 3.2 VTE workflow

Stage 3.2 is a read-only measurement layer over externalized behavioral traces.
It is intentionally separated from `stage3/`.

### Stage 3 step-log adapter

```bash
python -m vte.analysis.translate_stage3_steps_to_vte_trace \
  --input-csv logs/stage3/stage3_1_closure_raw/stage3_1b/<suite>/balanced/full/balanced_conflict_full_all_steps.csv \
  --output-csv logs/vte/raw/balanced_conflict_full_trace.csv \
  --run-id balanced_conflict_full
```

### VTE wrapper

```bash
python -m vte.analysis.run_stage3_2_vte \
  --input-csv logs/vte/raw/balanced_conflict_full_trace.csv \
  --output-dir logs/vte/stage3_2_smoke
```

### VTE analysis

```bash
python -m vte.analysis.analyze_stage3_2_vte \
  --metrics-csv logs/vte/stage3_2_smoke/vte_trial_metrics.csv \
  --output-dir logs/vte/stage3_2_smoke_analysis
```

### Tests

```bash
python -m pytest vte/tests
python -m pytest stage3/tests
```




## 🔬 Publications

### In Preparation (2026)
**Gate-Rheology: Inertia of Cognitive Control Explains Meta-Rigidity in Sequential Decision Making and Reversal Learning**

*Abstract:* We introduce Gate-Rheology, a mechanistic framework in which arbitration between computational modes possesses intrinsic inertia. Across 30 seeds, we demonstrate dissociable double dissociation between control-mode inertia ($V_G$) and action perseveration ($V_p$).

## 📈 Key Results (Stage 2)

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

## 🧪 Architectural components

| Component | Description | Status |
| :--- | :--- | :--- |
| **S-O-R Primitive** | Basic Units: States, Operations, Relations | ✅ Completed |
| **Gate (v2)** | Arbitration between MF/MB modes (Stage 2) | ✅ Completed |
| **Gate (v3)** | Threshold cascade + exposure field (Stage 3) | ✅ Completed |
| **Rheology ($V_G$, $V_p$)** | Viscosity of control and action | ✅ Completed |
| **Exposure Field** | Valence/observability as a unified field | ✅ Completed |
| **Temporal State** | Compressed temporal history ($h_t$) | ✅ Completed |
| **VTE Wrapper** | Read-only trajectory measurement layer over externalized traces | ✅ Stage 3.2A complete |
| **Stage 3 VTE Adapter** | Stage 3 step-log to VTE trace-schema translator | 🟡 Stage 3.2B final debug |
| **Biological Comparability Layer** | Lab-trace adapters, geometry registry, and fixed-threshold comparison reports | 🟡 Stage 3.2C in development |

---

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
5. Wilson, R. C., & Collins, A. G. E. (2019). Ten simple rules for the computational modeling of behavioral data. *eLife, 8*, e49547.

---

**Last updated:** May 2026
