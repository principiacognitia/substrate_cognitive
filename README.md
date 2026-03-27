# Principia Cognitia: Substrate-Independent Cognitive Architecture

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Status: Stage 2 Complete](https://img.shields.io/badge/status-stage--2--complete-green)](https://github.com/principiacognitia/substrate_cognitive)
[![Stage 3: Development](https://img.shields.io/badge/stage--3-development-orange)](stage3/)

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
| **Stage 2** | Two-Step + Reversal Task | ✅ **Complete** | [Preprint](docs/Gate-Rheology%20-%20Inertia%20of%20Cognitive%20Control%20Explains%20Meta-Rigidity%20in%20Sequential%20Decision%20Making%20and%20Reversal%20Learning.pdf), [stage2/README.md](stage2/README.md) |
| **Stage 3.0** | Architectural Refactor | 🟡 **In Development** | [SPECIFICATION3.md](docs/SPECIFICATION3.md), [stage3/README.md](stage3/README.md) |
| **Stage 3.1** | Open/Covered Choice Maze | ⬜ Planned | — |
| **Stage 3.2** | Shelter Cross / Absence Check | ⬜ Planned | — | ---

## 🗂️ Repository Structure

```
substrate_cognitive/
├── mvp/                        # ✅ Completed MVP (T-maze, v5.0)
├── stage2/                     # ✅ Stage 2: Experimental Validation
│   ├── core/                   # Core modules (gate, rheology, baselines)
│   ├── twostep/                # Two-Step Task (Daw et al., 2011)
│   ├── reversal/               # Block-Reversal Task (Le et al., 2023)
│   └── analysis/               # Analysis and visualization scripts
├── stage3/                     # 🟡 Stage 3: Architectural Refactor
│   ├── core/                   # Gate v3, exposure_field, temporal_state
│   ├── envs/                   # Dummy temporal environments
│   └── tests/                  # Integration tests
├── docs/                       # 📚 Documentation
│   ├── SPECIFICATION3.md       # Stage 3.0 Technical Specification
│   └── mapping.md              # Latent variables → observable metrics
├── logs/                       # Experimental logs (ignored by git)
├── tests/                      # Automated tests
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

### Stage 2: Reproducing Results

```bash
# Generate all figures for the preprint
python -m stage2.analysis.run_all

# Run a specific experiment
python -m stage2.analysis.run_all --experiment-id twostep_ablation_20260310_195529

# Figure 3 only (V_G dynamics)
python -m stage2.analysis.run_all --figure 3
```

### Stage 3: Running Tests

```bash
# Unit tests for Stage 3.0
python -m stage3.core.gate_inputs
python -m stage3.core.gate_modes
python -m stage3.core.exposure_field
python -m stage3.core.temporal_state
python -m stage3.core.gate_stage3
python -m stage3.core.agent_stage3
python -m stage3.core.compatibility
```

---

## 📈 Key Results (Stage 2)

| Metric | Result | Status |
| :--- | :--- |
| **MB/MF Signatures** | interaction coef = 0.312 ± 0.118, p = 0.008 | ✅ PASS |
| **V_G Hysteresis** | Latency = 35 trials (median), max = 699 | ✅ PASS |
| **V_G Ablation** | 35× difference in latency, p = 3.12×10⁻¹⁰ | ✅ PASS |
| **V_p Ablation** | Perseveration: 3 vs. 8, p = 5.85×10⁻⁵ | ✅ PASS |
| **Cross-task Generalization** | Identical parameters for Two-Step + Reversal | ✅ PASS |
| **Parameter Sensitivity** | 25 combinations × 30 seeds = 2,250 runs | ✅ PASS |

---

## 📄 Documentation

| Document | Description |
| :--- | :--- |
| [Stage 2 Overview](stage2/README.md) | Details on current development and result reproduction |
| [Stage 3.0 Overview](stage3/README.md) | Architectural refactor and quick start guide |
| [Stage 3.0 Specification](docs/SPECIFICATION3.md) | Complete technical specification for Stage 3.0 |
| [Mapping Document](docs/mapping.md) | Correspondence between latent variables and observable metrics |
| [Preprint Manuscript](docs/Gate-Rheology%20-%20Inertia%20of%20Cognitive%20Control%20Explains%20Meta-Rigidity%20in%20Sequential%20Decision%20Making%20and%20Reversal%20Learning.pdf) | Full Manuscript for bioRxiv |

---

## 🔬 Publications

### In Preparation (2026)
**Gate-Rheology: Inertia of Cognitive Control Explains Meta-Rigidity in Sequential Decision Making and Reversal Learning**

*Abstract:* We introduce Gate-Rheology, a mechanistic framework in which arbitration between computational modes possesses intrinsic inertia. Across 30 seeds, we demonstrate dissociable double dissociation between control-mode inertia ($V_G$) and action perseveration ($V_p$).

**Status:** ✅ Ready for bioRxiv submission
**Preprint:** [GitHub Repository](https://github.com/principiacognitia/substrate_cognitive)

---

## 🧪 Architectural components

| Component | Description | Status |
| :--- | :--- | :--- |
| **S-O-R Primitive** | Basic Units: States, Operations, Relations | ✅ Completed |
| **Gate (v2)** | Arbitration between MF/MB modes (Stage 2) | ✅ Completed |
| **Gate (v3)** | Threshold cascade + exposure field (Stage 3) | 🟡 In development |
| **Rheology ($V_G$, $V_p$)** | Viscosity of control and action | ✅ Completed |
| **Exposure Field** | Valence/observability as a unified field | 🟡 In development |
| **Temporal State** | Compressed temporal history ($h_t$) | 🟡 In development |

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

**Last updated:** March 2026
