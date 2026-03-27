# Stage 2: Experimental Validation Package

**Status:** ✅ **Complete** (Ready for bioRxiv submission)

**Preprint:** Gate-Rheology: Inertia of Cognitive Control Explains Meta-Rigidity in Sequential Decision Making and Reversal Learning

---

## 📖 Description

Stage 2 is a complete experimental validation package for **Gate-Rheology**—a mechanistic model of arbitration between computational modes with inherent inertia.

### Key Results

1. **MB/MF Signatures Replicated:** The RheologicalAgent exhibits the canonical reward × transition interaction (p = 0.008).
2. **Heavy-Tailed Latency:** The Full agent demonstrates a median latency of 35 trials (max = 699).
3. **Double Dissociation:** $V_G$ ablation eliminates latency; $V_p$ ablation reduces perseveration.
4. **Cross-Task Generalization:** Identical parameters used for both the Two-Step and Reversal tasks.

---

## 🗂️ Structure

```
stage2/
├── core/                       # Core modules
│   ├── gate.py                 # Gate v2 (Stage 2)
│   ├── rheology.py             # V_G, V_p dynamics
│   └── baselines.py            # MF, MB, Hybrid agents
├── twostep/                    # Two-Step Task
│   ├── env_twostep.py          # Two-Step environment
│   ├── run_twostep.py          # Run experiment
│   └── run_sanity_check.py     # Sanity check (MF/MB only)
├── reversal/                   # Reversal Task
│   ├── env_reversal.py         # Reversal environment
│   └── run_reversal.py         # Run experiment
├── analysis/                   # Analysis and visualization
│   ├── run_all.py              # Generate all plots
│   ├── loaders.py              # Data loaders
│   └── plots/                  # Plots by type
│       ├── stay_prob.py        # Figure 2
│       ├── vg_dynamics.py      # Figure 3
│       └── reversal.py         # Figure 4, 4B
└── README.md                   # This file
```

---

## 🚀 Quick Start

### Reproducing Results

```bash
# Generate all plots (automatically searches for experiments)
python -m stage2.analysis.run_all

# Run on a specific experiment
python -m stage2.analysis.run_all --experiment-id twostep_ablation_20260310_195529

# Figure 2 only (MB/MF Signatures)
python -m stage2.analysis.run_all --figure 2

# Figure 3 only (V_G Dynamics)
python -m stage2.analysis.run_all --figure 3

# Figure 4 only (Reversal)
python -m stage2.analysis.run_all --figure 4
```

### Running Experiments

```bash
# Two-Step Task (ablation study)
python -m stage2.twostep.run_twostep --n-seeds 30

# Two-Step Task (sanity check)
python -m stage2.twostep.run_sanity_check --n-seeds 5

# Reversal Task
python -m stage2.reversal.run_reversal --n-seeds 30
```

---

## 📊 Results

### Stage 2A: Two-Step Task

| Agent | MB Interaction | MF Main Effect | Switching Latency |
| :--- | :--- | :--- | :--- |
| **MF-only** | 0.028 (p = 0.455) | 0.998 (p < 0.001) | — |
| **MB-only** | 0.366 (p = 0.0006) | 0.002 (p = 0.989) | — |
| **Full** | 0.312 (p = 0.008) | 0.847 (p < 0.001) | 35 (median) |
| **NoVG** | — | — | 1 (median) |
| **NoVp** | — | — | 23 (median) | **Statistical comparison:**
- Full vs NoVG latency: p = 3.12 × 10⁻¹⁰ (large effect)
- Full vs NoVp latency: p = 0.528 (negligible effect)

### Stage 2B: Reversal Task

| Agent | Perseverative Errors | Latency to Explore | Stickiness |
| :--- | :--- | :--- | :--- |
| **Full** | 8 (median) | 537 (median) | 87.8% |
| **NoVG** | 2 (median) | 0 (median) | 64.4% |
| **NoVp** | 3 (median) | 201 (median) | 77.1% |

**Statistical comparison:**
- Full vs NoVG latency: p = 2.09 × 10⁻⁷
- Full vs NoVp perseveration: p = 5.85 × 10⁻⁵
- Full vs NoVp latency: p = 0.711 (not significant)

---

## 📈 Figures

All figures are saved in `logs/figures/`:

| File | Description |
| :--- | :--- |
| `Figure_2_Signatures.png` | MB/MF Signatures (3 panels: MF, MB, Rheo) |
| `Figure_3_VG_Dynamics.png` | V_G dynamics around the changepoint |
| `Figure_4_Reversal.png` | Reversal learning curves (2000 trials) |
| `Figure_4B_Reversal_Zoomed.png` | Zoomed view (-50/+100 around the reversal) |
| `Supplementary_Figure_1A_B.png` | Kaplan-Meier survival curves |
| `Supplementary_Figure_2_Heatmaps.png` | Parameter sensitivity analysis | ---

## 🧪 Analysis

### Generating Statistics

```python
from stage2.analysis.loaders import load_experiment_data, load_meta_data

# Load data
data = load_experiment_data('logs/twostep/twostep_ablation_20260310_195529/')
meta = load_meta_data('logs/twostep/twostep_ablation_20260310_195529/')

# Access data by agent type
full_df = data['Full']
novg_df = data['NoVG']
novp_df = data['NoVp']

# Statistics
print(f"Full: {len(full_df)} trials")
print(f"NoVG: {len(novg_df)} trials")
print(f"NoVp: {len(novp_df)} trials")
```

### Seed-level Regression

```python
import statsmodels.api as sm

# For each seed separately (avoiding pseudoreplication)
for seed in range(42, 72):
seed_df = full_df[full_df['seed'] == seed]

# Logistic regression: Stay ~ Reward × Transition
X = seed_df[['reward', 'trans_factor', 'interaction']]
y = seed_df['stay']

model = sm.Logit(y, sm.add_constant(X))
results = model.fit()

# Save coefficients
coefficients.append(results.params)
```

---

## 📄 Documents

| Document | Description |
| :--- | :--- |
| [Preprint Manuscript](../Gate-Rheology-Final.md) | Full manuscript for bioRxiv |
| [Supplementary Materials](../Gate-Rheology-Final.md#supplementary-materials) | Supplementary methods and tables |
| [Figure Captions](../Gate-Rheology-Final.md#figure-captions) | Captions for all figures |

---

## 🔧 Configuration

### Agent Parameters (Identical for all tasks)

| Parameter | Value | Description |
| :--- | :--- | :--- |
| $\alpha$ | 0.35 | Learning rate |
| $\beta$ | 4.0 | Inverse softmax temperature |
| $\theta_{MB}$ | 0.30 | Mode switch threshold |
| $\theta_U$ | 1.5 | Uncertainty baseline |
| $\tau_{vol}$ | 0.50 | Volatility threshold |
| $k_{use}$ | 0.08 | Hardening rate |
| $k_{melt}$ | 0.20 | Melting rate |
| $\lambda$ | 0.01 | Decay rate |

**Note:** No parameter tuning was performed between tasks. ---

## 📊 Statistical Analysis

### Seed-level analysis

- **30 independent seeds** (42–71)
- Logistic regression **per seed** (not per trial)
- Avoidance of pseudoreplication (Wilson & Collins, 2019)

### Planned comparisons

- **Full > NoVG** for switching latency ($V_G$ effect)
- **Full > NoVp** for perseverative errors ($V_p$ effect)
- **Bonferroni correction:** α_corrected = 0.017

### Distribution analysis

- **Median and IQR** as primary statistics
- **Mann-Whitney U** (non-parametric)
- **Kaplan-Meier survival** in Supplementary Materials

---

## 🐛 Troubleshooting

### Error: "Experiment not found"

```bash
# Ensure the folder exists
ls logs/twostep/
ls logs/reversal/

# Run using the full path
python -m stage2.analysis.run_all --input-dir logs/twostep/twostep_ablation_20260310_195529/
```

### Error: "No data for agent 'Full'"

```bash
# Check that the CSV files contain the correct agent names
# Format: {exp_id}_Full_seed{N}_trials.csv

# Rerun the experiment
python -m stage2.twostep.run_twostep --n-seeds 30
```

---

## 📬 Contact

**Author:** Alex Snow (Aleksey L. Snigirov)
**Email:** alex2saaba@gmail.com
**ORCID:** 0009-0001-3713-055X

---

**Last updated:** March 2026