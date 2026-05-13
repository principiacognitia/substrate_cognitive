# Stage 3 Reproduction Guide

Patch: 20A  
Scope: documentation only  
Target: external reproduction of Stage 3.1A and Stage 3.1B results  
Status: command guide and artifact map

## 1. Purpose

This document defines how an external reviewer should reproduce the Stage 3 results at a high level.

It distinguishes four layers:

1. environment setup;
2. smoke tests and architecture checks;
3. Stage 3.1A baseline reproduction;
4. Stage 3.1B matrix, one-shot, ablation, and reviewer-package reproduction.

Exact runner module paths may differ across branches. The commands below use the intended module names where available and direct-script fallbacks where needed.

## 2. Repository assumptions

Run commands from repository root:

```bash
cd substrate_cognitive
```

Expected project layout:

```text
stage3/
    core/
        agent_stage3.py
        gate_inputs.py
        gate_stage3.py
        temporal_state.py
    configs/
        config_stage3_1a.py
        config_stage3_1b.py
    envs/
    tests/
    analysis/
docs/
    results/
logs/
    stage3/
```

## 3. Environment setup

Create and activate an isolated Python environment.

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If the repository uses local editable imports, install editable mode:

```bash
python -m pip install -e .
```

If editable mode is unavailable, run from repository root and make sure the root is on `PYTHONPATH`.

Windows PowerShell:

```powershell
$env:PYTHONPATH = (Get-Location).Path
```

Linux/macOS:

```bash
export PYTHONPATH="$PWD"
```

## 4. Basic validation

### 4.1 Config smoke tests

Run Stage 3.1A config validation:

```bash
python stage3/configs/config_stage3_1a.py
```

Run Stage 3.1B config import smoke check:

```bash
python - <<'PY'
from stage3.configs.config_stage3_1b import CONFIG_3_1B, get_condition_grid, get_canonical_conditions
print("matrix cells:", len(get_condition_grid()))
print("canonical:", sorted(get_canonical_conditions().keys()))
print("ablation:", sorted(CONFIG_3_1B["ablation"].keys()))
PY
```

Expected:

```text
matrix cells: 9
canonical: ['balanced_conflict', 'reward_dominant', 'threat_dominant']
ablation: ['full', 'novg', 'novp', 'nox', 'one_shot_off']
```

### 4.2 Core smoke tests

Run the core module smoke tests:

```bash
python stage3/core/gate_stage3.py
python stage3/core/temporal_state.py
```

Expected: all embedded smoke tests pass.

### 4.3 Pytest suite

Run Stage 3 tests:

```bash
python -m pytest stage3/tests
```

Optional verbose run:

```bash
python -m pytest -v stage3/tests
```

## 5. Stage 3.1A baseline reproduction

### 5.1 Purpose

Stage 3.1A reproduces the frozen open/covered baseline.

Expected qualitative result:

- stable preference for covered path;
- nonzero junction deliberation proxies;
- Bernoulli reward variability;
- non-identical seed trajectories.

### 5.2 Typical command

If the repository exposes a module runner:

```bash
python -m stage3.analysis.run_stage3_1a
```

If the runner is a script:

```bash
python stage3/analysis/run_stage3_1a.py
```

If the local branch uses a different script name, use the Stage 3.1A runner that imports:

```python
from stage3.configs.config_stage3_1a import CONFIG_3_1A
```

and writes to:

```text
logs/stage3/stage3_1a/
```

### 5.3 Expected primary outputs

Expected output directory:

```text
logs/stage3/stage3_1a/
```

Expected artifact types:

```text
trial-level CSV
step-level CSV or JSONL
seed summary table
block dynamics table
covered-rate figure
```

Expected reviewer-facing figure:

```text
Figure_R1_Stage3_1A_Block_Covered_Rate.png
```

Expected qualitative block result:

```text
P(covered) remains approximately stable across four 25-trial blocks.
```

## 6. Stage 3.1B matrix reproduction

### 6.1 Purpose

The matrix run tests reward x threat tradeoff sensitivity across nine conditions.

Condition grid:

```text
R0_T1 R1_T1 R2_T1
R0_T2 R1_T2 R2_T2
R0_T3 R1_T3 R2_T3
```

### 6.2 Typical command

Module runner:

```bash
python -m stage3.analysis.run_stage3_1b
```

Script fallback:

```bash
python stage3/analysis/run_stage3_1b.py
```

The runner should import:

```python
from stage3.configs.config_stage3_1b import CONFIG_3_1B, get_condition_grid
```

### 6.3 Expected outputs

Expected artifact family:

```text
docs/results/stage3_1b_closure/
logs/stage3/stage3_1b/
```

Expected matrix tables:

```text
Table_Stage3_1B_Matrix_Cells.csv
Stage3_Reviewer_Tables.xlsx sheet: 3.1B_matrix_cells
```

Expected matrix figures:

```text
Figure_R2_*P_Open*.png
Figure_R3_*P_Timeout*.png
Figure_R4_*Commit_Latency*.png
```

Expected qualitative result:

- P(open) increases with reward level under mild/moderate threat;
- high threat suppresses open choice;
- timeout and latency peak near conflict rather than at extremes.

## 7. One-shot protocol reproduction

### 7.1 Purpose

One-shot protocols test whether a single high-amplitude event can deform subsequent behavior through continuous importance traces.

Default protocol:

```text
condition: balanced_conflict / R1_T2
event trial: 30
total trials: 100
pre block: 30 trials
post block: 69 trials
path: open
shock reward: -5.0
treat reward: +5.0
salience: 0.9
stakes: 10.0
```

### 7.2 Typical command

Module runner, if available:

```bash
python -m stage3.analysis.run_stage3_1b_one_shot
```

Script fallback:

```bash
python stage3/analysis/run_stage3_1b_one_shot.py
```

If the current closure branch uses a combined runner, the one-shot mode should call:

```python
get_one_shot_protocol(kind="shock")
get_one_shot_protocol(kind="treat")
```

from:

```python
stage3.configs.config_stage3_1b
```

### 7.3 Expected outputs

Expected effect tables:

```text
one_shot_effects
one_shot_carriers
ablation_localization
acceptance_summary
```

Expected figures:

```text
Figure_R5_*Shock_Target_Choice*.png
Figure_R6_*Shock_QNeg*.png
Figure_R7_*Shock_HRisk*.png
Figure_R8_*Treat_Target_Choice*.png
Figure_R9_*Treat_QPos*.png
```

Expected qualitative result:

- shock reduces post-event target-path choice;
- treat increases post-event target-path choice;
- `q_neg` and `h_risk` carry negative effects;
- `q_pos` and source-local opportunity carry positive effects.

## 8. Ablation reproduction

### 8.1 Purpose

Ablations localize which implementation-level channels carry the effects.

Ablation set:

```text
full
novg
novp
nox
one_shot_off
```

### 8.2 Typical command

Module runner:

```bash
python -m stage3.analysis.analyze_stage3_1b_ablation_suite
```

Script fallback:

```bash
python stage3/analysis/analyze_stage3_1b_ablation_suite.py
```

### 8.3 Expected outputs

Expected wide table:

```text
Stage3_Reviewer_Tables.xlsx sheet: 3.1B_ablation_wide
```

Expected localization table:

```text
Stage3_Reviewer_Tables.xlsx sheet: ablation_localization
```

Expected summary figure:

```text
Figure_R10_OneShot_Effect_By_Ablation.png
```

Expected qualitative pattern:

- `one_shot_off` removes both shock and treat one-shot effects;
- `novp` primarily affects source-local positive/treat effects;
- `nox` diagnoses exposure-derived contribution, especially for negative/shock effects;
- `novg` diagnoses accumulated temporal-risk contribution to Gate routing.

## 9. Acceptance validation

### 9.1 Purpose

Acceptance checks validate schema, directionality, architecture invariants, and expected diagnostic properties.

They are not identical to inferential statistical tests.

### 9.2 Typical command

Module runner:

```bash
python -m stage3.analysis.validate_stage3_1b_acceptance
```

Script fallback:

```bash
python stage3/analysis/validate_stage3_1b_acceptance.py
```

### 9.3 Expected outputs

Expected table:

```text
Stage3_Reviewer_Tables.xlsx sheet: acceptance_summary
```

Expected fields:

```text
protocol
ablation
check
status
value
threshold_or_expectation
note
```

Expected result for closure package:

```text
no safety/acceptance failures
```

## 10. Reviewer package generation

### 10.1 Recommended compact package

For external review, avoid uploading each figure separately. Use a compact package:

```text
Stage3_Reviewer_Report.md
Stage3_Reviewer_Tables.xlsx
Stage3_Reviewer_Figures.pdf
Stage3_Reviewer_Manifest.json
Stage3_Reproduction_Commands.md
```

Optional:

```text
Stage3_Limitations.md
```

### 10.2 Typical command

If available:

```bash
python -m stage3.analysis.export_stage3_reviewer_package
```

Script fallback:

```bash
python stage3/analysis/export_stage3_reviewer_package.py
```

If the current branch does not have a single exporter, assemble manually:

1. collect all Figure R1-R10 images;
2. combine them into `Stage3_Reviewer_Figures.pdf`;
3. export all CSV summaries into `Stage3_Reviewer_Tables.xlsx`;
4. write manifest with source commit, branch, run ids, file hashes, and output paths.

### 10.3 Minimal figure bundling script

Use this local helper if needed:

```python
from pathlib import Path
from PIL import Image

fig_dir = Path("docs/results/stage3_1b_closure")
figures = sorted(fig_dir.glob("Figure_R*.png"))

images = []
for path in figures:
    img = Image.open(path).convert("RGB")
    images.append(img)

out = Path("Stage3_Reviewer_Figures.pdf")
images[0].save(out, save_all=True, append_images=images[1:])
print(out)
```

## 11. Expected artifact map

| Stage | Main artifact | Reviewer-package location |
|---|---|---|
| 3.1A baseline | block dynamics | `Figure_R1`, XLSX `3.1A_block_dynamics` |
| 3.1B matrix | path choice / timeout / latency | `Figure_R2-R4`, XLSX `3.1B_matrix_cells` |
| one-shot shock | target choice, q_neg, h_risk | `Figure_R5-R7`, XLSX `one_shot_effects`, `one_shot_carriers` |
| one-shot treat | target choice, q_pos | `Figure_R8-R9`, XLSX `one_shot_effects`, `one_shot_carriers` |
| ablation localization | one-shot effect by ablation | `Figure_R10`, XLSX `3.1B_ablation_wide`, `ablation_localization` |
| acceptance | pass/fail checks | XLSX `acceptance_summary` |

## 12. Reproduction checklist

Before external release, verify:

- [ ] repository commit hash recorded;
- [ ] branch name recorded;
- [ ] working tree status recorded;
- [ ] dependency environment recorded;
- [ ] Stage 3 tests pass;
- [ ] Stage 3.1A baseline outputs regenerated;
- [ ] Stage 3.1B matrix outputs regenerated;
- [ ] one-shot outputs regenerated for shock and treat;
- [ ] all five ablations regenerated;
- [ ] acceptance summary has no unexpected failures;
- [ ] all figures R1-R10 present;
- [ ] tables exported to XLSX;
- [ ] SHA256 hashes generated for reviewer-package files;
- [ ] limitations documented separately from claims.

## 13. Claim-to-artifact map

| Claim | Primary evidence | Supporting artifact |
|---|---|---|
| Stage 3.1A has stable covered bias | block dynamics | `Figure_R1`, XLSX `3.1A_block_dynamics` |
| Stage 3.1B is reward/threat sensitive | 3 x 3 matrix | `Figure_R2-R4`, XLSX `3.1B_matrix_cells` |
| Shock produces negative one-shot deformation | pre/post target choice and carriers | `Figure_R5-R7`, XLSX `one_shot_effects`, `one_shot_carriers` |
| Treat produces positive one-shot deformation | pre/post target choice and q_pos | `Figure_R8-R9`, XLSX `one_shot_effects`, `one_shot_carriers` |
| One-shot effects require one-shot mechanism | `one_shot_off` ablation | `Figure_R10`, XLSX `ablation_localization` |
| Effects are component-localizable | ablation suite | XLSX `3.1B_ablation_wide`, `ablation_localization` |
| Engineering invariants hold | acceptance checks | XLSX `acceptance_summary` |

## 14. Known limitations to preserve in reproduction notes

Do not claim:

- biological equivalence of synthetic trajectories;
- direct rodent movement reproduction;
- symbolic episodic memory;
- object permanence;
- semantic object recognition;
- complete removal of psychological constructs by ablations.

Do claim only:

- reproducible synthetic control dynamics;
- reward/threat-sensitive path choice;
- continuous one-shot trace dynamics;
- component-level diagnostic ablation patterns;
- VTE-like proxy metrics suitable for later comparison, not direct biological identity.
