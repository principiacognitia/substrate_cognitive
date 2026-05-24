# Stage 3.2 Closure

## Status

Stage 3.2 is closed as a read-only VTE-style measurement and seed-level statistical analysis layer.

This closure does not mean that all possible visualization, biological comparison, or maze-generalization work is complete. It means that the current Stage 3.2 claim set is sufficiently fixed for article handoff.

---

## Scope

Stage 3.2 adds a measurement layer over externalized behavioral traces. It does not modify the Stage 3 cognitive kernel.

The closed Stage 3.2 package includes:

1. fixed trace-schema contract;
2. Stage 3 step-log adapter;
3. VTE-style trial metrics;
4. batch analysis;
5. biological/lab adapter scaffolding under a decision-level comparability boundary;
6. seed-level statistics;
7. wrapper-sanity / model-relevant / degenerate-ablation role separation;
8. compact reviewer package builders.

---

## Canonical artifact locations

Stage 3.2 statistics:

```text
docs/results/vte/stage3_2_seed_level_stats_analysis/
```

Reviewer packages:

```text
docs/reviewer_packages/stage3_1_3_2/
```

Main documents:

```text
docs/STAGE3_2_CLOSURE.md
docs/article_handoff/Stage3_Followup_Article_Outline.md
docs/stage3_2_TBD.md
```

---

## Main claim

Stage 3.2 supports the following claim:

> VTE-like deliberation can be measured as a behavioral regime in externalized Stage 3 traces without importing internal model state, and the resulting seed-level statistics can be separated into model-relevant effects, wrapper-sanity checks, and degenerate-ablation diagnostics.

This is a measurement and analysis claim, not a biological equivalence claim.

---

## Supported subclaims

### 1. Trace-schema independence

Stage 3 logs can be translated into a fixed VTE trace schema.

The VTE layer reads the trace as data. It does not call Stage 3 internals, agent objects, Gate state, or reward/threat configuration classes.

### 2. VTE-style behavioral measurement

The wrapper computes:

- IdPhi-like angular integration;
- pause duration;
- reorientation count;
- choice-point duration;
- VTE-like binary classification under fixed thresholding.

### 3. Seed-level inference

Patch 20B performs seed-level statistical tests.

Patch 20D/20E improve presentation and interpretation without changing the underlying statistical tests.

The seed is treated as the inferential unit. Trial rows are measurement observations.

### 4. Role-aware statistical interpretation

Patch 20E separates statistical rows into:

- `model_relevant_test`;
- `wrapper_sanity_check`;
- `degenerate_ablation_diagnostic`;
- `diagnostic`.

The separation prevents circular overinterpretation of IdPhi/pause/reorientation effects as independent model validation.

### 5. Biological decision-level comparability

Stage 3.2 includes biological/lab adapter work only under a restricted boundary.

The current claim is decision-level and schema-level comparability, not biological trajectory replay equivalence.

---

## Explicit non-claims

Stage 3.2 does not claim:

- rodent-level VTE equivalence;
- biological neural mechanism identity;
- allocentric spatial cognition;
- full W-maze or RROW equivalence;
- absence inference;
- self-model-based visibility reasoning;
- movement-level biological trajectory replay;
- dataset-specific threshold tuning.

---

## Relationship to Stage 3.1B

Stage 3.1B closes the valence/exposure kernel:

- balanced reward-threat conflict;
- one-shot shock/treat deformation;
- positive and negative carrier separation;
- ablation-localized effects;
- placebo-window controls.

Stage 3.2 does not change this kernel. It measures VTE-like regimes over traces produced by the Stage 3.1B system.

---

## Relationship to the Stage 2 Gate-Rheology paper

The Stage 2 Gate-Rheology paper established the core idea that control-mode arbitration has viscosity.

Stage 3 extends that idea into spatial/exposure-like choice:

- Stage 3.1B shows one-shot valence deformation and carrier persistence.
- Stage 3.2 shows that ambiguous or weakly dominated choice can be analyzed through VTE-like measurement.
- The follow-up article should treat deliberation-like behavior as a measurable regime of viscous Gate dynamics, not as a new special-purpose deliberation module.

---

## Reviewer package

Build the consolidated reviewer package with:

```bash
python -m stage3.analysis.build_stage3_reviewer_package \
  --preset stage3_1_3_2 \
  --profile llm5 \
  --results-root docs/results \
  --output-dir docs/reviewer_packages/stage3_1_3_2 \
  --clean
```

Expected output:

```text
Figure_Stage3_Reviewer_Page_Diagnostics.png
Figure_Stage3_Reviewer_Page_Main.png
Stage3_Key_Tables.md
Stage3_Reviewer_Report.md
reviewer_package_registry.json
```

---

## Closure tests

Recommended final checks:

```bash
python -m pytest stage3/tests
python -m pytest vte/tests
python -m stage3.analysis.build_stage3_reviewer_package \
  --preset stage3_1_3_2 \
  --profile llm5 \
  --results-root docs/results \
  --output-dir docs/reviewer_packages/stage3_1_3_2 \
  --clean
```

---

## Deferred work

Deferred work is tracked in:

```text
docs/stage3_2_TBD.md
```

Deferred work is not required for Stage 3.2 closure.