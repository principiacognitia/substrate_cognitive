
# Documentation and Result Packages

This directory contains project specifications, notes, manuscripts, and curated
result packages.

Raw experiment logs are not stored here. Raw outputs are generated under
`logs/`, while publication-facing selected outputs are copied into
`docs/results/`.

---

## Structure

```text
docs/
├── results/
│   ├── stage3_1a/
│   └── stage3_1b_closure/
├── SPECIFICATION3.md
├── mapping.md
└── README.md
```

---

## Curated results

### Stage 3.1A

```text
docs/results/stage3_1a/
├── figures/
├── tables/
├── stats/
├── reports/
├── artifact_registry.json
└── ARTIFACT_REGISTRY.md
```

Stage 3.1A in the closure package is a compatibility rerun. It is used to check
that Stage 3.1B changes preserve the calibrated Stage 3.1A baseline behavior.

### Stage 3.1B closure

```text
docs/results/stage3_1b_closure/
├── figures/
├── tables/
├── stats/
├── reports/
├── artifact_registry.json
└── ARTIFACT_REGISTRY.md
```

Stage 3.1B closure is split into four artifact layers:

1. `matrix`: 3x3 reward x threat conflict surface.
2. `balanced ablation`: balanced-conflict ablation metrics.
3. `one-shot shock/treat`: event-aligned carryover.
4. `diagnostics`: placebo-window, carrier, and ablation-localization checks.

---

## Regenerating curated result packages

Smoke profile:

```bash
python -m stage3.analysis.run_stage3_1_closure_package --mode smoke
python -m stage3.analysis.validate_stage3_1_artifacts --root docs/results
python -m stage3.analysis.validate_stage3_1b_acceptance --results-root docs/results --mode smoke
```

Full production profile:

```bash
python -m stage3.analysis.run_stage3_1_closure_package --mode full
python -m stage3.analysis.validate_stage3_1_artifacts --root docs/results
python -m stage3.analysis.validate_stage3_1b_acceptance --results-root docs/results --mode full
```

The full production profile should be run from a clean working tree, after code
and documentation patches have been committed.

---

## Raw outputs

Raw outputs are generated under:

```text
logs/stage3/stage3_1_closure_raw/
```

These outputs are local working artifacts and should not be treated as the
publication-facing package. The publication-facing package is the curated copy
under `docs/results/`.

---

## Artifact registry

Each curated result package contains:

```text
artifact_registry.json
ARTIFACT_REGISTRY.md
```

The registry is the canonical index for figures, tables, reports, and stats. File
names are stable, but article assembly should use the registry rather than
inferring scientific meaning only from filenames.
