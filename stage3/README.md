
# Stage 3: Gate v3, Exposure Field, and Stage 3.1 Closure

Stage 3 refactors the Stage 2 gate into a three-layer control interface and uses
that architecture in the Stage 3.1 open/covered choice protocols.

The current closure branch focuses on Stage 3.1B: the valence/exposure kernel.

---

## Architecture

Stage 3 separates three analytical layers:

1. Instantaneous diagnostics.
2. Exposure-field aggregates.
3. Temporal state.

The sensory interface must not contain ready-made object labels. The agent
receives diagnostic and exposure variables, not pre-classified semions such as
`snake`, `stick`, or `shelter`.

---

## Main modules

```text
stage3/
├── core/
│   ├── agent_stage3.py
│   ├── compatibility.py
│   ├── exposure_field.py
│   ├── gate_inputs.py
│   ├── gate_modes.py
│   ├── gate_stage3.py
│   └── temporal_state.py
├── configs/
│   ├── config_stage3_1a.py
│   └── config_stage3_1b.py
├── envs/
│   ├── maze_builder.py
│   └── open_covered_choice_env.py
├── analysis/
│   ├── run_stage3_1a.py
│   ├── run_stage3_1b.py
│   ├── run_stage3_1b_ablation_suite.py
│   ├── run_stage3_1_closure_package.py
│   ├── analyze_stage3_1a_baseline.py
│   ├── analyze_stage3_1b_ablation_suite.py
│   ├── analyze_stage3_1b_one_shot_publication.py
│   ├── analyze_stage3_1b_matrix_publication.py
│   ├── validate_stage3_1_artifacts.py
│   └── validate_stage3_1b_acceptance.py
└── tests/
    ├── test_action_basin_equivalence.py
    ├── test_backward_compatibility.py
    ├── test_gate_stage3.py
    ├── test_integration.py
    ├── test_no_ready_semions.py
    ├── test_stage3_1b_config.py
    ├── test_stage3_1b_one_shot.py
    └── test_temporal_state.py
```

---

## Stage 3.1A

Stage 3.1A is the open/covered baseline and compatibility layer.

In the closure package it is not treated as a new claim. It is rerun to verify
that Stage 3.1B changes preserve the calibrated Stage 3.1A behavior.

Manual run:

```bash
python -m stage3.analysis.run_stage3_1a \
  --n-seeds 3 \
  --n-trials 20 \
  --output-dir logs/stage3/stage3_1_closure_raw/stage3_1a_manual \
  --ablation full

python -m stage3.analysis.analyze_stage3_1a_baseline \
  --run-dir logs/stage3/stage3_1_closure_raw/stage3_1a_manual \
  --output-dir logs/stage3/stage3_1_closure_raw/stage3_1a_manual_analysis
```

---

## Stage 3.1B

Stage 3.1B closes the valence/exposure kernel. The closure package contains four
layers:

1. `matrix`: 3x3 reward x threat conflict surface.
2. `balanced ablation`: balanced-conflict ablation metrics.
3. `one-shot shock/treat`: event-aligned carryover.
4. `diagnostics`: placebo-window, carrier, and ablation-localization checks.

The one-shot mechanism is not a separate memory module. It is an
amplitude-dependent update regime applied to the same temporal and carrier
variables.

---

## Closure package

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

Analyze-only profile:

```bash
python -m stage3.analysis.run_stage3_1_closure_package \
  --mode analyze-only \
  --skip-stage3-1a \
  --stage3-1b-suite-dir logs/stage3/stage3_1_closure_raw/stage3_1b/<suite_dir> \
  --stage3-1b-matrix-run-dir logs/stage3/stage3_1_closure_raw/stage3_1b_matrix/<grid_dir>
```

---

## Matrix-only workflow

```bash
python -m stage3.analysis.run_stage3_1b \
  --grid 3x3 \
  --n-seeds 3 \
  --n-trials 20 \
  --output-dir logs/stage3/stage3_1_closure_raw/stage3_1b_matrix \
  --ablation full

python -m stage3.analysis.analyze_stage3_1b_matrix_publication \
  --input-dir logs/stage3/stage3_1_closure_raw/stage3_1b_matrix/<grid_dir> \
  --output-dir logs/stage3/stage3_1_closure_raw/stage3_1b_matrix/<grid_dir>/analysis_matrix \
  --ablation full
```

The matrix layer produces heatmaps for `P(open)`, `P(timeout)`, commit latency,
deliberation proxy, junction pause, reorientation, and mode-at-junction
diagnostics.

---

## Tests

```bash
python -m pytest stage3/tests
```

The Stage 3 test suite covers:

- backward compatibility;
- no-ready-semions constraints;
- Gate v3 routing;
- temporal state updates;
- action-basin equivalence;
- Stage 3.1B config;
- one-shot protocol behavior.

---

## Interpretation boundary

Stage 3.1B supports only the valence/exposure kernel claim. It does not claim
absence inference, allocentric spatial cognition, self-model-based visibility
reasoning, or rodent-level VTE equivalence.

