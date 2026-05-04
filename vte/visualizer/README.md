# VTE Visualizer - Deferred Stage 3.2D Draft

> **Status:** local draft / deferred module  
> **Do not commit or push as part of Stage 3.2C.**  
> This directory is a staging area for visualization scripts that may become Stage 3.2D after biological-lab comparability has been evaluated.

---

## Purpose

`vte/visualizer/` contains draft scripts for rendering Stage 3.2 VTE traces as static figures and toy-rat animations.

The visualizer is intended to support a future **Stage 3.2D: toy-rat visualizer / publication-facing replay layer**.

It is currently **not part of Stage 3.2A/B/C**.

The reason is methodological. Stage 3.2A/B validate the wrapper, trace schema, IdPhi-like metrics, batch analysis, and selected examples. Stage 3.2C evaluates comparability with biological laboratory tracking data. Until Stage 3.2C produces an acceptable positive comparison, a Redish-style “artificial rat” visualization would risk implying biological equivalence before it has been established.

---

## Current staging policy

This directory may contain draft scripts, but it should remain local until Stage 3.2D is explicitly opened.

Recommended status:

```text
vte/visualizer/
  README.md              # this file
  *.py                   # local draft scripts only
```

Do not include this directory in a public commit during Stage 3.2C unless the commit explicitly marks the module as deferred and non-operational.

Suggested Git policy:

```bash
git status
git diff -- vte/visualizer
```

Do not run:

```bash
git add vte/visualizer
git commit
git push
```

until Stage 3.2D is opened.

---

## Stage boundary

### Active now

```text
Stage 3.2A
  VTE wrapper core
  raw trace schema
  trial-level VTE metrics

Stage 3.2B
  Stage 3 adapter
  batch wrapper
  batch analysis
  selected examples table

Stage 3.2C
  biological-lab comparability layer
  external lab-data adapters
  geometry registry
  comparison reports
```

### Deferred

```text
Stage 3.2D
  toy-rat visualizer
  Redish-style dynamic replay
  static and animated selected examples
  side-by-side VTE vs non-VTE visual figures
```

---

## Why the visualizer is deferred

Stage 3 step-log derived traces use synthetic pose reconstruction.

This means:

```text
pose_source = synthetic_from_stage3_steps
```

The current trace can support IdPhi-like analysis and synthetic replay, but it is not biological head-tracking or body-tracking data.

Therefore, the visualizer must not be used to imply:

- direct rodent equivalence;
- biological head-sweep kinematics;
- hippocampal trajectory sweeps;
- neural homology;
- validated behavioral identity with laboratory VTE.

The visualizer may later be used to show:

- frozen Stage 3 traces;
- candidate-path switching;
- choice-point pauses;
- heading changes;
- trial-level VTE metrics;
- selected examples from the wrapper output.

---

## Intended data inputs

The visualizer should operate only on externalized VTE-layer files.

Allowed inputs:

```text
logs/vte/<batch>/traces/*.csv
logs/vte/<batch>/vte_trial_metrics_all.csv
logs/vte/<batch>/Table_3_2_VTE_Selected_Examples.csv
```

Typical batch manifest fields:

```text
combined_metrics_csv
selected_examples_csv
runs[*].vte_trace_csv
runs[*].vte_metrics_csv
```

A selected examples table should contain:

```text
example_type
run_id
seed
trial
committed_path
raw_idphi
z_idphi
pause_ticks
reorientation_count
vte_binary
trace_csv
recommended_output_name
```

The visualizer should not directly read Stage 3 internals.

Disallowed inputs:

```text
stage3.core.*
stage3.envs.*
GateStage3
AgentStage3
internal gate state
model configuration objects
reward/threat configuration objects
precomputed deliberation labels
```

If Stage 3 logs are needed, they must first be translated through the VTE adapter:

```text
Stage 3 all_steps.csv
  -> vte.analysis.translate_stage3_steps_to_vte_trace
  -> VTE trace CSV
  -> vte.visualizer
```

---

## Draft script layout

The draft implementation may use the following structure:

```text
vte/visualizer/
  README.md

  __init__.py

  io.py
    Load trace CSV.
    Load metrics CSV.
    Select one trial by run_id / seed / trial.
    Resolve final committed path.

  layout.py
    Define canonical open/covered maze layout.
    Store presentation-only coordinates.

  pose.py
    Reconstruct presentation pose from frozen VTE trace.
    Add draw_x, draw_y, draw_heading.
    Must not recompute VTE metrics.

  render_static.py
    Render one static trial figure.
    Panels:
      maze replay
      heading-over-time
      trial metrics

  animate_trial.py
    Render one GIF animation.
    Show rat marker, heading arrow, tail trace, choice-point status.

  select_examples.py
    Select top_vte, clean_non_vte, matched_pause_control examples.

  compare_trials.py
    Future Redish-style side-by-side figure:
      VTE example
      non-VTE example
      zIdPhi distribution / metrics panel

  cli_render_trial.py
    CLI entry point for one trial.

  cli_batch_render.py
    CLI entry point for Table_3_2_VTE_Selected_Examples.csv.
```

---

## Measurement vs presentation

The visualizer must preserve a strict distinction between measurement and presentation.

### Measurement trace

The wrapper metrics are computed from frozen VTE trace columns:

```text
run_id
seed
trial
tick
x
y
heading
choice_point_id
at_choice_point
action
committed_path
reward
done
```

### Presentation pose

The visualizer may create additional drawing-only columns:

```text
draw_x
draw_y
draw_heading
final_committed_path
presentation_pose_source
```

These columns are only for display.

They must not be used to recompute:

```text
raw_idphi
log_idphi
z_idphi
pause_ticks
reorientation_count
vte_binary
```

Required caption language for synthetic Stage 3 traces:

```text
Synthetic presentation trajectory reconstructed from Stage 3 VTE trace.
IdPhi metrics are computed from the frozen trace, not from the smoothed animation.
This is not biological tracking data.
```

---

## Intended visual outputs

### Static single-trial figure

Target output:

```text
docs/results/vte/visualizer/static/<example_name>.png
```

Panels:

```text
left:
  canonical maze
  trajectory replay
  heading arrow
  choice point marker

middle:
  heading over tick
  choice-point window

right:
  run_id
  seed
  trial
  committed_path
  raw_idphi
  z_idphi
  pause_ticks
  reorientation_count
  vte_binary
  pose_source
```

### Animated single-trial replay

Target output:

```text
docs/results/vte/visualizer/animations/<example_name>.gif
```

Animation elements:

```text
rat body marker
heading arrow
short fading tail
choice-point highlight
current tick
current action
final committed path
trial-level VTE metrics
```

### Future Redish-style comparison

Deferred Stage 3.2D target:

```text
docs/results/vte/visualizer/composite/Figure_3_2D_Redish_Style_Replay.gif
```

Suggested layout:

```text
left:
  VTE-like trial replay

middle:
  non-VTE-like trial replay

right top:
  zIdPhi distribution with selected trial markers

right bottom:
  metric table
```

This should only be promoted to publication-facing output after Stage 3.2C biological comparability is evaluated.

---

## Draft commands

These commands are for local testing only.

### Single static trial

```bash
python -m vte.visualizer.cli_render_trial ^
  --trace-csv logs\vte\stage3_2_batch_smoke\traces\balanced_full_balanced_conflict_full_all_steps_vte_trace.csv ^
  --metrics-csv logs\vte\stage3_2_batch_smoke\vte_trial_metrics_all.csv ^
  --run-id balanced_full_balanced_conflict_full_all_steps ^
  --seed 42 ^
  --trial 1 ^
  --output docs\results\vte\visualizer_smoke\ToyRat_seed42_trial1.png
```

### Single animated trial

```bash
python -m vte.visualizer.cli_render_trial ^
  --trace-csv logs\vte\stage3_2_batch_smoke\traces\balanced_full_balanced_conflict_full_all_steps_vte_trace.csv ^
  --metrics-csv logs\vte\stage3_2_batch_smoke\vte_trial_metrics_all.csv ^
  --run-id balanced_full_balanced_conflict_full_all_steps ^
  --seed 42 ^
  --trial 1 ^
  --animate ^
  --fps 2 ^
  --output docs\results\vte\visualizer_smoke\ToyRat_seed42_trial1.gif
```

### Batch render selected examples

```bash
python -m vte.visualizer.cli_batch_render ^
  --batch-root logs\vte\stage3_2_batch_smoke ^
  --output-dir docs\results\vte\visualizer_smoke ^
  --limit 3
```

### Batch render selected GIFs

```bash
python -m vte.visualizer.cli_batch_render ^
  --batch-root logs\vte\stage3_2_batch_smoke ^
  --output-dir docs\results\vte\visualizer_smoke_gif ^
  --animate ^
  --fps 2 ^
  --limit 3
```

---

## Smoke criteria

The local visualizer draft is usable if:

```text
1. It renders one static PNG from a VTE trace CSV.
2. It renders one GIF from the same trace CSV.
3. It can read Table_3_2_VTE_Selected_Examples.csv.
4. It can render top_vte, clean_non_vte, and matched_pause_control examples.
5. It never imports Stage 3 internals.
6. It never recomputes VTE metrics from presentation pose.
7. It marks synthetic traces as synthetic.
8. It keeps visual output separate from Stage 3.2C biological comparison reports.
```

---

## Activation criteria for Stage 3.2D

The visualizer should be promoted from local draft to active module only after Stage 3.2C has produced a clear comparability decision.

Minimum activation criteria:

```text
1. Biological-lab adapter exists.
2. External tracking data can be translated into the same VTE trace schema.
3. Measurement core remains frozen.
4. zIdPhi / pause / reorientation distributions can be compared without dataset-specific metric changes.
5. Stage 3 synthetic traces and biological traces can be displayed under one visual convention.
6. The report clearly distinguishes synthetic replay from biological tracking.
```

If Stage 3.2C is negative or inconclusive, this visualizer should remain a diagnostic/internal tool only.

---

## Non-goals

This module does not provide:

```text
biological pose tracking
neural decoding
hippocampal sweep visualization
proof of deliberation
proof of rodent equivalence
maze editing
metric computation
Stage 3 model execution
```

It visualizes already externalized traces.

---

## Recommended future rename

If the module is activated after Stage 3.2C, consider renaming the package from:

```text
vte/visualizer/
```

to:

```text
vte/visualization/
```

or keep `visualizer/` if the project convention favors executable tools over analysis modules.

No rename should be done while the module is deferred.

---

## References

Redish, A. D. (2016). Vicarious trial and error. *Nature Reviews Neuroscience, 17*(3), 147-159. https://doi.org/10.1038/nrn.2015.30