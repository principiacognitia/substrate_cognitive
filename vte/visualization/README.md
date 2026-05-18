# VTE Visualization Layer — Stage 3.2D, Patch 21

## Status
✅ Implemented as read-only measurement replay layer.  
Not part of Stage 3.2C biological comparability evaluation.

## Architectural boundary
This module:
- Reads only canonical VTE trace CSV and trial metrics CSV
- Never imports `stage3.core`, `stage3.envs`, or agent internals
- Never recomputes VTE metrics from presentation pose
- Always marks `pose_source` explicitly in output metadata
- Keeps visual output separate from Stage 3.2C comparison reports

## Input schema
Required columns in trace CSV:
```
run_id,seed,trial,tick,x,y,heading,choice_point_id,at_choice_point,
action,committed_path,reward,done,condition,ablation,protocol,pose_source
```

Required columns in metrics CSV:
```
run_id,seed,trial,raw_idphi,log_idphi,z_idphi,vte_binary,
pause_ticks,reorientation_count,commit_latency,committed_path
```

## Usage
### Single static trial
```bash
python -m vte.visualization.cli_render_trial `
  --trace-csv logs/vte/raw/balanced_conflict_full_vte_trace.csv `
  --metrics-csv logs/vte/analysis/balanced_full/vte_trial_metrics_all.csv `
  --seed 42 --trial 1 `
  --regime explore `
  --output docs/results/vte/visualization/regime_explore/seed42_trial1.png
```

### Single animation
```bash
python -m vte.visualization.cli_render_trial `
  --trace-csv logs/vte/raw/balanced_conflict_full_vte_trace.csv `
  --metrics-csv logs/vte/analysis/balanced_full/vte_trial_metrics_all.csv `
  --seed 42 --trial 1 `
  --animate --fps 4 `
  --output docs/results/vte/visualization/animations/seed42_trial1_vte.gif
```

### Batch rendering
```bash
python -m vte.visualization.cli_batch_render `
  --manifest docs/results/vte/visualization/selected_examples.json `
  --trace-root logs/vte/raw/ `
  --metrics-root logs/vte/analysis/ `
  --output-root docs/results/vte/visualization/ `
  --config vte/visualization/config_regime_selection.yaml
```

## Methodological statement
Visualizations in this layer display **normalized trajectory samples**, not biological animals or simulated agents. Model-derived traces and biological tracking traces are translated into the same canonical VTE trace schema and rendered by the same adapter. The adapter is origin-blind except for explicit source labels (`trace_origin`, `pose_source`).

A side-by-side replay is not a claim of biological identity. It is a visual inspection tool over a shared measurement schema. Claims of comparability must come from frozen metrics and predeclared comparison reports, not from visual similarity.

## Output artifacts
```
docs/results/vte/visualization/
├── regime_exploit/          # Low-deliberation examples
├── regime_explore/          # High-deliberation examples
├── animations/              # GIF/MP4 replay files
├── selected_examples.csv    # Reproducible selection metadata
├── comparison_report.md     # Side-by-side metrics summary
└── *.json                   # Per-figure metadata (pose_source, geometry_id, etc.)
```