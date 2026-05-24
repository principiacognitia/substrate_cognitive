# Stage 3.2 Deferred Work

This file lists useful work that can be done with Stage 3 materials after Stage 3.2 closure.

These items are not required for Stage 3.2 closure and should not block article handoff.

---

## 1. VTE-like trail visualization

### Goal

Produce static article figures showing representative VTE-like and non-VTE-like trajectories from Stage 3 traces.

### Possible outputs

```text
Figure_3_2_VTE_Like_Trail_Example.png
Figure_3_2_NonVTE_Trail_Example.png
Figure_3_2_VTE_ChoicePoint_Panel.png
```

### Boundary

Visualization must remain read-only.

The visualizer may read:

- trace CSV;
- trial metrics CSV;
- selected-example registry;
- geometry metadata.

The visualizer must not recompute VTE labels from presentation pose and must not import Stage 3 internals.

---

## 2. VTE-like trail animation

### Goal

Create short animation clips for demonstration and presentations.

### Possible outputs

```text
Animation_3_2_VTE_Like_Trail.mp4
Animation_3_2_NonVTE_Trail.mp4
```

### Boundary

Animations are communication artifacts, not statistical evidence.

They should not be part of the primary inferential package unless converted into fixed static figure panels.

---

## 3. Side-by-side VTE comparison

### Goal

Show Stage 3 synthetic trace and biological decision-level comparator in the same figure format.

### Possible outputs

```text
Figure_3_2_Synthetic_vs_Biological_Decision_Level_Comparison.png
```

### Boundary

The comparison must be labeled as decision-level / schema-level.

It must not imply:

- movement-level biological trajectory replay;
- rodent-level VTE equivalence;
- neural mechanism identity.

---

## 4. Biological-data extension

### Goal

Use available biological datasets to test whether decision-level variables can be mapped into the Stage 3.2 schema.

Potential sources include:

- choice-point behavioral records;
- tracking-derived pause/reorientation metrics;
- trial-level outcome records;
- electrophysiological or neural summaries, if available.

### Possible outputs

```text
Table_3_2_Biological_Adapter_Coverage.csv
Table_3_2_Biological_Decision_Level_Comparability.csv
Figure_3_2_Biological_Comparator_Summary.png
```

### Boundary

Biological data should not be used to retune IdPhi, z-scoring, or VTE thresholds per dataset.

---

## 5. Gate dynamics at biological choice points

### Goal

Explore whether Stage 3 Gate dynamics at ambiguous choice points can be compared to neural variables from uploaded or future datasets.

Possible targets:

- choice-point firing-rate changes;
- hippocampal/prefrontal activity around decision points;
- neural markers of pause/reorient regimes;
- trial-level uncertainty or conflict proxies.

### Boundary

This is exploratory.

It should not be folded into the closed Stage 3.2 claim unless a separate predeclared analysis plan is written.

---

## 6. Maze configuration variants

### Goal

Extend Stage 3 from Open/Covered choice to other maze topologies.

Possible approaches:

1. manual `config.py` files for a small number of frozen maze variants;
2. deterministic maze-builder script;
3. W-maze-like configuration;
4. T-maze / Y-maze variants for control comparisons.

### Recommended order

1. manual frozen config for one additional maze;
2. validate trace schema compatibility;
3. only then consider a deterministic maze builder.

### Boundary

Maze building must not become geometry p-hacking.

Any new maze must have:

- fixed config;
- stable geometry registry;
- seed policy;
- frozen task parameters;
- explicit interpretation boundary.

---

## 7. W-maze exploration

### Goal

Test whether the Stage 3 agent can run in a W-maze-like environment.

### Risk

This is likely a new experiment class rather than a small Stage 3.2 extension.

A W-maze adds:

- richer spatial topology;
- multiple decision points;
- sequence memory demands;
- alternation-like behavior;
- harder biological comparison constraints.

### Recommendation

Treat W-maze as a future Stage 3.x or Stage 4-adjacent task, not as a requirement for the current article.

---

## 8. RROW-like synthetic analogue

### Goal

Potentially compare with RROW-style biological tasks.

### Recommendation

Do not prioritize this for the current model.

RROW-like tasks involve too many simultaneous factors for the present configuration:

- alternation;
- reward history;
- spatial route memory;
- procedural automation;
- choice-point deliberation;
- multi-arm task geometry.

A synthetic RROW analogue would require a dedicated task model and should not be presented as a small extension of Stage 3.2.

---

## 9. Article-support figure cleanup

### Goal

Prepare only the minimal visual material needed for the Stage 3 follow-up article.

Recommended minimal figure set:

```text
Figure 1: Stage 3 architecture and closure boundary
Figure 2: Stage 3.1B valence/exposure closure
Figure 3: VTE-style measurement schema
Figure 4: seed-level statistics
Figure 5: biological decision-level comparability boundary
```

This is the only visualization work that should be considered article-critical.

---

## 10. Stage 4 bridge

Stage 4 should not start by modifying Stage 3.2.

A clean Stage 4 bridge should begin from:

- Stage 3.1B valence/exposure kernel;
- Stage 3.2 trace/measurement boundary;
- explicit new mechanism for absence / visibility / self-model reasoning.

Stage 4 should be a new branch and a new claim set.