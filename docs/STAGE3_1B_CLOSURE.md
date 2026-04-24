# Stage 3.1B Closure

## Status

Stage 3.1B is the closure point for the valence/exposure layer of the
S-O-R + Gate + Rheology architecture.

This document defines the closure boundary for Stage 3.1B. It does not
introduce new mechanisms. It freezes the current Stage 3.1B kernel for
smoke testing, paper-grade clean runs, figure/table generation, and
documentation.

After this closure, Stage 3.2 may add a spatial/VTE wrapper, but it must
not alter the Stage 3.1B cognitive kernel without explicitly reopening
the closure.

---

## 1. Closure purpose

Stage 3.1B closes three experimental claims:

1. tradeoff-sensitive path choice under reward/threat conflict;
2. persistent one-shot deformation through continuous importance traces;
3. ablation-localized positive/negative carrier asymmetry.

The purpose of closure is to separate:

- the validated Stage 3.1B kernel;
- paper-grade reproducibility artifacts;
- future Stage 3.2 wrapper work.

This boundary is necessary because Stage 3.2 will add spatial/VTE
comparison machinery. Without a closure tag, changes to the wrapper could
be confused with changes to the kernel.

---

## 2. Frozen kernel boundary

The Stage 3.1B kernel includes:

- Stage 3 open/covered choice environment;
- conflict protocol;
- positive and negative one-shot protocols;
- temporal traces:
  - `h_risk`;
  - `h_opp`;
  - `h_time`;
- continuous importance traces:
  - `q_neg`;
  - `q_pos`;
- gate arbitration;
- gate rheology `V_G`;
- pattern/action rheology `V_p`;
- Stage 3.1B configuration and analysis scripts.

The closure kernel must preserve the environment-agent boundary:

- the environment provides field potentials, path contingencies, exposure,
  reward, and event conditions;
- the agent performs inference, temporal accumulation, arbitration,
  rheological updating, and action selection;
- the environment must not evaluate importance on behalf of the agent.

---

## 3. What Stage 3.1B establishes

Stage 3.1B establishes that the agent can express non-trivial behavior in
the valence/exposure layer without adding symbolic event memory or
scheduler-based persistence.

The accepted architectural claims are:

### 3.1 Tradeoff-sensitive path choice

The agent can shift between open and covered paths as reward premium and
threat/exposure change.

This is not a claim that the agent has a full spatial model. It is a claim
that path choice responds to reward-threat structure in the current
valence/exposure field.

### 3.2 Persistent one-shot deformation

A single strong event can deform subsequent behavior through continuous
importance dynamics.

This persistence must be carried by endogenous traces and rheological
variables, not by explicit event counters, forced post-shock windows, or
symbolic shock labels.

### 3.3 Positive/negative carrier asymmetry

Positive and negative one-shot events need not produce symmetric effects.

Stage 3.1B allows asymmetry between `q_pos` and `q_neg`, and between their
effects on `h_opp`, `h_risk`, `V_G`, and `V_p`.

### 3.4 Ablation-localized carriers

The behavioral effect of one-shot events must be interpretable through
ablation comparisons.

At minimum, closure should distinguish:

- full model;
- no gate rheology (`NoVG`);
- no pattern/action rheology (`NoVp`);
- optional combined or targeted ablations if already implemented.

---

## 4. What Stage 3.1B does not establish

Stage 3.1B does not claim:

1. absence inference;
2. self-model-based "if it were there, I would see it" reasoning;
3. object permanence;
4. allocentric spatial cognition;
5. rodent-level VTE equivalence;
6. explicit episodic memory;
7. symbolic event representation;
8. general-purpose planning.

Absence inference and self-model-dependent counterfactual visibility are
deferred to Stage 4.

Spatial/VTE comparison with rodent choice-point behavior is deferred to
Stage 3.2.

---

## 5. Closure acceptance criteria

Stage 3.1B closure requires the following checks.

### 5.1 Smoke tests

Smoke tests must confirm that the current kernel runs without regression
and that core tests pass.

Required minimum:

```bash
python -m pytest stage3/tests
````

If additional repository-level tests are stable, run:

```
python -m pytest
```

### 5.2 Paper-grade clean runs

Paper-grade runs must be generated from a clean working tree on the  
closure branch.

Before running:

```
git status
```

Expected:

```
On branch stage3_1b_closure
nothing to commit, working tree clean
```

Runs should be generated with fixed seeds and recorded configuration.

### 5.3 Artifact separation

Generated artifacts must be separated from source code.

Allowed closure artifacts include:

*   summary tables;
*   final figure files;
*   aggregate statistics;
*   appendix-ready CSV/JSON summaries;
*   run metadata.

Raw debug traces, temporary notebooks, exploratory dumps, and ad hoc  
diagnostic scripts should not be committed unless explicitly moved into a  
documented debug or appendix location.

### 5.4 Documentation update

Closure requires updates to:

*   this document;
*   Stage 3 README if needed;
*   root README / roadmap if needed.

The roadmap must state that Stage 3.1B is frozen before Stage 3.2 begins.

* * *

6\. Required closure artifacts
------------------------------

The final closure commit should contain, at minimum:

1.  `docs/STAGE3_1B_CLOSURE.md`;
2.  updated README or roadmap text;
3.  clean run summaries;
4.  figure/table outputs selected for paper or appendix use;
5.  reproducibility notes:
    *   branch;
    *   commit hash;
    *   seed range;
    *   config file;
    *   command used to generate the run.

Suggested artifact structure:

```
docs/
  STAGE3_1B_CLOSURE.md

logs/
  figures/
    stage3/
      stage3_1b_closure/
        figures/
        tables/
        stats/
```

If the repository policy is to keep heavy artifacts out of Git, then this  
document should instead record where the artifacts are stored and which  
commit generated them.

* * *

7\. Version boundary
--------------------

The closure branch is:

```
stage3_1b_closure
```

The closure tag should be:

```
v0.3.1b-closure
```

The tag must be created only after:

1.  tests pass;
2.  paper-grade runs are complete;
3.  closure artifacts are generated;
4.  README/roadmap is updated;
5.  the working tree is clean.

Suggested tag command:

```
git tag -a v0.3.1b-closure -m "Stage 3.1B closure: valence/exposure kernel frozen"
git push origin v0.3.1b-closure
```

* * *

8\. Boundary to Stage 3.2
-------------------------

Stage 3.2 begins only after `v0.3.1b-closure`.

Stage 3.2 may add:

*   spatial/VTE wrapper;
*   rodent choice-point comparison layer;
*   additional trajectory-level diagnostics;
*   wrapper-level visualizations.

Stage 3.2 must not silently change:

*   Stage 3.1B gate logic;
*   temporal trace dynamics;
*   `q_neg` / `q_pos` semantics;
*   one-shot persistence mechanism;
*   `V_G` / `V_p` carrier interpretation.

Any such change reopens the kernel and must be documented as post-closure  
kernel revision.

* * *

9\. Roadmap text
----------------

Recommended roadmap wording:

> Stage 3.1B closes the valence/exposure layer of the architecture. It  
> demonstrates tradeoff-sensitive path choice, persistent one-shot  
> deformation, and ablation-localized positive/negative carrier asymmetry.  
> Stage 3.2 will not alter the cognitive kernel; it will add a spatial/VTE  
> wrapper for comparison with rodent choice-point behavior. Absence  
> inference and self-model-based "if it were there, I would see it"  
> reasoning are deferred to Stage 4.

* * *

10\. Closure statement
----------------------

Stage 3.1B is closed when the repository contains a reproducible,  
documented, and tagged version of the valence/exposure kernel.

The closure claim is deliberately narrow:

> Stage 3.1B demonstrates that valence/exposure dynamics, continuous  
> importance traces, and gate/pattern rheology can produce tradeoff-sensitive  
> behavior and persistent one-shot deformation without symbolic event memory  
> or scheduler-based persistence.

No stronger claim is made at this stage.

````