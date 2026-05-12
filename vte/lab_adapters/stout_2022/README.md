# Stout et al. 2022 / Griffin Lab VTE dataset adapter notes

## Status

This dataset was inspected as a possible biological VTE source, but it is not promoted to a direct Stage 3.2 biological-to-synthetic movement comparator.

The dataset is useful as a diagnostic trial-level VTE endpoint, but not as a direct trace-level behavioral comparator for the current toy-rat artificial environment.

## Source

Local source folder:

`logs\vte_datasets\GriffinLabCode\Manuscript Code\Published\Stout et al 2022 Re Inactivation during VTE`

Main files inspected:

- `data_oopsTrialsVTE_step1.mat`
- `data_oopsTrialsVTE_step2.mat`
- `data_vte_step2.mat`
- `data_behavior.mat`
- `data_remove.mat`
- `SCRIPT_VTE_final.m`

## Relevant variables

The useful trial-level variables are:

- `zIdPhi`
- `IdPhi`
- `oopsTrials`
- `accuracy`
- `turnDirection`
- `timeSpent_CP`
- `timeSpent`
- `remTrials`

These variables are organized by rat, condition, and session path, for example:

`subject.condition.session`

Typical conditions include:

- `Baseline`
- `Saline`
- `Muscimol`

## Patch history

### Patch 19A

Purpose: probe file inventory, root MATLAB variables, script references, and candidate VTE/trajectory fields.

Main result:

- Root variables and script references were readable.
- `SCRIPT_VTE_final.m` confirms that VTE-related analysis combines `oopsTrials` and `zIdPhi`.
- `tsPosOG`, `xPosOG`, and `yPosOG` exist but were not sufficient as complete trial-level trajectory sources under the current extraction pass.

### Patch 19B

Purpose: extract a trial-level candidate endpoint.

Main result:

- Endpoint rows: 1216
- Usable endpoint rows: 1152
- Trial-level VTE metrics are usable.
- Full movement trajectory coverage is insufficient for trace-level adapter use.

Observed diagnostic pattern:

- VTE-positive rows have higher `IdPhi`.
- VTE-positive rows have higher dwell/time-at-choice-point proxy.

This supports use as a VTE-marker diagnostic dataset, not as a direct behavioral trace comparator.

## Comparability policy

This dataset should not be mixed with the Redish LRA healthy-control baseline as a direct biological comparator.

Reasons:

1. The task design is different.
2. The dataset is organized around neurobiological perturbation conditions.
3. VTE is a secondary behavioral marker in a neurobiological study.
4. Full movement traces are not reliably available for all trials under the current extraction.
5. The action/task namespace does not match the current synthetic toy-rat environment.

## Ratdle note

`SCRIPT_VTE_final.m` contains an author-side comment indicating that `Ratdle` was excluded. Any future clean endpoint should either exclude `Ratdle` or preserve an explicit `author_exclusion_note`.

## Recommended use

Use this dataset only for future diagnostic or reverse-fitting work, for example:

- Build an analogous DNMP/Re-inactivation task in the artificial environment.
- Fit toy-rat parameters to reproduce trial-level VTE/IdPhi/dwell patterns.
- Compare perturbation-like regimes only after task geometry and reward schedule are explicitly modeled.

## Current decision

No further extraction patch is required for Stage 3.2.

The dataset is documented and parked as:

`diagnostic_trial_level_vte_endpoint_only`