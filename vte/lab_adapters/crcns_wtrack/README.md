# CRCNS W-track lab adapter probe

This directory contains Stage 3.2C utilities for inspecting CRCNS Frank-lab
W-track MATLAB v5 files before writing a frozen biological-lab VTE adapter.

## Patch 14: CRCNS W-track lab adapter
status: complete as ingestion / canonical trace / segmentation diagnostic layer
not complete as direct biological VTE validation

## Scope

Current utilities are probes only:

- read MATLAB v5 `.mat` files through `scipy.io.loadmat`;
- inventory available `pos`, `rawpos`, `task`, `spikes`, metadata, and EEG files;
- inspect top-level MATLAB variables for small non-EEG files;
- locate plausible position arrays;
- render a first XY position plot for one animal/day.

They do not compute VTE metrics, infer trials, fit parameters, or import Stage 3
model internals.

## Scope limitation

This adapter does not validate model VTE against pre-labeled biological VTE events.
The CRCNS hc-6 dataset provides W-track alternation behavior, position tracking,
task metadata, spikes, and LFP/EEG recordings, but it does not provide curated
VTE/head-sweep labels or IdPhi annotations.

Patch 14 closes only the biological tracking ingestion layer:

- MATLAB v5 file loading
- file inventory
- position probe
- task/epoch probe
- canonical lab-tracking trace export
- inferred W-track geometry
- choice-zone / route-zone segmentation
- diagnostic segmentation QA

Any comparison to model VTE must be treated as derived behavioral-proxy analysis
unless a dataset with explicit VTE labels is used.

## Commands

Inventory one animal directory:

```powershell
python -m vte.lab_adapters.crcns_wtrack.inventory_crcns_wtrack `
  --animal-dir logs\vte_datasets\hc-6\Fiv `
  --output-dir logs\vte\crcns_probe\hc6_Fiv
```

Probe one day of position data:

```powershell
python -m vte.lab_adapters.crcns_wtrack.extract_position_probe `
  --animal-dir logs\vte_datasets\hc-6\Fiv `
  --day 1 `
  --output-dir logs\vte\crcns_probe\hc6_Fiv_day01
```

If `pos` is unavailable or unreadable, try `rawpos`:

```powershell
python -m vte.lab_adapters.crcns_wtrack.extract_position_probe `
  --animal-dir logs\vte_datasets\hc-6\Fiv `
  --day 1 `
  --source-kind rawpos `
  --output-dir logs\vte\crcns_probe\hc6_Fiv_day01_rawpos
```

## Outputs

Inventory:

- `Table_CRCNS_WTrack_File_Inventory.csv`
- `Table_CRCNS_WTrack_Day_File_Matrix.csv`
- `crcns_wtrack_inventory_meta.json`

Position probe:

- `Table_CRCNS_WTrack_Position_Probe.csv`
- `Figure_CRCNS_WTrack_XY_DayXX_Epochs.png`
- `crcns_wtrack_position_probe_meta.json`

## Methodological note

This layer is a biological trace adapter, not a simulator. The target output for
future patches is a canonical VTE trace with explicit metadata:

- `trace_origin = biological`
- `pose_source = lab_tracking`
- `dataset_id`
- `subject_id`
- `session_id`
- `geometry_id`
- `coordinate_system`

The same downstream VTE wrapper and visualizer should operate on model-derived
and biological tracking traces through this shared contract.

## Report

Commit-rule stability is diagnostic only in Patch 14.
Route labels from segmentation and post-window zone labels are not yet mapped
through a frozen arm/zone equivalence table. Match-rate columns must not be used
as acceptance criteria.