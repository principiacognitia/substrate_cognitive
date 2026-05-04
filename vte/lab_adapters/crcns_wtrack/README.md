# CRCNS W-track lab adapter probe

This directory contains Stage 3.2C utilities for inspecting CRCNS Frank-lab
W-track MATLAB v5 files before writing a frozen biological-lab VTE adapter.

## Scope

Current utilities are probes only:

- read MATLAB v5 `.mat` files through `scipy.io.loadmat`;
- inventory available `pos`, `rawpos`, `task`, `spikes`, metadata, and EEG files;
- inspect top-level MATLAB variables for small non-EEG files;
- locate plausible position arrays;
- render a first XY position plot for one animal/day.

They do not compute VTE metrics, infer trials, fit parameters, or import Stage 3
model internals.

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