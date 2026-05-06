# DANDI 000115 behavioral NWB adapter probe

Patch 15A adds a diagnostic probe for the DANDI 000115 NWB behavioral layer.

Dataset role:

- biological flexible spatial task dataset;
- proxy-level biological comparability candidate;
- not an explicit VTE/head-scanning label benchmark.

The probe inspects a local NWB file and extracts only behavior-relevant layers:

- `intervals/epochs`
- `processing/behavior/position`
- `processing/behavior/behavioral_events`
- `processing/associated_files` StateScript content stored in attributes

It does not read large ecephys arrays.

## Outputs

The probe writes:

- `Table_DANDI000115_Position_Series_Probe.csv`
- `Table_DANDI000115_Behavioral_Events_Probe.csv`
- `Table_DANDI000115_StateScript_Attr_Summary.csv`
- `Table_DANDI000115_StateScript_Event_Probe.csv`
- `Table_DANDI000115_Epochs_Probe.csv`
- `Table_DANDI000115_Time_Alignment_Candidates.csv`
- `dandi000115_behavior_probe_meta.json`
- `DANDI000115_Behavior_Probe_Report.md`
- `statescript_text/*.txt`

## Interpretation

A high StateScript-to-position overlap supports future canonical trace export.

The adapter does not assume explicit biological VTE labels. DANDI 000115 is treated as an event-aligned behavioral-proxy dataset suitable for dwell, reorientation, IdPhi-like trajectory metrics, route/well choice, and reward-linked episode structure.

## Example

```powershell
python -m vte.lab_adapters.dandi_000115.nwb_behavior_probe `
  --nwb logs\vte_datasets\dandi_000115\nwb_probe\sub-despereaux_ses-despereaux-07_behavior+ecephys.nwb `
  --output-dir logs\vte\dandi_000115\despereaux_07_behavior_probe
```
