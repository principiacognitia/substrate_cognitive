# Patch 17A - Redish 2024 LRA probe and candidate extraction

Place files under the repository root.

Run probe:

```powershell
python -m vte.lab_adapters.redish_lra_2024.probe_lra_dataset `
  --root "E:\CRS-1\SUBSTRATE_COGNITIVE\LOGS\VTE_DATASETS\REDISH 2024" `
  --output-dir "logs\vte\redish_lra_2024\patch17a_probe"
```

Run candidate extraction:

```powershell
python -m vte.lab_adapters.redish_lra_2024.extract_lra_candidates `
  --root "E:\CRS-1\SUBSTRATE_COGNITIVE\LOGS\VTE_DATASETS\REDISH 2024" `
  --output-dir "logs\vte\redish_lra_2024\patch17a_candidates"
```

Control commands:

```powershell
Import-Csv logs\vte\redish_lra_2024\patch17a_probe\Table_Redish_LRA_Behavior_Endpoint_Precheck.csv |
  Format-Table -AutoSize

Import-Csv logs\vte\redish_lra_2024\patch17a_candidates\Table_Redish_LRA_Field_Alignment_Summary.csv |
  Sort-Object cohort,source_table,field_class,field_path |
  Select-Object -First 120 |
  Format-Table -AutoSize

Import-Csv logs\vte\redish_lra_2024\patch17a_candidates\Table_Redish_LRA_Endpoint_Draft.csv |
  Select-Object -First 80 |
  Format-Table -AutoSize
```

Patch 17A is conservative. It does not assume the final MATLAB schema. It finds candidate IdPhi/VTE, lap/trial, left/right choice, reward, contingency/rule, changepoint, stereotypy, and DREADD/treatment fields for both LRA and mPFC-DREADDs cohorts. Patch 17B should make the canonical left/right endpoint after the candidate tables show the real field paths.
