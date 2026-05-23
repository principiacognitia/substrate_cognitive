# Reviewer package builders

This directory contains post-analysis packaging tools.

The scripts do not rerun experiments and do not import project modules. They read committed `docs/results` artifacts and build compact reviewer-facing packages.

## Stage 3.2 seed-level statistics package

Default 5-file LLM package:

```powershell
python tools\reviewer_package\build_stage3_reviewer_package.py `
  --preset stage3_2_seed_level_stats `
  --profile llm5 `
  --results-root docs\results `
  --output-dir docs\reviewer_packages\stage3_2_seed_level_stats `
  --clean
```

Output:

```text
Stage3_2_Reviewer_Report.md
Stage3_2_Key_Tables.md
Figure_Stage3_2_Reviewer_Page_Stats.png
Figure_Stage3_2_Reviewer_Page_Diagnostics.png
reviewer_package_registry.json
```

Ultra-compact 3-file package:

```powershell
python tools\reviewer_package\build_stage3_reviewer_package.py `
  --preset stage3_2_seed_level_stats `
  --profile llm3 `
  --results-root docs\results `
  --output-dir docs\reviewer_packages\stage3_2_seed_level_stats_llm3 `
  --clean
```

Output:

```text
Stage3_2_Reviewer_Report.md
Stage3_2_Key_Tables.md
Figure_Stage3_2_Reviewer_Page_Stats.png
```

## Boundary

Reviewer packages are transport artifacts. They do not introduce new statistical tests, figures, or model claims.