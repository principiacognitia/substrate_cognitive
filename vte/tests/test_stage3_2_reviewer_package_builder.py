from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGURE_NAMES = [
    "Figure_3_2_Model_Relevant_Seed_Level_Effects.png",
    "Figure_3_2_Seed_Level_VTE_Rate_By_Ablation.png",
    "Figure_3_2_Degenerate_Ablation_Diagnostics.png",
    "Figure_3_2_Seed_Level_Ablation_Effect_Sizes.png",
    "Figure_3_2_Seed_Level_VTE_Delta_Effect_Sizes.png",
]

def _write_png(path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(3, 2))
    ax.plot([0, 1], [0, 1])
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)

def _write_stage3_2_source_tree(results_root: Path) -> Path:
    source_dir = results_root / "vte" / "stage3_2_seed_level_stats_analysis"
    source_dir.mkdir(parents=True)

    for name in FIGURE_NAMES:
        _write_png(source_dir / name, name)

    (source_dir / "stage3_2_seed_level_stats_analysis_meta.json").write_text(
        json.dumps(
            {
                "patch": "20E",
                "n_tests": 100,
                "n_model_relevant_tests": 45,
                "n_wrapper_sanity_tests": 48,
                "n_degenerate_ablation_diagnostics": 7,
            }
        ),
        encoding="utf-8",
    )

    (source_dir / "Stage3_2_Response_To_GLM_Stats_Critique.md").write_text(
        "# Stage 3.2 Response to GLM Statistical Critique\n\n"
        "Wrapper sanity checks and model-relevant tests are separated.\n",
        encoding="utf-8",
    )

    (source_dir / "Stage3_2_Seed_Level_Stats_Analysis_Report.md").write_text(
        "# Patch 20E report\n",
        encoding="utf-8",
    )

    (source_dir / "Table_3_2_Seed_Level_Stats_By_Test_Role.csv").write_text(
        "test_role,n_tests,n_fdr_lt_05,median_abs_effect_size\n"
        "model_relevant_test,45,17,0.15\n"
        "wrapper_sanity_check,48,48,15.4\n"
        "degenerate_ablation_diagnostic,7,6,3.38\n",
        encoding="utf-8",
    )

    common_csv = (
        "test_family,test_role,contrast,protocol,condition,ablation,metric,"
        "group_a,group_b,n_seed_pairs,mean_delta,effect_size,p_fdr_bh,direction,status\n"
        "vte_binary_within_seed,model_relevant_test,vte_minus_nonvte,"
        "stage3_steps,R1_T2,full,total_reward,nonvte,vte,50,0.1,0.5,0.01,positive,tested\n"
    )

    for csv_name in [
        "Table_3_2_Model_Relevant_Seed_Level_Tests.csv",
        "Table_3_2_Wrapper_Sanity_Tests.csv",
        "Table_3_2_Degenerate_Ablation_Diagnostics.csv",
    ]:
        (source_dir / csv_name).write_text(common_csv, encoding="utf-8")

    for md_name in [
        "Table_3_2_Model_Relevant_Seed_Level_Tests.md",
        "Table_3_2_Wrapper_Sanity_Tests.md",
        "Table_3_2_Degenerate_Ablation_Diagnostics.md",
    ]:
        (source_dir / md_name).write_text(
            f"# {md_name}\n\n| metric | effect_size |\n|---|---:|\n| total_reward | 0.5 |\n",
            encoding="utf-8",
        )

    return source_dir

def test_stage3_2_reviewer_package_builder_llm5(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "tools" / "reviewer_package" / "build_stage3_reviewer_package.py"

    results_root = tmp_path / "docs" / "results"
    output_dir = tmp_path / "docs" / "reviewer_packages" / "stage3_2_seed_level_stats"

    _write_stage3_2_source_tree(results_root)

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--preset",
            "stage3_2_seed_level_stats",
            "--profile",
            "llm5",
            "--results-root",
            str(results_root),
            "--output-dir",
            str(output_dir),
            "--clean",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Stage 3.2 reviewer package built successfully" in completed.stdout

    expected = {
        "Stage3_2_Reviewer_Report.md",
        "Stage3_2_Key_Tables.md",
        "Figure_Stage3_2_Reviewer_Page_Stats.png",
        "Figure_Stage3_2_Reviewer_Page_Diagnostics.png",
        "reviewer_package_registry.json",
    }
    produced = {p.name for p in output_dir.iterdir() if p.is_file()}
    assert produced == expected

    registry = json.loads((output_dir / "reviewer_package_registry.json").read_text())
    assert registry["package"] == "stage3_2_llm_reviewer_package"
    assert len(registry["files"]) == 4