from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


STAGE31_FIGURES = [
    "stage3_1a/figures/Figure_3_1A_Block_Covered_Rate.png",
    "stage3_1b_closure/figures/Figure_3_1B_Matrix_P_Open_Heatmap.png",
    "stage3_1b_closure/figures/Figure_3_1B_Shock_Target_Choice_SEM_Zoom.png",
    "stage3_1b_closure/figures/Figure_3_1B_Treat_Target_Choice_SEM_Zoom.png",
    "stage3_1b_closure/figures/Figure_3_1B_Matrix_P_Timeout_Heatmap.png",
    "stage3_1b_closure/figures/Figure_3_1B_Matrix_Commit_Latency_Heatmap.png",
    "stage3_1b_closure/figures/Figure_3_1B_Shock_QNeg_SEM_Zoom.png",
    "stage3_1b_closure/figures/Figure_3_1B_Shock_HRisk_SEM_Zoom.png",
    "stage3_1b_closure/figures/Figure_3_1B_Treat_QPos_SEM_Zoom.png",
    "stage3_1b_closure/figures/Figure_3_1B_OneShot_Effect_By_Ablation.png",
]

STAGE32_FIGURES = [
    "vte/stage3_2_seed_level_stats_analysis/Figure_3_2_Model_Relevant_Seed_Level_Effects.png",
    "vte/stage3_2_seed_level_stats_analysis/Figure_3_2_Seed_Level_VTE_Rate_By_Ablation.png",
    "vte/stage3_2_seed_level_stats_analysis/Figure_3_2_Degenerate_Ablation_Diagnostics.png",
    "vte/stage3_2_seed_level_stats_analysis/Figure_3_2_Seed_Level_Ablation_Effect_Sizes.png",
    "vte/stage3_2_seed_level_stats_analysis/Figure_3_2_Seed_Level_VTE_Delta_Effect_Sizes.png",
]

STAGE31_TABLES = [
    "stage3_1a/tables/Table_3_1A_Seed_Summary.csv",
    "stage3_1a/tables/Table_3_1A_Block_Dynamics.csv",
    "stage3_1b_closure/tables/Table_3_1B_matrix_cell_summary.csv",
    "stage3_1b_closure/tables/Table_3_1B_ablation_summary_wide.csv",
    "stage3_1b_closure/tables/Table_3_1B_one_shot_effect_stats.csv",
    "stage3_1b_closure/tables/Table_3_1B_one_shot_carrier_effect_stats.csv",
    "stage3_1b_closure/tables/Table_3_1B_one_shot_ablation_localization.csv",
    "stage3_1b_closure/tables/Table_3_1B_one_shot_acceptance_summary.csv",
]


def _write_png(path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(3, 2))
    ax.plot([0, 1], [0, 1])
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def _write_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "metric,group,n,effect_size,p_fdr_bh,status\n"
        "target_choice,full,50,0.42,0.01,tested\n",
        encoding="utf-8",
    )


def _write_fixture_results(results_root: Path) -> None:
    for rel in STAGE31_FIGURES + STAGE32_FIGURES:
        _write_png(results_root / rel, Path(rel).stem)

    for rel in STAGE31_TABLES:
        _write_csv(results_root / rel)

    (results_root / "stage3_1_closure_manifest.json").write_text(
        json.dumps({"mode": "test", "run_id": "fixture"}),
        encoding="utf-8",
    )

    stage32 = results_root / "vte" / "stage3_2_seed_level_stats_analysis"
    stage32.mkdir(parents=True, exist_ok=True)

    (stage32 / "stage3_2_seed_level_stats_analysis_meta.json").write_text(
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

    (stage32 / "Stage3_2_Response_To_GLM_Stats_Critique.md").write_text(
        "# Stage 3.2 Response to GLM Statistical Critique\n\n"
        "Wrapper sanity checks and model-relevant tests are separated.\n",
        encoding="utf-8",
    )

    (stage32 / "Table_3_2_Seed_Level_Stats_By_Test_Role.csv").write_text(
        "test_role,n_tests,n_fdr_lt_05,median_abs_effect_size\n"
        "model_relevant_test,45,17,0.15\n"
        "wrapper_sanity_check,48,48,15.4\n",
        encoding="utf-8",
    )

    for name in [
        "Table_3_2_Model_Relevant_Seed_Level_Tests.md",
        "Table_3_2_Degenerate_Ablation_Diagnostics.md",
        "Table_3_2_Wrapper_Sanity_Tests.md",
    ]:
        (stage32 / name).write_text(
            f"# {name}\n\n| metric | effect_size |\n|---|---:|\n| total_reward | 0.5 |\n",
            encoding="utf-8",
        )


def test_stage3_reviewer_package_builder_stage31_32_llm5(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    results_root = tmp_path / "docs" / "results"
    output_dir = tmp_path / "docs" / "reviewer_packages" / "stage3_1_3_2"

    _write_fixture_results(results_root)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "stage3.analysis.build_stage3_reviewer_package",
            "--preset",
            "stage3_1_3_2",
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

    assert "Stage 3 reviewer package built successfully" in completed.stdout

    expected = {
        "Stage3_Reviewer_Report.md",
        "Stage3_Key_Tables.md",
        "Figure_Stage3_Reviewer_Page_Main.png",
        "Figure_Stage3_Reviewer_Page_Diagnostics.png",
        "reviewer_package_registry.json",
    }
    produced = {p.name for p in output_dir.iterdir() if p.is_file()}
    assert produced == expected

    registry = json.loads((output_dir / "reviewer_package_registry.json").read_text())
    assert registry["preset"] == "stage3_1_3_2"
    assert registry["profile"] == "llm5"


def test_stage3_reviewer_package_builder_visualization_llm3(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    results_root = tmp_path / "docs" / "results"
    output_dir = tmp_path / "docs" / "reviewer_packages" / "stage3_visualization"

    vis_dir = results_root / "vte" / "visualization"
    for i in range(3):
        _write_png(vis_dir / f"visualization_example_{i}.png", f"example {i}")

    (vis_dir / "selected_examples.csv").write_text(
        "example_id,run_id,trial,metric\n"
        "ex1,run_a,1,z_idphi\n",
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "stage3.analysis.build_stage3_reviewer_package",
            "--preset",
            "stage3_visualization",
            "--profile",
            "llm3",
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

    assert "Stage 3 reviewer package built successfully" in completed.stdout

    expected = {
        "Stage3_Reviewer_Report.md",
        "Stage3_Key_Tables.md",
        "Figure_Stage3_Reviewer_Page_Main.png",
    }
    produced = {p.name for p in output_dir.iterdir() if p.is_file()}
    assert produced == expected