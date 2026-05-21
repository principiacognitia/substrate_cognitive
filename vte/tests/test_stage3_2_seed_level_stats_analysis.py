import pandas as pd

from vte.analysis.run_stage3_2_seed_level_stats import run_seed_level_stats
from vte.analysis.analyze_stage3_2_seed_level_stats import analyze_seed_level_stats


def _row(seed, trial, ablation, vte, reward, raw_idphi, z_idphi, pause):
    return {
        "source": "synthetic",
        "dataset_id": "stage3_2_test",
        "task_family": "balanced_fork",
        "run_id": f"balanced_{ablation}",
        "protocol": "balanced",
        "condition": "R1_T2",
        "ablation": ablation,
        "seed": seed,
        "trial": trial,
        "vte_binary": vte,
        "reward": reward,
        "raw_idphi": raw_idphi,
        "z_idphi": z_idphi,
        "pause_ticks": pause,
        "reorientation_count": 2 if vte else 0,
    }


def test_seed_level_stats_analyzer_writes_tables_and_figures(tmp_path):
    rows = []
    for seed in [1, 2, 3, 4]:
        rows.extend([
            _row(seed, 1, "full", 0, 0.6, 10, -0.5, 1),
            _row(seed, 2, "full", 1, 0.8, 40, 1.5, 4),
            _row(seed, 3, "novg", 0, 0.5, 8, -0.6, 1),
            _row(seed, 4, "novg", 1, 0.7, 30, 1.0, 3),
        ])
    metrics_csv = tmp_path / "metrics.csv"
    stats_dir = tmp_path / "patch20b"
    analysis_dir = tmp_path / "patch20e"
    pd.DataFrame(rows).to_csv(metrics_csv, index=False)
    run_seed_level_stats([metrics_csv], stats_dir, min_seed_pairs=3)
    meta = analyze_seed_level_stats(stats_dir, analysis_dir, top_n=10)
    assert meta["patch"] == "20E"
    assert meta["n_tests"] > 0
    expected = {
        "Table_3_2_Seed_Level_Stats_Top_Findings.csv",
        "Table_3_2_Seed_Level_Stats_By_Test_Family.csv",
        "Table_3_2_Seed_Level_Stats_By_Metric_Direction.csv",
        "stage3_2_seed_level_stats_analysis_meta.json",
        "Stage3_2_Seed_Level_Stats_Analysis_Report.md",
        "Table_3_2_Model_Relevant_Seed_Level_Tests.md",
        "Table_3_2_Wrapper_Sanity_Tests.md",
        "Table_3_2_Degenerate_Ablation_Diagnostics.md",
    }
    assert expected.issubset({p.name for p in analysis_dir.iterdir()})
    assert any(p.name.startswith("Figure_3_2_Seed_Level_") for p in analysis_dir.iterdir())
