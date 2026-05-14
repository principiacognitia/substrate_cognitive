import json

import pandas as pd

from vte.analysis.run_stage3_2_seed_level_stats import run_seed_level_stats


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
        "committed_path": "open" if trial % 2 else "covered",
        "vte_binary": vte,
        "reward": reward,
        "raw_idphi": raw_idphi,
        "z_idphi": z_idphi,
        "pause_ticks": pause,
        "reorientation_count": 2 if vte else 0,
    }


def test_seed_level_stats_writes_expected_outputs(tmp_path):
    rows = []
    for seed in [1, 2, 3, 4]:
        rows.extend([
            _row(seed, 1, "full", 0, 0.6, 10, -0.5, 1),
            _row(seed, 2, "full", 1, 0.8, 40, 1.5, 4),
            _row(seed, 3, "novg", 0, 0.5, 8, -0.6, 1),
            _row(seed, 4, "novg", 1, 0.7, 30, 1.0, 3),
        ])
    metrics_csv = tmp_path / "vte_trial_metrics_all.csv"
    output_dir = tmp_path / "patch20b"
    pd.DataFrame(rows).to_csv(metrics_csv, index=False)
    meta = run_seed_level_stats([metrics_csv], output_dir, min_seed_pairs=3)
    assert meta["seed_is_inferential_unit"] is True
    assert meta["n_trial_rows"] == 16
    assert meta["n_seed_aggregate_rows"] == 8
    assert meta["n_statistical_tests"] > 0
    expected = {
        "Table_3_2_Seed_Level_Aggregates.csv",
        "Table_3_2_Seed_Level_By_VTE.csv",
        "Table_3_2_Seed_Level_VTE_Contrasts.csv",
        "Table_3_2_Seed_Level_Ablation_Contrasts.csv",
        "Table_3_2_Seed_Level_Statistical_Tests.csv",
        "Table_3_2_Seed_Level_Effect_Sizes.csv",
        "stage3_2_seed_level_stats_meta.json",
        "Stage3_2_Seed_Level_Stats_Report.md",
    }
    assert expected.issubset({p.name for p in output_dir.iterdir()})
    tests = pd.read_csv(output_dir / "Table_3_2_Seed_Level_Statistical_Tests.csv")
    assert "p_fdr_bh" in tests.columns
    assert {"vte_binary_within_seed", "ablation_vs_reference_within_seed"}.issubset(set(tests["test_family"]))


def test_seed_level_stats_metadata_is_json(tmp_path):
    metrics_csv = tmp_path / "metrics.csv"
    output_dir = tmp_path / "out"
    pd.DataFrame([
        _row(1, 1, "full", 0, 0.0, 1.0, -1.0, 1),
        _row(1, 2, "full", 1, 1.0, 2.0, 1.0, 2),
        _row(2, 1, "full", 0, 0.0, 1.0, -1.0, 1),
        _row(2, 2, "full", 1, 1.0, 2.0, 1.0, 2),
        _row(3, 1, "full", 0, 0.0, 1.0, -1.0, 1),
        _row(3, 2, "full", 1, 1.0, 2.0, 1.0, 2),
    ]).to_csv(metrics_csv, index=False)
    run_seed_level_stats([metrics_csv], output_dir, min_seed_pairs=3)
    meta = json.loads((output_dir / "stage3_2_seed_level_stats_meta.json").read_text(encoding="utf-8"))
    assert meta["patch"] == "20B"
    assert meta["trial_rows_are_measurement_observations"] is True
