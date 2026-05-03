import json

import pandas as pd
import pytest

from vte.analysis.analyze_stage3_2_vte import analyze_vte_metrics


def _metric_row(
    *,
    seed=42,
    trial=1,
    condition="R1_T2",
    committed_path="open",
    raw_idphi=1.0,
    z_idphi=0.0,
    vte_binary=0,
):
    return {
        "run_id": "analysis_test",
        "protocol": "stage3_steps",
        "condition": condition,
        "ablation": "full",
        "pose_source": "synthetic_from_stage3_steps",
        "seed": seed,
        "trial": trial,
        "committed_path": committed_path,
        "raw_idphi": raw_idphi,
        "log_idphi": 0.69,
        "z_idphi": z_idphi,
        "pause_ticks": 4,
        "reorientation_count": 1,
        "vte_binary": vte_binary,
        "total_reward": 1.0,
    }


def test_analyze_vte_metrics_writes_core_tables(tmp_path):
    metrics_csv = tmp_path / "vte_trial_metrics.csv"
    output_dir = tmp_path / "analysis"

    pd.DataFrame(
        [
            _metric_row(seed=1, trial=1, committed_path="open", vte_binary=1),
            _metric_row(seed=1, trial=2, committed_path="covered", vte_binary=0),
            _metric_row(seed=2, trial=1, committed_path="open", vte_binary=1),
            _metric_row(seed=2, trial=2, committed_path="covered", vte_binary=0),
        ]
    ).to_csv(metrics_csv, index=False)

    meta = analyze_vte_metrics(metrics_csv, output_dir)

    assert meta["n_trials"] == 4
    assert meta["n_seeds"] == 2

    expected = {
        "Table_3_2_VTE_Overall_Summary.csv",
        "Table_3_2_VTE_By_Seed.csv",
        "Table_3_2_VTE_By_Condition.csv",
        "Table_3_2_VTE_By_Committed_Path.csv",
        "Table_3_2_VTE_By_Condition_x_Path.csv",
        "stage3_2_vte_analysis_meta.json",
        "Stage3_2_VTE_Analysis_Report.md",
        "Table_3_2_VTE_Distribution_By_Path.csv",
        "Figure_3_2_VTE_Rate_By_Path.png",
        "Figure_3_2_IdPhi_By_Path_Boxplot.png",
        "Figure_3_2_VTE_Rate_By_Seed.png",
        "Figure_3_2_IdPhi_vs_Pause.png",
    }

    assert expected.issubset({p.name for p in output_dir.iterdir()})


def test_analyze_vte_metrics_rejects_missing_required_columns(tmp_path):
    metrics_csv = tmp_path / "bad.csv"
    output_dir = tmp_path / "analysis"

    pd.DataFrame([{"run_id": "bad"}]).to_csv(metrics_csv, index=False)

    with pytest.raises(ValueError, match="Missing required VTE metric columns"):
        analyze_vte_metrics(metrics_csv, output_dir)


def test_analyze_vte_metrics_metadata_is_json(tmp_path):
    metrics_csv = tmp_path / "vte_trial_metrics.csv"
    output_dir = tmp_path / "analysis"

    pd.DataFrame([_metric_row()]).to_csv(metrics_csv, index=False)

    analyze_vte_metrics(metrics_csv, output_dir)

    meta = json.loads(
        (output_dir / "stage3_2_vte_analysis_meta.json").read_text(encoding="utf-8")
    )

    assert meta["n_trials"] == 1
    assert meta["run_ids"] == ["analysis_test"]