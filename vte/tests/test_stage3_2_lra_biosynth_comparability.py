from pathlib import Path

import pandas as pd
import pytest

from vte.lab_adapters.redish_lra_2024.build_lra_biosynth_comparability import (
    build_lra_biosynth_comparability,
)


def _write_biological(path: Path) -> Path:
    df = pd.DataFrame(
        [
            {
                "dataset_id": "redish_lra_2024",
                "task_family": "left_right_alternate",
                "subject_id": "R1",
                "session_id": "R1-2022-01-01",
                "trial": 1,
                "decision_stage": "choice_point",
                "outcome": "correct",
                "reward": 1,
                "vte_binary": 0,
                "lab_idphi": 50.0,
                "z_lab_idphi_by_session": -1.0,
                "dwell_proxy": 1.0,
                "z_dwell_proxy_by_session": -0.5,
                "native_action_code": "code_2",
                "choice_direction_policy": "raw_event_code_only_no_left_right_mapping",
            },
            {
                "dataset_id": "redish_lra_2024",
                "task_family": "left_right_alternate",
                "subject_id": "R1",
                "session_id": "R1-2022-01-01",
                "trial": 2,
                "decision_stage": "choice_point",
                "outcome": "error",
                "reward": 0,
                "vte_binary": 1,
                "lab_idphi": 250.0,
                "z_lab_idphi_by_session": 1.0,
                "dwell_proxy": 2.0,
                "z_dwell_proxy_by_session": 0.5,
                "native_action_code": "code_3",
                "choice_direction_policy": "raw_event_code_only_no_left_right_mapping",
            },
        ]
    )
    df.to_csv(path, index=False)
    return path


def _write_synthetic(path: Path) -> Path:
    df = pd.DataFrame(
        [
            {
                "dataset_id": "synthetic_balanced_fork",
                "task_family": "balanced_fork",
                "seed": "seed_1",
                "run_id": "run_1",
                "trial": 1,
                "decision_stage": "choice_point",
                "chosen_action_namespace": "left_right",
                "chosen_action": "left",
                "reward": 1,
                "vte_binary": 0,
                "raw_idphi": 1.0,
                "z_idphi": -0.7,
                "dwell_proxy": 1.2,
                "dwell_z": -0.4,
            },
            {
                "dataset_id": "synthetic_balanced_fork",
                "task_family": "balanced_fork",
                "seed": "seed_1",
                "run_id": "run_1",
                "trial": 2,
                "decision_stage": "choice_point",
                "chosen_action_namespace": "left_right",
                "chosen_action": "right",
                "reward": 0,
                "vte_binary": 1,
                "raw_idphi": 3.0,
                "z_idphi": 0.9,
                "dwell_proxy": 2.4,
                "dwell_z": 0.6,
            },
        ]
    )
    df.to_csv(path, index=False)
    return path


def test_lra_biosynth_marks_action_namespace_mismatch_without_failing(tmp_path):
    biological = _write_biological(tmp_path / "bio.csv")
    synthetic = _write_synthetic(tmp_path / "synthetic.csv")
    output_dir = tmp_path / "out"

    meta = build_lra_biosynth_comparability(
        biological_csv=biological,
        synthetic_csvs=[synthetic],
        output_dir=output_dir,
    )

    assert meta["n_comparable_rows"] == 4
    assert meta["n_biological_rows"] == 2
    assert meta["n_synthetic_rows"] == 2
    assert meta["action_labels_comparable"] is False
    assert meta["action_label_comparison_status"] == "not_comparable_namespace_mismatch"

    audit = pd.read_csv(output_dir / "Table_LRA_BioSynth_Action_Namespace_Audit.csv")
    assert set(audit["action_label_comparison_status"]) == {"not_comparable_namespace_mismatch"}
    assert set(audit["action_labels_comparable"].astype(str).str.lower()) <= {"false", "0"}


def test_lra_biosynth_can_fail_on_action_namespace_mismatch(tmp_path):
    biological = _write_biological(tmp_path / "bio.csv")
    synthetic = _write_synthetic(tmp_path / "synthetic.csv")

    with pytest.raises(ValueError, match="Action-label namespace mismatch"):
        build_lra_biosynth_comparability(
            biological_csv=biological,
            synthetic_csvs=[synthetic],
            output_dir=tmp_path / "out",
            fail_on_action_namespace_mismatch=True,
        )


def test_lra_biosynth_vte_contrast_is_metric_level_not_action_label_level(tmp_path):
    biological = _write_biological(tmp_path / "bio.csv")
    synthetic = _write_synthetic(tmp_path / "synthetic.csv")
    output_dir = tmp_path / "out"

    build_lra_biosynth_comparability(
        biological_csv=biological,
        synthetic_csvs=[synthetic],
        output_dir=output_dir,
    )

    contrast = pd.read_csv(output_dir / "Table_LRA_BioSynth_VTE_Contrast_By_Source.csv")

    bio = contrast[contrast["source"] == "biological"].iloc[0]
    syn = contrast[contrast["source"] == "synthetic"].iloc[0]

    assert bio["deliberation_z_vte_minus_nonvte"] > 0
    assert syn["deliberation_z_vte_minus_nonvte"] > 0

    rows = pd.read_csv(output_dir / "Table_LRA_BioSynth_Comparable_Decision_Rows.csv")
    assert "action_namespace" in rows.columns
    assert "chosen_action" in rows.columns
