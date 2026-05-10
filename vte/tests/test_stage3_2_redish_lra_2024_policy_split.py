from __future__ import annotations

from pathlib import Path

import pandas as pd

from vte.lab_adapters.redish_lra_2024.split_lra_comparability_policy import (
    build_lra_comparability_policy_split,
)


def _write_input(path: Path) -> Path:
    rows = [
        {
            "dataset_id": "redish_lra_2024",
            "cohort": "lra",
            "treatment": "control",
            "subject_id": "R1",
            "session_id": "R1-2022-01-01",
            "trial": 1,
            "decision_stage": "choice_point",
            "outcome": "correct",
            "reward": 1.0,
            "lab_idphi": 10.0,
            "lab_avg_idphi": 1.0,
            "lab_z_idphi": -0.5,
            "lab_robust_z_idphi": -0.4,
            "deliberation_proxy": 10.0,
            "dwell_proxy": 1.5,
            "pause_time_s": 1.4,
            "vte_binary": 0.0,
            "vte_binary_source": "VTELap.ChoicePoint",
            "switch_relation": "pre_switch",
        },
        {
            "dataset_id": "redish_lra_2024",
            "cohort": "lra",
            "treatment": "control",
            "subject_id": "R1",
            "session_id": "R1-2022-01-01",
            "trial": 2,
            "decision_stage": "choice_point",
            "outcome": "error",
            "reward": 0.0,
            "lab_idphi": 80.0,
            "lab_avg_idphi": 3.0,
            "lab_z_idphi": 1.5,
            "lab_robust_z_idphi": 1.4,
            "deliberation_proxy": 80.0,
            "dwell_proxy": 3.5,
            "pause_time_s": 3.4,
            "vte_binary": 1.0,
            "vte_binary_source": "VTELap.ChoicePoint",
            "switch_relation": "post_switch",
        },
        {
            "dataset_id": "redish_lra_2024",
            "cohort": "lra",
            "treatment": "control",
            "subject_id": "R1",
            "session_id": "R1-2022-01-01",
            "trial": 3,
            "decision_stage": "choice_point",
            "outcome": "correct",
            "reward": 1.0,
            "lab_idphi": 120.0,
            "lab_avg_idphi": 4.0,
            "lab_z_idphi": 2.0,
            "lab_robust_z_idphi": 2.1,
            "deliberation_proxy": 120.0,
            "dwell_proxy": 4.0,
            "pause_time_s": 4.0,
            "vte_binary": 1.0,
            "vte_binary_source": "IdPhi.ChoicePoint>=VTEThreshold",
            "switch_relation": "post_switch",
        },
        {
            "dataset_id": "redish_lra_2024",
            "cohort": "mpfc_dreadds",
            "treatment": "DCZ",
            "subject_id": "R2",
            "session_id": "R2-2022-02-01",
            "trial": 1,
            "decision_stage": "choice_point",
            "outcome": "correct",
            "reward": 1.0,
            "lab_idphi": 50.0,
            "lab_avg_idphi": 2.0,
            "lab_z_idphi": 0.5,
            "lab_robust_z_idphi": 0.4,
            "deliberation_proxy": 50.0,
            "dwell_proxy": 2.0,
            "pause_time_s": 2.0,
            "vte_binary": 1.0,
            "vte_binary_source": "IdPhi.ChoicePoint>=VTEThreshold",
            "switch_relation": "pre_switch",
        },
        {
            "dataset_id": "redish_lra_2024",
            "cohort": "mpfc_dreadds",
            "treatment": "VEH",
            "subject_id": "R3",
            "session_id": "R3-2022-02-01",
            "trial": 1,
            "decision_stage": "choice_point",
            "outcome": "error",
            "reward": 0.0,
            "lab_idphi": 60.0,
            "lab_avg_idphi": 2.5,
            "lab_z_idphi": 0.7,
            "lab_robust_z_idphi": 0.6,
            "deliberation_proxy": 60.0,
            "dwell_proxy": 2.5,
            "pause_time_s": 2.5,
            "vte_binary": 1.0,
            "vte_binary_source": "IdPhi.ChoicePoint>=VTEThreshold",
            "switch_relation": "post_switch",
        },
    ]
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_policy_split_keeps_only_native_lra_control_as_baseline(tmp_path):
    input_csv = _write_input(tmp_path / "lra17d.csv")
    output_dir = tmp_path / "out"

    meta = build_lra_comparability_policy_split(input_csv, output_dir)

    assert meta["n_input_rows"] == 5
    assert meta["n_control_baseline_rows"] == 2
    assert meta["n_perturbation_rows"] == 2
    assert meta["n_excluded_rows"] == 1

    control = pd.read_csv(output_dir / "redish_lra17e_healthy_control_baseline.csv")
    assert set(control["cohort"]) == {"lra"}
    assert set(control["treatment"]) == {"control"}
    assert set(control["vte_binary_source"]) == {"VTELap.ChoicePoint"}
    assert set(control["vte_label_policy"]) == {"native_control_vtelap_label"}
    assert set(control["synthetic_comparable_eligible"]) == {True}
    assert set(control["vte_binary_for_comparison"]) == {0.0, 1.0}


def test_policy_split_keeps_dreadd_only_as_continuous_perturbation(tmp_path):
    input_csv = _write_input(tmp_path / "lra17d.csv")
    output_dir = tmp_path / "out"

    build_lra_comparability_policy_split(input_csv, output_dir)

    perturbation = pd.read_csv(
        output_dir / "redish_lra17e_dreadd_perturbation_continuous.csv"
    )

    assert set(perturbation["cohort"]) == {"mpfc_dreadds"}
    assert set(perturbation["biological_comparison_role"]) == {
        "dreadd_perturbation_continuous_only"
    }
    assert set(perturbation["synthetic_comparable_eligible"]) == {False}
    assert set(perturbation["vte_binary_usable"]) == {False}
    assert perturbation["vte_binary_for_comparison"].isna().all()
    assert set(perturbation["treatment_family"]) == {
        "dreadd_perturbation",
        "dreadd_vehicle_control",
    }


def test_policy_split_audit_retains_reconstructed_labels_but_excludes_them(tmp_path):
    input_csv = _write_input(tmp_path / "lra17d.csv")
    output_dir = tmp_path / "out"

    build_lra_comparability_policy_split(input_csv, output_dir)

    excluded = pd.read_csv(output_dir / "redish_lra17e_excluded_from_healthy_baseline.csv")
    assert len(excluded) == 1
    assert excluded.iloc[0]["vte_binary_source"] == "IdPhi.ChoicePoint>=VTEThreshold"
    assert excluded.iloc[0]["biological_comparison_role"] == "excluded_from_healthy_baseline"

    audit = pd.read_csv(output_dir / "Table_Redish_LRA17E_VTE_Label_Source_Audit.csv")
    assert "native_control_vtelap_label" in set(audit["vte_label_policy"])
    assert "do_not_use_as_healthy_baseline_label" in set(audit["vte_label_policy"])
