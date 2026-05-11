from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from vte.lab_adapters.redish_lra_2024.validate_lra_biosynth_comparability import (
    validate_lra_biosynth_comparability,
)


def _write_comparable_csv(path: Path) -> Path:
    rows = [
        {
            "source": "biological",
            "dataset_id": "redish_lra_2024",
            "task_family": "left_right_alternate",
            "subject_or_seed": "R1",
            "session_or_run": "R1-2022-01-01",
            "trial": 1,
            "decision_stage": "choice_point",
            "choice_point_id": "choice_point",
            "action_namespace": "raw_lra_event_code",
            "chosen_action": "code_2",
            "action_comparison_policy": "compare_only_if_action_namespace_matches",
            "outcome": "correct",
            "reward": 1.0,
            "cost": 1.0,
            "native_cost_bin": "balanced",
            "comparable_cost_bin": "balanced",
            "dwell_proxy": 1.0,
            "deliberation_proxy": 10.0,
            "dwell_z": -1.0,
            "deliberation_z": -1.0,
            "vte_binary": 0.0,
            "source_file": "bio.csv",
        },
        {
            "source": "biological",
            "dataset_id": "redish_lra_2024",
            "task_family": "left_right_alternate",
            "subject_or_seed": "R1",
            "session_or_run": "R1-2022-01-01",
            "trial": 2,
            "decision_stage": "choice_point",
            "choice_point_id": "choice_point",
            "action_namespace": "raw_lra_event_code",
            "chosen_action": "code_4",
            "action_comparison_policy": "compare_only_if_action_namespace_matches",
            "outcome": "error",
            "reward": 0.0,
            "cost": 1.0,
            "native_cost_bin": "balanced",
            "comparable_cost_bin": "balanced",
            "dwell_proxy": 2.0,
            "deliberation_proxy": 30.0,
            "dwell_z": 1.0,
            "deliberation_z": 1.0,
            "vte_binary": 1.0,
            "source_file": "bio.csv",
        },
        {
            "source": "synthetic",
            "dataset_id": "stage3_synthetic",
            "task_family": "balanced_fork",
            "subject_or_seed": "seed_1",
            "session_or_run": "run_1",
            "trial": 1,
            "decision_stage": "junction",
            "choice_point_id": "junction",
            "action_namespace": "synthetic_left_right_or_model_native",
            "chosen_action": "open",
            "action_comparison_policy": "compare_only_if_action_namespace_matches",
            "outcome": "success",
            "reward": 0.7,
            "cost": 1.0,
            "native_cost_bin": "balanced",
            "comparable_cost_bin": "balanced",
            "dwell_proxy": 1.0,
            "deliberation_proxy": 5.0,
            "dwell_z": -0.5,
            "deliberation_z": -0.5,
            "vte_binary": 0.0,
            "source_file": "synthetic.csv",
        },
        {
            "source": "synthetic",
            "dataset_id": "stage3_synthetic",
            "task_family": "balanced_fork",
            "subject_or_seed": "seed_1",
            "session_or_run": "run_1",
            "trial": 2,
            "decision_stage": "junction",
            "choice_point_id": "junction",
            "action_namespace": "synthetic_left_right_or_model_native",
            "chosen_action": "covered",
            "action_comparison_policy": "compare_only_if_action_namespace_matches",
            "outcome": "success",
            "reward": 0.9,
            "cost": 1.0,
            "native_cost_bin": "balanced",
            "comparable_cost_bin": "balanced",
            "dwell_proxy": 2.0,
            "deliberation_proxy": 7.0,
            "dwell_z": 0.5,
            "deliberation_z": 0.5,
            "vte_binary": 1.0,
            "source_file": "synthetic.csv",
        },
    ]

    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _write_healthy_baseline_csv(path: Path) -> Path:
    rows = [
        {
            "dataset_id": "redish_lra_2024",
            "cohort": "lra",
            "treatment": "control",
            "vte_binary_source": "VTELap.ChoicePoint",
            "subject_id": "R1",
            "session_id": "R1-2022-01-01",
            "trial": 1,
        },
        {
            "dataset_id": "redish_lra_2024",
            "cohort": "lra",
            "treatment": "control",
            "vte_binary_source": "VTELap.ChoicePoint",
            "subject_id": "R1",
            "session_id": "R1-2022-01-01",
            "trial": 2,
        },
    ]
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _write_bad_baseline_csv(path: Path) -> Path:
    rows = [
        {
            "dataset_id": "redish_lra_2024",
            "cohort": "mpfc_dreadds",
            "treatment": "DCZ",
            "treatment_family": "dreadd_perturbation",
            "vte_binary_source": "IdPhi.ChoicePoint>=VTEThreshold",
            "subject_id": "R2",
            "session_id": "R2-2022-01-01",
            "trial": 1,
        }
    ]
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_lra_biosynth_validation_writes_outputs_and_blocks_action_comparison(tmp_path):
    comparable = _write_comparable_csv(tmp_path / "comparability.csv")
    baseline = _write_healthy_baseline_csv(tmp_path / "baseline.csv")
    output_dir = tmp_path / "out"

    meta = validate_lra_biosynth_comparability(
        comparability_csv=comparable,
        biological_baseline_csv=baseline,
        output_dir=output_dir,
    )

    assert meta["n_rows"] == 4
    assert meta["n_biological_rows"] == 2
    assert meta["n_synthetic_rows"] == 2
    assert meta["n_safety_failures"] == 0
    assert meta["action_labels_comparable"] is False
    assert meta["action_label_comparison_status"] == "not_comparable_namespace_mismatch"
    assert meta["direct_task_comparison_supported"] is False

    assert (output_dir / "Table_LRA_BioSynth18B_Comparable_Rows_Validated.csv").exists()
    assert (output_dir / "Table_LRA_BioSynth18B_Validation_Checks.csv").exists()
    assert (output_dir / "Table_LRA_BioSynth18B_VTE_Direction_Contrast.csv").exists()
    assert (output_dir / "Table_LRA_BioSynth18B_Action_Namespace_Audit.csv").exists()
    assert (output_dir / "Table_LRA_BioSynth18B_Conclusion.csv").exists()
    assert (output_dir / "lra_biosynth18b_validation_meta.json").exists()


def test_lra_biosynth_validation_maps_junction_to_choice_point_and_detects_reward_sign_mismatch(tmp_path):
    comparable = _write_comparable_csv(tmp_path / "comparability.csv")
    baseline = _write_healthy_baseline_csv(tmp_path / "baseline.csv")
    output_dir = tmp_path / "out"

    validate_lra_biosynth_comparability(
        comparability_csv=comparable,
        biological_baseline_csv=baseline,
        output_dir=output_dir,
    )

    validated = pd.read_csv(output_dir / "Table_LRA_BioSynth18B_Comparable_Rows_Validated.csv")
    assert set(validated["decision_stage_canonical"]) == {"choice_point"}
    assert "synthetic_junction_mapped_to_choice_point" in set(
        validated["decision_stage_mapping_policy"]
    )

    direction = pd.read_csv(output_dir / "Table_LRA_BioSynth18B_VTE_Direction_Contrast.csv")
    reward_row = direction[
        (direction["decision_stage_canonical"] == "choice_point")
        & (direction["metric"] == "reward")
    ].iloc[0]

    assert reward_row["biological_vte_minus_nonvte"] < 0
    assert reward_row["synthetic_vte_minus_nonvte"] > 0
    assert reward_row["comparison_status"] == "sign_mismatch"


def test_lra_biosynth_validation_rejects_dreadd_baseline_in_strict_mode(tmp_path):
    comparable = _write_comparable_csv(tmp_path / "comparability.csv")
    bad_baseline = _write_bad_baseline_csv(tmp_path / "bad_baseline.csv")
    output_dir = tmp_path / "out"

    with pytest.raises(ValueError):
        validate_lra_biosynth_comparability(
            comparability_csv=comparable,
            biological_baseline_csv=bad_baseline,
            output_dir=output_dir,
            strict=True,
        )

    checks = pd.read_csv(output_dir / "Table_LRA_BioSynth18B_Validation_Checks.csv")
    assert "fail" in set(checks["status"])