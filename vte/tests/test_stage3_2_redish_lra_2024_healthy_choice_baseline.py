from __future__ import annotations

from pathlib import Path

import pandas as pd

from vte.lab_adapters.redish_lra_2024.build_lra_healthy_choice_baseline import (
    build_lra_healthy_choice_baseline,
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
            "outcome": "correct",
            "reward": 1.0,
            "vte_binary_source": "VTELap.ChoicePoint",
            "vte_binary_for_comparison": 0.0,
            "lab_idphi": 10.0,
            "lab_avg_idphi": 1.0,
            "choice_point_dwell_s": 1.0,
            "pause_time_s": 0.5,
            "primary_event_kind": "feeder_fired",
            "primary_event_code_label": "code_2",
            "strict_event_status": "same_trial_event",
            "strict_primary_event_latency_s": 1.5,
        },
        {
            "dataset_id": "redish_lra_2024",
            "cohort": "lra",
            "treatment": "control",
            "subject_id": "R1",
            "session_id": "R1-2022-01-01",
            "trial": 2,
            "outcome": "error",
            "reward": 0.0,
            "vte_binary_source": "VTELap.ChoicePoint",
            "vte_binary_for_comparison": 1.0,
            "lab_idphi": 30.0,
            "lab_avg_idphi": 2.0,
            "choice_point_dwell_s": 2.0,
            "pause_time_s": 1.5,
            "primary_event_kind": "error_not_fired",
            "primary_event_code_label": "code_4",
            "strict_event_status": "same_trial_event",
            "strict_primary_event_latency_s": 2.5,
        },
        {
            "dataset_id": "redish_lra_2024",
            "cohort": "mpfc_dreadds",
            "treatment": "DCZ",
            "subject_id": "R2",
            "session_id": "R2-2022-02-01",
            "trial": 1,
            "outcome": "correct",
            "reward": 1.0,
            "vte_binary_source": "IdPhi.ChoicePoint>=VTEThreshold",
            "vte_binary_for_comparison": "",
            "lab_idphi": 50.0,
            "lab_avg_idphi": 3.0,
            "choice_point_dwell_s": 3.0,
            "pause_time_s": 2.5,
            "primary_event_kind": "feeder_fired",
            "primary_event_code_label": "code_3",
            "strict_event_status": "same_trial_event",
            "strict_primary_event_latency_s": 1.0,
        },
    ]
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_healthy_choice_baseline_excludes_dreadd_and_keeps_raw_codes(tmp_path):
    input_csv = _write_input(tmp_path / "same_trial.csv")
    output_dir = tmp_path / "out"

    meta = build_lra_healthy_choice_baseline(input_csv, output_dir)

    assert meta["n_input_rows"] == 3
    assert meta["n_output_rows"] == 2
    assert meta["choice_direction_usable"] is False

    endpoint = pd.read_csv(output_dir / "redish_lra17h_healthy_choice_baseline.csv")

    assert set(endpoint["cohort"]) == {"lra"}
    assert set(endpoint["treatment"]) == {"control"}
    assert set(endpoint["vte_binary_source"]) == {"VTELap.ChoicePoint"}
    assert set(endpoint["native_action_code"]) == {"code_2", "code_4"}
    assert set(endpoint["choice_direction_policy"]) == {
        "raw_event_code_only_no_left_right_mapping"
    }
    assert set(endpoint["synthetic_comparable_eligible"]) == {True}
    assert set(endpoint["vte_binary"]) == {0.0, 1.0}
    assert endpoint["z_lab_idphi_by_session"].notna().all()


def test_healthy_choice_baseline_writes_summaries(tmp_path):
    input_csv = _write_input(tmp_path / "same_trial.csv")
    output_dir = tmp_path / "out"

    build_lra_healthy_choice_baseline(input_csv, output_dir)

    assert (output_dir / "Table_Redish_LRA17H_By_VTE.csv").exists()
    assert (output_dir / "Table_Redish_LRA17H_By_Outcome.csv").exists()
    assert (output_dir / "Table_Redish_LRA17H_By_Action_Code.csv").exists()
    assert (output_dir / "Table_Redish_LRA17H_Field_Coverage.csv").exists()
