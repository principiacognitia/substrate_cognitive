from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from vte.lab_adapters.redish_lra_2024.build_lra_biosynth_visualization_bridge import (
    build_lra_biosynth_visualization_bridge,
)


def _write_18b_input(path: Path) -> Path:
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
            "action_label_comparison_status": "not_comparable_namespace_mismatch",
            "outcome": "correct",
            "reward": 1.0,
            "cost": 1.0,
            "native_cost_bin": "balanced",
            "comparable_cost_bin": "balanced",
            "vte_binary": 0.0,
            "dwell_proxy": 1.0,
            "deliberation_proxy": 10.0,
            "dwell_z": -0.5,
            "deliberation_z": -1.0,
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
            "chosen_action": "code_3",
            "action_label_comparison_status": "not_comparable_namespace_mismatch",
            "outcome": "error",
            "reward": 0.0,
            "cost": 1.0,
            "native_cost_bin": "balanced",
            "comparable_cost_bin": "balanced",
            "vte_binary": 1.0,
            "dwell_proxy": 2.0,
            "deliberation_proxy": 30.0,
            "dwell_z": 0.5,
            "deliberation_z": 1.0,
            "source_file": "bio.csv",
        },
        {
            "source": "synthetic",
            "dataset_id": "synthetic_balanced_fork",
            "task_family": "balanced_fork",
            "subject_or_seed": "seed_1",
            "session_or_run": "run_1",
            "trial": 1,
            "decision_stage": "junction",
            "choice_point_id": "junction",
            "action_namespace": "synthetic_left_right_or_model_native",
            "chosen_action": "left",
            "action_label_comparison_status": "not_comparable_namespace_mismatch",
            "outcome": "correct",
            "reward": 1.0,
            "cost": 1.0,
            "native_cost_bin": "balanced",
            "comparable_cost_bin": "balanced",
            "vte_binary": 0.0,
            "dwell_proxy": 1.0,
            "deliberation_proxy": 1.0,
            "dwell_z": -0.7,
            "deliberation_z": -0.8,
            "source_file": "synthetic.csv",
        },
        {
            "source": "synthetic",
            "dataset_id": "synthetic_balanced_fork",
            "task_family": "balanced_fork",
            "subject_or_seed": "seed_1",
            "session_or_run": "run_1",
            "trial": 2,
            "decision_stage": "junction",
            "choice_point_id": "junction",
            "action_namespace": "synthetic_left_right_or_model_native",
            "chosen_action": "right",
            "action_label_comparison_status": "not_comparable_namespace_mismatch",
            "outcome": "correct",
            "reward": 1.0,
            "cost": 1.0,
            "native_cost_bin": "balanced",
            "comparable_cost_bin": "balanced",
            "vte_binary": 1.0,
            "dwell_proxy": 3.0,
            "deliberation_proxy": 4.0,
            "dwell_z": 0.9,
            "deliberation_z": 1.1,
            "source_file": "synthetic.csv",
        },
    ]
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_visualization_bridge_writes_endpoint_and_metadata(tmp_path: Path):
    input_csv = _write_18b_input(tmp_path / "18b.csv")
    output_dir = tmp_path / "out"

    meta = build_lra_biosynth_visualization_bridge(
        input_csv=input_csv,
        output_dir=output_dir,
    )

    assert meta["patch"] == "18C"
    assert meta["n_output_rows"] == 4
    assert meta["n_biological_rows"] == 2
    assert meta["n_synthetic_rows"] == 2
    assert meta["movement_trace_available"] is False
    assert meta["task_equivalence_status"] == "diagnostic_only_not_direct_task_match"

    endpoint = pd.read_csv(
        output_dir / "Table_LRA_BioSynth18C_Visualization_Decision_Endpoint.csv"
    )

    assert set(endpoint["source"]) == {"biological", "synthetic"}
    assert set(endpoint["visualization_level"]) == {"decision_endpoint"}
    assert set(endpoint["movement_trace_available"].astype(str).str.lower()) <= {"false", "0"}
    assert set(endpoint["decision_stage_canonical"]) == {"choice_point"}
    assert set(endpoint["action_label_comparison_status"]) == {
        "not_comparable_namespace_mismatch"
    }

    meta_json = json.loads(
        (output_dir / "lra_biosynth18c_visualization_bridge_meta.json").read_text(
            encoding="utf-8"
        )
    )
    assert meta_json["outputs"]["endpoint"].endswith(
        "Table_LRA_BioSynth18C_Visualization_Decision_Endpoint.csv"
    )


def test_visualization_bridge_writes_dictionary_summary_and_coverage(tmp_path: Path):
    input_csv = _write_18b_input(tmp_path / "18b.csv")
    output_dir = tmp_path / "out"

    build_lra_biosynth_visualization_bridge(
        input_csv=input_csv,
        output_dir=output_dir,
    )

    assert (
        output_dir / "Table_LRA_BioSynth18C_Visualization_Source_Summary.csv"
    ).exists()
    assert (
        output_dir / "Table_LRA_BioSynth18C_Visualization_Field_Dictionary.csv"
    ).exists()
    assert (
        output_dir / "Table_LRA_BioSynth18C_Visualization_Field_Coverage.csv"
    ).exists()
    assert (
        output_dir / "LRA_BioSynth18C_Visualization_Bridge_Report.md"
    ).exists()

    dictionary = pd.read_csv(
        output_dir / "Table_LRA_BioSynth18C_Visualization_Field_Dictionary.csv"
    )
    assert "movement_trace_available" in set(dictionary["field"])
    assert "task_equivalence_status" in set(dictionary["field"])