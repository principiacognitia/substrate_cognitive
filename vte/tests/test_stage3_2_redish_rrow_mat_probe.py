from pathlib import Path

import pandas as pd
import pytest

from vte.lab_adapters.redish_rrow_2022.mat_probe import summarize_mat_file
from vte.lab_adapters.redish_rrow_2022.probe_processed_behavior import (
    build_behavior_endpoint_precheck,
    write_processed_behavior_probe_outputs,
)


scipy_io = pytest.importorskip("scipy.io")


def _write_mat(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    scipy_io.savemat(str(path), payload)
    return path


def _write_minimal_redish_mat_tree(tmp_path: Path) -> Path:
    root = tmp_path / "Redish 2022"
    data = root / "Version 002 - 2022-12-07" / "Data-revision-2022-12-07" / "Data"
    processed = data / "Processed Behavior"

    _write_mat(
        processed / "IdPhi_RRow.mat",
        {
            "IdPhiData": {
                "rat": ["R506"],
                "session": ["R506-2019-01-25"],
                "lap": [[1, 2, 3]],
                "zone": [[1, 2, 3]],
                "idphi": [[0.1, 0.4, 0.9]],
            }
        },
    )

    _write_mat(
        processed / "LapData_Behav_RRow.mat",
        {
            "LapData": {
                "trial": [[1, 2, 3]],
                "choice": [[1, 0, 1]],
                "reward": [[1, 0, 1]],
                "delay": [[1, 5, 10]],
                "zone": [[1, 2, 3]],
            }
        },
    )

    _write_mat(
        processed / "SessionData_RRow.mat",
        {
            "SessionData": {
                "rat": ["R506"],
                "session": ["R506-2019-01-25"],
            }
        },
    )

    _write_mat(
        data / "Misc" / "DataDef_RRow.mat",
        {
            "DataDef": {
                "zoneLabels": ["offer", "wait"],
                "task": ["restaurant_row"],
            }
        },
    )

    session = processed / "R506" / "R506-2019-01-25"
    _write_mat(
        session / "R506-2019-01-25-RRow.mat",
        {
            "RRow": {
                "choice": [[1, 0]],
                "reward": [[1, 0]],
                "offerZone": [[1, 2]],
                "waitZone": [[5, 6]],
            }
        },
    )
    _write_mat(
        session / "R506-2019-01-25-vt.mat",
        {
            "VT": {
                "x": [[1.0, 2.0, 3.0]],
                "y": [[4.0, 5.0, 6.0]],
                "time": [[0.0, 0.1, 0.2]],
            }
        },
    )

    return root


def test_summarize_mat_file_reports_nested_fields(tmp_path):
    mat_path = _write_mat(
        tmp_path / "IdPhi_RRow.mat",
        {
            "IdPhiData": {
                "idphi": [[0.1, 0.2]],
                "lap": [[1, 2]],
            }
        },
    )

    summary = summarize_mat_file(mat_path, max_depth=4)

    assert not summary.empty
    assert any(summary["field_path"].astype(str).str.contains("IdPhiData", case=False))
    assert any(summary["field_path"].astype(str).str.contains("idphi", case=False))


def test_build_behavior_endpoint_precheck_detects_core_concepts():
    structure = pd.DataFrame(
        [
            {"source_file": "IdPhi_RRow.mat", "field_path": "IdPhiData.idphi"},
            {"source_file": "LapData_Behav_RRow.mat", "field_path": "LapData.choice"},
            {"source_file": "LapData_Behav_RRow.mat", "field_path": "LapData.reward"},
            {"source_file": "LapData_Behav_RRow.mat", "field_path": "LapData.delay"},
            {"source_file": "LapData_Behav_RRow.mat", "field_path": "LapData.zone"},
            {"source_file": "SessionData_RRow.mat", "field_path": "SessionData.rat"},
            {"source_file": "R506-2019-01-25-vt.mat", "field_path": "VT.x"},
        ]
    )

    precheck = build_behavior_endpoint_precheck(structure)
    counts = dict(zip(precheck["concept"], precheck["n_matching_fields"]))

    false_positive = pd.DataFrame(
        [
            {"source_file": "DataDef_RRow.mat", "field_path": "Experimenter"},
            {"source_file": "DataDef_RRow.mat", "field_path": "Delays"},
        ]
    )
    false_precheck = build_behavior_endpoint_precheck(false_positive)
    false_counts = dict(zip(false_precheck["concept"], false_precheck["n_matching_fields"]))

    assert false_counts["choice"] == 0
    assert false_counts["tracking_or_pose"] == 0

    assert counts["idphi_or_vte_proxy"] >= 1
    assert counts["choice"] >= 1
    assert counts["reward_or_outcome"] >= 1
    assert counts["delay_or_cost"] >= 1
    assert counts["zone_or_choice_point"] >= 1
    assert counts["tracking_or_pose"] >= 1


def test_write_processed_behavior_probe_outputs(tmp_path):
    root = _write_minimal_redish_mat_tree(tmp_path)
    output_dir = tmp_path / "probe"

    meta = write_processed_behavior_probe_outputs(
        root=root,
        output_dir=output_dir,
        max_depth=4,
        max_rows_per_file=1000,
        max_session_files_per_kind=1,
    )

    assert meta["n_probe_files"] >= 4
    assert meta["n_successful_probe_files"] >= 4
    assert meta["readiness"] in {
        "likely_ready_for_patch_16b_adapter_design",
        "partially_ready_needs_field_mapping",
    }

    assert (output_dir / "Table_Redish_RRow_Mat_Structure_Probe.csv").exists()
    assert (output_dir / "Table_Redish_RRow_Behavior_Endpoint_Precheck.csv").exists()
    assert (output_dir / "Redish_RRow_Processed_Behavior_Probe_Report.md").exists()