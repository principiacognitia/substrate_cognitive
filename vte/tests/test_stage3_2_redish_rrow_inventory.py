from pathlib import Path

from vte.lab_adapters.redish_rrow_2022.inventory_redish_rrow import (
    build_aggregate_file_matrix,
    build_file_inventory,
    build_session_matrix,
    classify_redish_file,
    discover_data_roots,
)


def _write(path: Path, content: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _write_minimal_redish_tree(tmp_path: Path) -> Path:
    root = tmp_path / "Redish 2022"
    data = root / "Version 002 - 2022-12-07" / "Data-revision-2022-12-07" / "Data"

    _write(data / "Processed Behavior" / "IdPhi_RRow.mat")
    _write(data / "Processed Behavior" / "LapData_Behav_RRow.mat")
    _write(data / "Processed Behavior" / "SessionData_RRow.mat")
    _write(data / "Misc" / "DataDef_RRow.mat")

    session = data / "Processed Behavior" / "R506" / "R506-2019-01-25"
    _write(session / "R506-2019-01-25-RRow.mat")
    _write(session / "R506-2019-01-25-vt.mat")
    _write(session / "R506_2019_01_25_keys.m")

    unit_session = data / "Processed Phys" / "Processed Units" / "R506" / "R506-2019-01-25"
    _write(unit_session / "R506-2019-01-25-Si01_01.t")
    _write(unit_session / "R506-2019-01-25-Si01_01t_WF.mat")

    return root


def test_classify_redish_file_recognizes_core_files():
    assert classify_redish_file(Path("IdPhi_RRow.mat")) == "aggregate_idphi"
    assert classify_redish_file(Path("LapData_Behav_RRow.mat")) == "aggregate_lapdata_behavior"
    assert classify_redish_file(Path("SessionData_RRow.mat")) == "aggregate_sessiondata"
    assert classify_redish_file(Path("DataDef_RRow.mat")) == "data_definition"
    assert classify_redish_file(Path("R506-2019-01-25-RRow.mat")) == "session_rrow_behavior"
    assert classify_redish_file(Path("R506-2019-01-25-vt.mat")) == "session_tracking_vt"
    assert classify_redish_file(Path("R506_2019_01_25_keys.m")) == "session_keys"
    assert classify_redish_file(Path("R506-2019-01-25-Si01_01.t")) == "unit_spike_times"
    assert classify_redish_file(Path("R506-2019-01-25-Si01_01t_WF.mat")) == "unit_waveform"


def test_discover_data_roots_finds_nested_redish_data_root(tmp_path):
    root = _write_minimal_redish_tree(tmp_path)

    data_roots = discover_data_roots(root)

    assert len(data_roots) == 1
    assert data_roots[0].name == "Data"
    assert (data_roots[0] / "Processed Behavior").is_dir()


def test_build_file_inventory_extracts_subject_session_and_kinds(tmp_path):
    root = _write_minimal_redish_tree(tmp_path)

    inventory = build_file_inventory(root)

    assert not inventory.empty
    assert "aggregate_idphi" in set(inventory["file_kind"])
    assert "session_rrow_behavior" in set(inventory["file_kind"])
    assert "session_tracking_vt" in set(inventory["file_kind"])
    assert "unit_spike_times" in set(inventory["file_kind"])

    session_rows = inventory.loc[inventory["session_id"] == "R506-2019-01-25"]
    assert not session_rows.empty
    assert set(session_rows["subject_id"]) == {"R506"}
    assert set(session_rows["session_date"]) == {"2019-01-25"}


def test_build_session_matrix_marks_required_session_files(tmp_path):
    root = _write_minimal_redish_tree(tmp_path)
    inventory = build_file_inventory(root)

    matrix = build_session_matrix(inventory)

    row = matrix.loc[matrix["session_id"] == "R506-2019-01-25"].iloc[0]
    assert bool(row["has_rrow_behavior"]) is True
    assert bool(row["has_tracking_vt"]) is True
    assert bool(row["has_keys"]) is True
    assert int(row["n_unit_spike_files"]) == 1
    assert int(row["n_unit_waveform_files"]) == 1


def test_build_aggregate_file_matrix_marks_core_aggregate_files(tmp_path):
    root = _write_minimal_redish_tree(tmp_path)
    inventory = build_file_inventory(root)

    matrix = build_aggregate_file_matrix(inventory)

    assert len(matrix) == 1
    row = matrix.iloc[0]
    assert bool(row["has_idphi_rrow"]) is True
    assert bool(row["has_lapdata_behav_rrow"]) is True
    assert bool(row["has_sessiondata_rrow"]) is True
    assert bool(row["has_datadef_rrow"]) is True
    assert int(row["n_sessions"]) == 1