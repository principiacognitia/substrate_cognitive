from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vte.lab_adapters.redish_rrow_2022.extract_processed_behavior_endpoints import (
    _cell_list_for_sessions,
    extract_processed_behavior_endpoints,
)


scipy_io = pytest.importorskip("scipy.io")


def _cell(items):
    arr = np.empty((len(items),), dtype=object)
    for idx, item in enumerate(items):
        arr[idx] = np.asarray(item)
    return arr


def _write_mat(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    scipy_io.savemat(str(path), payload)
    return path


def _write_minimal_redish_endpoint_tree(tmp_path: Path) -> Path:
    root = tmp_path / "Redish 2022"
    data = root / "Version 002 - 2022-12-07" / "Data-revision-2022-12-07" / "Data"

    processed = data / "Processed Behavior"
    misc = data / "Misc"

    _write_mat(
        misc / "DataDef_RRow.mat",
        {
            "Directory": np.array(["R001-2020-01-01", "R002-2020-01-02"], dtype=object),
            "Date": np.array(["2020-01-01", "2020-01-02"], dtype=object),
            "ExpType": np.array(["RestaurantRow", "RestaurantRow"], dtype=object),
            "Delays": _cell([[1, 5, 10], [2, 6, 12]]),
        },
    )

    _write_mat(
        processed / "SessionData_RRow.mat",
        {
            "Rat": np.array(["R001", "R002"], dtype=object),
            "Session": np.array(["R001-2020-01-01", "R002-2020-01-02"], dtype=object),
        },
    )

    _write_mat(
        processed / "LapData_Behav_RRow.mat",
        {
            "AcceptOffer": _cell([[1, 0, 1], [0, 1]]),
            "SkipOffer": _cell([[0, 1, 0], [1, 0]]),
            "QuitOffer": _cell([[0, 0, 0], [0, 0]]),
            "EarnOffer": _cell([[1, 0, 1], [0, 1]]),
            "FoodReceived": _cell([[1, 0, 1], [0, 1]]),
            "Decision": _cell([[1, 0, 1], [0, 1]]),
            "CurrentLap": _cell([[1, 2, 3], [1, 2]]),
            "ZoneID": _cell([[5, 2, 6], [1, 8]]),
            "ZoneDelay": _cell([[1, 5, 10], [2, 12]]),
            "SiteRank": _cell([[1, 2, 3], [1, 4]]),
            "EnteringZoneTime": _cell([[10.0, 20.0, 30.0], [5.0, 15.0]]),
            "ExitZoneTime": _cell([[12.0, 22.0, 34.0], [7.0, 18.0]]),
            "TotalSiteTime": _cell([[2.0, 2.0, 4.0], [2.0, 3.0]]),
            "PauseTime": _cell([[0.5, 1.0, 2.0], [0.2, 1.5]]),
            "RunSpeed": _cell([[20.0, 18.0, 15.0], [22.0, 11.0]]),
        },
    )

    _write_mat(
        processed / "IdPhi_RRow.mat",
        {
            "IdPhi": _cell([[0.1, 0.4, 0.9], [0.2, 0.8]]),
            "AvgDPhi": _cell([[0.05, 0.2, 0.45], [0.1, 0.4]]),
            "CurrentLap": _cell([[1, 2, 3], [1, 2]]),
            "ZoneID": _cell([[5, 2, 6], [1, 8]]),
        },
    )

    session = processed / "R001" / "R001-2020-01-01"
    _write_mat(session / "R001-2020-01-01-RRow.mat", {"RRow": np.array([1, 2, 3])})
    _write_mat(session / "R001-2020-01-01-vt.mat", {"VT": np.array([[0, 1], [1, 2]])})

    return root


def test_extract_processed_behavior_endpoints_writes_outputs(tmp_path):
    root = _write_minimal_redish_endpoint_tree(tmp_path)
    output_dir = tmp_path / "endpoint"

    meta = extract_processed_behavior_endpoints(
        root=root,
        output_dir=output_dir,
        max_rows_per_session=None,
    )

    assert meta["n_sessions"] == 2
    assert meta["n_endpoint_rows"] == 5

    assert (output_dir / "Table_Redish_RRow_Session_Metadata.csv").exists()
    assert (output_dir / "Table_Redish_RRow_LapData_Long.csv").exists()
    assert (output_dir / "Table_Redish_RRow_IdPhi_Long.csv").exists()
    assert (output_dir / "Table_Redish_RRow_Choice_IdPhi_Endpoint.csv").exists()
    assert (output_dir / "Table_Redish_RRow_Endpoint_By_Choice.csv").exists()
    assert (output_dir / "Table_Redish_RRow_Endpoint_By_Session.csv").exists()
    assert (output_dir / "Redish_RRow_Processed_Behavior_Endpoint_Report.md").exists()


def test_endpoint_table_contains_choice_reward_delay_and_idphi(tmp_path):
    root = _write_minimal_redish_endpoint_tree(tmp_path)
    output_dir = tmp_path / "endpoint"

    extract_processed_behavior_endpoints(root=root, output_dir=output_dir)

    endpoint = pd.read_csv(output_dir / "Table_Redish_RRow_Choice_IdPhi_Endpoint.csv")

    assert set(endpoint["subject_id"]) == {"R001", "R002"}
    assert set(endpoint["choice"]) == {"accept", "skip"}
    assert set(endpoint["zone_type"]) >= {"offer_zone", "wait_zone"}

    first = endpoint.iloc[0]
    assert first["session_id"] == "R001-2020-01-01"
    assert first["trial"] == 1
    assert first["zone_id"] == 5
    assert first["zone_type"] == "offer_zone"
    assert first["choice"] == "accept"
    assert first["reward"] == 1
    assert first["zone_delay"] == 1
    assert abs(first["lab_idphi"] - 0.1) < 1e-9
    assert abs(first["lab_avg_dphi"] - 0.05) < 1e-9


def test_wait_and_offer_zone_mapping(tmp_path):
    root = _write_minimal_redish_endpoint_tree(tmp_path)
    output_dir = tmp_path / "endpoint"

    extract_processed_behavior_endpoints(root=root, output_dir=output_dir)

    endpoint = pd.read_csv(output_dir / "Table_Redish_RRow_Choice_IdPhi_Endpoint.csv")

    zone_map = dict(zip(endpoint["zone_id"], endpoint["zone_type"]))
    assert zone_map[1] == "wait_zone"
    assert zone_map[2] == "wait_zone"
    assert zone_map[5] == "offer_zone"
    assert zone_map[8] == "offer_zone"


def test_max_rows_per_session_caps_endpoint_rows(tmp_path):
    root = _write_minimal_redish_endpoint_tree(tmp_path)
    output_dir = tmp_path / "endpoint"

    meta = extract_processed_behavior_endpoints(
        root=root,
        output_dir=output_dir,
        max_rows_per_session=1,
    )

    endpoint = pd.read_csv(output_dir / "Table_Redish_RRow_Choice_IdPhi_Endpoint.csv")

    assert meta["n_endpoint_rows"] == 2
    assert len(endpoint) == 2
    assert list(endpoint["row_index"]) == [0, 0]

def test_cell_list_for_sessions_splits_transposed_session_matrix():
    matrix = np.array(
        [
            [1, 10],
            [2, 20],
            [3, 30],
        ]
    )

    cells = _cell_list_for_sessions(matrix, n_sessions=2)

    assert len(cells) == 2
    assert np.array_equal(cells[0], np.array([1, 2, 3]))
    assert np.array_equal(cells[1], np.array([10, 20, 30]))


def test_cell_list_for_sessions_does_not_replicate_unaligned_large_matrix():
    matrix = np.ones((5, 7))

    cells = _cell_list_for_sessions(matrix, n_sessions=2)

    assert cells == ["", ""]