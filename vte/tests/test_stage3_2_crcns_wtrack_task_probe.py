import pandas as pd
import numpy as np
from scipy.io import savemat

from vte.lab_adapters.crcns_wtrack.extract_task_probe import (
    build_position_epoch_summary,
    extract_task_probe,
    find_task_epoch_candidates,
)


def _write_minimal_task_probe_tree(root):
    animal = root / "Fiv"
    animal.mkdir(parents=True)

    pos = np.array(
        [
            [0.0, 10.0, 20.0],
            [1.0, 11.0, 21.0],
            [2.0, 12.0, 22.0],
        ]
    )

    savemat(animal / "Fivpos01.mat", {"pos": pos})
    savemat(
        animal / "Fivtask01.mat",
        {
            "task": {
                "type": "wtrack",
                "environment": "wtrack",
                "description": "minimal test task",
            }
        },
    )

    return animal


def test_find_task_epoch_candidates_accepts_metadata_dict():
    mat_data = {
        "task": [
            {
                "type": "wtrack",
                "environment": "wtrack",
                "description": "run epoch",
            },
            {
                "type": "sleep",
                "environment": "box",
                "description": "rest epoch",
            },
        ]
    }

    candidates = find_task_epoch_candidates(mat_data)

    assert len(candidates) == 2
    assert candidates[0].epoch == 1
    assert candidates[0].task_type == "wtrack"
    assert candidates[0].environment == "wtrack"
    assert "description" in candidates[0].task_keys


def test_find_task_epoch_candidates_accepts_numeric_task_array():
    mat_data = {"task": np.array([[1.0, 2.0]])}

    candidates = find_task_epoch_candidates(mat_data)

    assert len(candidates) == 1
    assert candidates[0].field_path == "task"
    assert candidates[0].value_type == "ndarray"


def test_build_position_epoch_summary_aligns_pos_file(tmp_path):
    animal = _write_minimal_task_probe_tree(tmp_path)

    summary = build_position_epoch_summary(
        animal,
        day=1,
        dataset_id="crcns_hc6",
        animal_id="Fiv",
    )

    assert len(summary) == 1
    row = summary.iloc[0]
    assert row["dataset_id"] == "crcns_hc6"
    assert row["animal_id"] == "Fiv"
    assert row["day"] == 1
    assert row["epoch"] == 1
    assert row["n_position_samples"] == 3
    assert row["time_min"] == 0.0
    assert row["time_max"] == 2.0
    assert row["x_min"] == 10.0
    assert row["x_max"] == 12.0
    assert row["y_min"] == 20.0
    assert row["y_max"] == 22.0


def test_extract_task_probe_writes_task_and_position_tables(tmp_path):
    animal = _write_minimal_task_probe_tree(tmp_path)
    output_dir = tmp_path / "task_probe"

    meta = extract_task_probe(
        animal_dir=animal,
        output_dir=output_dir,
        day=1,
        dataset_id="crcns_hc6",
        animal_id="Fiv",
    )

    assert meta["n_task_epoch_candidates"] == 1
    assert meta["n_position_epochs"] == 1

    task_csv = output_dir / "Table_CRCNS_WTrack_Task_Epoch_Probe.csv"
    position_csv = output_dir / "Table_CRCNS_WTrack_Position_Epoch_Summary.csv"
    meta_json = output_dir / "crcns_wtrack_task_probe_meta.json"

    assert task_csv.exists()
    assert position_csv.exists()
    assert meta_json.exists()

    task_probe = pd.read_csv(task_csv)
    assert len(task_probe) == 1

    row = task_probe.iloc[0]
    assert row["dataset_id"] == "crcns_hc6"
    assert row["animal_id"] == "Fiv"
    assert row["day"] == 1
    assert row["epoch"] == 1
    assert row["task_type"] == "wtrack"
    assert row["environment"] == "wtrack"
    assert row["description"] == "minimal test task"
    assert row["n_position_samples"] == 3
    assert row["x_min"] == 10.0
    assert row["x_max"] == 12.0

    position_summary = pd.read_csv(position_csv)
    assert len(position_summary) == 1
    assert set(["time_min", "time_max", "x_min", "x_max", "y_min", "y_max"]).issubset(
        position_summary.columns
    )