import pandas as pd
import numpy as np
from scipy.io import savemat

from vte.lab_adapters.crcns_wtrack.extract_position_probe import (
    extract_position_probe,
    find_position_candidates,
)
from vte.lab_adapters.crcns_wtrack.inventory_crcns_wtrack import (
    build_day_file_matrix,
    build_file_inventory,
    write_inventory,
)


def _write_minimal_crcns_tree(root):
    animal = root / "Fiv"
    eeg = animal / "EEG"
    eeg.mkdir(parents=True)

    savemat(animal / "Fivpos01.mat", {"pos": np.array([[0.0, 1.0, 2.0], [1.0, 3.0, 4.0]])})
    savemat(animal / "Fivrawpos01.mat", {"rawpos": np.array([[0.0, 10.0, 20.0], [1.0, 30.0, 40.0]])})
    savemat(animal / "Fivtask01.mat", {"task": np.array([[1.0]])})
    savemat(animal / "Fivspikes01.mat", {"spikes": np.array([[1.0]])})
    savemat(animal / "Fivcellinfo.mat", {"cellinfo": np.array([[1.0]])})
    savemat(eeg / "Fiveeg01-1-01.mat", {"eeg": np.array([[1.0]])})

    return animal


def test_build_file_inventory_classifies_crcns_files(tmp_path):
    animal = _write_minimal_crcns_tree(tmp_path)

    inventory = build_file_inventory(animal, dataset_id="crcns_hc6", animal_id="Fiv")

    assert set(inventory["file_kind"]) == {
        "pos",
        "rawpos",
        "task",
        "spikes",
        "metadata",
        "eeg",
    }
    assert set(inventory["animal_id"]) == {"Fiv"}
    assert inventory.loc[inventory["file_kind"] == "eeg", "inspected"].iloc[0] is False


def test_build_day_file_matrix_reports_day_availability(tmp_path):
    animal = _write_minimal_crcns_tree(tmp_path)
    inventory = build_file_inventory(animal, dataset_id="crcns_hc6", animal_id="Fiv")
    matrix = build_day_file_matrix(inventory)

    assert len(matrix) == 1

    row = matrix.iloc[0]
    assert row["dataset_id"] == "crcns_hc6"
    assert row["animal_id"] == "Fiv"
    assert row["day"] == 1
    assert bool(row["has_pos"]) is True
    assert bool(row["has_rawpos"]) is True
    assert bool(row["has_task"]) is True
    assert bool(row["has_spikes"]) is True
    assert row["n_eeg_files"] == 1


def test_write_inventory_outputs_expected_tables(tmp_path):
    animal = _write_minimal_crcns_tree(tmp_path)
    output_dir = tmp_path / "inventory"

    meta = write_inventory(
        animal_dir=animal,
        output_dir=output_dir,
        dataset_id="crcns_hc6",
        animal_id="Fiv",
    )

    assert meta["n_mat_files"] == 6
    assert (output_dir / "Table_CRCNS_WTrack_File_Inventory.csv").exists()
    assert (output_dir / "Table_CRCNS_WTrack_Day_File_Matrix.csv").exists()
    assert (output_dir / "crcns_wtrack_inventory_meta.json").exists()

    file_inventory = pd.read_csv(output_dir / "Table_CRCNS_WTrack_File_Inventory.csv")
    assert len(file_inventory) == 6


def test_find_position_candidates_accepts_time_xy_matrix():
    mat_data = {
        "pos": np.array(
            [
                [0.0, 1.0, 2.0],
                [1.0, 3.0, 4.0],
                [2.0, 5.0, 6.0],
            ]
        )
    }

    candidates = find_position_candidates(mat_data)

    assert len(candidates) == 1
    assert candidates[0].time_col == 0
    assert candidates[0].x_col == 1
    assert candidates[0].y_col == 2


def test_extract_position_probe_writes_table_and_figure(tmp_path):
    animal = _write_minimal_crcns_tree(tmp_path)
    output_dir = tmp_path / "probe"

    meta = extract_position_probe(
        animal_dir=animal,
        output_dir=output_dir,
        day=1,
        dataset_id="crcns_hc6",
        animal_id="Fiv",
    )

    assert meta["n_position_candidates"] >= 1
    assert (output_dir / "Table_CRCNS_WTrack_Position_Probe.csv").exists()
    assert (output_dir / "Figure_CRCNS_WTrack_XY_Day01_Epochs.png").exists()
    assert (output_dir / "crcns_wtrack_position_probe_meta.json").exists()

    probe = pd.read_csv(output_dir / "Table_CRCNS_WTrack_Position_Probe.csv")
    assert set(["x_min", "x_max", "y_min", "y_max"]).issubset(probe.columns)