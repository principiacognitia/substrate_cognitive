import pandas as pd
import numpy as np
from scipy.io import savemat

from vte.core.schema import REQUIRED_TRACE_COLUMNS
from vte.core.wrapper import read_trace_csv
from vte.lab_adapters.crcns_wtrack.convert_position_to_canonical_trace import (
    convert_position_to_canonical_trace,
)


def _write_minimal_canonical_trace_tree(root):
    animal = root / "Fiv"
    animal.mkdir(parents=True)

    pos_epoch_1 = np.array(
        [
            [0.0, 10.0, 20.0],
            [1.0, 11.0, 20.0],
            [2.0, 11.0, 21.0],
        ]
    )
    pos_epoch_2 = np.array(
        [
            [10.0, 30.0, 40.0],
            [11.0, 31.0, 40.0],
            [12.0, 32.0, 40.0],
        ]
    )

    pos_cell = np.empty((2,), dtype=object)
    pos_cell[0] = pos_epoch_1
    pos_cell[1] = pos_epoch_2

    savemat(animal / "Fivpos01.mat", {"pos": pos_cell})
    savemat(
        animal / "Fivtask01.mat",
        {
            "task": [
                {
                    "type": "run",
                    "environment": "TrackA",
                    "description": "epoch one",
                },
                {
                    "type": "sleep",
                    "environment": "Box",
                    "description": "epoch two",
                },
            ]
        },
    )

    return animal


def test_convert_position_to_canonical_trace_writes_required_outputs(tmp_path):
    animal = _write_minimal_canonical_trace_tree(tmp_path)
    output_dir = tmp_path / "canonical"

    meta = convert_position_to_canonical_trace(
        animal_dir=animal,
        output_dir=output_dir,
        day=1,
        dataset_id="crcns_hc6",
        animal_id="Fiv",
    )

    assert meta["n_epochs"] == 2
    assert meta["n_trace_rows"] == 6
    assert meta["raw_epoch_level"] is True
    assert meta["choice_zone_inference"] is False
    assert meta["trial_segmentation"] is False

    assert (output_dir / "crcns_wtrack_canonical_trace.csv").exists()
    assert (output_dir / "Table_CRCNS_WTrack_Canonical_Epoch_Summary.csv").exists()
    assert (output_dir / "crcns_wtrack_canonical_trace_meta.json").exists()


def test_canonical_trace_satisfies_vte_required_schema(tmp_path):
    animal = _write_minimal_canonical_trace_tree(tmp_path)
    output_dir = tmp_path / "canonical"

    convert_position_to_canonical_trace(
        animal_dir=animal,
        output_dir=output_dir,
        day=1,
        dataset_id="crcns_hc6",
        animal_id="Fiv",
    )

    trace_csv = output_dir / "crcns_wtrack_canonical_trace.csv"
    trace = read_trace_csv(trace_csv)

    for col in REQUIRED_TRACE_COLUMNS:
        assert col in trace.columns

    assert set(trace["run_id"]) == {"crcns_hc6_Fiv_day01"}
    assert set(trace["seed"]) == {"Fiv"}
    assert set(trace["pose_source"]) == {"lab_tracking"}
    assert set(trace["trace_origin"]) == {"biological"}
    assert set(trace["dataset_id"]) == {"crcns_hc6"}
    assert set(trace["subject_id"]) == {"Fiv"}
    assert set(trace["geometry_id"]) == {"crcns_wtrack_raw"}


def test_canonical_trace_epoch_level_trial_mapping_and_done_flags(tmp_path):
    animal = _write_minimal_canonical_trace_tree(tmp_path)
    output_dir = tmp_path / "canonical"

    convert_position_to_canonical_trace(
        animal_dir=animal,
        output_dir=output_dir,
        day=1,
        dataset_id="crcns_hc6",
        animal_id="Fiv",
    )

    trace = pd.read_csv(output_dir / "crcns_wtrack_canonical_trace.csv")

    assert set(trace["trial"]) == {1, 2}
    assert set(trace["epoch"]) == {1, 2}

    for _, g in trace.groupby("trial"):
        assert g["tick"].tolist() == [1, 2, 3]
        assert g["done"].tolist() == [False, False, True]
        assert g["at_choice_point"].tolist() == [True, True, True]
        assert g["choice_point_id"].iloc[0].startswith("epoch_")


def test_canonical_trace_preserves_task_metadata_and_epoch_summary(tmp_path):
    animal = _write_minimal_canonical_trace_tree(tmp_path)
    output_dir = tmp_path / "canonical"

    convert_position_to_canonical_trace(
        animal_dir=animal,
        output_dir=output_dir,
        day=1,
        dataset_id="crcns_hc6",
        animal_id="Fiv",
    )

    trace = pd.read_csv(output_dir / "crcns_wtrack_canonical_trace.csv")
    summary = pd.read_csv(output_dir / "Table_CRCNS_WTrack_Canonical_Epoch_Summary.csv")

    epoch_1 = trace.loc[trace["epoch"] == 1].iloc[0]
    epoch_2 = trace.loc[trace["epoch"] == 2].iloc[0]

    assert epoch_1["task_type"] == "run"
    assert epoch_1["task_environment"] == "TrackA"
    assert epoch_1["task_description"] == "epoch one"

    assert epoch_2["task_type"] == "sleep"
    assert epoch_2["task_environment"] == "Box"
    assert epoch_2["task_description"] == "epoch two"

    assert len(summary) == 2
    assert summary.loc[summary["epoch"] == 1, "n_samples"].iloc[0] == 3
    assert summary.loc[summary["epoch"] == 2, "n_samples"].iloc[0] == 3


def test_canonical_trace_optional_downsampling_cap(tmp_path):
    animal = _write_minimal_canonical_trace_tree(tmp_path)
    output_dir = tmp_path / "canonical"

    meta = convert_position_to_canonical_trace(
        animal_dir=animal,
        output_dir=output_dir,
        day=1,
        dataset_id="crcns_hc6",
        animal_id="Fiv",
        max_samples_per_epoch=2,
    )

    trace = pd.read_csv(output_dir / "crcns_wtrack_canonical_trace.csv")

    assert meta["n_trace_rows"] == 4
    assert all(len(g) == 2 for _, g in trace.groupby("trial"))
    assert all(g["done"].tolist() == [False, True] for _, g in trace.groupby("trial"))