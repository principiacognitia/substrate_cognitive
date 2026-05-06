from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from vte.lab_adapters.dandi_000115.nwb_behavior_probe import (
    classify_statescript_payload,
    parse_task_epoch_ids,
    probe_dandi000115_behavior,
)


def _write_minimal_dandi_nwb(path: Path) -> None:
    with h5py.File(path, "w") as f:
        intervals = f.create_group("intervals")
        epochs = intervals.create_group("epochs")
        epochs.create_dataset("id", data=np.array([1, 2], dtype=np.int64))
        epochs.create_dataset("start_time", data=np.array([1000.0, 2000.0]))
        epochs.create_dataset("stop_time", data=np.array([1200.0, 2300.0]))

        processing = f.create_group("processing")
        behavior = processing.create_group("behavior")

        pos = behavior.create_group("position")
        series = pos.create_group("series_0")
        series.attrs["description"] = "xloc, yloc, xloc2, yloc2"
        data = series.create_dataset(
            "data",
            data=np.array(
                [
                    [1, 2, 3, 4],
                    [2, 3, 4, 5],
                    [3, 4, 5, 6],
                    [4, 5, 6, 7],
                ],
                dtype=np.int16,
            ),
        )
        data.attrs["conversion"] = 0.5
        data.attrs["unit"] = "meters"
        series.create_dataset(
            "timestamps",
            data=np.array([2000.0, 2001.0, 2002.0, np.nan]),
        )

        events = behavior.create_group("behavioral_events")
        arm1beam = events.create_group("arm1beam")
        arm1beam.create_dataset("data", data=np.array([0, 1, 0], dtype=np.int8))
        arm1beam.create_dataset("timestamps", data=np.array([2000.0, 2001.0, 2002.0]))

        homepump = events.create_group("homepump")
        homepump.create_dataset("data", data=np.array([0, 1, 0], dtype=np.int8))
        homepump.create_dataset("timestamps", data=np.array([2000.5, 2001.5, 2002.5]))

        associated = processing.create_group("associated_files")
        statescript = associated.create_group("statescript_r1")
        statescript.attrs["description"] = "Statescript log r1"
        statescript.attrs["task_epochs"] = "2, "
        statescript.attrs[
            "content"
        ] = """# header
100 UP 10
101 waslock = 0
102 512 1024
200 DOWN 10
250 homeCount = 1
300 LOCKOUT 1
"""


def test_parse_helpers():
    assert parse_task_epoch_ids("2, ") == [2]
    assert parse_task_epoch_ids("1, 3,") == [1, 3]

    assert classify_statescript_payload("UP 10")["event_type"] == "well_transition"
    assert classify_statescript_payload("DOWN 12")["well"] == "12"
    assert classify_statescript_payload("homeCount = 3")["event_type"] == "variable_assignment"
    assert classify_statescript_payload("512 1024")["event_type"] == "dio_state"
    assert classify_statescript_payload("LOCKOUT 1")["event_type"] == "message"


def test_probe_writes_required_outputs(tmp_path):
    nwb = tmp_path / "sample.nwb"
    output_dir = tmp_path / "probe"
    _write_minimal_dandi_nwb(nwb)

    meta = probe_dandi000115_behavior(nwb, output_dir)

    assert meta["n_position_series"] == 1
    assert meta["n_behavior_event_channels"] == 2
    assert meta["n_statescript_sources"] == 1
    assert meta["n_statescript_event_rows"] == 6

    expected = {
        "Table_DANDI000115_Position_Series_Probe.csv",
        "Table_DANDI000115_Behavioral_Events_Probe.csv",
        "Table_DANDI000115_StateScript_Attr_Summary.csv",
        "Table_DANDI000115_StateScript_Event_Probe.csv",
        "Table_DANDI000115_Epochs_Probe.csv",
        "Table_DANDI000115_Time_Alignment_Candidates.csv",
        "dandi000115_behavior_probe_meta.json",
        "DANDI000115_Behavior_Probe_Report.md",
    }

    assert expected.issubset({p.name for p in output_dir.iterdir()})
    assert (output_dir / "statescript_text" / "statescript_r1.txt").exists()


def test_probe_handles_nan_position_timestamps(tmp_path):
    nwb = tmp_path / "sample.nwb"
    output_dir = tmp_path / "probe"
    _write_minimal_dandi_nwb(nwb)

    probe_dandi000115_behavior(nwb, output_dir)

    position = pd.read_csv(output_dir / "Table_DANDI000115_Position_Series_Probe.csv")
    row = position.iloc[0]

    assert int(row["n_samples_raw"]) == 4
    assert int(row["n_samples_finite"]) == 3
    assert int(row["n_nan_timestamps"]) == 1
    assert float(row["time_min"]) == 2000.0
    assert float(row["time_max"]) == 2002.0
    assert float(row["x_min"]) == 0.5
    assert float(row["x_max"]) == 1.5


def test_probe_extracts_statescript_events_and_alignment(tmp_path):
    nwb = tmp_path / "sample.nwb"
    output_dir = tmp_path / "probe"
    _write_minimal_dandi_nwb(nwb)

    meta = probe_dandi000115_behavior(nwb, output_dir)

    events = pd.read_csv(output_dir / "Table_DANDI000115_StateScript_Event_Probe.csv")
    assert set(events["event_type"]) == {
        "well_transition",
        "variable_assignment",
        "dio_state",
        "message",
    }

    align = pd.read_csv(output_dir / "Table_DANDI000115_Time_Alignment_Candidates.csv")
    assert not align.empty
    assert "overlap_fraction_of_statescript" in align.columns
    assert meta["best_alignment"] is not None
