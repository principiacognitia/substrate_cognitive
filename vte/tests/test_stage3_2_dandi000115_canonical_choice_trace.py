from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from vte.core.wrapper import read_trace_csv
from vte.lab_adapters.dandi_000115.convert_to_canonical_choice_trace import (
    convert_dandi000115_to_canonical_choice_trace,
)
from vte.lab_adapters.dandi_000115.nwb_behavior_probe import probe_dandi000115_behavior


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
        series = pos.create_group("series_1")
        series.attrs["description"] = "xloc, yloc, xloc2, yloc2"

        n = 20
        x = np.arange(n, dtype=np.int16)
        y = np.zeros(n, dtype=np.int16)
        x2 = x + 1
        y2 = np.zeros(n, dtype=np.int16)

        data = np.column_stack([x, y, x2, y2]).astype(np.int16)
        data_ds = series.create_dataset("data", data=data)
        data_ds.attrs["conversion"] = 1.0
        data_ds.attrs["unit"] = "meters"
        series.create_dataset("timestamps", data=np.arange(2000.0, 2020.0, 1.0))

        events = behavior.create_group("behavioral_events")

        arm1beam = events.create_group("arm1beam")
        arm1beam.create_dataset("data", data=np.array([0, 1, 0], dtype=np.int8))
        arm1beam.create_dataset("timestamps", data=np.array([2004.0, 2005.0, 2006.0]))

        arm1pump = events.create_group("arm1pump")
        arm1pump.create_dataset("data", data=np.array([0, 1, 0], dtype=np.int8))
        arm1pump.create_dataset("timestamps", data=np.array([2005.5, 2006.0, 2006.5]))

        arm2beam = events.create_group("arm2beam")
        arm2beam.create_dataset("data", data=np.array([0, 1, 0], dtype=np.int8))
        arm2beam.create_dataset("timestamps", data=np.array([2010.0, 2011.0, 2012.0]))

        arm2pump = events.create_group("arm2pump")
        arm2pump.create_dataset("data", data=np.array([0, 0, 0], dtype=np.int8))
        arm2pump.create_dataset("timestamps", data=np.array([2011.1, 2011.5, 2012.0]))

        associated = processing.create_group("associated_files")
        statescript = associated.create_group("statescript_r1")
        statescript.attrs["description"] = "Statescript log r1"
        statescript.attrs["task_epochs"] = "2, "
        statescript.attrs[
            "content"
        ] = """# header
100 UP 10
101 waslock = 0
200 DOWN 10
300 LOCKOUT 1
"""


def _prepare_probe(tmp_path: Path) -> tuple[Path, Path]:
    nwb = tmp_path / "sample.nwb"
    probe_dir = tmp_path / "probe"
    _write_minimal_dandi_nwb(nwb)
    probe_dandi000115_behavior(nwb, probe_dir)
    return nwb, probe_dir


def test_convert_dandi_choice_trace_writes_outputs(tmp_path):
    nwb, probe_dir = _prepare_probe(tmp_path)
    output_dir = tmp_path / "canonical"

    meta = convert_dandi000115_to_canonical_choice_trace(
        nwb_path=nwb,
        probe_dir=probe_dir,
        output_dir=output_dir,
        position_series="series_1",
        statescript_source="statescript_r1",
        alignment_offset_s=1900.0,
        pre_event_s=1.0,
        post_event_s=2.0,
    )

    assert meta["n_choice_events"] == 2
    assert meta["n_trace_rows"] > 0
    assert (output_dir / "dandi000115_canonical_choice_trace.csv").exists()
    assert (output_dir / "Table_DANDI000115_Choice_Events.csv").exists()
    assert (output_dir / "dandi000115_canonical_choice_trace_meta.json").exists()
    assert (output_dir / "DANDI000115_Canonical_Choice_Trace_Report.md").exists()


def test_canonical_choice_trace_satisfies_wrapper_schema(tmp_path):
    nwb, probe_dir = _prepare_probe(tmp_path)
    output_dir = tmp_path / "canonical"

    convert_dandi000115_to_canonical_choice_trace(
        nwb_path=nwb,
        probe_dir=probe_dir,
        output_dir=output_dir,
        position_series="series_1",
        statescript_source="statescript_r1",
        alignment_offset_s=1900.0,
        pre_event_s=1.0,
        post_event_s=2.0,
    )

    trace_csv = output_dir / "dandi000115_canonical_choice_trace.csv"
    trace = read_trace_csv(trace_csv)

    required = {
        "run_id",
        "seed",
        "trial",
        "tick",
        "x",
        "y",
        "heading",
        "choice_point_id",
        "at_choice_point",
        "done",
    }
    assert required.issubset(trace.columns)
    assert set(trace["choice_point_id"]) == {"dandi000115_event_centered_choice"}
    assert trace["trial"].nunique() == 2


def test_choice_trace_assigns_reward_from_matching_pump(tmp_path):
    nwb, probe_dir = _prepare_probe(tmp_path)
    output_dir = tmp_path / "canonical"

    convert_dandi000115_to_canonical_choice_trace(
        nwb_path=nwb,
        probe_dir=probe_dir,
        output_dir=output_dir,
        position_series="series_1",
        statescript_source="statescript_r1",
        alignment_offset_s=1900.0,
        pre_event_s=1.0,
        post_event_s=2.0,
        reward_window_after_s=2.0,
    )

    events = pd.read_csv(output_dir / "Table_DANDI000115_Choice_Events.csv")

    arm1 = events.loc[events["committed_path"] == "arm1"].iloc[0]
    arm2 = events.loc[events["committed_path"] == "arm2"].iloc[0]

    assert float(arm1["reward"]) == 1.0
    assert float(arm2["reward"]) == 0.0


def test_choice_trace_heading_is_finite_and_uses_body_vector(tmp_path):
    nwb, probe_dir = _prepare_probe(tmp_path)
    output_dir = tmp_path / "canonical"

    convert_dandi000115_to_canonical_choice_trace(
        nwb_path=nwb,
        probe_dir=probe_dir,
        output_dir=output_dir,
        position_series="series_1",
        statescript_source="statescript_r1",
        alignment_offset_s=1900.0,
        pre_event_s=1.0,
        post_event_s=2.0,
    )

    trace = pd.read_csv(output_dir / "dandi000115_canonical_choice_trace.csv")

    assert np.isfinite(trace["heading"]).all()
    assert np.allclose(trace["heading"], 0.0)


def test_choice_trace_can_limit_events_for_smoke(tmp_path):
    nwb, probe_dir = _prepare_probe(tmp_path)
    output_dir = tmp_path / "canonical"

    meta = convert_dandi000115_to_canonical_choice_trace(
        nwb_path=nwb,
        probe_dir=probe_dir,
        output_dir=output_dir,
        position_series="series_1",
        statescript_source="statescript_r1",
        alignment_offset_s=1900.0,
        pre_event_s=1.0,
        post_event_s=2.0,
        max_events=1,
    )

    events = pd.read_csv(output_dir / "Table_DANDI000115_Choice_Events.csv")
    trace = pd.read_csv(output_dir / "dandi000115_canonical_choice_trace.csv")

    assert meta["n_choice_events"] == 1
    assert len(events) == 1
    assert trace["trial"].nunique() == 1
