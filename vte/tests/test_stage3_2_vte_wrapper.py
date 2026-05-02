import json

import pandas as pd
import pytest

from vte.core.wrapper import read_trace_csv, run_vte_wrapper


def _row(trial: int, tick: int, heading: float, at_choice_point: bool = True):
    return {
        "run_id": "wrapper_test",
        "seed": 1,
        "trial": trial,
        "tick": tick,
        "x": float(tick),
        "y": 0.0,
        "heading": heading,
        "choice_point_id": "cp0",
        "at_choice_point": at_choice_point,
        "action": "observe",
        "committed_path": "open",
        "reward": 0.0,
        "done": False,
    }


def test_read_trace_csv_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_trace_csv(tmp_path / "missing.csv")


def test_read_trace_csv_rejects_missing_columns(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame([{"run_id": "r0"}]).to_csv(path, index=False)

    with pytest.raises(ValueError, match="missing required columns"):
        read_trace_csv(path)


def test_run_vte_wrapper_writes_trial_metrics(tmp_path):
    input_csv = tmp_path / "trace.csv"
    output_dir = tmp_path / "out"

    rows = [
        _row(trial=1, tick=0, heading=0.0),
        _row(trial=1, tick=1, heading=1.0),
        _row(trial=1, tick=2, heading=-1.0),
        _row(trial=2, tick=0, heading=0.0),
        _row(trial=2, tick=1, heading=0.0),
    ]
    pd.DataFrame(rows).to_csv(input_csv, index=False)

    result = run_vte_wrapper(input_csv, output_dir)

    assert result.n_trace_rows == 5
    assert result.n_trial_rows == 2
    assert result.output_path.exists()

    out = pd.read_csv(result.output_path)

    assert len(out) == 2
    assert "raw_idphi" in out.columns
    assert "z_idphi" in out.columns
    assert "vte_binary" in out.columns


def test_cli_writes_metrics_and_metadata(tmp_path):
    from vte.analysis.run_stage3_2_vte import main
    import sys

    input_csv = tmp_path / "trace.csv"
    output_dir = tmp_path / "cli_out"

    pd.DataFrame(
        [
            _row(trial=1, tick=0, heading=0.0),
            _row(trial=1, tick=1, heading=1.0),
            _row(trial=1, tick=2, heading=-1.0),
        ]
    ).to_csv(input_csv, index=False)

    old_argv = sys.argv
    try:
        sys.argv = [
            "run_stage3_2_vte",
            "--input-csv",
            str(input_csv),
            "--output-dir",
            str(output_dir),
        ]
        main()
    finally:
        sys.argv = old_argv

    metrics_path = output_dir / "vte_trial_metrics.csv"
    meta_path = output_dir / "vte_wrapper_meta.json"

    assert metrics_path.exists()
    assert meta_path.exists()

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["n_trace_rows"] == 3
    assert meta["n_trial_rows"] == 1