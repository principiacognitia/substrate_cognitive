from pathlib import Path

import pandas as pd

from vte.analysis.run_stage3_2_vte_batch import (
    SELECTED_EXAMPLE_COLUMNS,
    _write_selected_examples,
)


def _metric_row(
    *,
    run_id="run_a",
    seed=42,
    trial=1,
    committed_path="open",
    raw_idphi=1.0,
    z_idphi=0.0,
    pause_ticks=4,
    reorientation_count=1,
    vte_binary=0,
):
    return {
        "run_id": run_id,
        "protocol": "stage3_steps",
        "condition": "R1_T2",
        "ablation": "full",
        "pose_source": "synthetic_from_stage3_steps",
        "seed": seed,
        "trial": trial,
        "choice_point_id": "junction",
        "choice_point_duration": pause_ticks,
        "pause_ticks": pause_ticks,
        "raw_idphi": raw_idphi,
        "log_idphi": 0.69,
        "z_idphi": z_idphi,
        "reorientation_count": reorientation_count,
        "vte_binary": vte_binary,
        "committed_path": committed_path,
        "terminal_action": "commit",
        "terminal_reward": 1.0,
        "total_reward": 1.0,
        "done_observed": True,
        "n_trace_rows": pause_ticks,
    }


def test_selected_examples_schema_is_stable():
    assert SELECTED_EXAMPLE_COLUMNS == [
        "example_type",
        "run_id",
        "seed",
        "trial",
        "committed_path",
        "raw_idphi",
        "z_idphi",
        "pause_ticks",
        "reorientation_count",
        "vte_binary",
        "trace_csv",
        "recommended_output_name",
    ]


def test_batch_writes_selected_examples_table(tmp_path):
    output_dir = tmp_path / "batch"
    traces_dir = output_dir / "traces"
    traces_dir.mkdir(parents=True)

    trace_csv = traces_dir / "run_a_vte_trace.csv"
    trace_csv.write_text("run_id,seed,trial\n", encoding="utf-8")

    metrics = pd.DataFrame(
        [
            _metric_row(trial=1, committed_path="open", raw_idphi=10.0, z_idphi=3.0, pause_ticks=6, reorientation_count=4, vte_binary=1),
            _metric_row(trial=2, committed_path="open", raw_idphi=0.0, z_idphi=-2.0, pause_ticks=3, reorientation_count=0, vte_binary=0),
            _metric_row(trial=3, committed_path="open", raw_idphi=1.0, z_idphi=-1.0, pause_ticks=6, reorientation_count=0, vte_binary=0),
        ]
    )

    output_csv = output_dir / "Table_3_2_VTE_Selected_Examples.csv"
    n_rows = _write_selected_examples(
        metrics_df=metrics,
        batch_runs=[{"run_label": "run_a", "vte_trace_csv": str(trace_csv)}],
        output_csv=output_csv,
        output_dir=output_dir,
        max_per_type=1,
    )

    assert n_rows == 3
    assert output_csv.exists()

    selected = pd.read_csv(output_csv)
    assert list(selected.columns) == SELECTED_EXAMPLE_COLUMNS
    assert set(selected["example_type"]) == {
        "top_vte",
        "clean_non_vte",
        "matched_pause_control",
    }
    assert selected["trace_csv"].str.endswith("run_a_vte_trace.csv").all()
    assert not Path(selected["trace_csv"].iloc[0]).is_absolute()


def test_matched_pause_control_is_non_vte_and_pause_matched(tmp_path):
    output_dir = tmp_path / "batch"
    traces_dir = output_dir / "traces"
    traces_dir.mkdir(parents=True)

    trace_csv = traces_dir / "run_a_vte_trace.csv"
    trace_csv.write_text("run_id,seed,trial\n", encoding="utf-8")

    metrics = pd.DataFrame(
        [
            _metric_row(trial=1, committed_path="open", raw_idphi=10.0, z_idphi=3.0, pause_ticks=6, reorientation_count=4, vte_binary=1),
            _metric_row(trial=2, committed_path="covered", raw_idphi=0.5, z_idphi=-4.0, pause_ticks=6, reorientation_count=0, vte_binary=0),
            _metric_row(trial=3, committed_path="open", raw_idphi=1.0, z_idphi=-1.0, pause_ticks=6, reorientation_count=0, vte_binary=0),
            _metric_row(trial=4, committed_path="open", raw_idphi=2.0, z_idphi=-2.0, pause_ticks=9, reorientation_count=0, vte_binary=0),
        ]
    )

    output_csv = output_dir / "Table_3_2_VTE_Selected_Examples.csv"
    _write_selected_examples(
        metrics_df=metrics,
        batch_runs=[{"run_label": "run_a", "vte_trace_csv": str(trace_csv)}],
        output_csv=output_csv,
        output_dir=output_dir,
        max_per_type=1,
    )

    selected = pd.read_csv(output_csv)
    matched = selected.loc[selected["example_type"] == "matched_pause_control"].iloc[0]

    assert matched["vte_binary"] == 0
    assert matched["committed_path"] == "open"
    assert matched["pause_ticks"] == 6
    assert matched["recommended_output_name"].startswith("matched_pause_control_run_a_")
