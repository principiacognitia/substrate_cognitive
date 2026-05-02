import pandas as pd
import pytest

from vte.adapters.stage3_steps import (
    convert_stage3_steps_csv,
    stage3_steps_to_vte_trace,
)
from vte.core.metrics import compute_trial_vte_metrics
from vte.core.schema import REQUIRED_TRACE_COLUMNS


def _stage3_row(
    *,
    seed=42,
    trial=1,
    tick=1,
    at_junction=True,
    action=0,
    reward=0.0,
    candidate_path="open",
    committed_path="",
):
    return {
        "seed": seed,
        "condition_id": "R1_T2",
        "ablation": "full",
        "trial": trial,
        "tick": tick,
        "node_id": "junction" if at_junction else "start",
        "at_junction": at_junction,
        "deliberation_state": "",
        "candidate_path": candidate_path,
        "committed_path": committed_path,
        "action": action,
        "reward": reward,
    }


def test_stage3_adapter_outputs_required_vte_trace_columns():
    steps = pd.DataFrame(
        [
            _stage3_row(trial=1, tick=1, candidate_path="open"),
            _stage3_row(trial=1, tick=2, candidate_path="covered"),
        ]
    )

    trace = stage3_steps_to_vte_trace(steps, run_id="test_run")

    for col in REQUIRED_TRACE_COLUMNS:
        assert col in trace.columns

    assert set(trace["run_id"]) == {"test_run"}
    assert set(trace["choice_point_id"]) == {"junction"}
    assert trace["pose_source"].iloc[0] == "synthetic_from_stage3_steps"


def test_stage3_adapter_marks_done_on_last_tick_per_trial():
    steps = pd.DataFrame(
        [
            _stage3_row(trial=1, tick=1),
            _stage3_row(trial=1, tick=2),
            _stage3_row(trial=2, tick=1),
        ]
    )

    trace = stage3_steps_to_vte_trace(steps, run_id="test_run")

    done_by_trial = trace.groupby("trial")["done"].sum().to_dict()

    assert done_by_trial == {1: 1, 2: 1}


def test_stage3_adapter_candidate_switches_produce_idphi_signal():
    steps = pd.DataFrame(
        [
            _stage3_row(trial=1, tick=1, candidate_path="open"),
            _stage3_row(trial=1, tick=2, candidate_path="covered"),
            _stage3_row(trial=1, tick=3, candidate_path="open"),
            _stage3_row(trial=2, tick=1, candidate_path="open"),
            _stage3_row(trial=2, tick=2, candidate_path="open"),
            _stage3_row(trial=2, tick=3, candidate_path="open"),
        ]
    )

    trace = stage3_steps_to_vte_trace(steps, run_id="test_run")
    metrics = compute_trial_vte_metrics(trace)

    high = metrics.loc[metrics["trial"] == 1].iloc[0]
    low = metrics.loc[metrics["trial"] == 2].iloc[0]

    assert high["raw_idphi"] > low["raw_idphi"]


def test_stage3_adapter_rejects_missing_required_stage3_columns():
    steps = pd.DataFrame([{"seed": 1}])

    with pytest.raises(ValueError, match="missing required columns"):
        stage3_steps_to_vte_trace(steps, run_id="bad")


def test_convert_stage3_steps_csv_writes_trace_csv(tmp_path):
    input_csv = tmp_path / "steps.csv"
    output_csv = tmp_path / "trace.csv"

    pd.DataFrame(
        [
            _stage3_row(trial=1, tick=1, candidate_path="open"),
            _stage3_row(trial=1, tick=2, candidate_path="covered"),
        ]
    ).to_csv(input_csv, index=False)

    out = convert_stage3_steps_csv(
        input_csv=input_csv,
        output_csv=output_csv,
        run_id="csv_test",
    )

    assert out == output_csv
    assert output_csv.exists()

    trace = pd.read_csv(output_csv)
    assert "heading" in trace.columns
    assert set(trace["run_id"]) == {"csv_test"}