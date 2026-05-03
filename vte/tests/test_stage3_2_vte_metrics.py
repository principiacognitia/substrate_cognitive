import numpy as np
import pandas as pd
import pytest

from vte.core.metrics import (
    VTEThresholdConfig,
    angular_difference_radians,
    compute_raw_idphi,
    compute_reorientation_count,
    compute_trial_vte_metrics,
)


def _trace_row(
    *,
    trial: int,
    tick: int,
    heading: float,
    at_choice_point: bool = True,
):
    return {
        "run_id": "r0",
        "seed": 42,
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


def test_angular_difference_wraps_at_pi_boundary():
    values = [np.pi - 0.1, -np.pi + 0.1]

    diffs = angular_difference_radians(values)

    assert diffs.shape == (1,)
    assert diffs[0] == pytest.approx(0.2)


def test_compute_raw_idphi_sums_absolute_wrapped_heading_changes():
    headings = [0.0, 0.5, -0.5, 0.25]

    raw_idphi = compute_raw_idphi(headings)

    assert raw_idphi == pytest.approx(2.25)


def test_compute_reorientation_count_counts_sign_reversals():
    headings = [0.0, 0.5, -0.5, 0.25]

    count = compute_reorientation_count(headings)

    assert count == 2


def test_compute_trial_vte_metrics_returns_trial_rows():
    rows = [
        _trace_row(trial=1, tick=0, heading=0.0),
        _trace_row(trial=1, tick=1, heading=0.5),
        _trace_row(trial=1, tick=2, heading=-0.5),
        _trace_row(trial=2, tick=0, heading=0.0),
        _trace_row(trial=2, tick=1, heading=0.0),
        _trace_row(trial=2, tick=2, heading=0.0),
    ]

    metrics = compute_trial_vte_metrics(pd.DataFrame(rows))

    assert len(metrics) == 2
    assert set(metrics["trial"]) == {1, 2}
    assert "raw_idphi" in metrics.columns
    assert "log_idphi" in metrics.columns
    assert "z_idphi" in metrics.columns
    assert "vte_binary" in metrics.columns


def test_compute_trial_vte_metrics_uses_choice_point_rows_only():
    rows = [
        _trace_row(trial=1, tick=0, heading=0.0, at_choice_point=False),
        _trace_row(trial=1, tick=1, heading=3.0, at_choice_point=False),
        _trace_row(trial=1, tick=2, heading=0.0, at_choice_point=True),
        _trace_row(trial=1, tick=3, heading=0.5, at_choice_point=True),
    ]

    metrics = compute_trial_vte_metrics(pd.DataFrame(rows))

    assert metrics.loc[0, "choice_point_duration"] == 2
    assert metrics.loc[0, "raw_idphi"] == pytest.approx(0.5)


def test_compute_trial_vte_metrics_rejects_missing_required_columns():
    df = pd.DataFrame([{"run_id": "r0"}])

    with pytest.raises(ValueError, match="Missing required columns"):
        compute_trial_vte_metrics(df)


def test_vte_binary_can_be_thresholded_without_stage3_state():
    rows = [
        _trace_row(trial=1, tick=0, heading=0.0),
        _trace_row(trial=1, tick=1, heading=1.0),
        _trace_row(trial=1, tick=2, heading=-1.0),
        _trace_row(trial=1, tick=3, heading=1.0),
        _trace_row(trial=2, tick=0, heading=0.0),
        _trace_row(trial=2, tick=1, heading=0.0),
        _trace_row(trial=2, tick=2, heading=0.0),
    ]

    metrics = compute_trial_vte_metrics(
        pd.DataFrame(rows),
        threshold_config=VTEThresholdConfig(
            z_idphi_threshold=0.0,
            min_pause_ticks=2,
            min_reorientation_count=1,
        ),
    )

    high_idphi_trial = metrics.loc[metrics["trial"] == 1].iloc[0]
    flat_trial = metrics.loc[metrics["trial"] == 2].iloc[0]

    assert high_idphi_trial["vte_binary"] == 1
    assert flat_trial["vte_binary"] == 0

def test_compute_trial_vte_metrics_preserves_optional_trace_metadata():
    rows = [
        _trace_row(trial=1, tick=0, heading=0.0),
        _trace_row(trial=1, tick=1, heading=0.5),
        _trace_row(trial=2, tick=0, heading=0.0),
        _trace_row(trial=2, tick=1, heading=0.0),
    ]

    for row in rows:
        row.update(
            {
                "protocol": "stage3_steps",
                "condition": "R1_T2",
                "ablation": "full",
                "pose_source": "synthetic_from_stage3_steps",
                "target_path": "open",
            }
        )

    metrics = compute_trial_vte_metrics(pd.DataFrame(rows))

    assert set(metrics["protocol"]) == {"stage3_steps"}
    assert set(metrics["condition"]) == {"R1_T2"}
    assert set(metrics["ablation"]) == {"full"}
    assert set(metrics["pose_source"]) == {"synthetic_from_stage3_steps"}
    assert set(metrics["target_path"]) == {"open"}