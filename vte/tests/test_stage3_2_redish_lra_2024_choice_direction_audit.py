from __future__ import annotations

import numpy as np

from vte.lab_adapters.redish_lra_2024.audit_lra_choice_direction_events import (
    _event_code_label,
    _nearest_after,
    _primary_from_reward,
)


def test_event_code_label_normalizes_integer_codes():
    assert _event_code_label(1.0) == "code_1"
    assert _event_code_label(4) == "code_4"
    assert _event_code_label(float("nan")) == ""


def test_nearest_after_matches_first_event_after_choice_exit():
    out = _nearest_after(
        start_time_s=10.0,
        event_times=np.asarray([8.0, 11.5, 15.0]),
        event_codes=np.asarray([1.0, 2.0, 3.0]),
        max_latency_s=10.0,
    )

    assert out["match_status"] == "matched_after_choice_exit"
    assert out["event_time_s"] == 11.5
    assert out["event_code"] == 2.0
    assert out["event_latency_s"] == 1.5


def test_nearest_after_reports_outside_window():
    out = _nearest_after(
        start_time_s=10.0,
        event_times=np.asarray([30.0]),
        event_codes=np.asarray([4.0]),
        max_latency_s=5.0,
    )

    assert out["match_status"] == "nearest_outside_window"
    assert out["event_code"] == 4.0


def test_primary_event_uses_reward_to_select_feeder_or_error_stream():
    feeder = {
        "event_time_s": 12.0,
        "event_code": 1.0,
        "event_latency_s": 2.0,
        "match_status": "matched_after_choice_exit",
    }
    error = {
        "event_time_s": 13.0,
        "event_code": 3.0,
        "event_latency_s": 3.0,
        "match_status": "matched_after_choice_exit",
    }

    correct = _primary_from_reward(1.0, feeder, error)
    assert correct["primary_event_kind"] == "feeder_fired"
    assert correct["primary_event_code_label"] == "code_1"

    wrong = _primary_from_reward(0.0, feeder, error)
    assert wrong["primary_event_kind"] == "error_not_fired"
    assert wrong["primary_event_code_label"] == "code_3"
