from __future__ import annotations

import pandas as pd

from vte.lab_adapters.redish_lra_2024.filter_lra_choice_direction_same_trial import (
    _add_next_choice_entry,
    _classify_same_trial,
)


def test_add_next_choice_entry_groups_by_session():
    df = pd.DataFrame(
        {
            "subject_id": ["R1", "R1", "R1"],
            "session_id": ["S1", "S1", "S2"],
            "trial": [1, 2, 1],
            "choice_point_entry_s_audit": [10.0, 20.0, 100.0],
            "choice_point_exit_s_audit": [11.0, 21.0, 101.0],
            "primary_event_time_s": [12.0, 22.0, 102.0],
            "primary_event_latency_s": [1.0, 1.0, 1.0],
        }
    )

    out = _add_next_choice_entry(df)

    s1_t1 = out[(out["session_id"] == "S1") & (out["trial"] == 1)].iloc[0]
    s1_t2 = out[(out["session_id"] == "S1") & (out["trial"] == 2)].iloc[0]
    s2_t1 = out[(out["session_id"] == "S2") & (out["trial"] == 1)].iloc[0]

    assert s1_t1["next_choice_point_entry_s"] == 20.0
    assert pd.isna(s1_t2["next_choice_point_entry_s"])
    assert pd.isna(s2_t1["next_choice_point_entry_s"])


def test_classify_same_trial_accepts_event_before_next_entry():
    row = pd.Series(
        {
            "choice_point_exit_s_audit": 10.0,
            "primary_event_time_s": 12.0,
            "next_choice_point_entry_s": 20.0,
        }
    )

    status, latency = _classify_same_trial(
        row,
        max_primary_latency_s=30.0,
        next_entry_slack_s=0.25,
    )

    assert status == "same_trial_event"
    assert latency == 2.0


def test_classify_same_trial_rejects_event_after_next_entry():
    row = pd.Series(
        {
            "choice_point_exit_s_audit": 10.0,
            "primary_event_time_s": 25.0,
            "next_choice_point_entry_s": 20.0,
        }
    )

    status, latency = _classify_same_trial(
        row,
        max_primary_latency_s=30.0,
        next_entry_slack_s=0.25,
    )

    assert status == "after_next_choice_entry"
    assert latency == 15.0


def test_classify_same_trial_rejects_event_above_latency_cap():
    row = pd.Series(
        {
            "choice_point_exit_s_audit": 10.0,
            "primary_event_time_s": 50.0,
            "next_choice_point_entry_s": 100.0,
        }
    )

    status, latency = _classify_same_trial(
        row,
        max_primary_latency_s=30.0,
        next_entry_slack_s=0.25,
    )

    assert status == "exceeds_latency_cap"
    assert latency == 40.0
