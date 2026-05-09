from __future__ import annotations

import numpy as np

from vte.lab_adapters.redish_lra_2024.extract_lra_candidates import (
    field_classes,
    split_session_vectors,
    value_to_text,
)


def test_field_classes_detect_lra_vte_choice_and_rule_fields():
    assert "idphi" in field_classes("IdPhiData.IdPhi")
    assert "idphi" in field_classes("IdPhiData.AvgDPhi")
    assert "left_right_choice" in field_classes("LapData.Choice")
    assert "reward_outcome" in field_classes("LapData.Correct")
    assert "rule_contingency" in field_classes("SessionData.Contingency")
    assert "switch_changepoint" in field_classes("ChangePointBehavAll.ChangePoint")


def test_split_session_vectors_handles_object_array_of_lap_vectors():
    raw = np.array([np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0])], dtype=object)
    assert split_session_vectors(raw) == [[1.0, 2.0, 3.0], [4.0, 5.0]]


def test_split_session_vectors_handles_numeric_session_by_lap_matrix():
    raw = np.array([[1, 2, 3], [4, 5, 6]])
    assert split_session_vectors(raw) == [[1, 2, 3], [4, 5, 6]]


def test_value_to_text_keeps_scalar_and_short_vectors_readable():
    assert value_to_text(np.float64(2.5)) == "2.5"
    assert value_to_text(np.array([1, 2, 3])) == "[1, 2, 3]"
