from __future__ import annotations

import numpy as np
import pandas as pd

from vte.lab_adapters.redish_lra_2024.extract_canonical_choice_endpoint_vte_resolved import (
    _apply_vte_resolution,
    _threshold_vector_from_value,
)


def test_threshold_vector_from_scalar():
    vector, source = _threshold_vector_from_value(42.0, n_sessions=3)

    assert source == "scalar"
    assert list(vector) == [42.0, 42.0, 42.0]


def test_threshold_vector_from_session_vector():
    vector, source = _threshold_vector_from_value(np.array([1.0, 2.0, 3.0]), n_sessions=3)

    assert source == "session_vector"
    assert list(vector) == [1.0, 2.0, 3.0]


def test_apply_vte_resolution_keeps_native_label():
    df = pd.DataFrame(
        {
            "lab_vte_binary": [0.0, 1.0],
            "lab_idphi": [100.0, 100.0],
            "lab_vte_threshold": [50.0, 200.0],
        }
    )

    out = _apply_vte_resolution(df)

    assert list(out["vte_binary"]) == [0.0, 1.0]
    assert set(out["vte_binary_source"]) == {"VTELap.ChoicePoint"}


def test_apply_vte_resolution_uses_author_threshold_when_native_missing():
    df = pd.DataFrame(
        {
            "lab_vte_binary": [np.nan, np.nan],
            "lab_idphi": [10.0, 30.0],
            "lab_vte_threshold": [20.0, 20.0],
        }
    )

    out = _apply_vte_resolution(df)

    assert list(out["vte_binary"]) == [0.0, 1.0]
    assert set(out["vte_binary_source"]) == {"IdPhi.ChoicePoint>=VTEThreshold"}