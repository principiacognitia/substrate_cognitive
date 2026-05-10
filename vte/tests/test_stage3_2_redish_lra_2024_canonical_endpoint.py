from __future__ import annotations

import numpy as np
import pandas as pd

from vte.lab_adapters.redish_lra_2024.extract_canonical_choice_endpoint import (
    _add_group_zscore,
    _as_session_lap_matrix,
    _outcome_from_correct,
    _vte_binary,
)


def test_as_session_lap_matrix_transposes_lap_by_session_layout():
    source = np.arange(10, dtype=float).reshape(5, 2)

    out = _as_session_lap_matrix(source, n_sessions=2, max_laps=5)

    assert out.shape == (2, 5)
    assert out[0, 0] == 0.0
    assert out[1, 0] == 1.0
    assert out[0, 4] == 8.0
    assert out[1, 4] == 9.0


def test_as_session_lap_matrix_keeps_session_by_lap_layout():
    source = np.arange(10, dtype=float).reshape(2, 5)

    out = _as_session_lap_matrix(source, n_sessions=2, max_laps=5)

    assert out.shape == (2, 5)
    assert out[0, 0] == 0.0
    assert out[0, 4] == 4.0
    assert out[1, 0] == 5.0
    assert out[1, 4] == 9.0


def test_outcome_and_vte_normalization():
    assert _outcome_from_correct(1.0) == ("correct", 1.0)
    assert _outcome_from_correct(0.0) == ("error", 0.0)

    missing_outcome, missing_reward = _outcome_from_correct(float("nan"))
    assert missing_outcome == ""
    assert np.isnan(missing_reward)

    assert _vte_binary(1.0) == 1.0
    assert _vte_binary(0.0) == 0.0
    assert _vte_binary(0.49) == 0.0
    assert _vte_binary(0.5) == 1.0


def test_add_group_zscore_is_session_local():
    df = pd.DataFrame(
        {
            "cohort": ["a", "a", "a", "a"],
            "session_id": ["s1", "s1", "s2", "s2"],
            "deliberation_proxy": [1.0, 3.0, 10.0, 14.0],
        }
    )

    _add_group_zscore(
        df,
        value_col="deliberation_proxy",
        group_cols=["cohort", "session_id"],
        out_col="z",
    )

    assert list(df["z"].round(6)) == [-1.0, 1.0, -1.0, 1.0]