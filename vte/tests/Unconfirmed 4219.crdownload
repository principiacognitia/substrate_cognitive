"""Patch 21B: Unit tests for regime selection logic."""
from __future__ import annotations
import pandas as pd
import pytest
from vte.visualization.regime_selector import select_exploit_traces, select_explore_traces
from vte.visualization.config import load_regime_config


@pytest.fixture
def mock_config():
    return load_regime_config("vte/visualization/config_regime_selection.yaml")


@pytest.fixture
def mock_metrics():
    return pd.DataFrame({
        "run_id": ["r1"]*6, "seed": [1]*6, "trial": range(1, 7),
        "vte_binary": [0, 0, 1, 1, 0, 1],
        "z_idphi": [0.2, -0.1, 1.2, -1.5, 0.0, 1.8],
        "pause_ticks": [2, 1, 6, 8, 1, 9],
        "reorientation_count": [0, 0, 3, 4, 0, 2],
        "raw_idphi": [1.0, 1.5, 4.0, 5.0, 0.8, 6.0],
        "total_reward": [1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
        "committed_path": ["open", "covered", "open", "covered", "open", "open"]
    })


@pytest.fixture
def mock_traces(mock_metrics):
    rows = []
    for _, m in mock_metrics.iterrows():
        for tick in range(1, 4):
            rows.append({
                "run_id": m["run_id"], "seed": m["seed"], "trial": m["trial"],
                "tick": tick, "x": float(tick), "y": 0.0, "heading": 0.0,
                "at_choice_point": True
            })
    return pd.DataFrame(rows)


def test_exploit_filters_low_deliberation(mock_config, mock_metrics, mock_traces):
    result = select_exploit_traces(mock_metrics, mock_traces, mock_config)
    assert not result.empty
    assert all(result["vte_binary"] == 0)
    assert all(result["pause_ticks"] <= mock_config["regimes"]["exploit"]["pause_ticks_max"])


def test_explore_filters_high_deliberation(mock_config, mock_metrics, mock_traces):
    result = select_explore_traces(mock_metrics, mock_traces, mock_config)
    assert not result.empty
    # Should contain only vte_binary=1 or high pause/z_idphi/reorient
    mask = (result["vte_binary"] == 1) | \
           (result["pause_ticks"] >= mock_config["regimes"]["explore"]["pause_ticks_min"])
    assert all(mask)


def test_empty_regime_returns_empty(mock_config, mock_traces):
    metrics = pd.DataFrame({
        "run_id": ["r2"], "seed": [2], "trial": [1],
        "vte_binary": [1], "z_idphi": [0.0], "pause_ticks": [0], 
        "reorientation_count": [0], "raw_idphi": [1.0], "total_reward": [1.0],
        "committed_path": ["open"]
    })
    traces = mock_traces[mock_traces["seed"] == 2]
    result = select_exploit_traces(metrics, traces, mock_config)
    assert result.empty


def test_merge_preserves_trace_columns(mock_config, mock_metrics, mock_traces):
    result = select_exploit_traces(mock_metrics, mock_traces, mock_config)
    assert not result.empty
    assert "x" in result.columns
    assert "y" in result.columns
    assert "heading" in result.columns
    assert "tick" in result.columns