"""Patch 21: Smoke tests for VTE visualization layer."""
from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
from vte.visualization.regime_selector import select_exploit_traces, select_explore_traces, write_selected_examples
from vte.visualization.render_trial import render_static_trial
from vte.visualization.config import load_regime_config


def test_regime_selector_exploit():
    """Verify exploit regime selects low-deliberation trials."""
    metrics = pd.DataFrame({
        "run_id": ["r1"]*4, "seed": [1]*4, "trial": [1,2,3,4],
        "vte_binary": [0,0,1,0],
        "z_idphi": [0.2, -0.3, 1.5, 0.1],
        "pause_ticks": [2, 1, 8, 2],
        "reorientation_count": [0, 1, 3, 0],
        "raw_idphi": [1.0, 1.2, 5.0, 0.9],
        "total_reward": [1.0, 0.5, 0.0, 1.0],
        "committed_path": ["open", "covered", "open", "open"]
    })
    traces = pd.DataFrame({
        "run_id": ["r1"]*8, "seed": [1]*8, "trial": [1,1,2,2,3,3,4,4],
        "tick": [1,2]*4, "x": [0.0]*8, "y": [0.0]*8, "heading": [0.0]*8,
        "at_choice_point": [True]*8
    })
    config = load_regime_config("vte/visualization/config_regime_selection.yaml")
    
    result = select_exploit_traces(metrics, traces, config)
    assert not result.empty
    # Trial 3 excluded due to high z_idphi, pause, reorient
    assert set(result["trial"]) <= {1, 2, 4}


def test_render_static_output_exists():
    """Verify static renderer produces output file."""
    trace = pd.DataFrame({
        "tick": [1,2,3], "x": [0.0, 0.0, -0.5], "y": [-1.0, 0.0, 0.5],
        "heading": [-0.78, -0.78, -1.2], "at_choice_point": [False, True, False]
    })
    metrics = pd.Series({
        "run_id": "test", "seed": 1, "trial": 1,
        "raw_idphi": 2.1, "z_idphi": 0.3, "vte_binary": 0,
        "pause_ticks": 2, "reorientation_count": 0, "committed_path": "open",
        "total_reward": 1.0
    })
    config = load_regime_config("vte/visualization/config_regime_selection.yaml")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output = Path(tmpdir) / "test.png"
        render_static_trial(trace, metrics, output, config, 
                           pose_source="synthetic_from_stage3_steps")
        assert output.exists()
        assert output.stat().st_size > 1000