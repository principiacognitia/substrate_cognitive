"""Patch 21B: Regime-based trial selection for VTE visualization."""
from __future__ import annotations
import pandas as pd
from pathlib import Path
import yaml
from vte.visualization.config import load_regime_config


def load_regime_config(config_path: str | Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def select_exploit_traces(
    metrics_df: pd.DataFrame,
    trace_df: pd.DataFrame,
    config: dict,
    source: str = "synthetic"
) -> pd.DataFrame:
    """Select low-deliberation trials for exploit regime visualization."""
    cfg = config["regimes"]["exploit"]
    mask = (
        (metrics_df["vte_binary"] == cfg["vte_binary"]) &
        (metrics_df["z_idphi"].abs() <= cfg["z_idphi_abs_max"]) &
        (metrics_df["pause_ticks"] <= cfg["pause_ticks_max"]) &
        (metrics_df["reorientation_count"] <= cfg["reorientation_count_max"])
    )
    selected = metrics_df[mask].nlargest(
        config["selection"]["max_examples_per_regime"],
        "reward" if config["selection"]["prefer_high_reward"] else "raw_idphi"
    )
    return selected.merge(trace_df, on=["run_id", "seed", "trial"], how="inner")


def select_explore_traces(
    metrics_df: pd.DataFrame,
    trace_df: pd.DataFrame,
    config: dict,
    source: str = "synthetic"
) -> pd.DataFrame:
    """Select high-deliberation trials for explore regime visualization."""
    cfg = config["regimes"]["explore"]
    mask = (
        (metrics_df["vte_binary"] == cfg["vte_binary"]) |
        (
            (metrics_df["z_idphi"].abs() >= cfg["z_idphi_abs_min"]) &
            (metrics_df["pause_ticks"] >= cfg["pause_ticks_min"]) &
            (metrics_df["reorientation_count"] >= cfg["reorientation_count_min"])
        )
    )
    selected = metrics_df[mask].nlargest(
        config["selection"]["max_examples_per_regime"],
        "reward" if config["selection"]["prefer_high_reward"] else "raw_idphi"
    )
    return selected.merge(trace_df, on=["run_id", "seed", "trial"], how="inner")


def write_selected_examples(
    exploit_df: pd.DataFrame,
    explore_df: pd.DataFrame,
    output_path: Path
) -> None:
    """Export selection metadata for reproducibility."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    exploit_df["example_type"] = "clean_non_vte"
    explore_df["example_type"] = "top_vte"
    combined = pd.concat([exploit_df, explore_df], ignore_index=True)
    combined[["example_type", "run_id", "seed", "trial", "committed_path",
              "raw_idphi", "z_idphi", "pause_ticks", "reorientation_count",
              "vte_binary", "pose_source"]].to_csv(output_path, index=False)