"""Patch 21B: Regime-based trial selection for VTE visualization."""
from __future__ import annotations
import pandas as pd
from pathlib import Path
import yaml
from vte.visualization.config import load_regime_config


def load_regime_config(config_path: str | Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# vte/visualization/regime_selector.py

def _prepare_trace(trace_df: pd.DataFrame, metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Удаляет из trace_df столбцы, уже существующие в metrics_df, чтобы избежать суффиксов _x/_y."""
    overlap = [c for c in trace_df.columns if c in metrics_df.columns and c not in ("run_id", "seed", "trial")]
    return trace_df.drop(columns=overlap)


def select_exploit_traces(
    metrics_df: pd.DataFrame,
    trace_df: pd.DataFrame,
    config: dict,
    source: str = "synthetic"
) -> pd.DataFrame:
    cfg = config["regimes"]["exploit"]
    mask = (
        (metrics_df["vte_binary"] == cfg["vte_binary"]) &
        (metrics_df["z_idphi"].abs() <= cfg["z_idphi_abs_max"]) &
        (metrics_df["pause_ticks"] <= cfg["pause_ticks_max"]) &
        (metrics_df["reorientation_count"] <= cfg["reorientation_count_max"])
    )
    
    prefer_reward = config["selection"].get("prefer_high_reward", True)
    primary_col = "total_reward" if prefer_reward else "raw_idphi"
    
    # Fallback chain: primary -> alternative -> index
    if primary_col in metrics_df.columns:
        sort_col = primary_col
    elif "raw_idphi" in metrics_df.columns:
        sort_col = "raw_idphi"
    elif "total_reward" in metrics_df.columns:
        sort_col = "total_reward"
    else:
        sort_col = None

    masked = metrics_df[mask]
    if sort_col is not None and not masked.empty:
        selected = masked.nlargest(
            config["selection"]["max_examples_per_regime"],
            sort_col
        )
    else:
        # Fallback: return first N without sorting
        selected = masked.head(config["selection"]["max_examples_per_regime"])
        
    return selected.merge(_prepare_trace(trace_df, metrics_df), on=["run_id", "seed", "trial"], how="inner")


def select_explore_traces(
    metrics_df: pd.DataFrame,
    trace_df: pd.DataFrame,
    config: dict,
    source: str = "synthetic"
) -> pd.DataFrame:
    cfg = config["regimes"]["explore"]
    mask = (
        (metrics_df["vte_binary"] == cfg["vte_binary"]) |
        (
            (metrics_df["z_idphi"].abs() >= cfg["z_idphi_abs_min"]) &
            (metrics_df["pause_ticks"] >= cfg["pause_ticks_min"]) &
            (metrics_df["reorientation_count"] >= cfg["reorientation_count_min"])
        )
    )
    
    prefer_reward = config["selection"].get("prefer_high_reward", True)
    primary_col = "total_reward" if prefer_reward else "raw_idphi"
    
    # Fallback chain: primary -> alternative -> index
    if primary_col in metrics_df.columns:
        sort_col = primary_col
    elif "raw_idphi" in metrics_df.columns:
        sort_col = "raw_idphi"
    elif "total_reward" in metrics_df.columns:
        sort_col = "total_reward"
    else:
        sort_col = None

    masked = metrics_df[mask]
    if sort_col is not None and not masked.empty:
        selected = masked.nlargest(
            config["selection"]["max_examples_per_regime"],
            sort_col
        )
    else:
        # Fallback: return first N without sorting
        selected = masked.head(config["selection"]["max_examples_per_regime"])
        
    return selected.merge(_prepare_trace(trace_df, metrics_df), on=["run_id", "seed", "trial"], how="inner")


def write_selected_examples(
    exploit_df: pd.DataFrame,
    explore_df: pd.DataFrame,
    output_path: Path
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    exploit_df["example_type"] = "clean_non_vte"
    explore_df["example_type"] = "top_vte"
    combined = pd.concat([exploit_df, explore_df], ignore_index=True)
    
    # Защитный фильтр: выбираем только те столбцы, которые реально присутствуют после merge
    target_cols = [
        "example_type", "run_id", "seed", "trial", "committed_path",
        "raw_idphi", "z_idphi", "pause_ticks", "reorientation_count",
        "vte_binary", "pose_source"
    ]
    present_cols = [c for c in target_cols if c in combined.columns]
    missing = set(target_cols) - set(present_cols)
    if missing:
        import sys
        print(f"[WARNING] Columns missing after merge (will be skipped): {missing}", file=sys.stderr)
        
    combined[present_cols].to_csv(output_path, index=False)