"""Adapter from Stage 3 step logs to VTE raw trace schema.

This module reads already externalized Stage 3 CSV logs. It must not import
stage3 internals. Its output is a synthetic pose trace suitable for the VTE
wrapper.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from vte.core.schema import REQUIRED_TRACE_COLUMNS


REQUIRED_STAGE3_STEP_COLUMNS = (
    "seed",
    "trial",
    "tick",
    "at_junction",
    "action",
    "reward",
)

PATH_TO_HEADING = {
    "open": -np.pi / 4.0,
    "covered": np.pi / 4.0,
}


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        if np.isnan(value):
            return False
        return bool(value)
    text = str(value).strip().lower()
    return text in {"true", "1", "yes", "y", "t"}


def _clean_path(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    text = str(value).strip().lower()
    if text in {"open", "covered"}:
        return text
    return ""


def _path_from_action(value: Any) -> str:
    try:
        action = int(value)
    except (TypeError, ValueError):
        return ""

    if action == 0:
        return "open"
    if action == 1:
        return "covered"
    return ""


def _infer_choice_path(row: pd.Series) -> str:
    candidate = _clean_path(row.get("candidate_path", ""))
    if candidate:
        return candidate

    committed = _clean_path(row.get("committed_path", ""))
    if committed:
        return committed

    return _path_from_action(row.get("action", ""))


def _infer_heading(row: pd.Series) -> float:
    """Infer synthetic heading from candidate/committed path.

    This is not biological pose tracking. It is an explicit synthetic projection
    from model trace rows into a pose-like schema.
    """

    path = _infer_choice_path(row)
    return float(PATH_TO_HEADING.get(path, 0.0))


def _infer_y(row: pd.Series) -> float:
    path = _infer_choice_path(row)
    if path == "open":
        return -1.0
    if path == "covered":
        return 1.0
    return 0.0


def validate_stage3_step_columns(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_STAGE3_STEP_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Stage 3 step log is missing required columns: {missing}")


def stage3_steps_to_vte_trace(
    steps_df: pd.DataFrame,
    *,
    run_id: str,
    pose_source: str = "synthetic_from_stage3_steps",
) -> pd.DataFrame:
    """Convert Stage 3 all_steps rows into VTE raw trace rows."""

    validate_stage3_step_columns(steps_df)

    df = steps_df.copy()
    df = df.sort_values(["seed", "trial", "tick"]).reset_index(drop=True)

    at_choice = df["at_junction"].map(_as_bool)

    trace = pd.DataFrame(index=df.index)
    trace["run_id"] = str(run_id)
    trace["seed"] = df["seed"]
    trace["trial"] = df["trial"]
    trace["tick"] = df["tick"]

    # Synthetic pose. The VTE wrapper currently needs x/y/heading, but only
    # heading within choice-point rows affects raw_idphi.
    trace["heading"] = df.apply(_infer_heading, axis=1)
    trace["x"] = df["tick"].astype(float)
    trace["y"] = df.apply(_infer_y, axis=1)

    trace["choice_point_id"] = np.where(at_choice, "junction", "")
    trace["at_choice_point"] = at_choice

    trace["action"] = df["action"]
    trace["committed_path"] = (
        df["committed_path"].map(_clean_path)
        if "committed_path" in df.columns
        else df["action"].map(_path_from_action)
    )
    trace["reward"] = df["reward"]

    max_tick = df.groupby(["seed", "trial"])["tick"].transform("max")
    trace["done"] = df["tick"].eq(max_tick)

    # Optional metadata preserved for downstream grouping.
    if "condition_id" in df.columns:
        trace["condition"] = df["condition_id"]
    if "ablation" in df.columns:
        trace["ablation"] = df["ablation"]

    trace["protocol"] = "stage3_steps"
    trace["pose_source"] = pose_source

    # Keep required columns first, then optional metadata.
    required = list(REQUIRED_TRACE_COLUMNS)
    optional = [col for col in trace.columns if col not in required]
    return trace[required + optional]


def convert_stage3_steps_csv(
    input_csv: str | Path,
    output_csv: str | Path,
    *,
    run_id: str | None = None,
    pose_source: str = "synthetic_from_stage3_steps",
) -> Path:
    input_path = Path(input_csv)
    output_path = Path(output_csv)

    if not input_path.exists():
        raise FileNotFoundError(f"Stage 3 step log not found: {input_path}")

    if run_id is None:
        run_id = input_path.stem

    steps_df = pd.read_csv(input_path)
    trace_df = stage3_steps_to_vte_trace(
        steps_df,
        run_id=run_id,
        pose_source=pose_source,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    trace_df.to_csv(output_path, index=False)

    return output_path