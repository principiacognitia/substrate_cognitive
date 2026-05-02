"""Core VTE metrics for Stage 3.2.

This module is intentionally independent from stage3 internals.
It operates only on externalized trace rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from vte.core.schema import REQUIRED_TRACE_COLUMNS


DEFAULT_GROUP_COLUMNS = ("run_id", "seed", "trial")


@dataclass(frozen=True)
class VTEThresholdConfig:
    """Thresholds for deriving a binary VTE-like label."""

    z_idphi_threshold: float = 1.0
    min_pause_ticks: int = 2
    min_reorientation_count: int = 1


def _require_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def angular_difference_radians(values: Sequence[float]) -> np.ndarray:
    """Return wrapped first differences for heading angles in radians."""

    arr = np.asarray(values, dtype=float)
    if arr.size < 2:
        return np.asarray([], dtype=float)

    raw = np.diff(arr)
    return (raw + np.pi) % (2.0 * np.pi) - np.pi


def compute_raw_idphi(headings: Sequence[float]) -> float:
    """Compute IdPhi-like angular path integral.

    This is the sum of absolute wrapped heading changes.
    """

    diffs = angular_difference_radians(headings)
    if diffs.size == 0:
        return 0.0
    return float(np.sum(np.abs(diffs)))


def compute_reorientation_count(
    headings: Sequence[float],
    min_abs_delta: float = 1e-9,
) -> int:
    """Count sign reversals in non-trivial heading changes."""

    diffs = angular_difference_radians(headings)
    diffs = diffs[np.abs(diffs) > min_abs_delta]

    if diffs.size < 2:
        return 0

    signs = np.sign(diffs)
    reversals = signs[1:] != signs[:-1]
    return int(np.sum(reversals))


def compute_trial_vte_metrics(
    trace_df: pd.DataFrame,
    group_columns: Sequence[str] = DEFAULT_GROUP_COLUMNS,
    threshold_config: VTEThresholdConfig = VTEThresholdConfig(),
) -> pd.DataFrame:
    """Compute trial-level VTE metrics from raw trace rows.

    Metrics are computed only over rows where `at_choice_point` is truthy.
    """

    _require_columns(trace_df, REQUIRED_TRACE_COLUMNS)
    _require_columns(trace_df, group_columns)

    if trace_df.empty:
        return pd.DataFrame(
            columns=[
                *group_columns,
                "choice_point_id",
                "choice_point_duration",
                "pause_ticks",
                "raw_idphi",
                "log_idphi",
                "z_idphi",
                "reorientation_count",
                "vte_binary",
            ]
        )

    rows: list[dict[str, object]] = []

    sort_columns = [*group_columns, "tick"]
    df = trace_df.sort_values(sort_columns).copy()

    for key, g in df.groupby(list(group_columns), sort=False):
        key_tuple = key if isinstance(key, tuple) else (key,)
        key_data = dict(zip(group_columns, key_tuple))

        cp = g[g["at_choice_point"].astype(bool)].copy()

        if cp.empty:
            choice_point_id = None
            headings = []
            duration = 0
        else:
            choice_point_id = cp["choice_point_id"].iloc[0]
            headings = cp["heading"].astype(float).to_numpy()
            duration = int(cp["tick"].max() - cp["tick"].min() + 1)

        raw_idphi = compute_raw_idphi(headings)
        reorientation_count = compute_reorientation_count(headings)

        rows.append(
            {
                **key_data,
                "choice_point_id": choice_point_id,
                "choice_point_duration": duration,
                "pause_ticks": duration,
                "raw_idphi": raw_idphi,
                "log_idphi": float(np.log1p(raw_idphi)),
                "reorientation_count": reorientation_count,
            }
        )

    out = pd.DataFrame(rows)

    if out.empty:
        out["z_idphi"] = pd.Series(dtype=float)
        out["vte_binary"] = pd.Series(dtype=int)
        return out

    # Normalize within run/seed where possible. This avoids leaking condition-level
    # expectations into the metric.
    z_groups = [col for col in ("run_id", "seed") if col in out.columns]

    if z_groups:
        means = out.groupby(z_groups)["log_idphi"].transform("mean")
        stds = out.groupby(z_groups)["log_idphi"].transform("std").replace(0.0, np.nan)
        out["z_idphi"] = ((out["log_idphi"] - means) / stds).fillna(0.0)
    else:
        std = out["log_idphi"].std()
        if std == 0 or np.isnan(std):
            out["z_idphi"] = 0.0
        else:
            out["z_idphi"] = (out["log_idphi"] - out["log_idphi"].mean()) / std

    out["vte_binary"] = (
        (out["z_idphi"] >= threshold_config.z_idphi_threshold)
        & (out["pause_ticks"] >= threshold_config.min_pause_ticks)
        & (out["reorientation_count"] >= threshold_config.min_reorientation_count)
    ).astype(int)

    return out