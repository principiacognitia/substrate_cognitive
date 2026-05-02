"""File-level VTE wrapper.

The wrapper is deliberately separated from the toy model. It reads already
externalized trace rows and writes derived VTE metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from vte.core.metrics import (
    DEFAULT_GROUP_COLUMNS,
    VTEThresholdConfig,
    compute_trial_vte_metrics,
)
from vte.core.schema import REQUIRED_TRACE_COLUMNS


@dataclass(frozen=True)
class VTEWrapperResult:
    input_path: Path
    output_path: Path
    n_trace_rows: int
    n_trial_rows: int


def read_trace_csv(input_path: str | Path) -> pd.DataFrame:
    path = Path(input_path)

    if not path.exists():
        raise FileNotFoundError(f"Trace input not found: {path}")

    df = pd.read_csv(path)

    missing = [col for col in REQUIRED_TRACE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Trace input is missing required columns: {missing}")

    return df


def run_vte_wrapper(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    output_name: str = "vte_trial_metrics.csv",
    threshold_config: VTEThresholdConfig = VTEThresholdConfig(),
) -> VTEWrapperResult:
    """Read trace CSV, compute trial-level VTE metrics, write CSV."""

    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trace_df = read_trace_csv(input_path)

    metrics_df = compute_trial_vte_metrics(
        trace_df,
        group_columns=DEFAULT_GROUP_COLUMNS,
        threshold_config=threshold_config,
    )

    output_path = output_dir / output_name
    metrics_df.to_csv(output_path, index=False)

    return VTEWrapperResult(
        input_path=input_path,
        output_path=output_path,
        n_trace_rows=int(len(trace_df)),
        n_trial_rows=int(len(metrics_df)),
    )