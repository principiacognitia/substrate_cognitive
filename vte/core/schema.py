"""Schema helpers for Stage 3.2 VTE trace inputs.

This module intentionally has no dependency on stage3 internals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

REQUIRED_TRACE_COLUMNS = (
    "run_id",
    "seed",
    "trial",
    "tick",
    "x",
    "y",
    "heading",
    "choice_point_id",
    "at_choice_point",
    "action",
    "committed_path",
    "reward",
    "done",
)

OPTIONAL_TRACE_COLUMNS = (
    "protocol",
    "condition",
    "ablation",
    "trial_phase",
    "pose_source",
    "event_type",
    "event_trial",
    "target_path",
)

@dataclass(frozen=True)
class TraceSchemaValidation:
    ok: bool
    missing_columns: tuple[str, ...]
    extra_columns: tuple[str, ...]

def validate_trace_columns(columns: Iterable[str]) -> TraceSchemaValidation:
    """Validate that a raw VTE trace has the required columns.

    Extra columns are allowed because external adapters may preserve metadata.
    """

    observed = tuple(columns)
    observed_set = set(observed)
    required_set = set(REQUIRED_TRACE_COLUMNS)
    allowed_set = required_set | set(OPTIONAL_TRACE_COLUMNS)

    missing = tuple(col for col in REQUIRED_TRACE_COLUMNS if col not in observed_set)
    extra = tuple(col for col in observed if col not in allowed_set)

    return TraceSchemaValidation(
        ok=not missing,
        missing_columns=missing,
        extra_columns=extra,
    )