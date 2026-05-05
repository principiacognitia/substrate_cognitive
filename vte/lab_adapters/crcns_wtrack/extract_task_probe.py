"""Task/epoch probe for CRCNS Frank-lab W-track MATLAB files.

This Stage 3.2C utility inspects task metadata and aligns it with position
epoch summaries. It is not a trial parser and does not compute VTE metrics.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from vte.lab_adapters.crcns_wtrack.extract_position_probe import (
    PositionCandidate,
    find_position_candidates,
)
from vte.lab_adapters.crcns_wtrack.mat_loader import (
    infer_animal_prefix,
    infer_day_from_filename,
    infer_file_kind,
    list_animal_mat_files,
    read_mat,
)


TASK_EPOCH_PROBE_COLUMNS = [
    "dataset_id",
    "animal_id",
    "day",
    "epoch",
    "task_field_path",
    "task_type",
    "environment",
    "description",
    "task_keys",
    "task_value_type",
    "n_position_samples",
    "time_min",
    "time_max",
    "x_min",
    "x_max",
    "y_min",
    "y_max",
    "source_task_file",
    "source_pos_file",
]

POSITION_EPOCH_SUMMARY_COLUMNS = [
    "dataset_id",
    "animal_id",
    "day",
    "epoch",
    "position_field_path",
    "n_position_samples",
    "time_min",
    "time_max",
    "x_min",
    "x_max",
    "y_min",
    "y_max",
    "has_time",
    "data_shape",
    "source_pos_file",
]


@dataclass(frozen=True)
class TaskEpochCandidate:
    """One candidate task/epoch metadata node."""

    field_path: str
    epoch: int | None
    task_type: str
    environment: str
    description: str
    task_keys: str
    value_type: str


def _as_text(value: Any, *, max_len: int = 160) -> str:
    """Convert small MATLAB-derived values into compact text."""

    if value is None:
        return ""

    if isinstance(value, bytes):
        text = value.decode("utf-8", errors="replace")
        return text[:max_len]

    if isinstance(value, str):
        return value[:max_len]

    if isinstance(value, (int, float, np.integer, np.floating, bool, np.bool_)):
        if pd.isna(value):
            return ""
        return str(value)[:max_len]

    if isinstance(value, np.ndarray):
        arr = np.asarray(value)
        if arr.shape == ():
            return _as_text(arr.item(), max_len=max_len)
        if arr.size <= 8:
            return _as_text(arr.tolist(), max_len=max_len)
        return f"array(shape={'x'.join(str(v) for v in arr.shape)})"

    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return ""
        if len(value) <= 8:
            return ";".join(_as_text(v, max_len=40) for v in value)[:max_len]
        return f"{type(value).__name__}(len={len(value)})"

    text = str(value)
    return text[:max_len]


def _first_text(mapping: dict[str, Any], candidate_keys: tuple[str, ...]) -> str:
    lower_to_key = {str(k).lower(): k for k in mapping.keys()}
    for key in candidate_keys:
        actual = lower_to_key.get(key.lower())
        if actual is not None:
            text = _as_text(mapping.get(actual))
            if text:
                return text
    return ""


def _infer_epoch_from_path(field_path: str) -> int | None:
    bracket_matches = re.findall(r"\[(\d+)\]", field_path)
    if bracket_matches:
        return int(bracket_matches[-1]) + 1

    patterns = (
        r"epoch[_-]?(\d+)",
        r"ep[_-]?(\d+)",
        r"run[_-]?(\d+)",
        r"task[_-]?(\d+)",
    )
    for pattern in patterns:
        match = re.search(pattern, field_path, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))

    return None


def _looks_like_task_metadata(mapping: dict[str, Any]) -> bool:
    keys = {str(k).lower() for k in mapping.keys()}

    metadata_keys = {
        "type",
        "task_type",
        "tasktype",
        "task",
        "name",
        "taskname",
        "environment",
        "env",
        "arena",
        "track",
        "description",
        "desc",
        "notes",
        "linearcoord",
        "trajectories",
        "rewards",
        "well",
        "wells",
    }

    # Avoid treating the top-level {"task": [...]} wrapper as an epoch node
    # unless it carries additional task metadata.
    if keys == {"task"}:
        return False

    return bool(keys & metadata_keys)


def _candidate_from_mapping(field_path: str, mapping: dict[str, Any]) -> TaskEpochCandidate:
    task_type = _first_text(
        mapping,
        (
            "type",
            "task_type",
            "tasktype",
            "taskname",
            "name",
            "task",
        ),
    )
    environment = _first_text(
        mapping,
        (
            "environment",
            "env",
            "arena",
            "track",
        ),
    )
    description = _first_text(
        mapping,
        (
            "description",
            "desc",
            "notes",
            "comment",
            "comments",
        ),
    )

    return TaskEpochCandidate(
        field_path=field_path or "root",
        epoch=_infer_epoch_from_path(field_path),
        task_type=task_type,
        environment=environment,
        description=description,
        task_keys=";".join(sorted(str(k) for k in mapping.keys())),
        value_type=type(mapping).__name__,
    )


def _candidate_from_leaf(field_path: str, value: Any) -> TaskEpochCandidate:
    text = _as_text(value)
    return TaskEpochCandidate(
        field_path=field_path or "root",
        epoch=_infer_epoch_from_path(field_path),
        task_type=text,
        environment="",
        description="",
        task_keys="",
        value_type=type(value).__name__,
    )


def find_task_epoch_candidates(mat_data: dict[str, Any]) -> list[TaskEpochCandidate]:
    """Find candidate task/epoch metadata nodes.

    This is intentionally conservative: it reports candidate metadata nodes
    without trying to infer behavioral trials.
    """

    candidates: list[TaskEpochCandidate] = []

    def visit(obj: Any, field_path: str) -> None:
        if isinstance(obj, dict):
            if _looks_like_task_metadata(obj):
                candidates.append(_candidate_from_mapping(field_path, obj))
                return

            for key, value in obj.items():
                child = f"{field_path}.{key}" if field_path else str(key)
                visit(value, child)
            return

        if isinstance(obj, (list, tuple)):
            for idx, value in enumerate(obj):
                child = f"{field_path}[{idx}]" if field_path else f"[{idx}]"
                visit(value, child)
            return

        if isinstance(obj, np.ndarray):
            arr = np.asarray(obj)

            if arr.dtype == object:
                for idx, value in enumerate(arr.flat):
                    child = f"{field_path}[{idx}]" if field_path else f"[{idx}]"
                    visit(value, child)
                return

            # Numeric/string task arrays are valid weak probes for minimal files.
            if field_path.lower().endswith("task") or ".task" in field_path.lower():
                candidates.append(_candidate_from_leaf(field_path, arr))
            return

        if field_path.lower().endswith("task") or ".task" in field_path.lower():
            candidates.append(_candidate_from_leaf(field_path, obj))

    visit(mat_data, "")

    # Fallback for unusual files where task exists but no nested epoch-like
    # structure was recognized.
    if not candidates and "task" in mat_data:
        candidates.append(_candidate_from_leaf("task", mat_data["task"]))

    return candidates


def _finite_min(values: np.ndarray) -> float | str:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return ""
    return float(np.min(finite))


def _finite_max(values: np.ndarray) -> float | str:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return ""
    return float(np.max(finite))


def _select_mat_file(
    animal_dir: Path,
    *,
    day: int,
    file_kind: str,
) -> Path:
    matching = [
        p for p in list_animal_mat_files(animal_dir)
        if infer_file_kind(p) == file_kind and infer_day_from_filename(p) == day
    ]

    if not matching:
        raise FileNotFoundError(
            f"No {file_kind} MAT file found for day {day} under {animal_dir}"
        )

    return sorted(matching)[0]


def _position_candidate_to_row(
    candidate: PositionCandidate,
    *,
    dataset_id: str,
    animal_id: str,
    day: int,
    source_pos_file: Path,
    fallback_epoch: int,
) -> dict[str, Any]:
    arr = candidate.matrix
    x = arr[:, candidate.x_col]
    y = arr[:, candidate.y_col]

    if candidate.time_col is not None:
        time = arr[:, candidate.time_col]
        time_min = _finite_min(time)
        time_max = _finite_max(time)
    else:
        time_min = ""
        time_max = ""

    return {
        "dataset_id": dataset_id,
        "animal_id": animal_id,
        "day": int(day),
        "epoch": int(candidate.epoch or fallback_epoch),
        "position_field_path": candidate.field_path,
        "n_position_samples": int(arr.shape[0]),
        "time_min": time_min,
        "time_max": time_max,
        "x_min": _finite_min(x),
        "x_max": _finite_max(x),
        "y_min": _finite_min(y),
        "y_max": _finite_max(y),
        "has_time": bool(candidate.time_col is not None),
        "data_shape": "x".join(str(v) for v in arr.shape),
        "source_pos_file": str(source_pos_file),
    }


def build_position_epoch_summary(
    animal_dir: str | Path,
    *,
    day: int,
    dataset_id: str = "crcns_hc6",
    animal_id: str | None = None,
    source_kind: str = "pos",
) -> pd.DataFrame:
    """Build position summary by epoch for one animal/day."""

    root = Path(animal_dir)
    source_pos_file = _select_mat_file(root, day=day, file_kind=source_kind)
    resolved_animal_id = animal_id or infer_animal_prefix(source_pos_file)

    pos_data = read_mat(source_pos_file)
    candidates = find_position_candidates(pos_data)

    rows = [
        _position_candidate_to_row(
            candidate,
            dataset_id=dataset_id,
            animal_id=resolved_animal_id,
            day=day,
            source_pos_file=source_pos_file,
            fallback_epoch=idx,
        )
        for idx, candidate in enumerate(candidates, start=1)
    ]

    return pd.DataFrame(rows, columns=POSITION_EPOCH_SUMMARY_COLUMNS)


def _blank_position_row() -> dict[str, Any]:
    return {
        "n_position_samples": "",
        "time_min": "",
        "time_max": "",
        "x_min": "",
        "x_max": "",
        "y_min": "",
        "y_max": "",
    }


def _task_candidate_to_row(
    candidate: TaskEpochCandidate,
    *,
    dataset_id: str,
    animal_id: str,
    day: int,
    source_task_file: Path,
    source_pos_file: Path | None,
    position_by_epoch: dict[int, dict[str, Any]],
    fallback_epoch: int,
) -> dict[str, Any]:
    epoch = int(candidate.epoch or fallback_epoch)
    pos_row = position_by_epoch.get(epoch, _blank_position_row())

    return {
        "dataset_id": dataset_id,
        "animal_id": animal_id,
        "day": int(day),
        "epoch": epoch,
        "task_field_path": candidate.field_path,
        "task_type": candidate.task_type,
        "environment": candidate.environment,
        "description": candidate.description,
        "task_keys": candidate.task_keys,
        "task_value_type": candidate.value_type,
        "n_position_samples": pos_row.get("n_position_samples", ""),
        "time_min": pos_row.get("time_min", ""),
        "time_max": pos_row.get("time_max", ""),
        "x_min": pos_row.get("x_min", ""),
        "x_max": pos_row.get("x_max", ""),
        "y_min": pos_row.get("y_min", ""),
        "y_max": pos_row.get("y_max", ""),
        "source_task_file": str(source_task_file),
        "source_pos_file": str(source_pos_file) if source_pos_file else "",
    }


def extract_task_probe(
    animal_dir: str | Path,
    output_dir: str | Path,
    *,
    day: int,
    dataset_id: str = "crcns_hc6",
    animal_id: str | None = None,
    position_source_kind: str = "pos",
    allow_missing_position: bool = False,
) -> dict[str, Any]:
    """Write task/epoch probe tables for one CRCNS animal/day."""

    root = Path(animal_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    source_task_file = _select_mat_file(root, day=day, file_kind="task")
    resolved_animal_id = animal_id or infer_animal_prefix(source_task_file)

    task_data = read_mat(source_task_file)
    task_candidates = find_task_epoch_candidates(task_data)

    source_pos_file: Path | None = None
    position_summary = pd.DataFrame(columns=POSITION_EPOCH_SUMMARY_COLUMNS)

    try:
        source_pos_file = _select_mat_file(root, day=day, file_kind=position_source_kind)
        position_summary = build_position_epoch_summary(
            root,
            day=day,
            dataset_id=dataset_id,
            animal_id=resolved_animal_id,
            source_kind=position_source_kind,
        )
    except FileNotFoundError:
        if not allow_missing_position:
            raise

    position_by_epoch: dict[int, dict[str, Any]] = {}
    if not position_summary.empty:
        for _, row in position_summary.iterrows():
            try:
                epoch = int(row["epoch"])
            except (TypeError, ValueError):
                continue
            position_by_epoch[epoch] = row.to_dict()

    task_rows = [
        _task_candidate_to_row(
            candidate,
            dataset_id=dataset_id,
            animal_id=resolved_animal_id,
            day=day,
            source_task_file=source_task_file,
            source_pos_file=source_pos_file,
            position_by_epoch=position_by_epoch,
            fallback_epoch=idx,
        )
        for idx, candidate in enumerate(task_candidates, start=1)
    ]

    task_probe = pd.DataFrame(task_rows, columns=TASK_EPOCH_PROBE_COLUMNS)

    task_probe_csv = out_dir / "Table_CRCNS_WTrack_Task_Epoch_Probe.csv"
    position_summary_csv = out_dir / "Table_CRCNS_WTrack_Position_Epoch_Summary.csv"
    meta_json = out_dir / "crcns_wtrack_task_probe_meta.json"

    task_probe.to_csv(task_probe_csv, index=False)
    position_summary.to_csv(position_summary_csv, index=False)

    meta: dict[str, Any] = {
        "script": "vte.lab_adapters.crcns_wtrack.extract_task_probe",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_id": dataset_id,
        "animal_dir": str(root),
        "animal_id": resolved_animal_id,
        "day": int(day),
        "source_task_file": str(source_task_file),
        "source_pos_file": str(source_pos_file) if source_pos_file else "",
        "position_source_kind": position_source_kind,
        "output_dir": str(out_dir),
        "n_task_epoch_candidates": int(len(task_probe)),
        "n_position_epochs": int(len(position_summary)),
        "task_probe_csv": str(task_probe_csv),
        "position_summary_csv": str(position_summary_csv),
        "allow_missing_position": bool(allow_missing_position),
    }

    meta_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Task epoch probe saved: {task_probe_csv}")
    print(f"Position epoch summary saved: {position_summary_csv}")
    print(f"Metadata saved: {meta_json}")

    return meta


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Probe CRCNS W-track task/epoch metadata for Stage 3.2C."
    )
    parser.add_argument("--animal-dir", required=True, type=Path)
    parser.add_argument("--day", required=True, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dataset-id", default="crcns_hc6")
    parser.add_argument("--animal-id", default=None)
    parser.add_argument(
        "--position-source-kind",
        default="pos",
        choices=["pos", "rawpos"],
        help="Position MAT file kind to align against. Default: pos.",
    )
    parser.add_argument(
        "--allow-missing-position",
        action="store_true",
        help="Write task metadata even if no matching position file exists.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    extract_task_probe(
        animal_dir=args.animal_dir,
        output_dir=args.output_dir,
        day=args.day,
        dataset_id=args.dataset_id,
        animal_id=args.animal_id,
        position_source_kind=args.position_source_kind,
        allow_missing_position=args.allow_missing_position,
    )


if __name__ == "__main__":
    main()