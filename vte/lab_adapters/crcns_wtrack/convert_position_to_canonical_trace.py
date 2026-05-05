"""Convert CRCNS Frank-lab W-track position MAT files to canonical VTE trace.

Patch 14C scope:
- raw epoch-level conversion;
- no trial segmentation;
- no choice-zone inference;
- no VTE metric computation.

Each position epoch becomes one canonical trace trial. The resulting CSV satisfies
the current VTE wrapper required columns and preserves lab-specific metadata as
extra columns.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from vte.lab_adapters.crcns_wtrack.extract_position_probe import (
    PositionCandidate,
    find_position_candidates,
)
from vte.lab_adapters.crcns_wtrack.extract_task_probe import (
    find_task_epoch_candidates,
)
from vte.lab_adapters.crcns_wtrack.mat_loader import (
    infer_animal_prefix,
    infer_day_from_filename,
    infer_file_kind,
    list_animal_mat_files,
    read_mat,
)


CANONICAL_TRACE_COLUMNS = [
    # Existing vte.core.schema required columns.
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
    # Existing optional / Stage 3 compatible metadata.
    "protocol",
    "condition",
    "ablation",
    "trial_phase",
    "pose_source",
    "event_type",
    "event_trial",
    "target_path",
    # Lab/canonical extra metadata.
    "dataset_id",
    "trace_origin",
    "subject_id",
    "animal_id",
    "session_id",
    "day",
    "epoch",
    "sample_index",
    "time_s",
    "geometry_id",
    "route_id",
    "coordinate_system",
    "source_file",
    "task_type",
    "task_environment",
    "task_description",
]

EPOCH_SUMMARY_COLUMNS = [
    "dataset_id",
    "animal_id",
    "day",
    "epoch",
    "run_id",
    "trial",
    "n_samples",
    "time_min",
    "time_max",
    "x_min",
    "x_max",
    "y_min",
    "y_max",
    "has_time",
    "task_type",
    "task_environment",
    "task_description",
    "source_file",
]


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


def _compute_heading(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute per-sample heading in radians from x/y samples."""

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) == 0:
        return np.asarray([], dtype=float)

    if len(x) == 1:
        return np.asarray([0.0], dtype=float)

    dx = np.diff(x, prepend=x[0])
    dy = np.diff(y, prepend=y[0])

    # First diff is often zero because of prepend. Use the second step if present.
    if len(dx) > 1:
        dx[0] = dx[1]
        dy[0] = dy[1]

    heading = np.arctan2(dy, dx)

    # Replace non-finite headings caused by missing coordinates or zero-length
    # samples by nearest available value, then by 0 if the epoch is fully invalid.
    heading = pd.Series(heading).replace([np.inf, -np.inf], np.nan).ffill().bfill()
    heading = heading.fillna(0.0).to_numpy(dtype=float)

    return heading


def _task_metadata_by_epoch(task_file: Path | None) -> dict[int, dict[str, Any]]:
    if task_file is None or not task_file.exists():
        return {}

    task_data = read_mat(task_file)
    candidates = find_task_epoch_candidates(task_data)

    out: dict[int, dict[str, Any]] = {}
    for idx, candidate in enumerate(candidates, start=1):
        epoch = int(candidate.epoch or idx)
        out[epoch] = {
            "task_type": candidate.task_type,
            "task_environment": candidate.environment,
            "task_description": candidate.description,
        }

    return out


def _extract_candidate_arrays(candidate: PositionCandidate) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    arr = candidate.matrix

    x = arr[:, candidate.x_col].astype(float)
    y = arr[:, candidate.y_col].astype(float)

    if candidate.time_col is not None:
        time_s = arr[:, candidate.time_col].astype(float)
    else:
        time_s = None

    return x, y, time_s


def _candidate_to_trace_rows(
    candidate: PositionCandidate,
    *,
    dataset_id: str,
    animal_id: str,
    day: int,
    epoch: int,
    run_id: str,
    session_id: str,
    geometry_id: str,
    coordinate_system: str,
    source_file: Path,
    task_meta: dict[str, Any],
    trial: int,
    max_samples_per_epoch: int | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    x, y, time_s = _extract_candidate_arrays(candidate)
    n = len(x)

    if n == 0:
        return [], {}

    if max_samples_per_epoch is not None and max_samples_per_epoch > 0 and n > max_samples_per_epoch:
        keep = np.linspace(0, n - 1, max_samples_per_epoch).round().astype(int)
        x = x[keep]
        y = y[keep]
        if time_s is not None:
            time_s = time_s[keep]
        n = len(x)

    heading = _compute_heading(x, y)

    if time_s is None:
        time_values: list[float | str] = ["" for _ in range(n)]
        time_min: float | str = ""
        time_max: float | str = ""
    else:
        time_values = [float(v) if np.isfinite(v) else "" for v in time_s]
        time_min = _finite_min(time_s)
        time_max = _finite_max(time_s)

    task_type = task_meta.get("task_type", "")
    task_environment = task_meta.get("task_environment", "")
    task_description = task_meta.get("task_description", "")

    rows: list[dict[str, Any]] = []
    for i in range(n):
        rows.append(
            {
                "run_id": run_id,
                "seed": animal_id,
                "trial": int(trial),
                "tick": int(i + 1),
                "x": float(x[i]) if np.isfinite(x[i]) else "",
                "y": float(y[i]) if np.isfinite(y[i]) else "",
                "heading": float(heading[i]) if np.isfinite(heading[i]) else 0.0,
                "choice_point_id": f"epoch_{epoch:02d}",
                "at_choice_point": True,
                "action": "track",
                "committed_path": "",
                "reward": 0.0,
                "done": bool(i == n - 1),
                "protocol": "crcns_wtrack",
                "condition": "",
                "ablation": "",
                "trial_phase": "epoch",
                "pose_source": "lab_tracking",
                "event_type": "",
                "event_trial": "",
                "target_path": "",
                "dataset_id": dataset_id,
                "trace_origin": "biological",
                "subject_id": animal_id,
                "animal_id": animal_id,
                "session_id": session_id,
                "day": int(day),
                "epoch": int(epoch),
                "sample_index": int(i),
                "time_s": time_values[i],
                "geometry_id": geometry_id,
                "route_id": f"epoch_{epoch:02d}",
                "coordinate_system": coordinate_system,
                "source_file": str(source_file),
                "task_type": task_type,
                "task_environment": task_environment,
                "task_description": task_description,
            }
        )

    summary = {
        "dataset_id": dataset_id,
        "animal_id": animal_id,
        "day": int(day),
        "epoch": int(epoch),
        "run_id": run_id,
        "trial": int(trial),
        "n_samples": int(n),
        "time_min": time_min,
        "time_max": time_max,
        "x_min": _finite_min(x),
        "x_max": _finite_max(x),
        "y_min": _finite_min(y),
        "y_max": _finite_max(y),
        "has_time": bool(time_s is not None),
        "task_type": task_type,
        "task_environment": task_environment,
        "task_description": task_description,
        "source_file": str(source_file),
    }

    return rows, summary


def convert_position_to_canonical_trace(
    animal_dir: str | Path,
    output_dir: str | Path,
    *,
    day: int,
    dataset_id: str = "crcns_hc6",
    animal_id: str | None = None,
    position_source_kind: str = "pos",
    geometry_id: str = "crcns_wtrack_raw",
    coordinate_system: str = "lab_position_units",
    max_samples_per_epoch: int | None = None,
) -> dict[str, Any]:
    """Convert one CRCNS animal/day position file to canonical VTE trace."""

    root = Path(animal_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    source_pos_file = _select_mat_file(root, day=day, file_kind=position_source_kind)
    resolved_animal_id = animal_id or infer_animal_prefix(source_pos_file)

    try:
        source_task_file: Path | None = _select_mat_file(root, day=day, file_kind="task")
    except FileNotFoundError:
        source_task_file = None

    pos_data = read_mat(source_pos_file)
    candidates = find_position_candidates(pos_data)

    task_by_epoch = _task_metadata_by_epoch(source_task_file)

    session_id = f"{dataset_id}_{resolved_animal_id}_day{day:02d}"
    run_id = session_id

    trace_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for idx, candidate in enumerate(candidates, start=1):
        epoch = int(candidate.epoch or idx)
        rows, summary = _candidate_to_trace_rows(
            candidate,
            dataset_id=dataset_id,
            animal_id=resolved_animal_id,
            day=day,
            epoch=epoch,
            run_id=run_id,
            session_id=session_id,
            geometry_id=geometry_id,
            coordinate_system=coordinate_system,
            source_file=source_pos_file,
            task_meta=task_by_epoch.get(epoch, {}),
            trial=epoch,
            max_samples_per_epoch=max_samples_per_epoch,
        )
        trace_rows.extend(rows)
        if summary:
            summary_rows.append(summary)

    trace = pd.DataFrame(trace_rows, columns=CANONICAL_TRACE_COLUMNS)
    epoch_summary = pd.DataFrame(summary_rows, columns=EPOCH_SUMMARY_COLUMNS)

    trace_csv = out_dir / "crcns_wtrack_canonical_trace.csv"
    epoch_summary_csv = out_dir / "Table_CRCNS_WTrack_Canonical_Epoch_Summary.csv"
    meta_json = out_dir / "crcns_wtrack_canonical_trace_meta.json"

    trace.to_csv(trace_csv, index=False)
    epoch_summary.to_csv(epoch_summary_csv, index=False)

    meta: dict[str, Any] = {
        "script": "vte.lab_adapters.crcns_wtrack.convert_position_to_canonical_trace",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_id": dataset_id,
        "animal_dir": str(root),
        "animal_id": resolved_animal_id,
        "day": int(day),
        "source_pos_file": str(source_pos_file),
        "source_task_file": str(source_task_file) if source_task_file else "",
        "position_source_kind": position_source_kind,
        "geometry_id": geometry_id,
        "coordinate_system": coordinate_system,
        "output_dir": str(out_dir),
        "trace_csv": str(trace_csv),
        "epoch_summary_csv": str(epoch_summary_csv),
        "n_epochs": int(len(epoch_summary)),
        "n_trace_rows": int(len(trace)),
        "max_samples_per_epoch": max_samples_per_epoch,
        "raw_epoch_level": True,
        "choice_zone_inference": False,
        "trial_segmentation": False,
    }

    meta_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Canonical trace saved: {trace_csv}")
    print(f"Epoch summary saved: {epoch_summary_csv}")
    print(f"Metadata saved: {meta_json}")

    return meta


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert CRCNS W-track position MAT files to canonical VTE trace."
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
    )
    parser.add_argument("--geometry-id", default="crcns_wtrack_raw")
    parser.add_argument("--coordinate-system", default="lab_position_units")
    parser.add_argument(
        "--max-samples-per-epoch",
        type=int,
        default=None,
        help="Optional downsampling cap per epoch. Default keeps all samples.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    convert_position_to_canonical_trace(
        animal_dir=args.animal_dir,
        output_dir=args.output_dir,
        day=args.day,
        dataset_id=args.dataset_id,
        animal_id=args.animal_id,
        position_source_kind=args.position_source_kind,
        geometry_id=args.geometry_id,
        coordinate_system=args.coordinate_system,
        max_samples_per_epoch=args.max_samples_per_epoch,
    )


if __name__ == "__main__":
    main()