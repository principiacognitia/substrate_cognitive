"""Position probe for CRCNS Frank-lab W-track MATLAB files.

The probe tries to locate numeric position arrays and renders a first XY
inspection plot. It is not a trial parser and does not compute VTE metrics.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vte.lab_adapters.crcns_wtrack.mat_loader import (
    infer_animal_prefix,
    infer_day_from_filename,
    infer_file_kind,
    list_animal_mat_files,
    read_mat,
    walk_nested,
)


POSITION_PROBE_COLUMNS = [
    "dataset_id",
    "animal_id",
    "day",
    "epoch",
    "source_file",
    "field_path",
    "n_samples",
    "time_min",
    "time_max",
    "x_min",
    "x_max",
    "y_min",
    "y_max",
    "has_time",
    "has_x",
    "has_y",
    "data_shape",
]


@dataclass(frozen=True)
class PositionCandidate:
    field_path: str
    matrix: np.ndarray
    epoch: int | None
    time_col: int | None
    x_col: int
    y_col: int


def _is_numeric_array(value: Any) -> bool:
    return isinstance(value, np.ndarray) and value.dtype.kind in {"i", "u", "f"}


def _normalize_2d(value: Any) -> np.ndarray | None:
    if not _is_numeric_array(value):
        return None

    arr = np.asarray(value, dtype=float)
    arr = np.squeeze(arr)

    if arr.ndim != 2:
        return None

    if arr.shape[0] < 2 or arr.shape[1] < 2:
        return None

    # Common case: samples x columns.
    if arr.shape[1] >= 2:
        if arr.shape[0] <= 20 and arr.shape[1] > arr.shape[0]:
            # Possible columns x samples layout.
            return arr.T
        return arr

    return None


def _looks_monotonic_time(col: np.ndarray) -> bool:
    finite = col[np.isfinite(col)]
    if len(finite) < 3:
        return False

    diffs = np.diff(finite)
    if len(diffs) == 0:
        return False

    nonnegative_ratio = float((diffs >= 0).mean())
    distinct_ratio = float(len(np.unique(finite)) / max(len(finite), 1))
    return nonnegative_ratio >= 0.95 and distinct_ratio >= 0.50


def _infer_columns(arr: np.ndarray) -> tuple[int | None, int, int]:
    """Infer time/x/y columns.

    For Frank-lab position arrays the common layout is time, x, y, ...
    If the first column does not look like time, use x,y = columns 0,1.
    """

    if arr.shape[1] >= 3 and _looks_monotonic_time(arr[:, 0]):
        return 0, 1, 2
    return None, 0, 1


def _infer_epoch_from_path(field_path: str) -> int | None:
    bracket_matches = re.findall(r"\[(\d+)\]", field_path)
    if bracket_matches:
        return int(bracket_matches[-1]) + 1

    match = re.search(r"epoch[_-]?(\d+)", field_path, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))

    return None


def find_position_candidates(mat_data: dict[str, Any]) -> list[PositionCandidate]:
    """Find numeric arrays that plausibly contain position samples."""

    candidates: list[PositionCandidate] = []

    for field_path, value in walk_nested(mat_data):
        arr = _normalize_2d(value)
        if arr is None:
            continue

        n_samples, n_cols = arr.shape
        if n_samples < 10 or n_cols < 2:
            continue

        normalized_path = field_path.lower()
        if ".arg" in normalized_path or normalized_path.endswith("arg"):
            continue

        time_col, x_col, y_col = _infer_columns(arr)
        x = arr[:, x_col]
        y = arr[:, y_col]

        finite_xy = np.isfinite(x) & np.isfinite(y)
        if finite_xy.mean() < 0.50:
            continue

        # Frank-lab files can contain sentinel-like non-position values in
        # auxiliary arrays. Keep only plausible trajectory arrays.
        if np.nanmin(x[finite_xy]) <= -1e20 or np.nanmin(y[finite_xy]) <= -1e20:
            continue

        if np.nanmax(x[finite_xy]) >= 1e20 or np.nanmax(y[finite_xy]) >= 1e20:
            continue

        # Keep broad: this is a probe, not a definitive parser.
        candidates.append(
            PositionCandidate(
                field_path=field_path or "root",
                matrix=arr,
                epoch=_infer_epoch_from_path(field_path),
                time_col=time_col,
                x_col=x_col,
                y_col=y_col,
            )
        )

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


def _candidate_to_row(
    candidate: PositionCandidate,
    *,
    dataset_id: str,
    animal_id: str,
    day: int,
    source_file: Path,
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
        "source_file": str(source_file),
        "field_path": candidate.field_path,
        "n_samples": int(arr.shape[0]),
        "time_min": time_min,
        "time_max": time_max,
        "x_min": _finite_min(x),
        "x_max": _finite_max(x),
        "y_min": _finite_min(y),
        "y_max": _finite_max(y),
        "has_time": bool(candidate.time_col is not None),
        "has_x": True,
        "has_y": True,
        "data_shape": "x".join(str(v) for v in arr.shape),
    }


def _select_position_file(
    animal_dir: Path,
    *,
    day: int,
    source_kind: str = "pos",
) -> Path:
    files = list_animal_mat_files(animal_dir)

    kinds_to_try = [source_kind]
    if source_kind == "auto":
        kinds_to_try = ["pos", "rawpos"]

    for kind in kinds_to_try:
        matching = [
            p for p in files
            if infer_file_kind(p) == kind and infer_day_from_filename(p) == day
        ]
        if matching:
            return sorted(matching)[0]

    raise FileNotFoundError(
        f"No {source_kind} MAT file found for day {day} under {animal_dir}"
    )


def _plot_position_candidates(
    candidates: list[PositionCandidate],
    output_path: Path,
    *,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    for idx, candidate in enumerate(candidates, start=1):
        arr = candidate.matrix
        x = arr[:, candidate.x_col]
        y = arr[:, candidate.y_col]

        finite = np.isfinite(x) & np.isfinite(y)
        x = x[finite]
        y = y[finite]

        if len(x) == 0:
            continue

        stride = max(1, int(math.ceil(len(x) / 5000)))
        label = f"epoch {candidate.epoch or idx}"

        ax.plot(
            x[::stride],
            y[::stride],
            linewidth=0.8,
            alpha=0.85,
            label=label,
        )

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")

    if len(candidates) <= 12:
        ax.legend(fontsize=7, loc="best")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def extract_position_probe(
    animal_dir: str | Path,
    output_dir: str | Path,
    *,
    day: int,
    dataset_id: str = "crcns_hc6",
    animal_id: str | None = None,
    source_kind: str = "pos",
) -> dict[str, Any]:
    """Extract probe table and XY plot for one animal/day."""

    root = Path(animal_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    source_file = _select_position_file(root, day=day, source_kind=source_kind)
    resolved_animal_id = animal_id or infer_animal_prefix(source_file)

    mat_data = read_mat(source_file)
    candidates = find_position_candidates(mat_data)

    rows = [
        _candidate_to_row(
            candidate,
            dataset_id=dataset_id,
            animal_id=resolved_animal_id,
            day=day,
            source_file=source_file,
            fallback_epoch=idx,
        )
        for idx, candidate in enumerate(candidates, start=1)
    ]

    probe = pd.DataFrame(rows, columns=POSITION_PROBE_COLUMNS)

    probe_csv = out_dir / "Table_CRCNS_WTrack_Position_Probe.csv"
    figure_png = out_dir / f"Figure_CRCNS_WTrack_XY_Day{day:02d}_Epochs.png"
    meta_json = out_dir / "crcns_wtrack_position_probe_meta.json"

    probe.to_csv(probe_csv, index=False)

    if candidates:
        _plot_position_candidates(
            candidates,
            figure_png,
            title=f"CRCNS W-track position probe: {resolved_animal_id} day {day:02d}",
        )
    else:
        # Write an empty diagnostic figure rather than silently skipping output.
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.set_title(f"No position candidates found: {resolved_animal_id} day {day:02d}")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(figure_png, dpi=200)
        plt.close(fig)

    meta: dict[str, Any] = {
        "script": "vte.lab_adapters.crcns_wtrack.extract_position_probe",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_id": dataset_id,
        "animal_dir": str(root),
        "animal_id": resolved_animal_id,
        "day": int(day),
        "source_kind": source_kind,
        "source_file": str(source_file),
        "output_dir": str(out_dir),
        "n_position_candidates": int(len(candidates)),
        "probe_csv": str(probe_csv),
        "figure_png": str(figure_png),
    }

    meta_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Position probe saved: {probe_csv}")
    print(f"Figure saved: {figure_png}")
    print(f"Metadata saved: {meta_json}")

    return meta


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Probe CRCNS W-track position MAT files for Stage 3.2C."
    )
    parser.add_argument("--animal-dir", required=True, type=Path)
    parser.add_argument("--day", required=True, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dataset-id", default="crcns_hc6")
    parser.add_argument("--animal-id", default=None)
    parser.add_argument(
        "--source-kind",
        default="pos",
        choices=["pos", "rawpos", "auto"],
        help="MAT file kind to probe. Default: pos.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    extract_position_probe(
        animal_dir=args.animal_dir,
        output_dir=args.output_dir,
        day=args.day,
        dataset_id=args.dataset_id,
        animal_id=args.animal_id,
        source_kind=args.source_kind,
    )


if __name__ == "__main__":
    main()