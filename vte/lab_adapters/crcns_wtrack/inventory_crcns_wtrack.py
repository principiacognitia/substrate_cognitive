"""Inventory CRCNS Frank-lab W-track MATLAB files.

This is a Stage 3.2C probe utility. It inspects file availability and only
lightly inspects .mat top-level variables. It does not compute VTE metrics.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from vte.lab_adapters.crcns_wtrack.mat_loader import (
    infer_animal_prefix,
    infer_day_from_filename,
    infer_file_kind,
    list_animal_mat_files,
    summarize_mat_top_level,
)


INVENTORY_COLUMNS = [
    "dataset_id",
    "animal_id",
    "file_kind",
    "day",
    "relative_path",
    "path",
    "size_bytes",
    "inspected",
    "top_keys",
    "top_key_count",
    "root_type",
    "root_shape",
]

DAY_MATRIX_COLUMNS = [
    "dataset_id",
    "animal_id",
    "day",
    "has_pos",
    "has_rawpos",
    "has_task",
    "has_spikes",
    "n_eeg_files",
    "n_metadata_files",
    "n_other_files",
]


def _as_int_or_blank(value: Any) -> int | str:
    if value is None or pd.isna(value):
        return ""
    return int(value)


def _should_inspect(path: Path, file_kind: str, max_inspect_bytes: int) -> bool:
    if file_kind == "eeg":
        return False
    try:
        return path.stat().st_size <= max_inspect_bytes
    except OSError:
        return False


def build_file_inventory(
    animal_dir: str | Path,
    *,
    dataset_id: str = "crcns_hc6",
    animal_id: str | None = None,
    max_inspect_bytes: int = 50_000_000,
) -> pd.DataFrame:
    """Build file-level inventory for one CRCNS animal directory."""

    root = Path(animal_dir)
    mat_files = list_animal_mat_files(root)

    rows: list[dict[str, Any]] = []
    for path in mat_files:
        file_kind = infer_file_kind(path)
        inspect = _should_inspect(path, file_kind, max_inspect_bytes)
        inferred_animal = animal_id or infer_animal_prefix(path)

        if inspect:
            summary = summarize_mat_top_level(path, inspect=True)
        else:
            summary = summarize_mat_top_level(path, inspect=False)

        try:
            relative_path = str(path.resolve().relative_to(root.resolve()))
        except ValueError:
            relative_path = str(path)

        rows.append(
            {
                "dataset_id": dataset_id,
                "animal_id": inferred_animal,
                "file_kind": file_kind,
                "day": _as_int_or_blank(infer_day_from_filename(path)),
                "relative_path": relative_path,
                "path": str(path),
                "size_bytes": summary.get("size_bytes", path.stat().st_size),
                "inspected": bool(inspect),
                "top_keys": summary.get("top_keys", ""),
                "top_key_count": summary.get("top_key_count", 0),
                "root_type": summary.get("root_type", ""),
                "root_shape": summary.get("root_shape", ""),
            }
        )

    return pd.DataFrame(rows, columns=INVENTORY_COLUMNS)


def build_day_file_matrix(inventory_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize file availability by recording day."""

    if inventory_df.empty:
        return pd.DataFrame(columns=DAY_MATRIX_COLUMNS)

    df = inventory_df.copy()
    df["day_numeric"] = pd.to_numeric(df["day"], errors="coerce")
    df = df.loc[df["day_numeric"].notna()].copy()

    if df.empty:
        return pd.DataFrame(columns=DAY_MATRIX_COLUMNS)

    rows: list[dict[str, Any]] = []
    group_cols = ["dataset_id", "animal_id", "day_numeric"]

    for (dataset_id, animal_id, day), g in df.groupby(group_cols, sort=True):
        kinds = g["file_kind"].astype(str)
        rows.append(
            {
                "dataset_id": dataset_id,
                "animal_id": animal_id,
                "day": int(day),
                "has_pos": bool((kinds == "pos").any()),
                "has_rawpos": bool((kinds == "rawpos").any()),
                "has_task": bool((kinds == "task").any()),
                "has_spikes": bool((kinds == "spikes").any()),
                "n_eeg_files": int((kinds == "eeg").sum()),
                "n_metadata_files": int((kinds == "metadata").sum()),
                "n_other_files": int((kinds == "other").sum()),
            }
        )

    return pd.DataFrame(rows, columns=DAY_MATRIX_COLUMNS)


def write_inventory(
    animal_dir: str | Path,
    output_dir: str | Path,
    *,
    dataset_id: str = "crcns_hc6",
    animal_id: str | None = None,
    max_inspect_bytes: int = 50_000_000,
) -> dict[str, Any]:
    """Write inventory tables and metadata."""

    root = Path(animal_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    inventory = build_file_inventory(
        root,
        dataset_id=dataset_id,
        animal_id=animal_id,
        max_inspect_bytes=max_inspect_bytes,
    )
    day_matrix = build_day_file_matrix(inventory)

    inventory_csv = out_dir / "Table_CRCNS_WTrack_File_Inventory.csv"
    day_matrix_csv = out_dir / "Table_CRCNS_WTrack_Day_File_Matrix.csv"
    meta_json = out_dir / "crcns_wtrack_inventory_meta.json"

    inventory.to_csv(inventory_csv, index=False)
    day_matrix.to_csv(day_matrix_csv, index=False)

    meta: dict[str, Any] = {
        "script": "vte.lab_adapters.crcns_wtrack.inventory_crcns_wtrack",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_id": dataset_id,
        "animal_dir": str(root),
        "output_dir": str(out_dir),
        "n_mat_files": int(len(inventory)),
        "n_days": int(day_matrix["day"].nunique()) if not day_matrix.empty else 0,
        "inventory_csv": str(inventory_csv),
        "day_matrix_csv": str(day_matrix_csv),
        "max_inspect_bytes": int(max_inspect_bytes),
    }

    meta_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"File inventory saved: {inventory_csv}")
    print(f"Day matrix saved: {day_matrix_csv}")
    print(f"Metadata saved: {meta_json}")

    return meta


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inventory CRCNS W-track MATLAB files for Stage 3.2C probing."
    )
    parser.add_argument("--animal-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dataset-id", default="crcns_hc6")
    parser.add_argument("--animal-id", default=None)
    parser.add_argument(
        "--max-inspect-bytes",
        default=50_000_000,
        type=int,
        help="Only inspect non-EEG MAT files up to this size.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    write_inventory(
        animal_dir=args.animal_dir,
        output_dir=args.output_dir,
        dataset_id=args.dataset_id,
        animal_id=args.animal_id,
        max_inspect_bytes=args.max_inspect_bytes,
    )


if __name__ == "__main__":
    main()