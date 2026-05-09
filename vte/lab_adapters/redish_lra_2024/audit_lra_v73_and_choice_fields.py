from __future__ import annotations

import argparse
import csv
import json
import math
import warnings
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:
    import h5py
except ImportError:  # pragma: no cover
    h5py = None

try:
    from scipy.io import loadmat
except ImportError:  # pragma: no cover
    loadmat = None


DATASET_ID = "redish_lra_2024"


TARGET_FILES = [
    ("lra", "idphi", "Processed Data/Processed Data/LRA/IdPhiData_LRA.mat"),
    ("lra", "lapdata", "Processed Data/Processed Data/LRA/LapData_Behav_LRA.mat"),
    ("lra", "session", "Processed Data/Processed Data/LRA/SessionData_LRA.mat"),
    ("lra", "changepoint", "Processed Data/Processed Data/LRA/ChangePointBehavAll_LRA.mat"),
    ("mpfc_dreadds", "idphi", "Processed Data/Processed Data/mPFC-DREADDs/IdPhiData_LRA_DREADDs.mat"),
    ("mpfc_dreadds", "lapdata", "Processed Data/Processed Data/mPFC-DREADDs/LapData_Behav_LRA_DREADDs.mat"),
    ("mpfc_dreadds", "session", "Processed Data/Processed Data/mPFC-DREADDs/SessionData_LRA_DREADDs.mat"),
    ("mpfc_dreadds", "changepoint", "Processed Data/Processed Data/mPFC-DREADDs/ChangePointBehav_LRA_DREADDs.mat"),
]


IDPHI_PATTERNS = (
    "idphi",
    "dphi",
    "vte",
    "choicepoint",
    "robustz",
    "zidphi",
    "avgidphi",
    "meanidphi",
    "medianidphi",
)

CHOICE_PATTERNS = (
    "choice",
    "choicepoint",
    "correct",
    "contingency",
    "left",
    "right",
    "feeder",
    "centralpath",
    "zone",
    "error",
    "switch",
    "lap",
    "task",
    "phase",
    "path",
    "rail",
)


def _as_posix(path: Path) -> str:
    return str(path).replace("\\", "/")


def _format_number(value: Any) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(value):
        return ""
    return f"{value:.12g}"


def _shape_to_text(shape: Any) -> str:
    if shape is None:
        return ""
    if isinstance(shape, tuple):
        return "x".join(str(x) for x in shape)
    if isinstance(shape, list):
        return "x".join(str(x) for x in shape)
    return str(shape)


def _safe_sample_array(arr: np.ndarray, max_items: int = 6) -> str:
    if arr.size == 0:
        return ""

    try:
        flat = arr.reshape(-1)
    except Exception:
        flat = np.ravel(arr)

    out = []
    for x in flat[:max_items]:
        if isinstance(x, bytes):
            out.append(x.decode("utf-8", errors="replace"))
        elif isinstance(x, np.generic):
            out.append(str(x.item()))
        else:
            out.append(str(x))
    return "; ".join(out)


def _numeric_summary_from_array(arr: np.ndarray) -> tuple[str, str, str]:
    if arr is None:
        return "", "", ""

    try:
        flat = np.ravel(arr)
    except Exception:
        return "", "", ""

    values: list[float] = []

    for item in flat:
        try:
            if isinstance(item, np.ndarray):
                if item.size != 1:
                    continue
                item = item.reshape(-1)[0]

            if isinstance(item, np.generic):
                item = item.item()

            if isinstance(item, (bytes, str)):
                continue

            value = float(item)
        except (TypeError, ValueError):
            continue

        if math.isfinite(value):
            values.append(value)

    if not values:
        return "", "", "0"

    numeric = np.asarray(values, dtype=float)
    return (
        _format_number(float(np.nanmin(numeric))),
        _format_number(float(np.nanmax(numeric))),
        str(int(numeric.size)),
    )


def _read_small_hdf5_sample(ds: Any) -> np.ndarray:
    shape = tuple(ds.shape)
    if len(shape) == 0:
        return np.asarray(ds[()])

    slices = tuple(slice(0, min(int(dim), 4)) for dim in shape)
    return np.asarray(ds[slices])


def _hdf5_rows(path: Path, cohort: str, source_table: str, max_datasets: int | None) -> list[dict[str, str]]:
    if h5py is None:
        return [
            {
                "dataset_id": DATASET_ID,
                "cohort": cohort,
                "source_table": source_table,
                "source_file": str(path),
                "loader": "h5py",
                "field_path": "",
                "kind": "load_error",
                "shape": "",
                "dtype": "",
                "numeric_min": "",
                "numeric_max": "",
                "numeric_finite_count": "",
                "sample": "h5py is not installed",
            }
        ]

    rows: list[dict[str, str]] = []
    n_seen = 0

    def visit(name: str, obj: Any) -> None:
        nonlocal n_seen

        if max_datasets is not None and n_seen >= max_datasets:
            return

        if isinstance(obj, h5py.Dataset):
            n_seen += 1
            kind = "dataset"
            shape = tuple(obj.shape)
            dtype = str(obj.dtype)
            sample = ""
            numeric_min = ""
            numeric_max = ""
            numeric_finite_count = ""

            try:
                sample_arr = _read_small_hdf5_sample(obj)
                sample = _safe_sample_array(sample_arr)
                numeric_min, numeric_max, numeric_finite_count = _numeric_summary_from_array(sample_arr)
            except Exception as exc:
                sample = f"sample_error: {type(exc).__name__}: {exc}"

            rows.append(
                {
                    "dataset_id": DATASET_ID,
                    "cohort": cohort,
                    "source_table": source_table,
                    "source_file": str(path),
                    "loader": "h5py",
                    "field_path": name,
                    "kind": kind,
                    "shape": _shape_to_text(shape),
                    "dtype": dtype,
                    "numeric_min": numeric_min,
                    "numeric_max": numeric_max,
                    "numeric_finite_count": numeric_finite_count,
                    "sample": sample,
                }
            )

        elif isinstance(obj, h5py.Group):
            rows.append(
                {
                    "dataset_id": DATASET_ID,
                    "cohort": cohort,
                    "source_table": source_table,
                    "source_file": str(path),
                    "loader": "h5py",
                    "field_path": name,
                    "kind": "group",
                    "shape": "",
                    "dtype": "",
                    "numeric_min": "",
                    "numeric_max": "",
                    "numeric_finite_count": "",
                    "sample": "",
                }
            )

    try:
        with h5py.File(path, "r") as f:
            f.visititems(visit)
    except Exception as exc:
        rows.append(
            {
                "dataset_id": DATASET_ID,
                "cohort": cohort,
                "source_table": source_table,
                "source_file": str(path),
                "loader": "h5py",
                "field_path": "",
                "kind": "load_error",
                "shape": "",
                "dtype": "",
                "numeric_min": "",
                "numeric_max": "",
                "numeric_finite_count": "",
                "sample": f"{type(exc).__name__}: {exc}",
            }
        )

    return rows


def _iter_scipy_fields(prefix: str, value: Any, depth: int, max_depth: int) -> Iterable[tuple[str, Any]]:
    if depth > max_depth:
        return

    yield prefix, value

    if isinstance(value, dict):
        for key, child in value.items():
            if key.startswith("__"):
                continue
            child_path = f"{prefix}.{key}" if prefix else key
            yield from _iter_scipy_fields(child_path, child, depth + 1, max_depth)
        return

    if isinstance(value, np.ndarray):
        if value.dtype.names:
            for name in value.dtype.names:
                try:
                    child = value[name]
                except Exception:
                    continue
                child_path = f"{prefix}.{name}" if prefix else name
                yield from _iter_scipy_fields(child_path, child, depth + 1, max_depth)
            return

        if value.dtype == object and value.size > 0 and depth < max_depth:
            flat = value.reshape(-1)
            for idx, child in enumerate(flat[:20]):
                child_path = f"{prefix}.[{idx}]"
                yield from _iter_scipy_fields(child_path, child, depth + 1, max_depth)
            return

    if hasattr(value, "_fieldnames"):
        for name in getattr(value, "_fieldnames", []):
            try:
                child = getattr(value, name)
            except Exception:
                continue
            child_path = f"{prefix}.{name}" if prefix else name
            yield from _iter_scipy_fields(child_path, child, depth + 1, max_depth)


def _kind_shape_dtype_sample(value: Any) -> tuple[str, str, str, str, str, str, str]:
    if isinstance(value, np.ndarray):
        kind = "ndarray" if value.dtype != object else "object_ndarray"
        shape = _shape_to_text(value.shape)
        dtype = str(value.dtype)
        sample = _safe_sample_array(value)
        numeric_min, numeric_max, numeric_finite_count = _numeric_summary_from_array(value)
        return kind, shape, dtype, numeric_min, numeric_max, numeric_finite_count, sample

    if isinstance(value, (str, bytes)):
        sample = value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value
        return "str", "", type(value).__name__, "", "", "", sample[:200]

    if isinstance(value, (int, float, np.integer, np.floating, bool)):
        text = str(value.item() if isinstance(value, np.generic) else value)
        return "scalar", "", type(value).__name__, _format_number(value), _format_number(value), "1", text

    if isinstance(value, dict):
        return "dict", "", "", "", "", "", "; ".join(list(value.keys())[:20])

    if hasattr(value, "_fieldnames"):
        return "mat_struct", "", "", "", "", "", "; ".join(getattr(value, "_fieldnames", [])[:20])

    return type(value).__name__, "", "", "", "", "", ""


def _scipy_rows(path: Path, cohort: str, source_table: str) -> list[dict[str, str]]:
    if loadmat is None:
        return [
            {
                "dataset_id": DATASET_ID,
                "cohort": cohort,
                "source_table": source_table,
                "source_file": str(path),
                "loader": "scipy",
                "field_path": "",
                "kind": "load_error",
                "shape": "",
                "dtype": "",
                "numeric_min": "",
                "numeric_max": "",
                "numeric_finite_count": "",
                "sample": "scipy is not installed",
            }
        ]

    rows: list[dict[str, str]] = []

    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            data = loadmat(path, squeeze_me=True, struct_as_record=False)
        warn_text = " | ".join(str(w.message) for w in caught)
    except NotImplementedError as exc:
        return [
            {
                "dataset_id": DATASET_ID,
                "cohort": cohort,
                "source_table": source_table,
                "source_file": str(path),
                "loader": "scipy",
                "field_path": "",
                "kind": "load_error",
                "shape": "",
                "dtype": "",
                "numeric_min": "",
                "numeric_max": "",
                "numeric_finite_count": "",
                "sample": f"{type(exc).__name__}: {exc}",
            }
        ]
    except Exception as exc:
        return [
            {
                "dataset_id": DATASET_ID,
                "cohort": cohort,
                "source_table": source_table,
                "source_file": str(path),
                "loader": "scipy",
                "field_path": "",
                "kind": "load_error",
                "shape": "",
                "dtype": "",
                "numeric_min": "",
                "numeric_max": "",
                "numeric_finite_count": "",
                "sample": f"{type(exc).__name__}: {exc}",
            }
        ]

    for field_path, value in _iter_scipy_fields("", data, 0, 4):
        if not field_path:
            continue
        kind, shape, dtype, numeric_min, numeric_max, numeric_finite_count, sample = _kind_shape_dtype_sample(value)

        rows.append(
            {
                "dataset_id": DATASET_ID,
                "cohort": cohort,
                "source_table": source_table,
                "source_file": str(path),
                "loader": "scipy",
                "field_path": field_path,
                "kind": kind,
                "shape": shape,
                "dtype": dtype,
                "numeric_min": numeric_min,
                "numeric_max": numeric_max,
                "numeric_finite_count": numeric_finite_count,
                "sample": sample,
                "mat_warnings": warn_text,
            }
        )

    return rows


def _is_hdf5(path: Path) -> bool:
    if h5py is None:
        return False
    try:
        return bool(h5py.is_hdf5(path))
    except Exception:
        return False


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _filter_rows(rows: list[dict[str, str]], patterns: tuple[str, ...]) -> list[dict[str, str]]:
    out = []
    for row in rows:
        text = " ".join(
            [
                row.get("source_table", ""),
                row.get("field_path", ""),
                row.get("sample", ""),
            ]
        ).lower()
        if any(pattern.lower() in text for pattern in patterns):
            out.append(row)
    return out


def audit_lra_v73_and_choice_fields(root: Path, output_dir: Path, max_hdf5_datasets: int | None) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    loadability_rows: list[dict[str, str]] = []
    structure_rows: list[dict[str, str]] = []

    for cohort, source_table, rel in TARGET_FILES:
        path = root / Path(rel)
        exists = path.exists()
        is_hdf5 = _is_hdf5(path) if exists else False

        loadability_rows.append(
            {
                "dataset_id": DATASET_ID,
                "cohort": cohort,
                "source_table": source_table,
                "source_file": str(path),
                "exists": str(bool(exists)),
                "size_bytes": str(path.stat().st_size) if exists else "",
                "is_hdf5": str(bool(is_hdf5)),
                "preferred_loader": "h5py" if is_hdf5 else "scipy",
            }
        )

        if not exists:
            structure_rows.append(
                {
                    "dataset_id": DATASET_ID,
                    "cohort": cohort,
                    "source_table": source_table,
                    "source_file": str(path),
                    "loader": "",
                    "field_path": "",
                    "kind": "missing_file",
                    "shape": "",
                    "dtype": "",
                    "numeric_min": "",
                    "numeric_max": "",
                    "numeric_finite_count": "",
                    "sample": "",
                }
            )
            continue

        if is_hdf5:
            structure_rows.extend(_hdf5_rows(path, cohort, source_table, max_hdf5_datasets))
        else:
            structure_rows.extend(_scipy_rows(path, cohort, source_table))

    idphi_rows = _filter_rows(structure_rows, IDPHI_PATTERNS)
    choice_rows = _filter_rows(structure_rows, CHOICE_PATTERNS)

    loadability_csv = output_dir / "Table_Redish_LRA17B_File_Loadability.csv"
    structure_csv = output_dir / "Table_Redish_LRA17B_Structure_Audit.csv"
    idphi_csv = output_dir / "Table_Redish_LRA17B_IdPhi_Field_Candidates.csv"
    choice_csv = output_dir / "Table_Redish_LRA17B_Choice_Field_Candidates.csv"
    meta_json = output_dir / "redish_lra_17b_audit_meta.json"
    report_md = output_dir / "Redish_LRA_17B_Audit_Report.md"

    _write_csv(loadability_csv, loadability_rows)
    _write_csv(structure_csv, structure_rows)
    _write_csv(idphi_csv, idphi_rows)
    _write_csv(choice_csv, choice_rows)

    meta = {
        "dataset_id": DATASET_ID,
        "root": str(root),
        "n_files_checked": len(loadability_rows),
        "n_structure_rows": len(structure_rows),
        "n_idphi_candidate_rows": len(idphi_rows),
        "n_choice_candidate_rows": len(choice_rows),
        "max_hdf5_datasets": max_hdf5_datasets,
        "outputs": {
            "loadability": str(loadability_csv),
            "structure_audit": str(structure_csv),
            "idphi_candidates": str(idphi_csv),
            "choice_candidates": str(choice_csv),
            "report": str(report_md),
        },
    }

    meta_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    report = [
        "# Redish LRA 2024 Patch 17B audit",
        "",
        "Purpose: identify readable v7.3/HDF5 fields and isolate real left/right decision fields before building a canonical biological endpoint.",
        "",
        f"- Files checked: {len(loadability_rows)}",
        f"- Structure rows: {len(structure_rows)}",
        f"- IdPhi candidate rows: {len(idphi_rows)}",
        f"- Choice candidate rows: {len(choice_rows)}",
        "",
        "Interpretation rule: this audit does not build the endpoint. It only verifies whether the required fields exist and how they are stored.",
        "",
    ]
    report_md.write_text("\n".join(report), encoding="utf-8")

    print(f"Loadability saved: {loadability_csv}")
    print(f"Structure audit saved: {structure_csv}")
    print(f"IdPhi candidates saved: {idphi_csv}")
    print(f"Choice candidates saved: {choice_csv}")
    print(f"Metadata saved: {meta_json}")
    print(f"Report saved: {report_md}")

    return meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-hdf5-datasets", type=int, default=None)
    args = parser.parse_args()

    audit_lra_v73_and_choice_fields(
        root=args.root,
        output_dir=args.output_dir,
        max_hdf5_datasets=args.max_hdf5_datasets,
    )


if __name__ == "__main__":
    main()