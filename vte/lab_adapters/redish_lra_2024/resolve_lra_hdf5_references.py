from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

try:
    import h5py
except ImportError:  # pragma: no cover
    h5py = None


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

DEFAULT_VARIABLES = {
    "IdPhi",
    "zIdPhi",
    "RobustZIdPhi",
    "AvgIdPhi",
    "MeanIdPhi",
    "MedianIdPhi",
    "pVTE",
    "VTEMethod",
    "VTEThreshold",
    "SSNs",
    "RatID",
    "SessDate",
    "Task",
    "TaskPhase",
    "Contingency",
    "SwitchTimes",
    "SwitchLaps",
    "ChangePointLap",
    "CPTimes",
    "DCZ",
    "BEHAVIOR",
    "Correct",
    "ContingencyCorrect_L",
    "ContingencyCorrect_R",
    "ContingencyCorrect_A",
    "ZoneTimes_CentralPath",
    "ZoneTimes_FeederLeft",
    "ZoneTimes_FeederRight",
    "ZoneTimes_FeederRailLeftTop",
    "ZoneTimes_FeederRailLeftBottom",
    "ZoneTimes_FeederRailRightTop",
    "ZoneTimes_FeederRailRightBottom",
}


def _format_number(value: Any) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(value):
        return ""
    return f"{value:.12g}"


def _shape_text(shape: Any) -> str:
    if shape is None:
        return ""
    if isinstance(shape, tuple):
        return "x".join(str(int(x)) for x in shape)
    return str(shape)


def _attr_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        parts = [_attr_text(x) for x in value.reshape(-1)]
        return ";".join(x for x in parts if x)
    return str(value)


def _is_ref_dtype(dtype: Any) -> bool:
    if h5py is None:
        return False
    try:
        return h5py.check_dtype(ref=dtype) is not None
    except TypeError:
        return False


def _safe_read_sample(ds: Any, max_per_dim: int = 4) -> np.ndarray:
    shape = tuple(ds.shape)

    if len(shape) == 0:
        return np.asarray(ds[()])

    slices = tuple(slice(0, min(int(dim), max_per_dim)) for dim in shape)
    return np.asarray(ds[slices])


def _safe_read_for_summary(ds: Any, max_full_values: int) -> tuple[np.ndarray, str]:
    shape = tuple(ds.shape)
    n_values = int(np.prod(shape)) if shape else 1

    if n_values <= max_full_values:
        try:
            return np.asarray(ds[()]), "full"
        except Exception:
            pass

    return _safe_read_sample(ds), "sample"


def _numeric_summary(arr: Any) -> tuple[str, str, str]:
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


def _decode_matlab_char(arr: Any) -> str:
    try:
        flat = np.ravel(arr)
    except Exception:
        return ""

    chars: list[str] = []
    for item in flat:
        try:
            if isinstance(item, np.generic):
                item = item.item()
            code = int(item)
        except (TypeError, ValueError):
            return ""

        if code == 0:
            continue
        if 0 <= code <= 0x10FFFF:
            chars.append(chr(code))

    return "".join(chars)


def _sample_text(arr: Any, matlab_class: str, max_items: int = 8) -> str:
    if matlab_class == "char":
        text = _decode_matlab_char(arr)
        return text[:300]

    try:
        flat = np.ravel(arr)
    except Exception:
        return ""

    parts: list[str] = []

    for item in flat[:max_items]:
        if isinstance(item, np.ndarray):
            if item.size == 1:
                item = item.reshape(-1)[0]
            else:
                parts.append(f"ndarray{item.shape}")
                continue

        if isinstance(item, np.generic):
            item = item.item()

        if isinstance(item, bytes):
            parts.append(item.decode("utf-8", errors="replace"))
        else:
            parts.append(str(item))

    return "; ".join(parts)[:300]


def _row_base(
    cohort: str,
    source_table: str,
    source_file: Path,
    root_variable: str,
    logical_path: str,
    hdf5_path: str,
) -> dict[str, str]:
    return {
        "dataset_id": DATASET_ID,
        "cohort": cohort,
        "source_table": source_table,
        "source_file": str(source_file),
        "root_variable": root_variable,
        "logical_path": logical_path,
        "hdf5_path": hdf5_path,
    }


def _walk_node(
    f: Any,
    obj: Any,
    rows: list[dict[str, str]],
    cohort: str,
    source_table: str,
    source_file: Path,
    root_variable: str,
    logical_path: str,
    max_ref_items: int,
    max_depth: int,
    max_full_values: int,
    depth: int,
) -> None:
    if depth > max_depth:
        return

    matlab_class = _attr_text(obj.attrs.get("MATLAB_class", "")) if hasattr(obj, "attrs") else ""

    if h5py is not None and isinstance(obj, h5py.Group):
        row = _row_base(cohort, source_table, source_file, root_variable, logical_path, obj.name)
        row.update(
            {
                "node_type": "group",
                "matlab_class": matlab_class,
                "shape": "",
                "dtype": "",
                "n_ref_items": "",
                "numeric_min": "",
                "numeric_max": "",
                "numeric_finite_count": "",
                "summary_scope": "",
                "sample": "; ".join(list(obj.keys())[:20]),
            }
        )
        rows.append(row)

        for key in obj.keys():
            child = obj[key]
            child_logical = f"{logical_path}.{key}" if logical_path else key
            _walk_node(
                f=f,
                obj=child,
                rows=rows,
                cohort=cohort,
                source_table=source_table,
                source_file=source_file,
                root_variable=root_variable,
                logical_path=child_logical,
                max_ref_items=max_ref_items,
                max_depth=max_depth,
                max_full_values=max_full_values,
                depth=depth + 1,
            )
        return

    if h5py is not None and isinstance(obj, h5py.Dataset):
        dtype = str(obj.dtype)
        shape = tuple(obj.shape)
        is_ref = _is_ref_dtype(obj.dtype)

        sample = ""
        numeric_min = ""
        numeric_max = ""
        numeric_finite_count = ""
        summary_scope = ""
        n_ref_items = ""

        if is_ref:
            try:
                refs = np.ravel(np.asarray(obj[()]))
                n_ref_items = str(int(refs.size))
                sample = "; ".join(str(x) for x in refs[:8])
            except Exception as exc:
                sample = f"ref_read_error: {type(exc).__name__}: {exc}"
        else:
            try:
                arr, summary_scope = _safe_read_for_summary(obj, max_full_values=max_full_values)
                numeric_min, numeric_max, numeric_finite_count = _numeric_summary(arr)
                sample = _sample_text(arr, matlab_class=matlab_class)
            except Exception as exc:
                sample = f"read_error: {type(exc).__name__}: {exc}"

        row = _row_base(cohort, source_table, source_file, root_variable, logical_path, obj.name)
        row.update(
            {
                "node_type": "dataset_ref" if is_ref else "dataset",
                "matlab_class": matlab_class,
                "shape": _shape_text(shape),
                "dtype": dtype,
                "n_ref_items": n_ref_items,
                "numeric_min": numeric_min,
                "numeric_max": numeric_max,
                "numeric_finite_count": numeric_finite_count,
                "summary_scope": summary_scope,
                "sample": sample,
            }
        )
        rows.append(row)

        if is_ref and depth < max_depth:
            try:
                refs = np.ravel(np.asarray(obj[()]))
            except Exception:
                return

            for idx, ref in enumerate(refs[:max_ref_items]):
                try:
                    if not ref:
                        continue
                    target = f[ref]
                except Exception:
                    continue

                child_logical = f"{logical_path}[{idx}]"
                _walk_node(
                    f=f,
                    obj=target,
                    rows=rows,
                    cohort=cohort,
                    source_table=source_table,
                    source_file=source_file,
                    root_variable=root_variable,
                    logical_path=child_logical,
                    max_ref_items=max_ref_items,
                    max_depth=max_depth,
                    max_full_values=max_full_values,
                    depth=depth + 1,
                )

        return


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _wanted_root_variables(keys: list[str], requested: list[str] | None) -> list[str]:
    if requested:
        wanted = [x for x in requested if x in keys]
    else:
        wanted = [x for x in keys if x in DEFAULT_VARIABLES]

        extra_patterns = (
            "idphi",
            "vte",
            "choice",
            "correct",
            "contingency",
            "switch",
            "task",
            "phase",
            "zone",
            "feeder",
            "central",
            "behavior",
            "dcz",
        )
        for key in keys:
            lower = key.lower()
            if any(p in lower for p in extra_patterns) and key not in wanted:
                wanted.append(key)

    return [x for x in wanted if x != "#refs#"]


def resolve_lra_hdf5_references(
    root: Path,
    output_dir: Path,
    variables: list[str] | None,
    max_ref_items: int,
    max_depth: int,
    max_full_values: int,
) -> dict[str, Any]:
    if h5py is None:
        raise RuntimeError("h5py is required for MATLAB v7.3/HDF5 files")

    output_dir.mkdir(parents=True, exist_ok=True)

    root_rows: list[dict[str, str]] = []
    resolved_rows: list[dict[str, str]] = []
    skipped_rows: list[dict[str, str]] = []

    for cohort, source_table, rel_path in TARGET_FILES:
        source_file = root / Path(rel_path)

        if not source_file.exists():
            skipped_rows.append(
                {
                    "dataset_id": DATASET_ID,
                    "cohort": cohort,
                    "source_table": source_table,
                    "source_file": str(source_file),
                    "reason": "missing_file",
                }
            )
            continue

        if not h5py.is_hdf5(source_file):
            skipped_rows.append(
                {
                    "dataset_id": DATASET_ID,
                    "cohort": cohort,
                    "source_table": source_table,
                    "source_file": str(source_file),
                    "reason": "not_hdf5",
                }
            )
            continue

        with h5py.File(source_file, "r") as f:
            keys = list(f.keys())

            for key in keys:
                obj = f[key]
                matlab_class = _attr_text(obj.attrs.get("MATLAB_class", "")) if hasattr(obj, "attrs") else ""
                shape = _shape_text(tuple(obj.shape)) if isinstance(obj, h5py.Dataset) else ""
                dtype = str(obj.dtype) if isinstance(obj, h5py.Dataset) else ""

                root_rows.append(
                    {
                        "dataset_id": DATASET_ID,
                        "cohort": cohort,
                        "source_table": source_table,
                        "source_file": str(source_file),
                        "root_variable": key,
                        "node_type": "dataset" if isinstance(obj, h5py.Dataset) else "group",
                        "matlab_class": matlab_class,
                        "shape": shape,
                        "dtype": dtype,
                    }
                )

            wanted = _wanted_root_variables(keys, variables)

            for key in wanted:
                _walk_node(
                    f=f,
                    obj=f[key],
                    rows=resolved_rows,
                    cohort=cohort,
                    source_table=source_table,
                    source_file=source_file,
                    root_variable=key,
                    logical_path=key,
                    max_ref_items=max_ref_items,
                    max_depth=max_depth,
                    max_full_values=max_full_values,
                    depth=0,
                )

    root_csv = output_dir / "Table_Redish_LRA17B2_HDF5_Root_Variables.csv"
    resolved_csv = output_dir / "Table_Redish_LRA17B2_HDF5_Resolved_References.csv"
    skipped_csv = output_dir / "Table_Redish_LRA17B2_HDF5_Skipped_Files.csv"
    meta_json = output_dir / "redish_lra_17b2_hdf5_reference_meta.json"
    report_md = output_dir / "Redish_LRA_17B2_HDF5_Reference_Report.md"

    _write_csv(root_csv, root_rows)
    _write_csv(resolved_csv, resolved_rows)
    _write_csv(skipped_csv, skipped_rows)

    meta = {
        "dataset_id": DATASET_ID,
        "root": str(root),
        "n_root_rows": len(root_rows),
        "n_resolved_rows": len(resolved_rows),
        "n_skipped_files": len(skipped_rows),
        "variables": variables,
        "max_ref_items": max_ref_items,
        "max_depth": max_depth,
        "max_full_values": max_full_values,
        "outputs": {
            "root_variables": str(root_csv),
            "resolved_references": str(resolved_csv),
            "skipped_files": str(skipped_csv),
            "report": str(report_md),
        },
    }

    meta_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    report = [
        "# Redish LRA 2024 Patch 17B2 HDF5 reference audit",
        "",
        "Purpose: resolve MATLAB v7.3 HDF5 reference objects into logical MATLAB-style paths.",
        "",
        f"- Root variable rows: {len(root_rows)}",
        f"- Resolved rows: {len(resolved_rows)}",
        f"- Skipped files: {len(skipped_rows)}",
        f"- max_ref_items: {max_ref_items}",
        f"- max_depth: {max_depth}",
        "",
        "This is still an audit, not the canonical endpoint.",
        "",
    ]
    report_md.write_text("\n".join(report), encoding="utf-8")

    print(f"Root variables saved: {root_csv}")
    print(f"Resolved references saved: {resolved_csv}")
    print(f"Skipped files saved: {skipped_csv}")
    print(f"Metadata saved: {meta_json}")
    print(f"Report saved: {report_md}")
    print(f"Resolved rows: {len(resolved_rows)}")

    return meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--variables", nargs="*", default=None)
    parser.add_argument("--max-ref-items", type=int, default=12)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--max-full-values", type=int, default=200000)
    args = parser.parse_args()

    resolve_lra_hdf5_references(
        root=args.root,
        output_dir=args.output_dir,
        variables=args.variables,
        max_ref_items=args.max_ref_items,
        max_depth=args.max_depth,
        max_full_values=args.max_full_values,
    )


if __name__ == "__main__":
    main()