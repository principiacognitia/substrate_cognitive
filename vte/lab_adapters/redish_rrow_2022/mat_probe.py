from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import warnings


MATLAB_PRIVATE_PREFIXES = ("__",)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if math.isnan(float(value)):
            return None
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    return str(value)


def is_hdf5_mat(path: Path) -> bool:
    """Return True for MATLAB v7.3/HDF5-like files."""
    with Path(path).open("rb") as f:
        header = f.read(8)
    return header.startswith(b"\x89HDF")


def _shape_to_string(shape: Any) -> str:
    if shape is None:
        return ""
    if isinstance(shape, tuple):
        return "x".join(str(x) for x in shape)
    return str(shape)


def _safe_numeric_summary(value: Any, max_values: int = 10000) -> dict[str, Any]:
    out = {
        "numeric_min": "",
        "numeric_max": "",
        "numeric_mean": "",
        "n_finite": "",
    }

    try:
        arr = np.asarray(value)
    except Exception:
        return out

    if arr.size == 0 or not np.issubdtype(arr.dtype, np.number):
        return out

    flat = arr.ravel()
    if flat.size > max_values:
        flat = flat[:max_values]

    finite = flat[np.isfinite(flat)]
    if finite.size == 0:
        out["n_finite"] = 0
        return out

    out["numeric_min"] = float(np.min(finite))
    out["numeric_max"] = float(np.max(finite))
    out["numeric_mean"] = float(np.mean(finite))
    out["n_finite"] = int(finite.size)
    return out


def _value_kind(value: Any) -> str:
    if isinstance(value, dict):
        return "dict"
    if _is_mat_struct(value):
        return "mat_struct"
    if isinstance(value, np.ndarray):
        if value.dtype.names:
            return "structured_ndarray"
        if value.dtype == object:
            return "object_ndarray"
        return "ndarray"
    if isinstance(value, (list, tuple)):
        return "sequence"
    return type(value).__name__


def _is_mat_struct(value: Any) -> bool:
    return hasattr(value, "_fieldnames")


def _iter_mat_struct_fields(value: Any) -> Iterable[tuple[str, Any]]:
    fieldnames = getattr(value, "_fieldnames", None) or []
    for field in fieldnames:
        try:
            yield field, getattr(value, field)
        except Exception:
            continue


def _iter_children(value: Any, max_object_items: int) -> Iterable[tuple[str, Any]]:
    if isinstance(value, dict):
        for key in sorted(value.keys()):
            if str(key).startswith(MATLAB_PRIVATE_PREFIXES):
                continue
            yield str(key), value[key]
        return

    if _is_mat_struct(value):
        yield from _iter_mat_struct_fields(value)
        return

    if isinstance(value, np.ndarray):
        if value.dtype.names:
            for name in value.dtype.names:
                try:
                    yield str(name), value[name]
                except Exception:
                    continue
            return

        if value.dtype == object:
            flat = value.ravel()
            count = 0
            for idx, item in enumerate(flat):
                if count >= max_object_items:
                    break
                if item is None:
                    continue
                yield f"[{idx}]", item
                count += 1
            return

    if isinstance(value, (list, tuple)):
        for idx, item in enumerate(value[:max_object_items]):
            yield f"[{idx}]", item


def _describe_value(value: Any) -> dict[str, Any]:
    row = {
        "kind": _value_kind(value),
        "python_type": type(value).__name__,
        "shape": "",
        "dtype": "",
        "size": "",
        "n_fields": "",
    }

    try:
        arr = np.asarray(value)
        row["shape"] = _shape_to_string(arr.shape)
        row["dtype"] = str(arr.dtype)
        row["size"] = int(arr.size)
    except Exception:
        pass

    if isinstance(value, dict):
        row["n_fields"] = len([k for k in value.keys() if not str(k).startswith(MATLAB_PRIVATE_PREFIXES)])
    elif _is_mat_struct(value):
        row["n_fields"] = len(getattr(value, "_fieldnames", None) or [])
    elif isinstance(value, np.ndarray) and value.dtype.names:
        row["n_fields"] = len(value.dtype.names)

    row.update(_safe_numeric_summary(value))
    return row


def summarize_scipy_mat_file(
    mat_path: Path,
    *,
    max_depth: int = 5,
    max_rows: int = 5000,
    max_object_items: int = 20,
) -> pd.DataFrame:
    scipy_io = __import__("scipy.io", fromlist=["loadmat"])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mat_data = scipy_io.loadmat(
            str(mat_path),
            squeeze_me=True,
            struct_as_record=False,
        )

    warning_messages = "; ".join(
        dict.fromkeys(str(w.message) for w in caught)
    )

    rows: list[dict[str, Any]] = []

    def visit(value: Any, path: str, depth: int) -> None:
        if len(rows) >= max_rows:
            return

        desc = _describe_value(value)
        rows.append(
            {
                "source_file": str(mat_path),
                "field_path": path,
                "depth": depth,
                "mat_warnings": warning_messages,
                **desc,
            }
        )

        if depth >= max_depth:
            return

        for child_name, child_value in _iter_children(value, max_object_items=max_object_items):
            if len(rows) >= max_rows:
                return
            child_path = child_name if not path else f"{path}.{child_name}"
            visit(child_value, child_path, depth + 1)

    for key, value in sorted(mat_data.items()):
        if str(key).startswith(MATLAB_PRIVATE_PREFIXES):
            continue
        visit(value, str(key), 0)

    return pd.DataFrame(rows)


def summarize_hdf5_mat_file(
    mat_path: Path,
    *,
    max_depth: int = 8,
    max_rows: int = 5000,
) -> pd.DataFrame:
    h5py = __import__("h5py")

    rows: list[dict[str, Any]] = []

    def visit_h5(name: str, obj: Any) -> None:
        if len(rows) >= max_rows:
            return

        depth = 0 if not name else name.count("/")
        if depth > max_depth:
            return

        attrs = {}
        try:
            attrs = {str(k): str(v) for k, v in obj.attrs.items()}
        except Exception:
            attrs = {}

        shape = getattr(obj, "shape", None)
        dtype = getattr(obj, "dtype", "")

        rows.append(
            {
                "source_file": str(mat_path),
                "field_path": name or "/",
                "depth": depth,
                "mat_warnings": "",
                "kind": "hdf5_dataset" if hasattr(obj, "shape") else "hdf5_group",
                "python_type": type(obj).__name__,
                "shape": _shape_to_string(shape),
                "dtype": str(dtype) if dtype is not None else "",
                "size": int(np.prod(shape)) if shape else "",
                "n_fields": "",
                "numeric_min": "",
                "numeric_max": "",
                "numeric_mean": "",
                "n_finite": "",
                "attrs_json": json.dumps(attrs, ensure_ascii=False),
            }
        )

    with h5py.File(mat_path, "r") as h5:
        visit_h5("/", h5)
        h5.visititems(visit_h5)

    return pd.DataFrame(rows)


def summarize_mat_file(
    mat_path: Path,
    *,
    max_depth: int = 5,
    max_rows: int = 5000,
    max_object_items: int = 20,
) -> pd.DataFrame:
    mat_path = Path(mat_path)

    if is_hdf5_mat(mat_path):
        return summarize_hdf5_mat_file(
            mat_path,
            max_depth=max_depth,
            max_rows=max_rows,
        )

    return summarize_scipy_mat_file(
        mat_path,
        max_depth=max_depth,
        max_rows=max_rows,
        max_object_items=max_object_items,
    )


def write_mat_probe_outputs(
    mat_files: list[Path],
    output_dir: Path,
    *,
    max_depth: int = 5,
    max_rows_per_file: int = 5000,
    max_object_items: int = 20,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    errors: list[dict[str, str]] = []

    for mat_file in mat_files:
        try:
            frame = summarize_mat_file(
                mat_file,
                max_depth=max_depth,
                max_rows=max_rows_per_file,
                max_object_items=max_object_items,
            )
            frames.append(frame)
        except Exception as exc:
            errors.append({"source_file": str(mat_file), "error": repr(exc)})

    if frames:
        summary = pd.concat(frames, ignore_index=True)
    else:
        summary = pd.DataFrame(
            columns=[
                "source_file",
                "field_path",
                "depth",
                "mat_warnings",
                "kind",
                "python_type",
                "shape",
                "dtype",
                "size",
                "n_fields",
                "numeric_min",
                "numeric_max",
                "numeric_mean",
                "n_finite",
            ]
        )

    table_path = output_dir / "Table_Redish_RRow_Mat_Structure_Probe.csv"
    summary.to_csv(table_path, index=False)

    if errors:
        errors_path = output_dir / "Table_Redish_RRow_Mat_Probe_Errors.csv"
        pd.DataFrame(errors).to_csv(errors_path, index=False)
    else:
        errors_path = None

    meta = {
        "n_input_files": len(mat_files),
        "n_successful_files": len(frames),
        "n_failed_files": len(errors),
        "n_rows": int(len(summary)),
        "max_depth": max_depth,
        "max_rows_per_file": max_rows_per_file,
        "max_object_items": max_object_items,
        "table_path": str(table_path),
        "errors_path": str(errors_path) if errors_path else None,
    }

    meta_path = output_dir / "redish_rrow_mat_probe_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")

    print(f"MAT structure probe saved: {table_path}")
    if errors_path:
        print(f"MAT probe errors saved: {errors_path}")
    print(f"Metadata saved: {meta_path}")

    return meta


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe MATLAB files from Redish RRow 2022 without MATLAB.")
    parser.add_argument("--mat-file", action="append", default=[], help="MAT file to inspect. Can be repeated.")
    parser.add_argument("--mat-glob", action="append", default=[], help="Glob expression for MAT files.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--max-rows-per-file", type=int, default=5000)
    parser.add_argument("--max-object-items", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    mat_files = [Path(p) for p in args.mat_file]
    for pattern in args.mat_glob:
        mat_files.extend(Path().glob(pattern))

    mat_files = sorted({p.resolve() for p in mat_files if p.exists()})

    write_mat_probe_outputs(
        mat_files=mat_files,
        output_dir=Path(args.output_dir),
        max_depth=args.max_depth,
        max_rows_per_file=args.max_rows_per_file,
        max_object_items=args.max_object_items,
    )


if __name__ == "__main__":
    main()