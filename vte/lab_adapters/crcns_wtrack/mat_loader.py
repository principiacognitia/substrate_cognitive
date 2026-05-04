"""MATLAB v5 loader utilities for CRCNS W-track datasets.

This module is deliberately limited to file inspection and lightweight data
extraction. It does not compute VTE metrics and does not import Stage 3 internals.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from scipy.io import loadmat


MATLAB_INTERNAL_KEYS = {"__header__", "__version__", "__globals__"}


def _convert_mat_value(value: Any) -> Any:
    """Convert scipy mat_struct / object arrays into Python containers.

    scipy.io.loadmat with simplify_cells=True already handles most cases.
    This fallback keeps behavior stable on older scipy versions or unusual
    MATLAB cell/struct layouts.
    """

    if hasattr(value, "_fieldnames"):
        return {
            field: _convert_mat_value(getattr(value, field))
            for field in value._fieldnames
        }

    if isinstance(value, dict):
        return {k: _convert_mat_value(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [_convert_mat_value(v) for v in value]

    if isinstance(value, np.ndarray) and value.dtype == object:
        if value.shape == ():
            return _convert_mat_value(value.item())
        return [_convert_mat_value(v) for v in value.flat]

    return value


def read_mat(path: str | Path) -> dict[str, Any]:
    """Read a MATLAB v5 .mat file into a simplified Python dictionary."""

    mat_path = Path(path)
    if not mat_path.exists():
        raise FileNotFoundError(f"MAT file not found: {mat_path}")

    try:
        loaded = loadmat(
            mat_path,
            squeeze_me=True,
            struct_as_record=False,
            simplify_cells=True,
        )
    except TypeError:
        loaded = loadmat(
            mat_path,
            squeeze_me=True,
            struct_as_record=False,
        )

    return {
        key: _convert_mat_value(value)
        for key, value in loaded.items()
        if key not in MATLAB_INTERNAL_KEYS
    }


def top_level_keys(mat_data: dict[str, Any]) -> list[str]:
    """Return non-internal top-level MATLAB variable names."""

    return sorted(k for k in mat_data.keys() if k not in MATLAB_INTERNAL_KEYS)


def infer_animal_prefix(path: str | Path) -> str:
    """Infer animal prefix from common Frank-lab CRCNS file names.

    Examples:
    - Fivpos01.mat -> Fiv
    - Fivrawpos01.mat -> Fiv
    - Fivtask01.mat -> Fiv
    - Fiveeg01-1-01.mat -> Fiv
    """

    stem = Path(path).stem

    patterns = (
        r"^([A-Za-z]+?)rawpos\d+",
        r"^([A-Za-z]+?)pos\d+",
        r"^([A-Za-z]+?)task\d+",
        r"^([A-Za-z]+?)spikes\d+",
        r"^([A-Za-z]+?)cellinfo$",
        r"^([A-Za-z]+?)tetinfo$",
        r"^([A-Za-z]+?)eeg\d+",
    )

    for pattern in patterns:
        match = re.match(pattern, stem, flags=re.IGNORECASE)
        if match:
            return match.group(1)

    stripped = re.sub(r"[^A-Za-z].*$", "", stem)
    return stripped or Path(path).parent.name


def infer_day_from_filename(path: str | Path) -> int | None:
    """Infer recording day from common CRCNS W-track file names."""

    stem = Path(path).stem.lower()

    match = re.search(r"(?:rawpos|pos|task|spikes)(\d{2})", stem)
    if match:
        return int(match.group(1))

    match = re.search(r"eeg(\d{2})", stem)
    if match:
        return int(match.group(1))

    return None


def infer_file_kind(path: str | Path) -> str:
    """Classify a CRCNS W-track file by filename."""

    p = Path(path)
    stem = p.stem.lower()
    parent = p.parent.name.lower()

    if parent == "eeg" or "eeg" in stem:
        return "eeg"
    if "rawpos" in stem:
        return "rawpos"
    if "pos" in stem:
        return "pos"
    if "task" in stem:
        return "task"
    if "spikes" in stem:
        return "spikes"
    if stem.endswith("cellinfo") or stem.endswith("tetinfo"):
        return "metadata"

    return "other"


def list_animal_mat_files(animal_dir: str | Path) -> list[Path]:
    """List all .mat files under an animal directory."""

    root = Path(animal_dir)
    if not root.exists():
        raise FileNotFoundError(f"Animal directory not found: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Animal path is not a directory: {root}")

    return sorted(p.resolve() for p in root.rglob("*.mat"))


def _shape_of(value: Any) -> str:
    if isinstance(value, np.ndarray):
        return "x".join(str(x) for x in value.shape)
    return ""


def summarize_mat_top_level(path: str | Path, *, inspect: bool = True) -> dict[str, Any]:
    """Summarize top-level variables in a MATLAB file.

    Set inspect=False for large files, especially EEG, when only path-level
    inventory is needed.
    """

    mat_path = Path(path)
    size_bytes = mat_path.stat().st_size if mat_path.exists() else None
    file_kind = infer_file_kind(mat_path)
    day = infer_day_from_filename(mat_path)

    summary: dict[str, Any] = {
        "filename": mat_path.name,
        "path": str(mat_path),
        "file_kind": file_kind,
        "animal_id": infer_animal_prefix(mat_path),
        "day": day,
        "size_bytes": size_bytes,
        "inspected": bool(inspect),
        "top_keys": "",
        "top_key_count": 0,
        "root_type": "",
        "root_shape": "",
    }

    if not inspect:
        summary["inspected"] = False
        summary["root_type"] = "not_inspected"
        return summary

    mat_data = read_mat(mat_path)
    keys = top_level_keys(mat_data)

    summary["top_keys"] = ";".join(keys)
    summary["top_key_count"] = len(keys)

    if len(keys) == 1:
        root_value = mat_data[keys[0]]
        summary["root_type"] = type(root_value).__name__
        summary["root_shape"] = _shape_of(root_value)
    elif len(keys) > 1:
        summary["root_type"] = "multi"

    return summary


def walk_nested(obj: Any, prefix: str = "", max_depth: int = 8) -> Iterator[tuple[str, Any]]:
    """Yield leaf-like nested values from dict/list/object-array structures."""

    if max_depth < 0:
        return

    if isinstance(obj, dict):
        for key, value in obj.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            yield from walk_nested(value, child, max_depth=max_depth - 1)
        return

    if isinstance(obj, (list, tuple)):
        for idx, value in enumerate(obj):
            child = f"{prefix}[{idx}]" if prefix else f"[{idx}]"
            yield from walk_nested(value, child, max_depth=max_depth - 1)
        return

    if isinstance(obj, np.ndarray) and obj.dtype == object:
        for idx, value in enumerate(obj.flat):
            child = f"{prefix}[{idx}]" if prefix else f"[{idx}]"
            yield from walk_nested(value, child, max_depth=max_depth - 1)
        return

    yield prefix, obj
