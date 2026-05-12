from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.io import loadmat


DATASET_ID = "stout_2022_griffin_re_inactivation_vte"

TARGET_PATTERNS = [
    "tsPosOG", "xPosOG", "yPosOG",
    "VTE", "vte", "IdPhi", "idphi", "zIdPhi", "zidphi",
    "trial", "Trial", "laps", "Laps", "choice", "Choice",
    "correct", "Correct", "reward", "Reward", "outcome", "Outcome",
    "rat", "Rat", "session", "Session", "condition", "Condition",
    "oops", "OOPS", "remove", "Remove",
]

MAT_FILES_OF_INTEREST = [
    "data_oopsTrialsVTE_step1.mat",
    "data_oopsTrialsVTE_step2.mat",
    "data_vte_step2.mat",
    "data_behavior.mat",
    "data_remove.mat",
    "dataLFP_step2.mat",
]


def _shape_str(obj: Any) -> str:
    shape = getattr(obj, "shape", None)
    if shape is None:
        return ""
    return "x".join(str(int(x)) for x in shape)


def _kind(obj: Any) -> str:
    if isinstance(obj, np.ndarray):
        if obj.dtype == object:
            return "ndarray_object_or_cell"
        return "ndarray"
    if hasattr(obj, "_fieldnames"):
        return "mat_struct"
    return type(obj).__name__


def _safe_numeric_summary(obj: Any) -> dict[str, Any]:
    out = {
        "numeric_finite_count": "",
        "numeric_min": "",
        "numeric_max": "",
        "sample": "",
    }

    try:
        arr = np.asarray(obj)
    except Exception:
        return out

    if arr.dtype == object:
        flat = arr.ravel()
        samples = []
        for item in flat[:5]:
            try:
                samples.append(str(np.asarray(item).squeeze().ravel()[:3].tolist()))
            except Exception:
                samples.append(str(type(item).__name__))
        out["sample"] = "; ".join(samples)
        return out

    if not np.issubdtype(arr.dtype, np.number):
        try:
            out["sample"] = str(arr.squeeze().ravel()[:5].tolist())
        except Exception:
            pass
        return out

    try:
        flat = arr.astype(float, copy=False).ravel()
        finite = flat[np.isfinite(flat)]
        out["numeric_finite_count"] = int(finite.size)
        if finite.size:
            out["numeric_min"] = float(np.nanmin(finite))
            out["numeric_max"] = float(np.nanmax(finite))
            out["sample"] = str(finite[:5].tolist())
    except Exception:
        pass

    return out


def _summarize_variable(source_file: Path, loader: str, name: str, obj: Any) -> dict[str, Any]:
    row = {
        "dataset_id": DATASET_ID,
        "source_file": str(source_file),
        "file_name": source_file.name,
        "loader": loader,
        "variable": name,
        "kind": _kind(obj),
        "shape": _shape_str(obj),
        "dtype": str(getattr(obj, "dtype", "")),
        "fieldnames": "",
    }

    if hasattr(obj, "_fieldnames"):
        row["fieldnames"] = ";".join(str(x) for x in getattr(obj, "_fieldnames", []) or [])

    row.update(_safe_numeric_summary(obj))
    return row


def _load_mat_scipy(path: Path) -> tuple[dict[str, Any], str, str]:
    try:
        data = loadmat(path, struct_as_record=False, squeeze_me=False)
        data = {k: v for k, v in data.items() if not k.startswith("__")}
        return data, "scipy.io.loadmat", ""
    except Exception as exc:
        return {}, "load_error", f"{type(exc).__name__}: {exc}"


def _inventory(root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(root.glob("*")):
        if path.is_file() and path.suffix.lower() in {".mat", ".m"}:
            rows.append({
                "dataset_id": DATASET_ID,
                "file_name": path.name,
                "source_file": str(path),
                "suffix": path.suffix.lower(),
                "size_bytes": path.stat().st_size,
                "size_mb": round(path.stat().st_size / (1024 * 1024), 3),
                "priority": path.name in MAT_FILES_OF_INTEREST or path.suffix.lower() == ".m",
            })
    return pd.DataFrame(rows)


def _root_variables(root: Path) -> tuple[pd.DataFrame, dict[str, dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    loaded = {}
    errors = []

    for name in MAT_FILES_OF_INTEREST:
        path = root / name
        if not path.exists():
            errors.append({"file_name": name, "error": "missing"})
            continue

        data, loader, error = _load_mat_scipy(path)
        if error:
            errors.append({"file_name": name, "error": error})
            rows.append({
                "dataset_id": DATASET_ID,
                "source_file": str(path),
                "file_name": path.name,
                "loader": loader,
                "variable": "",
                "kind": "load_error",
                "shape": "",
                "dtype": "",
                "fieldnames": "",
                "numeric_finite_count": "",
                "numeric_min": "",
                "numeric_max": "",
                "sample": error,
            })
            continue

        loaded[name] = data
        for var, obj in sorted(data.items()):
            rows.append(_summarize_variable(path, loader, var, obj))

    return pd.DataFrame(rows), loaded, errors


def _flatten_cell_lengths(obj: Any) -> dict[str, Any]:
    arr = np.asarray(obj)

    result = {
        "kind": _kind(obj),
        "shape": _shape_str(obj),
        "dtype": str(getattr(arr, "dtype", "")),
        "n_items": "",
        "n_numeric_series": "",
        "total_points": "",
        "min_len": "",
        "median_len": "",
        "max_len": "",
        "first_lengths": "",
    }

    lengths = []

    if arr.dtype == object:
        items = arr.ravel()
        result["n_items"] = int(items.size)
        for item in items:
            try:
                sub = np.asarray(item).squeeze()
                if sub.size and np.issubdtype(sub.dtype, np.number):
                    lengths.append(int(sub.size))
            except Exception:
                continue
    else:
        result["n_items"] = 1
        if arr.size and np.issubdtype(arr.dtype, np.number):
            if arr.ndim <= 1:
                lengths.append(int(arr.size))
            else:
                # conservative: treat first dimension as observation axis
                lengths.append(int(arr.shape[0]))

    result["n_numeric_series"] = int(len(lengths))
    if lengths:
        result["total_points"] = int(np.sum(lengths))
        result["min_len"] = int(np.min(lengths))
        result["median_len"] = float(np.median(lengths))
        result["max_len"] = int(np.max(lengths))
        result["first_lengths"] = ";".join(str(x) for x in lengths[:20])

    return result


def _trajectory_precheck(root: Path, loaded: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows = []
    file_name = "data_oopsTrialsVTE_step1.mat"
    data = loaded.get(file_name, {})

    for var in ["tsPosOG", "xPosOG", "yPosOG"]:
        if var in data:
            row = {
                "dataset_id": DATASET_ID,
                "file_name": file_name,
                "source_file": str(root / file_name),
                "variable": var,
                "present": True,
            }
            row.update(_flatten_cell_lengths(data[var]))
        else:
            row = {
                "dataset_id": DATASET_ID,
                "file_name": file_name,
                "source_file": str(root / file_name),
                "variable": var,
                "present": False,
                "kind": "",
                "shape": "",
                "dtype": "",
                "n_items": "",
                "n_numeric_series": "",
                "total_points": "",
                "min_len": "",
                "median_len": "",
                "max_len": "",
                "first_lengths": "",
            }
        rows.append(row)

    if all(var in data for var in ["tsPosOG", "xPosOG", "yPosOG"]):
        t = np.asarray(data["tsPosOG"]).ravel()
        x = np.asarray(data["xPosOG"]).ravel()
        y = np.asarray(data["yPosOG"]).ravel()
        n = min(t.size, x.size, y.size)
        equal = 0
        checked = 0

        for i in range(n):
            try:
                lt = np.asarray(t[i]).squeeze().size
                lx = np.asarray(x[i]).squeeze().size
                ly = np.asarray(y[i]).squeeze().size
                checked += 1
                if lt == lx == ly:
                    equal += 1
            except Exception:
                pass

        rows.append({
            "dataset_id": DATASET_ID,
            "file_name": file_name,
            "source_file": str(root / file_name),
            "variable": "tsPosOG/xPosOG/yPosOG_alignment",
            "present": True,
            "kind": "triplet_alignment",
            "shape": "",
            "dtype": "",
            "n_items": int(n),
            "n_numeric_series": int(checked),
            "total_points": "",
            "min_len": "",
            "median_len": "",
            "max_len": "",
            "first_lengths": "",
            "triplets_equal_length": int(equal),
            "triplets_checked": int(checked),
            "triplet_equal_rate": float(equal / checked) if checked else math.nan,
        })

    return pd.DataFrame(rows)


def _target_candidates(root_rows: pd.DataFrame) -> pd.DataFrame:
    if root_rows.empty:
        return root_rows.copy()

    pattern = re.compile("|".join(re.escape(x) for x in TARGET_PATTERNS), flags=re.IGNORECASE)
    mask = (
        root_rows["variable"].astype(str).str.contains(pattern, regex=True, na=False)
        | root_rows["fieldnames"].astype(str).str.contains(pattern, regex=True, na=False)
    )
    return root_rows.loc[mask].copy()


def _script_hits(root: Path) -> pd.DataFrame:
    rows = []
    pattern = re.compile("|".join(re.escape(x) for x in TARGET_PATTERNS), flags=re.IGNORECASE)

    for path in sorted(root.glob("*.m")):
        try:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception:
            continue

        for line_no, line in enumerate(lines, start=1):
            hits = sorted(set(m.group(0) for m in pattern.finditer(line)))
            if hits:
                rows.append({
                    "dataset_id": DATASET_ID,
                    "script_file": str(path),
                    "file_name": path.name,
                    "line_no": line_no,
                    "hits": ";".join(hits),
                    "line": line.strip()[:500],
                })

    return pd.DataFrame(rows)


def _write_report(output_dir: Path, meta: dict[str, Any]) -> None:
    report = [
        "# Stout 2022 Griffin VTE dataset probe",
        "",
        f"Dataset id: `{DATASET_ID}`",
        "",
        "This probe inspects local MATLAB bundles and scripts only. It does not build a canonical endpoint.",
        "",
        "## Key counts",
        "",
        f"- MAT files found: {meta['n_mat_files']}",
        f"- MATLAB scripts found: {meta['n_m_scripts']}",
        f"- root variables inspected: {meta['n_root_variables']}",
        f"- target candidates: {meta['n_target_candidates']}",
        f"- script hit rows: {meta['n_script_hits']}",
        "",
        "## Interpretation policy",
        "",
        "- `dataLFP_step2.mat` is treated as neural/LFP-heavy and not a first-pass behavioral endpoint source.",
        "- `data_oopsTrialsVTE_step1.mat` is prioritized because it should contain `tsPosOG`, `xPosOG`, and `yPosOG`.",
        "- VTE labels are not trusted until the probe shows a joinable trial/session structure.",
        "- No left/right or task equivalence with the synthetic fork task is assumed.",
        "",
    ]
    (output_dir / "Stout2022_Probe_Report.md").write_text("\n".join(report), encoding="utf-8")


def run(root: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    inv = _inventory(root)
    root_rows, loaded, load_errors = _root_variables(root)
    targets = _target_candidates(root_rows)
    traj = _trajectory_precheck(root, loaded)
    scripts = _script_hits(root)

    inv_path = output_dir / "Table_Stout2022_File_Inventory.csv"
    root_path = output_dir / "Table_Stout2022_Mat_Root_Variables.csv"
    target_path = output_dir / "Table_Stout2022_Target_Field_Candidates.csv"
    traj_path = output_dir / "Table_Stout2022_Trajectory_Precheck.csv"
    script_path = output_dir / "Table_Stout2022_Script_Variable_Hits.csv"
    meta_path = output_dir / "stout2022_probe_meta.json"

    inv.to_csv(inv_path, index=False)
    root_rows.to_csv(root_path, index=False)
    targets.to_csv(target_path, index=False)
    traj.to_csv(traj_path, index=False)
    scripts.to_csv(script_path, index=False)

    meta = {
        "dataset_id": DATASET_ID,
        "root": str(root),
        "n_files": int(len(inv)),
        "n_mat_files": int((inv["suffix"] == ".mat").sum()) if not inv.empty else 0,
        "n_m_scripts": int((inv["suffix"] == ".m").sum()) if not inv.empty else 0,
        "n_root_variables": int(len(root_rows)),
        "n_target_candidates": int(len(targets)),
        "n_trajectory_precheck_rows": int(len(traj)),
        "n_script_hits": int(len(scripts)),
        "load_errors": load_errors,
        "outputs": {
            "inventory": str(inv_path),
            "root_variables": str(root_path),
            "target_candidates": str(target_path),
            "trajectory_precheck": str(traj_path),
            "script_hits": str(script_path),
            "report": str(output_dir / "Stout2022_Probe_Report.md"),
        },
    }

    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    _write_report(output_dir, meta)

    print(f"File inventory saved: {inv_path}")
    print(f"MAT root variables saved: {root_path}")
    print(f"Target candidates saved: {target_path}")
    print(f"Trajectory precheck saved: {traj_path}")
    print(f"Script variable hits saved: {script_path}")
    print(f"Metadata saved: {meta_path}")
    print(f"Report saved: {output_dir / 'Stout2022_Probe_Report.md'}")
    print(f"Root variables: {meta['n_root_variables']}")
    print(f"Target candidates: {meta['n_target_candidates']}")
    print(f"Script hits: {meta['n_script_hits']}")

    return meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    if not args.root.exists():
        raise FileNotFoundError(args.root)

    run(root=args.root, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
