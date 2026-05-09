from __future__ import annotations

import argparse
import csv
import json
import re
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.io import loadmat

DATASET_ID = "redish_lra_2024"

TARGET_FILES = [
    "SessionData_LRA.mat",
    "LapData_Behav_LRA.mat",
    "IdPhiData_LRA.mat",
    "ChangePointBehavAll_LRA.mat",
    "ComplexityVals_LRA.mat",
    "StereotypyData_LRA.mat",
    "RailStereotypyData_LRA.mat",
    "SessionData_LRA_DREADDs.mat",
    "LapData_Behav_LRA_DREADDs.mat",
    "IdPhiData_LRA_DREADDs.mat",
    "ChangePointBehav_LRA_DREADDs.mat",
    "StereotypyData_LRA_DREADDs.mat",
    "ExplorationData_LRA_DREADDs.mat",
]

OPTIONAL_NEURAL_FILES = [
    "ThetaDecoding_HC_LRA.mat",
    "ThetaPrediction_ToBehavior_HC_mPFC_LRA.mat",
    "ThetaScorePred_CentralPath_TopRail_HC_mPFC_LRA.mat",
    "ThetaSequenceScore_HCmPFC_LRA.mat",
    "ThetaSequenceSlope_HCmPFC_LRA.mat",
    "TaskBracketing_HCmPFCDLS_LRA.mat",
    "TaskBracketingByLap_DLS_LRA.mat",
    "AvgFRLinear_HCmPFCDLS_LRA.mat",
    "PopCorr_LapNumber_All_HCmPFCDLS_LRA.mat",
]

CONCEPT_PATTERNS = {
    "idphi_or_vte_proxy": [r"idphi", r"dphi", r"vte", r"zlog", r"head.*angle"],
    "trial_or_lap": [r"\blap\b", r"laps", r"trial", r"currentlap", r"lapnumber", r"pass"],
    "left_right_choice": [r"choice", r"decision", r"turn", r"path", r"left", r"right", r"arm", r"goal", r"route"],
    "reward_or_outcome": [r"reward", r"outcome", r"correct", r"error", r"food", r"pellet", r"pump"],
    "rule_or_contingency": [r"rule", r"conting", r"state", r"strategy", r"alternat", r"task", r"condition", r"epoch", r"block"],
    "switch_or_changepoint": [r"change", r"switch", r"transition", r"changepoint", r"reversal", r"cp"],
    "subject_or_session": [r"subject", r"rat", r"animal", r"session", r"date", r"ssn", r"directory"],
    "stereotypy_or_path_variability": [r"stereotyp", r"rail", r"path", r"variab", r"complex", r"entropy", r"speed"],
    "dreadd_or_treatment": [r"dreadd", r"cno", r"dcz", r"vehicle", r"veh", r"drug", r"injection", r"treatment"],
    "neural_optional": [r"theta", r"sequence", r"decode", r"lfp", r"spike", r"cell", r"unit", r"hc", r"mpfc", r"dls"],
}


def find_processed_root(root: Path) -> Path:
    for p in [root / "Processed Data" / "Processed Data", root / "Processed Data", root]:
        if (p / "LRA").exists() or (p / "mPFC-DREADDs").exists():
            return p
    raise FileNotFoundError("Could not find Processed Data/Processed Data with LRA or mPFC-DREADDs")


def cohort_from_path(path: Path) -> str:
    s = str(path).lower().replace("\\", "/")
    if "mpfc-dreadds" in s:
        return "mpfc_dreadds"
    if "/lra/" in s or s.endswith("/lra"):
        return "lra"
    return "unknown"


def source_table_from_path(path: Path) -> str:
    name = path.stem
    name = re.sub(r"_LRA_DREADDs$", "", name)
    name = re.sub(r"_LRA$", "", name)
    return name


def selected_files(processed_root: Path, include_neural: bool) -> list[Path]:
    patterns = TARGET_FILES + (OPTIONAL_NEURAL_FILES if include_neural else [])
    out: list[Path] = []
    for pat in patterns:
        out.extend(processed_root.rglob(pat))
    return sorted(set(out), key=lambda p: str(p).lower())


def load_mat(path: Path) -> tuple[dict[str, Any] | None, str]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            data = loadmat(path, squeeze_me=True, struct_as_record=False, simplify_cells=True)
        except MemoryError:
            return None, "MemoryError"
        except Exception as exc:
            return None, f"{type(exc).__name__}: {exc}"
    msg = "; ".join(f"{type(w.message).__name__}: {w.message}" for w in caught)
    return data, msg


def kind(v: Any) -> str:
    if isinstance(v, np.ndarray):
        return "object_ndarray" if v.dtype == object else "ndarray"
    if isinstance(v, dict):
        return "dict"
    if isinstance(v, (list, tuple)):
        return "list"
    if isinstance(v, str):
        return "str"
    if np.isscalar(v):
        return "scalar"
    if hasattr(v, "__dict__"):
        return "mat_struct"
    return type(v).__name__


def shape(v: Any) -> str:
    if isinstance(v, np.ndarray):
        return "x".join(str(x) for x in v.shape)
    if isinstance(v, (list, tuple, dict)):
        return str(len(v))
    if hasattr(v, "__dict__") and not isinstance(v, (str, bytes)):
        return str(len([k for k in vars(v) if not k.startswith("_")]))
    return ""


def dtype(v: Any) -> str:
    if isinstance(v, np.ndarray):
        return str(v.dtype)
    return type(v).__name__


def sample(v: Any, max_len: int = 140) -> str:
    if isinstance(v, (dict, list, tuple, np.ndarray)) or hasattr(v, "__dict__"):
        return ""
    try:
        text = str(v)
    except Exception:
        text = repr(type(v))
    text = text.replace("\n", " ").replace("\r", " ")
    return text if len(text) <= max_len else text[: max_len - 3] + "..."


def _format_number(value: float) -> str:
    if value is None:
        return ""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(value):
        return ""
    return f"{value:.12g}"


def numeric_stats(value: Any) -> tuple[str, str, str]:
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            return "", "", ""
        arr = value
    elif isinstance(value, (int, float, np.integer, np.floating, bool)):
        arr = np.asarray([value], dtype=float)
    else:
        return "", "", ""

    try:
        arr = np.asarray(arr, dtype=float)
    except (TypeError, ValueError):
        return "", "", ""

    if arr.size == 0:
        return "", "", "0"

    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return "", "", "0"

    return (
        _format_number(float(np.nanmin(finite))),
        _format_number(float(np.nanmax(finite))),
        str(int(finite.size)),
    )


def iter_children(v: Any, max_items: int) -> Iterable[tuple[str, Any]]:
    if isinstance(v, dict):
        for k in sorted(v.keys(), key=str):
            if not str(k).startswith("__"):
                yield str(k), v[k]
    elif hasattr(v, "__dict__") and not isinstance(v, (str, bytes)):
        for k, child in sorted(vars(v).items(), key=lambda kv: str(kv[0])):
            if not str(k).startswith("_"):
                yield str(k), child
    elif isinstance(v, (list, tuple)):
        for i, child in enumerate(v[:max_items]):
            yield f"[{i}]", child
    elif isinstance(v, np.ndarray) and v.dtype == object:
        for i, child in enumerate(v.ravel()[:max_items]):
            yield f"[{i}]", child


def probe_value(v: Any, *, source_file: Path, mat_variable: str, field_path: str, rows: list[dict[str, Any]], max_depth: int, max_items: int, depth: int = 0) -> None:
    nmin, nmax, nfinite = numeric_stats(v)
    rows.append({
        "dataset_id": DATASET_ID,
        "cohort": cohort_from_path(source_file),
        "source_file": str(source_file),
        "source_table": source_table_from_path(source_file),
        "mat_variable": mat_variable,
        "field_path": field_path,
        "kind": kind(v),
        "shape": shape(v),
        "dtype": dtype(v),
        "numeric_min": nmin,
        "numeric_max": nmax,
        "numeric_finite_count": nfinite,
        "sample": sample(v),
    })
    if depth >= max_depth:
        return
    for child_name, child in iter_children(v, max_items):
        probe_value(child, source_file=source_file, mat_variable=mat_variable, field_path=f"{field_path}.{child_name}" if field_path else child_name, rows=rows, max_depth=max_depth, max_items=max_items, depth=depth + 1)


def concept_matches(field_path: str) -> list[str]:
    s = field_path.lower()
    return [c for c, pats in CONCEPT_PATTERNS.items() if any(re.search(p, s) for p in pats)]


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys = []
        seen = set()
        for row in rows:
            for k in row:
                if k not in seen:
                    seen.add(k); keys.append(k)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)


def probe_redish_lra_dataset(root: Path, output_dir: Path, include_neural: bool = False, max_depth: int = 4, max_items_per_array: int = 20) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    processed_root = find_processed_root(root)
    files = selected_files(processed_root, include_neural)

    selected_rows = []
    probe_rows = []
    mat_warnings = {}
    for path in files:
        selected_rows.append({
            "dataset_id": DATASET_ID,
            "cohort": cohort_from_path(path),
            "source_table": source_table_from_path(path),
            "source_file": str(path),
            "relative_path": str(path.relative_to(processed_root)),
            "size_bytes": path.stat().st_size,
            "will_probe": path.suffix.lower() == ".mat",
        })
        if path.suffix.lower() != ".mat":
            continue
        data, warn = load_mat(path)
        mat_warnings[str(path)] = warn
        if data is None:
            probe_rows.append({"dataset_id": DATASET_ID, "cohort": cohort_from_path(path), "source_file": str(path), "source_table": source_table_from_path(path), "mat_variable": "", "field_path": "", "kind": "load_error", "shape": "", "dtype": "", "numeric_min": "", "numeric_max": "", "numeric_finite_count": "", "sample": warn})
            continue
        for key, value in data.items():
            if key.startswith("__"):
                continue
            probe_value(value, source_file=path, mat_variable=key, field_path=key, rows=probe_rows, max_depth=max_depth, max_items=max_items_per_array)

    pairs = defaultdict(set)
    paths = defaultdict(list)
    for row in probe_rows:
        fp = row.get("field_path", "")
        for concept in concept_matches(fp):
            pairs[concept].add((row.get("source_file", ""), fp))
            if fp not in paths[concept]:
                paths[concept].append(fp)
    precheck_rows = []
    for concept in CONCEPT_PATTERNS:
        precheck_rows.append({
            "concept": concept,
            "n_matching_fields": len(pairs.get(concept, set())),
            "n_source_files": len({x[0] for x in pairs.get(concept, set())}),
            "candidate_field_paths": "; ".join(paths.get(concept, [])[:100]),
        })

    dcz_t = list((processed_root / "mPFC-DREADDs").rglob("*.t")) if (processed_root / "mPFC-DREADDs").exists() else []
    dcz_keys = list((processed_root / "mPFC-DREADDs").rglob("*_keys.m")) if (processed_root / "mPFC-DREADDs").exists() else []

    selected_path = output_dir / "Table_Redish_LRA_Selected_Files.csv"
    probe_path = output_dir / "Table_Redish_LRA_Mat_Structure_Probe.csv"
    precheck_path = output_dir / "Table_Redish_LRA_Behavior_Endpoint_Precheck.csv"
    meta_path = output_dir / "redish_lra_2024_probe_meta.json"
    report_path = output_dir / "Redish_LRA_Probe_Report.md"
    write_csv(selected_path, selected_rows)
    write_csv(probe_path, probe_rows)
    write_csv(precheck_path, precheck_rows)

    meta = {
        "dataset_id": DATASET_ID,
        "root": str(root),
        "processed_root": str(processed_root),
        "include_neural": include_neural,
        "n_selected_files": len(selected_rows),
        "n_probe_rows": len(probe_rows),
        "n_dcz_timecourse_t_files": len(dcz_t),
        "n_dcz_timecourse_key_files": len(dcz_keys),
        "mat_warnings": mat_warnings,
        "outputs": {"selected_files": str(selected_path), "mat_structure_probe": str(probe_path), "endpoint_precheck": str(precheck_path), "report": str(report_path)},
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    report_path.write_text("# Redish LRA 2024 probe report\n\n" + json.dumps({k: v for k, v in meta.items() if k != "mat_warnings"}, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Selected files saved: {selected_path}")
    print(f"MAT structure probe saved: {probe_path}")
    print(f"Behavior endpoint precheck saved: {precheck_path}")
    print(f"Metadata saved: {meta_path}")
    print(f"Report saved: {report_path}")
    return meta


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--include-neural", action="store_true")
    p.add_argument("--max-depth", type=int, default=4)
    p.add_argument("--max-items-per-array", type=int, default=20)
    args = p.parse_args()
    probe_redish_lra_dataset(args.root, args.output_dir, args.include_neural, args.max_depth, args.max_items_per_array)


if __name__ == "__main__":
    main()
