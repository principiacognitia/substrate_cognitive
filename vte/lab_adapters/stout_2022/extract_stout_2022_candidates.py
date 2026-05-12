from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.io import loadmat


DATASET_ID = "stout_2022_griffin_re_inactivation_vte"


def _load(path: Path) -> dict[str, Any]:
    return {
        k: v
        for k, v in loadmat(path, struct_as_record=False, squeeze_me=True).items()
        if not k.startswith("__")
    }


def _is_struct(x: Any) -> bool:
    return hasattr(x, "_fieldnames")


def _unwrap(x: Any) -> Any:
    while isinstance(x, np.ndarray) and x.dtype == object and x.size == 1:
        x = x.item()
    return x


def _shape(x: Any) -> str:
    shp = getattr(x, "shape", None)
    if shp is None:
        return ""
    return "x".join(str(int(v)) for v in shp)


def _numeric_vector(x: Any) -> np.ndarray:
    x = _unwrap(x)
    try:
        arr = np.asarray(x)
    except Exception:
        return np.asarray([], dtype=float)

    if arr.dtype == object:
        if arr.size == 1:
            return _numeric_vector(arr.item())
        vals = []
        for item in arr.ravel():
            try:
                sub = np.asarray(_unwrap(item), dtype=float).ravel()
                if sub.size == 1:
                    vals.append(float(sub[0]))
                else:
                    vals.append(np.nan)
            except Exception:
                vals.append(np.nan)
        return np.asarray(vals, dtype=float)

    try:
        return np.asarray(arr, dtype=float).ravel()
    except Exception:
        return np.asarray([], dtype=float)


def _safe_get(root: Any, path: tuple[str, ...]) -> Any | None:
    x = _unwrap(root)
    for part in path:
        x = _unwrap(x)
        if not _is_struct(x):
            return None
        if not hasattr(x, part):
            return None
        x = getattr(x, part)
    return _unwrap(x)


def _collect_numeric_leaves(x: Any, path: tuple[str, ...] = ()) -> dict[tuple[str, ...], Any]:
    x = _unwrap(x)
    out: dict[tuple[str, ...], Any] = {}

    if _is_struct(x):
        for field in x._fieldnames or []:
            out.update(_collect_numeric_leaves(getattr(x, field), path + (str(field),)))
        return out

    arr = np.asarray(x)
    if arr.dtype == object:
        if arr.size == 1:
            out.update(_collect_numeric_leaves(arr.item(), path))
        return out

    if np.issubdtype(arr.dtype, np.number):
        out[path] = arr

    return out


def _summarize_leaf(file_name: str, variable: str, path: tuple[str, ...], value: Any) -> dict[str, Any]:
    arr = _numeric_vector(value)
    finite = arr[np.isfinite(arr)] if arr.size else np.asarray([], dtype=float)

    return {
        "dataset_id": DATASET_ID,
        "file_name": file_name,
        "root_variable": variable,
        "logical_path": ".".join(path),
        "path_depth": len(path),
        "kind": type(_unwrap(value)).__name__,
        "shape": _shape(value),
        "n_values": int(arr.size),
        "n_finite": int(finite.size),
        "coverage": float(finite.size / arr.size) if arr.size else math.nan,
        "numeric_min": float(np.nanmin(finite)) if finite.size else math.nan,
        "numeric_max": float(np.nanmax(finite)) if finite.size else math.nan,
        "sample": ";".join(str(float(x)) for x in finite[:5]) if finite.size else "",
    }


def _field_inventory(root: Path, loaded: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for file_name, data in loaded.items():
        for variable, value in sorted(data.items()):
            leaves = _collect_numeric_leaves(value)
            if not leaves:
                rows.append({
                    "dataset_id": DATASET_ID,
                    "file_name": file_name,
                    "root_variable": variable,
                    "logical_path": "",
                    "path_depth": 0,
                    "kind": type(_unwrap(value)).__name__,
                    "shape": _shape(value),
                    "n_values": 0,
                    "n_finite": 0,
                    "coverage": math.nan,
                    "numeric_min": math.nan,
                    "numeric_max": math.nan,
                    "sample": "",
                })
            else:
                for path, leaf in sorted(leaves.items()):
                    rows.append(_summarize_leaf(file_name, variable, path, leaf))
    return pd.DataFrame(rows)


def _value_at(vec: np.ndarray, idx: int) -> float:
    if idx < 0 or idx >= vec.size:
        return math.nan
    try:
        return float(vec[idx])
    except Exception:
        return math.nan


def _trajectory_length(root: Any, path: tuple[str, ...], idx: int) -> float:
    x = _safe_get(root, path)
    if x is None:
        return math.nan

    x = _unwrap(x)
    arr = np.asarray(x)

    try:
        if arr.dtype == object:
            flat = arr.ravel()
            if idx >= flat.size:
                return math.nan
            item = _unwrap(flat[idx])
            return float(np.asarray(item).size)

        if arr.ndim == 1:
            return float(arr.size) if idx == 0 else math.nan

        if arr.ndim >= 2:
            if idx < arr.shape[0]:
                return float(np.asarray(arr[idx]).size)
            if idx < arr.shape[-1]:
                return float(np.asarray(arr[..., idx]).size)
    except Exception:
        return math.nan

    return math.nan


def _status_from_finite(*vals: float) -> str:
    return "usable" if all(np.isfinite(v) for v in vals) else "partial"


def _endpoint(root: Path, loaded: dict[str, dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    step1 = loaded.get("data_oopsTrialsVTE_step1.mat", {})
    step2 = loaded.get("data_oopsTrialsVTE_step2.mat", {})
    vte2 = loaded.get("data_vte_step2.mat", {})
    beh = loaded.get("data_behavior.mat", {})
    rem = loaded.get("data_remove.mat", {})

    z_root = vte2.get("zIdPhi")
    if z_root is None:
        return pd.DataFrame(), pd.DataFrame()

    z_leaves = _collect_numeric_leaves(z_root)

    rows = []
    align_rows = []

    for path, z_arr_raw in sorted(z_leaves.items()):
        if len(path) < 3:
            continue

        subject_id = path[0]
        condition = path[1]
        session_label = path[2]
        session_id = f"{subject_id}_{condition}_{session_label}"

        z = _numeric_vector(z_arr_raw)
        n = int(z.size)

        idphi = _numeric_vector(_safe_get(step1.get("IdPhi"), path))
        oops_step2 = _numeric_vector(_safe_get(step2.get("oopsTrials"), path))
        oops_step1 = _numeric_vector(_safe_get(step1.get("oopsTrials"), path))
        oops = oops_step2 if oops_step2.size else oops_step1

        accuracy = _numeric_vector(_safe_get(beh.get("accuracy"), path))
        turn = _numeric_vector(_safe_get(beh.get("turnDirection"), path))
        time_spent_cp = _numeric_vector(_safe_get(beh.get("timeSpent_CP"), path))
        time_spent = _numeric_vector(_safe_get(step1.get("timeSpent"), path))
        rem_trials = _numeric_vector(_safe_get(rem.get("remTrials"), path))

        n_max = max(
            n,
            idphi.size,
            oops.size,
            accuracy.size,
            turn.size,
            time_spent_cp.size,
            time_spent.size,
        )

        align_rows.append({
            "dataset_id": DATASET_ID,
            "subject_id": subject_id,
            "condition": condition,
            "session_label": session_label,
            "session_id": session_id,
            "path": ".".join(path),
            "n_zidphi": int(z.size),
            "n_idphi": int(idphi.size),
            "n_oops": int(oops.size),
            "n_accuracy": int(accuracy.size),
            "n_turn_direction": int(turn.size),
            "n_time_spent_cp": int(time_spent_cp.size),
            "n_time_spent": int(time_spent.size),
            "n_rem_trials": int(rem_trials.size),
            "n_endpoint_rows": int(n_max),
        })

        for idx in range(n_max):
            trial = idx + 1

            z_val = _value_at(z, idx)
            idphi_val = _value_at(idphi, idx)
            oops_val = _value_at(oops, idx)
            acc_val = _value_at(accuracy, idx)
            turn_val = _value_at(turn, idx)
            cp_val = _value_at(time_spent_cp, idx)
            time_val = _value_at(time_spent, idx)

            idphi_vte = float(z_val > 0.0) if np.isfinite(z_val) else math.nan
            oops_vte = float(oops_val > 0.0) if np.isfinite(oops_val) else math.nan

            if np.isfinite(idphi_vte) or np.isfinite(oops_vte):
                combined_vte = float(
                    (np.isfinite(idphi_vte) and idphi_vte > 0)
                    or (np.isfinite(oops_vte) and oops_vte > 0)
                )
            else:
                combined_vte = math.nan

            reward = acc_val if np.isfinite(acc_val) else math.nan
            if np.isfinite(reward):
                outcome = "correct" if reward > 0 else "error"
            else:
                outcome = ""

            dwell = cp_val if np.isfinite(cp_val) else time_val

            removed_by_author = False
            if rem_trials.size:
                removed_by_author = bool(np.any(np.isclose(rem_trials, trial, equal_nan=False)))

            tx_len = _trajectory_length(step1.get("tsPosOG"), path, idx)
            x_len = _trajectory_length(step1.get("xPosOG"), path, idx)
            y_len = _trajectory_length(step1.get("yPosOG"), path, idx)

            rows.append({
                "source": "biological",
                "dataset_id": DATASET_ID,
                "task_family": "dnmp_re_inactivation_vte",
                "subject_id": subject_id,
                "condition": condition,
                "session_label": session_label,
                "session_id": session_id,
                "trial": trial,
                "decision_stage": "choice_point",
                "outcome": outcome,
                "reward": reward,
                "z_idphi": z_val,
                "lab_idphi": idphi_val,
                "vte_binary_oops": oops_vte,
                "vte_binary_idphi_positive": idphi_vte,
                "vte_binary_combined": combined_vte,
                "turn_direction_raw": turn_val,
                "chosen_action": f"turn_code_{int(turn_val)}" if np.isfinite(turn_val) else "",
                "dwell_proxy": dwell,
                "time_spent_cp": cp_val,
                "time_spent": time_val,
                "trajectory_ts_len": tx_len,
                "trajectory_x_len": x_len,
                "trajectory_y_len": y_len,
                "trajectory_len_match": bool(
                    np.isfinite(tx_len)
                    and np.isfinite(x_len)
                    and np.isfinite(y_len)
                    and tx_len == x_len == y_len
                ),
                "removed_by_author_script": removed_by_author,
                "endpoint_status": _status_from_finite(combined_vte, reward),
                "biological_comparison_role": "perturbation_dataset_candidate",
                "comparability_policy": "not_direct_lra_or_synthetic_task_match",
                "source_path": ".".join(path),
            })

    return pd.DataFrame(rows), pd.DataFrame(align_rows)


def _coverage(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    total = len(df)
    for col in df.columns:
        nonempty = int(df[col].notna().sum())
        if df[col].dtype == object:
            nonempty = int((df[col].astype(str) != "").sum())
        rows.append({
            "field": col,
            "nonempty": nonempty,
            "total": total,
            "coverage": float(nonempty / total) if total else math.nan,
        })
    return pd.DataFrame(rows)


def _summaries(endpoint: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if endpoint.empty:
        return {
            "by_condition": pd.DataFrame(),
            "by_vte": pd.DataFrame(),
            "by_status": pd.DataFrame(),
        }

    usable = endpoint[endpoint["endpoint_status"] == "usable"].copy()

    by_condition = usable.groupby(
        ["condition", "session_label"], dropna=False
    ).agg(
        n_rows=("trial", "size"),
        n_subjects=("subject_id", "nunique"),
        n_sessions=("session_id", "nunique"),
        reward_rate=("reward", "mean"),
        vte_rate_combined=("vte_binary_combined", "mean"),
        oops_vte_rate=("vte_binary_oops", "mean"),
        idphi_positive_rate=("vte_binary_idphi_positive", "mean"),
        mean_z_idphi=("z_idphi", "mean"),
        mean_lab_idphi=("lab_idphi", "mean"),
        mean_dwell_proxy=("dwell_proxy", "mean"),
    ).reset_index()

    by_vte = usable.groupby(
        ["vte_binary_combined"], dropna=False
    ).agg(
        n_rows=("trial", "size"),
        n_subjects=("subject_id", "nunique"),
        n_sessions=("session_id", "nunique"),
        reward_rate=("reward", "mean"),
        mean_z_idphi=("z_idphi", "mean"),
        mean_lab_idphi=("lab_idphi", "mean"),
        mean_dwell_proxy=("dwell_proxy", "mean"),
    ).reset_index()

    by_status = endpoint.groupby(
        ["endpoint_status", "removed_by_author_script"], dropna=False
    ).agg(
        n_rows=("trial", "size"),
        n_subjects=("subject_id", "nunique"),
        n_sessions=("session_id", "nunique"),
    ).reset_index()

    return {
        "by_condition": by_condition,
        "by_vte": by_vte,
        "by_status": by_status,
    }


def _write_report(output_dir: Path, meta: dict[str, Any]) -> None:
    lines = [
        "# Stout 2022 candidate extraction",
        "",
        f"Dataset id: `{DATASET_ID}`",
        "",
        "This extraction resolves MATLAB rat/condition/session structs and writes a trial-level candidate endpoint.",
        "",
        "The endpoint is diagnostic. It is not declared directly comparable to Redish LRA or the synthetic fork task.",
        "",
        "## Counts",
        "",
        f"- deep field rows: {meta['n_deep_field_rows']}",
        f"- session alignment rows: {meta['n_session_alignment_rows']}",
        f"- endpoint rows: {meta['n_endpoint_rows']}",
        f"- usable endpoint rows: {meta['n_usable_endpoint_rows']}",
        "",
        "## Policy",
        "",
        "- Re-inactivation, saline, and muscimol conditions are preserved.",
        "- Combined VTE is reconstructed as `oopsTrials > 0 OR zIdPhi > 0`, following the manuscript script logic.",
        "- Raw turn-direction codes are retained but not mapped to left/right.",
        "- Author removal markers are retained as `removed_by_author_script`.",
    ]
    (output_dir / "Stout2022_Candidate_Extraction_Report.md").write_text("\n".join(lines), encoding="utf-8")


def run(root: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    files = {
        "data_oopsTrialsVTE_step1.mat": _load(root / "data_oopsTrialsVTE_step1.mat"),
        "data_oopsTrialsVTE_step2.mat": _load(root / "data_oopsTrialsVTE_step2.mat"),
        "data_vte_step2.mat": _load(root / "data_vte_step2.mat"),
        "data_behavior.mat": _load(root / "data_behavior.mat"),
        "data_remove.mat": _load(root / "data_remove.mat"),
    }

    deep = _field_inventory(root, files)
    endpoint, alignment = _endpoint(root, files)
    coverage = _coverage(endpoint)
    summaries = _summaries(endpoint)

    deep_path = output_dir / "Table_Stout2022_Deep_Field_Inventory.csv"
    align_path = output_dir / "Table_Stout2022_Session_Path_Alignment.csv"
    endpoint_path = output_dir / "stout2022_trial_endpoint_candidate.csv"
    coverage_path = output_dir / "Table_Stout2022_Endpoint_Field_Coverage.csv"
    by_condition_path = output_dir / "Table_Stout2022_By_Condition.csv"
    by_vte_path = output_dir / "Table_Stout2022_By_VTE.csv"
    by_status_path = output_dir / "Table_Stout2022_By_Status.csv"
    meta_path = output_dir / "stout2022_candidate_extraction_meta.json"

    deep.to_csv(deep_path, index=False)
    alignment.to_csv(align_path, index=False)
    endpoint.to_csv(endpoint_path, index=False)
    coverage.to_csv(coverage_path, index=False)
    summaries["by_condition"].to_csv(by_condition_path, index=False)
    summaries["by_vte"].to_csv(by_vte_path, index=False)
    summaries["by_status"].to_csv(by_status_path, index=False)

    meta = {
        "dataset_id": DATASET_ID,
        "root": str(root),
        "patch": "19B",
        "policy": "Resolve MATLAB structs and write diagnostic trial-level endpoint; no direct task comparability assumed.",
        "vte_reconstruction": "combined_vte = (oopsTrials > 0) OR (zIdPhi > 0)",
        "n_deep_field_rows": int(len(deep)),
        "n_session_alignment_rows": int(len(alignment)),
        "n_endpoint_rows": int(len(endpoint)),
        "n_usable_endpoint_rows": int((endpoint["endpoint_status"] == "usable").sum()) if not endpoint.empty else 0,
        "outputs": {
            "deep_field_inventory": str(deep_path),
            "session_alignment": str(align_path),
            "endpoint_candidate": str(endpoint_path),
            "coverage": str(coverage_path),
            "by_condition": str(by_condition_path),
            "by_vte": str(by_vte_path),
            "by_status": str(by_status_path),
            "report": str(output_dir / "Stout2022_Candidate_Extraction_Report.md"),
        },
    }

    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    _write_report(output_dir, meta)

    print(f"Deep field inventory saved: {deep_path}")
    print(f"Session alignment saved: {align_path}")
    print(f"Candidate endpoint saved: {endpoint_path}")
    print(f"Coverage saved: {coverage_path}")
    print(f"Condition summary saved: {by_condition_path}")
    print(f"VTE summary saved: {by_vte_path}")
    print(f"Status summary saved: {by_status_path}")
    print(f"Metadata saved: {meta_path}")
    print(f"Report saved: {output_dir / 'Stout2022_Candidate_Extraction_Report.md'}")
    print(f"Endpoint rows: {meta['n_endpoint_rows']}")
    print(f"Usable endpoint rows: {meta['n_usable_endpoint_rows']}")

    return meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    run(args.root, args.output_dir)


if __name__ == "__main__":
    main()
