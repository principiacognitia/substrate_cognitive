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

try:
    import h5py
except ImportError:  # pragma: no cover
    h5py = None


DATASET_ID = "redish_lra_2024"

COHORT_SPECS = {
    "lra": {
        "relative_dir": Path("Processed Data") / "Processed Data" / "LRA",
        "idphi": "IdPhiData_LRA.mat",
        "lapdata": "LapData_Behav_LRA.mat",
        "session": "SessionData_LRA.mat",
        "changepoint": "ChangePointBehavAll_LRA.mat",
        "treatment_default": "control",
    },
    "mpfc_dreadds": {
        "relative_dir": Path("Processed Data") / "Processed Data" / "mPFC-DREADDs",
        "idphi": "IdPhiData_LRA_DREADDs.mat",
        "lapdata": "LapData_Behav_LRA_DREADDs.mat",
        "session": "SessionData_LRA_DREADDs.mat",
        "changepoint": "ChangePointBehav_LRA_DREADDs.mat",
        "treatment_default": "unknown_dreadd",
    },
}


def _is_hdf5(path: Path) -> bool:
    if h5py is None:
        return False
    try:
        return bool(h5py.is_hdf5(path))
    except Exception:
        return False


def _load_scipy_mat(path: Path) -> dict[str, Any]:
    raw = loadmat(
        path,
        squeeze_me=True,
        struct_as_record=False,
        simplify_cells=True,
    )
    return {k: v for k, v in raw.items() if not k.startswith("__")}


def _clean_scalar(value: Any) -> Any:
    if value is None:
        return ""

    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")

    if isinstance(value, np.generic):
        value = value.item()

    if isinstance(value, float):
        if math.isnan(value):
            return ""
        if value.is_integer():
            return int(value)
        return value

    if isinstance(value, np.ndarray):
        arr = np.asarray(value)
        if arr.size == 0:
            return ""
        if arr.size == 1:
            return _clean_scalar(arr.reshape(-1)[0])
        if arr.dtype.kind in {"U", "S"}:
            parts = [_clean_scalar(x) for x in arr.reshape(-1, order="F")]
            return "".join(str(x) for x in parts if str(x) != "")
        return str(arr.tolist())

    return value


def _as_text(value: Any) -> str:
    cleaned = _clean_scalar(value)
    if cleaned is None:
        return ""
    return str(cleaned)


def _decode_h5_char_array(arr: np.ndarray) -> str:
    arr = np.asarray(arr)

    if arr.dtype.kind in {"S", "U"}:
        return "".join(str(x) for x in arr.reshape(-1, order="F"))

    chars: list[str] = []
    for x in arr.reshape(-1, order="F"):
        try:
            code = int(x)
        except Exception:
            continue
        if code == 0:
            continue
        if 0 <= code <= 0x10FFFF:
            chars.append(chr(code))
    return "".join(chars)


def _h5_matlab_class(node: Any) -> str:
    raw = node.attrs.get("MATLAB_class", "")
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="replace")
    if isinstance(raw, np.ndarray) and raw.size:
        return _as_text(raw.reshape(-1)[0])
    return str(raw)


def _read_h5_ref_as_scalar_or_text(f: Any, ref: Any) -> Any:
    if not ref:
        return ""
    node = f[ref]

    if h5py is not None and isinstance(node, h5py.Dataset):
        arr = np.asarray(node)
        matlab_class = _h5_matlab_class(node)
        if matlab_class == "char":
            return _decode_h5_char_array(arr)
        if arr.size == 1:
            return _clean_scalar(arr.reshape(-1)[0])
        if arr.dtype.kind in {"u", "i"} and arr.size <= 256 and np.nanmax(arr) <= 65535:
            decoded = _decode_h5_char_array(arr)
            if decoded.strip():
                return decoded
        return arr

    return ""


def _read_h5_value(path: Path, logical_path: str) -> Any:
    if h5py is None:
        raise RuntimeError("h5py is required to read MATLAB v7.3 files")

    with h5py.File(path, "r") as f:
        node: Any = f
        for part in logical_path.split("."):
            if part not in node:
                return None
            node = node[part]

        if isinstance(node, h5py.Dataset):
            return np.asarray(node)

        return None


def _read_h5_string_list(path: Path, variable: str) -> list[str]:
    if h5py is None:
        return []

    with h5py.File(path, "r") as f:
        if variable not in f:
            return []

        node = f[variable]
        if isinstance(node, h5py.Dataset):
            arr = np.asarray(node)

            if arr.dtype == object:
                out: list[str] = []
                for ref in arr.reshape(-1, order="F"):
                    out.append(_as_text(_read_h5_ref_as_scalar_or_text(f, ref)))
                return out

            matlab_class = _h5_matlab_class(node)
            if matlab_class == "char":
                return [_decode_h5_char_array(arr)]

            if arr.ndim == 1:
                return [_as_text(x) for x in arr]

        return []


def _get_field(obj: Any, name: str) -> Any:
    if obj is None:
        return None

    if isinstance(obj, dict):
        return obj.get(name)

    if hasattr(obj, name):
        return getattr(obj, name)

    if isinstance(obj, np.ndarray) and obj.dtype.names and name in obj.dtype.names:
        return obj[name]

    return None


def _read_scipy_value(path: Path, logical_path: str) -> Any:
    data = _load_scipy_mat(path)
    node: Any = data
    for part in logical_path.split("."):
        node = _get_field(node, part)
        if node is None:
            return None
    return node


def _read_value(path: Path, logical_path: str) -> Any:
    if _is_hdf5(path):
        return _read_h5_value(path, logical_path)
    return _read_scipy_value(path, logical_path)


def _to_numeric_array(value: Any) -> np.ndarray:
    if value is None:
        return np.asarray([], dtype=float)

    try:
        arr = np.asarray(value, dtype=float)
    except Exception:
        return np.asarray([], dtype=float)

    return np.squeeze(arr)


def _as_session_lap_matrix(value: Any, n_sessions: int, max_laps: int = 250) -> np.ndarray:
    arr = _to_numeric_array(value)

    if arr.size == 0:
        return np.full((n_sessions, max_laps), np.nan, dtype=float)

    if arr.ndim == 0:
        return np.full((n_sessions, max_laps), float(arr), dtype=float)

    if arr.ndim == 1:
        if arr.size == n_sessions:
            return np.repeat(arr.reshape(n_sessions, 1), max_laps, axis=1)
        if arr.size == max_laps:
            return np.repeat(arr.reshape(1, max_laps), n_sessions, axis=0)

        out = np.full((n_sessions, max_laps), np.nan, dtype=float)
        n = min(arr.size, max_laps)
        out[:, :n] = arr[:n]
        return out

    if arr.ndim >= 2:
        arr2 = np.asarray(arr, dtype=float)
        if arr2.ndim > 2:
            arr2 = np.squeeze(arr2)

        if arr2.ndim != 2:
            return np.full((n_sessions, max_laps), np.nan, dtype=float)

        if arr2.shape[0] == n_sessions:
            matrix = arr2
        elif arr2.shape[1] == n_sessions:
            matrix = arr2.T
        elif arr2.shape[0] == max_laps:
            matrix = arr2.T
        else:
            matrix = arr2

        out = np.full((n_sessions, max_laps), np.nan, dtype=float)
        n_s = min(n_sessions, matrix.shape[0])
        n_l = min(max_laps, matrix.shape[1])
        out[:n_s, :n_l] = matrix[:n_s, :n_l]
        return out

    return np.full((n_sessions, max_laps), np.nan, dtype=float)


def _as_session_vector(value: Any, n_sessions: int) -> np.ndarray:
    arr = _to_numeric_array(value)

    if arr.size == 0:
        return np.full(n_sessions, np.nan, dtype=float)

    if arr.ndim == 0:
        return np.full(n_sessions, float(arr), dtype=float)

    arr = np.squeeze(arr)

    if arr.ndim == 1:
        out = np.full(n_sessions, np.nan, dtype=float)
        n = min(n_sessions, arr.size)
        out[:n] = arr[:n]
        return out

    if arr.ndim == 2:
        if arr.shape[0] == 1 and arr.shape[1] == n_sessions:
            return arr.reshape(-1)
        if arr.shape[1] == 1 and arr.shape[0] == n_sessions:
            return arr.reshape(-1)
        if arr.shape[0] == n_sessions:
            return arr[:, 0]
        if arr.shape[1] == n_sessions:
            return arr[0, :]

    return np.full(n_sessions, np.nan, dtype=float)


def _as_session_variable_matrix(value: Any, n_sessions: int) -> np.ndarray:
    arr = _to_numeric_array(value)

    if arr.size == 0:
        return np.full((n_sessions, 0), np.nan, dtype=float)

    arr = np.squeeze(arr)

    if arr.ndim == 0:
        return np.full((n_sessions, 1), float(arr), dtype=float)

    if arr.ndim == 1:
        if arr.size == n_sessions:
            return arr.reshape(n_sessions, 1)
        return np.repeat(arr.reshape(1, -1), n_sessions, axis=0)

    if arr.ndim >= 2:
        arr2 = np.asarray(arr, dtype=float)
        if arr2.shape[0] == n_sessions:
            return arr2
        if arr2.shape[1] == n_sessions:
            return arr2.T
        return arr2

    return np.full((n_sessions, 0), np.nan, dtype=float)


def _read_string_list_from_mat(path: Path, variable: str, n_sessions: int | None = None) -> list[str]:
    if _is_hdf5(path):
        values = _read_h5_string_list(path, variable)
    else:
        data = _load_scipy_mat(path)
        value = data.get(variable)
        if value is None:
            values = []
        else:
            arr = np.asarray(value, dtype=object)
            values = [_as_text(x) for x in arr.reshape(-1, order="F")]

    if n_sessions is None:
        return values

    if len(values) < n_sessions:
        values = values + [""] * (n_sessions - len(values))
    return values[:n_sessions]


def _read_session_field(path: Path, field: str, n_sessions: int) -> list[Any]:
    if not path.exists():
        return [""] * n_sessions

    try:
        if _is_hdf5(path):
            raw_values = _read_h5_string_list(path, field)
            if not raw_values:
                value = _read_h5_value(path, field)
                arr = np.asarray(value, dtype=object) if value is not None else np.asarray([])
                raw_values = [_clean_scalar(x) for x in arr.reshape(-1, order="F")]
        else:
            data = _load_scipy_mat(path)
            value = data.get(field)
            if value is None:
                return [""] * n_sessions
            arr = np.asarray(value, dtype=object)
            raw_values = [_clean_scalar(x) for x in arr.reshape(-1, order="F")]
    except Exception:
        return [""] * n_sessions

    if len(raw_values) == 1 and n_sessions > 1:
        return raw_values * n_sessions

    if len(raw_values) < n_sessions:
        raw_values = raw_values + [""] * (n_sessions - len(raw_values))

    return raw_values[:n_sessions]


def _read_numeric_matrix(path: Path, logical_path: str, n_sessions: int, max_laps: int) -> np.ndarray:
    try:
        value = _read_value(path, logical_path)
    except Exception:
        value = None
    return _as_session_lap_matrix(value, n_sessions=n_sessions, max_laps=max_laps)


def _read_numeric_vector(path: Path, logical_path: str, n_sessions: int) -> np.ndarray:
    try:
        value = _read_value(path, logical_path)
    except Exception:
        value = None
    return _as_session_vector(value, n_sessions=n_sessions)


def _flatten_numeric(value: Any) -> np.ndarray:
    arr = _to_numeric_array(value)
    if arr.size == 0:
        return np.asarray([], dtype=float)
    return np.asarray(arr, dtype=float).reshape(-1, order="F")


def _read_behavior_arrays(path: Path, n_sessions: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    entries: list[np.ndarray] = [np.asarray([], dtype=float) for _ in range(n_sessions)]
    exits: list[np.ndarray] = [np.asarray([], dtype=float) for _ in range(n_sessions)]

    if not path.exists():
        return entries, exits

    if _is_hdf5(path):
        if h5py is None:
            return entries, exits

        with h5py.File(path, "r") as f:
            if "BEHAVIOR" not in f:
                return entries, exits

            behavior = f["BEHAVIOR"]
            refs = np.asarray(behavior).reshape(-1, order="F")

            for session_index, ref in enumerate(refs[:n_sessions]):
                if not ref:
                    continue
                group = f[ref]
                if not hasattr(group, "keys"):
                    continue

                if "ChoicePointEntry" in group:
                    entries[session_index] = _flatten_numeric(np.asarray(group["ChoicePointEntry"]))

                if "ChoicePointExit" in group:
                    exits[session_index] = _flatten_numeric(np.asarray(group["ChoicePointExit"]))

        return entries, exits

    try:
        data = _load_scipy_mat(path)
        behavior = data.get("BEHAVIOR")
        if behavior is None:
            return entries, exits

        if isinstance(behavior, list):
            items = behavior
        else:
            arr = np.asarray(behavior, dtype=object)
            items = list(arr.reshape(-1, order="F"))

        for session_index, item in enumerate(items[:n_sessions]):
            entry = _get_field(item, "ChoicePointEntry")
            exit_ = _get_field(item, "ChoicePointExit")

            entries[session_index] = _flatten_numeric(entry)
            exits[session_index] = _flatten_numeric(exit_)

    except Exception:
        return entries, exits

    return entries, exits


def _parse_subject_and_date(session_id: str) -> tuple[str, str]:
    text = str(session_id)
    match = re.search(r"(R\d+)[-_](\d{4}-\d{2}-\d{2})", text)
    if match:
        return match.group(1), match.group(2)

    match = re.search(r"(R\d+)", text)
    if match:
        return match.group(1), ""

    return "", ""


def _finite_float(value: Any) -> float:
    try:
        x = float(value)
    except Exception:
        return float("nan")
    return x if math.isfinite(x) else float("nan")


def _outcome_from_correct(correct: Any) -> tuple[str, float]:
    x = _finite_float(correct)
    if not math.isfinite(x):
        return "", float("nan")
    if x >= 0.5:
        return "correct", 1.0
    return "error", 0.0


def _vte_binary(value: Any) -> float:
    x = _finite_float(value)
    if not math.isfinite(x):
        return float("nan")
    return 1.0 if x >= 0.5 else 0.0


def _nearest_switch_features(switch_laps: np.ndarray, lap: int) -> tuple[float, float, str]:
    if switch_laps.size == 0:
        return float("nan"), float("nan"), ""

    vals = np.asarray(switch_laps, dtype=float).reshape(-1)
    vals = vals[np.isfinite(vals)]
    vals = vals[vals > 0]

    if vals.size == 0:
        return float("nan"), float("nan"), ""

    nearest = vals[np.argmin(np.abs(vals - lap))]
    delta = float(lap - nearest)

    if delta < 0:
        relation = "pre_switch"
    elif delta == 0:
        relation = "switch_lap"
    else:
        relation = "post_switch"

    return float(nearest), delta, relation


def _safe_matrix_value(matrix: np.ndarray, session_index: int, lap_index: int) -> float:
    if matrix.size == 0:
        return float("nan")
    if session_index >= matrix.shape[0] or lap_index >= matrix.shape[1]:
        return float("nan")
    return _finite_float(matrix[session_index, lap_index])


def _safe_vector_value(values: np.ndarray, session_index: int) -> float:
    if values.size == 0 or session_index >= values.size:
        return float("nan")
    return _finite_float(values[session_index])


def _safe_array_lap_value(values: list[np.ndarray], session_index: int, lap_index: int) -> float:
    if session_index >= len(values):
        return float("nan")
    arr = values[session_index]
    if arr.size == 0 or lap_index >= arr.size:
        return float("nan")
    return _finite_float(arr[lap_index])


def _session_value(values: list[Any], session_index: int) -> Any:
    if session_index >= len(values):
        return ""
    return _clean_scalar(values[session_index])


def _treatment_from_dcz(value: Any, default: str) -> str:
    x = _finite_float(value)
    if math.isfinite(x):
        return "dcz" if x > 0 else "vehicle"
    if value not in ("", None):
        return _as_text(value)
    return default


def _zscore(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    finite = numeric[np.isfinite(numeric)]

    out = pd.Series(np.nan, index=values.index, dtype=float)
    if finite.empty:
        return out

    mean = float(finite.mean())
    std = float(finite.std(ddof=0))

    if not math.isfinite(std) or std <= 0:
        out.loc[finite.index] = 0.0
        return out

    out.loc[finite.index] = (finite - mean) / std
    return out


def _add_group_zscore(df: pd.DataFrame, value_col: str, group_cols: list[str], out_col: str) -> None:
    if value_col not in df.columns:
        df[out_col] = np.nan
        return

    df[out_col] = (
        df.groupby(group_cols, dropna=False, group_keys=False)[value_col]
        .apply(_zscore)
        .reindex(df.index)
    )


def _cohort_paths(root: Path, cohort: str) -> dict[str, Path]:
    spec = COHORT_SPECS[cohort]
    base = root / spec["relative_dir"]
    return {
        "base": base,
        "idphi": base / spec["idphi"],
        "lapdata": base / spec["lapdata"],
        "session": base / spec["session"],
        "changepoint": base / spec["changepoint"],
    }


def _determine_n_sessions(idphi_path: Path, lapdata_path: Path) -> int:
    session_ids = _read_string_list_from_mat(idphi_path, "SSNs")
    if session_ids:
        return len(session_ids)

    try:
        matrix = _read_value(idphi_path, "IdPhi.ChoicePoint")
        arr = _to_numeric_array(matrix)
        if arr.ndim == 2:
            return int(max(arr.shape))
    except Exception:
        pass

    try:
        correct = _read_value(lapdata_path, "Correct")
        arr = _to_numeric_array(correct)
        if arr.ndim == 2:
            return int(max(arr.shape))
    except Exception:
        pass

    return 0


def _build_cohort_rows(root: Path, cohort: str, max_laps: int) -> list[dict[str, Any]]:
    spec = COHORT_SPECS[cohort]
    paths = _cohort_paths(root, cohort)

    idphi_path = paths["idphi"]
    lapdata_path = paths["lapdata"]
    session_path = paths["session"]

    if not idphi_path.exists():
        raise FileNotFoundError(f"Missing IdPhi file for cohort {cohort}: {idphi_path}")
    if not lapdata_path.exists():
        raise FileNotFoundError(f"Missing LapData file for cohort {cohort}: {lapdata_path}")

    n_sessions = _determine_n_sessions(idphi_path, lapdata_path)
    if n_sessions <= 0:
        raise RuntimeError(f"Could not determine session count for cohort {cohort}")

    session_ids = _read_string_list_from_mat(idphi_path, "SSNs", n_sessions=n_sessions)
    if not any(session_ids):
        session_ids = [f"{cohort}_session_{idx:04d}" for idx in range(n_sessions)]

    rat_ids = _read_session_field(session_path, "RatID", n_sessions)
    sess_dates = _read_session_field(session_path, "SessDate", n_sessions)
    task_phases = _read_session_field(session_path, "TaskPhase", n_sessions)
    contingencies = _read_session_field(session_path, "Contingency", n_sessions)
    dcz_values = _read_session_field(session_path, "DCZ", n_sessions)

    idphi = _read_numeric_matrix(idphi_path, "IdPhi.ChoicePoint", n_sessions, max_laps)
    avg_idphi = _read_numeric_matrix(idphi_path, "AvgIdPhi.ChoicePoint", n_sessions, max_laps)
    vte_lap = _read_numeric_matrix(idphi_path, "VTELap.ChoicePoint", n_sessions, max_laps)

    z_idphi = _read_numeric_matrix(idphi_path, "zIdPhi.ChoicePoint", n_sessions, max_laps)
    robust_z_idphi = _read_numeric_matrix(idphi_path, "RobustZIdPhi.ChoicePoint", n_sessions, max_laps)

    p_vte = _read_numeric_vector(idphi_path, "pVTE.ChoicePoint", n_sessions)

    correct = _read_numeric_matrix(lapdata_path, "Correct", n_sessions, max_laps)
    pause_time = _read_numeric_matrix(lapdata_path, "PauseTime", n_sessions, max_laps)
    run_speed = _read_numeric_matrix(lapdata_path, "RunSpeed", n_sessions, max_laps)

    contingency_l = _read_numeric_matrix(lapdata_path, "ContingencyCorrect_L", n_sessions, max_laps)
    contingency_r = _read_numeric_matrix(lapdata_path, "ContingencyCorrect_R", n_sessions, max_laps)
    contingency_a = _read_numeric_matrix(lapdata_path, "ContingencyCorrect_A", n_sessions, max_laps)

    switch_laps_raw = _read_value(lapdata_path, "SwitchLaps")
    switch_laps = _as_session_variable_matrix(switch_laps_raw, n_sessions=n_sessions)

    entries, exits = _read_behavior_arrays(lapdata_path, n_sessions=n_sessions)

    rows: list[dict[str, Any]] = []

    for session_index in range(n_sessions):
        session_id = _as_text(session_ids[session_index])
        parsed_subject, parsed_date = _parse_subject_and_date(session_id)

        subject_id = _as_text(_session_value(rat_ids, session_index)) or parsed_subject
        session_date = _as_text(_session_value(sess_dates, session_index)) or parsed_date
        task_phase = _session_value(task_phases, session_index)
        contingency = _session_value(contingencies, session_index)
        treatment = _treatment_from_dcz(
            _session_value(dcz_values, session_index),
            default=spec["treatment_default"],
        )

        for lap_index in range(max_laps):
            lap = lap_index + 1

            lab_idphi = _safe_matrix_value(idphi, session_index, lap_index)
            lab_avg_idphi = _safe_matrix_value(avg_idphi, session_index, lap_index)
            lab_z_idphi = _safe_matrix_value(z_idphi, session_index, lap_index)
            lab_robust_z_idphi = _safe_matrix_value(robust_z_idphi, session_index, lap_index)

            lab_vte_binary = _vte_binary(_safe_matrix_value(vte_lap, session_index, lap_index))
            correct_value = _safe_matrix_value(correct, session_index, lap_index)
            outcome, reward = _outcome_from_correct(correct_value)

            entry_s = _safe_array_lap_value(entries, session_index, lap_index)
            exit_s = _safe_array_lap_value(exits, session_index, lap_index)
            dwell_s = exit_s - entry_s if math.isfinite(entry_s) and math.isfinite(exit_s) else float("nan")
            if math.isfinite(dwell_s) and dwell_s < 0:
                dwell_s = float("nan")

            nearest_switch_lap, lap_from_switch, switch_relation = _nearest_switch_features(
                switch_laps[session_index] if session_index < switch_laps.shape[0] else np.asarray([]),
                lap=lap,
            )

            row = {
                "dataset_id": DATASET_ID,
                "trace_origin": "biological",
                "task_family": "left_right_alternate",
                "cohort": cohort,
                "treatment": treatment,
                "subject_id": subject_id,
                "session_id": session_id,
                "session_index": session_index,
                "session_date": session_date,
                "trial": lap,
                "lap": lap,
                "decision_stage": "choice_point",
                "choice_point_id": "lra_choice_point",
                "chosen_action": "",
                "chosen_action_status": "not_resolved_from_aggregate_mat",
                "outcome": outcome,
                "reward": reward,
                "correct": correct_value,
                "cost": 0.0,
                "cost_context": "balanced_lra",
                "task_phase": task_phase,
                "contingency": contingency,
                "contingency_correct_l": _safe_matrix_value(contingency_l, session_index, lap_index),
                "contingency_correct_r": _safe_matrix_value(contingency_r, session_index, lap_index),
                "contingency_correct_a": _safe_matrix_value(contingency_a, session_index, lap_index),
                "lab_idphi": lab_idphi,
                "lab_avg_idphi": lab_avg_idphi,
                "lab_z_idphi": lab_z_idphi,
                "lab_robust_z_idphi": lab_robust_z_idphi,
                "lab_vte_binary": lab_vte_binary,
                "lab_pvte_session": _safe_vector_value(p_vte, session_index),
                "choice_point_entry_s": entry_s,
                "choice_point_exit_s": exit_s,
                "choice_point_dwell_s": dwell_s,
                "pause_time_s": _safe_matrix_value(pause_time, session_index, lap_index),
                "run_speed": _safe_matrix_value(run_speed, session_index, lap_index),
                "dwell_proxy": dwell_s if math.isfinite(dwell_s) else _safe_matrix_value(pause_time, session_index, lap_index),
                "deliberation_proxy": lab_idphi,
                "nearest_switch_lap": nearest_switch_lap,
                "lap_from_switch": lap_from_switch,
                "switch_relation": switch_relation,
                "source_idphi_file": str(idphi_path),
                "source_lapdata_file": str(lapdata_path),
            }
            rows.append(row)

    return rows


def _make_summary(
    df: pd.DataFrame,
    group_cols: list[str],
    output_csv: Path,
) -> pd.DataFrame:
    if df.empty:
        summary = pd.DataFrame(columns=group_cols)
        summary.to_csv(output_csv, index=False)
        return summary

    summary = (
        df.groupby(group_cols, dropna=False)
        .agg(
            n_rows=("dataset_id", "size"),
            n_subjects=("subject_id", "nunique"),
            n_sessions=("session_id", "nunique"),
            reward_rate=("reward", "mean"),
            mean_reward=("reward", "mean"),
            mean_lab_idphi=("lab_idphi", "mean"),
            mean_lab_avg_idphi=("lab_avg_idphi", "mean"),
            mean_lab_vte_binary=("lab_vte_binary", "mean"),
            mean_choice_point_dwell_s=("choice_point_dwell_s", "mean"),
            mean_pause_time_s=("pause_time_s", "mean"),
        )
        .reset_index()
    )

    summary.to_csv(output_csv, index=False)
    return summary


def build_lra_canonical_choice_endpoint(
    root: Path,
    output_dir: Path,
    cohorts: list[str] | None = None,
    max_laps: int = 250,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    if cohorts is None:
        cohorts = list(COHORT_SPECS)

    rows: list[dict[str, Any]] = []
    source_files: dict[str, dict[str, str]] = {}

    for cohort in cohorts:
        if cohort not in COHORT_SPECS:
            raise ValueError(f"Unknown cohort: {cohort}")

        paths = _cohort_paths(root, cohort)
        source_files[cohort] = {k: str(v) for k, v in paths.items() if k != "base"}

        rows.extend(_build_cohort_rows(root=root, cohort=cohort, max_laps=max_laps))

    df = pd.DataFrame(rows)

    numeric_cols = [
        "reward",
        "correct",
        "cost",
        "contingency_correct_l",
        "contingency_correct_r",
        "contingency_correct_a",
        "lab_idphi",
        "lab_avg_idphi",
        "lab_z_idphi",
        "lab_robust_z_idphi",
        "lab_vte_binary",
        "lab_pvte_session",
        "choice_point_entry_s",
        "choice_point_exit_s",
        "choice_point_dwell_s",
        "pause_time_s",
        "run_speed",
        "dwell_proxy",
        "deliberation_proxy",
        "nearest_switch_lap",
        "lap_from_switch",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    usable = df[
        df["deliberation_proxy"].notna()
        & df["reward"].notna()
        & df["lab_vte_binary"].notna()
    ].copy()

    _add_group_zscore(df, "deliberation_proxy", ["cohort", "session_id"], "z_deliberation_by_session")
    _add_group_zscore(df, "dwell_proxy", ["cohort", "session_id"], "z_dwell_by_session")
    _add_group_zscore(df, "deliberation_proxy", ["cohort"], "z_deliberation_by_cohort")
    _add_group_zscore(df, "dwell_proxy", ["cohort"], "z_dwell_by_cohort")

    _add_group_zscore(usable, "deliberation_proxy", ["cohort", "session_id"], "z_deliberation_by_session")
    _add_group_zscore(usable, "dwell_proxy", ["cohort", "session_id"], "z_dwell_by_session")
    _add_group_zscore(usable, "deliberation_proxy", ["cohort"], "z_deliberation_by_cohort")
    _add_group_zscore(usable, "dwell_proxy", ["cohort"], "z_dwell_by_cohort")

    endpoint_csv = output_dir / "redish_lra_canonical_choice_endpoint.csv"
    usable_csv = output_dir / "redish_lra_canonical_choice_endpoint_usable.csv"

    by_cohort_csv = output_dir / "Table_Redish_LRA_Canonical_By_Cohort.csv"
    by_subject_csv = output_dir / "Table_Redish_LRA_Canonical_By_Subject.csv"
    by_session_csv = output_dir / "Table_Redish_LRA_Canonical_By_Session.csv"
    by_vte_csv = output_dir / "Table_Redish_LRA_Canonical_By_VTE.csv"
    by_switch_csv = output_dir / "Table_Redish_LRA_Canonical_By_Switch_Relation.csv"

    df.to_csv(endpoint_csv, index=False)
    usable.to_csv(usable_csv, index=False)

    _make_summary(usable, ["cohort", "treatment"], by_cohort_csv)
    _make_summary(usable, ["cohort", "treatment", "subject_id"], by_subject_csv)
    _make_summary(usable, ["cohort", "treatment", "subject_id", "session_id"], by_session_csv)
    _make_summary(usable, ["cohort", "treatment", "lab_vte_binary"], by_vte_csv)
    _make_summary(usable, ["cohort", "treatment", "switch_relation"], by_switch_csv)

    meta = {
        "dataset_id": DATASET_ID,
        "root": str(root),
        "cohorts": cohorts,
        "max_laps": max_laps,
        "n_rows": int(len(df)),
        "n_usable_rows": int(len(usable)),
        "n_subjects": int(usable["subject_id"].nunique()) if not usable.empty else 0,
        "n_sessions": int(usable["session_id"].nunique()) if not usable.empty else 0,
        "n_cohorts": int(usable["cohort"].nunique()) if not usable.empty else 0,
        "source_files": source_files,
        "outputs": {
            "endpoint": str(endpoint_csv),
            "usable": str(usable_csv),
            "by_cohort": str(by_cohort_csv),
            "by_subject": str(by_subject_csv),
            "by_session": str(by_session_csv),
            "by_vte": str(by_vte_csv),
            "by_switch": str(by_switch_csv),
        },
    }

    meta_csv = output_dir / "redish_lra_canonical_choice_endpoint_meta.json"
    meta_csv.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    report = output_dir / "Redish_LRA_Canonical_Choice_Endpoint_Report.md"
    report.write_text(
        "\n".join(
            [
                "# Redish LRA 2024 canonical choice endpoint",
                "",
                f"- Dataset: `{DATASET_ID}`",
                f"- Cohorts: `{', '.join(cohorts)}`",
                f"- Rows: `{len(df)}`",
                f"- Usable rows: `{len(usable)}`",
                f"- Subjects: `{meta['n_subjects']}`",
                f"- Sessions: `{meta['n_sessions']}`",
                "",
                "## Interpretation",
                "",
                "This endpoint treats the LRA central choice point as the canonical biological choice point.",
                "The direct deliberation proxy is `IdPhi.ChoicePoint`; the lab VTE label is `VTELap.ChoicePoint`.",
                "The endpoint does not infer left/right chosen action from aggregate files unless a reliable field is later identified.",
                "",
                "## Main outputs",
                "",
                f"- `{endpoint_csv}`",
                f"- `{usable_csv}`",
                f"- `{by_cohort_csv}`",
                f"- `{by_subject_csv}`",
                f"- `{by_session_csv}`",
                f"- `{by_vte_csv}`",
                f"- `{by_switch_csv}`",
            ]
        ),
        encoding="utf-8",
    )

    print(f"Canonical LRA endpoint saved: {endpoint_csv}")
    print(f"Usable LRA endpoint saved: {usable_csv}")
    print(f"Cohort summary saved: {by_cohort_csv}")
    print(f"Subject summary saved: {by_subject_csv}")
    print(f"Session summary saved: {by_session_csv}")
    print(f"VTE summary saved: {by_vte_csv}")
    print(f"Switch summary saved: {by_switch_csv}")
    print(f"Metadata saved: {meta_csv}")
    print(f"Report saved: {report}")
    print(f"Rows: {len(df)}")
    print(f"Usable rows: {len(usable)}")

    return meta


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract Redish LRA 2024 canonical choice-point endpoint."
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Root directory of the Redish 2024 archive.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory.",
    )
    parser.add_argument(
        "--cohorts",
        nargs="+",
        choices=sorted(COHORT_SPECS),
        default=sorted(COHORT_SPECS),
        help="Cohorts to extract.",
    )
    parser.add_argument(
        "--max-laps",
        type=int,
        default=250,
        help="Maximum lap count per session.",
    )

    args = parser.parse_args()

    build_lra_canonical_choice_endpoint(
        root=args.root,
        output_dir=args.output_dir,
        cohorts=args.cohorts,
        max_laps=args.max_laps,
    )


if __name__ == "__main__":
    main()