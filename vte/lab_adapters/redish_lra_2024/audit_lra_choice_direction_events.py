from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from vte.lab_adapters.redish_lra_2024.extract_canonical_choice_endpoint import (
    COHORT_SPECS,
    DATASET_ID,
    _as_text,
    _cohort_paths,
    _determine_n_sessions,
    _finite_float,
    _flatten_numeric,
    _get_field,
    _is_hdf5,
    _load_scipy_mat,
    _read_behavior_arrays,
    _read_numeric_matrix,
    _read_string_list_from_mat,
    h5py,
)


def _event_code_label(value: Any) -> str:
    x = _finite_float(value)
    if not math.isfinite(x):
        return ""
    if float(x).is_integer():
        return f"code_{int(x)}"
    return f"code_{x:g}"


def _read_behavior_field_arrays(path: Path, n_sessions: int, field: str) -> list[np.ndarray]:
    out: list[np.ndarray] = [np.asarray([], dtype=float) for _ in range(n_sessions)]

    if not path.exists():
        return out

    if _is_hdf5(path):
        if h5py is None:
            return out

        with h5py.File(path, "r") as f:
            if "BEHAVIOR" not in f:
                return out

            refs = np.asarray(f["BEHAVIOR"]).reshape(-1, order="F")

            for session_index, ref in enumerate(refs[:n_sessions]):
                if not ref:
                    continue

                group = f[ref]
                if not hasattr(group, "keys"):
                    continue

                if field in group:
                    out[session_index] = _flatten_numeric(np.asarray(group[field]))

        return out

    try:
        data = _load_scipy_mat(path)
        behavior = data.get("BEHAVIOR")
        if behavior is None:
            return out

        if isinstance(behavior, list):
            items = behavior
        else:
            items = list(np.asarray(behavior, dtype=object).reshape(-1, order="F"))

        for session_index, item in enumerate(items[:n_sessions]):
            value = _get_field(item, field)
            out[session_index] = _flatten_numeric(value)

    except Exception:
        return out

    return out


def _align_codes_to_times(times: np.ndarray, codes: np.ndarray) -> np.ndarray:
    times = np.asarray(times, dtype=float).reshape(-1)
    codes = np.asarray(codes, dtype=float).reshape(-1)

    if times.size == 0:
        return np.asarray([], dtype=float)

    out = np.full(times.size, np.nan, dtype=float)
    n = min(times.size, codes.size)
    if n > 0:
        out[:n] = codes[:n]
    return out


def _nearest_after(
    start_time_s: Any,
    event_times: np.ndarray,
    event_codes: np.ndarray,
    max_latency_s: float,
) -> dict[str, Any]:
    start = _finite_float(start_time_s)
    if not math.isfinite(start):
        return {
            "event_time_s": np.nan,
            "event_code": np.nan,
            "event_latency_s": np.nan,
            "match_status": "missing_choice_exit",
        }

    times = np.asarray(event_times, dtype=float).reshape(-1)
    codes = _align_codes_to_times(times, event_codes)

    finite = np.isfinite(times)
    if not finite.any():
        return {
            "event_time_s": np.nan,
            "event_code": np.nan,
            "event_latency_s": np.nan,
            "match_status": "no_events",
        }

    times = times[finite]
    codes = codes[finite]

    latency = times - start
    after = latency >= 0
    within = after & (latency <= max_latency_s)

    if within.any():
        idx = int(np.argmin(latency[within]))
        source_indices = np.where(within)[0]
        chosen = int(source_indices[idx])
        return {
            "event_time_s": float(times[chosen]),
            "event_code": float(codes[chosen]) if math.isfinite(_finite_float(codes[chosen])) else np.nan,
            "event_latency_s": float(latency[chosen]),
            "match_status": "matched_after_choice_exit",
        }

    nearest_idx = int(np.argmin(np.abs(latency)))
    return {
        "event_time_s": float(times[nearest_idx]),
        "event_code": float(codes[nearest_idx]) if math.isfinite(_finite_float(codes[nearest_idx])) else np.nan,
        "event_latency_s": float(latency[nearest_idx]),
        "match_status": "nearest_outside_window",
    }


def _safe_array_lap_value(values: list[np.ndarray], session_index: int, lap_index: int) -> float:
    if session_index < 0 or session_index >= len(values):
        return np.nan
    arr = values[session_index]
    if arr.size == 0 or lap_index < 0 or lap_index >= arr.size:
        return np.nan
    return _finite_float(arr[lap_index])


def _safe_matrix_value(matrix: np.ndarray, session_index: int, lap_index: int) -> float:
    if matrix.size == 0:
        return np.nan
    if session_index < 0 or lap_index < 0:
        return np.nan
    if session_index >= matrix.shape[0] or lap_index >= matrix.shape[1]:
        return np.nan
    return _finite_float(matrix[session_index, lap_index])


def _primary_event_kind(reward: Any) -> str:
    r = _finite_float(reward)
    if not math.isfinite(r):
        return ""
    if r >= 0.5:
        return "feeder_fired"
    return "error_not_fired"


def _primary_from_reward(
    reward: Any,
    feeder_match: dict[str, Any],
    error_match: dict[str, Any],
) -> dict[str, Any]:
    kind = _primary_event_kind(reward)
    if kind == "feeder_fired":
        m = feeder_match
    elif kind == "error_not_fired":
        m = error_match
    else:
        return {
            "primary_event_kind": "",
            "primary_event_time_s": np.nan,
            "primary_event_code": np.nan,
            "primary_event_code_label": "",
            "primary_event_latency_s": np.nan,
            "primary_event_match_status": "missing_reward",
        }

    return {
        "primary_event_kind": kind,
        "primary_event_time_s": m["event_time_s"],
        "primary_event_code": m["event_code"],
        "primary_event_code_label": _event_code_label(m["event_code"]),
        "primary_event_latency_s": m["event_latency_s"],
        "primary_event_match_status": m["match_status"],
    }


def _coverage_table(df: pd.DataFrame, fields: list[str]) -> pd.DataFrame:
    rows = []
    total = len(df)
    for field in fields:
        if field not in df.columns:
            rows.append({"field": field, "nonempty": 0, "total": total, "coverage": 0.0})
            continue

        s = df[field]
        nonempty = int(s.notna().sum())
        if s.dtype == object:
            nonempty = int(s.map(lambda x: str(x).strip() != "" and str(x).lower() != "nan").sum())

        rows.append(
            {
                "field": field,
                "nonempty": nonempty,
                "total": total,
                "coverage": nonempty / total if total else 0.0,
            }
        )
    return pd.DataFrame(rows)


def _summary_by_code(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    work = df.copy()
    for col in [
        "reward",
        "lab_idphi",
        "vte_binary_for_comparison",
        "primary_event_latency_s",
        "feeder_event_latency_s",
        "error_event_latency_s",
    ]:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")

    return (
        work.groupby(
            [
                "primary_event_kind",
                "primary_event_code_label",
                "primary_event_match_status",
                "outcome",
            ],
            dropna=False,
        )
        .agg(
            n_rows=("dataset_id", "size"),
            n_subjects=("subject_id", "nunique"),
            n_sessions=("session_id", "nunique"),
            reward_rate=("reward", "mean"),
            vte_rate=("vte_binary_for_comparison", "mean"),
            mean_lab_idphi=("lab_idphi", "mean"),
            mean_primary_event_latency_s=("primary_event_latency_s", "mean"),
        )
        .reset_index()
    )


def _summary_crosswalk(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    work = df.copy()
    for col in ["reward", "lab_idphi", "vte_binary_for_comparison"]:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")

    return (
        work.groupby(
            [
                "feeder_event_code_label",
                "error_event_code_label",
                "outcome",
            ],
            dropna=False,
        )
        .agg(
            n_rows=("dataset_id", "size"),
            n_subjects=("subject_id", "nunique"),
            n_sessions=("session_id", "nunique"),
            reward_rate=("reward", "mean"),
            vte_rate=("vte_binary_for_comparison", "mean"),
            mean_lab_idphi=("lab_idphi", "mean"),
        )
        .reset_index()
    )


def audit_lra_choice_direction_events(
    root: Path,
    healthy_control_csv: Path,
    output_dir: Path,
    max_latency_s: float = 90.0,
    max_laps: int = 250,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    control = pd.read_csv(healthy_control_csv, low_memory=False)

    paths = _cohort_paths(root, "lra")
    idphi_path = paths["idphi"]
    lapdata_path = paths["lapdata"]

    n_sessions = _determine_n_sessions(idphi_path, lapdata_path)
    session_ids = _read_string_list_from_mat(idphi_path, "SSNs", n_sessions=n_sessions)
    session_index_by_id = {_as_text(session_id): idx for idx, session_id in enumerate(session_ids)}

    entries, exits = _read_behavior_arrays(lapdata_path, n_sessions=n_sessions)

    feeder_times = _read_behavior_field_arrays(lapdata_path, n_sessions, "FeederTime")
    feeders_fired = _read_behavior_field_arrays(lapdata_path, n_sessions, "FeedersFired")
    error_times = _read_behavior_field_arrays(lapdata_path, n_sessions, "ErrorTime")
    error_not_fired = _read_behavior_field_arrays(lapdata_path, n_sessions, "ErrorNotFired")

    contingency_l = _read_numeric_matrix(lapdata_path, "ContingencyCorrect_L", n_sessions, max_laps)
    contingency_r = _read_numeric_matrix(lapdata_path, "ContingencyCorrect_R", n_sessions, max_laps)
    contingency_a = _read_numeric_matrix(lapdata_path, "ContingencyCorrect_A", n_sessions, max_laps)

    rows: list[dict[str, Any]] = []

    for _, row in control.iterrows():
        session_id = _as_text(row.get("session_id", ""))

        session_index = _finite_float(row.get("session_index", np.nan))
        if not math.isfinite(session_index):
            session_index = session_index_by_id.get(session_id, np.nan)

        if not math.isfinite(session_index):
            continue

        session_index_i = int(session_index)

        trial = _finite_float(row.get("trial", np.nan))
        if not math.isfinite(trial):
            continue
        lap_index = int(trial) - 1

        choice_entry_s = _safe_array_lap_value(entries, session_index_i, lap_index)
        choice_exit_s = _safe_array_lap_value(exits, session_index_i, lap_index)

        feeder_match = _nearest_after(
            choice_exit_s,
            feeder_times[session_index_i],
            feeders_fired[session_index_i],
            max_latency_s=max_latency_s,
        )
        error_match = _nearest_after(
            choice_exit_s,
            error_times[session_index_i],
            error_not_fired[session_index_i],
            max_latency_s=max_latency_s,
        )
        primary = _primary_from_reward(row.get("reward", np.nan), feeder_match, error_match)

        out = row.to_dict()
        out.update(
            {
                "choice_direction_audit_stage": "raw_event_code_alignment",
                "choice_direction_status": "raw_code_only_not_left_right",
                "choice_point_entry_s_audit": choice_entry_s,
                "choice_point_exit_s_audit": choice_exit_s,
                "feeder_event_time_s": feeder_match["event_time_s"],
                "feeder_event_code": feeder_match["event_code"],
                "feeder_event_code_label": _event_code_label(feeder_match["event_code"]),
                "feeder_event_latency_s": feeder_match["event_latency_s"],
                "feeder_event_match_status": feeder_match["match_status"],
                "error_event_time_s": error_match["event_time_s"],
                "error_event_code": error_match["event_code"],
                "error_event_code_label": _event_code_label(error_match["event_code"]),
                "error_event_latency_s": error_match["event_latency_s"],
                "error_event_match_status": error_match["match_status"],
                "contingency_correct_l_audit": _safe_matrix_value(contingency_l, session_index_i, lap_index),
                "contingency_correct_r_audit": _safe_matrix_value(contingency_r, session_index_i, lap_index),
                "contingency_correct_a_audit": _safe_matrix_value(contingency_a, session_index_i, lap_index),
                **primary,
            }
        )
        rows.append(out)

    audit = pd.DataFrame(rows)

    usable = audit[
        audit["primary_event_match_status"].eq("matched_after_choice_exit")
        & audit["primary_event_code_label"].astype(str).str.len().gt(0)
    ].copy()

    audit_csv = output_dir / "redish_lra17f_choice_direction_event_audit.csv"
    usable_csv = output_dir / "redish_lra17f_choice_direction_event_audit_usable.csv"
    by_code_csv = output_dir / "Table_Redish_LRA17F_Primary_Event_Code_By_Outcome.csv"
    crosswalk_csv = output_dir / "Table_Redish_LRA17F_Feeder_Error_Code_Crosswalk.csv"
    coverage_csv = output_dir / "Table_Redish_LRA17F_Choice_Direction_Audit_Coverage.csv"

    audit.to_csv(audit_csv, index=False)
    usable.to_csv(usable_csv, index=False)

    _summary_by_code(usable).to_csv(by_code_csv, index=False)
    _summary_crosswalk(usable).to_csv(crosswalk_csv, index=False)

    coverage = _coverage_table(
        audit,
        [
            "subject_id",
            "session_id",
            "trial",
            "outcome",
            "reward",
            "choice_point_exit_s_audit",
            "feeder_event_code_label",
            "error_event_code_label",
            "primary_event_kind",
            "primary_event_code_label",
            "primary_event_latency_s",
            "contingency_correct_l_audit",
            "contingency_correct_r_audit",
            "contingency_correct_a_audit",
        ],
    )
    coverage.to_csv(coverage_csv, index=False)

    meta = {
        "dataset_id": DATASET_ID,
        "input_csv": str(healthy_control_csv),
        "root": str(root),
        "cohort": "lra",
        "policy": "audit raw feeder/error event codes; do not infer left/right labels yet",
        "max_latency_s": max_latency_s,
        "n_input_rows": int(len(control)),
        "n_audit_rows": int(len(audit)),
        "n_usable_rows": int(len(usable)),
        "source_files": {
            "idphi": str(idphi_path),
            "lapdata": str(lapdata_path),
        },
        "outputs": {
            "audit": str(audit_csv),
            "usable": str(usable_csv),
            "by_code": str(by_code_csv),
            "crosswalk": str(crosswalk_csv),
            "coverage": str(coverage_csv),
        },
    }

    meta_json = output_dir / "redish_lra17f_choice_direction_event_audit_meta.json"
    report_md = output_dir / "Redish_LRA17F_Choice_Direction_Event_Audit_Report.md"

    meta_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    report_md.write_text(
        "\n".join(
            [
                "# Redish LRA 2024 Patch 17F choice-direction event audit",
                "",
                "This audit aligns raw `FeedersFired` and `ErrorNotFired` event codes to each choice-point trial.",
                "",
                "It does not interpret event codes as left/right labels.",
                "The next patch may assign `chosen_action` only if the raw code mapping is stable and independently interpretable.",
                "",
                f"- Input rows: `{len(control)}`",
                f"- Audit rows: `{len(audit)}`",
                f"- Usable aligned rows: `{len(usable)}`",
                f"- Max post-choice latency: `{max_latency_s}` seconds",
                "",
                "Main outputs:",
                "",
                f"- `{audit_csv}`",
                f"- `{usable_csv}`",
                f"- `{by_code_csv}`",
                f"- `{crosswalk_csv}`",
                f"- `{coverage_csv}`",
            ]
        ),
        encoding="utf-8",
    )

    print(f"Choice-direction event audit saved: {audit_csv}")
    print(f"Usable event audit saved: {usable_csv}")
    print(f"Primary event-code summary saved: {by_code_csv}")
    print(f"Feeder/error crosswalk saved: {crosswalk_csv}")
    print(f"Coverage saved: {coverage_csv}")
    print(f"Metadata saved: {meta_json}")
    print(f"Report saved: {report_md}")
    print(f"Input rows: {len(control)}")
    print(f"Audit rows: {len(audit)}")
    print(f"Usable aligned rows: {len(usable)}")

    return meta


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit raw LRA feeder/error event codes aligned to choice-point trials."
    )
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--healthy-control-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-latency-s", type=float, default=90.0)
    parser.add_argument("--max-laps", type=int, default=250)

    args = parser.parse_args()

    audit_lra_choice_direction_events(
        root=args.root,
        healthy_control_csv=args.healthy_control_csv,
        output_dir=args.output_dir,
        max_latency_s=args.max_latency_s,
        max_laps=args.max_laps,
    )


if __name__ == "__main__":
    main()
