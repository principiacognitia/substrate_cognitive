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
    _add_group_zscore,
    _build_cohort_rows,
    _cohort_paths,
    _read_value,
    _to_numeric_array,
)


def _finite_float(value: Any) -> float:
    try:
        x = float(value)
    except Exception:
        return float("nan")
    return x if math.isfinite(x) else float("nan")


def _threshold_vector_from_value(value: Any, n_sessions: int) -> tuple[np.ndarray, str]:
    arr = _to_numeric_array(value)
    out = np.full(n_sessions, np.nan, dtype=float)

    if arr.size == 0:
        return out, ""

    arr = np.squeeze(arr)

    if arr.ndim == 0:
        x = _finite_float(arr)
        if math.isfinite(x):
            out[:] = x
            return out, "scalar"
        return out, ""

    if arr.ndim == 1:
        flat = np.asarray(arr, dtype=float).reshape(-1)
        if flat.size == n_sessions:
            out[:] = flat[:n_sessions]
            return out, "session_vector"

        finite = flat[np.isfinite(flat)]
        if finite.size == 1:
            out[:] = float(finite[0])
            return out, "single_value_vector"

        if finite.size > 1:
            out[:] = float(np.nanmedian(finite))
            return out, "median_vector_fallback"

        return out, ""

    if arr.ndim >= 2:
        matrix = np.asarray(arr, dtype=float)

        if matrix.shape[0] == n_sessions:
            out[:] = np.nanmedian(matrix, axis=1)
            return out, "session_by_value_matrix"

        if matrix.shape[1] == n_sessions:
            out[:] = np.nanmedian(matrix, axis=0)
            return out, "value_by_session_matrix"

        finite = matrix[np.isfinite(matrix)]
        if finite.size == 1:
            out[:] = float(finite[0])
            return out, "single_value_matrix"

        if finite.size > 1:
            out[:] = float(np.nanmedian(finite))
            return out, "median_matrix_fallback"

    return out, ""


def _read_vte_threshold_vector(idphi_path: Path, n_sessions: int) -> tuple[np.ndarray, str]:
    candidates = [
        "VTEThreshold.ChoicePoint",
        "VTEThreshold",
    ]

    for logical_path in candidates:
        try:
            value = _read_value(idphi_path, logical_path)
        except Exception:
            value = None

        vector, mode = _threshold_vector_from_value(value, n_sessions=n_sessions)
        if np.isfinite(vector).any():
            return vector, f"{logical_path}:{mode}"

    return np.full(n_sessions, np.nan, dtype=float), ""


def _apply_vte_resolution(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["lab_vte_binary"] = pd.to_numeric(out["lab_vte_binary"], errors="coerce")
    out["lab_idphi"] = pd.to_numeric(out["lab_idphi"], errors="coerce")
    out["lab_vte_threshold"] = pd.to_numeric(out["lab_vte_threshold"], errors="coerce")

    out["vte_binary"] = out["lab_vte_binary"]
    out["vte_binary_source"] = ""

    native_mask = out["lab_vte_binary"].notna()
    out.loc[native_mask, "vte_binary_source"] = "VTELap.ChoicePoint"

    threshold_mask = (
        out["vte_binary"].isna()
        & out["lab_idphi"].notna()
        & out["lab_vte_threshold"].notna()
    )

    out.loc[threshold_mask, "vte_binary"] = (
        out.loc[threshold_mask, "lab_idphi"]
        >= out.loc[threshold_mask, "lab_vte_threshold"]
    ).astype(float)

    out.loc[threshold_mask, "vte_binary_source"] = "IdPhi.ChoicePoint>=VTEThreshold"

    return out


def _add_threshold_columns(
    df: pd.DataFrame,
    root: Path,
    cohorts: list[str],
) -> pd.DataFrame:
    out = df.copy()
    out["lab_vte_threshold"] = np.nan
    out["lab_vte_threshold_source"] = ""

    for cohort in cohorts:
        cohort_mask = out["cohort"] == cohort
        if not cohort_mask.any():
            continue

        n_sessions = int(pd.to_numeric(out.loc[cohort_mask, "session_index"], errors="coerce").max()) + 1
        paths = _cohort_paths(root, cohort)
        vector, source = _read_vte_threshold_vector(paths["idphi"], n_sessions=n_sessions)

        session_indices = pd.to_numeric(out.loc[cohort_mask, "session_index"], errors="coerce").astype("Int64")
        thresholds = session_indices.map(
            lambda idx: vector[int(idx)] if pd.notna(idx) and int(idx) < len(vector) else np.nan
        )

        out.loc[cohort_mask, "lab_vte_threshold"] = thresholds.astype(float)
        out.loc[cohort_mask, "lab_vte_threshold_source"] = source

    return out


def _summary(df: pd.DataFrame, group_cols: list[str], output_csv: Path) -> pd.DataFrame:
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
            vte_rate=("vte_binary", "mean"),
            mean_lab_idphi=("lab_idphi", "mean"),
            mean_lab_avg_idphi=("lab_avg_idphi", "mean"),
            mean_dwell_proxy=("dwell_proxy", "mean"),
            mean_pause_time_s=("pause_time_s", "mean"),
        )
        .reset_index()
    )

    summary.to_csv(output_csv, index=False)
    return summary


def build_lra_canonical_choice_endpoint_vte_resolved(
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
    df = _add_threshold_columns(df, root=root, cohorts=cohorts)
    df = _apply_vte_resolution(df)

    numeric_cols = [
        "reward",
        "correct",
        "cost",
        "lab_idphi",
        "lab_avg_idphi",
        "lab_z_idphi",
        "lab_robust_z_idphi",
        "lab_vte_binary",
        "lab_vte_threshold",
        "vte_binary",
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
        & df["vte_binary"].notna()
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

    by_cohort_csv = output_dir / "Table_Redish_LRA17D_Canonical_By_Cohort.csv"
    by_subject_csv = output_dir / "Table_Redish_LRA17D_Canonical_By_Subject.csv"
    by_session_csv = output_dir / "Table_Redish_LRA17D_Canonical_By_Session.csv"
    by_vte_csv = output_dir / "Table_Redish_LRA17D_Canonical_By_VTE.csv"
    by_vte_source_csv = output_dir / "Table_Redish_LRA17D_Canonical_By_VTE_Source.csv"
    by_switch_csv = output_dir / "Table_Redish_LRA17D_Canonical_By_Switch_Relation.csv"

    df.to_csv(endpoint_csv, index=False)
    usable.to_csv(usable_csv, index=False)

    _summary(usable, ["cohort", "treatment"], by_cohort_csv)
    _summary(usable, ["cohort", "treatment", "subject_id"], by_subject_csv)
    _summary(usable, ["cohort", "treatment", "subject_id", "session_id"], by_session_csv)
    _summary(usable, ["cohort", "treatment", "vte_binary"], by_vte_csv)
    _summary(usable, ["cohort", "treatment", "vte_binary_source"], by_vte_source_csv)
    _summary(usable, ["cohort", "treatment", "switch_relation"], by_switch_csv)

    meta = {
        "dataset_id": DATASET_ID,
        "patch": "17D",
        "root": str(root),
        "cohorts": cohorts,
        "max_laps": max_laps,
        "n_rows": int(len(df)),
        "n_usable_rows": int(len(usable)),
        "n_subjects": int(usable["subject_id"].nunique()) if not usable.empty else 0,
        "n_sessions": int(usable["session_id"].nunique()) if not usable.empty else 0,
        "n_cohorts": int(usable["cohort"].nunique()) if not usable.empty else 0,
        "vte_binary_resolution": {
            "native": "VTELap.ChoicePoint when present",
            "fallback": "IdPhi.ChoicePoint >= VTEThreshold when native VTELap is absent",
            "no_arbitrary_threshold": True,
        },
        "source_files": source_files,
        "outputs": {
            "endpoint": str(endpoint_csv),
            "usable": str(usable_csv),
            "by_cohort": str(by_cohort_csv),
            "by_subject": str(by_subject_csv),
            "by_session": str(by_session_csv),
            "by_vte": str(by_vte_csv),
            "by_vte_source": str(by_vte_source_csv),
            "by_switch": str(by_switch_csv),
        },
    }

    meta_json = output_dir / "redish_lra_canonical_choice_endpoint_meta.json"
    meta_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    report = output_dir / "Redish_LRA17D_Canonical_Choice_Endpoint_Report.md"
    report.write_text(
        "\n".join(
            [
                "# Redish LRA 2024 canonical choice endpoint, Patch 17D",
                "",
                f"- Rows: `{len(df)}`",
                f"- Usable rows: `{len(usable)}`",
                f"- Cohorts in usable: `{', '.join(sorted(usable['cohort'].dropna().unique())) if not usable.empty else ''}`",
                "",
                "## VTE label policy",
                "",
                "Native `VTELap.ChoicePoint` is used when present.",
                "If native lap-level VTE labels are absent, `vte_binary` is reconstructed only from author-provided `VTEThreshold`.",
                "No synthetic or hand-tuned threshold is introduced.",
            ]
        ),
        encoding="utf-8",
    )

    print(f"Canonical LRA endpoint saved: {endpoint_csv}")
    print(f"Usable LRA endpoint saved: {usable_csv}")
    print(f"Cohort summary saved: {by_cohort_csv}")
    print(f"VTE summary saved: {by_vte_csv}")
    print(f"VTE source summary saved: {by_vte_source_csv}")
    print(f"Metadata saved: {meta_json}")
    print(f"Report saved: {report}")
    print(f"Rows: {len(df)}")
    print(f"Usable rows: {len(usable)}")

    return meta


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract Redish LRA 2024 canonical choice-point endpoint with resolved VTE labels."
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--cohorts",
        nargs="+",
        choices=sorted(COHORT_SPECS),
        default=sorted(COHORT_SPECS),
    )
    parser.add_argument("--max-laps", type=int, default=250)

    args = parser.parse_args()

    build_lra_canonical_choice_endpoint_vte_resolved(
        root=args.root,
        output_dir=args.output_dir,
        cohorts=args.cohorts,
        max_laps=args.max_laps,
    )


if __name__ == "__main__":
    main()