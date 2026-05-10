from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DATASET_ID = "redish_lra_2024"
NATIVE_CONTROL_VTE_SOURCE = "VTELap.ChoicePoint"


def _norm_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def _norm_key(value: Any) -> str:
    return _norm_text(value).lower()


def _ensure_column(df: pd.DataFrame, name: str, default: Any = "") -> None:
    if name not in df.columns:
        df[name] = default


def _to_numeric_columns(df: pd.DataFrame, columns: list[str]) -> None:
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")


def _copy_if_missing(df: pd.DataFrame, target: str, source: str) -> None:
    if target not in df.columns and source in df.columns:
        df[target] = df[source]


def _treatment_family(cohort: str, treatment: str) -> str:
    c = _norm_key(cohort)
    t = _norm_key(treatment)

    if c == "lra" and t == "control":
        return "healthy_control"

    if c == "mpfc_dreadds":
        if "dcz" in t:
            return "dreadd_perturbation"
        if "veh" in t or "vehicle" in t:
            return "dreadd_vehicle_control"
        return "dreadd_unknown"

    return "other"


def _is_native_control_vte_source(
    source: Any,
    native_source: str = NATIVE_CONTROL_VTE_SOURCE,
) -> bool:
    src = _norm_key(source)
    native = native_source.lower()
    return src == native or src.endswith(native)


def _summarize(
    df: pd.DataFrame,
    group_cols: list[str],
) -> pd.DataFrame:
    if df.empty:
        cols = group_cols + [
            "n_rows",
            "n_subjects",
            "n_sessions",
            "reward_rate",
            "mean_reward",
            "vte_rate_for_comparison",
            "mean_lab_idphi",
            "mean_lab_avg_idphi",
            "mean_lab_z_idphi",
            "mean_lab_robust_z_idphi",
            "mean_dwell_proxy",
            "mean_pause_time_s",
        ]
        return pd.DataFrame(columns=cols)

    work = df.copy()
    for col in [
        "reward",
        "vte_binary_for_comparison",
        "lab_idphi",
        "lab_avg_idphi",
        "lab_z_idphi",
        "lab_robust_z_idphi",
        "dwell_proxy",
        "pause_time_s",
    ]:
        _ensure_column(work, col, np.nan)
        work[col] = pd.to_numeric(work[col], errors="coerce")

    _ensure_column(work, "subject_id", "")
    _ensure_column(work, "session_id", "")

    return (
        work.groupby(group_cols, dropna=False)
        .agg(
            n_rows=("dataset_id", "size"),
            n_subjects=("subject_id", "nunique"),
            n_sessions=("session_id", "nunique"),
            reward_rate=("reward", "mean"),
            mean_reward=("reward", "mean"),
            vte_rate_for_comparison=("vte_binary_for_comparison", "mean"),
            mean_lab_idphi=("lab_idphi", "mean"),
            mean_lab_avg_idphi=("lab_avg_idphi", "mean"),
            mean_lab_z_idphi=("lab_z_idphi", "mean"),
            mean_lab_robust_z_idphi=("lab_robust_z_idphi", "mean"),
            mean_dwell_proxy=("dwell_proxy", "mean"),
            mean_pause_time_s=("pause_time_s", "mean"),
        )
        .reset_index()
    )


def build_lra_comparability_policy_split(
    input_csv: Path,
    output_dir: Path,
    native_control_vte_source: str = NATIVE_CONTROL_VTE_SOURCE,
    assume_control_native_when_source_missing: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv, low_memory=False)

    _copy_if_missing(df, "vte_binary", "lab_vte_binary")
    _copy_if_missing(df, "vte_binary_source", "lab_vte_binary_source")

    for col in [
        "dataset_id",
        "trace_origin",
        "task_family",
        "cohort",
        "treatment",
        "subject_id",
        "session_id",
        "trial",
        "decision_stage",
        "choice_point_id",
        "outcome",
        "vte_binary_source",
        "switch_relation",
    ]:
        _ensure_column(df, col, "")

    _ensure_column(df, "vte_binary", np.nan)

    _to_numeric_columns(
        df,
        [
            "reward",
            "correct",
            "cost",
            "lab_idphi",
            "lab_avg_idphi",
            "lab_z_idphi",
            "lab_robust_z_idphi",
            "vte_binary",
            "lab_vte_binary",
            "choice_point_dwell_s",
            "pause_time_s",
            "run_speed",
            "dwell_proxy",
            "deliberation_proxy",
            "nearest_switch_lap",
            "lap_from_switch",
        ],
    )

    if "dataset_id" in df.columns:
        df["dataset_id"] = df["dataset_id"].replace("", DATASET_ID)
    else:
        df["dataset_id"] = DATASET_ID

    cohort_key = df["cohort"].map(_norm_key)
    treatment_key = df["treatment"].map(_norm_key)
    source_key = df["vte_binary_source"].map(_norm_key)

    if assume_control_native_when_source_missing:
        missing_source_control = (
            (cohort_key == "lra")
            & (treatment_key == "control")
            & source_key.eq("")
            & df["vte_binary"].notna()
        )
        df.loc[missing_source_control, "vte_binary_source"] = native_control_vte_source
        source_key = df["vte_binary_source"].map(_norm_key)

    df["treatment_family"] = [
        _treatment_family(c, t) for c, t in zip(df["cohort"], df["treatment"])
    ]

    has_core_continuous = (
        df["reward"].notna()
        & df["deliberation_proxy"].notna()
    )

    is_lra_control = (cohort_key == "lra") & (treatment_key == "control")
    is_native_control_source = df["vte_binary_source"].map(
        lambda x: _is_native_control_vte_source(x, native_control_vte_source)
    )

    control_baseline_mask = (
        is_lra_control
        & is_native_control_source
        & has_core_continuous
        & df["vte_binary"].notna()
    )

    perturbation_mask = (
        (cohort_key == "mpfc_dreadds")
        & has_core_continuous
    )

    df["biological_comparison_role"] = "excluded_from_healthy_baseline"
    df.loc[control_baseline_mask, "biological_comparison_role"] = "healthy_lra_control_baseline"
    df.loc[perturbation_mask, "biological_comparison_role"] = "dreadd_perturbation_continuous_only"

    df["healthy_baseline_eligible"] = control_baseline_mask
    df["synthetic_comparable_eligible"] = control_baseline_mask

    df["vte_label_policy"] = "do_not_use_as_healthy_baseline_label"
    df.loc[control_baseline_mask, "vte_label_policy"] = "native_control_vtelap_label"
    df.loc[df["vte_binary_source"].map(_norm_text).eq(""), "vte_label_policy"] = "missing_vte_label_source"

    df["vte_binary_usable"] = control_baseline_mask
    df["vte_binary_for_comparison"] = np.nan
    df.loc[control_baseline_mask, "vte_binary_for_comparison"] = df.loc[
        control_baseline_mask, "vte_binary"
    ]

    control = df[control_baseline_mask].copy()
    perturbation = df[perturbation_mask].copy()
    excluded = df[~control_baseline_mask & ~perturbation_mask].copy()

    control_csv = output_dir / "redish_lra17e_healthy_control_baseline.csv"
    perturbation_csv = output_dir / "redish_lra17e_dreadd_perturbation_continuous.csv"
    excluded_csv = output_dir / "redish_lra17e_excluded_from_healthy_baseline.csv"

    control_by_vte_csv = output_dir / "Table_Redish_LRA17E_Control_By_VTE.csv"
    control_by_subject_csv = output_dir / "Table_Redish_LRA17E_Control_By_Subject.csv"
    control_by_switch_csv = output_dir / "Table_Redish_LRA17E_Control_By_Switch_Relation.csv"
    perturbation_by_treatment_csv = output_dir / "Table_Redish_LRA17E_Perturbation_By_Treatment.csv"
    vte_source_audit_csv = output_dir / "Table_Redish_LRA17E_VTE_Label_Source_Audit.csv"

    control.to_csv(control_csv, index=False)
    perturbation.to_csv(perturbation_csv, index=False)
    excluded.to_csv(excluded_csv, index=False)

    _summarize(control, ["cohort", "treatment", "vte_binary_for_comparison"]).to_csv(
        control_by_vte_csv, index=False
    )
    _summarize(control, ["cohort", "treatment", "subject_id"]).to_csv(
        control_by_subject_csv, index=False
    )
    _summarize(control, ["cohort", "treatment", "switch_relation"]).to_csv(
        control_by_switch_csv, index=False
    )
    _summarize(perturbation, ["cohort", "treatment", "treatment_family"]).to_csv(
        perturbation_by_treatment_csv, index=False
    )

    audit = (
        df.groupby(
            [
                "cohort",
                "treatment",
                "treatment_family",
                "vte_binary_source",
                "biological_comparison_role",
                "vte_label_policy",
            ],
            dropna=False,
        )
        .agg(
            n_rows=("dataset_id", "size"),
            n_subjects=("subject_id", "nunique"),
            n_sessions=("session_id", "nunique"),
            n_vte_nonempty=("vte_binary", "count"),
            reward_rate=("reward", "mean"),
            mean_lab_idphi=("lab_idphi", "mean"),
            mean_vte_binary_raw=("vte_binary", "mean"),
            mean_vte_binary_for_comparison=("vte_binary_for_comparison", "mean"),
        )
        .reset_index()
    )
    audit.to_csv(vte_source_audit_csv, index=False)

    meta = {
        "dataset_id": DATASET_ID,
        "input_csv": str(input_csv),
        "native_control_vte_source": native_control_vte_source,
        "assume_control_native_when_source_missing": assume_control_native_when_source_missing,
        "n_input_rows": int(len(df)),
        "n_control_baseline_rows": int(len(control)),
        "n_perturbation_rows": int(len(perturbation)),
        "n_excluded_rows": int(len(excluded)),
        "n_control_subjects": int(control["subject_id"].nunique()) if not control.empty else 0,
        "n_control_sessions": int(control["session_id"].nunique()) if not control.empty else 0,
        "n_perturbation_subjects": int(perturbation["subject_id"].nunique()) if not perturbation.empty else 0,
        "n_perturbation_sessions": int(perturbation["session_id"].nunique()) if not perturbation.empty else 0,
        "outputs": {
            "control_baseline": str(control_csv),
            "perturbation_continuous": str(perturbation_csv),
            "excluded": str(excluded_csv),
            "control_by_vte": str(control_by_vte_csv),
            "control_by_subject": str(control_by_subject_csv),
            "control_by_switch": str(control_by_switch_csv),
            "perturbation_by_treatment": str(perturbation_by_treatment_csv),
            "vte_source_audit": str(vte_source_audit_csv),
        },
    }

    meta_json = output_dir / "redish_lra17e_comparability_policy_meta.json"
    report_md = output_dir / "Redish_LRA17E_Comparability_Policy_Report.md"

    meta_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    report = [
        "# Redish LRA 2024 Patch 17E comparability policy split",
        "",
        "Patch 17E is a policy split, not a new extraction pass.",
        "",
        "Rules:",
        "",
        "1. Healthy biological baseline is only `cohort == lra`, `treatment == control`, and native `VTELap.ChoicePoint` VTE label.",
        "2. DREADD rows are not mixed into the healthy baseline.",
        "3. DREADD rows remain available as continuous perturbation data, but their binary VTE label is not used for healthy synthetic comparability.",
        "4. Labels reconstructed from `IdPhi.ChoicePoint >= VTEThreshold` are retained only in the audit/excluded layer unless scale validity is separately proven.",
        "",
        f"- Input rows: `{len(df)}`",
        f"- Healthy control baseline rows: `{len(control)}`",
        f"- DREADD perturbation continuous rows: `{len(perturbation)}`",
        f"- Excluded rows: `{len(excluded)}`",
        f"- Control subjects: `{meta['n_control_subjects']}`",
        f"- Control sessions: `{meta['n_control_sessions']}`",
        "",
        "Main files:",
        "",
        f"- `{control_csv}`",
        f"- `{perturbation_csv}`",
        f"- `{excluded_csv}`",
        f"- `{vte_source_audit_csv}`",
    ]
    report_md.write_text("\n".join(report), encoding="utf-8")

    print(f"Healthy control baseline saved: {control_csv}")
    print(f"DREADD perturbation continuous endpoint saved: {perturbation_csv}")
    print(f"Excluded endpoint saved: {excluded_csv}")
    print(f"Control VTE summary saved: {control_by_vte_csv}")
    print(f"Control subject summary saved: {control_by_subject_csv}")
    print(f"Control switch summary saved: {control_by_switch_csv}")
    print(f"Perturbation treatment summary saved: {perturbation_by_treatment_csv}")
    print(f"VTE source audit saved: {vte_source_audit_csv}")
    print(f"Metadata saved: {meta_json}")
    print(f"Report saved: {report_md}")
    print(f"Input rows: {len(df)}")
    print(f"Healthy control baseline rows: {len(control)}")
    print(f"DREADD perturbation rows: {len(perturbation)}")
    print(f"Excluded rows: {len(excluded)}")

    return meta


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split Redish LRA 2024 endpoint into healthy-control baseline and DREADD perturbation layers."
    )
    parser.add_argument("--input-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--native-control-vte-source",
        default=NATIVE_CONTROL_VTE_SOURCE,
        help="Native VTE label source accepted for healthy LRA-control baseline.",
    )
    parser.add_argument(
        "--assume-control-native-when-source-missing",
        action="store_true",
        help="Unsafe compatibility mode for older endpoints without vte_binary_source.",
    )
    args = parser.parse_args()

    build_lra_comparability_policy_split(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        native_control_vte_source=args.native_control_vte_source,
        assume_control_native_when_source_missing=args.assume_control_native_when_source_missing,
    )


if __name__ == "__main__":
    main()
