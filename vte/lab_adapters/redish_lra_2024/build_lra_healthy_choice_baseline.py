from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DATASET_ID = "redish_lra_2024"


def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _clean_text(value: Any) -> str:
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "null"}:
        return ""
    return text


def _zscore_by_group(df: pd.DataFrame, value_col: str, group_cols: list[str], out_col: str) -> None:
    if value_col not in df.columns:
        df[out_col] = np.nan
        return

    values = pd.to_numeric(df[value_col], errors="coerce")
    grouped = values.groupby([df[c] for c in group_cols], dropna=False)

    mean = grouped.transform("mean")
    std = grouped.transform(lambda x: x.std(ddof=0))

    z = (values - mean) / std
    z = z.where(std.notna() & (std != 0), 0.0)
    df[out_col] = z


def _coverage(df: pd.DataFrame, fields: list[str], output_csv: Path) -> pd.DataFrame:
    rows = []
    total = len(df)

    for field in fields:
        if field not in df.columns:
            rows.append({"field": field, "nonempty": 0, "total": total, "coverage": 0.0})
            continue

        s = df[field]
        if s.dtype == object:
            nonempty = int(
                s.map(lambda x: str(x).strip().lower() not in {"", "nan", "none", "null"}).sum()
            )
        else:
            nonempty = int(s.notna().sum())

        rows.append(
            {
                "field": field,
                "nonempty": nonempty,
                "total": total,
                "coverage": nonempty / total if total else 0.0,
            }
        )

    out = pd.DataFrame(rows)
    out.to_csv(output_csv, index=False)
    return out


def _summary(df: pd.DataFrame, group_cols: list[str], output_csv: Path) -> pd.DataFrame:
    if df.empty:
        out = pd.DataFrame(columns=group_cols)
        out.to_csv(output_csv, index=False)
        return out

    work = df.copy()

    numeric_cols = [
        "reward",
        "vte_binary",
        "lab_idphi",
        "lab_avg_idphi",
        "lab_z_idphi",
        "lab_robust_z_idphi",
        "dwell_proxy",
        "deliberation_proxy",
        "pause_time_s",
        "choice_point_dwell_s",
        "strict_primary_event_latency_s",
        "z_lab_idphi_by_session",
        "z_dwell_proxy_by_session",
        "z_pause_time_by_session",
    ]

    for col in numeric_cols:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")

    agg = {
        "n_rows": ("dataset_id", "size"),
        "n_subjects": ("subject_id", "nunique"),
        "n_sessions": ("session_id", "nunique"),
        "reward_rate": ("reward", "mean"),
        "vte_rate": ("vte_binary", "mean"),
        "mean_lab_idphi": ("lab_idphi", "mean"),
        "mean_lab_avg_idphi": ("lab_avg_idphi", "mean"),
        "mean_dwell_proxy": ("dwell_proxy", "mean"),
        "mean_deliberation_proxy": ("deliberation_proxy", "mean"),
        "mean_pause_time_s": ("pause_time_s", "mean"),
    }

    optional = {
        "mean_choice_point_dwell_s": ("choice_point_dwell_s", "mean"),
        "mean_event_latency_s": ("strict_primary_event_latency_s", "mean"),
        "mean_z_lab_idphi_by_session": ("z_lab_idphi_by_session", "mean"),
        "mean_z_dwell_proxy_by_session": ("z_dwell_proxy_by_session", "mean"),
        "mean_z_pause_time_by_session": ("z_pause_time_by_session", "mean"),
    }

    for out_col, spec in optional.items():
        if spec[0] in work.columns:
            agg[out_col] = spec

    out = work.groupby(group_cols, dropna=False).agg(**agg).reset_index()
    out.to_csv(output_csv, index=False)
    return out


def build_lra_healthy_choice_baseline(
    same_trial_csv: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(same_trial_csv, low_memory=False)

    for col in [
        "reward",
        "lab_idphi",
        "lab_avg_idphi",
        "lab_z_idphi",
        "lab_robust_z_idphi",
        "vte_binary_for_comparison",
        "choice_point_dwell_s",
        "pause_time_s",
        "strict_primary_event_latency_s",
    ]:
        if col in df.columns:
            df[col] = _to_numeric(df[col])

    baseline = df.copy()

    if "cohort" in baseline.columns:
        baseline = baseline[baseline["cohort"].astype(str) == "lra"].copy()
    if "treatment" in baseline.columns:
        baseline = baseline[baseline["treatment"].astype(str) == "control"].copy()
    if "vte_binary_source" in baseline.columns:
        baseline = baseline[
            baseline["vte_binary_source"].astype(str) == "VTELap.ChoicePoint"
        ].copy()

    if "strict_event_status" in baseline.columns:
        baseline = baseline[
            baseline["strict_event_status"].isin(
                ["same_trial_event", "same_trial_event_last_lap_latency_only"]
            )
        ].copy()

    baseline["dataset_id"] = DATASET_ID
    baseline["source"] = "biological"
    baseline["trace_origin"] = "biological"
    baseline["task_family"] = "left_right_alternate"
    baseline["decision_stage"] = "choice_point"
    baseline["biological_comparison_role"] = "healthy_lra_control_same_trial_action_subset"
    baseline["synthetic_comparable_eligible"] = True

    baseline["native_action_kind"] = baseline.get("primary_event_kind", "").map(_clean_text)
    baseline["native_action_code"] = baseline.get("primary_event_code_label", "").map(_clean_text)
    baseline["chosen_action"] = baseline["native_action_code"]
    baseline["chosen_action_namespace"] = "raw_lra_event_code"
    baseline["choice_direction_policy"] = "raw_event_code_only_no_left_right_mapping"
    baseline["choice_direction_usable"] = False

    baseline["cost"] = 1.0
    baseline["cost_source"] = "nominal_balanced_two_arm_no_per_trial_route_cost"
    baseline["comparable_cost_bin"] = "balanced"

    baseline["reward"] = _to_numeric(baseline["reward"])
    baseline["vte_binary"] = _to_numeric(baseline["vte_binary_for_comparison"])
    baseline["lab_vte_binary"] = baseline["vte_binary"]

    baseline["deliberation_proxy"] = _to_numeric(baseline["lab_idphi"])

    if "choice_point_dwell_s" in baseline.columns:
        baseline["dwell_proxy"] = _to_numeric(baseline["choice_point_dwell_s"])
    elif "dwell_proxy" in baseline.columns:
        baseline["dwell_proxy"] = _to_numeric(baseline["dwell_proxy"])
    else:
        baseline["dwell_proxy"] = np.nan

    if "pause_time_s" in baseline.columns:
        baseline["pause_time_s"] = _to_numeric(baseline["pause_time_s"])
    else:
        baseline["pause_time_s"] = np.nan

    _zscore_by_group(
        baseline,
        value_col="lab_idphi",
        group_cols=["subject_id", "session_id"],
        out_col="z_lab_idphi_by_session",
    )
    _zscore_by_group(
        baseline,
        value_col="dwell_proxy",
        group_cols=["subject_id", "session_id"],
        out_col="z_dwell_proxy_by_session",
    )
    _zscore_by_group(
        baseline,
        value_col="pause_time_s",
        group_cols=["subject_id", "session_id"],
        out_col="z_pause_time_by_session",
    )
    _zscore_by_group(
        baseline,
        value_col="strict_primary_event_latency_s",
        group_cols=["subject_id", "session_id"],
        out_col="z_event_latency_by_session",
    )

    sort_cols = [c for c in ["subject_id", "session_id", "trial"] if c in baseline.columns]
    baseline = baseline.sort_values(sort_cols).reset_index(drop=True)

    endpoint_csv = output_dir / "redish_lra17h_healthy_choice_baseline.csv"
    by_vte_csv = output_dir / "Table_Redish_LRA17H_By_VTE.csv"
    by_outcome_csv = output_dir / "Table_Redish_LRA17H_By_Outcome.csv"
    by_action_csv = output_dir / "Table_Redish_LRA17H_By_Action_Code.csv"
    by_subject_csv = output_dir / "Table_Redish_LRA17H_By_Subject.csv"
    by_session_csv = output_dir / "Table_Redish_LRA17H_By_Session.csv"
    coverage_csv = output_dir / "Table_Redish_LRA17H_Field_Coverage.csv"

    baseline.to_csv(endpoint_csv, index=False)

    _summary(baseline, ["vte_binary"], by_vte_csv)
    _summary(baseline, ["outcome"], by_outcome_csv)
    _summary(baseline, ["native_action_kind", "native_action_code", "outcome"], by_action_csv)
    _summary(baseline, ["subject_id"], by_subject_csv)
    _summary(baseline, ["subject_id", "session_id"], by_session_csv)

    coverage_fields = [
        "source",
        "dataset_id",
        "task_family",
        "subject_id",
        "session_id",
        "trial",
        "decision_stage",
        "outcome",
        "reward",
        "vte_binary",
        "lab_idphi",
        "lab_avg_idphi",
        "deliberation_proxy",
        "dwell_proxy",
        "pause_time_s",
        "native_action_kind",
        "native_action_code",
        "chosen_action",
        "choice_direction_policy",
        "strict_primary_event_latency_s",
        "z_lab_idphi_by_session",
        "z_dwell_proxy_by_session",
        "z_pause_time_by_session",
    ]
    _coverage(baseline, coverage_fields, coverage_csv)

    meta = {
        "dataset_id": DATASET_ID,
        "patch": "17H",
        "input_csv": str(same_trial_csv),
        "policy": (
            "Healthy LRA control baseline restricted to strict same-trial action-code subset. "
            "DREADD perturbation is excluded. Raw event codes are retained, but not mapped to left/right."
        ),
        "n_input_rows": int(len(df)),
        "n_output_rows": int(len(baseline)),
        "n_subjects": int(baseline["subject_id"].nunique()) if "subject_id" in baseline else 0,
        "n_sessions": int(baseline["session_id"].nunique()) if "session_id" in baseline else 0,
        "choice_direction_usable": False,
        "outputs": {
            "endpoint": str(endpoint_csv),
            "by_vte": str(by_vte_csv),
            "by_outcome": str(by_outcome_csv),
            "by_action": str(by_action_csv),
            "by_subject": str(by_subject_csv),
            "by_session": str(by_session_csv),
            "coverage": str(coverage_csv),
        },
    }

    meta_json = output_dir / "redish_lra17h_healthy_choice_baseline_meta.json"
    report_md = output_dir / "Redish_LRA17H_Healthy_Choice_Baseline_Report.md"

    meta_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    report_md.write_text(
        "\n".join(
            [
                "# Redish LRA 2024 Patch 17H healthy choice baseline",
                "",
                "This patch creates the healthy biological baseline subset with strict same-trial raw action codes.",
                "",
                "Policy:",
                "",
                "- include only LRA control rows",
                "- include only native `VTELap.ChoicePoint` VTE labels",
                "- exclude DREADD perturbation rows",
                "- retain raw event codes",
                "- do not infer left/right choice direction",
                "",
                f"- Input rows: `{len(df)}`",
                f"- Output rows: `{len(baseline)}`",
            ]
        ),
        encoding="utf-8",
    )

    print(f"Healthy LRA choice baseline saved: {endpoint_csv}")
    print(f"VTE summary saved: {by_vte_csv}")
    print(f"Outcome summary saved: {by_outcome_csv}")
    print(f"Action-code summary saved: {by_action_csv}")
    print(f"Subject summary saved: {by_subject_csv}")
    print(f"Session summary saved: {by_session_csv}")
    print(f"Coverage saved: {coverage_csv}")
    print(f"Metadata saved: {meta_json}")
    print(f"Report saved: {report_md}")
    print(f"Input rows: {len(df)}")
    print(f"Output rows: {len(baseline)}")

    return meta


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Redish LRA 2024 healthy control choice baseline from Patch 17G."
    )
    parser.add_argument("--same-trial-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)

    args = parser.parse_args()

    build_lra_healthy_choice_baseline(
        same_trial_csv=args.same_trial_csv,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
