from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DATASET_ID = "redish_lra_2024"


def _finite_float(value: Any) -> float:
    try:
        x = float(value)
    except Exception:
        return float("nan")
    return x if math.isfinite(x) else float("nan")


def _clean_code(value: Any) -> str:
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "null"}:
        return ""
    return text


def _add_next_choice_entry(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["trial"] = pd.to_numeric(out["trial"], errors="coerce")
    out["choice_point_entry_s_audit"] = pd.to_numeric(
        out["choice_point_entry_s_audit"], errors="coerce"
    )
    out["choice_point_exit_s_audit"] = pd.to_numeric(
        out["choice_point_exit_s_audit"], errors="coerce"
    )
    out["primary_event_time_s"] = pd.to_numeric(out["primary_event_time_s"], errors="coerce")
    out["primary_event_latency_s"] = pd.to_numeric(
        out["primary_event_latency_s"], errors="coerce"
    )

    out = out.sort_values(["subject_id", "session_id", "trial"]).reset_index(drop=True)

    out["next_trial"] = out.groupby(["subject_id", "session_id"])["trial"].shift(-1)
    out["next_choice_point_entry_s"] = out.groupby(["subject_id", "session_id"])[
        "choice_point_entry_s_audit"
    ].shift(-1)

    return out


def _classify_same_trial(
    row: pd.Series,
    max_primary_latency_s: float,
    next_entry_slack_s: float,
) -> tuple[str, float]:
    exit_s = _finite_float(row.get("choice_point_exit_s_audit"))
    event_s = _finite_float(row.get("primary_event_time_s"))
    next_entry_s = _finite_float(row.get("next_choice_point_entry_s"))

    if not math.isfinite(exit_s):
        return "missing_choice_exit", np.nan
    if not math.isfinite(event_s):
        return "missing_primary_event_time", np.nan

    latency = event_s - exit_s

    if latency < 0:
        return "event_before_choice_exit", latency

    if math.isfinite(max_primary_latency_s) and latency > max_primary_latency_s:
        return "exceeds_latency_cap", latency

    if math.isfinite(next_entry_s):
        if event_s <= next_entry_s + next_entry_slack_s:
            return "same_trial_event", latency
        return "after_next_choice_entry", latency

    return "same_trial_event_last_lap_latency_only", latency


def _summary(
    df: pd.DataFrame,
    group_cols: list[str],
    output_csv: Path,
) -> pd.DataFrame:
    if df.empty:
        out = pd.DataFrame(columns=group_cols)
        out.to_csv(output_csv, index=False)
        return out

    work = df.copy()

    numeric_cols = [
        "reward",
        "lab_idphi",
        "lab_avg_idphi",
        "vte_binary_for_comparison",
        "primary_event_latency_s",
        "strict_primary_event_latency_s",
        "choice_point_dwell_s",
        "pause_time_s",
        "contingency_correct_l_audit",
        "contingency_correct_r_audit",
        "contingency_correct_a_audit",
    ]

    for col in numeric_cols:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")

    agg = {
        "n_rows": ("dataset_id", "size"),
        "n_subjects": ("subject_id", "nunique"),
        "n_sessions": ("session_id", "nunique"),
        "reward_rate": ("reward", "mean"),
        "vte_rate": ("vte_binary_for_comparison", "mean"),
        "mean_lab_idphi": ("lab_idphi", "mean"),
        "mean_primary_latency_s": ("strict_primary_event_latency_s", "mean"),
    }

    optional = {
        "mean_lab_avg_idphi": ("lab_avg_idphi", "mean"),
        "mean_choice_dwell_s": ("choice_point_dwell_s", "mean"),
        "mean_pause_time_s": ("pause_time_s", "mean"),
        "mean_contingency_l": ("contingency_correct_l_audit", "mean"),
        "mean_contingency_r": ("contingency_correct_r_audit", "mean"),
        "mean_contingency_a": ("contingency_correct_a_audit", "mean"),
    }

    for out_col, spec in optional.items():
        if spec[0] in work.columns:
            agg[out_col] = spec

    out = work.groupby(group_cols, dropna=False).agg(**agg).reset_index()
    out.to_csv(output_csv, index=False)
    return out


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


def filter_lra_choice_direction_same_trial(
    audit_csv: Path,
    output_dir: Path,
    max_primary_latency_s: float = 30.0,
    next_entry_slack_s: float = 0.25,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(audit_csv, low_memory=False)
    df = _add_next_choice_entry(df)

    statuses = df.apply(
        lambda row: _classify_same_trial(
            row,
            max_primary_latency_s=max_primary_latency_s,
            next_entry_slack_s=next_entry_slack_s,
        ),
        axis=1,
    )

    df["strict_event_status"] = [x[0] for x in statuses]
    df["strict_primary_event_latency_s"] = [x[1] for x in statuses]

    df["primary_event_code_label"] = df["primary_event_code_label"].map(_clean_code)
    df["primary_event_kind"] = df["primary_event_kind"].map(_clean_code)

    accepted_statuses = {
        "same_trial_event",
        "same_trial_event_last_lap_latency_only",
    }

    same_trial = df[
        df["strict_event_status"].isin(accepted_statuses)
        & df["primary_event_code_label"].str.len().gt(0)
    ].copy()

    rejected = df[~df.index.isin(same_trial.index)].copy()

    same_trial["chosen_action"] = same_trial["primary_event_code_label"]
    same_trial["chosen_action_namespace"] = "raw_lra_event_code"
    same_trial["chosen_action_status"] = "strict_same_trial_raw_code_not_left_right"
    same_trial["choice_resolution_policy"] = (
        "same-trial raw event code only; no left/right semantic mapping"
    )

    all_csv = output_dir / "redish_lra17g_choice_direction_same_trial_audit.csv"
    same_trial_csv = output_dir / "redish_lra17g_choice_direction_same_trial_usable.csv"
    rejected_csv = output_dir / "redish_lra17g_choice_direction_same_trial_rejected.csv"

    by_status_csv = output_dir / "Table_Redish_LRA17G_By_Strict_Event_Status.csv"
    by_code_csv = output_dir / "Table_Redish_LRA17G_By_Primary_Event_Code.csv"
    by_code_contingency_csv = output_dir / "Table_Redish_LRA17G_By_Code_And_Contingency.csv"
    by_subject_code_csv = output_dir / "Table_Redish_LRA17G_By_Subject_Code.csv"
    coverage_csv = output_dir / "Table_Redish_LRA17G_Field_Coverage.csv"

    df.to_csv(all_csv, index=False)
    same_trial.to_csv(same_trial_csv, index=False)
    rejected.to_csv(rejected_csv, index=False)

    _summary(df, ["strict_event_status"], by_status_csv)

    _summary(
        same_trial,
        ["primary_event_kind", "primary_event_code_label", "outcome"],
        by_code_csv,
    )

    _summary(
        same_trial,
        [
            "primary_event_kind",
            "primary_event_code_label",
            "outcome",
            "contingency_correct_l_audit",
            "contingency_correct_r_audit",
            "contingency_correct_a_audit",
        ],
        by_code_contingency_csv,
    )

    _summary(
        same_trial,
        ["subject_id", "primary_event_kind", "primary_event_code_label", "outcome"],
        by_subject_code_csv,
    )

    _coverage(
        same_trial,
        [
            "subject_id",
            "session_id",
            "trial",
            "outcome",
            "reward",
            "choice_point_exit_s_audit",
            "next_choice_point_entry_s",
            "primary_event_kind",
            "primary_event_code_label",
            "primary_event_time_s",
            "strict_primary_event_latency_s",
            "strict_event_status",
            "chosen_action",
            "vte_binary_for_comparison",
            "lab_idphi",
            "contingency_correct_l_audit",
            "contingency_correct_r_audit",
            "contingency_correct_a_audit",
        ],
        coverage_csv,
    )

    meta = {
        "dataset_id": DATASET_ID,
        "patch": "17G",
        "input_csv": str(audit_csv),
        "max_primary_latency_s": max_primary_latency_s,
        "next_entry_slack_s": next_entry_slack_s,
        "policy": (
            "Keep only primary reward/error event codes that occur after ChoicePointExit "
            "and before the next ChoicePointEntry, with optional latency cap. "
            "Do not map raw codes to left/right."
        ),
        "n_input_rows": int(len(df)),
        "n_same_trial_rows": int(len(same_trial)),
        "n_rejected_rows": int(len(rejected)),
        "outputs": {
            "all": str(all_csv),
            "same_trial": str(same_trial_csv),
            "rejected": str(rejected_csv),
            "by_status": str(by_status_csv),
            "by_code": str(by_code_csv),
            "by_code_contingency": str(by_code_contingency_csv),
            "by_subject_code": str(by_subject_code_csv),
            "coverage": str(coverage_csv),
        },
    }

    meta_json = output_dir / "redish_lra17g_choice_direction_same_trial_meta.json"
    report_md = output_dir / "Redish_LRA17G_Choice_Direction_Same_Trial_Report.md"

    meta_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    report_md.write_text(
        "\n".join(
            [
                "# Redish LRA 2024 Patch 17G same-trial choice event filter",
                "",
                "This patch filters Patch 17F raw event-code alignment to same-trial events.",
                "",
                "Accepted event criterion:",
                "",
                "- event time is after `ChoicePointExit`",
                "- event time is before the next trial's `ChoicePointEntry`, with slack",
                "- event latency does not exceed the configured cap",
                "",
                "The output still uses raw event codes only; no left/right mapping is inferred.",
                "",
                f"- Input rows: `{len(df)}`",
                f"- Same-trial rows: `{len(same_trial)}`",
                f"- Rejected rows: `{len(rejected)}`",
                f"- Max latency: `{max_primary_latency_s}` s",
                f"- Next-entry slack: `{next_entry_slack_s}` s",
            ]
        ),
        encoding="utf-8",
    )

    print(f"Strict same-trial audit saved: {all_csv}")
    print(f"Strict same-trial usable endpoint saved: {same_trial_csv}")
    print(f"Rejected rows saved: {rejected_csv}")
    print(f"Status summary saved: {by_status_csv}")
    print(f"Code summary saved: {by_code_csv}")
    print(f"Code-contingency summary saved: {by_code_contingency_csv}")
    print(f"Subject-code summary saved: {by_subject_code_csv}")
    print(f"Coverage saved: {coverage_csv}")
    print(f"Metadata saved: {meta_json}")
    print(f"Report saved: {report_md}")
    print(f"Input rows: {len(df)}")
    print(f"Same-trial rows: {len(same_trial)}")
    print(f"Rejected rows: {len(rejected)}")

    return meta


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter Redish LRA 17F raw event-code audit to strict same-trial events."
    )
    parser.add_argument("--audit-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-primary-latency-s", type=float, default=30.0)
    parser.add_argument("--next-entry-slack-s", type=float, default=0.25)

    args = parser.parse_args()

    filter_lra_choice_direction_same_trial(
        audit_csv=args.audit_csv,
        output_dir=args.output_dir,
        max_primary_latency_s=args.max_primary_latency_s,
        next_entry_slack_s=args.next_entry_slack_s,
    )


if __name__ == "__main__":
    main()
