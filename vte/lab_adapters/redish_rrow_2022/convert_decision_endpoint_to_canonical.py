"""Convert Redish Restaurant Row decision endpoints to a canonical biological table.

Patch 16E.

This module does not create a pose trace. It creates a decision-level biological
comparability endpoint from the cleaned Patch 16D output.

Input:
    Table_Redish_RRow_Decision_Endpoint_Usable.csv

Output:
    redish_rrow_canonical_decision_endpoint.csv

The resulting table is intended for Stage 3.2C biological comparability:
choice, cost, reward/outcome, dwell proxy, and lab deliberation proxy.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CANONICAL_DECISION_COLUMNS = [
    "dataset_id",
    "trace_origin",
    "task_family",
    "adapter_stage",
    "comparability_level",
    "subject_id",
    "animal_id",
    "session_id",
    "run_id",
    "trial",
    "sample_index",
    "tick",
    "event_trial",
    "event_type",
    "trial_phase",
    "decision_stage",
    "choice_point_id",
    "restaurant_id",
    "route_id",
    "committed_path",
    "chosen_action",
    "choice",
    "restaurant_outcome",
    "outcome",
    "reward",
    "done",
    "condition",
    "protocol",
    "cost",
    "offer_delay_s",
    "dwell_proxy",
    "pause_time_s",
    "total_site_time_s",
    "run_speed",
    "deliberation_proxy",
    "lab_idphi",
    "lab_avg_dphi",
    "z_lab_idphi_by_session_stage",
    "z_pause_time_by_session_stage",
    "z_total_site_time_by_session_stage",
    "z_offer_delay_by_session_stage",
    "vte_proxy_available",
    "dwell_proxy_available",
    "choice_available",
    "reward_available",
    "geometry_id",
    "coordinate_system",
    "pose_source",
    "x",
    "y",
    "heading",
    "at_choice_point",
    "source_dataset",
    "source_file",
    "source_endpoint_csv",
    "source_row_index",
    "source_zone_slot",
    "source_slot_policy",
    "source_restaurant_visit_id",
    "wrapper_compatibility_note",
]


SUMMARY_NUMERIC_COLUMNS = [
    "reward",
    "offer_delay_s",
    "pause_time_s",
    "total_site_time_s",
    "run_speed",
    "lab_idphi",
    "lab_avg_dphi",
    "deliberation_proxy",
    "dwell_proxy",
]


MISSING_STRINGS = {"", "nan", "none", "null", "na", "n/a", "<na>"}


def _clean_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).strip()
    if text.lower() in MISSING_STRINGS:
        return ""
    return text


def _is_blank(value: Any) -> bool:
    return _clean_str(value) == ""


def _as_float(value: Any) -> float:
    text = _clean_str(value)
    if text == "":
        return float("nan")
    try:
        return float(text)
    except (TypeError, ValueError):
        return float("nan")


def _as_int_or_blank(value: Any) -> int | str:
    number = _as_float(value)
    if math.isnan(number):
        return ""
    return int(number)


def _truthy(value: Any) -> bool:
    text = _clean_str(value).lower()
    return text in {"true", "1", "yes", "y", "t"}


def _first_existing(row: pd.Series, keys: list[str], default: Any = "") -> Any:
    for key in keys:
        if key in row.index and not _is_blank(row[key]):
            return row[key]
    return default


def _run_id(dataset_id: str, subject_id: str, session_id: str) -> str:
    safe_dataset = dataset_id or "redish_rrow_2022"
    safe_subject = subject_id or "unknown_subject"
    safe_session = session_id or "unknown_session"
    return f"{safe_dataset}_{safe_subject}_{safe_session}"


def _choice_point_id(row: pd.Series, restaurant_id: str, decision_stage: str) -> str:
    explicit = _clean_str(row.get("choice_point_id", ""))
    if explicit:
        return explicit

    safe_restaurant = restaurant_id or "unknown"
    safe_stage = decision_stage or "unknown_stage"
    return f"redish_rrow_restaurant_{safe_restaurant}_{safe_stage}"


def _route_id(restaurant_id: str) -> str:
    if restaurant_id:
        return f"restaurant_{restaurant_id}"
    return "restaurant_unknown"


def _zscore(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    mean = numeric.mean(skipna=True)
    std = numeric.std(skipna=True, ddof=0)

    if pd.isna(mean) or pd.isna(std) or std == 0:
        return pd.Series(np.nan, index=values.index)

    return (numeric - mean) / std


def _add_group_zscores(canonical: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["subject_id", "session_id", "decision_stage"]

    z_specs = {
        "lab_idphi": "z_lab_idphi_by_session_stage",
        "pause_time_s": "z_pause_time_by_session_stage",
        "total_site_time_s": "z_total_site_time_by_session_stage",
        "offer_delay_s": "z_offer_delay_by_session_stage",
    }

    for source_col, z_col in z_specs.items():
        if source_col not in canonical.columns:
            canonical[z_col] = np.nan
            continue

        canonical[z_col] = (
            canonical.groupby(group_cols, dropna=False)[source_col]
            .transform(_zscore)
            .astype(float)
        )

    return canonical


def _canonical_row(
    row: pd.Series,
    *,
    input_csv: Path,
    adapter_stage: str,
    task_family: str,
) -> dict[str, Any]:
    dataset_id = _clean_str(row.get("dataset_id", "redish_rrow_2022")) or "redish_rrow_2022"
    subject_id = _clean_str(row.get("subject_id", ""))
    session_id = _clean_str(row.get("session_id", ""))

    decision_stage = _clean_str(row.get("decision_stage", ""))
    stage_decision = _clean_str(row.get("stage_decision", "")).lower()
    restaurant_outcome = _clean_str(row.get("restaurant_outcome", "")).lower()
    restaurant_id = _clean_str(row.get("restaurant_id", ""))

    trial = _as_int_or_blank(row.get("trial", ""))

    reward = _as_float(row.get("reward", ""))
    offer_delay_s = _as_float(row.get("offer_delay_s", ""))
    pause_time_s = _as_float(row.get("pause_time_s", ""))
    total_site_time_s = _as_float(row.get("total_site_time_s", ""))
    run_speed = _as_float(row.get("run_speed", ""))
    lab_idphi = _as_float(row.get("lab_idphi", ""))
    lab_avg_dphi = _as_float(row.get("lab_avg_dphi", ""))

    dwell_proxy = pause_time_s
    deliberation_proxy = lab_idphi

    source_row_index = _first_existing(row, ["source_row_index", "row_index"], "")
    source_zone_slot = _clean_str(row.get("source_zone_slot", ""))
    source_slot_policy = _clean_str(row.get("source_slot_policy", ""))
    source_restaurant_visit_id = _clean_str(row.get("restaurant_visit_id", ""))

    choice_point_id = _choice_point_id(row, restaurant_id, decision_stage)

    return {
        "dataset_id": dataset_id,
        "trace_origin": "biological",
        "task_family": task_family,
        "adapter_stage": adapter_stage,
        "comparability_level": "decision_endpoint_proxy",
        "subject_id": subject_id,
        "animal_id": subject_id,
        "session_id": session_id,
        "run_id": _run_id(dataset_id, subject_id, session_id),
        "trial": trial,
        "sample_index": 0,
        "tick": 0,
        "event_trial": trial,
        "event_type": "biological_decision_endpoint",
        "trial_phase": decision_stage,
        "decision_stage": decision_stage,
        "choice_point_id": choice_point_id,
        "restaurant_id": restaurant_id,
        "route_id": _route_id(restaurant_id),
        "committed_path": stage_decision,
        "chosen_action": stage_decision,
        "choice": stage_decision,
        "restaurant_outcome": restaurant_outcome,
        "outcome": restaurant_outcome,
        "reward": reward,
        "done": True,
        "condition": session_id,
        "protocol": "redish_rrow_2022",
        "cost": offer_delay_s,
        "offer_delay_s": offer_delay_s,
        "dwell_proxy": dwell_proxy,
        "pause_time_s": pause_time_s,
        "total_site_time_s": total_site_time_s,
        "run_speed": run_speed,
        "deliberation_proxy": deliberation_proxy,
        "lab_idphi": lab_idphi,
        "lab_avg_dphi": lab_avg_dphi,
        "z_lab_idphi_by_session_stage": np.nan,
        "z_pause_time_by_session_stage": np.nan,
        "z_total_site_time_by_session_stage": np.nan,
        "z_offer_delay_by_session_stage": np.nan,
        "vte_proxy_available": not math.isnan(lab_idphi),
        "dwell_proxy_available": not math.isnan(dwell_proxy),
        "choice_available": stage_decision != "",
        "reward_available": not math.isnan(reward),
        "geometry_id": "redish_rrow_restaurant_row",
        "coordinate_system": "not_applicable_decision_endpoint",
        "pose_source": "none_decision_endpoint",
        "x": np.nan,
        "y": np.nan,
        "heading": np.nan,
        "at_choice_point": True,
        "source_dataset": "redish_rrow_2022",
        "source_file": _clean_str(row.get("source_file", "")),
        "source_endpoint_csv": str(input_csv),
        "source_row_index": source_row_index,
        "source_zone_slot": source_zone_slot,
        "source_slot_policy": source_slot_policy,
        "source_restaurant_visit_id": source_restaurant_visit_id,
        "wrapper_compatibility_note": (
            "Decision-level biological endpoint. Not a pose trace; do not feed "
            "to Stage 3.2 VTE trace wrapper as trajectory data."
        ),
    }


def _filter_decision_rows(
    df: pd.DataFrame,
    *,
    require_stage_applicable: bool,
) -> pd.DataFrame:
    filtered = df.copy()

    if require_stage_applicable and "stage_applicable" in filtered.columns:
        mask = filtered["stage_applicable"].map(_truthy)
        filtered = filtered.loc[mask].copy()

    required = [
        "subject_id",
        "session_id",
        "trial",
        "decision_stage",
        "stage_decision",
        "restaurant_outcome",
        "reward",
        "offer_delay_s",
        "pause_time_s",
        "total_site_time_s",
        "lab_idphi",
        "lab_avg_dphi",
    ]

    existing_required = [col for col in required if col in filtered.columns]
    for col in existing_required:
        filtered = filtered.loc[~filtered[col].map(_is_blank)].copy()

    return filtered


def _make_summary(
    canonical: pd.DataFrame,
    group_cols: list[str],
) -> pd.DataFrame:
    if canonical.empty:
        return pd.DataFrame(columns=group_cols + ["n_rows"])

    grouped = canonical.groupby(group_cols, dropna=False)

    rows: list[dict[str, Any]] = []
    for key, part in grouped:
        if not isinstance(key, tuple):
            key = (key,)

        row = {col: value for col, value in zip(group_cols, key)}
        row["n_rows"] = int(len(part))
        row["n_subjects"] = int(part["subject_id"].nunique(dropna=True))
        row["n_sessions"] = int(part["session_id"].nunique(dropna=True))
        row["reward_rate"] = pd.to_numeric(part["reward"], errors="coerce").mean()

        for col in SUMMARY_NUMERIC_COLUMNS:
            if col in part.columns:
                row[f"mean_{col}"] = pd.to_numeric(part[col], errors="coerce").mean()
            else:
                row[f"mean_{col}"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def _add_delay_bin(canonical: pd.DataFrame) -> pd.DataFrame:
    out = canonical.copy()
    delay = pd.to_numeric(out["offer_delay_s"], errors="coerce")

    bins = [-np.inf, 4, 9, 14, 19, 24, np.inf]
    labels = [
        "delay_00_04",
        "delay_05_09",
        "delay_10_14",
        "delay_15_19",
        "delay_20_24",
        "delay_25_plus",
    ]

    out["delay_bin"] = pd.cut(delay, bins=bins, labels=labels).astype("object")
    out.loc[delay.isna(), "delay_bin"] = "missing"
    return out


def _write_report(
    *,
    output_path: Path,
    input_csv: Path,
    canonical_path: Path,
    meta: dict[str, Any],
) -> None:
    lines = [
        "# Redish RRow canonical biological decision endpoint",
        "",
        "Patch 16E converts the cleaned Redish Restaurant Row decision endpoint into a canonical biological decision table.",
        "",
        "This is not a trajectory trace and should not be interpreted as biological pose replay.",
        "",
        "## Input",
        "",
        f"- `{input_csv}`",
        "",
        "## Output",
        "",
        f"- `{canonical_path}`",
        "",
        "## Counts",
        "",
        f"- input rows: {meta['n_input_rows']}",
        f"- filtered decision rows: {meta['n_filtered_rows']}",
        f"- canonical rows: {meta['n_canonical_rows']}",
        f"- subjects: {meta['n_subjects']}",
        f"- sessions: {meta['n_sessions']}",
        "",
        "## Comparability fields",
        "",
        "- `choice`, `chosen_action`, `committed_path`: biological decision label.",
        "- `reward`, `restaurant_outcome`, `outcome`: outcome labels.",
        "- `cost`, `offer_delay_s`: delay/cost field.",
        "- `dwell_proxy`, `pause_time_s`, `total_site_time_s`: dwell/hesitation proxy.",
        "- `deliberation_proxy`, `lab_idphi`, `lab_avg_dphi`: lab-side deliberation proxy.",
        "",
        "## Methodological note",
        "",
        "The endpoint is suitable for Stage 3.2C decision-level comparability. It is not suitable for direct Stage 3.2 pose-trace wrapper execution.",
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def convert_decision_endpoint_to_canonical(
    *,
    input_csv: str | Path,
    output_dir: str | Path,
    dataset_id: str = "redish_rrow_2022",
    adapter_stage: str = "patch16e",
    task_family: str = "restaurant_row",
    require_stage_applicable: bool = True,
) -> dict[str, Any]:
    input_csv = Path(input_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source = pd.read_csv(input_csv, dtype=str)
    if "dataset_id" not in source.columns:
        source["dataset_id"] = dataset_id
    else:
        source["dataset_id"] = source["dataset_id"].map(lambda v: _clean_str(v) or dataset_id)

    filtered = _filter_decision_rows(
        source,
        require_stage_applicable=require_stage_applicable,
    )

    canonical_rows = [
        _canonical_row(
            row,
            input_csv=input_csv,
            adapter_stage=adapter_stage,
            task_family=task_family,
        )
        for _, row in filtered.iterrows()
    ]

    canonical = pd.DataFrame(canonical_rows)

    if canonical.empty:
        canonical = pd.DataFrame(columns=CANONICAL_DECISION_COLUMNS)
    else:
        for col in CANONICAL_DECISION_COLUMNS:
            if col not in canonical.columns:
                canonical[col] = np.nan
        canonical = canonical[CANONICAL_DECISION_COLUMNS]
        canonical = _add_group_zscores(canonical)

    canonical_with_delay = _add_delay_bin(canonical) if not canonical.empty else canonical.copy()
    if canonical.empty:
        by_stage = pd.DataFrame()
        by_delay = pd.DataFrame()
        by_subject = pd.DataFrame()
        by_session = pd.DataFrame()
    else:
        by_stage = _make_summary(
            canonical,
            ["decision_stage", "chosen_action", "restaurant_outcome"],
        )
        by_delay = _make_summary(
            canonical_with_delay,
            ["decision_stage", "chosen_action", "restaurant_outcome", "delay_bin"],
        )
        by_subject = _make_summary(
            canonical,
            ["subject_id", "decision_stage", "chosen_action", "restaurant_outcome"],
        )
        by_session = _make_summary(
            canonical,
            ["subject_id", "session_id", "decision_stage"],
        )

    canonical_path = output_dir / "redish_rrow_canonical_decision_endpoint.csv"
    by_stage_path = output_dir / "Table_Redish_RRow_Canonical_By_Stage.csv"
    by_delay_path = output_dir / "Table_Redish_RRow_Canonical_By_Delay.csv"
    by_subject_path = output_dir / "Table_Redish_RRow_Canonical_By_Subject.csv"
    by_session_path = output_dir / "Table_Redish_RRow_Canonical_By_Session.csv"
    meta_path = output_dir / "redish_rrow_canonical_decision_endpoint_meta.json"
    report_path = output_dir / "Redish_RRow_Canonical_Decision_Endpoint_Report.md"

    canonical.to_csv(canonical_path, index=False)
    by_stage.to_csv(by_stage_path, index=False)
    by_delay.to_csv(by_delay_path, index=False)
    by_subject.to_csv(by_subject_path, index=False)
    by_session.to_csv(by_session_path, index=False)

    meta = {
        "dataset_id": dataset_id,
        "adapter_stage": adapter_stage,
        "task_family": task_family,
        "input_csv": str(input_csv),
        "n_input_rows": int(len(source)),
        "n_filtered_rows": int(len(filtered)),
        "n_canonical_rows": int(len(canonical)),
        "n_subjects": int(canonical["subject_id"].nunique()) if not canonical.empty else 0,
        "n_sessions": int(canonical["session_id"].nunique()) if not canonical.empty else 0,
        "require_stage_applicable": bool(require_stage_applicable),
        "outputs": {
            "canonical": str(canonical_path),
            "by_stage": str(by_stage_path),
            "by_delay": str(by_delay_path),
            "by_subject": str(by_subject_path),
            "by_session": str(by_session_path),
            "report": str(report_path),
        },
        "methodological_status": (
            "decision-level biological comparability endpoint; not a pose trace"
        ),
    }

    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    _write_report(
        output_path=report_path,
        input_csv=input_csv,
        canonical_path=canonical_path,
        meta=meta,
    )

    print(f"Canonical decision endpoint saved: {canonical_path}")
    print(f"Stage summary saved: {by_stage_path}")
    print(f"Delay summary saved: {by_delay_path}")
    print(f"Subject summary saved: {by_subject_path}")
    print(f"Session summary saved: {by_session_path}")
    print(f"Metadata saved: {meta_path}")
    print(f"Report saved: {report_path}")
    print(f"Canonical rows: {len(canonical)}")

    return meta


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert Redish Restaurant Row decision endpoint to canonical "
            "biological decision endpoint."
        )
    )
    parser.add_argument(
        "--input-csv",
        required=True,
        help="Path to Table_Redish_RRow_Decision_Endpoint_Usable.csv.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for canonical endpoint outputs.",
    )
    parser.add_argument(
        "--dataset-id",
        default="redish_rrow_2022",
    )
    parser.add_argument(
        "--adapter-stage",
        default="patch16e",
    )
    parser.add_argument(
        "--task-family",
        default="restaurant_row",
    )
    parser.add_argument(
        "--include-inapplicable",
        action="store_true",
        help="Do not filter stage_applicable=False rows. Default keeps only applicable decisions.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    convert_decision_endpoint_to_canonical(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        dataset_id=args.dataset_id,
        adapter_stage=args.adapter_stage,
        task_family=args.task_family,
        require_stage_applicable=not args.include_inapplicable,
    )


if __name__ == "__main__":
    main()