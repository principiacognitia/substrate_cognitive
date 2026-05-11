from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


OUTPUT_ENDPOINT = "Table_LRA_BioSynth18C_Visualization_Decision_Endpoint.csv"
OUTPUT_SUMMARY = "Table_LRA_BioSynth18C_Visualization_Source_Summary.csv"
OUTPUT_DICTIONARY = "Table_LRA_BioSynth18C_Visualization_Field_Dictionary.csv"
OUTPUT_COVERAGE = "Table_LRA_BioSynth18C_Visualization_Field_Coverage.csv"
OUTPUT_META = "lra_biosynth18c_visualization_bridge_meta.json"
OUTPUT_REPORT = "LRA_BioSynth18C_Visualization_Bridge_Report.md"

VISUALIZATION_COLUMNS = [
    "source",
    "dataset_id",
    "trace_origin",
    "task_family",
    "comparison_scope",
    "task_equivalence_status",
    "subject_or_seed",
    "session_or_run",
    "trial",
    "decision_stage",
    "decision_stage_canonical",
    "choice_point_id",
    "action_namespace",
    "chosen_action",
    "action_label_comparison_status",
    "outcome",
    "reward",
    "cost",
    "native_cost_bin",
    "comparable_cost_bin",
    "vte_binary",
    "dwell_proxy",
    "deliberation_proxy",
    "dwell_z",
    "deliberation_z",
    "visualization_level",
    "movement_trace_available",
    "x",
    "y",
    "heading",
    "t",
    "source_file",
]


NUMERIC_COLUMNS = [
    "trial",
    "reward",
    "cost",
    "vte_binary",
    "dwell_proxy",
    "deliberation_proxy",
    "dwell_z",
    "deliberation_z",
    "x",
    "y",
    "heading",
    "t",
]


MISSING_TEXT = {"", "nan", "none", "null", "na", "n/a", "<na>"}


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in MISSING_TEXT:
        return ""
    return text


def _series(df: pd.DataFrame, col: str, default: Any = "") -> pd.Series:
    if col in df.columns:
        return df[col]
    return pd.Series([default] * len(df), index=df.index)


def _text_series(df: pd.DataFrame, col: str, default: Any = "") -> pd.Series:
    return _series(df, col, default).map(_clean_text)


def _numeric_series(df: pd.DataFrame, col: str, default: Any = np.nan) -> pd.Series:
    return pd.to_numeric(_series(df, col, default), errors="coerce")


def _canonical_stage(value: Any) -> str:
    text = _clean_text(value).lower().replace(" ", "_").replace("-", "_")
    mapping = {
        "choice": "choice_point",
        "choicepoint": "choice_point",
        "choice_point": "choice_point",
        "choice_zone": "choice_point",
        "junction": "choice_point",
        "event_centered_choice": "choice_point",
    }
    return mapping.get(text, text or "choice_point")


def _trace_origin(source: Any) -> str:
    text = _clean_text(source).lower()
    if text == "biological":
        return "biological"
    if text == "synthetic":
        return "synthetic"
    return text or "unknown"


def _field_coverage(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    total = len(df)

    for col in VISUALIZATION_COLUMNS:
        if col not in df.columns:
            nonempty = 0
        elif pd.api.types.is_numeric_dtype(df[col]):
            nonempty = int(df[col].notna().sum())
        else:
            nonempty = int(df[col].map(lambda x: _clean_text(x) != "").sum())

        rows.append(
            {
                "field": col,
                "nonempty": nonempty,
                "total": total,
                "coverage": nonempty / total if total else 0.0,
            }
        )

    return pd.DataFrame(rows)


def _summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "source",
                "task_family",
                "decision_stage_canonical",
                "n_rows",
                "n_subjects_or_seeds",
                "n_sessions_or_runs",
                "reward_rate",
                "vte_rate",
                "mean_dwell_z",
                "mean_deliberation_z",
                "movement_trace_available",
                "task_equivalence_status",
            ]
        )

    work = df.copy()
    for col in ["reward", "vte_binary", "dwell_z", "deliberation_z"]:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    return (
        work.groupby(["source", "task_family", "decision_stage_canonical"], dropna=False)
        .agg(
            n_rows=("source", "size"),
            n_subjects_or_seeds=("subject_or_seed", "nunique"),
            n_sessions_or_runs=("session_or_run", "nunique"),
            reward_rate=("reward", "mean"),
            vte_rate=("vte_binary", "mean"),
            mean_dwell_z=("dwell_z", "mean"),
            mean_deliberation_z=("deliberation_z", "mean"),
            movement_trace_available=("movement_trace_available", "max"),
            task_equivalence_status=("task_equivalence_status", "first"),
        )
        .reset_index()
    )


def _field_dictionary() -> pd.DataFrame:
    rows = [
        ("source", "biological or synthetic source marker."),
        ("dataset_id", "Native dataset identifier after adapter normalization."),
        ("trace_origin", "biological or synthetic origin; mirrors source at visualization level."),
        ("task_family", "Task family; LRA remains left_right_alternate, synthetic remains balanced_fork or configured family."),
        ("comparison_scope", "Explicit comparison scope. Patch 18C uses VTE, outcome, dwell, and deliberation only."),
        ("task_equivalence_status", "Direct task/action equivalence status; diagnostic_only_not_direct_task_match by default."),
        ("subject_or_seed", "Biological subject or synthetic seed."),
        ("session_or_run", "Biological session or synthetic run."),
        ("trial", "Trial index where available."),
        ("decision_stage", "Native or adapter-provided stage label."),
        ("decision_stage_canonical", "Canonical stage used for visualization grouping."),
        ("choice_point_id", "Choice-point identifier where available."),
        ("action_namespace", "Namespace for chosen_action. Used to prevent invalid left/right vs raw-event-code comparisons."),
        ("chosen_action", "Native action label. Not directly comparable across namespaces unless status says comparable."),
        ("action_label_comparison_status", "Action label comparability status inherited from 18A/18B or set to not_comparable_namespace_mismatch."),
        ("outcome", "Outcome label where available."),
        ("reward", "Numeric reward/outcome proxy."),
        ("cost", "Comparable cost proxy; nominal balanced cost for LRA 17H."),
        ("native_cost_bin", "Native cost bin before cross-source standardization."),
        ("comparable_cost_bin", "Cross-source cost bin for visualization grouping."),
        ("vte_binary", "Binary VTE label/proxy after adapter policy."),
        ("dwell_proxy", "Dwell or pause proxy."),
        ("deliberation_proxy", "Deliberation proxy; biological IdPhi or synthetic proxy."),
        ("dwell_z", "Within-source/session normalized dwell proxy where available."),
        ("deliberation_z", "Within-source/session normalized deliberation proxy where available."),
        ("visualization_level", "decision_endpoint for Patch 18C."),
        ("movement_trace_available", "False for Patch 18C; this is not movement replay."),
        ("x", "Reserved coordinate field; NaN for decision endpoint visualization."),
        ("y", "Reserved coordinate field; NaN for decision endpoint visualization."),
        ("heading", "Reserved heading field; NaN for decision endpoint visualization."),
        ("t", "Reserved time field; NaN for decision endpoint visualization."),
        ("source_file", "Source CSV path."),
    ]
    return pd.DataFrame(rows, columns=["field", "meaning"])


def _write_report(
    *,
    output_path: Path,
    meta: dict[str, Any],
    coverage: pd.DataFrame,
) -> None:
    lines = [
        "# LRA BioSynth 18C visualization bridge report",
        "",
        "Purpose: export Patch 18B validated biological-vs-synthetic comparable rows into a stable decision-level visualization schema.",
        "",
        "Interpretation:",
        "",
        "- This is a decision endpoint table, not a movement trace.",
        "- Biological LRA raw event codes are not mapped to left/right.",
        "- Synthetic left/right labels are not directly compared with biological raw event codes.",
        "- The intended visualization scope is VTE, reward/outcome, dwell, and deliberation signatures.",
        "",
        "Counts:",
        "",
        f"- input rows: {meta['n_input_rows']}",
        f"- output rows: {meta['n_output_rows']}",
        f"- biological rows: {meta['n_biological_rows']}",
        f"- synthetic rows: {meta['n_synthetic_rows']}",
        f"- movement trace available: {meta['movement_trace_available']}",
        f"- task equivalence status: {meta['task_equivalence_status']}",
        "",
        "Field coverage:",
        "",
        coverage.to_markdown(index=False),
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def build_lra_biosynth_visualization_bridge(
    *,
    input_csv: str | Path,
    output_dir: str | Path,
    comparison_scope: str = "vte_dwell_deliberation_outcome_only",
    task_equivalence_status: str = "diagnostic_only_not_direct_task_match",
) -> dict[str, Any]:
    input_csv = Path(input_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(input_csv, low_memory=False)
    out = pd.DataFrame(index=raw.index)

    out["source"] = _text_series(raw, "source")
    out["dataset_id"] = _text_series(raw, "dataset_id")
    out["trace_origin"] = out["source"].map(_trace_origin)
    out["task_family"] = _text_series(raw, "task_family")
    out["comparison_scope"] = comparison_scope
    out["task_equivalence_status"] = task_equivalence_status

    out["subject_or_seed"] = _text_series(raw, "subject_or_seed")
    out["session_or_run"] = _text_series(raw, "session_or_run")
    out["trial"] = _numeric_series(raw, "trial")
    out["decision_stage"] = _text_series(raw, "decision_stage", "choice_point")
    out["decision_stage_canonical"] = _series(
        raw,
        "decision_stage_canonical",
        "",
    ).map(lambda v: _canonical_stage(v) if _clean_text(v) else "")

    missing_stage = out["decision_stage_canonical"].map(_clean_text) == ""
    out.loc[missing_stage, "decision_stage_canonical"] = out.loc[
        missing_stage, "decision_stage"
    ].map(_canonical_stage)

    out["choice_point_id"] = _text_series(raw, "choice_point_id", "choice_point")
    out["action_namespace"] = _text_series(raw, "action_namespace")
    out["chosen_action"] = _text_series(raw, "chosen_action")

    if "action_label_comparison_status" in raw.columns:
        out["action_label_comparison_status"] = _text_series(
            raw,
            "action_label_comparison_status",
            "not_comparable_namespace_mismatch",
        )
    else:
        out["action_label_comparison_status"] = "not_comparable_namespace_mismatch"

    out["outcome"] = _text_series(raw, "outcome")
    out["reward"] = _numeric_series(raw, "reward")
    out["cost"] = _numeric_series(raw, "cost", 0.0)
    out["native_cost_bin"] = _text_series(raw, "native_cost_bin", "balanced")
    out["comparable_cost_bin"] = _text_series(raw, "comparable_cost_bin", "balanced")

    out["vte_binary"] = _numeric_series(raw, "vte_binary")
    out["dwell_proxy"] = _numeric_series(raw, "dwell_proxy")
    out["deliberation_proxy"] = _numeric_series(raw, "deliberation_proxy")
    out["dwell_z"] = _numeric_series(raw, "dwell_z")
    out["deliberation_z"] = _numeric_series(raw, "deliberation_z")

    out["visualization_level"] = "decision_endpoint"
    out["movement_trace_available"] = False
    out["x"] = np.nan
    out["y"] = np.nan
    out["heading"] = np.nan
    out["t"] = np.nan
    out["source_file"] = _text_series(raw, "source_file", str(input_csv))

    for col in VISUALIZATION_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan

    out = out[VISUALIZATION_COLUMNS]

    for col in NUMERIC_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    endpoint_path = output_dir / OUTPUT_ENDPOINT
    summary_path = output_dir / OUTPUT_SUMMARY
    dictionary_path = output_dir / OUTPUT_DICTIONARY
    coverage_path = output_dir / OUTPUT_COVERAGE
    meta_path = output_dir / OUTPUT_META
    report_path = output_dir / OUTPUT_REPORT

    summary = _summary(out)
    dictionary = _field_dictionary()
    coverage = _field_coverage(out)

    out.to_csv(endpoint_path, index=False)
    summary.to_csv(summary_path, index=False)
    dictionary.to_csv(dictionary_path, index=False)
    coverage.to_csv(coverage_path, index=False)

    meta = {
        "dataset_id": "redish_lra_2024",
        "patch": "18C",
        "input_csv": str(input_csv),
        "n_input_rows": int(len(raw)),
        "n_output_rows": int(len(out)),
        "n_biological_rows": int((out["source"] == "biological").sum()),
        "n_synthetic_rows": int((out["source"] == "synthetic").sum()),
        "comparison_scope": comparison_scope,
        "task_equivalence_status": task_equivalence_status,
        "visualization_level": "decision_endpoint",
        "movement_trace_available": False,
        "policy": (
            "Decision-level visualization bridge only. Do not render as movement replay. "
            "Do not compare biological raw LRA event codes to synthetic left/right action labels."
        ),
        "outputs": {
            "endpoint": str(endpoint_path),
            "summary": str(summary_path),
            "field_dictionary": str(dictionary_path),
            "coverage": str(coverage_path),
            "report": str(report_path),
        },
    }

    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    _write_report(output_path=report_path, meta=meta, coverage=coverage)

    print(f"Visualization endpoint saved: {endpoint_path}")
    print(f"Source summary saved: {summary_path}")
    print(f"Field dictionary saved: {dictionary_path}")
    print(f"Coverage saved: {coverage_path}")
    print(f"Metadata saved: {meta_path}")
    print(f"Report saved: {report_path}")
    print(f"Rows: {meta['n_output_rows']}")
    print(f"Biological rows: {meta['n_biological_rows']}")
    print(f"Synthetic rows: {meta['n_synthetic_rows']}")

    return meta


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Patch 18C LRA BioSynth visualization bridge table."
    )
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--comparison-scope",
        default="vte_dwell_deliberation_outcome_only",
    )
    parser.add_argument(
        "--task-equivalence-status",
        default="diagnostic_only_not_direct_task_match",
    )

    args = parser.parse_args()

    build_lra_biosynth_visualization_bridge(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        comparison_scope=args.comparison_scope,
        task_equivalence_status=args.task_equivalence_status,
    )


if __name__ == "__main__":
    main()