from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DATASET_ID = "redish_lra_2024"
PATCH = "18B"

OUTPUT_ENRICHED = "Table_LRA_BioSynth18B_Comparable_Rows_Validated.csv"
OUTPUT_CHECKS = "Table_LRA_BioSynth18B_Validation_Checks.csv"
OUTPUT_DIRECTION = "Table_LRA_BioSynth18B_VTE_Direction_Contrast.csv"
OUTPUT_ACTION = "Table_LRA_BioSynth18B_Action_Namespace_Audit.csv"
OUTPUT_COVERAGE = "Table_LRA_BioSynth18B_Field_Coverage.csv"
OUTPUT_CONCLUSION = "Table_LRA_BioSynth18B_Conclusion.csv"
OUTPUT_META = "lra_biosynth18b_validation_meta.json"
OUTPUT_REPORT = "LRA_BioSynth18B_Validation_Report.md"


REQUIRED_FIELDS = [
    "source",
    "dataset_id",
    "task_family",
    "subject_or_seed",
    "session_or_run",
    "trial",
    "decision_stage",
    "choice_point_id",
    "action_namespace",
    "chosen_action",
    "action_comparison_policy",
    "outcome",
    "reward",
    "cost",
    "native_cost_bin",
    "comparable_cost_bin",
    "dwell_proxy",
    "deliberation_proxy",
    "dwell_z",
    "deliberation_z",
    "vte_binary",
    "source_file",
]


def _clean_text(value: Any) -> str:
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "null"}:
        return ""
    return text


def _norm(value: Any) -> str:
    return _clean_text(value).lower().replace(" ", "_").replace("-", "_")


def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _finite_mean(series: pd.Series) -> float:
    numeric = _to_numeric(series)
    finite = numeric[np.isfinite(numeric)]
    if finite.empty:
        return np.nan
    return float(finite.mean())


def _sign(value: float) -> int:
    if not np.isfinite(value):
        return 0
    if abs(value) < 1e-12:
        return 0
    return 1 if value > 0 else -1


def _canonical_stage(value: Any) -> str:
    text = _norm(value)
    if text in {"junction", "choice", "choice_zone", "choicepoint", "choice_point"}:
        return "choice_point"
    return text or "choice_point"


def _stage_mapping_policy(raw: Any, canonical: str) -> str:
    raw_text = _norm(raw)
    if raw_text == canonical:
        return "no_mapping_needed"
    if raw_text == "junction" and canonical == "choice_point":
        return "synthetic_junction_mapped_to_choice_point"
    if raw_text in {"choice", "choice_zone", "choicepoint"} and canonical == "choice_point":
        return "choice_alias_mapped_to_choice_point"
    return "stage_name_preserved_or_unknown"


def _status_row(
    check_name: str,
    status: str,
    severity: str,
    detail: str,
    n_rows: int | None = None,
) -> dict[str, Any]:
    return {
        "check_name": check_name,
        "status": status,
        "severity": severity,
        "n_rows": n_rows if n_rows is not None else "",
        "detail": detail,
    }


def _field_coverage(df: pd.DataFrame, fields: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    total = len(df)

    for field in fields:
        if field not in df.columns:
            nonempty = 0
        else:
            s = df[field]
            if pd.api.types.is_numeric_dtype(s):
                nonempty = int(s.notna().sum())
            else:
                nonempty = int(
                    s.fillna("")
                    .astype(str)
                    .map(lambda x: x.strip().lower() not in {"", "nan", "none", "null"})
                    .sum()
                )

        rows.append(
            {
                "field": field,
                "nonempty": nonempty,
                "total": total,
                "coverage": nonempty / total if total else 0.0,
            }
        )

    return pd.DataFrame(rows)


def _unique_nonempty(df: pd.DataFrame, col: str) -> set[str]:
    if col not in df.columns:
        return set()
    return {
        _clean_text(x)
        for x in df[col].dropna().unique().tolist()
        if _clean_text(x) != ""
    }


def _validate_biological_baseline(
    biological_baseline_csv: Path | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    meta: dict[str, Any] = {
        "biological_baseline_csv": str(biological_baseline_csv) if biological_baseline_csv else None,
        "biological_baseline_checked": biological_baseline_csv is not None,
    }

    if biological_baseline_csv is None:
        checks.append(
            _status_row(
                "healthy_control_baseline_source",
                "warn",
                "safety",
                "No biological baseline CSV supplied; comparable table can be checked, but upstream DREADD exclusion cannot be independently verified.",
            )
        )
        meta["baseline_rows"] = None
        return checks, meta

    df = pd.read_csv(biological_baseline_csv, low_memory=False)
    meta["baseline_rows"] = int(len(df))

    if "cohort" in df.columns:
        cohorts = {_norm(x) for x in _unique_nonempty(df, "cohort")}
        bad = sorted(cohorts - {"lra"})
        if bad:
            checks.append(
                _status_row(
                    "baseline_cohort_is_lra_only",
                    "fail",
                    "safety",
                    f"Unexpected cohort values: {bad}",
                    len(df),
                )
            )
        else:
            checks.append(
                _status_row(
                    "baseline_cohort_is_lra_only",
                    "pass",
                    "safety",
                    "All baseline rows are LRA cohort.",
                    len(df),
                )
            )
    else:
        checks.append(
            _status_row(
                "baseline_cohort_is_lra_only",
                "warn",
                "safety",
                "Column `cohort` is absent; cohort cannot be checked directly.",
                len(df),
            )
        )

    if "treatment" in df.columns:
        treatments = {_norm(x) for x in _unique_nonempty(df, "treatment")}
        bad = sorted(treatments - {"control"})
        if bad:
            checks.append(
                _status_row(
                    "baseline_treatment_is_control_only",
                    "fail",
                    "safety",
                    f"Unexpected treatment values: {bad}",
                    len(df),
                )
            )
        else:
            checks.append(
                _status_row(
                    "baseline_treatment_is_control_only",
                    "pass",
                    "safety",
                    "All baseline rows are healthy control treatment.",
                    len(df),
                )
            )
    else:
        checks.append(
            _status_row(
                "baseline_treatment_is_control_only",
                "warn",
                "safety",
                "Column `treatment` is absent; treatment cannot be checked directly.",
                len(df),
            )
        )

    if "vte_binary_source" in df.columns:
        sources = _unique_nonempty(df, "vte_binary_source")
        bad = sorted(sources - {"VTELap.ChoicePoint"})
        if bad:
            checks.append(
                _status_row(
                    "baseline_uses_native_vtelap_labels",
                    "fail",
                    "safety",
                    f"Unexpected VTE label sources: {bad}",
                    len(df),
                )
            )
        else:
            checks.append(
                _status_row(
                    "baseline_uses_native_vtelap_labels",
                    "pass",
                    "safety",
                    "Baseline uses native VTELap.ChoicePoint labels.",
                    len(df),
                )
            )
    else:
        checks.append(
            _status_row(
                "baseline_uses_native_vtelap_labels",
                "warn",
                "safety",
                "Column `vte_binary_source` is absent; native VTE label source cannot be checked directly.",
                len(df),
            )
        )

    joined = " ".join(
        [
            " ".join(map(str, _unique_nonempty(df, col)))
            for col in ["cohort", "treatment", "treatment_family", "biological_comparison_role"]
            if col in df.columns
        ]
    ).lower()

    if any(token in joined for token in ["dreadd", "dcz", "veh", "mpfc"]):
        checks.append(
            _status_row(
                "baseline_contains_no_dreadd_or_vehicle_terms",
                "fail",
                "safety",
                "DREADD/vehicle-related terms were detected in baseline metadata.",
                len(df),
            )
        )
    else:
        checks.append(
            _status_row(
                "baseline_contains_no_dreadd_or_vehicle_terms",
                "pass",
                "safety",
                "No DREADD/vehicle-related terms detected in baseline metadata.",
                len(df),
            )
        )

    return checks, meta


def _prepare_comparable(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for field in REQUIRED_FIELDS:
        if field not in out.columns:
            out[field] = np.nan

    if out["action_namespace"].isna().all():
        out["action_namespace"] = np.where(
            out["source"].astype(str) == "biological",
            "raw_lra_event_code",
            "synthetic_left_right_or_model_native",
        )

    if out["action_comparison_policy"].isna().all():
        out["action_comparison_policy"] = "compare_only_if_action_namespace_matches"

    for col in [
        "reward",
        "cost",
        "dwell_proxy",
        "deliberation_proxy",
        "dwell_z",
        "deliberation_z",
        "vte_binary",
    ]:
        out[col] = _to_numeric(out[col])

    out["decision_stage_raw"] = out["decision_stage"].map(_clean_text)
    out["decision_stage_canonical"] = out["decision_stage_raw"].map(_canonical_stage)
    out["decision_stage_mapping_policy"] = [
        _stage_mapping_policy(raw, canonical)
        for raw, canonical in zip(out["decision_stage_raw"], out["decision_stage_canonical"])
    ]

    return out


def _build_action_audit(df: pd.DataFrame) -> tuple[pd.DataFrame, bool, str]:
    audit = (
        df.groupby(["source", "action_namespace", "chosen_action"], dropna=False)
        .size()
        .reset_index(name="n_rows")
        .sort_values(["source", "action_namespace", "chosen_action"])
        .reset_index(drop=True)
    )

    namespaces_by_source = (
        df.groupby("source", dropna=False)["action_namespace"]
        .apply(lambda x: sorted({_clean_text(v) for v in x if _clean_text(v)}))
        .to_dict()
    )

    all_namespaces = {
        namespace
        for namespaces in namespaces_by_source.values()
        for namespace in namespaces
        if namespace
    }

    action_labels_comparable = len(all_namespaces) == 1
    status = (
        "comparable_namespace_match"
        if action_labels_comparable
        else "not_comparable_namespace_mismatch"
    )

    audit["action_label_comparison_status"] = status
    return audit, action_labels_comparable, status


def _build_vte_direction_contrast(df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "reward",
        "dwell_proxy",
        "deliberation_proxy",
        "dwell_z",
        "deliberation_z",
    ]

    rows: list[dict[str, Any]] = []

    for stage in sorted(df["decision_stage_canonical"].dropna().unique()):
        stage_df = df[df["decision_stage_canonical"] == stage]

        for metric in metrics:
            source_effects: dict[str, float] = {}

            for source in ["biological", "synthetic"]:
                src = stage_df[stage_df["source"] == source]
                nonvte = src[src["vte_binary"] == 0]
                vte = src[src["vte_binary"] == 1]

                nonvte_mean = _finite_mean(nonvte[metric]) if metric in nonvte.columns else np.nan
                vte_mean = _finite_mean(vte[metric]) if metric in vte.columns else np.nan
                effect = vte_mean - nonvte_mean if np.isfinite(nonvte_mean) and np.isfinite(vte_mean) else np.nan
                source_effects[source] = effect

            bio = source_effects.get("biological", np.nan)
            syn = source_effects.get("synthetic", np.nan)
            bio_sign = _sign(bio)
            syn_sign = _sign(syn)

            if bio_sign == 0 or syn_sign == 0:
                status = "insufficient_or_zero_effect"
                sign_agreement = np.nan
            elif bio_sign == syn_sign:
                status = "sign_match"
                sign_agreement = True
            else:
                status = "sign_mismatch"
                sign_agreement = False

            rows.append(
                {
                    "decision_stage_canonical": stage,
                    "metric": metric,
                    "biological_vte_minus_nonvte": bio,
                    "synthetic_vte_minus_nonvte": syn,
                    "biological_sign": bio_sign,
                    "synthetic_sign": syn_sign,
                    "sign_agreement": sign_agreement,
                    "comparison_status": status,
                }
            )

    return pd.DataFrame(rows)


def _build_conclusion(
    checks: pd.DataFrame,
    direction: pd.DataFrame,
    action_labels_comparable: bool,
    action_status: str,
    comparable: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    sources = set(comparable["source"].dropna().astype(str))
    rows.append(
        {
            "criterion": "source_presence",
            "status": "pass" if {"biological", "synthetic"}.issubset(sources) else "fail",
            "interpretation": f"Observed sources: {sorted(sources)}",
        }
    )

    safety_failures = checks[(checks["severity"] == "safety") & (checks["status"] == "fail")]
    rows.append(
        {
            "criterion": "healthy_control_safety",
            "status": "pass" if safety_failures.empty else "fail",
            "interpretation": (
                "Biological baseline is restricted to healthy LRA control."
                if safety_failures.empty
                else "Biological baseline failed safety checks."
            ),
        }
    )

    rows.append(
        {
            "criterion": "action_label_comparability",
            "status": "not_comparable" if not action_labels_comparable else "comparable",
            "interpretation": action_status,
        }
    )

    canonical_stages = sorted(set(comparable["decision_stage_canonical"].dropna().astype(str)))
    rows.append(
        {
            "criterion": "stage_mapping",
            "status": "pass" if canonical_stages == ["choice_point"] else "warn",
            "interpretation": f"Canonical decision stages: {canonical_stages}",
        }
    )

    for metric in ["reward", "dwell_z", "deliberation_z"]:
        sub = direction[
            (direction["decision_stage_canonical"] == "choice_point")
            & (direction["metric"] == metric)
        ]
        if sub.empty:
            status = "missing"
            interpretation = "No contrast row available."
        else:
            row = sub.iloc[0]
            status = str(row["comparison_status"])
            interpretation = (
                f"biological={row['biological_vte_minus_nonvte']}; "
                f"synthetic={row['synthetic_vte_minus_nonvte']}"
            )

        rows.append(
            {
                "criterion": f"vte_direction_{metric}",
                "status": status,
                "interpretation": interpretation,
            }
        )

    reward_row = direction[
        (direction["decision_stage_canonical"] == "choice_point")
        & (direction["metric"] == "reward")
    ]
    reward_match = (
        not reward_row.empty
        and str(reward_row.iloc[0]["comparison_status"]) == "sign_match"
    )

    rows.append(
        {
            "criterion": "overall",
            "status": (
                "diagnostic_signature_match"
                if reward_match and action_labels_comparable
                else "diagnostic_only_not_direct_task_match"
            ),
            "interpretation": (
                "Use as direct action/task comparison only if action labels and reward-direction signatures match."
            ),
        }
    )

    return pd.DataFrame(rows)


def _write_report(
    output_path: Path,
    meta: dict[str, Any],
    conclusion: pd.DataFrame,
    direction: pd.DataFrame,
) -> None:
    lines = [
        "# LRA BioSynth Patch 18B validation report",
        "",
        "Purpose: validate Patch 18A as a safety-constrained biological-vs-synthetic diagnostic comparison.",
        "",
        "Policy:",
        "",
        "- biological baseline must be healthy LRA control only",
        "- DREADD and vehicle perturbation rows are not eligible for healthy baseline comparison",
        "- raw LRA action codes are not mapped to left/right",
        "- synthetic action labels are not compared to biological raw event codes unless namespaces match",
        "- synthetic `junction` may be canonically interpreted as `choice_point` for VTE-level summaries",
        "- direct movement-level or task-level equivalence is not claimed",
        "",
        f"- Comparable rows: `{meta['n_rows']}`",
        f"- Biological rows: `{meta['n_biological_rows']}`",
        f"- Synthetic rows: `{meta['n_synthetic_rows']}`",
        f"- Action labels comparable: `{meta['action_labels_comparable']}`",
        f"- Direct task comparison supported: `{meta['direct_task_comparison_supported']}`",
        "",
        "Conclusion:",
        "",
        conclusion.to_string(index=False),
        "",
        "VTE direction contrast:",
        "",
        direction.to_string(index=False),
        "",
    ]

    output_path.write_text("\n".join(lines), encoding="utf-8")


def validate_lra_biosynth_comparability(
    comparability_csv: str | Path,
    output_dir: str | Path,
    biological_baseline_csv: str | Path | None = None,
    strict: bool = True,
) -> dict[str, Any]:
    comparability_csv = Path(comparability_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    biological_baseline_path = Path(biological_baseline_csv) if biological_baseline_csv else None

    raw = pd.read_csv(comparability_csv, low_memory=False)
    comparable = _prepare_comparable(raw)

    baseline_checks, baseline_meta = _validate_biological_baseline(biological_baseline_path)

    checks: list[dict[str, Any]] = baseline_checks

    n_bio = int((comparable["source"] == "biological").sum())
    n_syn = int((comparable["source"] == "synthetic").sum())

    checks.append(
        _status_row(
            "comparable_has_biological_rows",
            "pass" if n_bio > 0 else "fail",
            "safety",
            f"Biological rows: {n_bio}",
            n_bio,
        )
    )
    checks.append(
        _status_row(
            "comparable_has_synthetic_rows",
            "pass" if n_syn > 0 else "fail",
            "safety",
            f"Synthetic rows: {n_syn}",
            n_syn,
        )
    )

    bio = comparable[comparable["source"] == "biological"]
    if not bio.empty:
        bio_datasets = {_clean_text(x) for x in bio["dataset_id"].dropna().unique()}
        checks.append(
            _status_row(
                "biological_dataset_is_redish_lra_2024",
                "pass" if bio_datasets == {DATASET_ID} else "fail",
                "safety",
                f"Biological dataset_id values: {sorted(bio_datasets)}",
                len(bio),
            )
        )

        bio_tasks = {_clean_text(x) for x in bio["task_family"].dropna().unique()}
        checks.append(
            _status_row(
                "biological_task_is_left_right_alternate",
                "pass" if bio_tasks == {"left_right_alternate"} else "warn",
                "safety",
                f"Biological task_family values: {sorted(bio_tasks)}",
                len(bio),
            )
        )

    action_audit, action_labels_comparable, action_status = _build_action_audit(comparable)
    direction = _build_vte_direction_contrast(comparable)

    checks_df = pd.DataFrame(checks)
    conclusion = _build_conclusion(
        checks=checks_df,
        direction=direction,
        action_labels_comparable=action_labels_comparable,
        action_status=action_status,
        comparable=comparable,
    )
    coverage = _field_coverage(comparable, REQUIRED_FIELDS + [
        "decision_stage_raw",
        "decision_stage_canonical",
        "decision_stage_mapping_policy",
    ])

    enriched_path = output_dir / OUTPUT_ENRICHED
    checks_path = output_dir / OUTPUT_CHECKS
    direction_path = output_dir / OUTPUT_DIRECTION
    action_path = output_dir / OUTPUT_ACTION
    coverage_path = output_dir / OUTPUT_COVERAGE
    conclusion_path = output_dir / OUTPUT_CONCLUSION
    meta_path = output_dir / OUTPUT_META
    report_path = output_dir / OUTPUT_REPORT

    comparable.to_csv(enriched_path, index=False)
    checks_df.to_csv(checks_path, index=False)
    direction.to_csv(direction_path, index=False)
    action_audit.to_csv(action_path, index=False)
    coverage.to_csv(coverage_path, index=False)
    conclusion.to_csv(conclusion_path, index=False)

    n_safety_failures = int(
        ((checks_df["severity"] == "safety") & (checks_df["status"] == "fail")).sum()
    )

    reward_row = direction[
        (direction["decision_stage_canonical"] == "choice_point")
        & (direction["metric"] == "reward")
    ]
    reward_sign_match = (
        not reward_row.empty
        and str(reward_row.iloc[0]["comparison_status"]) == "sign_match"
    )

    meta = {
        "dataset_id": DATASET_ID,
        "patch": PATCH,
        "comparability_csv": str(comparability_csv),
        **baseline_meta,
        "n_rows": int(len(comparable)),
        "n_biological_rows": n_bio,
        "n_synthetic_rows": n_syn,
        "n_safety_failures": n_safety_failures,
        "action_labels_comparable": bool(action_labels_comparable),
        "action_label_comparison_status": action_status,
        "direct_task_comparison_supported": bool(action_labels_comparable and reward_sign_match),
        "vte_reward_direction_sign_match": bool(reward_sign_match),
        "policy": (
            "Validate healthy LRA-vs-synthetic comparability as VTE/outcome/proxy diagnostic only. "
            "Do not compare raw LRA event codes against synthetic action labels unless namespaces match."
        ),
        "outputs": {
            "validated_rows": str(enriched_path),
            "checks": str(checks_path),
            "vte_direction": str(direction_path),
            "action_namespace_audit": str(action_path),
            "coverage": str(coverage_path),
            "conclusion": str(conclusion_path),
            "report": str(report_path),
        },
    }

    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    _write_report(report_path, meta, conclusion, direction)

    print(f"Validated comparable rows saved: {enriched_path}")
    print(f"Validation checks saved: {checks_path}")
    print(f"VTE direction contrast saved: {direction_path}")
    print(f"Action namespace audit saved: {action_path}")
    print(f"Coverage saved: {coverage_path}")
    print(f"Conclusion saved: {conclusion_path}")
    print(f"Metadata saved: {meta_path}")
    print(f"Report saved: {report_path}")
    print(f"Rows: {len(comparable)}")
    print(f"Biological rows: {n_bio}")
    print(f"Synthetic rows: {n_syn}")
    print(f"Safety failures: {n_safety_failures}")

    if strict and n_safety_failures > 0:
        raise ValueError(
            f"Patch 18B safety validation failed: {n_safety_failures} safety check(s) failed."
        )

    return meta


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate Redish LRA 2024 biological-vs-synthetic comparability tables."
    )
    parser.add_argument("--comparability-csv", type=Path, required=True)
    parser.add_argument("--biological-baseline-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--no-strict",
        action="store_true",
        help="Write validation tables even if safety checks fail.",
    )

    args = parser.parse_args()

    validate_lra_biosynth_comparability(
        comparability_csv=args.comparability_csv,
        biological_baseline_csv=args.biological_baseline_csv,
        output_dir=args.output_dir,
        strict=not args.no_strict,
    )


if __name__ == "__main__":
    main()