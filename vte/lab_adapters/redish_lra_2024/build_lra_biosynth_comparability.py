from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


OUTPUT_COMPARABLE = "Table_LRA_BioSynth_Comparable_Decision_Rows.csv"
OUTPUT_BY_VTE = "Table_LRA_BioSynth_Summary_By_Source_VTE.csv"
OUTPUT_BY_OUTCOME = "Table_LRA_BioSynth_Summary_By_Source_Outcome.csv"
OUTPUT_ACTION_AUDIT = "Table_LRA_BioSynth_Action_Namespace_Audit.csv"
OUTPUT_VTE_CONTRAST = "Table_LRA_BioSynth_VTE_Contrast_By_Source.csv"
OUTPUT_COVERAGE = "Table_LRA_BioSynth_Field_Coverage.csv"
OUTPUT_META = "lra_biosynth_comparability_meta.json"
OUTPUT_REPORT = "LRA_BioSynth_Comparability_Report.md"

COMPARABLE_COLUMNS = [
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


def _first_existing(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    colset = set(columns)
    for candidate in candidates:
        if candidate in colset:
            return candidate
    return None


def _empty_series(n: int, value=np.nan) -> pd.Series:
    return pd.Series([value] * n)


def _as_string_series(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str)


def _as_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _normalize_text(series: pd.Series) -> pd.Series:
    return (
        _as_string_series(series)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
    )


def _normalize_outcome(series: pd.Series) -> pd.Series:
    text = _normalize_text(series)
    mapping = {
        "correct": "correct",
        "rewarded": "correct",
        "reward": "correct",
        "earned": "correct",
        "earn": "correct",
        "1": "correct",
        "true": "correct",
        "error": "error",
        "incorrect": "error",
        "unrewarded": "error",
        "no_reward": "error",
        "0": "error",
        "false": "error",
        "quit": "quit",
        "skip": "skip",
    }
    return text.map(lambda x: mapping.get(x, x))


def _reward_to_outcome(reward: pd.Series) -> pd.Series:
    numeric = _as_numeric(reward)
    out = pd.Series([""] * len(numeric), dtype="object")
    out.loc[numeric > 0] = "correct"
    out.loc[numeric == 0] = "error"
    return out


def _normalize_action(series: pd.Series) -> pd.Series:
    text = _normalize_text(series)
    return text.replace({"": np.nan})


def _standardize_stage(series: pd.Series, default: str = "choice_point") -> pd.Series:
    text = _normalize_text(series)
    text = text.replace("", default)
    mapping = {
        "choice": "choice_point",
        "choice_zone": "choice_point",
        "choicepoint": "choice_point",
        "choice_point": "choice_point",
        "event_centered_choice": "choice_point",
    }
    return text.map(lambda x: mapping.get(x, x))


def _standardize_cost_bin(series: pd.Series) -> pd.Series:
    text = _normalize_text(series)

    def convert(value: str) -> str:
        if value in {"", "nan", "none", "null"}:
            return "balanced"
        if value in {"balanced", "equal", "same", "flat"}:
            return "balanced"
        if "low" in value or "easy" in value:
            return "low"
        if "medium" in value or "mid" in value:
            return "medium"
        if "high" in value or "hard" in value:
            return "high"
        return value

    return text.map(convert)


def _zscore_within_groups(
    df: pd.DataFrame,
    value_col: str,
    output_col: str,
    group_cols: list[str],
) -> pd.DataFrame:
    out = df.copy()

    values = _as_numeric(out[value_col]) if value_col in out.columns else _empty_series(len(out))
    z = pd.Series(np.nan, index=out.index, dtype="float64")

    for _, idx in out.groupby(group_cols, dropna=False).groups.items():
        idx = list(idx)
        x = values.loc[idx]
        finite = x[np.isfinite(x)]
        if len(finite) < 2:
            continue
        sd = finite.std(ddof=0)
        if not np.isfinite(sd) or sd == 0:
            z.loc[idx] = 0.0
        else:
            z.loc[idx] = (x - finite.mean()) / sd

    if output_col in out.columns:
        old = _as_numeric(out[output_col])
        out[output_col] = old.where(np.isfinite(old), z)
    else:
        out[output_col] = z

    return out


def _load_biological_lra(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    n = len(raw)

    reward = _as_numeric(raw.get("reward", _empty_series(n)))

    action_col = _first_existing(raw.columns, ["native_action_code", "chosen_action", "action"])
    outcome_col = _first_existing(raw.columns, ["outcome", "restaurant_outcome"])

    cost_col = _first_existing(raw.columns, ["cost", "condition_cost"])
    cost = _as_numeric(raw[cost_col]) if cost_col else pd.Series([0.0] * n)

    native_cost_col = _first_existing(raw.columns, ["native_cost_bin", "cost_bin", "condition"])
    native_cost_bin = raw[native_cost_col] if native_cost_col else pd.Series(["balanced"] * n)

    dwell_col = _first_existing(
        raw.columns,
        ["dwell_proxy", "choice_point_dwell_s", "pause_time_s"],
    )
    deliberation_col = _first_existing(
        raw.columns,
        ["deliberation_proxy", "lab_idphi", "idphi"],
    )
    dwell_z_col = _first_existing(
        raw.columns,
        ["dwell_z", "z_dwell_proxy_by_session", "z_pause_time_by_session"],
    )
    deliberation_z_col = _first_existing(
        raw.columns,
        ["deliberation_z", "z_lab_idphi_by_session", "z_lab_idphi_by_session_stage", "z_idphi"],
    )
    vte_col = _first_existing(raw.columns, ["vte_binary", "vte_binary_for_comparison", "lab_vte_binary"])

    df = pd.DataFrame(index=raw.index)
    df["source"] = "biological"
    df["dataset_id"] = raw.get("dataset_id", _empty_series(n, "redish_lra_2024"))
    df["task_family"] = raw.get("task_family", _empty_series(n, "left_right_alternate"))
    df["subject_or_seed"] = raw.get("subject_id", _empty_series(n))
    df["session_or_run"] = raw.get("session_id", _empty_series(n))
    df["trial"] = raw.get("trial", _empty_series(n))
    df["decision_stage"] = _standardize_stage(
        raw.get("decision_stage", _empty_series(n, "choice_point"))
    )
    df["choice_point_id"] = raw.get("choice_point_id", df["decision_stage"])

    if "chosen_action_namespace" in raw.columns:
        df["action_namespace"] = _normalize_text(raw["chosen_action_namespace"])
    elif "choice_direction_usable" in raw.columns and not raw["choice_direction_usable"].astype(str).str.lower().isin({"true", "1"}).all():
        df["action_namespace"] = "raw_event_code_not_left_right"
    elif action_col == "native_action_code":
        df["action_namespace"] = "raw_event_code_not_left_right"
    else:
        df["action_namespace"] = "biological_action_unknown"

    df["chosen_action"] = _normalize_action(raw[action_col]) if action_col else np.nan
    df["action_comparison_policy"] = raw.get(
        "choice_direction_policy",
        _empty_series(n, "do_not_compare_action_labels_without_namespace_match"),
    )

    df["outcome"] = _normalize_outcome(raw[outcome_col]) if outcome_col else _reward_to_outcome(reward)
    df["reward"] = reward
    df["cost"] = cost
    df["native_cost_bin"] = native_cost_bin
    df["comparable_cost_bin"] = _standardize_cost_bin(native_cost_bin)

    df["dwell_proxy"] = _as_numeric(raw[dwell_col]) if dwell_col else np.nan
    df["deliberation_proxy"] = _as_numeric(raw[deliberation_col]) if deliberation_col else np.nan
    df["dwell_z"] = _as_numeric(raw[dwell_z_col]) if dwell_z_col else np.nan
    df["deliberation_z"] = _as_numeric(raw[deliberation_z_col]) if deliberation_z_col else np.nan
    df["vte_binary"] = _as_numeric(raw[vte_col]) if vte_col else np.nan
    df["source_file"] = str(path)

    df = _zscore_within_groups(
        df,
        value_col="dwell_proxy",
        output_col="dwell_z",
        group_cols=["source", "session_or_run", "decision_stage"],
    )
    df = _zscore_within_groups(
        df,
        value_col="deliberation_proxy",
        output_col="deliberation_z",
        group_cols=["source", "session_or_run", "decision_stage"],
    )
    return df


def _load_one_synthetic(
    path: Path,
    synthetic_dataset_id: str,
    synthetic_task_family: str,
    synthetic_stage_default: str,
    synthetic_cost_column: str | None,
) -> pd.DataFrame:
    raw = pd.read_csv(path)
    n = len(raw)

    reward_col = _first_existing(raw.columns, ["reward", "total_reward", "outcome_reward"])
    reward = _as_numeric(raw[reward_col]) if reward_col else _empty_series(n)

    stage_col = _first_existing(raw.columns, ["decision_stage", "trial_phase", "choice_point_id", "event_type"])
    action_col = _first_existing(raw.columns, ["chosen_action", "committed_path", "action", "choice", "chosen_arm"])
    outcome_col = _first_existing(raw.columns, ["outcome", "event_type"])

    cost_col = synthetic_cost_column or _first_existing(
        raw.columns,
        ["cost", "condition_cost", "risk", "shock", "threat", "delay"],
    )
    native_cost_col = _first_existing(raw.columns, ["native_cost_bin", "cost_bin", "condition"])

    dwell_col = _first_existing(raw.columns, ["dwell_proxy", "pause_ticks", "pause_time_s", "choice_point_dwell"])
    dwell_z_col = _first_existing(raw.columns, ["dwell_z", "z_dwell", "z_pause_time_by_session"])
    deliberation_col = _first_existing(raw.columns, ["deliberation_proxy", "raw_idphi", "idphi", "lab_idphi"])
    deliberation_z_col = _first_existing(raw.columns, ["deliberation_z", "z_idphi", "z_lab_idphi_by_session"])

    df = pd.DataFrame(index=raw.index)
    df["source"] = "synthetic"
    df["dataset_id"] = raw.get("dataset_id", _empty_series(n, synthetic_dataset_id))
    df["task_family"] = raw.get("task_family", _empty_series(n, synthetic_task_family))
    df["subject_or_seed"] = raw.get("seed", raw.get("subject_id", raw.get("animal_id", _empty_series(n))))
    df["session_or_run"] = raw.get("run_id", raw.get("session_id", raw.get("run", _empty_series(n, path.stem))))
    df["trial"] = raw.get("trial", _empty_series(n))
    df["decision_stage"] = _standardize_stage(raw[stage_col], synthetic_stage_default) if stage_col else synthetic_stage_default
    df["choice_point_id"] = raw.get("choice_point_id", df["decision_stage"])

    if "chosen_action_namespace" in raw.columns:
        df["action_namespace"] = _normalize_text(raw["chosen_action_namespace"])
    elif "action_namespace" in raw.columns:
        df["action_namespace"] = _normalize_text(raw["action_namespace"])
    elif action_col:
        df["action_namespace"] = "synthetic_left_right_or_model_native"
    else:
        df["action_namespace"] = "synthetic_action_unknown"

    df["chosen_action"] = _normalize_action(raw[action_col]) if action_col else np.nan
    df["action_comparison_policy"] = "synthetic_action_labels_not_compared_to_biological_raw_event_codes"

    df["outcome"] = _normalize_outcome(raw[outcome_col]) if outcome_col else _reward_to_outcome(reward)
    df["reward"] = reward
    df["cost"] = _as_numeric(raw[cost_col]) if cost_col and cost_col in raw.columns else pd.Series([0.0] * n)
    df["native_cost_bin"] = raw[native_cost_col] if native_cost_col else pd.Series(["balanced"] * n)
    df["comparable_cost_bin"] = _standardize_cost_bin(df["native_cost_bin"])

    df["dwell_proxy"] = _as_numeric(raw[dwell_col]) if dwell_col else np.nan
    df["deliberation_proxy"] = _as_numeric(raw[deliberation_col]) if deliberation_col else np.nan
    df["dwell_z"] = _as_numeric(raw[dwell_z_col]) if dwell_z_col else np.nan
    df["deliberation_z"] = _as_numeric(raw[deliberation_z_col]) if deliberation_z_col else np.nan
    df["vte_binary"] = _as_numeric(raw.get("vte_binary", _empty_series(n)))
    df["source_file"] = str(path)

    df = _zscore_within_groups(
        df,
        value_col="dwell_proxy",
        output_col="dwell_z",
        group_cols=["source", "session_or_run", "decision_stage"],
    )
    df = _zscore_within_groups(
        df,
        value_col="deliberation_proxy",
        output_col="deliberation_z",
        group_cols=["source", "session_or_run", "decision_stage"],
    )
    return df


def _load_synthetic_tables(
    paths: list[Path],
    synthetic_dataset_id: str,
    synthetic_task_family: str,
    synthetic_stage_default: str,
    synthetic_cost_column: str | None,
) -> pd.DataFrame:
    frames = [
        _load_one_synthetic(
            path=path,
            synthetic_dataset_id=synthetic_dataset_id,
            synthetic_task_family=synthetic_task_family,
            synthetic_stage_default=synthetic_stage_default,
            synthetic_cost_column=synthetic_cost_column,
        )
        for path in paths
    ]
    if not frames:
        return pd.DataFrame(columns=COMPARABLE_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def _namespace_is_comparable(value: str) -> bool:
    value = str(value).strip().lower()
    if not value:
        return False
    forbidden = ["raw_event", "not_left_right", "unknown", "do_not_compare"]
    return not any(token in value for token in forbidden)


def _action_namespace_audit(df: pd.DataFrame) -> tuple[pd.DataFrame, bool, str]:
    bio_ns = sorted(set(df.loc[df["source"] == "biological", "action_namespace"].dropna().astype(str)))
    syn_ns = sorted(set(df.loc[df["source"] == "synthetic", "action_namespace"].dropna().astype(str)))

    comparable = (
        bool(bio_ns)
        and bool(syn_ns)
        and set(bio_ns) == set(syn_ns)
        and all(_namespace_is_comparable(x) for x in bio_ns + syn_ns)
    )
    status = "comparable" if comparable else "not_comparable_namespace_mismatch"

    rows = (
        df.groupby(["source", "action_namespace", "chosen_action"], dropna=False)
        .agg(
            n_rows=("source", "size"),
            n_subjects=("subject_or_seed", "nunique"),
            n_sessions=("session_or_run", "nunique"),
            reward_rate=("reward", "mean"),
            vte_rate=("vte_binary", "mean"),
            mean_deliberation_z=("deliberation_z", "mean"),
        )
        .reset_index()
    )
    rows["action_labels_comparable"] = comparable
    rows["action_label_comparison_status"] = status
    rows["biological_action_namespaces"] = ";".join(bio_ns)
    rows["synthetic_action_namespaces"] = ";".join(syn_ns)

    return rows, comparable, status


def _summary(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    work = df.copy()
    for col in ["reward", "cost", "dwell_proxy", "deliberation_proxy", "dwell_z", "deliberation_z", "vte_binary"]:
        work[col] = _as_numeric(work[col])

    return (
        work.groupby(group_cols, dropna=False)
        .agg(
            n_rows=("source", "size"),
            n_subjects=("subject_or_seed", "nunique"),
            n_sessions=("session_or_run", "nunique"),
            reward_rate=("reward", "mean"),
            mean_reward=("reward", "mean"),
            mean_cost=("cost", "mean"),
            mean_dwell_proxy=("dwell_proxy", "mean"),
            mean_deliberation_proxy=("deliberation_proxy", "mean"),
            mean_dwell_z=("dwell_z", "mean"),
            mean_deliberation_z=("deliberation_z", "mean"),
            vte_rate=("vte_binary", "mean"),
        )
        .reset_index()
    )


def _vte_contrast(df: pd.DataFrame) -> pd.DataFrame:
    metrics = ["reward", "dwell_proxy", "deliberation_proxy", "dwell_z", "deliberation_z"]
    rows = []

    work = df.copy()
    work["vte_binary"] = _as_numeric(work["vte_binary"])

    for keys, group in work.groupby(["source", "task_family", "decision_stage"], dropna=False):
        source, task_family, decision_stage = keys
        v0 = group[group["vte_binary"] == 0]
        v1 = group[group["vte_binary"] == 1]

        row = {
            "source": source,
            "task_family": task_family,
            "decision_stage": decision_stage,
            "n_rows": int(len(group)),
            "n_nonvte": int(len(v0)),
            "n_vte": int(len(v1)),
            "vte_rate": float(group["vte_binary"].mean()) if group["vte_binary"].notna().any() else np.nan,
        }

        for metric in metrics:
            x0 = _as_numeric(v0[metric])
            x1 = _as_numeric(v1[metric])
            m0 = float(x0.mean()) if x0.notna().any() else np.nan
            m1 = float(x1.mean()) if x1.notna().any() else np.nan
            row[f"{metric}_nonvte_mean"] = m0
            row[f"{metric}_vte_mean"] = m1
            row[f"{metric}_vte_minus_nonvte"] = m1 - m0 if np.isfinite(m0) and np.isfinite(m1) else np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def _field_coverage(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    total = len(df)

    for col in COMPARABLE_COLUMNS:
        values = df[col] if col in df.columns else pd.Series([np.nan] * total)
        if pd.api.types.is_numeric_dtype(values):
            nonempty = int(values.notna().sum())
        else:
            text = values.fillna("").astype(str).str.strip()
            nonempty = int((text != "").sum())

        rows.append(
            {
                "field": col,
                "nonempty": nonempty,
                "total": total,
                "coverage": nonempty / total if total else np.nan,
            }
        )

    return pd.DataFrame(rows)


def _write_report(
    output_dir: Path,
    comparable: pd.DataFrame,
    action_status: str,
    action_labels_comparable: bool,
) -> Path:
    n_bio = int((comparable["source"] == "biological").sum())
    n_syn = int((comparable["source"] == "synthetic").sum())

    lines = [
        "# LRA BioSynth comparability report",
        "",
        "Purpose: compare healthy biological LRA VTE/outcome/deliberation fields against synthetic balanced-fork decision rows.",
        "",
        f"- Comparable rows: {len(comparable)}",
        f"- Biological rows: {n_bio}",
        f"- Synthetic rows: {n_syn}",
        f"- Action labels comparable: {action_labels_comparable}",
        f"- Action label status: {action_status}",
        "",
        "Interpretation constraints:",
        "",
        "- DREADD perturbation rows must not be mixed into the healthy baseline.",
        "- Biological raw action codes are not left/right labels.",
        "- Synthetic left/right actions are not compared to biological raw action codes unless namespaces explicitly match.",
        "- Primary comparable fields are reward/outcome, VTE binary or proxy, deliberation z-score, and dwell z-score.",
        "- Absolute biological IdPhi and synthetic raw IdPhi scales are not treated as commensurable.",
        "",
    ]

    report_path = output_dir / OUTPUT_REPORT
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def build_lra_biosynth_comparability(
    biological_csv: str | Path,
    synthetic_csvs: list[str | Path],
    output_dir: str | Path,
    synthetic_dataset_id: str = "stage3_synthetic",
    synthetic_task_family: str = "stage3_synthetic",
    synthetic_stage_default: str = "choice_point",
    synthetic_cost_column: str | None = None,
    fail_on_action_namespace_mismatch: bool = False,
) -> dict:
    biological_csv = Path(biological_csv)
    synthetic_paths = [Path(p) for p in synthetic_csvs]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bio = _load_biological_lra(biological_csv)
    syn = _load_synthetic_tables(
        paths=synthetic_paths,
        synthetic_dataset_id=synthetic_dataset_id,
        synthetic_task_family=synthetic_task_family,
        synthetic_stage_default=synthetic_stage_default,
        synthetic_cost_column=synthetic_cost_column,
    )

    comparable = pd.concat([bio, syn], ignore_index=True)

    for col in COMPARABLE_COLUMNS:
        if col not in comparable.columns:
            comparable[col] = np.nan

    comparable = comparable[COMPARABLE_COLUMNS]

    action_audit, action_labels_comparable, action_status = _action_namespace_audit(comparable)

    if fail_on_action_namespace_mismatch and not action_labels_comparable:
        raise ValueError(
            "Action-label namespace mismatch: biological and synthetic chosen_action labels "
            "must not be compared directly."
        )

    by_vte = _summary(comparable, ["source", "task_family", "decision_stage", "vte_binary"])
    by_outcome = _summary(comparable, ["source", "task_family", "decision_stage", "outcome"])
    vte_contrast = _vte_contrast(comparable)
    coverage = _field_coverage(comparable)

    comparable_path = output_dir / OUTPUT_COMPARABLE
    by_vte_path = output_dir / OUTPUT_BY_VTE
    by_outcome_path = output_dir / OUTPUT_BY_OUTCOME
    action_audit_path = output_dir / OUTPUT_ACTION_AUDIT
    vte_contrast_path = output_dir / OUTPUT_VTE_CONTRAST
    coverage_path = output_dir / OUTPUT_COVERAGE
    meta_path = output_dir / OUTPUT_META

    comparable.to_csv(comparable_path, index=False)
    by_vte.to_csv(by_vte_path, index=False)
    by_outcome.to_csv(by_outcome_path, index=False)
    action_audit.to_csv(action_audit_path, index=False)
    vte_contrast.to_csv(vte_contrast_path, index=False)
    coverage.to_csv(coverage_path, index=False)

    report_path = _write_report(
        output_dir=output_dir,
        comparable=comparable,
        action_status=action_status,
        action_labels_comparable=action_labels_comparable,
    )

    meta = {
        "dataset_id": "redish_lra_2024",
        "patch": "18A",
        "biological_csv": str(biological_csv),
        "synthetic_csvs": [str(p) for p in synthetic_paths],
        "n_comparable_rows": int(len(comparable)),
        "n_biological_rows": int((comparable["source"] == "biological").sum()),
        "n_synthetic_rows": int((comparable["source"] == "synthetic").sum()),
        "action_labels_comparable": bool(action_labels_comparable),
        "action_label_comparison_status": action_status,
        "policy": (
            "Compare healthy biological LRA and synthetic rows on VTE/outcome/deliberation/dwell only. "
            "Do not compare chosen_action labels unless action namespaces explicitly match."
        ),
        "outputs": {
            "comparable": str(comparable_path),
            "by_vte": str(by_vte_path),
            "by_outcome": str(by_outcome_path),
            "action_namespace_audit": str(action_audit_path),
            "vte_contrast": str(vte_contrast_path),
            "coverage": str(coverage_path),
            "report": str(report_path),
        },
    }

    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Comparable decision rows saved: {comparable_path}")
    print(f"VTE summary saved: {by_vte_path}")
    print(f"Outcome summary saved: {by_outcome_path}")
    print(f"Action namespace audit saved: {action_audit_path}")
    print(f"VTE contrast saved: {vte_contrast_path}")
    print(f"Coverage saved: {coverage_path}")
    print(f"Metadata saved: {meta_path}")
    print(f"Report saved: {report_path}")
    print(f"Comparable rows: {meta['n_comparable_rows']}")
    print(f"Biological rows: {meta['n_biological_rows']}")
    print(f"Synthetic rows: {meta['n_synthetic_rows']}")
    print(f"Action labels comparable: {meta['action_labels_comparable']}")
    print(f"Action label status: {meta['action_label_comparison_status']}")

    return meta


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build LRA healthy-control biological-vs-synthetic comparability tables."
    )
    parser.add_argument("--biological-csv", required=True)
    parser.add_argument("--synthetic-csv", required=True, action="append")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--synthetic-dataset-id", default="stage3_synthetic")
    parser.add_argument("--synthetic-task-family", default="stage3_synthetic")
    parser.add_argument("--synthetic-stage-default", default="choice_point")
    parser.add_argument("--synthetic-cost-column", default=None)
    parser.add_argument("--fail-on-action-namespace-mismatch", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    build_lra_biosynth_comparability(
        biological_csv=args.biological_csv,
        synthetic_csvs=args.synthetic_csv,
        output_dir=args.output_dir,
        synthetic_dataset_id=args.synthetic_dataset_id,
        synthetic_task_family=args.synthetic_task_family,
        synthetic_stage_default=args.synthetic_stage_default,
        synthetic_cost_column=args.synthetic_cost_column,
        fail_on_action_namespace_mismatch=args.fail_on_action_namespace_mismatch,
    )


if __name__ == "__main__":
    main()
