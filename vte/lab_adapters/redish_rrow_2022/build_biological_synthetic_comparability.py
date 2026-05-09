from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


COMPARABLE_COLUMNS = [
    "source",
    "dataset_id",
    "task_family",
    "subject_or_seed",
    "session_or_run",
    "trial",
    "decision_stage",
    "choice_point_id",
    "restaurant_id",
    "chosen_action",
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


OUTPUT_COMPARABLE = "Table_BioSynth_Comparable_Decision_Rows.csv"
OUTPUT_BY_COST = "Table_BioSynth_Summary_By_Source_Stage_Cost.csv"
OUTPUT_BY_CHOICE = "Table_BioSynth_Summary_By_Source_Choice.csv"
OUTPUT_DIRECTION = "Table_BioSynth_Direction_Agreement.csv"
OUTPUT_COVERAGE = "Table_BioSynth_Field_Coverage.csv"
OUTPUT_META = "biosynth_comparability_meta.json"
OUTPUT_REPORT = "BioSynth_Comparability_Report.md"


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


def _normalize_choice(series: pd.Series) -> pd.Series:
    text = _normalize_text(series)
    mapping = {
        "earn": "earn",
        "earned": "earn",
        "accept": "accept",
        "accepted": "accept",
        "quit": "quit",
        "skip": "skip",
        "reject": "skip",
        "rejected": "skip",
        "abort": "quit",
    }
    return text.map(lambda x: mapping.get(x, x))


def _normalize_outcome(series: pd.Series) -> pd.Series:
    text = _normalize_text(series)
    mapping = {
        "rewarded": "earn",
        "reward": "earn",
        "earned": "earn",
        "earn": "earn",
        "unrewarded": "no_reward",
        "no_reward": "no_reward",
        "quit": "quit",
        "skip": "skip",
        "accepted": "accept",
        "accept": "accept",
    }
    return text.map(lambda x: mapping.get(x, x))


def _reward_to_outcome(reward: pd.Series) -> pd.Series:
    numeric = _as_numeric(reward)
    out = pd.Series([""] * len(numeric))
    out.loc[numeric > 0] = "earn"
    out.loc[numeric == 0] = "no_reward"
    return out


def _standardize_stage(series: pd.Series, default: str = "choice_point") -> pd.Series:
    text = _normalize_text(series)
    text = text.replace("", default)
    mapping = {
        "offerzone": "offer_zone",
        "offer_zone": "offer_zone",
        "waitzone": "wait_zone",
        "wait_zone": "wait_zone",
        "choice": "choice_point",
        "choice_zone": "choice_point",
    }
    return text.map(lambda x: mapping.get(x, x))


def _standardize_native_cost_bin(series: pd.Series) -> pd.Series:
    text = _normalize_text(series)

    def convert(value: str) -> str:
        if value in {"", "nan", "none", "null"}:
            return ""
        if value in {"delay_00_04", "delay_0_4", "00_04", "0_4"}:
            return "low"
        if value in {"delay_05_09", "delay_5_9", "05_09", "5_9"}:
            return "low"
        if value in {"delay_10_14", "10_14", "delay_15_19", "15_19"}:
            return "medium"
        if value in {"delay_20_24", "20_24", "delay_25_plus", "25_plus"}:
            return "high"
        if "low" in value or "easy" in value or "obvious" in value:
            return "low"
        if "medium" in value or "mid" in value:
            return "medium"
        if "high" in value or "hard" in value:
            return "high"
        return ""

    return text.map(convert)


def _assign_quantile_cost_bins(df: pd.DataFrame) -> pd.Series:
    result = pd.Series([""] * len(df), index=df.index, dtype="object")

    if "cost" not in df.columns:
        return result

    for _, idx in df.groupby(["source", "decision_stage"], dropna=False).groups.items():
        idx = list(idx)
        cost = _as_numeric(df.loc[idx, "cost"])
        finite = cost[np.isfinite(cost)]
        if finite.nunique() < 3:
            continue

        try:
            bins = pd.qcut(cost, q=3, labels=["low", "medium", "high"], duplicates="drop")
        except ValueError:
            continue

        result.loc[idx] = bins.astype("object").fillna("").astype(str).values

    return result


def _fill_comparable_cost_bin(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    native = _standardize_native_cost_bin(out.get("native_cost_bin", _empty_series(len(out))))
    quantile = _assign_quantile_cost_bins(out)

    comparable = native.copy()
    missing = comparable == ""
    comparable.loc[missing] = quantile.loc[missing]
    out["comparable_cost_bin"] = comparable
    return out


def _zscore_within_groups(
    df: pd.DataFrame,
    value_col: str,
    output_col: str,
    group_cols: list[str],
) -> pd.DataFrame:
    out = df.copy()
    if output_col in out.columns and out[output_col].notna().all():
        return out

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


def _load_biological_canonical(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    n = len(raw)

    df = pd.DataFrame(index=raw.index)
    df["source"] = "biological"
    df["dataset_id"] = raw.get("dataset_id", _empty_series(n, "redish_rrow_2022"))
    df["task_family"] = raw.get("task_family", _empty_series(n, "restaurant_row"))
    df["subject_or_seed"] = raw.get("subject_id", _empty_series(n))
    df["session_or_run"] = raw.get("session_id", _empty_series(n))
    df["trial"] = raw.get("trial", _empty_series(n))
    df["decision_stage"] = _standardize_stage(raw.get("decision_stage", _empty_series(n)))
    df["choice_point_id"] = raw.get("choice_point_id", df["decision_stage"])
    df["restaurant_id"] = raw.get("restaurant_id", _empty_series(n))
    df["chosen_action"] = _normalize_choice(raw.get("chosen_action", _empty_series(n)))
    df["outcome"] = _normalize_outcome(raw.get("outcome", _empty_series(n)))
    df["reward"] = _as_numeric(raw.get("reward", _empty_series(n)))
    df["cost"] = _as_numeric(raw.get("cost", raw.get("offer_delay_s", _empty_series(n))))

    native_cost_col = _first_existing(raw.columns, ["delay_bin", "cost_bin", "native_cost_bin"])
    df["native_cost_bin"] = raw[native_cost_col] if native_cost_col else ""

    df["dwell_proxy"] = _as_numeric(
        raw.get("dwell_proxy", raw.get("pause_time_s", _empty_series(n)))
    )
    df["deliberation_proxy"] = _as_numeric(
        raw.get("deliberation_proxy", raw.get("lab_idphi", _empty_series(n)))
    )

    dwell_z_col = _first_existing(
        raw.columns,
        [
            "dwell_z",
            "z_pause_time_by_session_stage",
            "z_total_site_time_by_session_stage",
        ],
    )
    deliberation_z_col = _first_existing(
        raw.columns,
        ["deliberation_z", "z_lab_idphi_by_session_stage", "z_idphi"],
    )

    df["dwell_z"] = _as_numeric(raw[dwell_z_col]) if dwell_z_col else np.nan
    df["deliberation_z"] = (
        _as_numeric(raw[deliberation_z_col]) if deliberation_z_col else np.nan
    )
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


def _load_one_synthetic_csv(
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

    stage_col = _first_existing(
        raw.columns,
        ["decision_stage", "trial_phase", "choice_point_id", "event_type"],
    )
    choice_col = _first_existing(
        raw.columns,
        ["chosen_action", "committed_path", "action", "choice", "chosen_arm"],
    )
    outcome_col = _first_existing(raw.columns, ["outcome", "restaurant_outcome", "event_type"])
    cost_col = synthetic_cost_column or _first_existing(
        raw.columns,
        ["cost", "offer_delay_s", "delay", "risk", "shock", "threat", "condition_cost"],
    )
    native_cost_col = _first_existing(raw.columns, ["delay_bin", "cost_bin", "condition"])

    deliberation_col = _first_existing(
        raw.columns,
        ["deliberation_proxy", "raw_idphi", "idphi", "lab_idphi"],
    )
    deliberation_z_col = _first_existing(
        raw.columns,
        ["deliberation_z", "z_idphi", "z_lab_idphi_by_session_stage"],
    )
    dwell_col = _first_existing(
        raw.columns,
        ["dwell_proxy", "pause_ticks", "pause_time_s", "choice_point_dwell"],
    )
    dwell_z_col = _first_existing(
        raw.columns,
        ["dwell_z", "z_pause_time_by_session_stage", "z_dwell"],
    )

    df = pd.DataFrame(index=raw.index)
    df["source"] = "synthetic"
    df["dataset_id"] = raw.get("dataset_id", _empty_series(n, synthetic_dataset_id))
    df["task_family"] = raw.get("task_family", _empty_series(n, synthetic_task_family))
    df["subject_or_seed"] = raw.get(
        "seed",
        raw.get("subject_id", raw.get("animal_id", _empty_series(n))),
    )
    df["session_or_run"] = raw.get(
        "run_id",
        raw.get("session_id", raw.get("run", _empty_series(n, path.stem))),
    )
    df["trial"] = raw.get("trial", _empty_series(n))
    df["decision_stage"] = (
        _standardize_stage(raw[stage_col], default=synthetic_stage_default)
        if stage_col
        else synthetic_stage_default
    )
    df["choice_point_id"] = raw.get("choice_point_id", df["decision_stage"])
    df["restaurant_id"] = raw.get("restaurant_id", _empty_series(n))
    df["chosen_action"] = _normalize_choice(raw[choice_col]) if choice_col else ""
    df["outcome"] = (
        _normalize_outcome(raw[outcome_col]) if outcome_col else _reward_to_outcome(reward)
    )
    df["reward"] = reward
    df["cost"] = _as_numeric(raw[cost_col]) if cost_col and cost_col in raw.columns else np.nan
    df["native_cost_bin"] = raw[native_cost_col] if native_cost_col else ""
    df["dwell_proxy"] = _as_numeric(raw[dwell_col]) if dwell_col else np.nan
    df["deliberation_proxy"] = (
        _as_numeric(raw[deliberation_col]) if deliberation_col else np.nan
    )
    df["dwell_z"] = _as_numeric(raw[dwell_z_col]) if dwell_z_col else np.nan
    df["deliberation_z"] = (
        _as_numeric(raw[deliberation_z_col]) if deliberation_z_col else np.nan
    )
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
        _load_one_synthetic_csv(
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


def _field_coverage(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    total = len(df)
    for col in COMPARABLE_COLUMNS:
        if col not in df.columns:
            nonempty = 0
        else:
            values = df[col]
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


def _summary(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    work = df.copy()
    for col in ["reward", "cost", "dwell_proxy", "deliberation_proxy", "dwell_z", "deliberation_z", "vte_binary"]:
        if col in work.columns:
            work[col] = _as_numeric(work[col])

    grouped = work.groupby(group_cols, dropna=False)
    out = grouped.agg(
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
    ).reset_index()
    return out


def _direction_agreement(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = ["deliberation_z", "dwell_z", "reward"]

    valid = df[df["comparable_cost_bin"].isin(["low", "high"])].copy()
    if valid.empty:
        return pd.DataFrame(
            columns=[
                "decision_stage",
                "metric",
                "biological_high_minus_low",
                "synthetic_high_minus_low",
                "direction_agreement",
            ]
        )

    grouped = (
        valid.groupby(["source", "decision_stage", "comparable_cost_bin"], dropna=False)[metrics]
        .mean()
        .reset_index()
    )

    for decision_stage in sorted(grouped["decision_stage"].dropna().unique()):
        stage = grouped[grouped["decision_stage"] == decision_stage]
        for metric in metrics:
            effects = {}
            for source in ["biological", "synthetic"]:
                src = stage[stage["source"] == source]
                lows = src[src["comparable_cost_bin"] == "low"][metric]
                highs = src[src["comparable_cost_bin"] == "high"][metric]
                if lows.empty or highs.empty:
                    effects[source] = np.nan
                else:
                    effects[source] = float(highs.iloc[0] - lows.iloc[0])

            bio = effects.get("biological", np.nan)
            syn = effects.get("synthetic", np.nan)
            if np.isfinite(bio) and np.isfinite(syn):
                agreement = np.sign(bio) == np.sign(syn)
            else:
                agreement = np.nan

            rows.append(
                {
                    "decision_stage": decision_stage,
                    "metric": metric,
                    "biological_high_minus_low": bio,
                    "synthetic_high_minus_low": syn,
                    "direction_agreement": agreement,
                }
            )

    return pd.DataFrame(rows)


def _write_report(
    output_dir: Path,
    comparable: pd.DataFrame,
    coverage: pd.DataFrame,
    direction: pd.DataFrame,
) -> Path:
    n_bio = int((comparable["source"] == "biological").sum())
    n_syn = int((comparable["source"] == "synthetic").sum())

    high_low_rows = direction.dropna(subset=["direction_agreement"])
    n_agreement = int((high_low_rows["direction_agreement"] == True).sum())
    n_comparable = int(len(high_low_rows))

    lines = [
        "# BioSynth comparability report",
        "",
        "Purpose: create decision-level biological-vs-synthetic comparability tables without changing the frozen VTE wrapper.",
        "",
        f"- Comparable rows: {len(comparable)}",
        f"- Biological rows: {n_bio}",
        f"- Synthetic rows: {n_syn}",
        f"- Direction agreement rows: {n_agreement}/{n_comparable}",
        "",
        "Interpretation constraints:",
        "",
        "- Absolute biological `lab_idphi` and synthetic `raw_idphi` scales are not treated as commensurable.",
        "- Primary comparison fields are z-normalized deliberation and dwell proxies.",
        "- Cost-bin comparisons require both sources to have low/high comparable bins.",
        "- Missing synthetic cost means the table remains usable for choice/outcome and VTE-rate summaries, but not high-minus-low cost effects.",
        "",
        "Field coverage:",
        "",
        coverage.to_markdown(index=False),
        "",
    ]

    if not direction.empty:
        lines.extend(
            [
                "Direction agreement:",
                "",
                direction.to_markdown(index=False),
                "",
            ]
        )

    report_path = output_dir / OUTPUT_REPORT
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def build_biological_synthetic_comparability(
    biological_csv: str | Path,
    synthetic_csvs: list[str | Path],
    output_dir: str | Path,
    synthetic_dataset_id: str = "stage3_synthetic",
    synthetic_task_family: str = "stage3_synthetic",
    synthetic_stage_default: str = "choice_point",
    synthetic_cost_column: str | None = None,
) -> dict:
    biological_csv = Path(biological_csv)
    synthetic_paths = [Path(p) for p in synthetic_csvs]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bio = _load_biological_canonical(biological_csv)
    syn = _load_synthetic_tables(
        paths=synthetic_paths,
        synthetic_dataset_id=synthetic_dataset_id,
        synthetic_task_family=synthetic_task_family,
        synthetic_stage_default=synthetic_stage_default,
        synthetic_cost_column=synthetic_cost_column,
    )

    comparable = pd.concat([bio, syn], ignore_index=True)
    comparable = _fill_comparable_cost_bin(comparable)

    for col in COMPARABLE_COLUMNS:
        if col not in comparable.columns:
            comparable[col] = np.nan
    comparable = comparable[COMPARABLE_COLUMNS]

    comparable_path = output_dir / OUTPUT_COMPARABLE
    by_cost_path = output_dir / OUTPUT_BY_COST
    by_choice_path = output_dir / OUTPUT_BY_CHOICE
    direction_path = output_dir / OUTPUT_DIRECTION
    coverage_path = output_dir / OUTPUT_COVERAGE
    meta_path = output_dir / OUTPUT_META

    by_cost = _summary(
        comparable,
        ["source", "task_family", "decision_stage", "comparable_cost_bin"],
    )
    by_choice = _summary(
        comparable,
        ["source", "task_family", "decision_stage", "chosen_action", "outcome"],
    )
    direction = _direction_agreement(comparable)
    coverage = _field_coverage(comparable)

    comparable.to_csv(comparable_path, index=False)
    by_cost.to_csv(by_cost_path, index=False)
    by_choice.to_csv(by_choice_path, index=False)
    direction.to_csv(direction_path, index=False)
    coverage.to_csv(coverage_path, index=False)

    report_path = _write_report(output_dir, comparable, coverage, direction)

    meta = {
        "biological_csv": str(biological_csv),
        "synthetic_csvs": [str(p) for p in synthetic_paths],
        "n_comparable_rows": int(len(comparable)),
        "n_biological_rows": int((comparable["source"] == "biological").sum()),
        "n_synthetic_rows": int((comparable["source"] == "synthetic").sum()),
        "outputs": {
            "comparable": str(comparable_path),
            "by_cost": str(by_cost_path),
            "by_choice": str(by_choice_path),
            "direction_agreement": str(direction_path),
            "coverage": str(coverage_path),
            "report": str(report_path),
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Comparable decision rows saved: {comparable_path}")
    print(f"Cost summary saved: {by_cost_path}")
    print(f"Choice summary saved: {by_choice_path}")
    print(f"Direction agreement saved: {direction_path}")
    print(f"Field coverage saved: {coverage_path}")
    print(f"Metadata saved: {meta_path}")
    print(f"Report saved: {report_path}")
    print(f"Comparable rows: {meta['n_comparable_rows']}")
    print(f"Biological rows: {meta['n_biological_rows']}")
    print(f"Synthetic rows: {meta['n_synthetic_rows']}")

    return meta


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build biological-vs-synthetic decision comparability tables."
    )
    parser.add_argument("--biological-csv", required=True)
    parser.add_argument(
        "--synthetic-csv",
        required=True,
        action="append",
        help="Synthetic VTE metrics or decision endpoint CSV. Can be passed multiple times.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--synthetic-dataset-id", default="stage3_synthetic")
    parser.add_argument("--synthetic-task-family", default="stage3_synthetic")
    parser.add_argument("--synthetic-stage-default", default="choice_point")
    parser.add_argument(
        "--synthetic-cost-column",
        default=None,
        help="Optional explicit synthetic cost column. If omitted, common cost columns are detected.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    build_biological_synthetic_comparability(
        biological_csv=args.biological_csv,
        synthetic_csvs=args.synthetic_csv,
        output_dir=args.output_dir,
        synthetic_dataset_id=args.synthetic_dataset_id,
        synthetic_task_family=args.synthetic_task_family,
        synthetic_stage_default=args.synthetic_stage_default,
        synthetic_cost_column=args.synthetic_cost_column,
    )


if __name__ == "__main__":
    main()