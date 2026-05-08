"""Build semantic Restaurant Row decision endpoints for Stage 3.2C.

This module consumes the diagnostic endpoint table produced by
extract_processed_behavior_endpoints.py and rewrites it into an author-level
decision schema.

Important semantic distinction
-----------------------------
Restaurant Row is not a single flat choice variable.

Offer Zone:
    Skip   -> reject offer and move on
    Earn   -> implies Offer-Zone accept, followed by waiting to reward
    Quit   -> implies Offer-Zone accept, followed by quitting in Wait Zone

Wait Zone:
    Earn   -> wait-zone outcome is earn, reward received
    Quit   -> wait-zone outcome is quit, no reward
    Skip   -> no actual wait-zone decision; kept only as diagnostic row

The final comparability artifact should use oz_choice, wz_outcome,
restaurant_outcome, and reward rather than a collapsed choice field.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


OUTPUT_ENDPOINT = "Table_Redish_RRow_Decision_Endpoint.csv"
OUTPUT_USABLE = "Table_Redish_RRow_Decision_Endpoint_Usable.csv"
OUTPUT_BY_STAGE = "Table_Redish_RRow_Decision_By_Stage.csv"
OUTPUT_BY_DELAY = "Table_Redish_RRow_Decision_By_Delay.csv"
OUTPUT_BY_SESSION = "Table_Redish_RRow_Decision_By_Session.csv"
OUTPUT_META = "redish_rrow_decision_endpoint_meta.json"
OUTPUT_REPORT = "Redish_RRow_Decision_Endpoint_Report.md"


VECTOR_FIELDS = [
    "trial",
    "zone_delay",
    "total_site_time",
    "pause_time",
    "run_speed",
    "lab_idphi",
    "lab_avg_dphi",
    "site_rank",
    "entering_zone_time",
    "exit_zone_time",
    "zone_time",
    "adj_wz_exit_time",
    "current_cycle",
]

ID_FIELDS = [
    "dataset_id",
    "subject_id",
    "session_id",
    "row_index",
]


MISSING_TOKENS = {"", "nan", "none", "null", "<na>", "na", "n/a"}


def _is_blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    text = str(value).strip().lower()
    return text in MISSING_TOKENS


def _clean_scalar(value: Any) -> Any:
    if _is_blank(value):
        return ""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and value.is_integer():
        return int(value)
    text = str(value).strip()
    if text.lower() in MISSING_TOKENS:
        return ""
    return text


def _parse_vector(value: Any) -> list[Any]:
    """Parse simple MATLAB/Pandas-style vector cells.

    Handles strings such as:
        "[NaN, NaN, 3.0, 4.0]"
        "['A', 'B']"
        "WaitZone"
    """

    if _is_blank(value):
        return [""]

    if isinstance(value, (list, tuple, np.ndarray)):
        return [_clean_scalar(v) for v in list(value)]

    text = str(value).strip()

    if not (text.startswith("[") and text.endswith("]")):
        return [_clean_scalar(text)]

    body = text[1:-1].strip()
    if not body:
        return [""]

    parts = [p.strip().strip("'").strip('"') for p in body.split(",")]
    return [_clean_scalar(p) for p in parts]


def _vector_value(values: list[Any], index: int) -> Any:
    if not values:
        return ""
    if len(values) == 1:
        return values[0]
    if index < len(values):
        return values[index]
    return ""


def _to_float(value: Any) -> float:
    if _is_blank(value):
        return float("nan")
    try:
        return float(value)
    except Exception:
        return float("nan")


def _to_int_or_blank(value: Any) -> Any:
    number = _to_float(value)
    if math.isnan(number):
        return ""
    if float(number).is_integer():
        return int(number)
    return number


def _first_existing(row: pd.Series, names: Iterable[str]) -> Any:
    for name in names:
        if name in row.index and not _is_blank(row.get(name, "")):
            return row.get(name, "")
    return ""


def _zone_context(row: pd.Series) -> str:
    raw = _first_existing(
        row,
        [
            "source_zone_context",
            "zone_context",
            "zone_id",
            "choice_point_id",
            "zone_type",
        ],
    )
    text = str(raw).strip()
    if text:
        return text
    return "unknown"


def _zone_type_from_context(context: str) -> str:
    text = str(context).strip().lower().replace(" ", "").replace("_", "")
    if text in {"offerzone", "oz"}:
        return "offer_zone"
    if text in {"waitzone", "wz"}:
        return "wait_zone"
    if text in {"rewardzone", "rz"}:
        return "reward_zone"
    if text in {"transitionzone", "tz"}:
        return "transition_zone"
    return "unknown"


def _author_decision_label(row: pd.Series) -> str:
    raw = _first_existing(
        row,
        [
            "author_decision_label",
            "restaurant_outcome",
            "choice",
            "decision",
            "Decision",
        ],
    )
    text = str(raw).strip().lower()

    mapping = {
        "earn": "earn",
        "earned": "earn",
        "accept": "earn",  # legacy diagnostic alias; not author-preferred
        "accepted": "earn",
        "skip": "skip",
        "skipped": "skip",
        "quit": "quit",
        "quitted": "quit",
    }
    return mapping.get(text, "")


def _semantic_fields(author_label: str) -> dict[str, Any]:
    """Return Restaurant Row decision semantics from author outcome label."""

    if author_label == "earn":
        return {
            "oz_choice": "accept",
            "wz_outcome": "earn",
            "restaurant_outcome": "earn",
            "reward": 1,
        }

    if author_label == "quit":
        return {
            "oz_choice": "accept",
            "wz_outcome": "quit",
            "restaurant_outcome": "quit",
            "reward": 0,
        }

    if author_label == "skip":
        return {
            "oz_choice": "skip",
            "wz_outcome": "",
            "restaurant_outcome": "skip",
            "reward": 0,
        }

    return {
        "oz_choice": "",
        "wz_outcome": "",
        "restaurant_outcome": "",
        "reward": "",
    }


def _stage_decision(zone_type: str, semantics: dict[str, Any]) -> tuple[str, bool]:
    """Return stage-local decision and applicability flag."""

    outcome = str(semantics.get("restaurant_outcome", "")).strip().lower()

    if zone_type == "offer_zone":
        decision = str(semantics.get("oz_choice", "")).strip().lower()
        return decision, decision in {"accept", "skip"}

    if zone_type == "wait_zone":
        if outcome == "skip":
            return "", False
        decision = str(semantics.get("wz_outcome", "")).strip().lower()
        return decision, decision in {"earn", "quit"}

    return "", False


def _restaurant_id_from_slot(zone_slot: int) -> int:
    """Restaurant Row has four restaurants.

    The source matrices often expose eight vector slots. Until a better
    author-level slot map is recovered, slots are treated as two repetitions of
    restaurant positions 1..4 rather than eight restaurants.
    """

    return int(zone_slot % 4) + 1


def _max_vector_len(parsed: dict[str, list[Any]]) -> int:
    lengths = [len(v) for v in parsed.values() if v]
    if not lengths:
        return 1
    return max(lengths)


def _explode_source_row(row: pd.Series) -> list[dict[str, Any]]:
    parsed = {field: _parse_vector(row.get(field, "")) for field in VECTOR_FIELDS}
    n_slots = _max_vector_len(parsed)

    zone_context = _zone_context(row)
    zone_type = _zone_type_from_context(zone_context)
    author_label = _author_decision_label(row)
    semantics = _semantic_fields(author_label)

    rows: list[dict[str, Any]] = []

    for zone_slot in range(n_slots):
        trial = _to_int_or_blank(_vector_value(parsed.get("trial", [""]), zone_slot))
        if _is_blank(trial):
            continue

        restaurant_id = _restaurant_id_from_slot(zone_slot)
        stage_decision, stage_applicable = _stage_decision(zone_type, semantics)

        out: dict[str, Any] = {}
        for field in ID_FIELDS:
            out[field] = row.get(field, "") if field in row.index else ""

        if _is_blank(out.get("row_index", "")):
            out["row_index"] = int(row.name) if row.name is not None else ""

        out.update(
            {
                "restaurant_visit_id": (
                    f"{out.get('session_id', 'session')}_"
                    f"r{int(out['row_index']):05d}_s{zone_slot}"
                    if not _is_blank(out.get("row_index", ""))
                    else f"{out.get('session_id', 'session')}_s{zone_slot}"
                ),
                "trial": trial,
                "source_zone_slot": zone_slot,
                "restaurant_id": restaurant_id,
                "source_zone_context": zone_context,
                "decision_stage": zone_type,
                "choice_point_id": f"redish_rrow_restaurant_{restaurant_id}_{zone_type}",
                "author_decision_label": author_label,
                "oz_choice": semantics["oz_choice"],
                "wz_outcome": semantics["wz_outcome"],
                "restaurant_outcome": semantics["restaurant_outcome"],
                "stage_decision": stage_decision,
                "stage_applicable": bool(stage_applicable),
                "reward": semantics["reward"],
            }
        )

        numeric_map = {
            "offer_delay_s": "zone_delay",
            "total_site_time_s": "total_site_time",
            "pause_time_s": "pause_time",
            "run_speed": "run_speed",
            "lab_idphi": "lab_idphi",
            "lab_avg_dphi": "lab_avg_dphi",
            "site_rank": "site_rank",
            "entering_zone_time": "entering_zone_time",
            "exit_zone_time": "exit_zone_time",
            "zone_time_s": "zone_time",
            "adj_wz_exit_time": "adj_wz_exit_time",
            "current_cycle": "current_cycle",
        }

        for out_col, source_col in numeric_map.items():
            value = _vector_value(parsed.get(source_col, [""]), zone_slot)
            out[out_col] = _to_float(value)

        threshold = _first_existing(row, ["threshold_sec", "threshold", "accept_threshold"])
        threshold_num = _to_float(threshold)
        out["threshold_sec"] = threshold_num

        if not math.isnan(threshold_num) and not math.isnan(out["offer_delay_s"]):
            out["subjective_value"] = threshold_num - out["offer_delay_s"]
        else:
            out["subjective_value"] = float("nan")

        rows.append(out)

    return rows


def _numeric_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _build_usable(endpoint: pd.DataFrame) -> pd.DataFrame:
    if endpoint.empty:
        return endpoint.copy()

    usable = endpoint.copy()

    required_nonempty = [
        "trial",
        "restaurant_id",
        "decision_stage",
        "restaurant_outcome",
        "reward",
        "offer_delay_s",
        "pause_time_s",
        "lab_idphi",
        "lab_avg_dphi",
    ]

    mask = pd.Series(True, index=usable.index)

    for col in required_nonempty:
        if col not in usable.columns:
            return usable.iloc[0:0].copy()

        if col in {"offer_delay_s", "pause_time_s", "lab_idphi", "lab_avg_dphi", "reward"}:
            mask = mask & pd.to_numeric(usable[col], errors="coerce").notna()
        else:
            text = usable[col].astype("string").str.strip().str.lower()
            mask = mask & ~text.isin(MISSING_TOKENS)

    mask = mask & usable["stage_applicable"].astype(bool)

    return usable.loc[mask].reset_index(drop=True)


def _delay_bin(value: float) -> str:
    if math.isnan(value):
        return "missing"
    if value < 5:
        return "delay_00_04"
    if value < 10:
        return "delay_05_09"
    if value < 15:
        return "delay_10_14"
    if value < 20:
        return "delay_15_19"
    if value < 25:
        return "delay_20_24"
    return "delay_25_plus"


def _write_summaries(endpoint: pd.DataFrame, output_dir: Path) -> dict[str, str]:
    paths: dict[str, str] = {}

    if endpoint.empty:
        by_stage = pd.DataFrame()
        by_delay = pd.DataFrame()
        by_session = pd.DataFrame()
    else:
        work = endpoint.copy()
        work = _numeric_columns(
            work,
            [
                "reward",
                "offer_delay_s",
                "pause_time_s",
                "lab_idphi",
                "lab_avg_dphi",
                "run_speed",
            ],
        )

        by_stage = (
            work.groupby(
                [
                    "dataset_id",
                    "decision_stage",
                    "stage_applicable",
                    "stage_decision",
                    "restaurant_outcome",
                ],
                dropna=False,
            )
            .agg(
                n_rows=("restaurant_visit_id", "size"),
                n_subjects=("subject_id", "nunique"),
                n_sessions=("session_id", "nunique"),
                reward_rate=("reward", "mean"),
                mean_offer_delay_s=("offer_delay_s", "mean"),
                mean_pause_time_s=("pause_time_s", "mean"),
                mean_lab_idphi=("lab_idphi", "mean"),
                mean_lab_avg_dphi=("lab_avg_dphi", "mean"),
            )
            .reset_index()
        )

        work["delay_bin"] = work["offer_delay_s"].apply(_delay_bin)

        by_delay = (
            work.groupby(
                [
                    "dataset_id",
                    "decision_stage",
                    "stage_applicable",
                    "delay_bin",
                    "stage_decision",
                    "restaurant_outcome",
                ],
                dropna=False,
            )
            .agg(
                n_rows=("restaurant_visit_id", "size"),
                n_subjects=("subject_id", "nunique"),
                n_sessions=("session_id", "nunique"),
                reward_rate=("reward", "mean"),
                mean_offer_delay_s=("offer_delay_s", "mean"),
                mean_pause_time_s=("pause_time_s", "mean"),
                mean_lab_idphi=("lab_idphi", "mean"),
                mean_lab_avg_dphi=("lab_avg_dphi", "mean"),
            )
            .reset_index()
        )

        by_session = (
            work.groupby(["dataset_id", "subject_id", "session_id"], dropna=False)
            .agg(
                n_rows=("restaurant_visit_id", "size"),
                n_usable_stage_rows=("stage_applicable", "sum"),
                n_restaurants=("restaurant_id", "nunique"),
                n_outcomes=("restaurant_outcome", "nunique"),
                reward_rate=("reward", "mean"),
                mean_offer_delay_s=("offer_delay_s", "mean"),
                mean_pause_time_s=("pause_time_s", "mean"),
                mean_lab_idphi=("lab_idphi", "mean"),
                mean_lab_avg_dphi=("lab_avg_dphi", "mean"),
            )
            .reset_index()
        )

    by_stage_path = output_dir / OUTPUT_BY_STAGE
    by_delay_path = output_dir / OUTPUT_BY_DELAY
    by_session_path = output_dir / OUTPUT_BY_SESSION

    by_stage.to_csv(by_stage_path, index=False)
    by_delay.to_csv(by_delay_path, index=False)
    by_session.to_csv(by_session_path, index=False)

    paths["by_stage"] = str(by_stage_path)
    paths["by_delay"] = str(by_delay_path)
    paths["by_session"] = str(by_session_path)

    return paths


def _write_report(meta: dict[str, Any], output_dir: Path) -> Path:
    report_path = output_dir / OUTPUT_REPORT

    lines = [
        "# Redish Restaurant Row decision endpoint",
        "",
        "## Purpose",
        "",
        "This artifact rewrites diagnostic MATLAB-derived endpoint rows into an author-level Restaurant Row decision schema.",
        "",
        "## Semantic rules",
        "",
        "- `Skip`: offer-zone rejection; no reward.",
        "- `Earn`: offer-zone accept followed by waiting to reward; reward = 1.",
        "- `Quit`: offer-zone accept followed by wait-zone quit; reward = 0.",
        "- `Accept` is not treated as reward.",
        "",
        "## Main outputs",
        "",
        f"- `{OUTPUT_ENDPOINT}`: all exploded decision endpoint rows.",
        f"- `{OUTPUT_USABLE}`: rows with usable stage decision, delay, pause, and IdPhi proxies.",
        f"- `{OUTPUT_BY_STAGE}`: stage/outcome summary.",
        f"- `{OUTPUT_BY_DELAY}`: delay-binned summary.",
        f"- `{OUTPUT_BY_SESSION}`: session-level summary.",
        "",
        "## Counts",
        "",
        f"- source rows: {meta.get('n_source_rows')}",
        f"- decision endpoint rows: {meta.get('n_endpoint_rows')}",
        f"- usable rows: {meta.get('n_usable_rows')}",
        f"- subjects: {meta.get('n_subjects')}",
        f"- sessions: {meta.get('n_sessions')}",
        "",
        "## Methodological note",
        "",
        "Rows with `decision_stage=wait_zone` and `restaurant_outcome=skip` are not usable wait-zone decisions because a skipped offer does not enter the wait zone.",
        "",
    ]

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def build_decision_endpoint(
    input_csv: str | Path,
    output_dir: str | Path,
    dataset_id: str = "redish_rrow_2022",
) -> dict[str, Any]:
    input_csv = Path(input_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source = pd.read_csv(input_csv, low_memory=False)
    if "dataset_id" not in source.columns:
        source["dataset_id"] = dataset_id

    rows: list[dict[str, Any]] = []
    for _, row in source.iterrows():
        rows.extend(_explode_source_row(row))

    endpoint = pd.DataFrame(rows)

    if not endpoint.empty:
        endpoint = _numeric_columns(
            endpoint,
            [
                "trial",
                "source_zone_slot",
                "restaurant_id",
                "reward",
                "offer_delay_s",
                "total_site_time_s",
                "pause_time_s",
                "run_speed",
                "lab_idphi",
                "lab_avg_dphi",
                "site_rank",
                "entering_zone_time",
                "exit_zone_time",
                "zone_time_s",
                "adj_wz_exit_time",
                "current_cycle",
                "threshold_sec",
                "subjective_value",
            ],
        )

    usable = _build_usable(endpoint)

    endpoint_path = output_dir / OUTPUT_ENDPOINT
    usable_path = output_dir / OUTPUT_USABLE

    endpoint.to_csv(endpoint_path, index=False)
    usable.to_csv(usable_path, index=False)

    summary_paths = _write_summaries(endpoint, output_dir)

    meta: dict[str, Any] = {
        "dataset_id": dataset_id,
        "input_csv": str(input_csv),
        "n_source_rows": int(len(source)),
        "n_endpoint_rows": int(len(endpoint)),
        "n_usable_rows": int(len(usable)),
        "n_subjects": int(endpoint["subject_id"].nunique()) if "subject_id" in endpoint.columns and not endpoint.empty else 0,
        "n_sessions": int(endpoint["session_id"].nunique()) if "session_id" in endpoint.columns and not endpoint.empty else 0,
        "outputs": {
            "decision_endpoint": str(endpoint_path),
            "usable_endpoint": str(usable_path),
            **summary_paths,
        },
    }

    report_path = _write_report(meta, output_dir)
    meta["outputs"]["report"] = str(report_path)

    meta_path = output_dir / OUTPUT_META
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Decision endpoint saved: {endpoint_path}")
    print(f"Usable decision endpoint saved: {usable_path}")
    print(f"Stage summary saved: {summary_paths['by_stage']}")
    print(f"Delay summary saved: {summary_paths['by_delay']}")
    print(f"Session summary saved: {summary_paths['by_session']}")
    print(f"Metadata saved: {meta_path}")
    print(f"Report saved: {report_path}")
    print(f"Decision rows: {meta['n_endpoint_rows']}")
    print(f"Usable rows: {meta['n_usable_rows']}")

    return meta


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build semantic Redish Restaurant Row decision endpoint."
    )
    parser.add_argument("--input-csv", required=True, help="Raw 16B endpoint CSV.")
    parser.add_argument("--output-dir", required=True, help="Output directory.")
    parser.add_argument("--dataset-id", default="redish_rrow_2022")
    args = parser.parse_args()

    build_decision_endpoint(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        dataset_id=args.dataset_id,
    )


if __name__ == "__main__":
    main()