from __future__ import annotations

import argparse
import ast
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


VECTOR_FIELDS = [
    "trial",
    "zone_id",
    "zone_delay",
    "site_rank",
    "entering_zone_time",
    "exit_zone_time",
    "total_site_time",
    "zone_time",
    "pause_time",
    "run_speed",
    "adj_wz_exit_time",
    "lab_idphi",
    "lab_avg_dphi",
    "current_cycle",
]

ID_FIELDS = [
    "dataset_id",
    "source_version",
    "trace_origin",
    "adapter",
    "subject_id",
    "session_id",
    "session_date",
    "session_index",
    "row_index",
    "choice_point_id",
    "choice",
    "accept_offer",
    "skip_offer",
    "quit_offer",
    "earn_offer",
    "food_received",
    "reward",
    "decision",
    "delays_json",
    "exp_type",
    "source_lapdata_file",
    "source_idphi_file",
]


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if not np.isfinite(float(value)):
            return None
        return float(value)
    return str(value)


def _is_blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    text = str(value).strip()
    return text == "" or text.lower() in {"nan", "none", "null"}


def _parse_vector(value: Any) -> list[Any]:
    if _is_blank(value):
        return [""]

    if isinstance(value, (list, tuple, np.ndarray)):
        return list(value)

    text = str(value).strip()

    if text.startswith("[") and text.endswith("]"):
        normalized = (
            text.replace("NaN", "nan")
            .replace("nan", "None")
            .replace("Inf", "None")
            .replace("-Inf", "None")
        )
        try:
            parsed = ast.literal_eval(normalized)
            if isinstance(parsed, (list, tuple)):
                return list(parsed)
        except Exception:
            pass

        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass

    return [value]


def _as_scalar(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        val = float(value)
        if not np.isfinite(val):
            return ""
        return int(val) if val.is_integer() else val

    text = str(value).strip()
    if text.lower() in {"nan", "none", "null"}:
        return ""

    try:
        val = float(text)
        if not np.isfinite(val):
            return ""
        return int(val) if val.is_integer() else val
    except Exception:
        return text


def _choice_from_row(row: pd.Series) -> str:
    choice = str(row.get("choice", "")).strip()
    decision = str(row.get("decision", "")).strip()

    for value in [choice, decision]:
        if value and value.lower() not in {"nan", "none", "null"}:
            low = value.lower()
            if low in {"earn", "earned", "accept", "accepted"}:
                return "accept"
            if low in {"skip", "reject", "rejected"}:
                return "skip"
            if low == "quit":
                return "quit"
            if low == "n/a":
                return "n/a"
            return value

    accept = str(row.get("accept_offer", "")).strip()
    skip = str(row.get("skip_offer", "")).strip()
    quit_offer = str(row.get("quit_offer", "")).strip()

    if accept in {"1", "1.0", "True", "true"}:
        return "accept"
    if skip in {"1", "1.0", "True", "true"}:
        return "skip"
    if quit_offer in {"1", "1.0", "True", "true"}:
        return "quit"

    return ""


def _zone_type(zone_value: Any, choice_point_id: Any = "") -> str:
    text = str(zone_value).strip().lower()
    cp = str(choice_point_id).strip().lower()

    combined = f"{text} {cp}"

    if "wait" in combined:
        return "wait_zone"
    if "offer" in combined:
        return "offer_zone"
    if "linger" in combined:
        return "linger_zone"

    try:
        zone = int(float(text))
        if 1 <= zone <= 4:
            return "wait_zone"
        if 5 <= zone <= 8:
            return "offer_zone"
    except Exception:
        pass

    return ""


def _reward_from_row(row: pd.Series) -> Any:
    reward = _as_scalar(row.get("reward", ""))
    if reward != "":
        return reward

    for field in ["food_received", "earn_offer"]:
        value = _as_scalar(row.get(field, ""))
        if value != "":
            try:
                return int(float(value) != 0.0)
            except Exception:
                pass

    choice = _choice_from_row(row)
    if choice == "accept":
        return _as_scalar(row.get("earn_offer", ""))

    return ""


def _explode_row(row: pd.Series) -> list[dict[str, Any]]:
    parsed = {field: _parse_vector(row.get(field, "")) for field in VECTOR_FIELDS}

    max_len = max(len(values) for values in parsed.values()) if parsed else 1

    # A row with only scalar fields stays a single row.
    out_rows: list[dict[str, Any]] = []

    for idx in range(max_len):
        out: dict[str, Any] = {}

        for field in ID_FIELDS:
            if field in row.index:
                out[field] = row.get(field, "")

        out["zone_slot"] = idx

        for field, values in parsed.items():
            if len(values) == 1:
                value = values[0]
            elif idx < len(values):
                value = values[idx]
            else:
                value = ""

            out[field] = _as_scalar(value)

        out["choice"] = _choice_from_row(row)
        out["reward"] = _reward_from_row(row)
        out["zone_type"] = _zone_type(out.get("zone_id", ""), row.get("choice_point_id", ""))

        zone_id = out.get("zone_id", "")
        if zone_id != "":
            out["choice_point_id"] = f"redish_rrow_{out['zone_type'] or 'zone'}_{zone_id}"

        out_rows.append(out)

    return out_rows


def normalize_endpoint_table(
    input_csv: Path,
    output_dir: Path,
    *,
    drop_empty_vector_rows: bool = True,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    source = pd.read_csv(input_csv, low_memory=False)

    rows: list[dict[str, Any]] = []
    for _, row in source.iterrows():
        rows.extend(_explode_row(row))

    normalized = pd.DataFrame(rows)

    if drop_empty_vector_rows and not normalized.empty:
        informative_cols = [
            "trial",
            "zone_id",
            "zone_delay",
            "total_site_time",
            "pause_time",
            "lab_idphi",
            "lab_avg_dphi",
        ]

        mask = pd.Series(False, index=normalized.index)
        for col in informative_cols:
            if col in normalized.columns:
                mask = mask | normalized[col].astype(str).str.strip().ne("")

        normalized = normalized.loc[mask].reset_index(drop=True)

    numeric_cols = [
        "trial",
        "zone_id",
        "zone_delay",
        "site_rank",
        "entering_zone_time",
        "exit_zone_time",
        "total_site_time",
        "zone_time",
        "pause_time",
        "run_speed",
        "adj_wz_exit_time",
        "lab_idphi",
        "lab_avg_dphi",
        "current_cycle",
        "reward",
    ]

    for col in numeric_cols:
        if col in normalized.columns:
            normalized[col] = pd.to_numeric(normalized[col], errors="ignore")

    endpoint_path = output_dir / "Table_Redish_RRow_Choice_IdPhi_Endpoint_Normalized.csv"
    by_choice_path = output_dir / "Table_Redish_RRow_Normalized_By_Choice.csv"
    by_session_path = output_dir / "Table_Redish_RRow_Normalized_By_Session.csv"
    meta_path = output_dir / "redish_rrow_normalized_endpoint_meta.json"

    normalized.to_csv(endpoint_path, index=False)

    if normalized.empty:
        by_choice = pd.DataFrame()
        by_session = pd.DataFrame()
    else:
        work = normalized.copy()
        for col in ["reward", "lab_idphi", "lab_avg_dphi", "zone_delay", "pause_time", "total_site_time"]:
            if col in work.columns:
                work[col] = pd.to_numeric(work[col], errors="coerce")

        by_choice = (
            work.groupby(["dataset_id", "subject_id", "session_id", "zone_type", "choice"], dropna=False)
            .agg(
                n_rows=("choice", "size"),
                reward_rate=("reward", "mean"),
                mean_zone_delay=("zone_delay", "mean"),
                mean_pause_time=("pause_time", "mean"),
                mean_total_site_time=("total_site_time", "mean"),
                mean_lab_idphi=("lab_idphi", "mean"),
                mean_lab_avg_dphi=("lab_avg_dphi", "mean"),
            )
            .reset_index()
        )

        by_session = (
            work.groupby(["dataset_id", "subject_id", "session_id"], dropna=False)
            .agg(
                n_rows=("choice", "size"),
                n_zone_slots=("zone_slot", "nunique"),
                n_zone_values=("zone_id", lambda s: int(pd.Series(s).nunique(dropna=True))),
                n_choices=("choice", lambda s: int(pd.Series(s).astype(str).replace("", np.nan).nunique(dropna=True))),
                reward_rate=("reward", "mean"),
                mean_zone_delay=("zone_delay", "mean"),
                mean_pause_time=("pause_time", "mean"),
                mean_lab_idphi=("lab_idphi", "mean"),
            )
            .reset_index()
        )

    by_choice.to_csv(by_choice_path, index=False)
    by_session.to_csv(by_session_path, index=False)

    meta = {
        "input_csv": str(input_csv),
        "n_source_rows": int(len(source)),
        "n_normalized_rows": int(len(normalized)),
        "drop_empty_vector_rows": drop_empty_vector_rows,
        "outputs": {
            "normalized_endpoint": str(endpoint_path),
            "by_choice": str(by_choice_path),
            "by_session": str(by_session_path),
        },
    }

    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")

    print(f"Normalized endpoint saved: {endpoint_path}")
    print(f"Normalized choice summary saved: {by_choice_path}")
    print(f"Normalized session summary saved: {by_session_path}")
    print(f"Metadata saved: {meta_path}")

    return meta


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize Redish RRow vector-valued endpoint rows into atomic endpoint rows."
    )
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--keep-empty-vector-rows", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    normalize_endpoint_table(
        input_csv=Path(args.input_csv),
        output_dir=Path(args.output_dir),
        drop_empty_vector_rows=not args.keep_empty_vector_rows,
    )


if __name__ == "__main__":
    main()