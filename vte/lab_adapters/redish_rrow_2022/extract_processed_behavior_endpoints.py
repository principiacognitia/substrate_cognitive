from __future__ import annotations

import argparse
import json
import math
import re
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .inventory_redish_rrow import DATASET_ID_DEFAULT, build_file_inventory


PUBLIC_MATLAB_PREFIXES = ("__",)

SESSION_METADATA_FIELDS = [
    "Directory",
    "Date",
    "ExpType",
    "Experimenter",
    "Rat",
    "Rats",
    "Subject",
    "Delays",
]

LAP_ENDPOINT_FIELDS = [
    "AcceptOffer",
    "SkipOffer",
    "QuitOffer",
    "EarnOffer",
    "FoodReceived",
    "Decision",
    "Laps",
    "CurrentLap",
    "CurrentCycle",
    "OfferZone",
    "ZoneID",
    "ZoneDelay",
    "SiteRank",
    "EnteringZoneTime",
    "ExitZoneTime",
    "TotalSiteTime",
    "ZoneTime",
    "PauseTime",
    "RunSpeed",
    "AdjWZExitTime",
    "AvgDPhi",
    "IdPhi",
]

IDPHI_FIELDS = [
    "AvgDPhi",
    "IdPhi",
    "Laps",
    "CurrentLap",
    "ZoneID",
    "ZoneDelay",
    "OfferZone",
    "Decision",
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
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def _public_key(key: str) -> bool:
    return not str(key).startswith(PUBLIC_MATLAB_PREFIXES)


def _is_mat_struct(value: Any) -> bool:
    return hasattr(value, "_fieldnames")


def _mat_struct_to_dict(value: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for field in getattr(value, "_fieldnames", None) or []:
        try:
            out[str(field)] = getattr(value, field)
        except Exception:
            continue
    return out


def _loadmat_public(path: Path) -> tuple[dict[str, Any], str]:
    scipy_io = __import__("scipy.io", fromlist=["loadmat"])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        data = scipy_io.loadmat(
            str(path),
            squeeze_me=True,
            struct_as_record=False,
        )

    warning_messages = "; ".join(dict.fromkeys(str(w.message) for w in caught))

    public = {str(k): v for k, v in data.items() if _public_key(str(k))}

    # If a file stores one top-level MATLAB struct, expose both the struct name and its fields.
    if len(public) == 1:
        only_key, only_value = next(iter(public.items()))
        if _is_mat_struct(only_value):
            expanded = _mat_struct_to_dict(only_value)
            expanded[only_key] = only_value
            return expanded, warning_messages

    expanded = dict(public)
    for key, value in list(public.items()):
        if _is_mat_struct(value):
            for child_key, child_value in _mat_struct_to_dict(value).items():
                expanded.setdefault(child_key, child_value)

    return expanded, warning_messages


def _as_scalar(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        arr = np.asarray(value)
        if arr.size == 0:
            return ""
        if arr.size == 1:
            return _as_scalar(arr.ravel()[0])
        return value

    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        val = float(value)
        return "" if not np.isfinite(val) else val
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        return value

    return value


def _is_scalar_like(value: Any) -> bool:
    if isinstance(value, (str, bytes, int, float, bool, np.integer, np.floating)):
        return True
    if isinstance(value, np.ndarray):
        return value.size <= 1
    return False


def _flatten_nonempty(value: Any) -> list[Any]:
    if value is None:
        return []

    if _is_mat_struct(value):
        return [value]

    if isinstance(value, np.ndarray):
        if value.size == 0:
            return []

        if value.dtype == object:
            return [item for item in value.ravel().tolist()]

        if value.ndim == 0:
            return [_as_scalar(value.item())]

        return value.ravel().tolist()

    if isinstance(value, (list, tuple)):
        return list(value)

    return [value]


def _cell_list_for_sessions(value: Any, n_sessions: int) -> list[Any]:
    """Return one cell/vector per session when possible.

    Redish aggregate files can store session-wise data in several forms:

    1. MATLAB cell-like object array of length n_sessions.
    2. Numeric/object matrix with sessions on axis 0.
    3. Numeric/object matrix with sessions on axis 1.
    4. Scalar/session metadata value.

    The important safety rule is: never replicate a large matrix across all
    sessions. If a field cannot be aligned to sessions, return blanks and let
    downstream diagnostics reveal the missing field.
    """
    if value is None:
        return [""] * n_sessions

    if _is_scalar_like(value):
        return [_to_row_value(value)] * n_sessions

    if isinstance(value, np.ndarray):
        arr = np.asarray(value)

        if arr.size == 0:
            return [""] * n_sessions

        if arr.dtype == object:
            flat = arr.ravel().tolist()

            if len(flat) == n_sessions:
                return flat

            if arr.ndim >= 2 and arr.shape[0] == n_sessions:
                return [arr[i, ...] for i in range(n_sessions)]

            if arr.ndim >= 2 and arr.shape[1] == n_sessions:
                return [arr[:, i, ...] for i in range(n_sessions)]

            if len(flat) == 1:
                return flat * n_sessions

            # Do not replicate unaligned object arrays.
            return [""] * n_sessions

        if arr.ndim >= 2 and arr.shape[0] == n_sessions:
            return [arr[i, ...] for i in range(n_sessions)]

        if arr.ndim >= 2 and arr.shape[1] == n_sessions:
            return [arr[:, i, ...] for i in range(n_sessions)]

        if arr.ndim == 1 and arr.shape[0] == n_sessions:
            return arr.tolist()

        if arr.size == n_sessions:
            return arr.ravel().tolist()

        # Do not replicate unaligned numeric matrices.
        return [""] * n_sessions

    if isinstance(value, (list, tuple)):
        if len(value) == n_sessions:
            return list(value)
        if len(value) == 1:
            return list(value) * n_sessions
        return [""] * n_sessions

    return [_to_row_value(value)] * n_sessions


def _session_count_from_data_def(data_def: dict[str, Any]) -> int:
    preferred = ["Directory", "Date", "ExpType", "Delays"]
    for field in preferred:
        if field in data_def:
            values = _flatten_nonempty(data_def[field])
            if values:
                return len(values)

    max_len = 0
    for value in data_def.values():
        if isinstance(value, np.ndarray):
            max_len = max(max_len, int(value.size))
        elif isinstance(value, (list, tuple)):
            max_len = max(max_len, len(value))

    return max(max_len, 1)


def _to_row_value(value: Any) -> Any:
    value = _as_scalar(value)

    if _is_mat_struct(value):
        return json.dumps(_mat_struct_to_dict(value), ensure_ascii=False, default=_json_default)

    if isinstance(value, np.ndarray):
        arr = np.asarray(value)
        if arr.size == 0:
            return ""
        if arr.size == 1:
            return _to_row_value(arr.ravel()[0])
        return json.dumps(arr.tolist(), ensure_ascii=False, default=_json_default)

    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return ""
        if len(value) == 1:
            return _to_row_value(value[0])
        return json.dumps(list(value), ensure_ascii=False, default=_json_default)

    return value


def _vector_for_field(value: Any) -> list[Any]:
    if value is None:
        return []

    if _is_scalar_like(value):
        return [_to_row_value(value)]

    if isinstance(value, np.ndarray):
        arr = np.asarray(value)
        if arr.size == 0:
            return []

        if arr.dtype == object:
            return [_to_row_value(item) for item in arr.ravel().tolist()]

        if arr.ndim == 1:
            return [_to_row_value(item) for item in arr.tolist()]

        if arr.ndim == 2 and 1 in arr.shape:
            return [_to_row_value(item) for item in arr.ravel().tolist()]

        # For 2D matrices, preserve row records if possible.
        return [_to_row_value(arr[i, :]) for i in range(arr.shape[0])]

    if isinstance(value, (list, tuple)):
        return [_to_row_value(item) for item in value]

    return [_to_row_value(value)]


def _safe_get(values: list[Any], idx: int) -> Any:
    if not values:
        return ""
    if len(values) == 1:
        return values[0]
    if idx < len(values):
        return values[idx]
    return ""


def _num_or_blank(value: Any) -> Any:
    if value == "" or value is None:
        return ""
    try:
        val = float(value)
        if not np.isfinite(val):
            return ""
        if val.is_integer():
            return int(val)
        return val
    except Exception:
        return value


def _bool01(value: Any) -> int | str:
    value = _num_or_blank(value)
    if value == "":
        return ""
    try:
        return int(float(value) != 0.0)
    except Exception:
        text = str(value).strip().lower()
        if text in {"true", "yes", "accept", "accepted"}:
            return 1
        if text in {"false", "no", "skip", "quit", "reject"}:
            return 0
    return ""


def _session_id_from_directory(value: Any) -> str:
    text = str(value).strip()
    if not text:
        return ""

    m = re.search(r"(R\d+[-_]\d{4}[-_]\d{2}[-_]\d{2})", text)
    if m:
        return m.group(1).replace("_", "-")

    m = re.search(r"(R\d+)", text)
    if m:
        return text

    return text


def _subject_from_session(session_id: str, fallback: Any = "") -> str:
    text = str(session_id or fallback)
    m = re.search(r"(R\d+)", text)
    return m.group(1) if m else str(fallback or "")


def _date_from_session(session_id: str, fallback: Any = "") -> str:
    text = str(session_id or fallback)
    m = re.search(r"(\d{4}-\d{2}-\d{2})", text)
    if m:
        return m.group(1)

    m = re.search(r"(\d{4}_\d{2}_\d{2})", text)
    if m:
        return m.group(1).replace("_", "-")

    return str(fallback or "")


def _zone_type(zone_id: Any) -> str:
    val = _num_or_blank(zone_id)
    if val == "":
        return ""

    try:
        zone = int(float(val))
    except Exception:
        return ""

    if 1 <= zone <= 4:
        return "wait_zone"
    if 5 <= zone <= 8:
        return "offer_zone"
    return "other_zone"


def _choice_from_row(row: dict[str, Any]) -> str:
    accept = _bool01(row.get("AcceptOffer", ""))
    skip = _bool01(row.get("SkipOffer", ""))
    quit_offer = _bool01(row.get("QuitOffer", ""))
    decision = row.get("Decision", "")

    if accept == 1:
        return "accept"
    if skip == 1:
        return "skip"
    if quit_offer == 1:
        return "quit"

    if decision != "":
        return str(decision)

    return ""


def _reward_from_row(row: dict[str, Any]) -> int | str:
    for field in ["FoodReceived", "EarnOffer"]:
        value = _bool01(row.get(field, ""))
        if value != "":
            if value == 1:
                return 1

    values = [_bool01(row.get(field, "")) for field in ["FoodReceived", "EarnOffer"]]
    if any(v == 0 for v in values):
        return 0

    return ""


def _find_single_file(inventory: pd.DataFrame, file_kind: str, version_label_contains: str | None) -> Path | None:
    subset = inventory.loc[inventory["file_kind"] == file_kind].copy()

    if version_label_contains:
        subset = subset.loc[
            subset["version_label"].astype(str).str.contains(
                version_label_contains,
                case=False,
                regex=False,
            )
        ]

    if subset.empty:
        return None

    subset = subset.sort_values(["version_label", "file_path"])
    return Path(subset.iloc[-1]["file_path"])


def _build_session_metadata(data_def: dict[str, Any], n_sessions: int) -> pd.DataFrame:
    cells_by_field = {
        field: _cell_list_for_sessions(data_def.get(field), n_sessions)
        for field in SESSION_METADATA_FIELDS
        if field in data_def
    }

    rows = []
    for session_index in range(n_sessions):
        row: dict[str, Any] = {"session_index": session_index}

        for field, cells in cells_by_field.items():
            raw = cells[session_index] if session_index < len(cells) else ""
            if field == "Delays":
                row["delays_json"] = _to_row_value(raw)
            else:
                row[field] = _to_row_value(raw)

        directory = row.get("Directory", "")
        session_id = _session_id_from_directory(directory)
        row["session_id"] = session_id
        row["subject_id"] = _subject_from_session(session_id, row.get("Rat", row.get("Rats", row.get("Subject", ""))))
        row["session_date"] = _date_from_session(session_id, row.get("Date", ""))

        rows.append(row)

    return pd.DataFrame(rows)


def _build_long_table(
    payload: dict[str, Any],
    *,
    n_sessions: int,
    fields: list[str],
    source_file: Path,
    table_role: str,
    max_rows_per_session: int | None = None,
) -> pd.DataFrame:
    cells_by_field = {
        field: _cell_list_for_sessions(payload.get(field), n_sessions)
        for field in fields
        if field in payload
    }

    rows: list[dict[str, Any]] = []

    for session_index in range(n_sessions):
        vectors: dict[str, list[Any]] = {}

        for field, cells in cells_by_field.items():
            cell = cells[session_index] if session_index < len(cells) else ""
            vectors[field] = _vector_for_field(cell)

        n_rows = 0
        for values in vectors.values():
            if len(values) > 1:
                n_rows = max(n_rows, len(values))

        if n_rows == 0 and vectors:
            n_rows = 1

        if max_rows_per_session is not None:
            n_rows = min(n_rows, max_rows_per_session)

        for row_index in range(n_rows):
            row = {
                "session_index": session_index,
                "row_index": row_index,
                "table_role": table_role,
                "source_file": str(source_file),
            }

            for field, values in vectors.items():
                row[field] = _safe_get(values, row_index)

            rows.append(row)

    return pd.DataFrame(rows)


def _prefix_columns(df: pd.DataFrame, prefix: str, keep: set[str]) -> pd.DataFrame:
    rename = {
        col: f"{prefix}{col}"
        for col in df.columns
        if col not in keep
    }
    return df.rename(columns=rename)


def _coalesce(row: pd.Series, *columns: str) -> Any:
    for col in columns:
        if col in row.index:
            value = row[col]
            if value is not None and value == value and str(value) != "":
                return value
    return ""


def _build_endpoint_table(
    lap_long: pd.DataFrame,
    idphi_long: pd.DataFrame,
    session_meta: pd.DataFrame,
    *,
    dataset_id: str,
    source_version: str,
) -> pd.DataFrame:
    if lap_long.empty:
        return pd.DataFrame()

    idphi_prefixed = _prefix_columns(
        idphi_long,
        "idphi_",
        keep={"session_index", "row_index"},
    )

    merged = lap_long.merge(
        idphi_prefixed,
        on=["session_index", "row_index"],
        how="left",
        suffixes=("", "_idphi"),
    )

    merged = merged.merge(
        session_meta,
        on="session_index",
        how="left",
        suffixes=("", "_session"),
    )

    endpoint_rows: list[dict[str, Any]] = []

    for _, row in merged.iterrows():
        zone_id = _coalesce(row, "ZoneID", "OfferZone", "idphi_ZoneID", "idphi_OfferZone")
        trial = _coalesce(row, "CurrentLap", "Laps", "idphi_CurrentLap", "idphi_Laps", "row_index")
        lab_idphi = _coalesce(row, "IdPhi", "idphi_IdPhi")
        lab_avg_dphi = _coalesce(row, "AvgDPhi", "idphi_AvgDPhi")

        out = {
            "dataset_id": dataset_id,
            "source_version": source_version,
            "trace_origin": "biological",
            "adapter": "redish_rrow_2022_processed_behavior",
            "subject_id": row.get("subject_id", ""),
            "session_id": row.get("session_id", ""),
            "session_date": row.get("session_date", ""),
            "session_index": row.get("session_index", ""),
            "row_index": row.get("row_index", ""),
            "trial": _num_or_blank(trial),
            "choice_point_id": f"rrow_zone_{zone_id}" if str(zone_id) != "" else "",
            "zone_id": _num_or_blank(zone_id),
            "zone_type": _zone_type(zone_id),
            "choice": _choice_from_row(row.to_dict()),
            "accept_offer": _bool01(row.get("AcceptOffer", "")),
            "skip_offer": _bool01(row.get("SkipOffer", "")),
            "quit_offer": _bool01(row.get("QuitOffer", "")),
            "earn_offer": _bool01(row.get("EarnOffer", "")),
            "food_received": _bool01(row.get("FoodReceived", "")),
            "reward": _reward_from_row(row.to_dict()),
            "decision": row.get("Decision", ""),
            "zone_delay": _num_or_blank(_coalesce(row, "ZoneDelay", "idphi_ZoneDelay")),
            "site_rank": _num_or_blank(row.get("SiteRank", "")),
            "entering_zone_time": _num_or_blank(row.get("EnteringZoneTime", "")),
            "exit_zone_time": _num_or_blank(row.get("ExitZoneTime", "")),
            "total_site_time": _num_or_blank(row.get("TotalSiteTime", "")),
            "zone_time": _num_or_blank(row.get("ZoneTime", "")),
            "pause_time": _num_or_blank(row.get("PauseTime", "")),
            "run_speed": _num_or_blank(row.get("RunSpeed", "")),
            "adj_wz_exit_time": _num_or_blank(row.get("AdjWZExitTime", "")),
            "lab_idphi": _num_or_blank(lab_idphi),
            "lab_avg_dphi": _num_or_blank(lab_avg_dphi),
            "current_cycle": _num_or_blank(row.get("CurrentCycle", "")),
            "delays_json": row.get("delays_json", ""),
            "exp_type": row.get("ExpType", ""),
            "source_lapdata_file": row.get("source_file", ""),
            "source_idphi_file": row.get("idphi_source_file", ""),
        }

        endpoint_rows.append(out)

    endpoint = pd.DataFrame(endpoint_rows)

    # Keep rows that at least contain a choice, a zone, or an IdPhi-like metric.
    if not endpoint.empty:
        informative = (
            endpoint["choice"].astype(str).ne("")
            | endpoint["zone_id"].astype(str).ne("")
            | endpoint["lab_idphi"].astype(str).ne("")
            | endpoint["lab_avg_dphi"].astype(str).ne("")
        )
        endpoint = endpoint.loc[informative].reset_index(drop=True)

    return endpoint


def _build_summary_tables(endpoint: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if endpoint.empty:
        empty_choice = pd.DataFrame(
            columns=[
                "dataset_id",
                "subject_id",
                "session_id",
                "zone_type",
                "choice",
                "n_rows",
                "reward_rate",
                "mean_lab_idphi",
                "mean_lab_avg_dphi",
            ]
        )
        empty_session = pd.DataFrame(
            columns=[
                "dataset_id",
                "subject_id",
                "session_id",
                "n_rows",
                "n_zones",
                "n_choices",
                "reward_rate",
                "mean_lab_idphi",
            ]
        )
        return empty_choice, empty_session

    work = endpoint.copy()

    for col in ["reward", "lab_idphi", "lab_avg_dphi"]:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    by_choice = (
        work.groupby(["dataset_id", "subject_id", "session_id", "zone_type", "choice"], dropna=False)
        .agg(
            n_rows=("choice", "size"),
            reward_rate=("reward", "mean"),
            mean_lab_idphi=("lab_idphi", "mean"),
            mean_lab_avg_dphi=("lab_avg_dphi", "mean"),
        )
        .reset_index()
    )

    by_session = (
        work.groupby(["dataset_id", "subject_id", "session_id"], dropna=False)
        .agg(
            n_rows=("choice", "size"),
            n_zones=("zone_id", lambda s: int(pd.Series(s).nunique(dropna=True))),
            n_choices=("choice", lambda s: int(pd.Series(s).astype(str).replace("", np.nan).nunique(dropna=True))),
            reward_rate=("reward", "mean"),
            mean_lab_idphi=("lab_idphi", "mean"),
        )
        .reset_index()
    )

    return by_choice, by_session


def extract_processed_behavior_endpoints(
    root: Path,
    output_dir: Path,
    *,
    dataset_id: str = DATASET_ID_DEFAULT,
    version_label_contains: str = "Version 002",
    max_rows_per_session: int | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    inventory = build_file_inventory(root, dataset_id=dataset_id)

    data_def_file = _find_single_file(inventory, "data_definition", version_label_contains)
    lapdata_file = _find_single_file(inventory, "aggregate_lapdata_behavior", version_label_contains)
    idphi_file = _find_single_file(inventory, "aggregate_idphi", version_label_contains)
    sessiondata_file = _find_single_file(inventory, "aggregate_sessiondata", version_label_contains)

    missing = [
        name
        for name, path in [
            ("DataDef_RRow.mat", data_def_file),
            ("LapData_Behav_RRow.mat", lapdata_file),
            ("IdPhi_RRow.mat", idphi_file),
            ("SessionData_RRow.mat", sessiondata_file),
        ]
        if path is None
    ]

    if data_def_file is None or lapdata_file is None or idphi_file is None:
        raise FileNotFoundError(f"Missing required Redish RRow files: {missing}")

    data_def, data_def_warnings = _loadmat_public(data_def_file)
    lapdata, lapdata_warnings = _loadmat_public(lapdata_file)
    idphi, idphi_warnings = _loadmat_public(idphi_file)

    sessiondata = {}
    sessiondata_warnings = ""
    if sessiondata_file is not None:
        sessiondata, sessiondata_warnings = _loadmat_public(sessiondata_file)

    n_sessions = _session_count_from_data_def(data_def)

    session_meta = _build_session_metadata(data_def, n_sessions)

    # SessionData can add fields, but DataDef remains the primary session map.
    if sessiondata:
        session_extra = _build_long_table(
            sessiondata,
            n_sessions=n_sessions,
            fields=[field for field in ["Session", "Sessions", "Rat", "Rats", "Date", "Directory"] if field in sessiondata],
            source_file=sessiondata_file or Path(""),
            table_role="sessiondata",
            max_rows_per_session=1,
        )
        if not session_extra.empty:
            session_extra_wide = session_extra.drop(columns=["row_index", "table_role", "source_file"], errors="ignore")
            session_meta = session_meta.merge(
                session_extra_wide,
                on="session_index",
                how="left",
                suffixes=("", "_sessiondata"),
            )

    lap_long = _build_long_table(
        lapdata,
        n_sessions=n_sessions,
        fields=LAP_ENDPOINT_FIELDS,
        source_file=lapdata_file,
        table_role="lapdata_behavior",
        max_rows_per_session=max_rows_per_session,
    )

    idphi_long = _build_long_table(
        idphi,
        n_sessions=n_sessions,
        fields=IDPHI_FIELDS,
        source_file=idphi_file,
        table_role="idphi",
        max_rows_per_session=max_rows_per_session,
    )

    endpoint = _build_endpoint_table(
        lap_long=lap_long,
        idphi_long=idphi_long,
        session_meta=session_meta,
        dataset_id=dataset_id,
        source_version=version_label_contains,
    )

    by_choice, by_session = _build_summary_tables(endpoint)

    session_meta_path = output_dir / "Table_Redish_RRow_Session_Metadata.csv"
    lap_long_path = output_dir / "Table_Redish_RRow_LapData_Long.csv"
    idphi_long_path = output_dir / "Table_Redish_RRow_IdPhi_Long.csv"
    endpoint_path = output_dir / "Table_Redish_RRow_Choice_IdPhi_Endpoint.csv"
    by_choice_path = output_dir / "Table_Redish_RRow_Endpoint_By_Choice.csv"
    by_session_path = output_dir / "Table_Redish_RRow_Endpoint_By_Session.csv"
    meta_path = output_dir / "redish_rrow_processed_behavior_endpoint_meta.json"
    report_path = output_dir / "Redish_RRow_Processed_Behavior_Endpoint_Report.md"

    session_meta.to_csv(session_meta_path, index=False)
    lap_long.to_csv(lap_long_path, index=False)
    idphi_long.to_csv(idphi_long_path, index=False)
    endpoint.to_csv(endpoint_path, index=False)
    by_choice.to_csv(by_choice_path, index=False)
    by_session.to_csv(by_session_path, index=False)

    meta = {
        "dataset_id": dataset_id,
        "root": str(Path(root).resolve()),
        "version_label_contains": version_label_contains,
        "n_sessions": int(n_sessions),
        "n_lap_rows": int(len(lap_long)),
        "n_idphi_rows": int(len(idphi_long)),
        "n_endpoint_rows": int(len(endpoint)),
        "n_subjects": int(endpoint["subject_id"].nunique()) if not endpoint.empty and "subject_id" in endpoint else 0,
        "n_endpoint_sessions": int(endpoint["session_id"].nunique()) if not endpoint.empty and "session_id" in endpoint else 0,
        "max_rows_per_session": max_rows_per_session,
        "source_files": {
            "data_def": str(data_def_file),
            "lapdata": str(lapdata_file),
            "idphi": str(idphi_file),
            "sessiondata": str(sessiondata_file) if sessiondata_file else None,
        },
        "mat_warnings": {
            "data_def": data_def_warnings,
            "lapdata": lapdata_warnings,
            "idphi": idphi_warnings,
            "sessiondata": sessiondata_warnings,
        },
        "outputs": {
            "session_meta": str(session_meta_path),
            "lap_long": str(lap_long_path),
            "idphi_long": str(idphi_long_path),
            "endpoint": str(endpoint_path),
            "by_choice": str(by_choice_path),
            "by_session": str(by_session_path),
            "report": str(report_path),
        },
    }

    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    report_path.write_text(_make_report(meta, endpoint, by_choice, by_session), encoding="utf-8")

    print(f"Session metadata saved: {session_meta_path}")
    print(f"LapData long table saved: {lap_long_path}")
    print(f"IdPhi long table saved: {idphi_long_path}")
    print(f"Choice-IdPhi endpoint saved: {endpoint_path}")
    print(f"Choice summary saved: {by_choice_path}")
    print(f"Session summary saved: {by_session_path}")
    print(f"Metadata saved: {meta_path}")
    print(f"Report saved: {report_path}")

    return meta


def _make_report(
    meta: dict[str, Any],
    endpoint: pd.DataFrame,
    by_choice: pd.DataFrame,
    by_session: pd.DataFrame,
) -> str:
    lines = [
        "# Redish RRow processed behavior endpoint report",
        "",
        "Patch 16B output.",
        "",
        "## Purpose",
        "",
        "This extractor maps Redish Restaurant Row processed behavioral files into a Stage 3.2C biological comparison endpoint. It preserves Redish-authored IdPhi / AvgDPhi values as laboratory measurements rather than recomputing them with the frozen VTE wrapper.",
        "",
        "## Summary",
        "",
        f"- Dataset ID: `{meta['dataset_id']}`",
        f"- Root: `{meta['root']}`",
        f"- Version filter: `{meta['version_label_contains']}`",
        f"- Sessions in DataDef: `{meta['n_sessions']}`",
        f"- Lap rows: `{meta['n_lap_rows']}`",
        f"- IdPhi rows: `{meta['n_idphi_rows']}`",
        f"- Endpoint rows: `{meta['n_endpoint_rows']}`",
        f"- Endpoint subjects: `{meta['n_subjects']}`",
        f"- Endpoint sessions: `{meta['n_endpoint_sessions']}`",
        "",
        "## Interpretation",
        "",
        "This is not a biological trace replay table. It is a processed endpoint table for comparing choice context, reward/outcome, delay/cost, and Redish-authored IdPhi-like deliberation measures.",
        "",
        "The table should be used for Stage 3.2C gate-viscosity comparability, not for visual similarity claims.",
        "",
        "## Main endpoint columns",
        "",
        "The primary endpoint table is `Table_Redish_RRow_Choice_IdPhi_Endpoint.csv`.",
        "",
        "Key fields:",
        "",
        "- `subject_id`, `session_id`, `trial`",
        "- `zone_id`, `zone_type`, `choice_point_id`",
        "- `choice`, `accept_offer`, `skip_offer`, `quit_offer`",
        "- `reward`, `earn_offer`, `food_received`",
        "- `zone_delay`, `site_rank`, `total_site_time`, `pause_time`, `run_speed`",
        "- `lab_idphi`, `lab_avg_dphi`",
        "",
        "## Choice summary preview",
        "",
    ]

    if by_choice.empty:
        lines.append("No choice summary rows produced.")
    else:
        lines.append(by_choice.head(30).to_markdown(index=False))

    lines.extend(["", "## Session summary preview", ""])

    if by_session.empty:
        lines.append("No session summary rows produced.")
    else:
        lines.append(by_session.head(30).to_markdown(index=False))

    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- Field alignment is based on session index and row index from processed Redish aggregate files.",
            "- Patch 16B does not yet validate every field against the original MATLAB plotting pipeline.",
            "- Neural proxy fields are intentionally excluded here.",
            "- The frozen Stage 3.2 VTE wrapper is not modified and is not required for this endpoint.",
            "",
        ]
    )

    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract Redish RRow processed behavior into Stage 3.2C Choice-IdPhi endpoint tables."
    )
    parser.add_argument("--root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-id", default=DATASET_ID_DEFAULT)
    parser.add_argument("--version-label-contains", default="Version 002")
    parser.add_argument("--max-rows-per-session", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    extract_processed_behavior_endpoints(
        root=Path(args.root),
        output_dir=Path(args.output_dir),
        dataset_id=args.dataset_id,
        version_label_contains=args.version_label_contains,
        max_rows_per_session=args.max_rows_per_session,
    )


if __name__ == "__main__":
    main()