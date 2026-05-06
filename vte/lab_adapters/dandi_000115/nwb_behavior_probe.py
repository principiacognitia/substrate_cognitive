"""DANDI 000115 behavioral NWB probe.

This module inspects a local NWB file from DANDI 000115
(Gillespie et al., 2021) without reading large ecephys arrays.

It extracts only behavior-relevant layers:
- intervals/epochs
- processing/behavior/position
- processing/behavior/behavioral_events
- processing/associated_files StateScript content stored in attributes

The output is diagnostic. It establishes whether the dataset can support a
canonical biological VTE-trace adapter. It does not assert explicit VTE labels.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import h5py
import numpy as np


EVENT_RE = re.compile(r"^\s*(\d+)\s+(.+?)\s*$")
ASSIGN_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+)$")
WELL_RE = re.compile(r"^(UP|DOWN)\s+(\d+)$", re.IGNORECASE)
DIO_RE = re.compile(r"^(-?\d+)\s+(-?\d+)$")


POSITION_TABLE = "Table_DANDI000115_Position_Series_Probe.csv"
BEHAVIOR_EVENTS_TABLE = "Table_DANDI000115_Behavioral_Events_Probe.csv"
STATESCRIPT_SUMMARY_TABLE = "Table_DANDI000115_StateScript_Attr_Summary.csv"
STATESCRIPT_EVENTS_TABLE = "Table_DANDI000115_StateScript_Event_Probe.csv"
EPOCHS_TABLE = "Table_DANDI000115_Epochs_Probe.csv"
ALIGNMENT_TABLE = "Table_DANDI000115_Time_Alignment_Candidates.csv"
META_JSON = "dandi000115_behavior_probe_meta.json"
REPORT_MD = "DANDI000115_Behavior_Probe_Report.md"


def _decode_value(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        if value.dtype.kind in {"S", "O", "U"}:
            return "; ".join(_decode_value(v) for v in value.ravel())
        return str(value.tolist())
    return str(value)


def _attrs(obj: Any) -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        for key, value in obj.attrs.items():
            out[str(key)] = _decode_value(value)
    except Exception:
        pass
    return out


def _float_attr(obj: Any, key: str, default: float = 1.0) -> float:
    try:
        value = obj.attrs.get(key, default)
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="replace")
        return float(value)
    except Exception:
        return default


def _finite_min(arr: np.ndarray) -> float | str:
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.min(arr)) if arr.size else ""


def _finite_max(arr: np.ndarray) -> float | str:
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.max(arr)) if arr.size else ""


def parse_task_epoch_ids(text: str) -> list[int]:
    return [int(x) for x in re.findall(r"\d+", text or "")]


def classify_statescript_payload(payload: str) -> dict[str, str]:
    payload = payload.strip()

    well_match = WELL_RE.match(payload)
    if well_match:
        return {
            "event_type": "well_transition",
            "label": well_match.group(1).upper(),
            "well": well_match.group(2),
            "variable": "",
            "value": "",
        }

    assign_match = ASSIGN_RE.match(payload)
    if assign_match:
        return {
            "event_type": "variable_assignment",
            "label": "",
            "well": "",
            "variable": assign_match.group(1),
            "value": assign_match.group(2),
        }

    dio_match = DIO_RE.match(payload)
    if dio_match:
        return {
            "event_type": "dio_state",
            "label": "",
            "well": "",
            "variable": "",
            "value": payload,
        }

    return {
        "event_type": "message",
        "label": payload,
        "well": "",
        "variable": "",
        "value": "",
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _overlap_seconds(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _extract_epochs(handle: h5py.File) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    epochs = handle.get("intervals/epochs")

    if epochs is None:
        return rows

    if "start_time" not in epochs or "stop_time" not in epochs:
        return rows

    ids = epochs["id"][()] if "id" in epochs else np.arange(len(epochs["start_time"]))
    starts = epochs["start_time"][()]
    stops = epochs["stop_time"][()]

    for idx in range(len(starts)):
        start = float(starts[idx])
        stop = float(stops[idx]) if idx < len(stops) else ""
        rows.append(
            {
                "epoch_id": int(ids[idx]) if idx < len(ids) else idx,
                "start_time": start,
                "stop_time": stop,
                "duration_s": float(stop - start) if stop != "" else "",
            }
        )

    return rows


def _extract_position_series(handle: h5py.File) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pos_base = handle.get("processing/behavior/position")

    if pos_base is None:
        return rows

    for name in sorted(pos_base.keys()):
        if not name.startswith("series_"):
            continue

        grp = pos_base[name]
        if "data" not in grp or "timestamps" not in grp:
            continue

        data_ds = grp["data"]
        ts_ds = grp["timestamps"]

        ts = np.asarray(ts_ds[()], dtype=float)
        finite_ts_mask = np.isfinite(ts)
        ts_finite = ts[finite_ts_mask]

        data = np.asarray(data_ds[()])
        if data.shape[0] == ts.shape[0]:
            data = data[finite_ts_mask]

        if data.ndim == 1:
            data = data.reshape(-1, 1)

        conversion = _float_attr(data_ds, "conversion", 1.0)
        coords = data.astype(float) * conversion

        row: dict[str, Any] = {
            "series": name,
            "description": _attrs(grp).get("description", ""),
            "n_samples_raw": int(ts.shape[0]),
            "n_samples_finite": int(ts_finite.shape[0]),
            "n_nan_timestamps": int(np.count_nonzero(~finite_ts_mask)),
            "time_min": float(ts_finite[0]) if ts_finite.size else "",
            "time_max": float(ts_finite[-1]) if ts_finite.size else "",
            "duration_s": float(ts_finite[-1] - ts_finite[0]) if ts_finite.size > 1 else "",
            "conversion": conversion,
            "unit": _attrs(data_ds).get("unit", ""),
        }

        for col_idx, label in enumerate(("x", "y", "x2", "y2")):
            if coords.shape[1] > col_idx and coords.shape[0] > 0:
                values = coords[:, col_idx]
                row[f"{label}_min"] = _finite_min(values)
                row[f"{label}_max"] = _finite_max(values)
            else:
                row[f"{label}_min"] = ""
                row[f"{label}_max"] = ""

        rows.append(row)

    return rows


def _extract_behavior_events(handle: h5py.File) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ev_base = handle.get("processing/behavior/behavioral_events")

    if ev_base is None:
        return rows

    for name in sorted(ev_base.keys()):
        grp = ev_base[name]
        if "data" not in grp or "timestamps" not in grp:
            continue

        data = np.asarray(grp["data"][()])
        ts = np.asarray(grp["timestamps"][()], dtype=float)
        finite_ts = ts[np.isfinite(ts)]

        flat = data.reshape(-1).astype(float) if data.size else np.array([], dtype=float)
        flat_finite = flat[np.isfinite(flat)]

        transitions = (
            int(np.count_nonzero(np.diff(flat_finite) != 0))
            if flat_finite.size > 1
            else 0
        )

        rows.append(
            {
                "event_channel": name,
                "n_samples": int(ts.shape[0]),
                "n_finite_timestamps": int(finite_ts.shape[0]),
                "n_nan_timestamps": int(ts.shape[0] - finite_ts.shape[0]),
                "time_min": float(finite_ts[0]) if finite_ts.size else "",
                "time_max": float(finite_ts[-1]) if finite_ts.size else "",
                "data_min": _finite_min(flat_finite),
                "data_max": _finite_max(flat_finite),
                "n_nonzero": int(np.count_nonzero(flat_finite)) if flat_finite.size else 0,
                "n_transitions": transitions,
                "description": _attrs(grp).get("description", ""),
            }
        )

    return rows


def _extract_statescripts(
    handle: h5py.File,
    output_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []

    assoc_base = handle.get("processing/associated_files")
    if assoc_base is None:
        return summary_rows, event_rows

    text_dir = output_dir / "statescript_text"
    text_dir.mkdir(parents=True, exist_ok=True)

    for name, grp in assoc_base.items():
        attrs = _attrs(grp)
        content = attrs.get("content", "")
        task_epochs = attrs.get("task_epochs", "")

        txt_path = text_dir / f"{name}.txt"
        txt_path.write_text(content, encoding="utf-8", errors="replace")

        lines = content.splitlines()
        n_code_lines = sum(1 for line in lines if line.lstrip().startswith("#"))
        n_event_like_lines = 0
        event_times_s: list[float] = []

        for line_idx, raw_line in enumerate(lines, start=1):
            stripped = raw_line.strip()
            if not stripped or stripped == "~~~" or stripped.startswith("#"):
                continue

            match = EVENT_RE.match(stripped)
            if not match:
                continue

            n_event_like_lines += 1
            time_raw = int(match.group(1))
            time_s_guess = time_raw / 1000.0
            event_times_s.append(time_s_guess)

            payload = match.group(2).strip()
            classified = classify_statescript_payload(payload)

            event_rows.append(
                {
                    "source": name,
                    "task_epochs": task_epochs,
                    "line_idx": line_idx,
                    "time_raw": time_raw,
                    "time_s_guess": time_s_guess,
                    "event_type": classified["event_type"],
                    "label": classified["label"],
                    "well": classified["well"],
                    "variable": classified["variable"],
                    "value": classified["value"],
                    "payload": payload,
                    "raw_line": raw_line,
                }
            )

        summary_rows.append(
            {
                "source": name,
                "description": attrs.get("description", ""),
                "task_epochs": task_epochs,
                "task_epoch_ids": ";".join(str(x) for x in parse_task_epoch_ids(task_epochs)),
                "content_chars": len(content),
                "n_lines": len(lines),
                "n_code_lines": n_code_lines,
                "n_event_like_lines": n_event_like_lines,
                "time_s_guess_min": min(event_times_s) if event_times_s else "",
                "time_s_guess_max": max(event_times_s) if event_times_s else "",
                "duration_s_guess": (
                    max(event_times_s) - min(event_times_s)
                    if len(event_times_s) > 1
                    else ""
                ),
                "txt_path": str(txt_path),
            }
        )

    return summary_rows, event_rows


def _compute_alignment_candidates(
    position_rows: list[dict[str, Any]],
    event_rows: list[dict[str, Any]],
    statescript_summary_rows: list[dict[str, Any]],
    epoch_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    alignment_rows: list[dict[str, Any]] = []

    position_ranges = []
    for row in position_rows:
        if row["time_min"] == "" or row["time_max"] == "":
            continue
        position_ranges.append(
            {
                "series": row["series"],
                "time_min": float(row["time_min"]),
                "time_max": float(row["time_max"]),
                "duration_s": float(row["duration_s"]) if row["duration_s"] != "" else 0.0,
            }
        )

    behavior_starts: list[float] = []
    behavior_starts.extend(float(row["time_min"]) for row in position_rows if row["time_min"] != "")
    behavior_starts.extend(float(row["time_min"]) for row in event_rows if row["time_min"] != "")
    behavior_starts.extend(float(row["start_time"]) for row in epoch_rows if row["start_time"] != "")

    global_behavior_start = min(behavior_starts) if behavior_starts else None

    for state_row in statescript_summary_rows:
        if state_row["time_s_guess_min"] == "" or state_row["time_s_guess_max"] == "":
            continue

        ss_min = float(state_row["time_s_guess_min"])
        ss_max = float(state_row["time_s_guess_max"])
        ss_duration = max(0.0, ss_max - ss_min)
        task_epoch_ids = parse_task_epoch_ids(str(state_row.get("task_epochs", "")))

        candidates: list[dict[str, Any]] = []

        if global_behavior_start is not None:
            candidates.append(
                {
                    "candidate": "global_behavior_start_plus_statescript_seconds",
                    "offset_s": float(global_behavior_start),
                }
            )

        for pos_range in position_ranges:
            candidates.append(
                {
                    "candidate": f"align_statescript_min_to_{pos_range['series']}_start",
                    "offset_s": pos_range["time_min"] - ss_min,
                }
            )

        for epoch_row in epoch_rows:
            epoch_id = int(epoch_row["epoch_id"])
            if not task_epoch_ids or epoch_id in task_epoch_ids:
                candidates.append(
                    {
                        "candidate": f"align_statescript_min_to_epoch_{epoch_id}_start",
                        "offset_s": float(epoch_row["start_time"]) - ss_min,
                    }
                )

        for candidate in candidates:
            mapped_min = ss_min + candidate["offset_s"]
            mapped_max = ss_max + candidate["offset_s"]

            for pos_range in position_ranges:
                overlap_s = _overlap_seconds(
                    mapped_min,
                    mapped_max,
                    pos_range["time_min"],
                    pos_range["time_max"],
                )

                alignment_rows.append(
                    {
                        "statescript_source": state_row["source"],
                        "task_epochs": state_row["task_epochs"],
                        "candidate": candidate["candidate"],
                        "position_series": pos_range["series"],
                        "offset_s": candidate["offset_s"],
                        "mapped_time_min": mapped_min,
                        "mapped_time_max": mapped_max,
                        "statescript_duration_s": ss_duration,
                        "position_time_min": pos_range["time_min"],
                        "position_time_max": pos_range["time_max"],
                        "position_duration_s": pos_range["duration_s"],
                        "overlap_s": overlap_s,
                        "overlap_fraction_of_statescript": (
                            overlap_s / ss_duration if ss_duration > 0 else ""
                        ),
                    }
                )

    alignment_rows.sort(
        key=lambda row: float(row["overlap_s"]) if row["overlap_s"] != "" else -1.0,
        reverse=True,
    )

    return alignment_rows


def _best_alignment(alignment_rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    return alignment_rows[0] if alignment_rows else None


def _build_report(meta: dict[str, Any], best: dict[str, Any] | None) -> str:
    lines = [
        "# DANDI 000115 Behavior Probe Report",
        "",
        "## Scope",
        "",
        "This report summarizes behavior-relevant NWB layers for DANDI 000115.",
        "It is a probe for future canonical trace export, not a direct VTE-label validation.",
        "",
        "## Input",
        "",
        f"- NWB: `{meta['nwb_path']}`",
        f"- Output directory: `{meta['output_dir']}`",
        "",
        "## Detected layers",
        "",
        f"- Position series: {meta['n_position_series']}",
        f"- Behavioral event channels: {meta['n_behavior_event_channels']}",
        f"- StateScript sources: {meta['n_statescript_sources']}",
        f"- StateScript event rows: {meta['n_statescript_event_rows']}",
        f"- Epoch rows: {meta['n_epochs']}",
        "",
    ]

    if best:
        lines.extend(
            [
                "## Best alignment candidate",
                "",
                f"- StateScript source: `{best['statescript_source']}`",
                f"- Candidate: `{best['candidate']}`",
                f"- Position series: `{best['position_series']}`",
                f"- Overlap fraction: {float(best['overlap_fraction_of_statescript']):.6f}",
                f"- Offset seconds: {float(best['offset_s']):.6f}",
                "",
            ]
        )

    lines.extend(
        [
            "## Interpretation",
            "",
            "DANDI 000115 can support proxy-level biological comparability if position, well/reward events, and StateScript timing align.",
            "Explicit VTE/head-scanning labels are not assumed by this probe.",
            "",
            "## Output tables",
            "",
            f"- `{POSITION_TABLE}`",
            f"- `{BEHAVIOR_EVENTS_TABLE}`",
            f"- `{STATESCRIPT_SUMMARY_TABLE}`",
            f"- `{STATESCRIPT_EVENTS_TABLE}`",
            f"- `{EPOCHS_TABLE}`",
            f"- `{ALIGNMENT_TABLE}`",
            "",
        ]
    )

    return "\n".join(lines)


def probe_dandi000115_behavior(nwb_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    nwb = Path(nwb_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    if not nwb.exists():
        raise FileNotFoundError(f"NWB file not found: {nwb}")

    with h5py.File(nwb, "r") as handle:
        epoch_rows = _extract_epochs(handle)
        position_rows = _extract_position_series(handle)
        behavior_event_rows = _extract_behavior_events(handle)
        statescript_summary_rows, statescript_event_rows = _extract_statescripts(handle, output)

    alignment_rows = _compute_alignment_candidates(
        position_rows=position_rows,
        event_rows=behavior_event_rows,
        statescript_summary_rows=statescript_summary_rows,
        epoch_rows=epoch_rows,
    )
    best = _best_alignment(alignment_rows)

    _write_csv(
        output / POSITION_TABLE,
        position_rows,
        [
            "series",
            "description",
            "n_samples_raw",
            "n_samples_finite",
            "n_nan_timestamps",
            "time_min",
            "time_max",
            "duration_s",
            "conversion",
            "unit",
            "x_min",
            "x_max",
            "y_min",
            "y_max",
            "x2_min",
            "x2_max",
            "y2_min",
            "y2_max",
        ],
    )

    _write_csv(
        output / BEHAVIOR_EVENTS_TABLE,
        behavior_event_rows,
        [
            "event_channel",
            "n_samples",
            "n_finite_timestamps",
            "n_nan_timestamps",
            "time_min",
            "time_max",
            "data_min",
            "data_max",
            "n_nonzero",
            "n_transitions",
            "description",
        ],
    )

    _write_csv(
        output / STATESCRIPT_SUMMARY_TABLE,
        statescript_summary_rows,
        [
            "source",
            "description",
            "task_epochs",
            "task_epoch_ids",
            "content_chars",
            "n_lines",
            "n_code_lines",
            "n_event_like_lines",
            "time_s_guess_min",
            "time_s_guess_max",
            "duration_s_guess",
            "txt_path",
        ],
    )

    _write_csv(
        output / STATESCRIPT_EVENTS_TABLE,
        statescript_event_rows,
        [
            "source",
            "task_epochs",
            "line_idx",
            "time_raw",
            "time_s_guess",
            "event_type",
            "label",
            "well",
            "variable",
            "value",
            "payload",
            "raw_line",
        ],
    )

    _write_csv(
        output / EPOCHS_TABLE,
        epoch_rows,
        ["epoch_id", "start_time", "stop_time", "duration_s"],
    )

    _write_csv(
        output / ALIGNMENT_TABLE,
        alignment_rows,
        [
            "statescript_source",
            "task_epochs",
            "candidate",
            "position_series",
            "offset_s",
            "mapped_time_min",
            "mapped_time_max",
            "statescript_duration_s",
            "position_time_min",
            "position_time_max",
            "position_duration_s",
            "overlap_s",
            "overlap_fraction_of_statescript",
        ],
    )

    max_overlap_fraction = (
        float(best["overlap_fraction_of_statescript"])
        if best and best["overlap_fraction_of_statescript"] != ""
        else None
    )

    meta: dict[str, Any] = {
        "nwb_path": str(nwb),
        "output_dir": str(output),
        "n_position_series": len(position_rows),
        "n_behavior_event_channels": len(behavior_event_rows),
        "n_statescript_sources": len(statescript_summary_rows),
        "n_statescript_event_rows": len(statescript_event_rows),
        "n_epochs": len(epoch_rows),
        "n_alignment_rows": len(alignment_rows),
        "max_alignment_overlap_fraction": max_overlap_fraction,
        "best_alignment": best,
        "tables": [
            POSITION_TABLE,
            BEHAVIOR_EVENTS_TABLE,
            STATESCRIPT_SUMMARY_TABLE,
            STATESCRIPT_EVENTS_TABLE,
            EPOCHS_TABLE,
            ALIGNMENT_TABLE,
        ],
        "report": REPORT_MD,
        "interpretation": (
            "Behavioral proxy dataset candidate. Explicit VTE/head-scanning labels are not assumed."
        ),
    }

    (output / META_JSON).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (output / REPORT_MD).write_text(_build_report(meta, best), encoding="utf-8")

    print(f"Position probe saved: {output / POSITION_TABLE}")
    print(f"Behavioral events probe saved: {output / BEHAVIOR_EVENTS_TABLE}")
    print(f"StateScript summary saved: {output / STATESCRIPT_SUMMARY_TABLE}")
    print(f"StateScript events saved: {output / STATESCRIPT_EVENTS_TABLE}")
    print(f"Epochs probe saved: {output / EPOCHS_TABLE}")
    print(f"Alignment candidates saved: {output / ALIGNMENT_TABLE}")
    print(f"Metadata saved: {output / META_JSON}")
    print(f"Report saved: {output / REPORT_MD}")

    return meta


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Probe DANDI 000115 NWB behavioral layers."
    )
    parser.add_argument("--nwb", required=True, help="Local NWB file path.")
    parser.add_argument("--output-dir", required=True, help="Output directory.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    probe_dandi000115_behavior(args.nwb, args.output_dir)


if __name__ == "__main__":
    main()
