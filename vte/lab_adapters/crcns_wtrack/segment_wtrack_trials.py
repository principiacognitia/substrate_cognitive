"""Segment CRCNS W-track canonical trace into choice-zone visit trials.

Patch 14D scope:
- consume raw epoch-level canonical trace;
- consume inferred or manual W-track geometry JSON;
- mark choice-point samples and route zones;
- split trace into choice-zone visit windows suitable for the VTE wrapper.

This is still a heuristic segmentation layer. Final biological comparability
requires frozen zone definitions and a predeclared comparison protocol.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SEGMENT_SUMMARY_COLUMNS = [
    "dataset_id",
    "animal_id",
    "run_id",
    "day",
    "epoch",
    "trial",
    "choice_visit_index",
    "choice_point_id",
    "committed_path",
    "n_samples",
    "n_choice_samples",
    "start_time_s",
    "end_time_s",
    "choice_start_time_s",
    "choice_end_time_s",
    "source_start_tick",
    "source_end_tick",
    "pre_samples",
    "post_samples",
]


def _load_geometry(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _finite_xy(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["x"] = pd.to_numeric(out["x"], errors="coerce")
    out["y"] = pd.to_numeric(out["y"], errors="coerce")
    return out[np.isfinite(out["x"]) & np.isfinite(out["y"])].copy()


def _filter_run_epochs(df: pd.DataFrame, *, run_epochs_only: bool) -> pd.DataFrame:
    if not run_epochs_only or "task_type" not in df.columns:
        return df

    task_type = df["task_type"].fillna("").astype(str).str.lower()
    run_df = df[task_type == "run"].copy()
    return run_df if len(run_df) > 0 else df


def _as_float_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series([np.nan] * len(df), index=df.index)
    return pd.to_numeric(df[col], errors="coerce")


def _distance_to(x: np.ndarray, y: np.ndarray, zone: dict[str, Any]) -> np.ndarray:
    return np.sqrt((x - float(zone["x"])) ** 2 + (y - float(zone["y"])) ** 2)


def _assign_route_ids(
    x: np.ndarray,
    y: np.ndarray,
    *,
    geometry: dict[str, Any],
    at_choice: np.ndarray,
) -> list[str]:
    route_zones = geometry.get("route_zones", {})
    if not route_zones:
        return ["wtrack_junction" if bool(v) else "track" for v in at_choice]

    zone_ids = list(route_zones.keys())
    centers = np.asarray(
        [[route_zones[z]["x"], route_zones[z]["y"]] for z in zone_ids],
        dtype=float,
    )

    points = np.column_stack([x, y])
    distances = np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=2)
    nearest = np.argmin(distances, axis=1)

    route_ids: list[str] = []
    for i, is_choice in enumerate(at_choice):
        if bool(is_choice):
            route_ids.append("wtrack_junction")
        else:
            route_ids.append(zone_ids[int(nearest[i])])

    return route_ids


def _true_runs(mask: np.ndarray, *, merge_gap_samples: int) -> list[tuple[int, int]]:
    """Return inclusive true-run intervals, optionally merging short false gaps."""

    runs: list[tuple[int, int]] = []
    start: int | None = None

    for idx, value in enumerate(mask):
        if bool(value) and start is None:
            start = idx
        elif not bool(value) and start is not None:
            runs.append((start, idx - 1))
            start = None

    if start is not None:
        runs.append((start, len(mask) - 1))

    if not runs or merge_gap_samples <= 0:
        return runs

    merged: list[tuple[int, int]] = [runs[0]]
    for start, end in runs[1:]:
        prev_start, prev_end = merged[-1]
        gap = start - prev_end - 1
        if gap <= merge_gap_samples:
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))

    return merged


def _first_non_choice_route_after(
    route_ids: list[str],
    at_choice: np.ndarray,
    *,
    start: int,
    end: int,
) -> str:
    for i in range(start, end + 1):
        route_id = route_ids[i]
        if not bool(at_choice[i]) and route_id not in {"", "track", "wtrack_junction"}:
            return route_id
    return ""


def _safe_value(row: pd.Series, col: str, default: Any = "") -> Any:
    if col not in row.index:
        return default
    value = row[col]
    if pd.isna(value):
        return default
    return value


def _segment_group(
    group: pd.DataFrame,
    *,
    geometry: dict[str, Any],
    trial_offset: int,
    pre_samples: int,
    post_samples: int,
    min_choice_samples: int,
    merge_gap_samples: int,
) -> tuple[list[pd.DataFrame], list[dict[str, Any]], int]:
    group = group.copy().reset_index(drop=True)

    x = pd.to_numeric(group["x"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(group["y"], errors="coerce").to_numpy(dtype=float)

    cp_id = "wtrack_junction"
    cp = geometry["choice_points"][cp_id]
    at_choice = _distance_to(x, y, cp) <= float(cp["radius"])
    route_ids = _assign_route_ids(x, y, geometry=geometry, at_choice=at_choice)

    runs = [
        (s, e)
        for s, e in _true_runs(at_choice, merge_gap_samples=merge_gap_samples)
        if int(np.sum(at_choice[s:e + 1])) >= min_choice_samples
    ]

    segments: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    trial = trial_offset

    time_s = _as_float_series(group, "time_s")

    for visit_idx, (choice_start, choice_end) in enumerate(runs, start=1):
        start = max(0, choice_start - pre_samples)
        end = min(len(group) - 1, choice_end + post_samples)

        seg = group.iloc[start:end + 1].copy().reset_index(drop=True)
        local_n = len(seg)

        local_choice = at_choice[start:end + 1]
        local_route_ids = route_ids[start:end + 1]

        trial += 1
        committed_path = _first_non_choice_route_after(
            route_ids,
            at_choice,
            start=choice_end + 1,
            end=end,
        )

        if "source_trial" not in seg.columns and "trial" in seg.columns:
            seg["source_trial"] = seg["trial"]
        if "source_tick" not in seg.columns and "tick" in seg.columns:
            seg["source_tick"] = seg["tick"]

        seg["trial"] = trial
        seg["tick"] = np.arange(1, local_n + 1, dtype=int)
        seg["choice_visit_index"] = int(visit_idx)
        seg["choice_point_id"] = cp_id
        seg["at_choice_point"] = local_choice.astype(bool)
        seg["route_id"] = local_route_ids
        seg["committed_path"] = committed_path
        seg["event_type"] = "choice_zone_visit"
        seg["event_trial"] = trial
        seg["trial_phase"] = "post_choice"
        seg.loc[np.arange(local_n) < (choice_start - start), "trial_phase"] = "pre_choice"
        seg.loc[local_choice.astype(bool), "trial_phase"] = "choice"
        seg["done"] = False
        seg.loc[local_n - 1, "done"] = True

        first_row = seg.iloc[0]
        last_row = seg.iloc[-1]

        choice_time = time_s.iloc[choice_start:choice_end + 1]
        segment_time = time_s.iloc[start:end + 1]

        summaries.append(
            {
                "dataset_id": _safe_value(first_row, "dataset_id"),
                "animal_id": _safe_value(first_row, "animal_id"),
                "run_id": _safe_value(first_row, "run_id"),
                "day": _safe_value(first_row, "day"),
                "epoch": _safe_value(first_row, "epoch"),
                "trial": int(trial),
                "choice_visit_index": int(visit_idx),
                "choice_point_id": cp_id,
                "committed_path": committed_path,
                "n_samples": int(local_n),
                "n_choice_samples": int(np.sum(local_choice)),
                "start_time_s": float(segment_time.iloc[0]) if np.isfinite(segment_time.iloc[0]) else "",
                "end_time_s": float(segment_time.iloc[-1]) if np.isfinite(segment_time.iloc[-1]) else "",
                "choice_start_time_s": float(choice_time.iloc[0]) if np.isfinite(choice_time.iloc[0]) else "",
                "choice_end_time_s": float(choice_time.iloc[-1]) if np.isfinite(choice_time.iloc[-1]) else "",
                "source_start_tick": _safe_value(first_row, "source_tick", _safe_value(first_row, "tick")),
                "source_end_tick": _safe_value(last_row, "source_tick", _safe_value(last_row, "tick")),
                "pre_samples": int(choice_start - start),
                "post_samples": int(end - choice_end),
            }
        )

        segments.append(seg)

    return segments, summaries, trial


def segment_wtrack_trials(
    input_trace: str | Path,
    geometry_json: str | Path,
    output_dir: str | Path,
    *,
    run_epochs_only: bool = True,
    pre_samples: int = 15,
    post_samples: int = 45,
    min_choice_samples: int = 3,
    merge_gap_samples: int = 2,
) -> dict[str, Any]:
    trace_path = Path(input_trace)
    geom_path = Path(geometry_json)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    geometry = _load_geometry(geom_path)
    raw = pd.read_csv(trace_path, low_memory=False)
    df = _finite_xy(raw)
    df = _filter_run_epochs(df, run_epochs_only=run_epochs_only)

    if len(df) == 0:
        raise ValueError("No finite x/y samples available for segmentation.")

    if "source_row_index" not in df.columns:
        df["source_row_index"] = df.index

    sort_cols = [c for c in ["run_id", "day", "epoch", "time_s", "tick"] if c in df.columns]
    if sort_cols:
        for col in ["time_s", "tick"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.sort_values(sort_cols).reset_index(drop=True)

    group_cols = [c for c in ["run_id", "day", "epoch"] if c in df.columns]
    if not group_cols:
        group_iter = [(("all",), df)]
    else:
        group_iter = list(df.groupby(group_cols, dropna=False, sort=True))

    all_segments: list[pd.DataFrame] = []
    all_summaries: list[dict[str, Any]] = []
    trial_counter = 0

    for _, group in group_iter:
        segments, summaries, trial_counter = _segment_group(
            group,
            geometry=geometry,
            trial_offset=trial_counter,
            pre_samples=pre_samples,
            post_samples=post_samples,
            min_choice_samples=min_choice_samples,
            merge_gap_samples=merge_gap_samples,
        )
        all_segments.extend(segments)
        all_summaries.extend(summaries)

    if all_segments:
        segmented = pd.concat(all_segments, ignore_index=True)
    else:
        segmented = pd.DataFrame(columns=list(raw.columns) + ["source_trial", "source_tick", "choice_visit_index"])

    summary = pd.DataFrame(all_summaries, columns=SEGMENT_SUMMARY_COLUMNS)

    segmented_csv = out_dir / "crcns_wtrack_segmented_trace.csv"
    summary_csv = out_dir / "Table_CRCNS_WTrack_Segmented_Trials.csv"
    meta_json = out_dir / "crcns_wtrack_segmentation_meta.json"

    segmented.to_csv(segmented_csv, index=False)
    summary.to_csv(summary_csv, index=False)

    meta = {
        "script": "vte.lab_adapters.crcns_wtrack.segment_wtrack_trials",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_trace": str(trace_path),
        "geometry_json": str(geom_path),
        "output_dir": str(out_dir),
        "segmented_trace_csv": str(segmented_csv),
        "trial_summary_csv": str(summary_csv),
        "geometry_id": geometry.get("geometry_id", ""),
        "run_epochs_only": bool(run_epochs_only),
        "pre_samples": int(pre_samples),
        "post_samples": int(post_samples),
        "min_choice_samples": int(min_choice_samples),
        "merge_gap_samples": int(merge_gap_samples),
        "n_input_rows": int(len(raw)),
        "n_rows_after_filter": int(len(df)),
        "n_segmented_rows": int(len(segmented)),
        "n_choice_trials": int(len(summary)),
        "heuristic": True,
    }

    meta_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Segmented trace saved: {segmented_csv}")
    print(f"Trial summary saved: {summary_csv}")
    print(f"Metadata saved: {meta_json}")

    return meta


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Segment CRCNS W-track canonical trace by choice-zone visits.")
    parser.add_argument("--input-trace", required=True, type=Path)
    parser.add_argument("--geometry-json", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--all-epochs", action="store_true", help="Use all epochs, not only task_type == run.")
    parser.add_argument("--pre-samples", type=int, default=15)
    parser.add_argument("--post-samples", type=int, default=45)
    parser.add_argument("--min-choice-samples", type=int, default=3)
    parser.add_argument("--merge-gap-samples", type=int, default=2)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    segment_wtrack_trials(
        input_trace=args.input_trace,
        geometry_json=args.geometry_json,
        output_dir=args.output_dir,
        run_epochs_only=not args.all_epochs,
        pre_samples=args.pre_samples,
        post_samples=args.post_samples,
        min_choice_samples=args.min_choice_samples,
        merge_gap_samples=args.merge_gap_samples,
    )


if __name__ == "__main__":
    main()