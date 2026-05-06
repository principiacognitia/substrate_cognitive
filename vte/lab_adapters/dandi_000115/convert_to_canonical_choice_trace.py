"""Export an event-centered canonical choice trace from DANDI 000115 NWB.

Patch 15B scope:
- use Patch 15A probe output to select the best StateScript/position alignment;
- extract one behavior position series from NWB;
- derive heading from xloc/yloc/xloc2/yloc2;
- select beam-on behavioral events as event-centered choice episodes;
- assign committed_path from the event channel;
- assign reward from matching pump events in a post-event window;
- write canonical trace readable by the frozen VTE wrapper.

This is a proxy biological choice-point trace. It does not claim explicit
VTE/head-scanning labels and does not reconstruct the full eight-arm task.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd


TRACE_CSV = "dandi000115_canonical_choice_trace.csv"
CHOICE_EVENTS_CSV = "Table_DANDI000115_Choice_Events.csv"
META_JSON = "dandi000115_canonical_choice_trace_meta.json"
REPORT_MD = "DANDI000115_Canonical_Choice_Trace_Report.md"

ALIGNMENT_TABLE = "Table_DANDI000115_Time_Alignment_Candidates.csv"


def _attrs(obj: Any) -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        for key, value in obj.attrs.items():
            out[str(key)] = str(value)
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


def _path_stem(path: str) -> str:
    # sub-despereaux/sub-despereaux_ses-despereaux-07_behavior+ecephys.nwb
    stem = Path(path).name
    if stem.endswith(".nwb"):
        stem = stem[:-4]
    return stem.replace("_behavior+ecephys", "")


def _infer_subject_session(nwb_path: Path, dataset_id: str) -> dict[str, str]:
    parts = nwb_path.parts
    subject_id = ""
    session_id = ""

    for part in parts:
        if part.startswith("sub-"):
            subject_id = part.replace("sub-", "")
            break

    stem = nwb_path.name
    if "_ses-" in stem:
        session_id = stem.split("_ses-", 1)[1].replace("_behavior+ecephys.nwb", "")
    else:
        session_id = _path_stem(str(nwb_path))

    if not subject_id:
        if session_id and "-" in session_id:
            subject_id = session_id.split("-", 1)[0]
        else:
            subject_id = "unknown_subject"

    run_id = f"{dataset_id}_{session_id}".replace("-", "_")

    return {
        "subject_id": subject_id,
        "animal_id": subject_id,
        "session_id": session_id,
        "run_id": run_id,
    }


def _read_best_alignment(probe_dir: Path) -> dict[str, Any] | None:
    path = probe_dir / ALIGNMENT_TABLE
    if not path.exists():
        return None

    df = pd.read_csv(path)
    if df.empty:
        return None

    df["overlap_s_num"] = pd.to_numeric(df["overlap_s"], errors="coerce")
    df = df.sort_values("overlap_s_num", ascending=False)

    row = df.iloc[0].to_dict()
    return row


def _load_position_series(nwb_path: Path, series_name: str) -> pd.DataFrame:
    with h5py.File(nwb_path, "r") as handle:
        series_path = f"processing/behavior/position/{series_name}"
        if series_path not in handle:
            raise KeyError(f"Position series not found: {series_path}")

        grp = handle[series_path]
        data_ds = grp["data"]
        ts_ds = grp["timestamps"]

        conversion = _float_attr(data_ds, "conversion", 1.0)
        data = np.asarray(data_ds[()])
        ts = np.asarray(ts_ds[()], dtype=float)

    if data.ndim == 1:
        data = data.reshape(-1, 1)

    valid = np.isfinite(ts)
    data = data[valid]
    ts = ts[valid]

    coords = data.astype(float) * conversion

    def col_or_nan(index: int) -> np.ndarray:
        if coords.shape[1] > index:
            return coords[:, index]
        return np.full(coords.shape[0], np.nan, dtype=float)

    x = col_or_nan(0)
    y = col_or_nan(1)
    x2 = col_or_nan(2)
    y2 = col_or_nan(3)

    heading = np.arctan2(y2 - y, x2 - x)

    bad_heading = ~np.isfinite(heading)
    if np.any(bad_heading):
        dx = np.gradient(x)
        dy = np.gradient(y)
        velocity_heading = np.arctan2(dy, dx)
        heading[bad_heading] = velocity_heading[bad_heading]

    df = pd.DataFrame(
        {
            "source_row_index": np.arange(len(ts), dtype=int),
            "sample_index": np.arange(len(ts), dtype=int),
            "time_s": ts,
            "x": x,
            "y": y,
            "x2": x2,
            "y2": y2,
            "heading": heading,
        }
    )

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["time_s", "x", "y", "heading"]).reset_index(drop=True)
    df["sample_index"] = np.arange(len(df), dtype=int)

    return df


def _event_prefix(channel: str) -> str:
    for suffix in ("beam", "pump", "light"):
        if channel.endswith(suffix):
            return channel[: -len(suffix)]
    return channel


def _load_behavior_events(nwb_path: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    with h5py.File(nwb_path, "r") as handle:
        base = handle.get("processing/behavior/behavioral_events")
        if base is None:
            return pd.DataFrame()

        for channel in sorted(base.keys()):
            grp = base[channel]
            if "data" not in grp or "timestamps" not in grp:
                continue

            data = np.asarray(grp["data"][()]).reshape(-1)
            ts = np.asarray(grp["timestamps"][()], dtype=float).reshape(-1)

            n = min(len(data), len(ts))
            data = data[:n]
            ts = ts[:n]

            valid = np.isfinite(ts) & np.isfinite(data.astype(float))
            data = data[valid].astype(float)
            ts = ts[valid]

            if n == 0:
                continue

            for idx, (timestamp, value) in enumerate(zip(ts, data)):
                previous = data[idx - 1] if idx > 0 else 0.0
                is_on = bool(value > 0)
                is_rising = bool(value > 0 and previous <= 0)

                rows.append(
                    {
                        "event_channel": channel,
                        "event_family": _event_prefix(channel),
                        "event_kind": (
                            "beam"
                            if channel.endswith("beam")
                            else "pump"
                            if channel.endswith("pump")
                            else "light"
                            if channel.endswith("light")
                            else "other"
                        ),
                        "event_index": idx,
                        "time_s": float(timestamp),
                        "value": float(value),
                        "is_on": is_on,
                        "is_rising": is_rising,
                    }
                )

    return pd.DataFrame(rows)


def _select_choice_events(
    events: pd.DataFrame,
    position_time_min: float,
    position_time_max: float,
    include_event_regex: str,
    max_events: int | None,
) -> pd.DataFrame:
    if events.empty:
        return events

    choices = events.copy()
    choices = choices[
        (choices["event_kind"] == "beam")
        & (choices["is_rising"] == True)
        & (choices["time_s"] >= position_time_min)
        & (choices["time_s"] <= position_time_max)
    ].copy()

    if include_event_regex:
        choices = choices[choices["event_channel"].str.match(include_event_regex)].copy()

    choices = choices.sort_values("time_s").reset_index(drop=True)
    choices["trial"] = np.arange(1, len(choices) + 1, dtype=int)
    choices["committed_path"] = choices["event_family"].astype(str)

    if max_events is not None and max_events > 0:
        choices = choices.head(max_events).copy()
        choices["trial"] = np.arange(1, len(choices) + 1, dtype=int)

    return choices


def _assign_reward_to_events(
    choice_events: pd.DataFrame,
    events: pd.DataFrame,
    reward_window_after_s: float,
) -> pd.DataFrame:
    if choice_events.empty:
        return choice_events

    pumps = events[(events["event_kind"] == "pump") & (events["is_rising"] == True)].copy()

    out = choice_events.copy()
    reward_values: list[float] = []
    reward_channels: list[str] = []
    reward_times: list[str] = []

    for _, row in out.iterrows():
        committed_path = str(row["committed_path"])
        t0 = float(row["time_s"])
        t1 = t0 + float(reward_window_after_s)

        matching = pumps[
            (pumps["event_family"].astype(str) == committed_path)
            & (pumps["time_s"] >= t0)
            & (pumps["time_s"] <= t1)
        ]

        if matching.empty:
            reward_values.append(0.0)
            reward_channels.append("")
            reward_times.append("")
        else:
            first = matching.iloc[0]
            reward_values.append(1.0)
            reward_channels.append(str(first["event_channel"]))
            reward_times.append(str(float(first["time_s"])))

    out["reward"] = reward_values
    out["reward_channel"] = reward_channels
    out["reward_time_s"] = reward_times

    return out


def _nearest_statescript_event(
    statescript_events: pd.DataFrame,
    event_time_s: float,
    match_window_s: float,
) -> dict[str, Any]:
    if statescript_events.empty:
        return {
            "nearest_statescript_time_s": "",
            "nearest_statescript_dt_s": "",
            "nearest_statescript_event_type": "",
            "nearest_statescript_payload": "",
        }

    dt = (statescript_events["mapped_time_s"] - event_time_s).abs()
    idx = dt.idxmin()
    nearest = statescript_events.loc[idx]
    nearest_dt = float(nearest["mapped_time_s"] - event_time_s)

    if abs(nearest_dt) > match_window_s:
        return {
            "nearest_statescript_time_s": "",
            "nearest_statescript_dt_s": "",
            "nearest_statescript_event_type": "",
            "nearest_statescript_payload": "",
        }

    return {
        "nearest_statescript_time_s": float(nearest["mapped_time_s"]),
        "nearest_statescript_dt_s": nearest_dt,
        "nearest_statescript_event_type": str(nearest.get("event_type", "")),
        "nearest_statescript_payload": str(nearest.get("payload", "")),
    }


def _load_statescript_events(
    probe_dir: Path,
    statescript_source: str,
    offset_s: float,
) -> pd.DataFrame:
    path = probe_dir / "Table_DANDI000115_StateScript_Event_Probe.csv"
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)
    if df.empty:
        return df

    if statescript_source:
        df = df[df["source"].astype(str) == statescript_source].copy()

    if df.empty:
        return df

    df["time_s_guess"] = pd.to_numeric(df["time_s_guess"], errors="coerce")
    df = df.dropna(subset=["time_s_guess"]).copy()
    df["mapped_time_s"] = df["time_s_guess"] + float(offset_s)

    return df


def _event_window_trace(
    position: pd.DataFrame,
    choice_events: pd.DataFrame,
    statescript_events: pd.DataFrame,
    identifiers: dict[str, str],
    dataset_id: str,
    geometry_id: str,
    position_series: str,
    statescript_source: str,
    alignment_offset_s: float,
    pre_event_s: float,
    post_event_s: float,
    statescript_match_window_s: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    trace_rows: list[dict[str, Any]] = []
    choice_rows: list[dict[str, Any]] = []

    for _, event in choice_events.iterrows():
        trial = int(event["trial"])
        event_time = float(event["time_s"])
        committed_path = str(event["committed_path"])
        reward = float(event.get("reward", 0.0))

        start = event_time - float(pre_event_s)
        stop = event_time + float(post_event_s)

        window = position[(position["time_s"] >= start) & (position["time_s"] <= stop)].copy()
        if window.empty:
            continue

        window = window.reset_index(drop=True)
        nearest = _nearest_statescript_event(
            statescript_events=statescript_events,
            event_time_s=event_time,
            match_window_s=statescript_match_window_s,
        )

        for idx, sample in window.iterrows():
            done = bool(idx == len(window) - 1)
            row_reward = reward if done else 0.0

            trace_rows.append(
                {
                    "run_id": identifiers["run_id"],
                    "seed": identifiers["subject_id"],
                    "trial": trial,
                    "tick": int(idx + 1),
                    "x": float(sample["x"]),
                    "y": float(sample["y"]),
                    "heading": float(sample["heading"]),
                    "choice_point_id": "dandi000115_event_centered_choice",
                    "at_choice_point": True,
                    "action": "track",
                    "committed_path": committed_path,
                    "reward": row_reward,
                    "done": done,
                    "protocol": "dandi000115",
                    "condition": identifiers["session_id"],
                    "ablation": "",
                    "trial_phase": "event_centered_choice",
                    "pose_source": "lab_tracking",
                    "event_type": "beam_on_choice_event",
                    "event_trial": trial,
                    "target_path": "",
                    "dataset_id": dataset_id,
                    "trace_origin": "biological",
                    "subject_id": identifiers["subject_id"],
                    "animal_id": identifiers["animal_id"],
                    "session_id": identifiers["session_id"],
                    "sample_index": int(sample["sample_index"]),
                    "time_s": float(sample["time_s"]),
                    "time_from_event_s": float(sample["time_s"] - event_time),
                    "event_time_s": event_time,
                    "geometry_id": geometry_id,
                    "route_id": committed_path,
                    "coordinate_system": "nwb_position_meters",
                    "source_file": "",
                    "source_position_series": position_series,
                    "source_statescript": statescript_source,
                    "alignment_offset_s": alignment_offset_s,
                    "source_row_index": int(sample["source_row_index"]),
                    "x2": float(sample["x2"]) if np.isfinite(sample["x2"]) else "",
                    "y2": float(sample["y2"]) if np.isfinite(sample["y2"]) else "",
                    "choice_event_channel": str(event["event_channel"]),
                    "choice_event_time_s": event_time,
                    "reward_channel": str(event.get("reward_channel", "")),
                    "reward_time_s": str(event.get("reward_time_s", "")),
                    "nearest_statescript_time_s": nearest["nearest_statescript_time_s"],
                    "nearest_statescript_dt_s": nearest["nearest_statescript_dt_s"],
                    "nearest_statescript_event_type": nearest["nearest_statescript_event_type"],
                    "nearest_statescript_payload": nearest["nearest_statescript_payload"],
                }
            )

        choice_rows.append(
            {
                "dataset_id": dataset_id,
                "trace_origin": "biological",
                "run_id": identifiers["run_id"],
                "subject_id": identifiers["subject_id"],
                "animal_id": identifiers["animal_id"],
                "session_id": identifiers["session_id"],
                "trial": trial,
                "event_channel": str(event["event_channel"]),
                "committed_path": committed_path,
                "event_time_s": event_time,
                "window_start_s": float(window["time_s"].min()),
                "window_stop_s": float(window["time_s"].max()),
                "n_trace_samples": int(len(window)),
                "reward": reward,
                "reward_channel": str(event.get("reward_channel", "")),
                "reward_time_s": str(event.get("reward_time_s", "")),
                **nearest,
            }
        )

    return pd.DataFrame(trace_rows), pd.DataFrame(choice_rows)


def _write_report(meta: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# DANDI 000115 Canonical Choice Trace Report",
            "",
            "## Scope",
            "",
            "This export creates an event-centered biological canonical trace for the frozen VTE wrapper.",
            "It does not reconstruct the full eight-arm task and does not assume explicit VTE labels.",
            "",
            "## Input",
            "",
            f"- NWB: `{meta['nwb_path']}`",
            f"- Probe directory: `{meta['probe_dir']}`",
            "",
            "## Selected sources",
            "",
            f"- Position series: `{meta['position_series']}`",
            f"- StateScript source: `{meta['statescript_source']}`",
            f"- Alignment offset: {meta['alignment_offset_s']}",
            "",
            "## Output",
            "",
            f"- Trace rows: {meta['n_trace_rows']}",
            f"- Choice events: {meta['n_choice_events']}",
            f"- Trace CSV: `{TRACE_CSV}`",
            f"- Choice events CSV: `{CHOICE_EVENTS_CSV}`",
            "",
            "## Interpretation",
            "",
            "The output supports proxy-level biological comparison of choice-episode trajectory metrics.",
            "The appropriate next step is to run the existing VTE wrapper on the exported trace.",
            "",
        ]
    )


def convert_dandi000115_to_canonical_choice_trace(
    nwb_path: str | Path,
    probe_dir: str | Path,
    output_dir: str | Path,
    dataset_id: str = "dandi_000115",
    geometry_id: str = "dandi000115_event_centered_eight_arm",
    position_series: str | None = None,
    statescript_source: str | None = None,
    alignment_offset_s: float | None = None,
    include_event_regex: str = r"^(arm[1-8]|R|W|home)beam$",
    pre_event_s: float = 2.0,
    post_event_s: float = 4.0,
    reward_window_after_s: float = 2.0,
    statescript_match_window_s: float = 0.1,
    max_events: int | None = None,
) -> dict[str, Any]:
    nwb = Path(nwb_path)
    probe = Path(probe_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    if not nwb.exists():
        raise FileNotFoundError(f"NWB file not found: {nwb}")
    if not probe.exists():
        raise FileNotFoundError(f"Probe directory not found: {probe}")

    best_alignment = _read_best_alignment(probe)

    if position_series is None:
        if best_alignment is not None:
            position_series = str(best_alignment["position_series"])
        else:
            position_series = "series_1"

    if statescript_source is None:
        if best_alignment is not None:
            statescript_source = str(best_alignment["statescript_source"])
        else:
            statescript_source = "statescript_r1"

    if alignment_offset_s is None:
        if best_alignment is not None:
            alignment_offset_s = float(best_alignment["offset_s"])
        else:
            alignment_offset_s = 0.0

    identifiers = _infer_subject_session(nwb, dataset_id)

    position = _load_position_series(nwb, position_series)
    if position.empty:
        raise ValueError(f"Position series is empty after finite filtering: {position_series}")

    events = _load_behavior_events(nwb)
    if events.empty:
        raise ValueError("No behavioral events found in NWB")

    statescript_events = _load_statescript_events(
        probe_dir=probe,
        statescript_source=statescript_source,
        offset_s=float(alignment_offset_s),
    )

    choice_events = _select_choice_events(
        events=events,
        position_time_min=float(position["time_s"].min()),
        position_time_max=float(position["time_s"].max()),
        include_event_regex=include_event_regex,
        max_events=max_events,
    )

    choice_events = _assign_reward_to_events(
        choice_events=choice_events,
        events=events,
        reward_window_after_s=reward_window_after_s,
    )

    trace, choice_summary = _event_window_trace(
        position=position,
        choice_events=choice_events,
        statescript_events=statescript_events,
        identifiers=identifiers,
        dataset_id=dataset_id,
        geometry_id=geometry_id,
        position_series=position_series,
        statescript_source=statescript_source,
        alignment_offset_s=float(alignment_offset_s),
        pre_event_s=pre_event_s,
        post_event_s=post_event_s,
        statescript_match_window_s=statescript_match_window_s,
    )

    trace_path = output / TRACE_CSV
    events_path = output / CHOICE_EVENTS_CSV
    meta_path = output / META_JSON
    report_path = output / REPORT_MD

    trace.to_csv(trace_path, index=False)
    choice_summary.to_csv(events_path, index=False)

    path_counts = (
        choice_summary["committed_path"].value_counts().to_dict()
        if not choice_summary.empty and "committed_path" in choice_summary
        else {}
    )

    meta: dict[str, Any] = {
        "dataset_id": dataset_id,
        "nwb_path": str(nwb),
        "probe_dir": str(probe),
        "output_dir": str(output),
        "position_series": position_series,
        "statescript_source": statescript_source,
        "alignment_offset_s": float(alignment_offset_s),
        "include_event_regex": include_event_regex,
        "pre_event_s": pre_event_s,
        "post_event_s": post_event_s,
        "reward_window_after_s": reward_window_after_s,
        "statescript_match_window_s": statescript_match_window_s,
        "max_events": max_events,
        "run_id": identifiers["run_id"],
        "subject_id": identifiers["subject_id"],
        "session_id": identifiers["session_id"],
        "n_position_samples": int(len(position)),
        "n_behavior_event_rows": int(len(events)),
        "n_candidate_choice_events": int(len(choice_events)),
        "n_choice_events": int(len(choice_summary)),
        "n_trace_rows": int(len(trace)),
        "committed_path_counts": path_counts,
        "trace_csv": str(trace_path),
        "choice_events_csv": str(events_path),
        "report": str(report_path),
        "interpretation": (
            "Event-centered biological canonical trace for proxy-level VTE wrapper comparison. "
            "Not an explicit VTE-label dataset and not a full maze reconstruction."
        ),
    }

    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    report_path.write_text(_write_report(meta), encoding="utf-8")

    print(f"Canonical choice trace saved: {trace_path}")
    print(f"Choice events saved: {events_path}")
    print(f"Metadata saved: {meta_path}")
    print(f"Report saved: {report_path}")
    print(f"Trace rows: {len(trace)}")
    print(f"Choice events: {len(choice_summary)}")

    return meta


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert DANDI 000115 NWB behavior into an event-centered canonical VTE trace."
    )
    parser.add_argument("--nwb", required=True, help="Local NWB file path.")
    parser.add_argument("--probe-dir", required=True, help="Patch 15A probe output directory.")
    parser.add_argument("--output-dir", required=True, help="Output directory.")
    parser.add_argument("--dataset-id", default="dandi_000115")
    parser.add_argument("--geometry-id", default="dandi000115_event_centered_eight_arm")
    parser.add_argument("--position-series", default=None)
    parser.add_argument("--statescript-source", default=None)
    parser.add_argument("--alignment-offset-s", type=float, default=None)
    parser.add_argument("--include-event-regex", default=r"^(arm[1-8]|R|W|home)beam$")
    parser.add_argument("--pre-event-s", type=float, default=2.0)
    parser.add_argument("--post-event-s", type=float, default=4.0)
    parser.add_argument("--reward-window-after-s", type=float, default=2.0)
    parser.add_argument("--statescript-match-window-s", type=float, default=0.1)
    parser.add_argument("--max-events", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    convert_dandi000115_to_canonical_choice_trace(
        nwb_path=args.nwb,
        probe_dir=args.probe_dir,
        output_dir=args.output_dir,
        dataset_id=args.dataset_id,
        geometry_id=args.geometry_id,
        position_series=args.position_series,
        statescript_source=args.statescript_source,
        alignment_offset_s=args.alignment_offset_s,
        include_event_regex=args.include_event_regex,
        pre_event_s=args.pre_event_s,
        post_event_s=args.post_event_s,
        reward_window_after_s=args.reward_window_after_s,
        statescript_match_window_s=args.statescript_match_window_s,
        max_events=args.max_events,
    )


if __name__ == "__main__":
    main()
