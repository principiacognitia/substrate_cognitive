"""QA diagnostics for CRCNS W-track heuristic segmentation.

Patch 14E.

This module does not change segmentation output and does not compute model
metrics. It audits an existing segmented biological trace against a canonical
tracking trace and an inferred/manual geometry JSON.

Main outputs:
- route usage by epoch;
- per-choice exit diagnostics;
- commit-rule stability diagnostics;
- QA figures;
- metadata JSON.

The goal is to distinguish "real task/animal route imbalance" from a possible
commit-rule artifact.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROUTE_USAGE_COLUMNS = [
    "dataset_id",
    "animal_id",
    "run_id",
    "day",
    "epoch",
    "committed_path",
    "n_trials",
    "epoch_trials",
    "route_rate",
]

CHOICE_EXIT_COLUMNS = [
    "dataset_id",
    "animal_id",
    "run_id",
    "day",
    "epoch",
    "trial",
    "choice_visit_index",
    "existing_committed_path",
    "choice_duration_samples",
    "choice_start_source_row_index",
    "choice_end_source_row_index",
    "entry_x",
    "entry_y",
    "exit_x",
    "exit_y",
    "exit_angle_rad",
    "exit_distance_from_choice",
    "post_window_n_samples",
    "post_window_max_distance_from_choice",
    "first_non_choice_route",
    "first_non_choice_distance_to_route",
    "dominant_post_route",
    "dominant_post_route_fraction",
    "farthest_post_route",
    "farthest_post_distance_from_choice",
    "first_non_choice_matches_existing",
    "dominant_post_matches_existing",
    "farthest_post_matches_existing",
    "rule_available_count",
    "rule_agreement_count",
    "route_assignment_method",
]

COMMIT_RULE_COLUMNS = [
    "group",
    "epoch",
    "n_trials",
    "first_non_choice_available_rate",
    "first_non_choice_match_rate",
    "dominant_post_available_rate",
    "dominant_post_match_rate",
    "farthest_post_available_rate",
    "farthest_post_match_rate",
    "all_rules_available_rate",
    "unanimous_rule_rate",
]


@dataclass(frozen=True)
class Zone:
    zone_id: str
    role: str
    x: float
    y: float
    radius: float | None = None


@dataclass(frozen=True)
class GeometrySpec:
    geometry_id: str
    choice_zone: Zone
    route_zones: list[Zone]


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip() == "":
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    return v


def _safe_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, float) and math.isnan(value):
        return default
    text = str(value)
    return text if text != "nan" else default


def _xy_from_dict(d: dict[str, Any]) -> tuple[float, float] | None:
    direct_x = _safe_float(d.get("x"))
    direct_y = _safe_float(d.get("y"))
    if direct_x is not None and direct_y is not None:
        return direct_x, direct_y

    cx = _safe_float(d.get("center_x"))
    cy = _safe_float(d.get("center_y"))
    if cx is not None and cy is not None:
        return cx, cy

    center = d.get("center")
    if isinstance(center, dict):
        cx = _safe_float(center.get("x") or center.get("center_x"))
        cy = _safe_float(center.get("y") or center.get("center_y"))
        if cx is not None and cy is not None:
            return cx, cy

    if isinstance(center, (list, tuple)) and len(center) >= 2:
        cx = _safe_float(center[0])
        cy = _safe_float(center[1])
        if cx is not None and cy is not None:
            return cx, cy

    return None


def _radius_from_dict(d: dict[str, Any]) -> float | None:
    for key in ("radius", "r", "radius_px", "zone_radius"):
        value = _safe_float(d.get(key))
        if value is not None:
            return value
    return None


def _zone_id_from_dict(d: dict[str, Any], fallback: str) -> str:
    for key in ("zone_id", "id", "name", "route_id", "choice_point_id", "label"):
        value = _safe_str(d.get(key))
        if value:
            return value
    return fallback


def _role_from_dict(d: dict[str, Any], zone_id: str) -> str:
    for key in ("role", "zone_type", "type", "kind"):
        value = _safe_str(d.get(key))
        if value:
            return value
    if "choice" in zone_id.lower() or "junction" in zone_id.lower():
        return "choice"
    if "arm" in zone_id.lower() or "route" in zone_id.lower():
        return "route"
    return ""


def _collect_zone_dicts(obj: Any, out: list[dict[str, Any]]) -> None:
    if isinstance(obj, dict):
        if _xy_from_dict(obj) is not None:
            out.append(obj)
        for value in obj.values():
            _collect_zone_dicts(value, out)
    elif isinstance(obj, list):
        for value in obj:
            _collect_zone_dicts(value, out)


def load_geometry(geometry_json: str | Path) -> GeometrySpec:
    """Load inferred or manual W-track geometry.

    The parser is intentionally permissive: it accepts several likely JSON
    layouts with zone dictionaries containing x/y or center coordinates.
    """

    path = Path(geometry_json)
    data = json.loads(path.read_text(encoding="utf-8"))

    geometry_id = _safe_str(data.get("geometry_id"), path.stem)

    zone_dicts: list[dict[str, Any]] = []
    _collect_zone_dicts(data, zone_dicts)

    zones: list[Zone] = []
    for idx, d in enumerate(zone_dicts, start=1):
        xy = _xy_from_dict(d)
        if xy is None:
            continue

        zone_id = _zone_id_from_dict(d, f"zone_{idx}")
        role = _role_from_dict(d, zone_id)
        radius = _radius_from_dict(d)

        zones.append(
            Zone(
                zone_id=zone_id,
                role=role,
                x=float(xy[0]),
                y=float(xy[1]),
                radius=radius,
            )
        )

    if not zones:
        raise ValueError(f"No usable zones found in geometry JSON: {path}")

    choice_candidates = [
        z for z in zones
        if "choice" in z.role.lower()
        or "choice" in z.zone_id.lower()
        or "junction" in z.zone_id.lower()
    ]

    if choice_candidates:
        choice_zone = choice_candidates[0]
    else:
        # W-track choice zone is usually the lower central zone in tracking
        # coordinates. This is only fallback behavior for incomplete JSON.
        choice_zone = sorted(zones, key=lambda z: z.y)[0]

    route_zones = [
        z for z in zones
        if z.zone_id != choice_zone.zone_id
        and (
            "route" in z.role.lower()
            or "arm" in z.role.lower()
            or "reward" in z.role.lower()
            or "terminal" in z.role.lower()
            or "arm" in z.zone_id.lower()
        )
    ]

    if not route_zones:
        route_zones = [z for z in zones if z.zone_id != choice_zone.zone_id]

    if not route_zones:
        raise ValueError(f"No route zones found in geometry JSON: {path}")

    return GeometrySpec(
        geometry_id=geometry_id,
        choice_zone=choice_zone,
        route_zones=route_zones,
    )


def _distance(x: float, y: float, zone: Zone) -> float:
    return float(math.hypot(x - zone.x, y - zone.y))


def _assign_route(x: float, y: float, route_zones: list[Zone]) -> tuple[str, float, bool]:
    """Assign a point to the nearest route zone.

    Returns route_id, distance_to_zone_center, inside_any_declared_radius.
    """

    distances = [(z, _distance(x, y, z)) for z in route_zones]
    if not distances:
        return "", float("nan"), False

    inside = [
        (z, d) for z, d in distances
        if z.radius is not None and d <= z.radius
    ]
    if inside:
        z, d = min(inside, key=lambda item: item[1])
        return z.zone_id, float(d), True

    z, d = min(distances, key=lambda item: item[1])
    return z.zone_id, float(d), False


def _coerce_numeric_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _trial_group_columns(df: pd.DataFrame) -> list[str]:
    preferred = ["dataset_id", "animal_id", "run_id", "day", "epoch", "trial"]
    cols = [c for c in preferred if c in df.columns]
    if "trial" not in cols:
        raise ValueError("Segmented trace must contain a 'trial' column.")
    return cols


def _first_nonblank(series: pd.Series, default: str = "") -> str:
    for value in series:
        text = _safe_str(value)
        if text:
            return text
    return default


def build_route_usage_by_epoch(segmented_df: pd.DataFrame) -> pd.DataFrame:
    """Count existing committed_path labels per epoch."""

    if segmented_df.empty:
        return pd.DataFrame(columns=ROUTE_USAGE_COLUMNS)

    df = segmented_df.copy()
    group_cols = _trial_group_columns(df)

    trial_rows = []
    for key, g in df.groupby(group_cols, sort=True, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(group_cols, key))
        row["committed_path"] = _first_nonblank(g.get("committed_path", pd.Series(dtype=object)))
        trial_rows.append(row)

    trials = pd.DataFrame(trial_rows)
    if trials.empty:
        return pd.DataFrame(columns=ROUTE_USAGE_COLUMNS)

    epoch_cols = [c for c in ["dataset_id", "animal_id", "run_id", "day", "epoch"] if c in trials.columns]
    rows: list[dict[str, Any]] = []

    for key, g in trials.groupby(epoch_cols, sort=True, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        base = dict(zip(epoch_cols, key))
        epoch_total = int(len(g))

        counts = (
            g["committed_path"]
            .fillna("")
            .replace("", "unassigned")
            .value_counts()
            .sort_index()
        )

        for committed_path, count in counts.items():
            row = {col: base.get(col, "") for col in ROUTE_USAGE_COLUMNS}
            row.update(base)
            row["committed_path"] = committed_path
            row["n_trials"] = int(count)
            row["epoch_trials"] = epoch_total
            row["route_rate"] = float(count / epoch_total) if epoch_total else float("nan")
            rows.append(row)

    return pd.DataFrame(rows, columns=ROUTE_USAGE_COLUMNS)


def _subset_canonical_for_segment(
    canonical_df: pd.DataFrame,
    seg_group: pd.DataFrame,
    *,
    post_window_samples: int,
) -> tuple[pd.DataFrame, pd.DataFrame, float | None, float | None]:
    """Return full aligned window and post-choice rows for one segmented trial."""

    subset = canonical_df

    for col in ["dataset_id", "animal_id", "run_id", "day", "epoch"]:
        if col in subset.columns and col in seg_group.columns:
            value = seg_group[col].iloc[0]
            subset = subset.loc[subset[col].astype(str) == str(value)]

    if "source_row_index" in subset.columns and "source_row_index" in seg_group.columns:
        seg_idx = pd.to_numeric(seg_group["source_row_index"], errors="coerce")
        start_idx = _safe_float(seg_idx.min())
        end_idx = _safe_float(seg_idx.max())

        if start_idx is not None and end_idx is not None:
            source_idx = pd.to_numeric(subset["source_row_index"], errors="coerce")
            full = subset.loc[
                (source_idx >= start_idx)
                & (source_idx <= end_idx + post_window_samples)
            ].copy()
            post = subset.loc[
                (source_idx > end_idx)
                & (source_idx <= end_idx + post_window_samples)
            ].copy()
            return full, post, float(start_idx), float(end_idx)

    # Fallback: use segmented rows only. This still yields exit geometry
    # diagnostics, but commit-rule re-evaluation will be mostly unavailable.
    return seg_group.copy(), pd.DataFrame(columns=canonical_df.columns), None, None


def _post_route_series(
    post_df: pd.DataFrame,
    geometry: GeometrySpec,
) -> pd.DataFrame:
    if post_df.empty:
        return pd.DataFrame(columns=["route_id", "distance_to_route", "distance_to_choice"])

    rows = []
    for _, row in post_df.iterrows():
        x = _safe_float(row.get("x"))
        y = _safe_float(row.get("y"))
        if x is None or y is None:
            continue

        dist_choice = _distance(x, y, geometry.choice_zone)
        choice_radius = geometry.choice_zone.radius or 0.0

        if dist_choice <= choice_radius:
            continue

        route_id, dist_route, inside = _assign_route(x, y, geometry.route_zones)
        rows.append(
            {
                "route_id": route_id,
                "distance_to_route": dist_route,
                "distance_to_choice": dist_choice,
                "inside_route_radius": bool(inside),
            }
        )

    return pd.DataFrame(rows)

def _as_bool_series(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.strip().str.lower()
    return text.isin({"true", "1", "yes", "y"})

def _route_alias_map(geometry: GeometrySpec) -> dict[str, str]:
    """Map internal geometry zone ids to externally stable route labels.

    In inferred W-track geometry, route zones may have internal ids such as
    zone_2 / zone_4 while tables and segmented traces use arm_2 / arm_3.
    The QA comparison must use the stable committed-path namespace.
    """
    aliases: dict[str, str] = {}

    route_zones = getattr(geometry, "route_zones", [])
    for zone in route_zones:
        zone_id = _safe_str(getattr(zone, "zone_id", ""))
        label = _safe_str(getattr(zone, "label", ""))

        if zone_id:
            aliases[zone_id] = label or zone_id
        if label:
            aliases[label] = label

    # Defensive fallback for older inferred geometry where labels were not
    # serialized but zone ids follow top-arm order.
    # This should only affect QA display, not segmentation itself.
    for i in range(1, 10):
        aliases.setdefault(f"arm_{i}", f"arm_{i}")

    return aliases


def _canonical_route_label(route_id: Any, aliases: dict[str, str]) -> str:
    value = _safe_str(route_id)
    if not value:
        return ""
    return aliases.get(value, value)

def build_choice_exit_diagnostics(
    canonical_df: pd.DataFrame,
    segmented_df: pd.DataFrame,
    geometry: GeometrySpec,
    *,
    post_window_samples: int = 300,
) -> pd.DataFrame:
    """Build per-choice diagnostics and commit-rule re-evaluation.

    Primary rule source is the segmented trial itself:
    rows with at_choice_point == False after the choice-zone visit.

    Canonical post-window lookup is only a fallback for traces that contain
    choice-zone rows but not route rows.
    """

    if segmented_df.empty:
        return pd.DataFrame(columns=CHOICE_EXIT_COLUMNS)

    canonical = _coerce_numeric_columns(
        canonical_df,
        ["x", "y", "heading", "source_row_index", "sample_index", "time_s"],
    )
    route_aliases = _route_alias_map(geometry)
    segmented = _coerce_numeric_columns(
        segmented_df,
        ["x", "y", "heading", "source_row_index", "sample_index", "trial", "epoch", "day"],
    )

    group_cols = _trial_group_columns(segmented)
    rows: list[dict[str, Any]] = []

    for key, g in segmented.groupby(group_cols, sort=True, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        base = dict(zip(group_cols, key))

        sort_cols = [c for c in ["source_row_index", "sample_index", "tick"] if c in g.columns]
        if sort_cols:
            g = g.sort_values(sort_cols)

        existing = _first_nonblank(g.get("committed_path", pd.Series(dtype=object)))
        choice_visit_index = _first_nonblank(
            g.get("choice_visit_index", pd.Series(dtype=object)),
            default="",
        )

        if "at_choice_point" in g.columns:
            choice_mask = _as_bool_series(g["at_choice_point"])
        else:
            choice_mask = pd.Series(True, index=g.index)

        choice_rows = g.loc[choice_mask].copy()
        if choice_rows.empty:
            choice_rows = g.head(1).copy()

        in_trial_post = g.loc[~choice_mask].copy()

        start_idx = None
        end_idx = None
        if "source_row_index" in choice_rows.columns:
            choice_idx = pd.to_numeric(choice_rows["source_row_index"], errors="coerce")
            start_idx = _safe_float(choice_idx.min())
            end_idx = _safe_float(choice_idx.max())

        # Fallback: if segmented group has no post-choice rows, take rows
        # after choice-zone exit from canonical trace.
        if in_trial_post.empty and end_idx is not None:
            subset = canonical
            for col in ["dataset_id", "animal_id", "run_id", "day", "epoch"]:
                if col in subset.columns and col in g.columns:
                    value = g[col].iloc[0]
                    subset = subset.loc[subset[col].astype(str) == str(value)]

            if "source_row_index" in subset.columns:
                source_idx = pd.to_numeric(subset["source_row_index"], errors="coerce")
                in_trial_post = subset.loc[
                    (source_idx > end_idx)
                    & (source_idx <= end_idx + post_window_samples)
                ].copy()

        entry_row = choice_rows.iloc[0] if not choice_rows.empty else g.iloc[0]
        if not in_trial_post.empty:
            exit_row = in_trial_post.iloc[0]
        else:
            exit_row = choice_rows.iloc[-1] if not choice_rows.empty else g.iloc[-1]

        entry_x = _safe_float(entry_row.get("x"))
        entry_y = _safe_float(entry_row.get("y"))
        exit_x = _safe_float(exit_row.get("x"))
        exit_y = _safe_float(exit_row.get("y"))

        if exit_x is not None and exit_y is not None:
            exit_angle = float(math.atan2(exit_y - geometry.choice_zone.y, exit_x - geometry.choice_zone.x))
            exit_distance = _distance(exit_x, exit_y, geometry.choice_zone)
        else:
            exit_angle = float("nan")
            exit_distance = float("nan")

        post_routes = _post_route_series(in_trial_post, geometry)

        # Fallback when geometry assignment cannot produce routes but segmented
        # trace already carries route_id labels.
        if post_routes.empty and "route_id" in in_trial_post.columns:
            route_rows = in_trial_post.copy()
            route_rows["route_id"] = route_rows["route_id"].astype(str)
            route_rows = route_rows.loc[
                (route_rows["route_id"] != "")
                & (route_rows["route_id"].str.lower() != "nan")
                & (route_rows["route_id"] != geometry.choice_zone.zone_id)
            ]

            fallback_rows = []
            for _, rr in route_rows.iterrows():
                x = _safe_float(rr.get("x"))
                y = _safe_float(rr.get("y"))
                if x is None or y is None:
                    dist_choice = float("nan")
                else:
                    dist_choice = _distance(x, y, geometry.choice_zone)

                fallback_rows.append(
                    {
                        "route_id": _safe_str(rr.get("route_id")),
                        "distance_to_route": float("nan"),
                        "distance_to_choice": dist_choice,
                    }
                )

            post_routes = pd.DataFrame(fallback_rows)

        first_route = ""
        first_route_dist = float("nan")
        dominant_route = ""
        dominant_fraction = float("nan")
        farthest_route = ""
        farthest_distance = float("nan")
        max_post_distance = float("nan")

        if not post_routes.empty:
            first_route = _safe_str(post_routes["route_id"].iloc[0])
            first_route_dist = float(post_routes["distance_to_route"].iloc[0])

            counts = post_routes["route_id"].value_counts()
            dominant_route = _safe_str(counts.index[0])
            dominant_fraction = float(counts.iloc[0] / len(post_routes))

            if "distance_to_choice" in post_routes.columns:
                dist = pd.to_numeric(post_routes["distance_to_choice"], errors="coerce")
                if dist.notna().any():
                    farthest_idx = dist.idxmax()
                    farthest_route = _safe_str(post_routes.loc[farthest_idx, "route_id"])
                    farthest_distance = float(dist.loc[farthest_idx])
                    max_post_distance = float(dist.max())

            if not farthest_route:
                farthest_route = dominant_route

        first_route_label = _canonical_route_label(first_route, route_aliases)
        dominant_route_label = _canonical_route_label(dominant_route, route_aliases)
        farthest_route_label = _canonical_route_label(farthest_route, route_aliases)

        rule_values = {
            "first_non_choice": first_route_label,
            "dominant_post": dominant_route_label,
            "farthest_post": farthest_route_label,
        }

        available = {name: bool(value) for name, value in rule_values.items()}
        matches = {
            name: (bool(value) and value == existing)
            for name, value in rule_values.items()
        }

        row = {col: "" for col in CHOICE_EXIT_COLUMNS}
        row.update(base)
        row.update(
            {
                "choice_visit_index": choice_visit_index,
                "existing_committed_path": existing,
                "choice_duration_samples": int(len(choice_rows)),
                "choice_start_source_row_index": start_idx if start_idx is not None else "",
                "choice_end_source_row_index": end_idx if end_idx is not None else "",
                "entry_x": entry_x if entry_x is not None else "",
                "entry_y": entry_y if entry_y is not None else "",
                "exit_x": exit_x if exit_x is not None else "",
                "exit_y": exit_y if exit_y is not None else "",
                "exit_angle_rad": exit_angle,
                "exit_distance_from_choice": exit_distance,
                "post_window_n_samples": int(len(post_routes)),
                "post_window_max_distance_from_choice": max_post_distance,
                "first_non_choice_route": first_route_label,
                "first_non_choice_distance_to_route": first_route_dist,
                "dominant_post_route": dominant_route_label,
                "dominant_post_route_fraction": dominant_fraction,
                "farthest_post_route": farthest_route_label,
                "farthest_post_distance_from_choice": farthest_distance,
                "first_non_choice_matches_existing": matches["first_non_choice"] if available["first_non_choice"] else "",
                "dominant_post_matches_existing": matches["dominant_post"] if available["dominant_post"] else "",
                "farthest_post_matches_existing": matches["farthest_post"] if available["farthest_post"] else "",
                "rule_available_count": int(sum(available.values())),
                "rule_agreement_count": int(sum(matches.values())),
                "route_assignment_method": "segmented_trial_post_choice_rows_then_canonical_fallback",
            }
        )
        rows.append(row)

    return pd.DataFrame(rows, columns=CHOICE_EXIT_COLUMNS)


def build_commit_rule_stability(exit_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate commit-rule agreement diagnostics."""

    if exit_df.empty:
        return pd.DataFrame(columns=COMMIT_RULE_COLUMNS)

    rows: list[dict[str, Any]] = []

    def _rate(series: pd.Series) -> float:
        valid = series.loc[series != ""]
        if len(valid) == 0:
            return float("nan")
        return float(valid.astype(bool).mean())

    def _available(series: pd.Series) -> float:
        return float((series != "").mean()) if len(series) else float("nan")

    def _make_row(group: str, epoch: str | int, g: pd.DataFrame) -> dict[str, Any]:
        all_available = (
            (g["first_non_choice_matches_existing"] != "")
            & (g["dominant_post_matches_existing"] != "")
            & (g["farthest_post_matches_existing"] != "")
        )

        unanimous = (
            all_available
            & (g["first_non_choice_route"] == g["dominant_post_route"])
            & (g["dominant_post_route"] == g["farthest_post_route"])
        )

        return {
            "group": group,
            "epoch": epoch,
            "n_trials": int(len(g)),
            "first_non_choice_available_rate": _available(g["first_non_choice_matches_existing"]),
            "first_non_choice_match_rate": _rate(g["first_non_choice_matches_existing"]),
            "dominant_post_available_rate": _available(g["dominant_post_matches_existing"]),
            "dominant_post_match_rate": _rate(g["dominant_post_matches_existing"]),
            "farthest_post_available_rate": _available(g["farthest_post_matches_existing"]),
            "farthest_post_match_rate": _rate(g["farthest_post_matches_existing"]),
            "all_rules_available_rate": float(all_available.mean()) if len(g) else float("nan"),
            "unanimous_rule_rate": float(unanimous.mean()) if len(g) else float("nan"),
        }

    rows.append(_make_row("overall", "all", exit_df))

    if "epoch" in exit_df.columns:
        for epoch, g in exit_df.groupby("epoch", sort=True, dropna=False):
            rows.append(_make_row("epoch", epoch, g))

    return pd.DataFrame(rows, columns=COMMIT_RULE_COLUMNS)


def _plot_route_usage(route_usage: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))

    if route_usage.empty:
        ax.set_title("CRCNS W-track route usage by epoch: no data")
        ax.axis("off")
    else:
        pivot = route_usage.pivot_table(
            index="epoch",
            columns="committed_path",
            values="route_rate",
            aggfunc="sum",
            fill_value=0.0,
        ).sort_index()

        bottom = np.zeros(len(pivot))
        x = np.arange(len(pivot.index))

        for col in pivot.columns:
            values = pivot[col].to_numpy(dtype=float)
            ax.bar(x, values, bottom=bottom, label=str(col))
            bottom += values

        ax.set_title("CRCNS W-track route usage by epoch")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Route fraction")
        ax.set_xticks(x)
        ax.set_xticklabels([str(v) for v in pivot.index])
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8, loc="best")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_commit_rule_stability(stability: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))

    if stability.empty:
        ax.set_title("CRCNS W-track commit-rule stability: no data")
        ax.axis("off")
    else:
        epoch_rows = stability.loc[stability["group"] == "epoch"].copy()
        if epoch_rows.empty:
            epoch_rows = stability.loc[stability["group"] == "overall"].copy()

        x = np.arange(len(epoch_rows))
        labels = epoch_rows["epoch"].astype(str).tolist()

        ax.plot(x, epoch_rows["first_non_choice_match_rate"], marker="o", label="first_non_choice")
        ax.plot(x, epoch_rows["dominant_post_match_rate"], marker="o", label="dominant_post")
        ax.plot(x, epoch_rows["farthest_post_match_rate"], marker="o", label="farthest_post")
        ax.plot(x, epoch_rows["unanimous_rule_rate"], marker="o", label="unanimous_rule")

        ax.set_title("CRCNS W-track commit-rule stability")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Rate")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(-0.02, 1.02)
        ax.legend(fontsize=8, loc="best")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def validate_wtrack_segmentation(
    *,
    canonical_trace_csv: str | Path,
    segmented_trace_csv: str | Path,
    geometry_json: str | Path,
    output_dir: str | Path,
    post_window_samples: int = 300,
) -> dict[str, Any]:
    """Run CRCNS W-track segmentation QA."""

    canonical_path = Path(canonical_trace_csv)
    segmented_path = Path(segmented_trace_csv)
    geometry_path = Path(geometry_json)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    canonical = pd.read_csv(canonical_path, low_memory=False)
    segmented = pd.read_csv(segmented_path, low_memory=False)
    geometry = load_geometry(geometry_path)

    route_usage = build_route_usage_by_epoch(segmented)
    exit_diag = build_choice_exit_diagnostics(
        canonical,
        segmented,
        geometry,
        post_window_samples=post_window_samples,
    )
    stability = build_commit_rule_stability(exit_diag)

    route_usage_csv = out_dir / "Table_CRCNS_WTrack_Route_Usage_By_Epoch.csv"
    exit_diag_csv = out_dir / "Table_CRCNS_WTrack_Choice_Exit_Diagnostics.csv"
    stability_csv = out_dir / "Table_CRCNS_WTrack_Commit_Rule_Stability.csv"
    route_usage_fig = out_dir / "Figure_CRCNS_WTrack_Route_Usage_By_Epoch.png"
    stability_fig = out_dir / "Figure_CRCNS_WTrack_Commit_Rule_Stability.png"
    meta_json = out_dir / "crcns_wtrack_segmentation_qa_meta.json"

    route_usage.to_csv(route_usage_csv, index=False)
    exit_diag.to_csv(exit_diag_csv, index=False)
    stability.to_csv(stability_csv, index=False)

    _plot_route_usage(route_usage, route_usage_fig)
    _plot_commit_rule_stability(stability, stability_fig)

    meta: dict[str, Any] = {
        "script": "vte.lab_adapters.crcns_wtrack.validate_wtrack_segmentation",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "canonical_trace_csv": str(canonical_path),
        "segmented_trace_csv": str(segmented_path),
        "geometry_json": str(geometry_path),
        "output_dir": str(out_dir),
        "post_window_samples": int(post_window_samples),
        "geometry_id": geometry.geometry_id,
        "choice_zone": {
            "zone_id": geometry.choice_zone.zone_id,
            "x": geometry.choice_zone.x,
            "y": geometry.choice_zone.y,
            "radius": geometry.choice_zone.radius,
        },
        "route_zones": [
            {
                "zone_id": z.zone_id,
                "x": z.x,
                "y": z.y,
                "radius": z.radius,
            }
            for z in geometry.route_zones
        ],
        "n_canonical_rows": int(len(canonical)),
        "n_segmented_rows": int(len(segmented)),
        "n_segmented_trials": int(exit_diag["trial"].nunique()) if not exit_diag.empty else 0,
        "route_usage_csv": str(route_usage_csv),
        "choice_exit_diagnostics_csv": str(exit_diag_csv),
        "commit_rule_stability_csv": str(stability_csv),
        "route_usage_figure_png": str(route_usage_fig),
        "commit_rule_stability_figure_png": str(stability_fig),
    }

    meta_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Route usage saved: {route_usage_csv}")
    print(f"Choice-exit diagnostics saved: {exit_diag_csv}")
    print(f"Commit-rule stability saved: {stability_csv}")
    print(f"Figure saved: {route_usage_fig}")
    print(f"Figure saved: {stability_fig}")
    print(f"Metadata saved: {meta_json}")

    return meta


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate CRCNS W-track heuristic segmentation against canonical trace and geometry."
    )
    parser.add_argument("--canonical-trace-csv", required=True, type=Path)
    parser.add_argument("--segmented-trace-csv", required=True, type=Path)
    parser.add_argument("--geometry-json", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--post-window-samples",
        type=int,
        default=300,
        help="Number of canonical samples after choice-zone exit used for route-rule QA.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    validate_wtrack_segmentation(
        canonical_trace_csv=args.canonical_trace_csv,
        segmented_trace_csv=args.segmented_trace_csv,
        geometry_json=args.geometry_json,
        output_dir=args.output_dir,
        post_window_samples=args.post_window_samples,
    )


if __name__ == "__main__":
    main()