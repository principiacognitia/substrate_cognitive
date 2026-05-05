"""Infer a lightweight CRCNS W-track geometry from canonical lab position trace.

Patch 14D scope:
- infer one choice zone from position density;
- infer distal route zones from farthest-point clustering;
- write geometry JSON usable by segmentation;
- no biological claim about exact maze semantics.

The output is an inspectable heuristic layer. It should be replaced or corrected
by a manual/metadata geometry registry when exact lab geometry is available.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


GEOMETRY_ZONE_COLUMNS = [
    "geometry_id",
    "zone_kind",
    "zone_id",
    "x",
    "y",
    "radius",
    "angle_rad",
    "n_support_samples",
]


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

    # Real CRCNS task files usually mark run epochs. Synthetic/minimal fixtures may not.
    return run_df if len(run_df) > 0 else df


def _sample_rows(df: pd.DataFrame, *, max_samples: int) -> pd.DataFrame:
    if max_samples <= 0 or len(df) <= max_samples:
        return df
    return df.sample(n=max_samples, random_state=1729).sort_index()


def _infer_density_center(x: np.ndarray, y: np.ndarray, *, bins: int = 48) -> tuple[float, float]:
    hist, x_edges, y_edges = np.histogram2d(x, y, bins=bins)
    idx = np.unravel_index(np.argmax(hist), hist.shape)

    cx = float((x_edges[idx[0]] + x_edges[idx[0] + 1]) / 2.0)
    cy = float((y_edges[idx[1]] + y_edges[idx[1] + 1]) / 2.0)
    return cx, cy


def _robust_scale(x: np.ndarray, y: np.ndarray, cx: float, cy: float) -> float:
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    finite = dist[np.isfinite(dist)]
    if len(finite) == 0:
        return 1.0
    scale = float(np.nanpercentile(finite, 95))
    return scale if scale > 0 else 1.0


def _infer_route_zones(
    points: np.ndarray,
    *,
    center: np.ndarray,
    n_route_zones: int,
    route_radius: float,
) -> list[dict[str, Any]]:
    """Infer distal route zones by farthest-point clustering."""

    if len(points) == 0:
        return []

    dist_from_center = np.linalg.norm(points - center[None, :], axis=1)
    cutoff = np.nanpercentile(dist_from_center, 80)
    far = points[dist_from_center >= cutoff]

    if len(far) == 0:
        far = points

    k = int(max(1, min(n_route_zones, len(far))))

    centroids: list[np.ndarray] = []
    first_idx = int(np.argmax(np.linalg.norm(far - center[None, :], axis=1)))
    centroids.append(far[first_idx])

    while len(centroids) < k:
        existing = np.vstack(centroids)
        min_dist = np.min(
            np.linalg.norm(far[:, None, :] - existing[None, :, :], axis=2),
            axis=1,
        )
        radial = np.linalg.norm(far - center[None, :], axis=1)
        score = min_dist + 0.15 * radial
        centroids.append(far[int(np.argmax(score))])

    centers = np.vstack(centroids)

    for _ in range(5):
        d = np.linalg.norm(far[:, None, :] - centers[None, :, :], axis=2)
        labels = np.argmin(d, axis=1)
        for i in range(k):
            cluster = far[labels == i]
            if len(cluster) > 0:
                centers[i] = cluster.mean(axis=0)

    d = np.linalg.norm(far[:, None, :] - centers[None, :, :], axis=2)
    labels = np.argmin(d, axis=1)

    zones: list[dict[str, Any]] = []
    for i in range(k):
        cluster = far[labels == i]
        cx, cy = centers[i]
        angle = math.atan2(float(cy - center[1]), float(cx - center[0]))

        zones.append(
            {
                "zone_id": f"arm_{i + 1}",
                "x": float(cx),
                "y": float(cy),
                "radius": float(route_radius),
                "angle_rad": float(angle),
                "n_support_samples": int(len(cluster)),
            }
        )

    zones.sort(key=lambda z: z["angle_rad"])

    for idx, zone in enumerate(zones, start=1):
        zone["zone_id"] = f"arm_{idx}"

    return zones


def _write_geometry_plot(
    df: pd.DataFrame,
    *,
    geometry: dict[str, Any],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.8))

    stride = max(1, int(math.ceil(len(df) / 25000)))
    ax.scatter(
        df["x"].to_numpy()[::stride],
        df["y"].to_numpy()[::stride],
        s=1,
        alpha=0.25,
        label="tracking samples",
    )

    cp = geometry["choice_points"]["wtrack_junction"]
    ax.scatter([cp["x"]], [cp["y"]], s=70, marker="x", label="choice zone")
    choice_circle = plt.Circle(
        (cp["x"], cp["y"]),
        cp["radius"],
        fill=False,
        linewidth=1.5,
    )
    ax.add_patch(choice_circle)

    for zone_id, zone in geometry["route_zones"].items():
        ax.scatter([zone["x"]], [zone["y"]], s=50, marker="o")
        ax.text(zone["x"], zone["y"], zone_id, fontsize=8)
        route_circle = plt.Circle(
            (zone["x"], zone["y"]),
            zone["radius"],
            fill=False,
            linewidth=1.0,
            alpha=0.7,
        )
        ax.add_patch(route_circle)

    ax.set_title(f"Inferred CRCNS W-track geometry: {geometry['geometry_id']}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def infer_wtrack_geometry(
    input_trace: str | Path,
    output_dir: str | Path,
    *,
    geometry_id: str = "crcns_wtrack_inferred",
    dataset_id: str | None = None,
    animal_id: str | None = None,
    run_epochs_only: bool = True,
    n_route_zones: int = 4,
    max_samples: int = 60000,
    junction_x: float | None = None,
    junction_y: float | None = None,
    choice_radius: float | None = None,
    route_radius: float | None = None,
) -> dict[str, Any]:
    trace_path = Path(input_trace)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(trace_path, low_memory=False)
    df = _finite_xy(raw)
    df = _filter_run_epochs(df, run_epochs_only=run_epochs_only)
    df = _sample_rows(df, max_samples=max_samples)

    if len(df) == 0:
        raise ValueError("No finite x/y samples available for geometry inference.")

    x = df["x"].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=float)

    if junction_x is None or junction_y is None:
        inferred_x, inferred_y = _infer_density_center(x, y)
        cx = inferred_x if junction_x is None else float(junction_x)
        cy = inferred_y if junction_y is None else float(junction_y)
    else:
        cx = float(junction_x)
        cy = float(junction_y)

    scale = _robust_scale(x, y, cx, cy)

    resolved_choice_radius = float(choice_radius) if choice_radius is not None else 0.10 * scale
    resolved_route_radius = float(route_radius) if route_radius is not None else 0.12 * scale

    points = np.column_stack([x, y])
    route_zones = _infer_route_zones(
        points,
        center=np.asarray([cx, cy], dtype=float),
        n_route_zones=n_route_zones,
        route_radius=resolved_route_radius,
    )

    resolved_dataset_id = dataset_id
    if resolved_dataset_id is None and "dataset_id" in raw.columns:
        vals = raw["dataset_id"].dropna().astype(str).unique()
        resolved_dataset_id = vals[0] if len(vals) else ""

    resolved_animal_id = animal_id
    if resolved_animal_id is None and "animal_id" in raw.columns:
        vals = raw["animal_id"].dropna().astype(str).unique()
        resolved_animal_id = vals[0] if len(vals) else ""

    geometry = {
        "geometry_id": geometry_id,
        "dataset_id": resolved_dataset_id or "",
        "animal_id": resolved_animal_id or "",
        "source_trace_csv": str(trace_path),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "coordinate_system": "lab_position_units",
        "inference_method": "density_center_plus_farthest_point_route_zones",
        "run_epochs_only": bool(run_epochs_only),
        "n_support_samples": int(len(df)),
        "choice_points": {
            "wtrack_junction": {
                "x": float(cx),
                "y": float(cy),
                "radius": float(resolved_choice_radius),
            }
        },
        "route_zones": {
            zone["zone_id"]: {
                "x": zone["x"],
                "y": zone["y"],
                "radius": zone["radius"],
                "angle_rad": zone["angle_rad"],
                "n_support_samples": zone["n_support_samples"],
            }
            for zone in route_zones
        },
    }

    rows: list[dict[str, Any]] = [
        {
            "geometry_id": geometry_id,
            "zone_kind": "choice_point",
            "zone_id": "wtrack_junction",
            "x": float(cx),
            "y": float(cy),
            "radius": float(resolved_choice_radius),
            "angle_rad": "",
            "n_support_samples": int(len(df)),
        }
    ]

    for zone in route_zones:
        rows.append(
            {
                "geometry_id": geometry_id,
                "zone_kind": "route_zone",
                "zone_id": zone["zone_id"],
                "x": zone["x"],
                "y": zone["y"],
                "radius": zone["radius"],
                "angle_rad": zone["angle_rad"],
                "n_support_samples": zone["n_support_samples"],
            }
        )

    geometry_json = out_dir / "crcns_wtrack_geometry.json"
    zones_csv = out_dir / "Table_CRCNS_WTrack_Geometry_Zones.csv"
    figure_png = out_dir / "Figure_CRCNS_WTrack_Inferred_Geometry.png"
    meta_json = out_dir / "crcns_wtrack_geometry_meta.json"

    geometry_json.write_text(json.dumps(geometry, indent=2), encoding="utf-8")
    pd.DataFrame(rows, columns=GEOMETRY_ZONE_COLUMNS).to_csv(zones_csv, index=False)
    _write_geometry_plot(df, geometry=geometry, output_path=figure_png)

    meta = {
        "script": "vte.lab_adapters.crcns_wtrack.infer_wtrack_geometry",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_trace": str(trace_path),
        "output_dir": str(out_dir),
        "geometry_json": str(geometry_json),
        "zones_csv": str(zones_csv),
        "figure_png": str(figure_png),
        "geometry_id": geometry_id,
        "n_support_samples": int(len(df)),
        "n_route_zones": int(len(route_zones)),
        "choice_radius": float(resolved_choice_radius),
        "route_radius": float(resolved_route_radius),
        "heuristic": True,
    }

    meta_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Geometry saved: {geometry_json}")
    print(f"Zones table saved: {zones_csv}")
    print(f"Figure saved: {figure_png}")
    print(f"Metadata saved: {meta_json}")

    return meta


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Infer CRCNS W-track geometry from canonical trace.")
    parser.add_argument("--input-trace", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--geometry-id", default="crcns_wtrack_inferred")
    parser.add_argument("--dataset-id", default=None)
    parser.add_argument("--animal-id", default=None)
    parser.add_argument("--all-epochs", action="store_true", help="Use all epochs, not only task_type == run.")
    parser.add_argument("--n-route-zones", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=60000)
    parser.add_argument("--junction-x", type=float, default=None)
    parser.add_argument("--junction-y", type=float, default=None)
    parser.add_argument("--choice-radius", type=float, default=None)
    parser.add_argument("--route-radius", type=float, default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    infer_wtrack_geometry(
        input_trace=args.input_trace,
        output_dir=args.output_dir,
        geometry_id=args.geometry_id,
        dataset_id=args.dataset_id,
        animal_id=args.animal_id,
        run_epochs_only=not args.all_epochs,
        n_route_zones=args.n_route_zones,
        max_samples=args.max_samples,
        junction_x=args.junction_x,
        junction_y=args.junction_y,
        choice_radius=args.choice_radius,
        route_radius=args.route_radius,
    )


if __name__ == "__main__":
    main()