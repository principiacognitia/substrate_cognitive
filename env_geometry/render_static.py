"""Render static task-topology schematics from env_geometry registry."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch

from env_geometry.registry import (
    DEFAULT_REGISTRY_PATH,
    get_geometry,
    iter_geometries,
    list_geometries,
    write_registry_export,
)


def _node_lookup(geometry: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {node["id"]: node for node in geometry["nodes"]}


def _draw_edge(ax, source: Dict[str, Any], target: Dict[str, Any], label: str, kind: str) -> None:
    sx, sy = float(source["x"]), float(source["y"])
    tx, ty = float(target["x"]), float(target["y"])

    linestyle = "--" if "rare" in kind else "-"
    alpha = 0.55 if "rare" in kind else 0.85

    arrow = FancyArrowPatch(
        (sx, sy),
        (tx, ty),
        arrowstyle="->",
        mutation_scale=12,
        linewidth=1.4,
        linestyle=linestyle,
        alpha=alpha,
    )
    ax.add_patch(arrow)

    mx = sx + (tx - sx) * 0.55
    my = sy + (ty - sy) * 0.55
    ax.text(mx, my, label, fontsize=8, ha="center", va="center")


def _draw_node(ax, node: Dict[str, Any]) -> None:
    x, y = float(node["x"]), float(node["y"])
    kind = node.get("kind", "")

    radius = 0.16
    if kind in {"start", "start_choice", "choice_point"}:
        radius = 0.20
    elif kind in {"terminal", "goal", "reward_action"}:
        radius = 0.15

    circle = Circle((x, y), radius=radius, fill=False, linewidth=1.8)
    ax.add_patch(circle)
    ax.text(x, y, node["label"], fontsize=9, ha="center", va="center")


def _draw_zone_labels(ax, geometry: Dict[str, Any], nodes: Dict[str, Dict[str, Any]]) -> None:
    for zone in geometry.get("zones", []):
        node_ids = [nid for nid in zone.get("node_ids", []) if nid in nodes]
        if not node_ids:
            continue

        xs = [float(nodes[nid]["x"]) for nid in node_ids]
        ys = [float(nodes[nid]["y"]) for nid in node_ids]
        x = sum(xs) / len(xs)
        y = sum(ys) / len(ys)

        ax.text(
            x,
            y + 0.32,
            zone.get("label", zone.get("id", "")),
            fontsize=8,
            ha="center",
            va="bottom",
            alpha=0.75,
        )


def render_geometry(
    geometry: Dict[str, Any],
    output_dir: str | Path,
    *,
    dpi: int = 200,
) -> Path:
    """Render one geometry to a PNG file."""

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    nodes = _node_lookup(geometry)

    fig, ax = plt.subplots(figsize=(7.0, 5.0))

    for edge in geometry["edges"]:
        _draw_edge(
            ax,
            nodes[edge["source"]],
            nodes[edge["target"]],
            str(edge.get("label", "")),
            str(edge.get("kind", "")),
        )

    for node in geometry["nodes"]:
        _draw_node(ax, node)

    _draw_zone_labels(ax, geometry, nodes)

    xs = [float(node["x"]) for node in geometry["nodes"]]
    ys = [float(node["y"]) for node in geometry["nodes"]]
    pad = 0.75
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)

    ax.set_title(f"{geometry['title']} ({geometry['env_id']})", fontsize=11)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    notes = geometry.get("notes", [])
    if notes:
        fig.text(0.02, 0.02, " / ".join(notes), fontsize=7, ha="left", va="bottom")

    fig.tight_layout(rect=(0, 0.05, 1, 1))

    out_path = out_dir / f"Figure_EnvGeometry_{geometry['env_id']}.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return out_path


def render_geometries(
    output_dir: str | Path,
    *,
    env_ids: Iterable[str] | None = None,
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
    dpi: int = 200,
) -> Dict[str, Any]:
    """Render selected geometries and write manifest + registry export."""

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rendered: List[Dict[str, str]] = []
    for geometry in iter_geometries(env_ids=env_ids, path=registry_path):
        fig_path = render_geometry(geometry, out_dir, dpi=dpi)
        rendered.append(
            {
                "env_id": geometry["env_id"],
                "stage": geometry["stage"],
                "task_family": geometry["task_family"],
                "figure": str(fig_path),
            }
        )

    registry_export = write_registry_export(
        out_dir / "env_geometry_registry_export.json",
        path=registry_path,
    )

    manifest = {
        "schema_version": "env_geometry_render_manifest_v1",
        "created_at": datetime.now().isoformat(),
        "registry_path": str(Path(registry_path)),
        "registry_export": str(registry_export),
        "output_dir": str(out_dir),
        "rendered": rendered,
    }

    manifest_path = out_dir / "env_geometry_render_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    manifest["manifest_path"] = str(manifest_path)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render static environment schematics from env_geometry registry."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="logs/env_geometry",
        help="Directory for rendered figures and registry export.",
    )
    parser.add_argument(
        "--env-id",
        action="append",
        default=None,
        help="Environment ID to render. Can be passed multiple times. Default: all.",
    )
    parser.add_argument(
        "--registry",
        type=str,
        default=str(DEFAULT_REGISTRY_PATH),
        help="Path to geometry registry JSON.",
    )
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available env_id values and exit.",
    )

    args = parser.parse_args()

    if args.list:
        for env_id in list_geometries(args.registry):
            print(env_id)
        return

    if args.env_id:
        for env_id in args.env_id:
            get_geometry(env_id, args.registry)

    manifest = render_geometries(
        args.output_dir,
        env_ids=args.env_id,
        registry_path=args.registry,
        dpi=args.dpi,
    )

    print(f"Rendered {len(manifest['rendered'])} environment schematic(s)")
    print(f"Manifest: {manifest['manifest_path']}")


if __name__ == "__main__":
    main()