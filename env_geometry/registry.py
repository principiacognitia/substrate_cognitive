"""Static environment-geometry registry.

This module is a service layer. It must not import stage2, stage3, or vte
runtime internals. Geometry is data, not simulator state.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_REGISTRY_PATH = PACKAGE_ROOT / "registries" / "builtin_env_geometries.json"

REQUIRED_GEOMETRY_KEYS = {
    "env_id",
    "stage",
    "task_family",
    "title",
    "coordinate_system",
    "nodes",
    "edges",
    "zones",
}

REQUIRED_NODE_KEYS = {"id", "label", "x", "y", "kind"}
REQUIRED_EDGE_KEYS = {"source", "target", "label", "kind"}


def load_registry(path: str | Path = DEFAULT_REGISTRY_PATH) -> Dict[str, Any]:
    """Load an environment-geometry registry JSON file."""

    registry_path = Path(path)
    if not registry_path.exists():
        raise FileNotFoundError(f"Geometry registry not found: {registry_path}")

    with registry_path.open("r", encoding="utf-8") as f:
        registry = json.load(f)

    validate_registry(registry)
    return registry


def validate_registry(registry: Dict[str, Any]) -> None:
    """Validate the minimal static registry schema."""

    if "schema_version" not in registry:
        raise ValueError("Geometry registry is missing schema_version")
    if "geometries" not in registry or not isinstance(registry["geometries"], list):
        raise ValueError("Geometry registry must contain a geometries list")

    seen_ids = set()
    for geometry in registry["geometries"]:
        missing = sorted(REQUIRED_GEOMETRY_KEYS - set(geometry))
        if missing:
            raise ValueError(f"Geometry is missing required keys: {missing}")

        env_id = geometry["env_id"]
        if env_id in seen_ids:
            raise ValueError(f"Duplicate geometry env_id: {env_id}")
        seen_ids.add(env_id)

        nodes = geometry["nodes"]
        if not nodes:
            raise ValueError(f"Geometry {env_id} has no nodes")

        node_ids = set()
        for node in nodes:
            missing_node = sorted(REQUIRED_NODE_KEYS - set(node))
            if missing_node:
                raise ValueError(f"Geometry {env_id} node is missing keys: {missing_node}")
            node_id = node["id"]
            if node_id in node_ids:
                raise ValueError(f"Geometry {env_id} has duplicate node id: {node_id}")
            node_ids.add(node_id)

        for edge in geometry["edges"]:
            missing_edge = sorted(REQUIRED_EDGE_KEYS - set(edge))
            if missing_edge:
                raise ValueError(f"Geometry {env_id} edge is missing keys: {missing_edge}")
            if edge["source"] not in node_ids:
                raise ValueError(f"Geometry {env_id} edge source not found: {edge['source']}")
            if edge["target"] not in node_ids:
                raise ValueError(f"Geometry {env_id} edge target not found: {edge['target']}")

        for zone in geometry["zones"]:
            for node_id in zone.get("node_ids", []):
                if node_id not in node_ids:
                    raise ValueError(
                        f"Geometry {env_id} zone {zone.get('id')} references missing node: {node_id}"
                    )


def list_geometries(path: str | Path = DEFAULT_REGISTRY_PATH) -> List[str]:
    """Return registered environment IDs."""

    registry = load_registry(path)
    return [g["env_id"] for g in registry["geometries"]]


def get_geometry(env_id: str, path: str | Path = DEFAULT_REGISTRY_PATH) -> Dict[str, Any]:
    """Return one geometry by environment ID."""

    registry = load_registry(path)
    for geometry in registry["geometries"]:
        if geometry["env_id"] == env_id:
            return geometry

    available = ", ".join(list_geometries(path))
    raise KeyError(f"Unknown env_id: {env_id}. Available: {available}")


def iter_geometries(
    env_ids: Iterable[str] | None = None,
    path: str | Path = DEFAULT_REGISTRY_PATH,
) -> Iterable[Dict[str, Any]]:
    """Iterate over selected geometries, or all geometries if env_ids is None."""

    registry = load_registry(path)
    selected = set(env_ids) if env_ids else None

    for geometry in registry["geometries"]:
        if selected is None or geometry["env_id"] in selected:
            yield geometry


def write_registry_export(
    output_path: str | Path,
    path: str | Path = DEFAULT_REGISTRY_PATH,
) -> Path:
    """Write a normalized copy of the registry for logs/results."""

    registry = load_registry(path)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)

    return out