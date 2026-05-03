import json
from pathlib import Path

from env_geometry.registry import (
    DEFAULT_REGISTRY_PATH,
    get_geometry,
    list_geometries,
    load_registry,
)
from env_geometry.render_static import render_geometries


def test_builtin_registry_loads_and_has_expected_envs():
    registry = load_registry()

    assert registry["schema_version"] == "env_geometry_registry_v1"

    env_ids = set(list_geometries())
    assert "stage2_twostep_daw2011" in env_ids
    assert "stage2_reversal" in env_ids
    assert "stage3_open_covered_choice" in env_ids


def test_stage3_geometry_has_open_and_covered_routes():
    geometry = get_geometry("stage3_open_covered_choice")

    route_ids = {route["id"] for route in geometry["route_labels"]}
    assert {"open", "covered"}.issubset(route_ids)

    node_ids = {node["id"] for node in geometry["nodes"]}
    assert {"junction", "open_path", "covered_path"}.issubset(node_ids)


def test_stage2_twostep_has_common_and_rare_edges():
    geometry = get_geometry("stage2_twostep_daw2011")

    edge_kinds = {edge["kind"] for edge in geometry["edges"]}
    assert "common_transition" in edge_kinds
    assert "rare_transition" in edge_kinds


def test_render_geometries_writes_figures_and_manifest(tmp_path):
    manifest = render_geometries(
        tmp_path,
        env_ids=["stage2_reversal", "stage3_open_covered_choice"],
        registry_path=DEFAULT_REGISTRY_PATH,
        dpi=100,
    )

    assert len(manifest["rendered"]) == 2

    manifest_path = Path(manifest["manifest_path"])
    assert manifest_path.exists()

    with manifest_path.open("r", encoding="utf-8") as f:
        loaded = json.load(f)

    assert loaded["schema_version"] == "env_geometry_render_manifest_v1"

    for item in loaded["rendered"]:
        assert Path(item["figure"]).exists()

    assert (tmp_path / "env_geometry_registry_export.json").exists()