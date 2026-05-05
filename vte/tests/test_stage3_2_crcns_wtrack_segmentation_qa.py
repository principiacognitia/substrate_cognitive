from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from vte.lab_adapters.crcns_wtrack.validate_wtrack_segmentation import (
    build_choice_exit_diagnostics,
    build_commit_rule_stability,
    build_route_usage_by_epoch,
    load_geometry,
    validate_wtrack_segmentation,
)


def _write_geometry(path: Path) -> Path:
    geometry = {
        "geometry_id": "test_wtrack_geometry",
        "zones": [
            {
                "zone_id": "wtrack_junction",
                "zone_type": "choice",
                "center": [0.0, 0.0],
                "radius": 1.0,
            },
            {
                "zone_id": "arm_a",
                "zone_type": "route",
                "center": [5.0, 0.0],
                "radius": 2.0,
            },
            {
                "zone_id": "arm_b",
                "zone_type": "route",
                "center": [-5.0, 0.0],
                "radius": 2.0,
            },
        ],
    }
    path.write_text(json.dumps(geometry), encoding="utf-8")
    return path


def _canonical_rows() -> list[dict]:
    rows = []

    # Trial 1: exits to arm_a.
    for source_row_index, x, y in [
        (10, 0.0, 0.0),
        (11, 0.2, 0.0),
        (12, 0.8, 0.0),
        (13, 1.4, 0.0),
        (14, 3.0, 0.0),
        (15, 5.0, 0.0),
    ]:
        rows.append(
            {
                "dataset_id": "crcns_hc6",
                "animal_id": "Fiv",
                "run_id": "crcns_hc6_Fiv_day01",
                "day": 1,
                "epoch": 2,
                "source_row_index": source_row_index,
                "sample_index": source_row_index,
                "x": x,
                "y": y,
                "heading": 0.0,
            }
        )

    # Trial 2: exits to arm_b.
    for source_row_index, x, y in [
        (20, 0.0, 0.0),
        (21, -0.2, 0.0),
        (22, -0.8, 0.0),
        (23, -1.4, 0.0),
        (24, -3.0, 0.0),
        (25, -5.0, 0.0),
    ]:
        rows.append(
            {
                "dataset_id": "crcns_hc6",
                "animal_id": "Fiv",
                "run_id": "crcns_hc6_Fiv_day01",
                "day": 1,
                "epoch": 2,
                "source_row_index": source_row_index,
                "sample_index": source_row_index,
                "x": x,
                "y": y,
                "heading": 3.14159,
            }
        )

    return rows


def _segmented_rows() -> list[dict]:
    rows = []

    for source_row_index, x in [(10, 0.0), (11, 0.2), (12, 0.8)]:
        rows.append(
            {
                "dataset_id": "crcns_hc6",
                "animal_id": "Fiv",
                "run_id": "crcns_hc6_Fiv_day01",
                "day": 1,
                "epoch": 2,
                "trial": 1,
                "choice_visit_index": 1,
                "source_row_index": source_row_index,
                "sample_index": source_row_index,
                "x": x,
                "y": 0.0,
                "choice_point_id": "wtrack_junction",
                "at_choice_point": True,
                "committed_path": "arm_a",
            }
        )

    for source_row_index, x in [(20, 0.0), (21, -0.2), (22, -0.8)]:
        rows.append(
            {
                "dataset_id": "crcns_hc6",
                "animal_id": "Fiv",
                "run_id": "crcns_hc6_Fiv_day01",
                "day": 1,
                "epoch": 2,
                "trial": 2,
                "choice_visit_index": 2,
                "source_row_index": source_row_index,
                "sample_index": source_row_index,
                "x": x,
                "y": 0.0,
                "choice_point_id": "wtrack_junction",
                "at_choice_point": True,
                "committed_path": "arm_b",
            }
        )

    return rows


def test_load_geometry_accepts_choice_and_route_zones(tmp_path):
    geometry_json = _write_geometry(tmp_path / "geometry.json")

    geometry = load_geometry(geometry_json)

    assert geometry.geometry_id == "test_wtrack_geometry"
    assert geometry.choice_zone.zone_id == "wtrack_junction"
    assert {z.zone_id for z in geometry.route_zones} == {"arm_a", "arm_b"}


def test_route_usage_by_epoch_keeps_observed_route_imbalance():
    segmented = pd.DataFrame(_segmented_rows())

    usage = build_route_usage_by_epoch(segmented)

    assert set(usage["committed_path"]) == {"arm_a", "arm_b"}
    assert usage["n_trials"].sum() == 2
    assert set(usage["route_rate"]) == {0.5}


def test_choice_exit_diagnostics_recomputes_commit_rules(tmp_path):
    geometry = load_geometry(_write_geometry(tmp_path / "geometry.json"))
    canonical = pd.DataFrame(_canonical_rows())
    segmented = pd.DataFrame(_segmented_rows())

    diagnostics = build_choice_exit_diagnostics(
        canonical,
        segmented,
        geometry,
        post_window_samples=10,
    )

    assert len(diagnostics) == 2

    t1 = diagnostics.loc[diagnostics["trial"] == 1].iloc[0]
    assert t1["existing_committed_path"] == "arm_a"
    assert t1["first_non_choice_route"] == "arm_a"
    assert t1["dominant_post_route"] == "arm_a"
    assert t1["farthest_post_route"] == "arm_a"
    assert t1["rule_agreement_count"] == 3

    t2 = diagnostics.loc[diagnostics["trial"] == 2].iloc[0]
    assert t2["existing_committed_path"] == "arm_b"
    assert t2["first_non_choice_route"] == "arm_b"
    assert t2["dominant_post_route"] == "arm_b"
    assert t2["farthest_post_route"] == "arm_b"
    assert t2["rule_agreement_count"] == 3


def test_commit_rule_stability_reports_overall_and_epoch_rows(tmp_path):
    geometry = load_geometry(_write_geometry(tmp_path / "geometry.json"))
    canonical = pd.DataFrame(_canonical_rows())
    segmented = pd.DataFrame(_segmented_rows())

    diagnostics = build_choice_exit_diagnostics(
        canonical,
        segmented,
        geometry,
        post_window_samples=10,
    )
    stability = build_commit_rule_stability(diagnostics)

    assert set(stability["group"]) == {"overall", "epoch"}
    overall = stability.loc[stability["group"] == "overall"].iloc[0]
    assert overall["n_trials"] == 2
    assert overall["first_non_choice_match_rate"] == 1.0
    assert overall["dominant_post_match_rate"] == 1.0
    assert overall["farthest_post_match_rate"] == 1.0
    assert overall["unanimous_rule_rate"] == 1.0


def test_validate_wtrack_segmentation_writes_outputs(tmp_path):
    canonical_csv = tmp_path / "canonical.csv"
    segmented_csv = tmp_path / "segmented.csv"
    geometry_json = _write_geometry(tmp_path / "geometry.json")
    output_dir = tmp_path / "qa"

    pd.DataFrame(_canonical_rows()).to_csv(canonical_csv, index=False)
    pd.DataFrame(_segmented_rows()).to_csv(segmented_csv, index=False)

    meta = validate_wtrack_segmentation(
        canonical_trace_csv=canonical_csv,
        segmented_trace_csv=segmented_csv,
        geometry_json=geometry_json,
        output_dir=output_dir,
        post_window_samples=10,
    )

    assert meta["n_segmented_trials"] == 2

    expected = {
        "Table_CRCNS_WTrack_Route_Usage_By_Epoch.csv",
        "Table_CRCNS_WTrack_Choice_Exit_Diagnostics.csv",
        "Table_CRCNS_WTrack_Commit_Rule_Stability.csv",
        "Figure_CRCNS_WTrack_Route_Usage_By_Epoch.png",
        "Figure_CRCNS_WTrack_Commit_Rule_Stability.png",
        "crcns_wtrack_segmentation_qa_meta.json",
    }

    assert expected.issubset({p.name for p in output_dir.iterdir()})