from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from vte.lab_adapters.redish_rrow_2022.convert_decision_endpoint_to_canonical import (
    CANONICAL_DECISION_COLUMNS,
    convert_decision_endpoint_to_canonical,
)


def _write_decision_endpoint_fixture(path: Path) -> None:
    rows = [
        {
            "dataset_id": "redish_rrow_2022",
            "subject_id": "R001",
            "session_id": "R001-2020-01-01",
            "restaurant_visit_id": "R001-2020-01-01_r00001_s0",
            "row_index": "1",
            "trial": "1",
            "source_zone_slot": "0",
            "source_slot_policy": "first_four_decision_slots",
            "restaurant_id": "1",
            "decision_stage": "offer_zone",
            "stage_decision": "accept",
            "stage_applicable": "True",
            "restaurant_outcome": "earn",
            "reward": "1.0",
            "choice_point_id": "redish_rrow_restaurant_1_offer_zone",
            "offer_delay_s": "5.0",
            "pause_time_s": "1.0",
            "total_site_time_s": "4.0",
            "run_speed": "12.0",
            "lab_idphi": "1.0",
            "lab_avg_dphi": "0.1",
        },
        {
            "dataset_id": "redish_rrow_2022",
            "subject_id": "R001",
            "session_id": "R001-2020-01-01",
            "restaurant_visit_id": "R001-2020-01-01_r00002_s1",
            "row_index": "2",
            "trial": "2",
            "source_zone_slot": "1",
            "source_slot_policy": "first_four_decision_slots",
            "restaurant_id": "2",
            "decision_stage": "offer_zone",
            "stage_decision": "skip",
            "stage_applicable": "True",
            "restaurant_outcome": "skip",
            "reward": "0.0",
            "choice_point_id": "redish_rrow_restaurant_2_offer_zone",
            "offer_delay_s": "10.0",
            "pause_time_s": "2.0",
            "total_site_time_s": "5.0",
            "run_speed": "13.0",
            "lab_idphi": "3.0",
            "lab_avg_dphi": "0.3",
        },
        {
            "dataset_id": "redish_rrow_2022",
            "subject_id": "R001",
            "session_id": "R001-2020-01-01",
            "restaurant_visit_id": "R001-2020-01-01_r00003_s2",
            "row_index": "3",
            "trial": "3",
            "source_zone_slot": "2",
            "source_slot_policy": "first_four_decision_slots",
            "restaurant_id": "3",
            "decision_stage": "wait_zone",
            "stage_decision": "earn",
            "stage_applicable": "True",
            "restaurant_outcome": "earn",
            "reward": "1.0",
            "choice_point_id": "redish_rrow_restaurant_3_wait_zone",
            "offer_delay_s": "15.0",
            "pause_time_s": "3.0",
            "total_site_time_s": "6.0",
            "run_speed": "14.0",
            "lab_idphi": "5.0",
            "lab_avg_dphi": "0.5",
        },
        {
            "dataset_id": "redish_rrow_2022",
            "subject_id": "R001",
            "session_id": "R001-2020-01-01",
            "restaurant_visit_id": "R001-2020-01-01_r00004_s3",
            "row_index": "4",
            "trial": "4",
            "source_zone_slot": "3",
            "source_slot_policy": "first_four_decision_slots",
            "restaurant_id": "4",
            "decision_stage": "wait_zone",
            "stage_decision": "quit",
            "stage_applicable": "True",
            "restaurant_outcome": "quit",
            "reward": "0.0",
            "choice_point_id": "redish_rrow_restaurant_4_wait_zone",
            "offer_delay_s": "25.0",
            "pause_time_s": "4.0",
            "total_site_time_s": "7.0",
            "run_speed": "15.0",
            "lab_idphi": "7.0",
            "lab_avg_dphi": "0.7",
        },
        {
            "dataset_id": "redish_rrow_2022",
            "subject_id": "R001",
            "session_id": "R001-2020-01-01",
            "restaurant_visit_id": "R001-2020-01-01_r00005_s0",
            "row_index": "5",
            "trial": "5",
            "source_zone_slot": "0",
            "source_slot_policy": "first_four_decision_slots",
            "restaurant_id": "1",
            "decision_stage": "offer_zone",
            "stage_decision": "",
            "stage_applicable": "False",
            "restaurant_outcome": "",
            "reward": "",
            "choice_point_id": "redish_rrow_restaurant_1_offer_zone",
            "offer_delay_s": "",
            "pause_time_s": "",
            "total_site_time_s": "",
            "run_speed": "",
            "lab_idphi": "",
            "lab_avg_dphi": "",
        },
    ]

    pd.DataFrame(rows).to_csv(path, index=False)


def test_convert_decision_endpoint_to_canonical_writes_outputs(tmp_path: Path):
    input_csv = tmp_path / "Table_Redish_RRow_Decision_Endpoint_Usable.csv"
    output_dir = tmp_path / "canonical"
    _write_decision_endpoint_fixture(input_csv)

    meta = convert_decision_endpoint_to_canonical(
        input_csv=input_csv,
        output_dir=output_dir,
    )

    assert meta["n_input_rows"] == 5
    assert meta["n_filtered_rows"] == 4
    assert meta["n_canonical_rows"] == 4

    assert (output_dir / "redish_rrow_canonical_decision_endpoint.csv").exists()
    assert (output_dir / "Table_Redish_RRow_Canonical_By_Stage.csv").exists()
    assert (output_dir / "Table_Redish_RRow_Canonical_By_Delay.csv").exists()
    assert (output_dir / "Table_Redish_RRow_Canonical_By_Subject.csv").exists()
    assert (output_dir / "Table_Redish_RRow_Canonical_By_Session.csv").exists()
    assert (output_dir / "redish_rrow_canonical_decision_endpoint_meta.json").exists()
    assert (output_dir / "Redish_RRow_Canonical_Decision_Endpoint_Report.md").exists()


def test_canonical_decision_endpoint_has_required_columns(tmp_path: Path):
    input_csv = tmp_path / "input.csv"
    output_dir = tmp_path / "canonical"
    _write_decision_endpoint_fixture(input_csv)

    convert_decision_endpoint_to_canonical(input_csv=input_csv, output_dir=output_dir)

    canonical = pd.read_csv(output_dir / "redish_rrow_canonical_decision_endpoint.csv")

    for col in CANONICAL_DECISION_COLUMNS:
        assert col in canonical.columns

    assert set(canonical["trace_origin"]) == {"biological"}
    assert set(canonical["task_family"]) == {"restaurant_row"}
    assert set(canonical["comparability_level"]) == {"decision_endpoint_proxy"}
    assert set(canonical["pose_source"]) == {"none_decision_endpoint"}
    assert set(canonical["wrapper_compatibility_note"].str.contains("Not a pose trace")) == {
        True
    }


def test_canonical_decision_endpoint_maps_decision_fields(tmp_path: Path):
    input_csv = tmp_path / "input.csv"
    output_dir = tmp_path / "canonical"
    _write_decision_endpoint_fixture(input_csv)

    convert_decision_endpoint_to_canonical(input_csv=input_csv, output_dir=output_dir)

    canonical = pd.read_csv(output_dir / "redish_rrow_canonical_decision_endpoint.csv")

    first = canonical.loc[canonical["trial"] == 1].iloc[0]
    assert first["decision_stage"] == "offer_zone"
    assert first["committed_path"] == "accept"
    assert first["chosen_action"] == "accept"
    assert first["choice"] == "accept"
    assert first["restaurant_outcome"] == "earn"
    assert first["outcome"] == "earn"
    assert first["reward"] == 1.0
    assert first["cost"] == 5.0
    assert first["dwell_proxy"] == 1.0
    assert first["deliberation_proxy"] == 1.0
    assert first["at_choice_point"] in {True, "True", "true", 1}


def test_canonical_decision_endpoint_filters_inapplicable_rows(tmp_path: Path):
    input_csv = tmp_path / "input.csv"
    output_dir = tmp_path / "canonical"
    _write_decision_endpoint_fixture(input_csv)

    convert_decision_endpoint_to_canonical(input_csv=input_csv, output_dir=output_dir)

    canonical = pd.read_csv(output_dir / "redish_rrow_canonical_decision_endpoint.csv")

    assert set(canonical["trial"]) == {1, 2, 3, 4}
    assert "" not in set(canonical["chosen_action"].astype(str))


def test_canonical_decision_endpoint_writes_stage_summary(tmp_path: Path):
    input_csv = tmp_path / "input.csv"
    output_dir = tmp_path / "canonical"
    _write_decision_endpoint_fixture(input_csv)

    convert_decision_endpoint_to_canonical(input_csv=input_csv, output_dir=output_dir)

    by_stage = pd.read_csv(output_dir / "Table_Redish_RRow_Canonical_By_Stage.csv")

    grouped = {
        (row.decision_stage, row.chosen_action, row.restaurant_outcome): row.n_rows
        for row in by_stage.itertuples(index=False)
    }

    assert grouped[("offer_zone", "accept", "earn")] == 1
    assert grouped[("offer_zone", "skip", "skip")] == 1
    assert grouped[("wait_zone", "earn", "earn")] == 1
    assert grouped[("wait_zone", "quit", "quit")] == 1


def test_canonical_decision_endpoint_metadata_is_json(tmp_path: Path):
    input_csv = tmp_path / "input.csv"
    output_dir = tmp_path / "canonical"
    _write_decision_endpoint_fixture(input_csv)

    convert_decision_endpoint_to_canonical(input_csv=input_csv, output_dir=output_dir)

    meta_path = output_dir / "redish_rrow_canonical_decision_endpoint_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    assert meta["methodological_status"] == (
        "decision-level biological comparability endpoint; not a pose trace"
    )
    assert meta["outputs"]["canonical"].endswith(
        "redish_rrow_canonical_decision_endpoint.csv"
    )