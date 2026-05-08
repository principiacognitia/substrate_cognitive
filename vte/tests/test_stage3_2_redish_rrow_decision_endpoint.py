from pathlib import Path

import pandas as pd

from vte.lab_adapters.redish_rrow_2022.build_decision_endpoint import (
    build_decision_endpoint,
)


def _write_raw_endpoint(path: Path) -> None:
    pd.DataFrame(
        [
            {
                "dataset_id": "redish_rrow_2022",
                "subject_id": "R001",
                "session_id": "R001-2020-01-01",
                "row_index": 0,
                "zone_id": "OfferZone",
                "choice_point_id": "OfferZone",
                "choice": "Skip",
                "trial": "[1.0, NaN, 3.0, 4.0]",
                "zone_delay": "[5.0, NaN, 15.0, 20.0]",
                "pause_time": "[0.5, NaN, 0.7, 0.8]",
                "total_site_time": "[2.0, NaN, 3.0, 4.0]",
                "run_speed": "[1.0, NaN, 1.1, 1.2]",
                "lab_idphi": "[0.1, NaN, 0.3, 0.4]",
                "lab_avg_dphi": "[0.01, NaN, 0.03, 0.04]",
            },
            {
                "dataset_id": "redish_rrow_2022",
                "subject_id": "R001",
                "session_id": "R001-2020-01-01",
                "row_index": 1,
                "zone_id": "WaitZone",
                "choice_point_id": "WaitZone",
                "choice": "Earn",
                "trial": "[5.0, 6.0]",
                "zone_delay": "[10.0, 11.0]",
                "pause_time": "[1.5, 1.6]",
                "total_site_time": "[6.0, 6.5]",
                "run_speed": "[0.8, 0.9]",
                "lab_idphi": "[0.5, 0.6]",
                "lab_avg_dphi": "[0.05, 0.06]",
            },
            {
                "dataset_id": "redish_rrow_2022",
                "subject_id": "R001",
                "session_id": "R001-2020-01-01",
                "row_index": 2,
                "zone_id": "WaitZone",
                "choice_point_id": "WaitZone",
                "choice": "Quit",
                "trial": "[7.0]",
                "zone_delay": "[12.0]",
                "pause_time": "[1.7]",
                "total_site_time": "[4.0]",
                "run_speed": "[0.7]",
                "lab_idphi": "[0.7]",
                "lab_avg_dphi": "[0.07]",
            },
            {
                "dataset_id": "redish_rrow_2022",
                "subject_id": "R001",
                "session_id": "R001-2020-01-01",
                "row_index": 3,
                "zone_id": "WaitZone",
                "choice_point_id": "WaitZone",
                "choice": "Skip",
                "trial": "[8.0]",
                "zone_delay": "[13.0]",
                "pause_time": "[1.8]",
                "total_site_time": "[2.0]",
                "run_speed": "[0.6]",
                "lab_idphi": "[0.8]",
                "lab_avg_dphi": "[0.08]",
            },
        ]
    ).to_csv(path, index=False)


def test_build_decision_endpoint_writes_outputs(tmp_path: Path):
    input_csv = tmp_path / "raw_endpoint.csv"
    output_dir = tmp_path / "decision"
    _write_raw_endpoint(input_csv)

    meta = build_decision_endpoint(input_csv=input_csv, output_dir=output_dir)

    assert meta["n_source_rows"] == 4
    assert meta["n_endpoint_rows"] == 7
    assert meta["n_usable_rows"] == 6

    assert (output_dir / "Table_Redish_RRow_Decision_Endpoint.csv").exists()
    assert (output_dir / "Table_Redish_RRow_Decision_Endpoint_Usable.csv").exists()
    assert (output_dir / "Table_Redish_RRow_Decision_By_Stage.csv").exists()
    assert (output_dir / "Table_Redish_RRow_Decision_By_Delay.csv").exists()
    assert (output_dir / "Table_Redish_RRow_Decision_By_Session.csv").exists()
    assert (output_dir / "redish_rrow_decision_endpoint_meta.json").exists()
    assert (output_dir / "Redish_RRow_Decision_Endpoint_Report.md").exists()


def test_decision_endpoint_preserves_author_semantics(tmp_path: Path):
    input_csv = tmp_path / "raw_endpoint.csv"
    output_dir = tmp_path / "decision"
    _write_raw_endpoint(input_csv)

    build_decision_endpoint(input_csv=input_csv, output_dir=output_dir)

    endpoint = pd.read_csv(output_dir / "Table_Redish_RRow_Decision_Endpoint.csv")

    skip = endpoint.loc[
        (endpoint["decision_stage"] == "offer_zone")
        & (endpoint["restaurant_outcome"] == "skip")
    ].iloc[0]

    assert skip["oz_choice"] == "skip"
    assert pd.isna(skip["wz_outcome"]) or skip["wz_outcome"] == ""
    assert skip["stage_decision"] == "skip"
    assert bool(skip["stage_applicable"]) is True
    assert skip["reward"] == 0

    earn = endpoint.loc[
        (endpoint["decision_stage"] == "wait_zone")
        & (endpoint["restaurant_outcome"] == "earn")
    ].iloc[0]

    assert earn["oz_choice"] == "accept"
    assert earn["wz_outcome"] == "earn"
    assert earn["stage_decision"] == "earn"
    assert bool(earn["stage_applicable"]) is True
    assert earn["reward"] == 1

    quit_row = endpoint.loc[
        (endpoint["decision_stage"] == "wait_zone")
        & (endpoint["restaurant_outcome"] == "quit")
    ].iloc[0]

    assert quit_row["oz_choice"] == "accept"
    assert quit_row["wz_outcome"] == "quit"
    assert quit_row["stage_decision"] == "quit"
    assert bool(quit_row["stage_applicable"]) is True
    assert quit_row["reward"] == 0


def test_wait_zone_skip_is_not_stage_applicable(tmp_path: Path):
    input_csv = tmp_path / "raw_endpoint.csv"
    output_dir = tmp_path / "decision"
    _write_raw_endpoint(input_csv)

    build_decision_endpoint(input_csv=input_csv, output_dir=output_dir)

    endpoint = pd.read_csv(output_dir / "Table_Redish_RRow_Decision_Endpoint.csv")
    usable = pd.read_csv(output_dir / "Table_Redish_RRow_Decision_Endpoint_Usable.csv")

    wz_skip = endpoint.loc[
        (endpoint["decision_stage"] == "wait_zone")
        & (endpoint["restaurant_outcome"] == "skip")
    ].iloc[0]

    assert bool(wz_skip["stage_applicable"]) is False
    assert wz_skip["restaurant_visit_id"] not in set(usable["restaurant_visit_id"])


def test_restaurant_id_uses_four_restaurants_not_eight_slots(tmp_path: Path):
    input_csv = tmp_path / "raw_endpoint.csv"
    output_dir = tmp_path / "decision"
    _write_raw_endpoint(input_csv)

    build_decision_endpoint(input_csv=input_csv, output_dir=output_dir)

    endpoint = pd.read_csv(output_dir / "Table_Redish_RRow_Decision_Endpoint.csv")

    assert set(endpoint["restaurant_id"]).issubset({1, 2, 3, 4})

    row = endpoint.loc[
        (endpoint["row_index"] == 0)
        & (endpoint["source_zone_slot"] == 3)
    ].iloc[0]

    assert row["restaurant_id"] == 4


def test_decision_endpoint_numeric_fields_are_available(tmp_path: Path):
    input_csv = tmp_path / "raw_endpoint.csv"
    output_dir = tmp_path / "decision"
    _write_raw_endpoint(input_csv)

    build_decision_endpoint(input_csv=input_csv, output_dir=output_dir)

    usable = pd.read_csv(output_dir / "Table_Redish_RRow_Decision_Endpoint_Usable.csv")

    for col in ["offer_delay_s", "pause_time_s", "lab_idphi", "lab_avg_dphi"]:
        assert col in usable.columns
        assert usable[col].notna().all()

    assert set(usable["restaurant_outcome"]) == {"skip", "earn", "quit"}