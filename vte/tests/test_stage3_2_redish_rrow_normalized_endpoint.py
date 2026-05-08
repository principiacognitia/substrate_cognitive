from pathlib import Path

import pandas as pd

from vte.lab_adapters.redish_rrow_2022.normalize_processed_behavior_endpoint import (
    normalize_endpoint_table,
)


def test_normalize_endpoint_table_explodes_vector_cells(tmp_path: Path):
    input_csv = tmp_path / "endpoint.csv"
    output_dir = tmp_path / "normalized"

    pd.DataFrame(
        [
            {
                "dataset_id": "redish_rrow_2022",
                "source_version": "Version 002",
                "trace_origin": "biological",
                "adapter": "redish_rrow_2022_processed_behavior",
                "subject_id": "R001",
                "session_id": "R001-2020-01-01",
                "session_date": "2020-01-01",
                "session_index": 0,
                "row_index": 0,
                "trial": "[1.0, 2.0, NaN]",
                "choice_point_id": "WaitZone",
                "zone_id": "[1.0, 2.0, NaN]",
                "zone_type": "",
                "choice": "Skip",
                "reward": "",
                "zone_delay": "[5.0, 10.0, NaN]",
                "total_site_time": "[1.5, 2.5, NaN]",
                "pause_time": "[0.2, 0.3, NaN]",
                "run_speed": "[20.0, 18.0, NaN]",
                "lab_idphi": "[0.1, 0.2, NaN]",
                "lab_avg_dphi": "[0.01, 0.02, NaN]",
            }
        ]
    ).to_csv(input_csv, index=False)

    meta = normalize_endpoint_table(input_csv=input_csv, output_dir=output_dir)

    assert meta["n_source_rows"] == 1
    assert meta["n_normalized_rows"] == 2

    normalized = pd.read_csv(output_dir / "Table_Redish_RRow_Choice_IdPhi_Endpoint_Normalized.csv")

    assert list(normalized["zone_slot"]) == [0, 1]
    assert list(normalized["trial"]) == [1, 2]
    assert list(normalized["zone_id"]) == [1, 2]
    assert list(normalized["zone_delay"]) == [5.0, 10.0]
    assert list(normalized["choice"]) == ["skip", "skip"]
    assert list(normalized["zone_type"]) == ["wait_zone", "wait_zone"]
    assert list(normalized["lab_idphi"]) == [0.1, 0.2]


def test_normalize_endpoint_table_keeps_scalar_rows(tmp_path: Path):
    input_csv = tmp_path / "endpoint.csv"
    output_dir = tmp_path / "normalized"

    pd.DataFrame(
        [
            {
                "dataset_id": "redish_rrow_2022",
                "subject_id": "R001",
                "session_id": "R001-2020-01-01",
                "trial": 1,
                "choice_point_id": "OfferZone",
                "zone_id": 5,
                "choice": "Earn",
                "reward": 1,
                "zone_delay": 12,
                "lab_idphi": 0.4,
                "lab_avg_dphi": 0.2,
            }
        ]
    ).to_csv(input_csv, index=False)

    normalize_endpoint_table(input_csv=input_csv, output_dir=output_dir)

    normalized = pd.read_csv(output_dir / "Table_Redish_RRow_Choice_IdPhi_Endpoint_Normalized.csv")

    assert len(normalized) == 1
    assert normalized.iloc[0]["choice"] == "accept"
    assert normalized.iloc[0]["zone_type"] == "offer_zone"
    assert normalized.iloc[0]["reward"] == 1

def test_normalize_endpoint_table_drops_blank_trial_slots_and_uses_zone_context(tmp_path: Path):
    input_csv = tmp_path / "endpoint.csv"
    output_dir = tmp_path / "normalized"

    pd.DataFrame(
        [
            {
                "dataset_id": "redish_rrow_2022",
                "subject_id": "R001",
                "session_id": "R001-2020-01-01",
                "row_index": 0,
                "trial": "[NaN, 2.0, 3.0]",
                "choice_point_id": "WaitZone",
                "zone_id": "WaitZone",
                "choice": "Skip",
                "zone_delay": "[NaN, 10.0, 20.0]",
                "pause_time": "[NaN, 0.4, 0.5]",
                "lab_idphi": "[NaN, 0.2, 0.3]",
            }
        ]
    ).to_csv(input_csv, index=False)

    normalize_endpoint_table(input_csv=input_csv, output_dir=output_dir)

    normalized = pd.read_csv(output_dir / "Table_Redish_RRow_Choice_IdPhi_Endpoint_Normalized.csv")

    assert len(normalized) == 2
    assert list(normalized["trial"]) == [2, 3]
    assert list(normalized["zone_id"]) == [2, 3]
    assert list(normalized["restaurant_id"]) == [2, 3]
    assert list(normalized["source_zone_context"]) == ["WaitZone", "WaitZone"]
    assert list(normalized["zone_type"]) == ["wait_zone", "wait_zone"]
    assert list(normalized["choice"]) == ["skip", "skip"]    