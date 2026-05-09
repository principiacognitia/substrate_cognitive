from pathlib import Path

import pandas as pd

from vte.lab_adapters.redish_rrow_2022.build_biological_synthetic_comparability import (
    build_biological_synthetic_comparability,
)


def _write_biological_csv(path: Path) -> Path:
    df = pd.DataFrame(
        [
            {
                "dataset_id": "redish_rrow_2022",
                "trace_origin": "biological",
                "task_family": "restaurant_row",
                "subject_id": "R1",
                "session_id": "R1-2020-01-01",
                "trial": 1,
                "decision_stage": "offer_zone",
                "restaurant_id": 1,
                "chosen_action": "skip",
                "outcome": "skip",
                "reward": 0,
                "cost": 2,
                "delay_bin": "delay_00_04",
                "dwell_proxy": 1.0,
                "deliberation_proxy": 10.0,
                "z_pause_time_by_session_stage": -1.0,
                "z_lab_idphi_by_session_stage": -1.2,
            },
            {
                "dataset_id": "redish_rrow_2022",
                "trace_origin": "biological",
                "task_family": "restaurant_row",
                "subject_id": "R1",
                "session_id": "R1-2020-01-01",
                "trial": 2,
                "decision_stage": "offer_zone",
                "restaurant_id": 2,
                "chosen_action": "accept",
                "outcome": "earn",
                "reward": 1,
                "cost": 28,
                "delay_bin": "delay_25_plus",
                "dwell_proxy": 4.0,
                "deliberation_proxy": 40.0,
                "z_pause_time_by_session_stage": 1.0,
                "z_lab_idphi_by_session_stage": 1.3,
            },
            {
                "dataset_id": "redish_rrow_2022",
                "trace_origin": "biological",
                "task_family": "restaurant_row",
                "subject_id": "R2",
                "session_id": "R2-2020-01-01",
                "trial": 1,
                "decision_stage": "wait_zone",
                "restaurant_id": 1,
                "chosen_action": "quit",
                "outcome": "quit",
                "reward": 0,
                "cost": 30,
                "delay_bin": "delay_25_plus",
                "dwell_proxy": 2.0,
                "deliberation_proxy": 20.0,
                "z_pause_time_by_session_stage": 0.5,
                "z_lab_idphi_by_session_stage": 0.7,
            },
        ]
    )
    df.to_csv(path, index=False)
    return path


def _write_synthetic_csv(path: Path) -> Path:
    df = pd.DataFrame(
        [
            {
                "run_id": "synthetic_run_1",
                "seed": "seed_1",
                "trial": 1,
                "decision_stage": "offer_zone",
                "condition": "low",
                "cost": 1,
                "committed_path": "skip",
                "total_reward": 0,
                "raw_idphi": 3.0,
                "z_idphi": -0.8,
                "pause_ticks": 2,
                "vte_binary": 0,
            },
            {
                "run_id": "synthetic_run_1",
                "seed": "seed_1",
                "trial": 2,
                "decision_stage": "offer_zone",
                "condition": "high",
                "cost": 30,
                "committed_path": "accept",
                "total_reward": 1,
                "raw_idphi": 9.0,
                "z_idphi": 1.1,
                "pause_ticks": 6,
                "vte_binary": 1,
            },
            {
                "run_id": "synthetic_run_2",
                "seed": "seed_2",
                "trial": 1,
                "decision_stage": "wait_zone",
                "condition": "high",
                "cost": 25,
                "committed_path": "quit",
                "total_reward": 0,
                "raw_idphi": 8.0,
                "z_idphi": 0.9,
                "pause_ticks": 5,
                "vte_binary": 1,
            },
        ]
    )
    df.to_csv(path, index=False)
    return path


def test_build_biosynth_comparability_writes_outputs(tmp_path):
    biological = _write_biological_csv(tmp_path / "bio.csv")
    synthetic = _write_synthetic_csv(tmp_path / "synthetic.csv")
    output_dir = tmp_path / "comparability"

    meta = build_biological_synthetic_comparability(
        biological_csv=biological,
        synthetic_csvs=[synthetic],
        output_dir=output_dir,
    )

    assert meta["n_comparable_rows"] == 6
    assert meta["n_biological_rows"] == 3
    assert meta["n_synthetic_rows"] == 3

    assert (output_dir / "Table_BioSynth_Comparable_Decision_Rows.csv").exists()
    assert (output_dir / "Table_BioSynth_Summary_By_Source_Stage_Cost.csv").exists()
    assert (output_dir / "Table_BioSynth_Summary_By_Source_Choice.csv").exists()
    assert (output_dir / "Table_BioSynth_Direction_Agreement.csv").exists()
    assert (output_dir / "Table_BioSynth_Field_Coverage.csv").exists()
    assert (output_dir / "biosynth_comparability_meta.json").exists()
    assert (output_dir / "BioSynth_Comparability_Report.md").exists()


def test_comparable_rows_have_expected_sources_and_cost_bins(tmp_path):
    biological = _write_biological_csv(tmp_path / "bio.csv")
    synthetic = _write_synthetic_csv(tmp_path / "synthetic.csv")
    output_dir = tmp_path / "comparability"

    build_biological_synthetic_comparability(
        biological_csv=biological,
        synthetic_csvs=[synthetic],
        output_dir=output_dir,
    )

    rows = pd.read_csv(output_dir / "Table_BioSynth_Comparable_Decision_Rows.csv")

    assert set(rows["source"]) == {"biological", "synthetic"}
    assert {"low", "high"}.issubset(set(rows["comparable_cost_bin"]))
    assert "deliberation_z" in rows.columns
    assert "dwell_z" in rows.columns


def test_direction_agreement_detects_high_minus_low_effect(tmp_path):
    biological = _write_biological_csv(tmp_path / "bio.csv")
    synthetic = _write_synthetic_csv(tmp_path / "synthetic.csv")
    output_dir = tmp_path / "comparability"

    build_biological_synthetic_comparability(
        biological_csv=biological,
        synthetic_csvs=[synthetic],
        output_dir=output_dir,
    )

    direction = pd.read_csv(output_dir / "Table_BioSynth_Direction_Agreement.csv")
    offer_idphi = direction[
        (direction["decision_stage"] == "offer_zone")
        & (direction["metric"] == "deliberation_z")
    ].iloc[0]

    assert offer_idphi["biological_high_minus_low"] > 0
    assert offer_idphi["synthetic_high_minus_low"] > 0
    assert str(offer_idphi["direction_agreement"]).lower() in {"true", "1.0", "1"}