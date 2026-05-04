from pathlib import Path

import numpy as np
from scipy.io import savemat

from vte.lab_adapters.crcns_wtrack.mat_loader import (
    infer_animal_prefix,
    infer_day_from_filename,
    infer_file_kind,
    list_animal_mat_files,
    read_mat,
    summarize_mat_top_level,
    top_level_keys,
)

def test_infer_crcns_filename_fields():
    assert infer_animal_prefix("Fiv/Fivpos01.mat") == "Fiv"
    assert infer_animal_prefix("Fiv/Fivrawpos09.mat") == "Fiv"
    assert infer_animal_prefix("Fiv/Fivtask02.mat") == "Fiv"
    assert infer_animal_prefix("Fiv/EEG/Fiveeg03-7-25.mat") == "Fiv"

    assert infer_day_from_filename("Fiv/Fivpos01.mat") == 1
    assert infer_day_from_filename("Fiv/Fivrawpos09.mat") == 9
    assert infer_day_from_filename("Fiv/EEG/Fiveeg03-7-25.mat") == 3
    assert infer_day_from_filename("Fiv/Fivcellinfo.mat") is None

    assert infer_file_kind("Fiv/Fivpos01.mat") == "pos"
    assert infer_file_kind("Fiv/Fivrawpos01.mat") == "rawpos"
    assert infer_file_kind("Fiv/Fivtask01.mat") == "task"
    assert infer_file_kind("Fiv/Fivspikes01.mat") == "spikes"
    assert infer_file_kind("Fiv/Fivcellinfo.mat") == "metadata"
    assert infer_file_kind("Fiv/EEG/Fiveeg01-1-01.mat") == "eeg"

def test_read_mat_returns_non_internal_keys(tmp_path):
    mat_path = tmp_path / "Fivpos01.mat"
    savemat(
        mat_path,
        {
            "pos": np.array(
                [
                    [0.0, 1.0, 2.0],
                    [1.0, 3.0, 4.0],
                ]
            )
        },
    )

    loaded = read_mat(mat_path)

    assert "pos" in loaded
    assert "__header__" not in loaded
    assert top_level_keys(loaded) == ["pos"]

def test_list_animal_mat_files_is_recursive(tmp_path):
    animal = tmp_path / "Fiv"
    eeg = animal / "EEG"
    eeg.mkdir(parents=True)

    (animal / "notes.txt").write_text("not a mat", encoding="utf-8")
    savemat(animal / "Fivpos01.mat", {"pos": np.array([[0.0, 1.0, 2.0]])})
    savemat(eeg / "Fiveeg01-1-01.mat", {"eeg": np.array([[1.0]])})

    files = list_animal_mat_files(animal)

    assert len(files) == 2
    assert {p.name for p in files} == {"Fivpos01.mat", "Fiveeg01-1-01.mat"}

def test_summarize_mat_top_level_can_skip_inspection(tmp_path):
    mat_path = tmp_path / "Fivpos01.mat"
    savemat(mat_path, {"pos": np.array([[0.0, 1.0, 2.0]])})

    inspected = summarize_mat_top_level(mat_path, inspect=True)
    skipped = summarize_mat_top_level(mat_path, inspect=False)

    assert inspected["file_kind"] == "pos"
    assert inspected["day"] == 1
    assert inspected["top_keys"] == "pos"
    assert inspected["top_key_count"] == 1

    assert skipped["inspected"] is False
    assert skipped["top_keys"] == ""
    assert skipped["root_type"] == "not_inspected"