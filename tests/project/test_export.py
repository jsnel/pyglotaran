from __future__ import annotations

import itertools
import shutil
from typing import TYPE_CHECKING

import pytest
import xarray as xr

from glotaran.io import load_dataset
from glotaran.io import load_parameters
from glotaran.io import load_result
from glotaran.io import save_dataset
from glotaran.project import record as record_module
from glotaran.project.export import export
from glotaran.project.recompute import recompute
from glotaran.project.record import read_record
from glotaran.testing.simulated_data.sequential_spectral_decay import DATASET
from tests.project.fits import fit

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.usefixtures("recording")


def test_export(recording: Path):
    """By default the export is self-contained and carries the project description."""
    (recording / "project.gta").write_text("title: Test project\n")
    notebook = recording / "analysis.ipynb"
    notebook.touch()
    record_module.record_settings.source = str(notebook)
    result = fit(name="best")

    folder = export(result)

    assert folder == recording / "exports" / "last_result"
    assert (folder / "project.gta").read_text() == "title: Test project\n"
    assert (folder / "optimization_results" / "sequential-decay" / "input_data.nc").is_file()
    assert (folder / "optimization_results" / "sequential-decay" / "residuals.nc").is_file()
    metadata = read_record(folder)
    record = read_record(result.record.path)
    assert metadata["id"] == "001"
    assert metadata["uid"] == result.record.uid
    assert metadata["name"] == "best"
    assert metadata["summary"] == record["summary"]
    assert metadata["data"] == record["data"]
    assert metadata["changed_parameters"] == []
    assert "exported" in metadata


def test_export_names_only_the_source_file(recording: Path):
    (recording / "notebooks").mkdir()
    (recording / "notebooks" / "analysis.ipynb").touch()
    record_module.record_settings.source = str(recording / "notebooks" / "analysis.ipynb")
    result = fit()

    assert read_record(result.record.path)["source"] == "../../notebooks/analysis.ipynb"
    assert read_record(export(result))["source"] == "analysis.ipynb"


def test_export_folder_names(recording: Path):
    result = fit()

    assert export(result, "fig3") == recording / "exports" / "fig3"
    with pytest.raises(FileExistsError, match="overwrite=True"):
        export(result, "fig3")
    assert export(result, "fig3", overwrite=True) == recording / "exports" / "fig3"
    assert export(result, recording / "elsewhere") == recording / "elsewhere"

    (recording / "exports" / "not_an_export").mkdir()
    with pytest.raises(FileExistsError, match="is not an export"):
        export(result, "not_an_export", overwrite=True)


@pytest.mark.parametrize(
    ("input_data", "arrays", "source_files"), list(itertools.product([True, False], repeat=3))
)
def test_export_without_the_original_files(
    recording: Path, input_data: bool, arrays: bool, source_files: bool
):
    """Every combination loads or recomputes after the data file and records are gone."""
    save_dataset(DATASET.copy(), recording / "data" / "raw.nc")
    dataset = load_dataset(recording / "data" / "raw.nc")
    result = fit(dataset=dataset)

    folder = export(result, input_data=input_data, arrays=arrays, source_files=source_files)
    shutil.rmtree(recording / "data")
    shutil.rmtree(recording / "results")

    assert (folder / "source_files" / "sequential-decay" / "raw.nc").is_file() is source_files
    if input_data:
        loaded = load_result(folder).optimization_results["sequential-decay"]
        xr.testing.assert_equal(
            loaded.input_data.data,
            result.optimization_results["sequential-decay"].input_data,
        )
        assert (loaded.residuals is not None) is arrays
    else:
        assert not list(folder.rglob("input_data.nc"))
        with pytest.raises(ValueError, match="Input data cannot be None"):
            load_result(folder)
    recomputed = recompute(folder, {"sequential-decay": DATASET})
    xr.testing.assert_equal(
        recomputed.optimization_results["sequential-decay"].fitted_data,
        result.optimization_results["sequential-decay"].fitted_data,
    )


def test_export_lists_missing_source_files(recording: Path):
    dataset = DATASET.copy()
    dataset.attrs |= {"source_path": "gone.nc", "io_plugin_name": "glotaran.builtin.io.netCDF"}

    with pytest.warns(UserWarning, match=r"sequential-decay \(gone.nc\)"):
        folder = export(fit(dataset=dataset))

    assert read_record(folder)["missing_source_files"] == {"sequential-decay": "gone.nc"}


def test_export_in_memory_data_has_no_source_file(recording: Path):
    folder = export(fit())

    assert read_record(folder)["source_files"] == {}
    assert read_record(folder)["missing_source_files"] == {}


def test_export_warns_about_parameters_changed_in_place():
    result = fit()
    result.optimized_parameters.get("rates.species_1").value = 0.7

    with pytest.warns(UserWarning, match="changed values of: rates.species_1"):
        folder = export(result)

    assert read_record(folder)["changed_parameters"] == ["rates.species_1"]
    exported = load_parameters(folder / "optimized_parameters.csv")
    assert exported.get("rates.species_1").value == 0.7


def test_export_recomputed_result():
    result = fit(name="original")
    recomputed = recompute(result.record, {"sequential-decay": DATASET})

    folder = export(recomputed)

    metadata = read_record(folder)
    assert metadata["uid"] == result.record.uid
    assert metadata["name"] == "original"
    assert load_result(folder).reconstruction == recomputed.reconstruction


def test_export_unrecorded_fit():
    result = fit(record=False)

    folder = export(result)

    metadata = read_record(folder)
    assert metadata["uid"] is None
    assert metadata["data"]["sequential-decay"]["shape"] == {"time": 2100, "spectral": 72}
    assert metadata["optimizer"]["maximum_number_function_evaluations"] == 1
    recomputed = recompute(folder, {"sequential-decay": DATASET})
    assert recomputed.optimization_info.cost == result.optimization_info.cost
