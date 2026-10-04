from __future__ import annotations

import shutil
from copy import deepcopy
from typing import TYPE_CHECKING

import pytest

from glotaran.builtin.io.yml.utils import write_dict
from glotaran.project import record as record_module
from glotaran.project.compare import compare_results
from glotaran.project.compare import list_results
from glotaran.project.record import read_record
from glotaran.testing.simulated_data.sequential_spectral_decay import DATASET
from glotaran.testing.simulated_data.sequential_spectral_decay import SCHEME_DICT
from tests.project.fits import fit

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.usefixtures("recording")


@pytest.fixture
def lineage(recording: Path) -> None:
    """Fit three times from two sources; the third fit is on replaced data."""
    record_module.record_settings.source = "analysis_a.ipynb"
    first = fit(name="model A")
    second = fit(first.optimized_parameters, name="model A")
    record_module.record_settings.source = "analysis_b.ipynb"
    fit(second.optimized_parameters, DATASET * 1.5, name="model B")


@pytest.mark.usefixtures("lineage")
def test_list_results():
    table = list_results()

    assert table.index.tolist() == ["001", "002", "003"]
    assert table["source"].tolist() == ["analysis_a.ipynb", "analysis_a.ipynb", "analysis_b.ipynb"]
    assert table["status"].tolist() == ["success"] * 3
    assert table["parent"].tolist() == ["", "001", "002"]
    assert table["sequential-decay shape"].tolist() == ["2100 x 72"] * 3
    rms = table["sequential-decay rms"]
    assert rms["001"] == rms["002"]
    assert rms["003"] == pytest.approx(1.5 * rms["001"])
    assert table["number_of_function_evaluations"].tolist() == [1, 1, 1]


@pytest.mark.usefixtures("lineage")
def test_list_results_filters():
    assert list_results(source="_b").index.tolist() == ["003"]
    assert list_results(name="model A").index.tolist() == ["001", "002"]
    empty = list_results(status="failed")
    assert empty.empty
    assert "status" in empty.columns
    assert list_results(scheme_source="scheme.yml").empty


def test_list_results_in_a_folder(recording: Path):
    fit(output_folder="other")

    assert list_results().empty
    assert list_results("other").index.tolist() == ["001"]
    assert list_results(recording / "other").index.tolist() == ["001"]


def test_list_results_sorts_by_number(recording: Path):
    fit()
    shutil.copytree(recording / "results" / "001", recording / "results" / "999")
    fit()

    assert list_results().index.tolist() == ["001", "999", "1000"]


def test_list_results_skips_other_output(recording: Path):
    """Output of a v0.7 project and damaged records are not listed."""
    fit()
    v07_result = recording / "results" / "my_result_run_0000"
    v07_result.mkdir()
    write_dict({"glotaran_version": "0.7.4"}, v07_result / "result.yml")
    (recording / "results" / "002").mkdir()
    fit()
    (recording / "results" / "003" / "record.yml").write_text("{damaged: [")

    with pytest.warns(UserWarning, match="Skipping the damaged record"):
        assert list_results().index.tolist() == ["001"]


def test_list_results_marks_incomplete_records_and_changed_parents(recording: Path):
    first = fit()
    second = fit(first.optimized_parameters)
    fit(second.optimized_parameters)
    # A fit that never returned, after which the parent's number was reused
    record = read_record(recording / "results" / "002")
    write_dict(
        record | {"status": "running", "uid": "other"},
        recording / "results" / "002" / "record.yml",
    )
    shutil.rmtree(recording / "results" / "001")

    table = list_results()

    assert table["status"].tolist() == ["incomplete", "success"]
    assert table["parent"].tolist() == ["001 (missing)", "002 (replaced)"]
    assert list_results(status="incomplete").index.tolist() == ["002"]


@pytest.mark.usefixtures("lineage")
def test_compare_results():
    comparison = compare_results("002", "003")

    assert comparison.summary.columns.tolist() == ["002", "003"]
    assert comparison.summary.loc["cost", "003"] > comparison.summary.loc["cost", "002"]
    assert "sequential-decay weighted_root_mean_square_error" in comparison.summary.index
    assert set(comparison.parameters["field"]) == {"value", "standard_error"}
    assert comparison.data_differences["sequential-decay"][0].startswith("data min")
    assert comparison.scheme_diff == ""
    markdown = comparison._repr_markdown_()
    assert "### Parameters" in markdown
    assert "sequential-decay: data min" in markdown


def test_compare_results_with_a_result_and_another_scheme():
    first = fit()
    scheme_dict = deepcopy(SCHEME_DICT)
    data_model = scheme_dict["experiments"]["sequential-decay"]["datasets"]["sequential-decay"]
    data_model["activations"]["irf"]["compartments"] = {"s1": 1}
    second = fit(scheme_dict=scheme_dict, record=False)

    comparison = compare_results(first, second)

    assert comparison.summary.columns.tolist() == ["001", "result"]
    assert "-              s2: 1" in comparison.scheme_diff
    assert comparison.data_differences == {}
    assert comparison.parameters.columns.tolist() == ["label", "field", "001", "result"]
