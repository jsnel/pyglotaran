from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from glotaran.project.data_summary import compare_data_summaries
from glotaran.project.data_summary import summarize_data


@pytest.fixture
def dataset() -> xr.Dataset:
    rng = np.random.default_rng(0)
    return xr.DataArray(
        rng.normal(size=(50, 4)),
        coords={"time": np.linspace(-1, 10, 50), "spectral": [400.0, 500.0, 600.0, 700.0]},
    ).to_dataset(name="data")


def compare(recorded: xr.Dataset, current: xr.Dataset) -> list[str]:
    return compare_data_summaries(
        {"ta": summarize_data(recorded)}, {"ta": summarize_data(current)}
    ).get("ta", [])


def test_summarize_data(dataset: xr.Dataset):
    summary = summarize_data(dataset)

    assert list(summary) == ["shape", "time", "spectral", "data"]
    assert summary["shape"] == {"time": 50, "spectral": 4}
    assert summary["time"] == {
        "min": -1.0,
        "max": 10.0,
        "mean": 4.5,
        "rms": pytest.approx(np.sqrt(np.mean(np.linspace(-1, 10, 50) ** 2))),
    }
    assert summary["spectral"]["rms"] == pytest.approx(
        np.sqrt(np.mean(np.square([4, 5, 6, 7]))) * 100
    )
    assert summary == summarize_data(dataset.data)


def test_summary_is_equal_for_equivalent_data(dataset: xr.Dataset):
    """Transposed data, another dtype and differences below the tolerance count as equal."""
    assert compare(dataset, dataset) == []
    assert compare(dataset, dataset.transpose()) == []
    assert compare(dataset, dataset.astype(np.dtype(">f8"))) == []
    assert compare(dataset, dataset + 1e-9 * float(np.sqrt(np.mean(dataset.data**2)))) == []


def test_summary_reports_changed_data(dataset: xr.Dataset):
    """A changed shape, coordinate range or values are reported with what changed."""
    changes = compare(dataset, dataset.isel(time=slice(0, 40)))

    assert changes[0] == "shape time 50 x spectral 4 -> time 40 x spectral 4"
    assert changes[1] == "time max 10 -> 7.7551 (-22 %)"
    assert not any(change.startswith("spectral") for change in changes)
    assert any(change.startswith("data rms") for change in changes)
    assert compare(dataset, dataset * 1.01)[-1].endswith("(+1 %)")


def test_summary_reports_weight(dataset: xr.Dataset):
    weighted = dataset.assign(weight=xr.full_like(dataset.data, 0.5))

    assert compare(dataset, weighted) == ["weight added"]
    assert compare(weighted, dataset) == ["weight removed"]
    assert compare(weighted, weighted.assign(weight=weighted.weight * 2)) == [
        "weight min 0.5 -> 1 (+1e+02 %)",
        "weight max 0.5 -> 1 (+1e+02 %)",
        "weight mean 0.5 -> 1 (+1e+02 %)",
        "weight rms 0.5 -> 1 (+1e+02 %)",
    ]


def test_compare_data_summaries_reports_missing_and_new_datasets(dataset: xr.Dataset):
    summary = summarize_data(dataset)

    assert compare_data_summaries({"a": summary}, {"a": summary}) == {}
    assert compare_data_summaries({"a": summary}, {"b": summary}) == {
        "a": ["dataset missing"],
        "b": ["new dataset"],
    }
