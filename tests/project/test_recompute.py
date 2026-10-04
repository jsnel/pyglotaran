from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from glotaran.builtin.io.yml.utils import write_dict
from glotaran.io import load_result
from glotaran.project.recompute import recompute
from glotaran.project.record import read_record
from glotaran.testing.simulated_data.sequential_spectral_decay import ACTIVATION
from glotaran.testing.simulated_data.sequential_spectral_decay import DATASET
from glotaran.testing.simulated_data.sequential_spectral_decay import SCHEME_DICT
from glotaran.testing.simulated_data.shared_decay import LIBRARY
from glotaran.testing.simulated_data.shared_decay import PARAMETERS
from glotaran.testing.simulated_data.shared_decay import SIMULATION_PARAMETERS
from tests.project.fits import fit

if TYPE_CHECKING:
    from pathlib import Path

    from glotaran.project.result import Result

pytestmark = pytest.mark.usefixtures("recording")


def scheme_dict(data_model: dict) -> dict:
    return {
        "library": LIBRARY | {"ca": {"type": "coherent-artifact", "order": 2}},
        "experiments": {"sequential-decay": {"datasets": {"sequential-decay": data_model}}},
    }


# Schemes built in code, with parameters to start from.
SCHEMES = {
    "kinetic": (SCHEME_DICT, PARAMETERS),
    "coherent artifact plugin element": (
        scheme_dict(
            {
                "elements": ["sequential", "ca"],
                "activations": {
                    "irf": ACTIVATION | {"compartments": ACTIVATION["compartments"] | {"ca": 1}}
                },
            }
        ),
        PARAMETERS,
    ),
    "global model": (
        scheme_dict(
            {
                "elements": ["sequential"],
                "global_elements": ["spectral"],
                "activations": {"irf": ACTIVATION},
            }
        ),
        SIMULATION_PARAMETERS,
    ),
}


def assert_same_arrays(result: Result, expected: Result):
    for label, optimization_result in expected.optimization_results.items():
        recomputed = result.optimization_results[label]
        xr.testing.assert_equal(recomputed.fitted_data, optimization_result.fitted_data)
        assert recomputed.elements.keys() == optimization_result.elements.keys()
        for name, element in optimization_result.elements.items():
            xr.testing.assert_equal(recomputed.elements[name], element)


@pytest.mark.parametrize("scheme", SCHEMES)
def test_recompute_reproduces_the_fit(scheme: str):
    """The scheme reloaded from the record gives the same fitted data and element arrays."""
    scheme_dict, parameters = SCHEMES[scheme]
    result = fit(parameters, scheme_dict=scheme_dict, maximum_number_function_evaluations=3)

    recomputed = recompute(result.record, {"sequential-decay": DATASET})

    assert_same_arrays(recomputed, result)
    assert recomputed.optimization_info.cost == result.optimization_info.cost
    assert recomputed.reconstruction is not None
    assert all(
        values["relative_difference"] == 0 for values in recomputed.reconstruction.drift.values()
    )


def test_recompute_reproduces_a_weighted_fit():
    dataset = DATASET.assign(weight=xr.full_like(DATASET.data, 0.5))
    result = fit(dataset=dataset)

    assert_same_arrays(recompute(result.record, {"sequential-decay": dataset}), result)


@pytest.mark.parametrize("reference", ["id", "path", "record"])
def test_recompute_by_reference(recording: Path, reference: str):
    result = fit()
    record = {"id": "001", "path": recording / "results" / "001", "record": result.record}

    recomputed = recompute(record[reference], {"sequential-decay": DATASET})

    assert_same_arrays(recomputed, result)


def test_recompute_provenance(recording: Path):
    result = fit(name="first")

    recomputed = recompute(result.record, {"sequential-decay": DATASET})

    reconstruction = recomputed.reconstruction
    assert reconstruction.original == read_record(result.record.path)
    assert reconstruction.original["name"] == "first"
    assert reconstruction.original_cost_history == (
        pd.read_csv(result.record.path / "cost_history.csv")["cost"].tolist()
    )
    assert reconstruction.data_differences == {}
    assert set(reconstruction.drift) == {
        "cost",
        "sequential-decay root_mean_square_error",
        "sequential-decay weighted_root_mean_square_error",
    }
    # The statistics of the evaluation, the standard errors of the original fit.
    info = recomputed.optimization_info
    assert info.number_of_function_evaluations == 1
    assert info.reduced_chi_square == pytest.approx(info.chi_square / info.degrees_of_freedom)
    assert recomputed.optimized_parameters == result.optimized_parameters
    # Neither recompute nor the dry run it uses writes a record.
    assert [path.name for path in (recording / "results").iterdir()] == ["001"]
    # A fit from the recomputed parameters continues the lineage of the original fit.
    assert recomputed.optimized_parameters.record_uid == result.record.uid
    child = fit(recomputed.optimized_parameters)
    assert read_record(child.record.path)["parent"]["uid"] == result.record.uid


def test_recompute_provenance_survives_saving(tmp_path: Path):
    recomputed = recompute(fit().record, {"sequential-decay": DATASET})
    recomputed.save(tmp_path / "saved")

    loaded = load_result(tmp_path / "saved")

    assert loaded.reconstruction == recomputed.reconstruction


def test_recompute_rejects_other_data():
    result = fit()

    with pytest.raises(ValueError, match=r"sequential-decay: shape time 2100 x spectral 72 -> "):
        recompute(result.record, {"sequential-decay": DATASET.isel(time=slice(0, 500))})

    with pytest.warns(UserWarning, match="differ from the recorded"):
        recomputed = recompute(
            result.record, {"sequential-decay": DATASET * 1.01}, allow_data_mismatch=True
        )
    differences = recomputed.reconstruction.data_differences["sequential-decay"]
    assert differences[-1].startswith("data rms")


def test_recompute_warns_about_drift():
    result = fit()
    record = read_record(result.record.path)
    record["summary"]["cost"] *= 1.001
    record["summary"]["datasets"]["sequential-decay"]["root_mean_square_error"] = np.nan
    write_dict(record, result.record.path / "record.yml")

    with pytest.warns(UserWarning, match="differ from the recorded") as warnings:
        recomputed = recompute(result.record, {"sequential-decay": DATASET})

    message = str(warnings[0].message)
    assert "cost: recorded" in message
    assert "9.99e-04" in message
    assert "root_mean_square_error: recorded nan" in message
    assert "not comparable" in message
    assert recomputed.reconstruction.drift["cost"]["relative_difference"] == pytest.approx(
        1e-3 / 1.001
    )
