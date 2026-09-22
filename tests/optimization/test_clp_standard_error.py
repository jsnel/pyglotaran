"""Tests for the staging CLP uncertainty calculation."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
import pytest
import xarray as xr

from glotaran.model.data_model import DataModel
from glotaran.model.experiment_model import ExperimentModel
from glotaran.optimization.clp_standard_error import ClpStandardErrorSettings
from glotaran.optimization.clp_standard_error import _evaluate_with_restoration
from glotaran.optimization.clp_standard_error import calculate_clp_standard_error
from glotaran.optimization.objective import FitDecomposition
from glotaran.optimization.optimization import Optimization
from glotaran.parameter import Parameters
from glotaran.project.scheme import Scheme
from glotaran.simulation import simulate
from tests.optimization.library import test_library

if TYPE_CHECKING:
    from pathlib import Path


def _result(value: float) -> dict[str, SimpleNamespace]:
    clp = xr.DataArray(
        [[value]],
        dims=("global", "amplitude_label"),
        coords={"global": [0.0], "amplitude_label": ["c"]},
    )
    return {"dataset": SimpleNamespace(fit_decomposition=SimpleNamespace(clp=clp))}


def test_clp_standard_error_propagates_covariance_without_mutating_parameters() -> None:
    parameters = Parameters.from_dict({"fit": {"p": [1.0]}})
    base_results = _result(1.0)
    objective = SimpleNamespace(
        model=SimpleNamespace(residual_function="variable_projection"),
        data=SimpleNamespace(),
    )

    def evaluate_at(candidate: Parameters):
        value = candidate.get("fit.p.1").value ** 2
        return _result(value)

    result = calculate_clp_standard_error(
        objective,
        parameters,
        ["fit.p.1"],
        np.asarray([[4.0]]),
        ClpStandardErrorSettings(relative_step=1e-6),
        base_results=base_results,
        evaluate_at=evaluate_at,
    )

    np.testing.assert_allclose(result["dataset"].data, [[4.0]], rtol=1e-5)
    assert result["dataset"].attrs["method"] == "linear_plus_nonlinear_propagation"
    assert parameters.get("fit.p.1").value == 1.0


def test_clp_standard_error_rejects_covariance_shape_mismatch() -> None:
    parameters = Parameters.from_dict({"fit": {"p": [1.0]}})
    objective = SimpleNamespace(
        model=SimpleNamespace(residual_function="variable_projection"),
        data=SimpleNamespace(),
    )
    with pytest.raises(ValueError, match="covariance matrix shape"):
        calculate_clp_standard_error(
            objective,
            parameters,
            ["fit.p.1"],
            np.zeros((2, 2)),
            ClpStandardErrorSettings(),
            base_results=_result(1.0),
            evaluate_at=lambda _: _result(1.0),
        )


def test_clp_standard_error_skips_non_variable_projection() -> None:
    objective = SimpleNamespace(
        model=SimpleNamespace(residual_function="non_negative_least_squares"),
        data=SimpleNamespace(),
    )
    with pytest.warns(UserWarning, match="variable_projection"):
        result = calculate_clp_standard_error(
            objective,
            Parameters.from_dict({"fit": {"p": [1.0]}}),
            ["fit.p.1"],
            np.zeros((1, 1)),
            ClpStandardErrorSettings(),
            base_results=_result(1.0),
        )
    assert result == {}


def test_clp_standard_error_uses_optimizer_coordinate_for_non_negative_parameter() -> None:
    parameters = Parameters.from_dict({"fit": {"p": [np.e, {"non-negative": True}]}})
    objective = SimpleNamespace(
        model=SimpleNamespace(residual_function="variable_projection"),
        data=SimpleNamespace(),
    )
    base_results = _result(np.e)

    def evaluate_at(candidate: Parameters):
        return _result(candidate.get("fit.p.1").value)

    result = calculate_clp_standard_error(
        objective,
        parameters,
        ["fit.p.1"],
        np.asarray([[4.0]]),
        ClpStandardErrorSettings(),
        base_results=base_results,
        evaluate_at=evaluate_at,
    )

    # d(exp(theta))/d(theta) = exp(theta) at theta=1; covariance is 4.
    np.testing.assert_allclose(result["dataset"].data, [[2 * np.e]], rtol=1e-5)


def test_scheme_optimizer_attaches_unlinked_clp_standard_errors() -> None:
    data_model = DataModel(elements=["decay_independent"])
    experiment = ExperimentModel(datasets={"dataset": data_model})
    parameters = Parameters.from_dict({"rates": {"decay": [0.8, 0.04]}})
    axes = {"global": np.arange(3), "model": np.arange(40)}
    clp = xr.DataArray(
        [[1.0, 2.0]] * 3,
        coords=(("global", axes["global"]), ("clp_label", ["c1", "c2"])),
    )
    data_model.data = simulate(data_model, test_library, parameters, axes, clp)
    data_model.data["data"].data += 1e-3 * np.sin(np.arange(120)).reshape(40, 3)

    scheme = Scheme.model_construct(library=test_library, experiments={"experiment": experiment})
    result = scheme.optimize(
        Parameters.from_dict({"rates": {"decay": [0.82, 0.05]}}),
        {"dataset": data_model.data},
        compute_clp_standard_error=True,
        clp_standard_error_finite_difference_relative_step=2e-6,
        verbose=False,
        raise_exception=True,
        maximum_number_function_evaluations=5,
    )
    results = result.optimization_results
    info = result.optimization_info
    assert info.clp_standard_error_finite_difference_relative_step == 2e-6

    uncertainty = results["dataset"].fit_decomposition.clp_standard_error
    assert uncertainty is not None
    assert uncertainty.dims == results["dataset"].fit_decomposition.clp.dims
    assert np.all(np.isfinite(uncertainty))
    assert np.any(uncertainty > 0)
    assert info.clp_standard_error_method == "linear_plus_nonlinear_propagation"


def test_linked_optimizer_attaches_clp_standard_errors_to_each_dataset() -> None:
    parameters = Parameters.from_dict({"rates": {"decay": [0.8, 0.04]}})
    datasets = {}
    for label in ["first", "second"]:
        data_model = DataModel(elements=["decay_independent"])
        axes = {"global": np.arange(3), "model": np.arange(40)}
        clp = xr.DataArray(
            [[1.0, 2.0]] * 3,
            coords=(("global", axes["global"]), ("clp_label", ["c1", "c2"])),
        )
        data_model.data = simulate(data_model, test_library, parameters, axes, clp)
        data_model.data["data"].data += 1e-3 * np.sin(np.arange(120)).reshape(40, 3)
        datasets[label] = data_model
    experiment = ExperimentModel(datasets=datasets)

    _, results, _ = Optimization(
        models=[experiment],
        parameters=Parameters.from_dict({"rates": {"decay": [0.82, 0.05]}}),
        library=test_library,
        compute_clp_standard_error=True,
        verbose=False,
        raise_exception=True,
        maximum_number_function_evaluations=5,
    ).run()

    assert set(results) == {"first", "second"}
    for result in results.values():
        uncertainty = result.fit_decomposition.clp_standard_error
        assert uncertainty is not None
        assert np.all(np.isfinite(uncertainty))
        assert np.any(uncertainty > 0)


def test_clp_standard_error_rejects_non_negative_least_squares() -> None:
    data_model = DataModel(
        elements=["decay_independent"],
        residual_function="non_negative_least_squares",
    )
    experiment = ExperimentModel(datasets={"dataset": data_model})
    parameters = Parameters.from_dict({"rates": {"decay": [0.8, 0.04]}})
    axes = {"global": np.arange(2), "model": np.arange(20)}
    clp = xr.DataArray(
        [[1.0, 2.0]] * 2,
        coords=(("global", axes["global"]), ("clp_label", ["c1", "c2"])),
    )
    data_model.data = simulate(data_model, test_library, parameters, axes, clp)
    with pytest.raises(ValueError, match="variable_projection"):
        Optimization(
            models=[experiment],
            parameters=parameters,
            library=test_library,
            compute_clp_standard_error=True,
        )


def test_perturbed_objective_restores_values_after_exception() -> None:
    parameters = Parameters.from_dict({"fit": {"p": [1.0]}})
    candidate = parameters.copy()
    candidate.get("fit.p.1").value = 2.0

    class FailingObjective:
        def get_result(self):
            raise RuntimeError("synthetic evaluation failure")

    with pytest.raises(RuntimeError, match="synthetic"):
        _evaluate_with_restoration(FailingObjective(), parameters, candidate)
    assert parameters.get("fit.p.1").value == 1.0


def test_fit_decomposition_serializes_and_loads_clp_standard_error(tmp_path: Path) -> None:
    clp = xr.DataArray([[1.0]], dims=("global", "amplitude_label"))
    uncertainty = xr.DataArray([[0.1]], dims=clp.dims)
    decomposition = FitDecomposition(clp=clp, matrix=clp, clp_standard_error=uncertainty)
    serialized = decomposition.model_dump(mode="json", context={"save_folder": tmp_path})

    assert serialized["clp_standard_error"] == "clp_standard_error.nc"
    restored = FitDecomposition.model_validate(serialized, context={"save_folder": tmp_path})
    xr.testing.assert_equal(restored.clp_standard_error, uncertainty)


def test_global_paired_optimizer_attaches_clp_standard_errors() -> None:
    data_model = DataModel(
        elements=["decay_independent"],
        global_elements=["gaussian"],
        global_composition="paired",
    )
    experiment = ExperimentModel(datasets={"dataset": data_model})
    parameters = Parameters.from_dict(
        {
            "rates": {"decay": [0.8, 0.04]},
            "gaussian": {
                "amplitude": [2.0, 3.0],
                "location": [3.0, 6.0],
                "width": [2.0, 4.0],
            },
        }
    )
    axes = {"global": np.arange(3), "model": np.arange(20)}
    data_model.data = simulate(data_model, test_library, parameters, axes)
    data_model.data["data"].data += 1e-3 * np.sin(np.arange(60)).reshape(20, 3)

    _, results, info = Optimization(
        models=[experiment],
        parameters=parameters.copy(),
        library=test_library,
        compute_clp_standard_error=True,
        verbose=False,
        raise_exception=True,
        maximum_number_function_evaluations=3,
    ).run()

    uncertainty = results["dataset"].fit_decomposition.clp_standard_error
    assert uncertainty is not None
    assert uncertainty.shape == results["dataset"].fit_decomposition.clp.shape
    assert np.all(np.isfinite(uncertainty))
    assert np.any(uncertainty > 0)
    assert info.clp_standard_error_method == "linear_plus_nonlinear_propagation"
