from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import xarray as xr

from glotaran.model.data_model import DataModel
from glotaran.model.experiment_model import ExperimentModel
from glotaran.optimization.objective import OptimizationObjective
from glotaran.optimization.optimization import Optimization
from glotaran.parameter import Parameters
from glotaran.simulation import simulate
from tests.optimization.library import test_library


def test_single_data():
    data_model = DataModel(elements=["decay_independent"])
    experiment = ExperimentModel(datasets={"decay_independent": data_model})
    parameters = Parameters.from_dict({"rates": {"decay": [0.8, 0.04]}})

    global_axis = np.arange(10)
    model_axis = np.arange(0, 150, 1)
    clp = xr.DataArray(
        [[1, 10]] * global_axis.size,
        coords=(("global", global_axis), ("clp_label", ["c1", "c2"])),
    )
    data_model.data = simulate(
        data_model, test_library, parameters, {"global": global_axis, "model": model_axis}, clp
    )

    initial_parameters = Parameters.from_dict({"rates": {"decay": [0.9, 0.02]}})
    print(initial_parameters)
    optimization = Optimization(
        models=[experiment],
        parameters=initial_parameters,
        library=test_library,
        raise_exception=True,
        maximum_number_function_evaluations=10,
    )
    optimized_parameters, optimization_results, optimization_info = optimization.run()
    print(optimized_parameters)
    assert optimization_info.success
    assert initial_parameters != optimized_parameters
    assert optimized_parameters.close_or_equal(parameters)
    assert "decay_independent" in optimization_results
    optimization_result = optimization_results["decay_independent"]
    print(optimization_result)
    assert optimization_result.residuals is not None
    assert optimization_result.fitted_data is not None


def test_only_unused_free_parameter_evaluates_model_successfully():
    data_model = DataModel(elements=["decay_independent"])
    experiment = ExperimentModel(datasets={"decay_independent": data_model})
    parameters = Parameters.from_dict(
        {
            "rates": {
                "decay": [
                    [0.8, {"vary": False}],
                    [0.04, {"vary": False}],
                ]
            },
            "unused": [1.0],
        }
    )

    global_axis = np.arange(10)
    model_axis = np.arange(0, 150, 1)
    clp = xr.DataArray(
        [[1, 10]] * global_axis.size,
        coords=(("global", global_axis), ("clp_label", ["c1", "c2"])),
    )
    data_model.data = simulate(
        data_model, test_library, parameters, {"global": global_axis, "model": model_axis}, clp
    )

    optimized_parameters, optimization_results, optimization_info = Optimization(
        models=[experiment],
        parameters=parameters,
        library=test_library,
        raise_exception=True,
    ).run()

    assert optimization_info.success
    assert optimization_info.termination_reason == "No free parameters to optimize."
    assert optimization_info.free_parameter_labels == []
    assert optimization_info.number_of_function_evaluations == 1
    assert optimization_info.number_of_jacobian_evaluations == 0
    assert optimization_info.number_of_parameters == 0
    assert optimization_info.jacobian.shape == (optimization_info.number_of_data_points, 0)
    assert optimization_info.covariance_matrix.shape == (0, 0)
    assert parameters.get_label_value_and_bounds_arrays(exclude_non_vary=True)[0] == ["unused.1"]
    assert [parameter.label for parameter in optimized_parameters.all()] == [
        "rates.decay.1",
        "rates.decay.2",
    ]
    assert "decay_independent" in optimization_results


def test_multiple_experiments():
    data_model = DataModel(elements=["decay_independent"])
    experiments = [
        ExperimentModel(datasets={"decay_independent_1": data_model}),
        ExperimentModel(datasets={"decay_independent_2": data_model}),
    ]
    parameters = Parameters.from_dict({"rates": {"decay": [0.8, 0.04]}})

    global_axis = np.arange(10)
    model_axis = np.arange(0, 150, 1)
    clp = xr.DataArray(
        [[1, 10]] * global_axis.size,
        coords=(("global", global_axis), ("clp_label", ["c1", "c2"])),
    )
    data_model.data = simulate(
        data_model, test_library, parameters, {"global": global_axis, "model": model_axis}, clp
    )

    initial_parameters = Parameters.from_dict({"rates": {"decay": [0.9, 0.02]}})
    print(initial_parameters)
    optimization = Optimization(
        models=experiments,
        parameters=initial_parameters,
        library=test_library,
        raise_exception=True,
        maximum_number_function_evaluations=10,
    )
    optimized_parameters, optimized_data, result = optimization.run()
    assert "decay_independent_1" in optimized_data
    assert "decay_independent_2" in optimized_data
    print(optimized_parameters)
    assert result.success
    assert initial_parameters != optimized_parameters
    assert optimized_parameters.close_or_equal(parameters)


def test_global_data():
    data_model = DataModel(elements=["decay_independent"], global_elements=["gaussian"])
    experiment = ExperimentModel(datasets={"decay_independent": data_model})
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

    global_axis = np.arange(10)
    model_axis = np.arange(0, 150, 1)
    data_model.data = simulate(
        data_model, test_library, parameters, {"global": global_axis, "model": model_axis}
    )

    initial_parameters = Parameters.from_dict(
        {
            "rates": {"decay": [0.8, 0.04]},
            "gaussian": {
                "amplitude": [2.0, 3.0],
                "location": [3.0, 6.0],
                "width": [2.0, 4.0],
            },
        }
    )
    print(initial_parameters)
    optimization = Optimization(
        models=[experiment],
        parameters=initial_parameters,
        library=test_library,
        raise_exception=True,
        maximum_number_function_evaluations=10,
    )
    optimized_parameters, optimized_data, result = optimization.run()
    assert "decay_independent" in optimized_data
    print(optimized_parameters)
    assert result.success
    assert initial_parameters != optimized_parameters
    assert optimized_parameters.close_or_equal(parameters)


def test_multiple_data():
    data_model_one = DataModel(elements=["decay_independent"])
    data_model_two = DataModel(elements=["decay_dependent"])
    experiment = ExperimentModel(
        datasets={"decay_independent": data_model_one, "decay_dependent": data_model_two}
    )
    parameters = Parameters.from_dict({"rates": {"decay": [0.8, 0.04]}})

    global_axis = np.arange(10)
    model_axis = np.arange(0, 150, 1)
    clp = xr.DataArray(
        [[1, 10]] * global_axis.size,
        coords=(("global", global_axis), ("clp_label", ["c1", "c2"])),
    )
    data_model_one.data = simulate(
        data_model_one, test_library, parameters, {"global": global_axis, "model": model_axis}, clp
    )
    data_model_two.data = simulate(
        data_model_two, test_library, parameters, {"global": global_axis, "model": model_axis}, clp
    )

    initial_parameters = Parameters.from_dict({"rates": {"decay": [0.9, 0.02]}})
    print(initial_parameters)
    optimization = Optimization(
        models=[experiment],
        parameters=initial_parameters,
        library=test_library,
        raise_exception=True,
        maximum_number_function_evaluations=10,
    )
    optimized_parameters, optimized_data, result = optimization.run()
    assert "decay_independent" in optimized_data
    assert "decay_dependent" in optimized_data
    print(optimized_parameters)
    assert result.success
    assert initial_parameters != optimized_parameters
    assert optimized_parameters.close_or_equal(parameters)


def test_failed_optimization_reports_the_optimizer_error(monkeypatch: pytest.MonkeyPatch):
    """The evaluation after a failed optimization does not replace the optimizer's error."""
    data_model = DataModel(elements=["decay_independent"])
    experiment = ExperimentModel(datasets={"decay_independent": data_model})
    parameters = Parameters.from_dict({"rates": {"decay": [0.8, 0.04]}})
    data_model.data = simulate(
        data_model,
        test_library,
        parameters,
        {"global": np.arange(2), "model": np.arange(0, 50, 1)},
        xr.DataArray(
            [[1, 10]] * 2, coords=(("global", np.arange(2)), ("clp_label", ["c1", "c2"]))
        ),
    )
    errors = iter([ValueError("optimizer error")])

    def failing_calculate(self: OptimizationObjective):
        raise next(errors, RuntimeError("error of the evaluation after the optimization"))

    monkeypatch.setattr(OptimizationObjective, "calculate", failing_calculate)
    optimization = Optimization(
        models=[experiment], parameters=parameters, library=test_library, raise_exception=False
    )

    with (
        pytest.warns(UserWarning, match="Optimization failed"),
        pytest.raises(ValueError, match="optimizer error"),
    ):
        optimization.run()


def create_decay_optimization(parameters: Parameters | None = None, **kwargs: Any) -> Optimization:
    """Create an optimization of a two-rate decay, by default from non-negative rates."""
    data_model = DataModel(elements=["decay_independent"])
    experiment = ExperimentModel(datasets={"decay_independent": data_model})
    data_model.data = simulate(
        data_model,
        test_library,
        Parameters.from_dict({"rates": {"decay": [0.8, 0.04]}}),
        {"global": np.arange(2), "model": np.arange(0, 50, 1)},
        xr.DataArray(
            [[1, 10]] * 2, coords=(("global", np.arange(2)), ("clp_label", ["c1", "c2"]))
        ),
    )
    if parameters is None:
        parameters = Parameters.from_dict(
            {"rates": {"decay": [[0.9, {"non-negative": True}], [0.02, {"non-negative": True}]]}}
        )
    return Optimization(
        models=[experiment],
        parameters=parameters,
        library=test_library,
        verbose=False,
        raise_exception=True,
        **kwargs,
    )


@pytest.mark.parametrize("method", ["TrustRegionReflection", "Dogbox", "Levenberg-Marquardt"])
def test_optimization_histories(method: str):
    """The cost of every evaluation is recorded, the parameter values only when switched on."""
    optimization = create_decay_optimization(optimization_method=method)
    _, _, info = optimization.run()

    assert len(optimization.cost_history) == optimization.number_of_evaluations
    assert optimization.number_of_evaluations > info.number_of_function_evaluations
    assert optimization.cost_history[-1] == pytest.approx(info.cost)
    assert optimization.parameter_history.number_of_records == 1

    optimization = create_decay_optimization(
        optimization_method=method, record_parameter_history=True
    )
    optimized_parameters, _, _ = optimization.run()
    history = optimization.parameter_history.to_dataframe()

    assert len(history) == optimization.number_of_evaluations + 1
    assert history["evaluation"].tolist() == list(range(optimization.number_of_evaluations + 1))
    assert history["rates.decay.1"].iloc[0] == 0.9
    assert history["rates.decay.1"].iloc[-1] == optimized_parameters.get("rates.decay.1").value


def test_optimization_info_converged_differs_from_success():
    """A fit stopped by the evaluation budget returns (success) without converging."""
    _, _, info = create_decay_optimization(maximum_number_function_evaluations=1).run()
    assert info.success is True
    assert info.converged is False

    _, _, info = create_decay_optimization().run()
    assert info.success is True
    assert info.converged is True


def test_dry_run_calculates_residual_statistics():
    """A dry run at the optimized parameters gives the statistics of the fit."""
    optimized_parameters, _, fit_info = create_decay_optimization().run()
    _, _, info = create_decay_optimization(optimized_parameters).dry_run()

    assert info.success is False
    assert info.converged is None
    assert info.number_of_function_evaluations == 1
    assert info.covariance_matrix is None
    for name in [
        "number_of_data_points",
        "number_of_parameters",
        "number_of_clps",
        "degrees_of_freedom",
    ]:
        assert getattr(info, name) == getattr(fit_info, name)
    # The fit's cost is that of its last evaluation, which the dry run repeats. Its chi-square
    # is that of SciPy's solution, which can be another point of the noise-free fit.
    assert info.cost == fit_info.cost
    assert info.chi_square == pytest.approx(2 * info.cost)
    assert info.reduced_chi_square == pytest.approx(info.chi_square / info.degrees_of_freedom)
    assert info.root_mean_square_error == pytest.approx(np.sqrt(info.reduced_chi_square))
