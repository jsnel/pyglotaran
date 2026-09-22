from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from glotaran.builtin.elements.kinetic import KineticElement
from glotaran.builtin.elements.kinetic.matrix import calculate_matrix_gaussian_activation_on_index
from glotaran.builtin.items.activation import Activation
from glotaran.builtin.items.activation import ActivationDataModel
from glotaran.builtin.items.activation import GaussianActivation
from glotaran.builtin.items.activation import InstantActivation
from glotaran.builtin.items.activation import MultiGaussianActivation
from glotaran.builtin.items.activation import NormConvMultiMultiGaussianActivation
from glotaran.model.experiment_model import ExperimentModel
from glotaran.optimization import Optimization
from glotaran.parameter import Parameters
from glotaran.simulation import simulate

test_library = {
    "parallel": KineticElement(
        **{
            "label": "parallel",
            "type": "kinetic",
            "rates": {
                ("s1", "s1"): "rates.1",
                ("s2", "s2"): "rates.2",
            },
        }
    ),
    "sequential": KineticElement(
        **{
            "label": "sequential",
            "type": "kinetic",
            "rates": {
                ("s2", "s1"): "rates.1",
                ("s2", "s2"): "rates.2",
            },
        }
    ),
    "equilibrium": KineticElement(
        **{
            "label": "equilibrium",
            "type": "kinetic",
            "rates": {
                ("s2", "s1"): "rates.1",
                ("s2", "s2"): "rates.2",
                ("s1", "s2"): "rates.3",
            },
            "clp_constraints": [
                {"type": "zero", "target": "s1", "interval": (1, 1)},
                {"type": "zero", "target": "s2", "interval": (0, 0)},
            ],
        }
    ),
}


test_parameters_simulation = Parameters.from_dict(
    {"rates": [0.2, 0.01, 0.09], "gaussian": [["center", 50], ["width", 20]]}
)
test_parameters = Parameters.from_dict(
    {"rates": [0.1, 0.02, 0.08, {"min": 0}], "gaussian": [["center", 60], ["width", 8]]}
)

test_global_axis = np.array([0, 1])
test_model_axis = np.arange(-10, 1500, 1)
test_axies = {"spectral": test_global_axis, "time": test_model_axis}
test_clp = xr.DataArray(
    [
        [1, 0],
        [0, 1],
    ],
    coords=[("clp_label", ["s1", "s2"]), ("spectral", test_global_axis.data)],
).T


def test_initial_concentration_is_normalized_across_activation_compartments():
    element = KineticElement(
        label="subset",
        type="kinetic",
        rates={("s1", "s1"): 1},
    )
    model = ActivationDataModel(
        elements=["subset"],
        activations={
            "instant": InstantActivation(
                type="instant",
                compartments={"s1": 1, "outside": 3},
            )
        },
    )

    _, matrix = element.calculate_matrix(model, np.array([0]), np.array([0]))

    assert matrix[0, 0] == pytest.approx(0.25)


def test_result_initial_concentration_uses_normalized_activation_values():
    element = KineticElement(
        label="subset",
        type="kinetic",
        rates={("s1", "s1"): 1},
    )
    model = ActivationDataModel(
        elements=["subset"],
        activations={
            "instant": InstantActivation(type="instant", compartments={"s1": 1, "outside": 3})
        },
    )
    amplitudes = xr.DataArray(
        [[1.0]],
        coords={"wavelength": [0.0], "amplitude_label": ["s1"]},
        dims=("wavelength", "amplitude_label"),
    )
    concentrations = xr.DataArray(
        [[1.0]],
        coords={"time": [0.0], "amplitude_label": ["s1"]},
        dims=("time", "amplitude_label"),
    )
    result = element.create_result(model, "wavelength", "time", amplitudes, concentrations)
    initial_concentration = result["initial_concentrations"].sel(
        activation="instant", compartment="s1"
    )
    assert initial_concentration == pytest.approx(0.25)


def test_zero_initial_concentration_sum_is_rejected():
    element = KineticElement(label="subset", type="kinetic", rates={("s1", "s1"): 1})
    model = ActivationDataModel(
        elements=["subset"],
        activations={"instant": InstantActivation(type="instant", compartments={"s1": 0})},
    )
    with pytest.raises(ValueError, match="sum is zero"):
        element.calculate_matrix(model, np.array([0]), np.array([0]))


def test_grouped_convolution_without_shift_uses_each_index_scale_in_matrix():
    element = KineticElement(
        label="single",
        type="kinetic",
        rates={("s1", "s1"): 1.0},
    )
    activation = NormConvMultiMultiGaussianActivation(
        type="norm-conv-multi-multi-gaussian",
        compartments={"s1": 1.0},
        center=[[0.0]],
        width=[[0.1]],
        scale=[[1.0]],
        convwidth=[0.05, 0.2],
    )
    model = ActivationDataModel(elements=["single"], activations={"irf": activation})
    global_axis = np.array([400.0, 500.0])
    model_axis = np.linspace(-1.0, 1.0, 31)
    labels, matrix = element.calculate_matrix(model, global_axis, model_axis)
    assert labels == ["s1"]
    assert matrix.shape == (2, model_axis.size, 1)

    parameters = activation.parameters(global_axis)
    expected = np.zeros_like(matrix)
    for index, per_index in enumerate(parameters):
        calculate_matrix_gaussian_activation_on_index(
            expected[index],
            np.array([1.0]),
            model_axis,
            np.array([per_index[0].center]),
            np.array([per_index[0].width]),
            np.array([per_index[0].scale]),
            per_index[0].backsweep_period,
        )
    assert np.allclose(matrix, expected)
    assert not np.allclose(matrix[0], matrix[1])


@pytest.mark.parametrize("decay_method", ["parallel", "sequential", "equilibrium"])
@pytest.mark.parametrize(
    "activation",
    [
        InstantActivation(type="instant", compartments={}),
        GaussianActivation(
            type="gaussian",
            compartments={},
            center="gaussian.center",
            width="gaussian.width",
        ),
        GaussianActivation(
            type="gaussian",
            compartments={},
            center="gaussian.center",
            width="gaussian.width",
            shift=[1, 0],
        ),
        MultiGaussianActivation(
            type="multi-gaussian",
            compartments={},
            center=["gaussian.center"],
            width=["gaussian.width", "gaussian.width"],
        ),
    ],
)
def test_decay(decay_method: str, activation: Activation):
    dataset_label = "dataset1"
    if decay_method == "parallel":
        activation.compartments = {"s1": 1, "s2": 1}
    else:
        activation.compartments = {"s1": 1}
    data_model = ActivationDataModel(elements=[decay_method], activations={"irf": activation})
    data_model.data = simulate(
        data_model, test_library, test_parameters_simulation, test_axies, clp=test_clp
    )
    experiments = [ExperimentModel(datasets={"dataset1": data_model})]
    optimization = Optimization(
        models=experiments,
        parameters=test_parameters,
        library=test_library,
        raise_exception=True,
        maximum_number_function_evaluations=25,
    )
    optimized_parameters, optimization_results, optimization_info = optimization.run()
    assert optimization_info.success
    assert optimized_parameters.close_or_equal(test_parameters_simulation)
    assert dataset_label in optimization_results
    optimization_result = optimization_results[dataset_label]

    assert optimization_result.residuals is not None
    assert optimization_result.elements is not None
    assert optimization_result.elements[decay_method] is not None
    decay_result = optimization_result.elements[decay_method]
    assert "amplitudes" in decay_result
    assert "concentrations" in decay_result
    assert "initial_concentrations" in decay_result
    assert "k_matrix" in decay_result
    assert "reduced_k_matrix" in decay_result
    assert "a_matrix" in decay_result
    assert "kinetic_amplitudes" in decay_result

    if isinstance(activation, MultiGaussianActivation):
        assert optimization_result.activations["irf"] is not None
        # TODO: check if we need more tests here
        # assert "gaussian_activation" in optimization_result["dataset1"].coords
        # assert "gaussian_activation_function" in optimization_result["dataset1"]
