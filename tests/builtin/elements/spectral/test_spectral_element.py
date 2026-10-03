from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from glotaran.builtin.elements.spectral import SpectralDataModel
from glotaran.builtin.elements.spectral import SpectralElement
from glotaran.builtin.elements.spectral.shape import SpectralShapeSkewedGaussian
from glotaran.builtin.elements.spectral.shape import SpectralShapeSkewedGaussianSum
from glotaran.model.experiment_model import ExperimentModel
from glotaran.model.item import get_item_issues
from glotaran.model.item import resolve_item_parameters
from glotaran.optimization import Optimization
from glotaran.parameter import Parameters
from glotaran.simulation import simulate

test_library = {
    "gaussian": SpectralElement(
        **{
            "label": "gaussian",
            "type": "spectral",
            "shapes": {
                "s1": {
                    "type": "gaussian",
                    "amplitude": "shape.amplitude",
                    "location": "shape.location",
                    "width": "shape.width",
                }
            },
        }
    ),
    "skewed_gaussian_neg": SpectralElement(
        **{
            "label": "skewed_gaussian_neg",
            "type": "spectral",
            "shapes": {
                "s1": {
                    "type": "skewed-gaussian",
                    "location": "shape.location",
                    "width": "shape.width",
                    "skewness": -1,
                }
            },
        }
    ),
    "skewed_gaussian_pos": SpectralElement(
        **{
            "label": "skewed_gaussian_pos",
            "type": "spectral",
            "shapes": {
                "s1": {
                    "type": "skewed-gaussian",
                    "location": "shape.location",
                    "width": "shape.width",
                    "skewness": 1,
                }
            },
        }
    ),
    "skewed_gaussian_zero": SpectralElement(
        **{
            "label": "skewed_gaussian_zero",
            "type": "spectral",
            "shapes": {
                "s1": {
                    "type": "skewed-gaussian",
                    "location": "shape.location",
                    "width": "shape.width",
                    "skewness": 0,
                }
            },
        }
    ),
}


test_parameters_simulation = Parameters.from_dict(
    {
        "shape": [
            ["amplitude", 50],
            ["location", 50],
            ["width", 20],
        ],
    }
)
test_parameters = Parameters.from_dict(
    {
        "shape": [
            ["amplitude", 50, {"vary": False}],
            ["location", 51],
            ["width", 21],
        ],
    }
)

test_global_axis = np.array([0])
test_model_axis = np.arange(0, 100, 1)
test_axies = {"time": test_global_axis, "spectral": test_model_axis}
test_clp = xr.DataArray(
    [[1]],
    coords=[
        ("clp_label", ["s1"]),
        ("time", test_global_axis.data),
    ],
).T


@pytest.mark.parametrize(
    "shape",
    [
        "gaussian",
        "skewed_gaussian_neg",
        "skewed_gaussian_pos",
        "skewed_gaussian_zero",
    ],
)
def test_spectral(shape: str):
    dataset_label = "dataset1"
    data_model = SpectralDataModel(elements=[shape])
    data_model.data = simulate(
        data_model, test_library, test_parameters_simulation, test_axies, clp=test_clp
    )
    experiments = [
        ExperimentModel(
            datasets={dataset_label: data_model},
        )
    ]
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
    assert shape in optimization_results[dataset_label].elements
    element_result = optimization_results[dataset_label].elements[shape]
    assert "amplitudes" in element_result
    assert "concentrations" in element_result


def test_skewed_gaussian_sum_shape_sums_components():
    axis = np.array([640.0, 650.0, 660.0])
    shape = SpectralShapeSkewedGaussianSum(
        type="skewed-gaussian-sum",
        amplitude=[2.0, 3.0],
        location=[650.0, 660.0],
        width=[20.0, 10.0],
        skewness=[0.0, 0.5],
    )
    first = SpectralShapeSkewedGaussian(
        type="skewed-gaussian",
        amplitude=2.0,
        location=650.0,
        width=20.0,
        skewness=0.0,
    ).calculate(axis)
    second = SpectralShapeSkewedGaussian(
        type="skewed-gaussian",
        amplitude=3.0,
        location=660.0,
        width=10.0,
        skewness=0.5,
    ).calculate(axis)

    np.testing.assert_allclose(shape.calculate(axis), first + second)


def test_skewed_gaussian_sum_shape_rejects_mismatched_parameter_lists():
    shape = SpectralShapeSkewedGaussianSum(
        type="skewed-gaussian-sum",
        amplitude=[1.0, 2.0],
        location=[650.0],
        width=[20.0, 10.0],
        skewness=[0.0, 0.5],
    )

    with pytest.raises(ValueError, match="same number of components"):
        shape.calculate(np.array([650.0]))


def test_skewed_gaussian_sum_shape_resolves_parameter_lists():
    element = SpectralElement(
        label="sum",
        type="spectral",
        shapes={
            "s1": {
                "type": "skewed-gaussian-sum",
                "amplitude": ["p.1", "p.2"],
                "location": ["p.3", "p.4"],
                "width": ["p.5", "p.6"],
                "skewness": ["p.7", "p.8"],
            }
        },
    )
    parameters = Parameters.from_dict({"p": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]})

    assert get_item_issues(element, parameters) == []
    shape = resolve_item_parameters(element, parameters).shapes["s1"]

    assert [p.value for p in shape.amplitude] == [1.0, 2.0]
    assert [p.value for p in shape.location] == [3.0, 4.0]
    assert [p.value for p in shape.width] == [5.0, 6.0]
    assert [p.value for p in shape.skewness] == [7.0, 8.0]
    assert shape.calculate(np.array([3.0])).shape == (1,)
    assert get_item_issues(element, Parameters.from_dict({"p": [1.0]})) != []


if __name__ == "__main__":
    pytest.main([__file__])
