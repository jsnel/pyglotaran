import numpy as np
import pytest
import xarray as xr

from glotaran.builtin.megacomplexes.clp_guide import ClpGuideMegacomplex
from glotaran.builtin.megacomplexes.clp_guide import SpectralModelClpGuideMegacomplex
from glotaran.builtin.megacomplexes.decay import DecaySequentialMegacomplex
from glotaran.builtin.megacomplexes.decay.test.test_decay_megacomplex import create_gaussian_clp
from glotaran.model import Model
from glotaran.optimization.data_provider import DataProvider
from glotaran.optimization.data_provider import prepare_generated_datasets
from glotaran.optimization.optimize import optimize
from glotaran.parameter import Parameters
from glotaran.project import Scheme
from glotaran.simulation.simulation import simulate


def test_clp_guide():
    model = Model.create_class_from_megacomplexes(
        [DecaySequentialMegacomplex, ClpGuideMegacomplex]
    )(
        **{
            "dataset_groups": {"default": {"link_clp": True}},
            "megacomplex": {
                "mc1": {
                    "type": "decay-sequential",
                    "compartments": ["s1", "s2"],
                    "rates": ["1", "2"],
                },
                "mc2": {"type": "clp-guide", "dimension": "time", "target": "s1"},
            },
            "dataset": {
                "dataset1": {"megacomplex": ["mc1"]},
                "dataset2": {"megacomplex": ["mc2"]},
            },
        },
    )

    initial_parameters = Parameters.from_list(
        [101e-5, 501e-4, [1, {"vary": False, "non-negative": False}]]
    )
    wanted_parameters = Parameters.from_list(
        [101e-4, 501e-3, [1, {"vary": False, "non-negative": False}]]
    )

    time = np.arange(0, 50, 1.5)
    pixel = np.arange(600, 750, 5)
    axis = {"time": time, "pixel": pixel}

    clp = create_gaussian_clp(["s1", "s2"], [7, 30], [620, 720], [10, 50], pixel)

    dataset1 = simulate(model, "dataset1", wanted_parameters, axis, clp)
    dataset2 = clp.sel(clp_label=["s1"]).rename(clp_label="time")
    data = {"dataset1": dataset1, "dataset2": dataset2}

    scheme = Scheme(
        model=model,
        parameters=initial_parameters,
        data=data,
        maximum_number_function_evaluations=20,
    )
    result = optimize(scheme)
    print(result.optimized_parameters)
    for param in result.optimized_parameters.all():
        assert np.allclose(param.value, wanted_parameters.get(param.label).value, rtol=1e-1)


def test_spectral_model_clp_guide_generates_and_refreshes_missing_dataset():
    model = Model.create_class_from_megacomplexes([SpectralModelClpGuideMegacomplex])(
        dataset_groups={"default": {"link_clp": True}},
        megacomplex={
            "guide": {
                "type": "spectral-model-clp-guide",
                "target": "s1",
                "shape": {"s1": "shape1"},
            },
        },
        shape={
            "shape1": {
                "type": "gaussian",
                "amplitude": "shape.amplitude",
                "location": "shape.location",
                "width": "shape.width",
            },
        },
        dataset={"guide_data": {"megacomplex": ["guide"]}},
    )
    parameters = Parameters.from_list(
        [
            ["shape.amplitude", 2.0],
            ["shape.location", 650.0],
            ["shape.width", 20.0],
        ]
    )
    real_data = xr.Dataset(
        {"data": (("time", "spectral"), np.zeros((2, 3)))},
        coords={"time": [0.0, 1.0], "spectral": [640.0, 650.0, 660.0]},
    )
    scheme = Scheme(model=model, parameters=parameters, data={"real_data": real_data})

    prepare_generated_datasets(scheme)
    guide_data = scheme.data["guide_data"]
    np.testing.assert_allclose(guide_data.data.values, [[1.0, 2.0, 1.0]])
    np.testing.assert_array_equal(guide_data.spectral.values, real_data.spectral.values)

    dataset_group = model.get_dataset_groups()["default"]
    dataset_group.set_parameters(parameters)
    data_provider = DataProvider(scheme, dataset_group)
    parameters.get("shape.amplitude").value = 3.0
    dataset_group.set_parameters(parameters)
    data_provider.update_generated_data()

    assert data_provider.get_data("guide_data")[0, 1] == 3.0
    np.testing.assert_allclose(real_data.data.values, 0.0)

    optimization_scheme = Scheme(model=model, parameters=parameters, data={"real_data": real_data})
    result = optimize(optimization_scheme, verbose=False, raise_exception=True)
    assert "guide_data" in result.data


def test_spectral_model_clp_guide_applies_spectral_axis_transform():
    model = Model.create_class_from_megacomplexes([SpectralModelClpGuideMegacomplex])(
        dataset={
            "guide_data": {
                "megacomplex": ["guide"],
                "spectral_axis_inverted": True,
                "spectral_axis_scale": 1e7,
            }
        },
        megacomplex={
            "guide": {
                "type": "spectral-model-clp-guide",
                "target": "s1",
                "shape": {"s1": "shape1"},
            }
        },
        shape={
            "shape1": {
                "type": "gaussian",
                "amplitude": "shape.amplitude",
                "location": "shape.location",
                "width": "shape.width",
            }
        },
    )
    parameters = Parameters.from_list(
        [
            ["shape.amplitude", 2.0],
            ["shape.location", 1e7 / 650.0],
            ["shape.width", 100.0],
        ]
    )
    real_data = xr.Dataset(
        {"data": (("time", "spectral"), np.zeros((1, 3)))},
        coords={"time": [0.0], "spectral": [640.0, 650.0, 660.0]},
    )
    scheme = Scheme(model=model, parameters=parameters, data={"real_data": real_data})

    prepare_generated_datasets(scheme)

    generated = scheme.data["guide_data"].data.values[0]
    assert generated[1] == pytest.approx(2.0)
    assert generated[1] > generated[0]
    assert generated[1] > generated[2]
