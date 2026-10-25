from copy import deepcopy

import numpy as np
import pytest
import xarray as xr

from glotaran.optimization.optimization_group import OptimizationGroup
from glotaran.optimization.optimize import optimize
from glotaran.optimization.test.models import DecayModel
from glotaran.optimization.test.models import SimpleTestModel
from glotaran.optimization.test.suites import FullModel
from glotaran.optimization.test.suites import MultichannelMulticomponentDecay
from glotaran.optimization.test.suites import OneCompartmentDecay
from glotaran.optimization.test.suites import OneCompartmentDecayScaleList
from glotaran.optimization.test.suites import ThreeDatasetDecay
from glotaran.optimization.test.suites import TwoCompartmentDecay
from glotaran.parameter import Parameters
from glotaran.project import Scheme
from glotaran.simulation import simulate


@pytest.mark.parametrize("is_index_dependent", [True, False])
@pytest.mark.parametrize("link_clp", [True, False])
@pytest.mark.parametrize("weight", [True, False])
@pytest.mark.parametrize(
    "method",
    [
        "TrustRegionReflection",
        "Dogbox",
        "Levenberg-Marquardt",
    ],
)
@pytest.mark.parametrize(
    "suite",
    [OneCompartmentDecay, TwoCompartmentDecay, ThreeDatasetDecay, MultichannelMulticomponentDecay],
)
def test_optimization(suite, is_index_dependent, link_clp, weight, method):
    model = suite.model

    model.megacomplex["m1"].is_index_dependent = is_index_dependent

    print("Link CLP:", link_clp)  # T201
    print("Index dependent:", is_index_dependent)  # T201

    sim_model = suite.sim_model
    sim_model.megacomplex["m1"].is_index_dependent = is_index_dependent

    print(model.validate())  # T201
    assert model.valid()

    print(sim_model.validate())  # T201
    assert sim_model.valid()

    wanted_parameters = suite.wanted_parameters
    print(wanted_parameters)  # T201
    print(sim_model.validate(wanted_parameters))  # T201
    assert sim_model.valid(wanted_parameters)

    initial_parameters = suite.initial_parameters
    print(initial_parameters)  # T201
    print(model.validate(initial_parameters))  # T201
    assert model.valid(initial_parameters)

    nr_datasets = 3 if issubclass(suite, ThreeDatasetDecay) else 1
    data = {}
    for i in range(nr_datasets):
        global_axis = getattr(suite, "global_axis" if i == 0 else f"global_axis{i+1}")
        model_axis = getattr(suite, "model_axis" if i == 0 else f"model_axis{i+1}")

        dataset = simulate(
            sim_model,
            f"dataset{i+1}",
            wanted_parameters,
            {"global": global_axis, "model": model_axis},
        )
        print(f"Dataset {i+1}")  # T201
        print("=============")  # T201
        print(dataset.data)  # T201

        if hasattr(suite, "scale"):
            dataset["data"] /= suite.scale

        if weight:
            dataset["weight"] = xr.DataArray(
                np.ones_like(dataset.data) * 0.5, coords=dataset.data.coords
            )

        assert dataset.data.shape == (model_axis.size, global_axis.size)

        data[f"dataset{i+1}"] = dataset

    scheme = Scheme(
        model=model,
        parameters=initial_parameters,
        data=data,
        maximum_number_function_evaluations=10,
        clp_link_tolerance=0.1,
        optimization_method=method,
    )

    model.dataset_groups["default"].link_clp = link_clp

    result = optimize(scheme, raise_exception=True)
    print(result.optimized_parameters)  # T201
    print(result.data["dataset1"].fitted_data)  # T201
    assert result.success
    optimized_scheme = result.get_scheme()
    assert result.optimized_parameters != initial_parameters
    assert result.optimized_parameters == optimized_scheme.parameters
    for dataset in optimized_scheme.data.values():
        assert "fitted_data" not in dataset
        if weight:
            assert "weight" in dataset
    for param in result.optimized_parameters.all():
        if param.vary:
            assert np.allclose(param.value, wanted_parameters.get(param.label).value, rtol=1e-1)

    for i, dataset in enumerate(data.values()):
        resultdata = result.data[f"dataset{i+1}"]
        print(f"Result Data {i+1}")  # T201k
        print("=================")  # T201
        print(resultdata)  # T201
        assert "matrix" in resultdata
        assert len(resultdata.matrix.shape) == (3 if is_index_dependent else 2)
        assert "residual" in resultdata
        assert "residual_left_singular_vectors" in resultdata
        assert "residual_right_singular_vectors" in resultdata
        assert "residual_singular_values" in resultdata
        assert np.array_equal(dataset.coords["model"], resultdata.coords["model"])
        assert np.array_equal(dataset.coords["global"], resultdata.coords["global"])
        assert dataset.data.shape == resultdata.data.shape
        print(dataset.data[0, 0], resultdata.data[0, 0])  # T201
        assert np.allclose(dataset.data, resultdata.data)
        if weight:
            assert "weight" in resultdata
            assert "weighted_residual" in resultdata
            assert "weighted_residual_left_singular_vectors" in resultdata
            assert "weighted_residual_right_singular_vectors" in resultdata
            assert "weighted_residual_singular_values" in resultdata


@pytest.mark.parametrize("index_dependent", [True, False])
def test_optimization_full_model(index_dependent):
    model = FullModel.model
    model.megacomplex["m1"].is_index_dependent = index_dependent

    print(model.validate())  # T201
    assert model.valid()

    parameters = FullModel.parameters
    assert model.valid(parameters)

    dataset = simulate(model, "dataset1", parameters, FullModel.coordinates)

    scheme = Scheme(
        model=model,
        parameters=parameters,
        data={"dataset1": dataset},
        maximum_number_function_evaluations=10,
    )

    result = optimize(scheme, raise_exception=True)
    assert result.success
    optimized_scheme = result.get_scheme()
    assert result.optimized_parameters == optimized_scheme.parameters

    result_data = result.data["dataset1"]
    assert "fitted_data" in result_data
    for param in result.optimized_parameters.all():
        if param.vary:
            assert np.allclose(param.value, parameters.get(param.label).value, rtol=1e-1)

    clp = result_data.clp
    print(clp)  # T201
    assert clp.shape == (4, 4)
    assert all(np.isclose(1.0, c) for c in np.diagonal(clp))


def test_link_clp_full_model_requires_consistent_single_amplitude_setting():
    model = DecayModel(
        **{
            "dataset_groups": {"default": {"link_clp": True}},
            "megacomplex": {
                "m1": {"type": "simple-kinetic-test-mc", "is_index_dependent": False},
                "m2": {"type": "simple-spectral-test-mc"},
            },
            "dataset": {
                "dataset1": {
                    "megacomplex": ["m1"],
                    "global_megacomplex": ["m2"],
                    "kinetic": ["1"],
                    "single_amplitude_model": True,
                },
                "dataset2": {
                    "megacomplex": ["m1"],
                    "global_megacomplex": ["m2"],
                    "kinetic": ["1"],
                },
            },
        }
    )
    data = xr.DataArray(
        np.ones((2, 2)), coords=[("model", [1, 2]), ("global", [1, 2])]
    ).to_dataset(name="data")
    scheme = Scheme(
        model,
        Parameters.from_list([0.1]),
        {"dataset1": data, "dataset2": data.copy()},
    )

    with pytest.raises(ValueError, match="single_amplitude_model.*consistently"):
        optimize(scheme)


def test_link_clp_full_model_applies_dataset_scale():
    model = deepcopy(FullModel.model)
    model.dataset_groups["default"].link_clp = True
    model.dataset["dataset1"].single_amplitude_model = True
    model.dataset["dataset1"].scale = "scale.1"

    parameter_dicts = FullModel.parameters.to_parameter_dict_list()
    parameter_dicts.append({"label": "scale.1", "value": 1.0})
    parameters = Parameters.from_parameter_dict_list(parameter_dicts)
    coordinates = FullModel.coordinates
    data = xr.DataArray(
        np.ones((coordinates["model"].size, coordinates["global"].size)),
        coords=[("model", coordinates["model"]), ("global", coordinates["global"])],
    ).to_dataset(name="data")
    scheme = Scheme(
        model=model,
        parameters=parameters,
        data={"dataset1": data},
        add_svd=False,
    )
    group = OptimizationGroup(scheme, model.get_dataset_groups()["default"])

    group.calculate(parameters)
    unscaled_matrix = group._matrix_provider.get_aligned_full_matrix("dataset1").copy()
    parameters.get("scale.1").value = 2.0
    group.calculate(parameters)
    scaled_matrix = group._matrix_provider.get_aligned_full_matrix("dataset1")

    assert np.allclose(scaled_matrix, 2.0 * unscaled_matrix)


@pytest.mark.parametrize("single_amplitude_model", [False, True])
@pytest.mark.parametrize(
    "residual_function",
    ["variable_projection", "non_negative_least_squares"],
)
def test_link_clp_full_model(single_amplitude_model, residual_function):
    model_dict = {
        "dataset_groups": {
            "default": {
                "link_clp": True,
                "residual_function": residual_function,
            }
        },
        "megacomplex": {
            "m1": {"type": "simple-kinetic-test-mc", "is_index_dependent": False},
            "m2": {
                "type": "shaped-spectral-test-mc",
                "location": ["loc.1", "loc.2", "loc.3", "loc.4"],
                "delta": ["del.1", "del.2", "del.3", "del.4"],
                "amplitude": ["amp.1", "amp.2", "amp.3", "amp.4"],
            },
        },
        "dataset": {},
    }
    dataset_labels = (
        ["dataset1", "dataset2"] if single_amplitude_model else ["dataset1", "dataset3"]
    )
    for label in dataset_labels:
        model_dict["dataset"][label] = {
            "megacomplex": ["m1"],
            "global_megacomplex": ["m2"],
            "kinetic": ["k.1", "k.2", "k.3", "k.4"],
            "single_amplitude_model": single_amplitude_model,
        }

    model = DecayModel(**model_dict)
    data = {
        label: simulate(model, label, FullModel.parameters, FullModel.coordinates)
        for label in dataset_labels
    }
    scheme = Scheme(
        model=model,
        parameters=FullModel.parameters,
        data=data,
        maximum_number_function_evaluations=10,
    )

    result = optimize(scheme, raise_exception=True)

    assert result.success
    clp1 = result.data[dataset_labels[0]].clp
    clp2 = result.data[dataset_labels[1]].clp
    assert clp1.dims == ("global_clp_label", "clp_label")
    assert clp2.dims == ("global_clp_label", "clp_label")
    expected_global_labels = ["s1", "s2", "s3", "s4"]
    if single_amplitude_model:
        assert clp1.coords["global_clp_label"].values.tolist() == expected_global_labels
    else:
        assert clp1.coords["global_clp_label"].values.tolist() == expected_global_labels
        assert clp2.coords["global_clp_label"].values.tolist() == ["s2", "s3", "s4", "s5"]
        assert clp2.coords["clp_label"].values.tolist() == ["s2", "s3", "s4", "s5"]
    if single_amplitude_model:
        assert np.allclose(clp1.values, clp2.values)
        assert np.allclose(clp1.values, np.diag(np.diag(clp1.values)))
    else:
        for global_label in ["s2", "s3", "s4"]:
            for clp_label in ["s2", "s3", "s4"]:
                assert clp1.sel(
                    global_clp_label=global_label,
                    clp_label=clp_label,
                ).item() == pytest.approx(
                    clp2.sel(
                        global_clp_label=global_label,
                        clp_label=clp_label,
                    ).item()
                )


@pytest.mark.parametrize("model_weight", [True, False])
@pytest.mark.parametrize("index_dependent", [True, False])
def test_result_data(model_weight: bool, index_dependent: bool):
    global_axis = [1, 5, 6]
    model_axis = [5, 7, 9, 12]

    data = xr.DataArray(
        np.ones((4, 3)), coords=[("model", model_axis), ("global", global_axis)]
    ).to_dataset(name="data")

    model_dict = {
        "megacomplex": {"m1": {"type": "simple-test-mc", "is_index_dependent": index_dependent}},
        "dataset": {"dataset1": {"megacomplex": ["m1"]}},
    }

    if model_weight:
        model_dict["weights"] = [{"datasets": ["dataset1"], "value": 0.5}]
    else:
        data["weight"] = xr.ones_like(data.data) * 0.5

    model = SimpleTestModel(**model_dict)
    assert model.valid()
    parameters = Parameters.from_list([1])

    scheme = Scheme(model, parameters, {"dataset1": data}, maximum_number_function_evaluations=1)
    result = optimize(scheme, raise_exception=True)
    result_data = result.data["dataset1"]
    wanted = [
        ("data", ("model", "global")),
        ("data_left_singular_vectors", ("model", "left_singular_value_index")),
        ("data_singular_values", ("singular_value_index",)),
        ("data_right_singular_vectors", ("global", "right_singular_value_index")),
        ("clp", ("global", "clp_label")),
        ("weight", ("model", "global")),
        ("weighted_residual", ("model", "global")),
        ("residual", ("model", "global")),
        ("weighted_residual_left_singular_vectors", ("model", "left_singular_value_index")),
        ("weighted_residual_singular_values", ("singular_value_index",)),
        ("weighted_residual_right_singular_vectors", ("global", "right_singular_value_index")),
        ("residual_left_singular_vectors", ("model", "left_singular_value_index")),
        ("residual_singular_values", ("singular_value_index",)),
        ("residual_right_singular_vectors", ("global", "right_singular_value_index")),
        ("fitted_data", ("model", "global")),
    ]

    for label, dims in wanted:
        print("Check label", label)
        assert label in result_data
        print("Check dims", result_data[label].dims, dims)
        assert result_data[label].dims == dims

    assert "matrix" in result_data

    if index_dependent:
        assert result_data.matrix.dims == ("global", "model", "clp_label")
    else:
        assert result_data.matrix.dims == ("model", "clp_label")


def test_result_data_with_clp_standard_error():
    model = OneCompartmentDecay.model
    model.megacomplex["m1"].is_index_dependent = False

    sim_model = OneCompartmentDecay.sim_model
    sim_model.megacomplex["m1"].is_index_dependent = False

    data = simulate(
        sim_model,
        "dataset1",
        OneCompartmentDecay.wanted_parameters,
        {"global": OneCompartmentDecay.global_axis, "model": OneCompartmentDecay.model_axis},
    )
    scheme = Scheme(
        model=model,
        parameters=OneCompartmentDecay.initial_parameters,
        data={"dataset1": data},
        maximum_number_function_evaluations=10,
        compute_clp_standard_error=True,
        clp_standard_error_finite_difference_relative_step=1e-6,
    )

    result = optimize(scheme, raise_exception=True)

    assert result.success
    assert result.clp_standard_error_method == "linear_plus_nonlinear_propagation"
    assert result.clp_standard_error_finite_difference_relative_step == pytest.approx(1e-6)

    result_data = result.data["dataset1"]
    assert "clp_standard_error" in result_data
    assert result_data["clp_standard_error"].dims == result_data["clp"].dims
    assert np.isfinite(result_data["clp_standard_error"].values).all()
    assert result_data["clp_standard_error"].attrs["method"] == "linear_plus_nonlinear_propagation"


@pytest.mark.parametrize("link_clp", [True, False])
def test_optimization_scale_list(link_clp):
    """Verify that a per-wavelength scale list is applied correctly during optimization."""
    suite = OneCompartmentDecayScaleList

    model = suite.model
    model.megacomplex["m1"].is_index_dependent = False

    sim_model = suite.sim_model
    sim_model.megacomplex["m1"].is_index_dependent = False

    assert model.valid(suite.initial_parameters)

    data = simulate(
        sim_model,
        "dataset1",
        suite.wanted_parameters,
        {"global": suite.global_axis, "model": suite.model_axis},
    )
    # Divide each wavelength by its scale so the model (which applies that scale) recovers
    # the original rates.
    data["data"] /= suite.scale

    model.dataset_groups["default"].link_clp = link_clp

    scheme = Scheme(
        model=model,
        parameters=suite.initial_parameters,
        data={"dataset1": data},
        maximum_number_function_evaluations=10,
        clp_link_tolerance=0.1,
    )
    result = optimize(scheme, raise_exception=True)
    assert result.success

    # Check that the kinetic rate was recovered.
    for param in result.optimized_parameters.all():
        if param.vary:
            assert np.allclose(
                param.value, suite.wanted_parameters.get(param.label).value, rtol=1e-1
            )

    # Check that dataset_scale_list stores the list of per-wavelength scale values.
    result_data = result.data["dataset1"]
    result_scale = result_data.attrs["dataset_scale_list"]
    assert isinstance(result_scale, list)
    assert np.allclose(result_scale, suite.scale)

    # The stored matrix must reflect the same per-wavelength scaling used during the solve.
    # This should apply each scale exactly once, not quadratically.
    assert result_data.matrix.dims == ("global", "model", "clp_label")
    assert np.allclose(result_data.matrix.isel(model=0, clp_label=0).values, suite.scale)


def test_optimization_dataset_scale_list_serializable_without_scale_list():
    """Store dataset_scale_list as an empty list when no per-wavelength scale list exists."""
    suite = OneCompartmentDecay

    model = suite.model
    model.megacomplex["m1"].is_index_dependent = False

    sim_model = suite.sim_model
    sim_model.megacomplex["m1"].is_index_dependent = False

    data = simulate(
        sim_model,
        "dataset1",
        suite.wanted_parameters,
        {"global": suite.global_axis, "model": suite.model_axis},
    )

    scheme = Scheme(
        model=model,
        parameters=suite.initial_parameters,
        data={"dataset1": data},
        maximum_number_function_evaluations=5,
        clp_link_tolerance=0.1,
    )

    result = optimize(scheme, raise_exception=True)
    assert result.success
    assert result.data["dataset1"].attrs["dataset_scale_list"] == []
