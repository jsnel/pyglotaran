from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
import xarray as xr

from glotaran.model.data_model import resolve_data_model
from glotaran.optimization.data import LinkedOptimizationData
from glotaran.optimization.data import OptimizationData
from glotaran.parameter import Parameters
from tests.optimization.data import TestDataModelConstantIndexDependent
from tests.optimization.data import TestDataModelConstantIndexIndependent
from tests.optimization.data import TestDataModelGlobal
from tests.optimization.library import test_library


def test_data_model_has_original_dataset_attributes():
    """Test that original dataset attributes are stored in OptimizationData."""
    data_model = deepcopy(TestDataModelConstantIndexIndependent)
    data = OptimizationData(data_model)

    assert data.original_dataset_attributes == {"test_attr": "test_value"}


@pytest.mark.parametrize("weight", [True, False])
def test_optimization_data(weight: bool):
    data_model = deepcopy(TestDataModelConstantIndexIndependent)
    if weight:
        data_model.data["weight"] = xr.ones_like(data_model.data.data) * 0.5
    data = OptimizationData(data_model)

    dataset = data_model.data
    assert data.model_dimension == "model_dim"
    assert data.global_dimension == "global_dim"
    assert np.array_equal(dataset.coords["model_dim"], data.model_axis)
    assert np.array_equal(dataset.coords["global_dim"], data.global_axis)
    if weight:
        assert np.array_equal(dataset.data * dataset.weight, data.data)
        assert np.array_equal(dataset.weight, data.weight)
    else:
        assert np.array_equal(dataset.data, data.data)


def test_scale_list_resolves_parameter_values_at_access_time():
    data_model = deepcopy(TestDataModelConstantIndexIndependent)
    data_model.scale_list = ["scale.1", "scale.2", "scale.3"]
    parameters = Parameters.from_list(
        [["scale.1", 1.0], ["scale.2", 2.0], ["scale.3", 3.0], ["dataset_scale", 4.0]]
    )

    resolved = resolve_data_model(data_model, test_library, parameters)
    data = OptimizationData(resolved, scale=parameters.get("dataset_scale"))

    np.testing.assert_allclose(data.scale_list, [1.0, 2.0, 3.0])
    assert data.dataset_scale == 4.0
    parameters.get("scale.2").value = 7.0
    parameters.get("dataset_scale").value = 8.0
    np.testing.assert_allclose(data.scale_list, [1.0, 7.0, 3.0])
    assert data.dataset_scale == 8.0


@pytest.mark.parametrize("weight", [True, False])
def test_optimization_data_global_model(weight: bool):
    data_model = deepcopy(TestDataModelGlobal)
    if weight:
        data_model.data["weight"] = xr.ones_like(data_model.data.data) * 0.5
    data = OptimizationData(data_model)

    dataset = data_model.data
    print(dataset.data)
    assert data.model_dimension == "model_dim"
    assert data.global_dimension == "global_dim"
    assert np.array_equal(dataset.coords["model_dim"], data.model_axis)
    assert np.array_equal(dataset.coords["global_dim"], data.global_axis)
    if weight:
        assert np.array_equal(
            dataset.data.data.T.flatten() * dataset.weight.data.T.flatten(), data.flat_data
        )
        assert np.array_equal(dataset.weight.data.T.flatten(), data.flat_weight)
    else:
        assert np.array_equal(dataset.data.data.T.flatten(), data.flat_data)


def test_linked_optimization_data():
    data_model_one = deepcopy(TestDataModelConstantIndexIndependent)
    data_model_one.data["weight"] = xr.ones_like(data_model_one.data.data) * 0.5
    data_model_two = deepcopy(TestDataModelConstantIndexDependent)
    all_data = {
        "dataset1": OptimizationData(data_model_one),
        "dataset2": OptimizationData(data_model_two),
    }
    tolerance, method = 1, "nearest"
    data = LinkedOptimizationData(all_data, tolerance, method, scales={"dataset2": 4})

    dataset_one = data_model_one.data
    dataset_two = data_model_two.data

    assert "dataset1" in data.group_definitions
    assert data.group_definitions["dataset1"] == ["dataset1"]
    assert "dataset2" in data.group_definitions
    assert data.group_definitions["dataset2"] == ["dataset2"]
    assert "dataset1dataset2" in data.group_definitions
    assert data.group_definitions["dataset1dataset2"] == ["dataset1", "dataset2"]

    assert np.array_equal(data.global_axis, [1, 3, 5, 6, 10])

    assert len(data.group_labels) == data.global_axis.size
    assert data.group_labels[0] == "dataset1dataset2"
    assert data.group_labels[1] == "dataset2"
    assert data.group_labels[2] == "dataset1"
    assert data.group_labels[3] == "dataset1dataset2"
    assert data.group_labels[4] == "dataset2"

    assert len(data.data_indices) == data.global_axis.size
    assert np.array_equal(data.data_indices[0], [0, 0])
    assert np.array_equal(data.data_indices[1], [1])
    assert np.array_equal(data.data_indices[2], [1])
    assert np.array_equal(data.data_indices[3], [2, 2])
    assert np.array_equal(data.data_indices[4], [3])

    dataset1_size = dataset_one.coords["model_dim"].size
    dataset2_size = dataset_two.coords["model_dim"].size

    assert data.data_slices[0].size == dataset1_size + dataset2_size
    assert data.data_slices[1].size == dataset2_size
    assert data.data_slices[2].size == dataset1_size
    assert data.data_slices[3].size == dataset1_size + dataset2_size
    assert data.data_slices[4].size == dataset2_size


@pytest.mark.parametrize("method", ["nearest", "backward", "forward"])
def test_linking_methods(method: str):
    all_data = {
        "dataset1": OptimizationData(TestDataModelConstantIndexIndependent),
        "dataset2": OptimizationData(TestDataModelConstantIndexDependent),
    }
    tolerance = 1
    data = LinkedOptimizationData(all_data, tolerance, method, {})

    wanted_global_axis = [1, 3, 5, 6, 10]
    if method == "backward":
        wanted_global_axis = [0, 1, 3, 5, 6, 10]
    elif method == "forward":
        wanted_global_axis = [1, 3, 5, 6, 7, 10]
    assert np.array_equal(data.global_axis, wanted_global_axis)
