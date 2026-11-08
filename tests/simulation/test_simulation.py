from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

import numpy as np
import pytest
import xarray as xr

from glotaran.model.errors import GlotaranUserError
from glotaran.simulation import simulate
from tests.optimization.data import TestDataModelConstantIndexDependent
from tests.optimization.data import TestDataModelConstantIndexIndependent
from tests.optimization.data import TestDataModelGlobal
from tests.optimization.elements import TestElementConstant
from tests.optimization.elements import TestElementExponential

if TYPE_CHECKING:
    from glotaran.model.data_model import DataModel
    from glotaran.typing.types import ArrayLike


@pytest.mark.parametrize("is_index_dependent", [True, False])
@pytest.mark.parametrize("noise", [True, False])
def test_simulate_from_clp(is_index_dependent: bool, noise: bool):
    model = (
        TestDataModelConstantIndexDependent
        if is_index_dependent
        else TestDataModelConstantIndexIndependent
    )
    global_axis = np.asarray([1, 2, 3, 4])
    model_axis = np.asarray([1, 2, 3])

    clp = xr.DataArray(
        [[10], [20], [30], [40]],
        coords=(
            ("global_dim", global_axis),
            ("clp_label", [f"c{2 if is_index_dependent else 1}"]),
        ),
    )

    data = simulate(
        model,
        {},
        None,
        {"global_dim": global_axis, "model_dim": model_axis},
        clp=clp,
        noise=noise,
        noise_std_dev=0.1,
    )
    assert np.array_equal(data["global_dim"], global_axis)
    assert np.array_equal(data["model_dim"], model_axis)
    assert data.data.shape == (3, 4)
    print(data.data)
    if not noise:
        if is_index_dependent:
            assert np.array_equal(
                data.data.data,
                np.array(
                    [
                        [20.0, 40.0, 60.0, 80.0],
                        [20.0, 40.0, 60.0, 80.0],
                        [20.0, 40.0, 60.0, 80.0],
                    ]
                ),
            )
        else:
            assert np.array_equal(
                data.data.data,
                np.array(
                    [
                        [50, 100, 150, 200],
                        [50, 100, 150, 200],
                        [50, 100, 150, 200],
                    ]
                ),
            )


def test_simulate_from_global_model():
    model = TestDataModelGlobal
    global_axis = np.asarray([1, 2, 3, 4])
    model_axis = np.asarray([1, 2, 3])
    data = simulate(model, {}, None, {"global_dim": global_axis, "model_dim": model_axis})
    assert np.array_equal(data["global_dim"], global_axis)
    assert np.array_equal(data["model_dim"], model_axis)
    assert data.data.shape == (3, 4)


def test_noise_without_seed_is_added_and_reproducible():
    coordinates = {"global_dim": np.asarray([1, 2, 3, 4]), "model_dim": np.asarray([1, 2, 3])}

    noise_free = simulate(TestDataModelGlobal, {}, None, coordinates)
    first = simulate(TestDataModelGlobal, {}, None, coordinates, noise=True, noise_std_dev=0.1)
    second = simulate(TestDataModelGlobal, {}, None, coordinates, noise=True, noise_std_dev=0.1)

    assert not np.allclose(first.data, noise_free.data)
    assert first.data.identical(second.data)


def test_simulate_applies_scale_list_per_global_coordinate():
    model = deepcopy(TestDataModelConstantIndexIndependent)
    model.scale_list = [2, 3, 4, 5]
    global_axis = np.asarray([1, 2, 3, 4])
    model_axis = np.asarray([1, 2, 3])
    clp = xr.DataArray(
        np.ones((global_axis.size, 1)),
        coords=(("global_dim", global_axis), ("clp_label", ["c1"])),
    )

    data = simulate(
        model,
        {},
        None,
        {"global_dim": global_axis, "model_dim": model_axis},
        clp=clp,
    )

    np.testing.assert_allclose(data.data.to_numpy()[0], [10, 15, 20, 25])


def test_simulate_paired_full_model_uses_one_amplitude_per_pair():
    model = deepcopy(TestDataModelGlobal)
    model.elements[0].compartments = ["m1", "m2"]
    model.global_elements[0].compartments = ["g1", "g2"]
    model.global_composition = "paired"
    global_axis = np.asarray([1, 2, 3, 4])
    model_axis = np.asarray([1, 2, 3])

    data = simulate(
        model,
        {},
        None,
        {"global_dim": global_axis, "model_dim": model_axis},
    )

    assert data.data.shape == (3, 4)
    np.testing.assert_allclose(data.data.to_numpy(), 40)


def test_simulate_paired_full_model_aligns_supplied_clps_by_label():
    model = deepcopy(TestDataModelGlobal)
    model.elements = [
        TestElementExponential(
            type="test-element-exponential",
            label="test_ele_model_full",
            dimension="model_dim",
            is_index_dependent=False,
            compartments=["a", "b"],
            rates=[0.1, 1.0],
        )
    ]
    model.global_elements[0].compartments = ["a", "b"]
    model.global_composition = "paired"
    model_axis = np.array([0.5, 1.0, 2.0])
    global_axis = np.array([1.0, 2.0])
    coordinates = {"model_dim": model_axis, "global_dim": global_axis}
    clp = xr.DataArray(
        [[1.0, 2.0, 100.0], [1.0, 2.0, 100.0]],
        coords=(("global_dim", global_axis), ("clp_label", ["a", "b", "unused"])),
    )

    original = simulate(model, {}, None, coordinates, clp=clp)
    reordered = simulate(model, {}, None, coordinates, clp=clp.sel(clp_label=["unused", "b", "a"]))

    expected = 4 * (np.exp(-0.1 * model_axis) + 2 * np.exp(-model_axis))
    np.testing.assert_allclose(original.data.to_numpy(), np.repeat(expected[:, None], 2, axis=1))
    xr.testing.assert_identical(original, reordered)
    with pytest.raises(KeyError):
        simulate(model, {}, None, coordinates, clp=clp.sel(clp_label=["a", "unused"]))


def test_simulate_supplied_clps_do_not_evaluate_unused_global_model():
    class GlobalElementMustBeUnused(TestElementConstant):
        def calculate_matrix(
            self, data_model: DataModel, global_axis: ArrayLike, model_axis: ArrayLike
        ) -> tuple[list[str], ArrayLike]:
            _ = data_model, global_axis, model_axis
            raise AssertionError("The unused global matrix was evaluated")

    model = deepcopy(TestDataModelGlobal)
    model.global_elements = [
        GlobalElementMustBeUnused(
            type="test-element-constant",
            label="test_ele_global_full",
            dimension="global_dim",
            compartments=["c4"],
            value=4,
            is_index_dependent=False,
        )
    ]
    model_axis = np.array([1.0, 2.0, 3.0])
    global_axis = np.array([1.0, 2.0])
    clp = xr.DataArray([[2.0], [3.0]], coords=(("global_dim", global_axis), ("clp_label", ["c4"])))

    result = simulate(
        model, {}, None, {"model_dim": model_axis, "global_dim": global_axis}, clp=clp
    )

    np.testing.assert_array_equal(result.data.to_numpy(), [[10, 15], [10, 15], [10, 15]])


def test_simulate_paired_model_requires_global_elements():
    model = deepcopy(TestDataModelConstantIndexIndependent)
    model.global_composition = "paired"

    with pytest.raises(GlotaranUserError, match="requires global elements"):
        simulate(
            model,
            {},
            None,
            {"global_dim": np.asarray([1, 2]), "model_dim": np.asarray([1, 2, 3])},
            clp=xr.DataArray(
                np.ones((2, 1)),
                coords=(("global_dim", [1, 2]), ("clp_label", ["c1"])),
            ),
        )


def test_seeded_noise_is_independent_of_full_model_dimension_order():
    original = deepcopy(TestDataModelGlobal)
    swapped = deepcopy(TestDataModelGlobal)
    swapped.elements, swapped.global_elements = swapped.global_elements, swapped.elements
    swapped.element_scale, swapped.global_element_scale = (
        swapped.global_element_scale,
        swapped.element_scale,
    )
    coordinates = {
        "global_dim": np.asarray([0, 3, 7, 10]),
        "model_dim": np.asarray([4, 11, 15]),
    }

    original_result = simulate(
        original, {}, None, coordinates, noise=True, noise_std_dev=0.1, noise_seed=42
    )
    swapped_result = simulate(
        swapped, {}, None, coordinates, noise=True, noise_std_dev=0.1, noise_seed=42
    )

    assert original_result.data.dims == ("model_dim", "global_dim")
    assert swapped_result.data.dims == ("global_dim", "model_dim")
    assert original_result.data.transpose(*sorted(original_result.data.dims)).identical(
        swapped_result.data.transpose(*sorted(swapped_result.data.dims))
    )
