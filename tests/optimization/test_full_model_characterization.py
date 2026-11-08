from __future__ import annotations

from itertools import product
from typing import TYPE_CHECKING

import numpy as np
import pytest
import xarray as xr

from glotaran.model.data_model import DataModel
from glotaran.model.experiment_model import ExperimentModel
from glotaran.optimization.clp_standard_error import _calculate_linear_variance
from glotaran.optimization.data import OptimizationData
from glotaran.optimization.matrix import OptimizationMatrix
from glotaran.optimization.objective import OptimizationObjective
from glotaran.simulation import simulate
from tests.optimization.elements import TestElementExponential as ExponentialElement

if TYPE_CHECKING:
    from typing import Literal


COMPOSITIONS = [
    pytest.param("paired", False, id="paired-by-position"),
    pytest.param("paired", True, id="paired-by-label"),
    pytest.param("cartesian", False, id="cartesian"),
]


def _model_and_basis(mode: Literal["paired", "cartesian"], align: bool, offset: float = 0):
    time = np.array([0.0, 0.7, 2.0, 4.5, 7.0]) + offset
    spectral = np.array([0.0, 0.9, 2.4, 5.0]) + offset
    local_rates = {"a": 0.15, "b": 0.9}
    global_rates = {"b": 0.65, "a": 0.08}
    if mode == "cartesian":
        global_rates["extra"] = 1.4

    model = DataModel(
        global_composition=mode,
        align_paired_clp_labels=align,
        scale_list=[0.9, 1.7, 0.6, 2.1],
        elements=[
            ExponentialElement(
                type="test-element-exponential",
                label="local",
                dimension="time",
                is_index_dependent=True,
                compartments=list(local_rates),
                rates=list(local_rates.values()),
            )
        ],
        global_elements=[
            ExponentialElement(
                type="test-element-exponential",
                label="global",
                dimension="spectral",
                is_index_dependent=False,
                compartments=list(global_rates),
                rates=list(global_rates.values()),
            )
        ],
    )
    local = {label: np.exp(-rate * time) for label, rate in local_rates.items()}
    global_ = {label: np.exp(-rate * spectral) for label, rate in global_rates.items()}
    return model, {"time": time, "spectral": spectral}, local, global_


@pytest.mark.parametrize(("mode", "align"), COMPOSITIONS)
@pytest.mark.parametrize("linked", [False, True], ids=["single", "linked"])
def test_full_fit_matrix_residual_amplitudes_and_covariance(
    mode: Literal["paired", "cartesian"], align: bool, linked: bool
):
    models = {}
    scales = {}
    columns_by_dataset = {}
    weighted_matrices = []
    weighted_observations = []
    for index in range(2 if linked else 1):
        label = f"dataset_{index}"
        model, coordinates, local, global_ = _model_and_basis(mode, align, offset=index / 3)
        global_labels = list(local) if align else list(global_)
        pairs = (
            list(zip(global_labels, local, strict=True))
            if mode == "paired"
            else list(product(global_labels, local))
        )
        scales[label] = 1.3 + index
        columns = np.stack([np.outer(local[m], global_[g]) for g, m in pairs], axis=-1)
        columns *= scales[label] * np.asarray(model.scale_list)[None, :, None]
        sample = np.arange(columns.shape[0] * columns.shape[1]).reshape(columns.shape[:2])
        observations = columns @ np.linspace(0.5, 2.0, len(pairs)) + 0.01 * np.sin(sample + index)
        weights = 0.5 + (sample + index) / sample.size
        dims = ("time", "spectral")
        model.data = xr.Dataset(
            {"data": (dims, observations), "weight": (dims, weights)}, coords=coordinates
        )
        weighted = (columns * weights[..., None]).transpose(1, 0, 2).reshape(-1, len(pairs))
        if not linked:
            _, _, full = OptimizationMatrix.from_global_data(
                OptimizationData(model, scale=scales[label])
            )
            assert full.clp_axis == [f"{g}@{m}" for g, m in pairs]
            np.testing.assert_allclose(full.array, weighted)
        models[label] = model
        columns_by_dataset[label] = columns
        weighted_matrices.append(weighted)
        weighted_observations.append((observations * weights).T.ravel())

    design = np.vstack(weighted_matrices)
    observed = np.concatenate(weighted_observations)
    amplitudes, _, rank, _ = np.linalg.lstsq(design, observed, rcond=None)
    assert rank == len(pairs)
    residual = observed - design @ amplitudes
    residual_variance = np.dot(residual, residual) / (observed.size - rank)
    variance = np.diag(np.linalg.inv(design.T @ design)) * residual_variance
    clp_shape = (len(global_labels), len(local))
    expected_clp = np.diag(amplitudes) if mode == "paired" else amplitudes.reshape(clp_shape)
    expected_variance = np.diag(variance) if mode == "paired" else variance.reshape(clp_shape)

    objective = OptimizationObjective(ExperimentModel(datasets=models, scale=scales))
    np.testing.assert_allclose(objective.calculate(), residual, atol=1e-10)
    result = objective.get_result()
    assert result.clp_size == len(pairs)
    clps = {}
    for label, dataset in result.optimization_results.items():
        decomposition = dataset.fit_decomposition
        assert decomposition is not None
        clp = decomposition.clp
        assert clp.coords["global_clp_label"].values.tolist() == global_labels
        assert clp.coords["clp_label"].values.tolist() == list(local)
        np.testing.assert_allclose(clp, expected_clp)
        np.testing.assert_allclose(dataset.fitted_data, columns_by_dataset[label] @ amplitudes)
        clps[label] = clp

    for actual in _calculate_linear_variance(objective, clps).values():
        np.testing.assert_allclose(actual, expected_variance, rtol=1e-9, atol=0)


@pytest.mark.parametrize(("mode", "align"), COMPOSITIONS)
def test_full_simulation_preserves_pairing_and_coordinate_scales(
    mode: Literal["paired", "cartesian"], align: bool
):
    model, coordinates, local, global_ = _model_and_basis(mode, align)
    # Cartesian simulation contracts equal labels even when a global-only column exists.
    global_labels = list(global_) if mode == "paired" and not align else list(local)
    expected = sum(
        np.outer(local[m], global_[g]) for g, m in zip(global_labels, local, strict=True)
    )
    expected *= np.asarray(model.scale_list)[None, :]

    result = simulate(model, {}, None, coordinates)

    assert result.data.dims == ("time", "spectral")
    np.testing.assert_allclose(result.data, expected)
