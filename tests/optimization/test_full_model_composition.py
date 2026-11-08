from __future__ import annotations

import tracemalloc

import numpy as np
import pytest
import xarray as xr

from glotaran.model.clp_relation import ClpRelation
from glotaran.model.data_model import DataModel
from glotaran.model.experiment_model import ExperimentModel
from glotaran.optimization.clp_standard_error import _calculate_linear_variance
from glotaran.optimization.data import OptimizationData
from glotaran.optimization.linked_full import _prepare_dataset
from glotaran.optimization.linked_full import _relation_expansion
from glotaran.optimization.matrix import OptimizationMatrix
from glotaran.optimization.matrix import compose_full_matrix
from glotaran.optimization.objective import OptimizationObjective
from tests.optimization.elements import TestElementExponential as ExponentialElement


@pytest.mark.parametrize(
    ("paired", "expected_pairs", "expected"),
    [
        (
            True,
            [("g1", "m1"), ("g2", "m2")],
            [[2, 6], [6, 12], [55, 78], [77, 104]],
        ),
        (
            False,
            [("g1", "m1"), ("g1", "m2"), ("g2", "m1"), ("g2", "m2")],
            [[2, 4, 3, 6], [6, 8, 9, 12], [55, 66, 65, 78], [77, 88, 91, 104]],
        ),
    ],
)
def test_compose_full_matrix(
    paired: bool, expected_pairs: list[tuple[str, str]], expected: list[list[int]]
):
    local = OptimizationMatrix(["m1", "m2"], np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]))
    global_matrix = OptimizationMatrix(["g1", "g2"], np.array([[2, 3], [11, 13]]))

    pairs, matrix = compose_full_matrix(local, global_matrix, paired=paired)
    slice_pairs, matrix_slice = compose_full_matrix(
        local, global_matrix, paired=paired, global_index=1
    )

    assert pairs == slice_pairs == expected_pairs
    np.testing.assert_array_equal(matrix, expected)
    np.testing.assert_array_equal(matrix_slice, expected[2:])


@pytest.mark.parametrize("dtype", [np.int64, np.float32])
def test_relation_expansion_preserves_dense_promotion_and_zero_targets(dtype: type):
    pairs = [(label, label) for label in ("a", "b", "c", "d")]
    relations = [
        ClpRelation(source="a", target="b", parameter=0.5),
        ClpRelation(source="a", target="c", parameter=0.0),
    ]
    matrix = np.array([[2, 3, 4, 5], [7, -1, 0, 2]], dtype=dtype)
    dense = np.array([[1, 0], [0.5, 0], [0, 0], [0, 1]])
    amplitudes = np.array([2, -3], dtype=dtype)
    covariance = np.array([[5, 7], [7, 11]], dtype=dtype)

    expansion = _relation_expansion(pairs, relations, np.array([0.0, 1.0]), paired=True)

    assert expansion is not None
    assert expansion.kept == [0, 3]
    reduced = expansion.reduce(matrix)
    assert reduced.dtype == np.float64
    np.testing.assert_array_equal(reduced, matrix @ dense)
    np.testing.assert_array_equal(expansion.expand(amplitudes), dense @ amplitudes)
    np.testing.assert_array_equal(
        expansion.expand(np.diag(covariance), variance=True),
        np.diag(dense @ covariance @ dense.T),
    )


def _exponential(dimension: str, labels: list[str], rates: list[float]):
    return ExponentialElement(
        type="test-element-exponential",
        label=dimension,
        dimension=dimension,
        is_index_dependent=False,
        compartments=labels,
        rates=rates,
    )


def test_paired_variance_uses_column_order_when_combined_labels_collide():
    time = np.arange(3.0)
    spectral = np.arange(4.0)
    values = np.arange(12.0).reshape(3, 4)
    model = DataModel(
        global_composition="paired",
        elements=[_exponential("time", ["c", "b@c"], [0.2, 0.7])],
        global_elements=[_exponential("spectral", ["a@b", "a"], [0.1, 1.1])],
        data=xr.Dataset(
            {"data": (("time", "spectral"), values)}, coords={"time": time, "spectral": spectral}
        ),
    )
    objective = OptimizationObjective(ExperimentModel(datasets={"data": model}))
    decomposition = objective.get_result().optimization_results["data"].fit_decomposition
    assert decomposition is not None

    design = np.stack(
        [
            np.outer(np.exp(-0.1 * spectral), np.exp(-0.2 * time)).ravel(),
            np.outer(np.exp(-1.1 * spectral), np.exp(-0.7 * time)).ravel(),
        ],
        axis=1,
    )
    data = values.T.ravel()
    estimate, _, _, _ = np.linalg.lstsq(design, data, rcond=None)
    residual = data - design @ estimate
    covariance = np.linalg.inv(design.T @ design) * np.dot(residual, residual) / (data.size - 2)

    variance = _calculate_linear_variance(objective, {"data": decomposition.clp})["data"]

    # Both fitted pairs serialize to "a@b@c" but keep their own variances.
    assert covariance[0, 0] != pytest.approx(covariance[1, 1])
    np.testing.assert_allclose(variance, np.diag(np.diag(covariance)))


def test_cartesian_design_without_relations_avoids_dense_identity():
    count = 50
    axes = {"time": np.array([0.0, 1.0]), "spectral": np.array([0.0, 1.0, 2.0])}
    dims = ("time", "spectral")
    rates = [0.01 * (i + 1) for i in range(count)]
    model = DataModel(
        elements=[_exponential("time", [f"m{i}" for i in range(count)], rates)],
        global_elements=[_exponential("spectral", [f"g{i}" for i in range(count)], rates)],
        data=xr.Dataset({"data": (dims, np.ones((2, 3)))}, coords=axes),
    )
    data = OptimizationData(model)
    was_tracing = tracemalloc.is_tracing()
    if not was_tracing:
        tracemalloc.start()
    try:
        before, _ = tracemalloc.get_traced_memory()
        tracemalloc.reset_peak()
        dataset = _prepare_dataset(data, [])
        _, peak = tracemalloc.get_traced_memory()
    finally:
        if not was_tracing:
            tracemalloc.stop()

    assert dataset.matrix.shape == (6, count**2)
    # The design itself is 120 kB. A dense identity would allocate another 50 MB.
    assert peak - before < 10 * 1024**2
