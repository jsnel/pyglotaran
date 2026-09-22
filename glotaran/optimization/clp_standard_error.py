"""Utilities for conditionally linear parameter (CLP) uncertainty estimates.

The nonlinear optimizer exposes a covariance matrix for the fitted parameters.  CLPs are
solved inside every objective evaluation, so their uncertainty has two parts: the linear
least-squares covariance at the optimum and the covariance propagated through the
nonlinear model.  This module keeps that calculation separate from result construction.

The finite-difference evaluator supplied by
:class:`~glotaran.optimization.optimization.Optimization` temporarily updates the parameter
objects shared by an objective. It restores every value in a ``finally`` block, including when
matrix construction or result creation raises. Callers that already have a pure evaluator can
pass it through ``evaluate_at`` instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from warnings import warn

import numpy as np
import xarray as xr

from glotaran.model.data_model import is_data_model_single_amplitude
from glotaran.optimization.data import LinkedOptimizationData
from glotaran.optimization.data import OptimizationData
from glotaran.optimization.estimation import OptimizationEstimation
from glotaran.optimization.matrix import OptimizationMatrix

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Mapping

    from glotaran.optimization.objective import OptimizationObjective
    from glotaran.optimization.objective import OptimizationResult
    from glotaran.parameter import Parameters
    from glotaran.typing.types import ArrayLike


@dataclass(frozen=True)
class ClpStandardErrorSettings:
    """Numerical settings for CLP standard-error calculation."""

    relative_step: float = 1e-6

    def __post_init__(self) -> None:
        """Validate the finite-difference step."""
        if not np.isfinite(self.relative_step) or self.relative_step <= 0:
            msg = "The CLP standard-error finite-difference step must be positive."
            raise ValueError(msg)


def calculate_clp_standard_error(
    optimization_objective: OptimizationObjective,
    parameters: Parameters,
    free_parameter_labels: list[str],
    covariance_matrix: ArrayLike,
    settings: ClpStandardErrorSettings,
    *,
    base_results: Mapping[str, OptimizationResult] | None = None,
    evaluate_at: Callable[[Parameters], Mapping[str, OptimizationResult]] | None = None,
) -> dict[str, xr.DataArray]:
    """Calculate CLP standard errors for one staging optimization objective.

    ``covariance_matrix`` must be the *scaled* covariance of the nonlinear parameters,
    ``sigma**2 * (J.T @ J)^-1``.  Staging stores the unscaled pseudo-inverse in
    :class:`~glotaran.optimization.info.OptimizationInfo`; the optimizer applies the reduced
    chi-square once before calling this function.

    The result contains one ``xarray.DataArray`` per dataset, with coordinates and dimensions
    copied from that dataset's fitted CLP array.  Variable-projection objectives support
    unlinked, linked, global, constrained, and relation-reduced matrices.  Other residual
    functions are explicitly skipped because their CLP solve is not the linear least-squares
    solve used by the uncertainty calculation.
    """
    if getattr(optimization_objective.model, "residual_function", None) != ("variable_projection"):
        warn(
            "Skipping CLP standard-error calculation because residual function is not "
            "'variable_projection'.",
            stacklevel=2,
        )
        return {}

    if not np.isfinite(settings.relative_step) or settings.relative_step <= 0:
        msg = "The CLP standard-error finite-difference step must be positive."
        raise ValueError(msg)

    if base_results is None:
        base_results = optimization_objective.get_result().optimization_results
    base_clps = _get_clps(base_results)
    if not base_clps:
        return {}

    base_vector, slices = _flatten_clps(base_clps)
    covariance = np.asarray(covariance_matrix, dtype=float)
    parameter_count = len(free_parameter_labels)
    if covariance.shape != (parameter_count, parameter_count):
        msg = (
            "The nonlinear covariance matrix shape must match the free parameter count: "
            f"expected {(parameter_count, parameter_count)}, got {covariance.shape}."
        )
        raise ValueError(msg)

    sensitivity = _calculate_sensitivity(
        optimization_objective,
        parameters,
        free_parameter_labels,
        base_vector,
        slices,
        settings,
        evaluate_at=evaluate_at,
    )
    propagated_variance = np.einsum(
        "ij,jk,ik->i", sensitivity, covariance, sensitivity, optimize=True
    )
    propagated_variance = np.maximum(propagated_variance, 0.0)

    linear_variance = _calculate_linear_variance(optimization_objective, base_clps)
    result: dict[str, xr.DataArray] = {}
    for label, clp in base_clps.items():
        dataset_slice = slices[label]
        nonlinear = propagated_variance[dataset_slice].reshape(clp.shape)
        total = np.maximum(linear_variance[label] + nonlinear, 0.0)
        result[label] = xr.DataArray(
            np.sqrt(total),
            coords=clp.coords,
            dims=clp.dims,
            attrs={
                "method": "linear_plus_nonlinear_propagation",
                "linear_covariance": "pseudo_inverse",
                "nonlinear_covariance": "finite_difference",
                "fd_relative_step": settings.relative_step,
            },
        )
    return result


def _get_clps(
    results: Mapping[str, OptimizationResult],
) -> dict[str, xr.DataArray]:
    result: dict[str, xr.DataArray] = {}
    for label, optimization_result in results.items():
        decomposition = optimization_result.fit_decomposition
        if decomposition is None:
            continue
        result[label] = decomposition.clp
    return result


def _calculate_sensitivity(
    objective: OptimizationObjective,
    parameters: Parameters,
    free_parameter_labels: list[str],
    base_vector: np.ndarray,
    expected_slices: dict[str, slice],
    settings: ClpStandardErrorSettings,
    *,
    evaluate_at: Callable[[Parameters], Mapping[str, OptimizationResult]] | None,
) -> np.ndarray:
    sensitivity = np.zeros((base_vector.size, len(free_parameter_labels)), dtype=float)
    if not free_parameter_labels:
        return sensitivity

    labels, optimization_values, lower_bounds, upper_bounds = (
        parameters.get_label_value_and_bounds_arrays(exclude_non_vary=True)
    )
    for parameter_index, label in enumerate(free_parameter_labels):
        try:
            value_index = labels.index(label)
        except ValueError as error:
            msg = f"Free parameter '{label}' is not present in the parameter set."
            raise ValueError(msg) from error
        base_value = float(optimization_values[value_index])
        step = _finite_difference_step(
            base_value,
            float(lower_bounds[value_index]),
            float(upper_bounds[value_index]),
            settings.relative_step,
        )
        if step == 0:
            continue
        candidate = parameters.copy()
        candidate_values = optimization_values.copy()
        candidate_values[value_index] += step
        candidate.set_from_label_and_value_arrays(labels, candidate_values)

        if evaluate_at is None:
            perturbed_results = _evaluate_with_restoration(objective, parameters, candidate)
        else:
            perturbed_results = evaluate_at(candidate)
        perturbed_clps = _get_clps(perturbed_results)
        perturbed_vector, _ = _flatten_clps(perturbed_clps, expected_slices=expected_slices)
        sensitivity[:, parameter_index] = (perturbed_vector - base_vector) / step
    return sensitivity


def _finite_difference_step(  # noqa: PLR0911
    value: float,
    lower_bound: float,
    upper_bound: float,
    relative_step: float,
) -> float:
    """Select a transformed-coordinate step that stays within finite bounds."""
    nominal = relative_step * max(1.0, abs(value))
    upper_distance = upper_bound - value
    lower_distance = value - lower_bound
    if np.isfinite(upper_distance) and upper_distance < nominal:
        if not np.isfinite(lower_distance):
            return -nominal
        if lower_distance > 0:
            return -min(nominal, lower_distance / 2)
        return min(nominal, upper_distance / 2) if upper_distance > 0 else 0.0
    if np.isfinite(lower_distance) and lower_distance < nominal:
        if not np.isfinite(upper_distance):
            return nominal
        if lower_distance > 0:
            return -min(nominal, lower_distance / 2)
        return min(nominal, upper_distance / 2) if upper_distance > 0 else 0.0
    return nominal


def _evaluate_with_restoration(
    objective: OptimizationObjective,
    parameters: Parameters,
    candidate: Parameters,
) -> Mapping[str, OptimizationResult]:
    """Evaluate a perturbed objective while restoring shared parameter objects."""
    original_values = {parameter.label: parameter.value for parameter in parameters.all()}
    try:
        for parameter in candidate.all():
            parameters.get(parameter.label).value = parameter.value
        parameters.update_parameter_expression()
        return objective.get_result().optimization_results
    finally:
        for label, value in original_values.items():
            parameters.get(label).value = value
        parameters.update_parameter_expression()


def _calculate_linear_variance(
    objective: OptimizationObjective,
    clps: Mapping[str, xr.DataArray],
) -> dict[str, np.ndarray]:
    data = objective.data
    if isinstance(data, OptimizationData):
        if data.is_global:
            return _calculate_global_linear_variance(objective, clps)
        return _calculate_unlinked_linear_variance(objective, clps)
    if isinstance(data, LinkedOptimizationData):
        return _calculate_linked_linear_variance(objective, clps)
    return {label: np.zeros(clp.shape, dtype=float) for label, clp in clps.items()}


def _calculate_global_linear_variance(
    objective: OptimizationObjective,
    clps: Mapping[str, xr.DataArray],
) -> dict[str, np.ndarray]:
    data = objective.data
    assert isinstance(data, OptimizationData)
    _, _, full_matrix = OptimizationMatrix.from_global_data(data)
    assert data.flat_data is not None
    estimation = OptimizationEstimation.calculate(
        full_matrix.array, data.flat_data, "variable_projection"
    )
    sigma_squared = _residual_variance(estimation.residual, full_matrix.array.shape[1])
    covariance = sigma_squared * _safe_pinv_xtx(full_matrix.array)
    variances = np.diag(covariance)
    full_labels = full_matrix.clp_axis
    model_matrix = OptimizationMatrix.from_data(data)
    global_matrix = OptimizationMatrix.from_data(data, global_matrix=True)
    result: dict[str, np.ndarray] = {}
    for label, clp in clps.items():
        if is_data_model_single_amplitude(data.model):
            values = np.zeros(clp.shape, dtype=float)
            for global_index, global_label in enumerate(global_matrix.clp_axis):
                for model_index, model_label in enumerate(model_matrix.clp_axis):
                    full_label = f"{global_label}@{model_label}"
                    if full_label in full_labels:
                        values[global_index, model_index] = variances[
                            full_labels.index(full_label)
                        ]
        else:
            values = variances.reshape((len(global_matrix.clp_axis), len(model_matrix.clp_axis)))
        result[label] = _align_variance_to_clp(values, clp)
    return result


def _calculate_unlinked_linear_variance(
    objective: OptimizationObjective,
    clps: Mapping[str, xr.DataArray],
) -> dict[str, np.ndarray]:
    data = objective.data
    assert isinstance(data, OptimizationData)
    matrices = objective.calculate_matrices()
    reduced_matrices = objective.calculate_reduced_matrices(matrices)
    label = next(iter(clps))
    clp = clps[label]
    values = np.zeros(clp.shape, dtype=float)
    for index, (matrix, reduced_matrix, data_slice) in enumerate(
        zip(matrices, reduced_matrices, data.data_slices, strict=True)
    ):
        estimation = OptimizationEstimation.calculate(
            reduced_matrix.array, data_slice, "variable_projection"
        )
        sigma_squared = _residual_variance(estimation.residual, reduced_matrix.array.shape[1])
        reduced_covariance = sigma_squared * _safe_pinv_xtx(reduced_matrix.array)
        expansion = _expansion_matrix(
            matrix.clp_axis,
            reduced_matrix.clp_axis,
            float(data.global_axis[index]),
            objective.model.clp_relations,
        )
        full_variance = np.diag(expansion @ reduced_covariance @ expansion.T)
        values[index, :] = _map_variance(matrix.clp_axis, full_variance, clp, index)
    return {label: values}


def _calculate_linked_linear_variance(
    objective: OptimizationObjective,
    clps: Mapping[str, xr.DataArray],
) -> dict[str, np.ndarray]:
    data = objective.data
    assert isinstance(data, LinkedOptimizationData)
    matrices = objective.calculate_matrices()
    reduced_matrices = objective.calculate_reduced_matrices(matrices)
    result = {label: np.zeros(clp.shape, dtype=float) for label, clp in clps.items()}
    for aligned_index, (matrix, reduced_matrix, data_slice) in enumerate(
        zip(matrices, reduced_matrices, data.data_slices, strict=True)
    ):
        estimation = OptimizationEstimation.calculate(
            reduced_matrix.array, data_slice, "variable_projection"
        )
        sigma_squared = _residual_variance(estimation.residual, reduced_matrix.array.shape[1])
        reduced_covariance = sigma_squared * _safe_pinv_xtx(reduced_matrix.array)
        expansion = _expansion_matrix(
            matrix.clp_axis,
            reduced_matrix.clp_axis,
            float(data.global_axis[aligned_index]),
            objective.model.clp_relations,
        )
        full_variance = np.diag(expansion @ reduced_covariance @ expansion.T)
        group_label = data.group_labels[aligned_index]
        group = data.group_definitions[group_label]
        positions = data.data_indices[aligned_index]
        for dataset_position, dataset_label in enumerate(group):
            if dataset_label not in result:
                continue
            dataset_index = int(positions[dataset_position])
            result[dataset_label][dataset_index, :] = _map_variance(
                matrix.clp_axis,
                full_variance,
                clps[dataset_label],
                dataset_index,
            )
    return result


def _map_variance(
    full_labels: list[str],
    full_variance: np.ndarray,
    clp: xr.DataArray,
    index: int,
) -> np.ndarray:
    del index  # The index is part of the call contract for future index-dependent coordinates.
    amplitude_labels = clp.coords.get("amplitude_label")
    if amplitude_labels is None:
        return full_variance
    return np.asarray(
        [
            full_variance[full_labels.index(str(label))] if str(label) in full_labels else 0.0
            for label in amplitude_labels.data
        ],
        dtype=float,
    )


def _align_variance_to_clp(values: np.ndarray, clp: xr.DataArray) -> np.ndarray:
    if values.shape == clp.shape:
        return values
    return np.asarray(values).reshape(clp.shape)


def _expansion_matrix(
    full_labels: list[str],
    reduced_labels: list[str],
    global_axis_value: float,
    relations: list,
) -> np.ndarray:
    expansion = np.zeros((len(full_labels), len(reduced_labels)), dtype=float)
    for reduced_index in range(len(reduced_labels)):
        basis = np.zeros(len(reduced_labels), dtype=float)
        basis[reduced_index] = 1.0
        expansion[:, reduced_index] = (
            OptimizationEstimation(basis, np.empty(0))
            .resolve_clp(full_labels, reduced_labels, global_axis_value, relations)
            .clp
        )
    return expansion


def _residual_variance(residual: ArrayLike, number_of_clps: int) -> float:
    residual_array = np.asarray(residual, dtype=float)
    degrees_of_freedom = max(1, residual_array.size - number_of_clps)
    return float(np.dot(residual_array, residual_array) / degrees_of_freedom)


def _safe_pinv_xtx(matrix: ArrayLike) -> np.ndarray:
    matrix_array = np.asarray(matrix, dtype=float)
    return np.linalg.pinv(matrix_array.T @ matrix_array)


def _flatten_clps(
    clps: Mapping[str, xr.DataArray],
    *,
    expected_slices: dict[str, slice] | None = None,
) -> tuple[np.ndarray, dict[str, slice]]:
    parts: list[np.ndarray] = []
    slices: dict[str, slice] = {}
    cursor = 0
    for label, clp in clps.items():
        flattened = np.asarray(clp.data, dtype=float).reshape(-1)
        if expected_slices is not None:
            if label not in expected_slices:
                msg = f"Dataset '{label}' disappeared during CLP evaluation."
                raise ValueError(msg)
            dataset_slice = expected_slices[label]
            if dataset_slice.stop - dataset_slice.start != flattened.size:
                msg = (
                    f"CLP shape changed during finite-difference evaluation for dataset '{label}'."
                )
                raise ValueError(msg)
            slices[label] = dataset_slice
        else:
            slices[label] = slice(cursor, cursor + flattened.size)
        parts.append(flattened)
        cursor += flattened.size
    return np.concatenate(parts) if parts else np.empty(0), slices
