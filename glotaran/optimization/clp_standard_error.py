"""Utilities to calculate conditionally linear parameter (CLP) standard errors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import cast
from warnings import warn

import numpy as np
import xarray as xr

from glotaran.optimization.data_provider import DataProviderLinked
from glotaran.optimization.estimation_provider import EstimationProviderLinked
from glotaran.optimization.estimation_provider import EstimationProviderUnlinked
from glotaran.optimization.matrix_provider import MatrixProviderLinked
from glotaran.optimization.matrix_provider import MatrixProviderUnlinked
from glotaran.optimization.variable_projection import residual_variable_projection

if TYPE_CHECKING:
    from glotaran.optimization.optimization_group import OptimizationGroup
    from glotaran.parameter import Parameters


@dataclass
class ClpStandardErrorSettings:
    """Numerical settings for CLP standard-error calculation."""

    relative_step: float = 1e-6


def calculate_clp_standard_error(
    optimization_group: OptimizationGroup,
    parameters: Parameters,
    free_parameter_labels: list[str],
    covariance_matrix,
    settings: ClpStandardErrorSettings,
) -> dict[str, xr.DataArray]:
    """Calculate CLP standard errors for all datasets of one optimization group.

    Parameters
    ----------
    optimization_group : OptimizationGroup
        Optimization group containing providers and current fit state.
    parameters : Parameters
        Parameter set at the converged solution.
    free_parameter_labels : list[str]
        Labels of free nonlinear parameters corresponding to ``covariance_matrix``
        row and column order.
    covariance_matrix : array-like
        Covariance matrix of free nonlinear parameters.
    settings : ClpStandardErrorSettings
        Numerical settings for finite-difference sensitivity evaluation.

    Returns
    -------
    dict[str, xr.DataArray]
        Mapping of dataset label to CLP standard-error array. Returns an empty
        mapping if CLP standard errors cannot be evaluated for the current
        optimization configuration.
    """
    estimation_provider = optimization_group._estimation_provider

    if estimation_provider._residual_function is not residual_variable_projection:
        warn(
            "Skipping CLP standard-error calculation because residual function is not "
            "'variable_projection'."
        )
        return {}

    base_clps, _ = estimation_provider.get_result()
    if not base_clps:
        return {}

    base_vector, slices = _flatten_clps(base_clps)
    if base_vector.size == 0:
        return {}

    propagated_variance = _calculate_propagated_variance(
        optimization_group,
        parameters,
        free_parameter_labels,
        covariance_matrix,
        base_vector,
        slices,
        settings,
    )
    linear_variance = _calculate_linear_variance(optimization_group, base_clps)

    clp_standard_error: dict[str, xr.DataArray] = {}
    for dataset_label, dataset_clp in base_clps.items():
        dataset_slice = slices[dataset_label]
        nonlinear_var = propagated_variance[dataset_slice].reshape(dataset_clp.shape)
        total_variance = np.maximum(linear_variance[dataset_label] + nonlinear_var, 0.0)
        clp_standard_error[dataset_label] = xr.DataArray(
            np.sqrt(total_variance),
            coords=dataset_clp.coords,
            dims=dataset_clp.dims,
            attrs={
                "method": "linear_plus_nonlinear_propagation",
                "linear_covariance": "pseudo_inverse",
                "nonlinear_covariance": "finite_difference",
                "fd_relative_step": settings.relative_step,
            },
        )

    optimization_group.calculate(parameters)
    return clp_standard_error


def _calculate_propagated_variance(
    optimization_group: OptimizationGroup,
    parameters: Parameters,
    free_parameter_labels: list[str],
    covariance_matrix,
    base_vector: np.ndarray,
    slices: dict[str, slice],
    settings: ClpStandardErrorSettings,
) -> np.ndarray:
    n_clp = base_vector.size
    n_parameters = len(free_parameter_labels)
    if n_parameters == 0:
        return np.zeros(n_clp)

    sensitivity = np.zeros((n_clp, n_parameters), dtype=float)

    for parameter_index, parameter_label in enumerate(free_parameter_labels):
        base_value = float(parameters.get(parameter_label).value)
        step = settings.relative_step * max(1.0, abs(base_value))
        if step == 0.0:
            continue

        perturbed = parameters.copy()
        perturbed_parameter = perturbed.get(parameter_label)
        perturbed_parameter.value = base_value + step
        perturbed.update_parameter_expression()

        optimization_group.calculate(perturbed)
        perturbed_clps, _ = optimization_group._estimation_provider.get_result()
        perturbed_vector, _ = _flatten_clps(perturbed_clps, expected_slices=slices)
        sensitivity[:, parameter_index] = (perturbed_vector - base_vector) / step

    covariance_matrix_np = np.asarray(covariance_matrix)
    propagated_covariance = sensitivity @ covariance_matrix_np @ sensitivity.T
    return np.clip(np.diag(propagated_covariance), a_min=0.0, a_max=None)


def _calculate_linear_variance(
    optimization_group: OptimizationGroup,
    clps: dict[str, xr.DataArray],
) -> dict[str, np.ndarray]:
    matrix_provider = optimization_group._matrix_provider

    if isinstance(matrix_provider, MatrixProviderUnlinked):
        return _calculate_unlinked_linear_variance(optimization_group, clps)
    if isinstance(matrix_provider, MatrixProviderLinked):
        return _calculate_linked_linear_variance(optimization_group, clps)

    return {label: np.zeros(clp.shape) for label, clp in clps.items()}


def _calculate_unlinked_linear_variance(
    optimization_group: OptimizationGroup,
    clps: dict[str, xr.DataArray],
) -> dict[str, np.ndarray]:
    data_provider = optimization_group._data_provider
    matrix_provider = cast(MatrixProviderUnlinked, optimization_group._matrix_provider)
    estimation_provider = cast(EstimationProviderUnlinked, optimization_group._estimation_provider)
    linear_variance = {label: np.zeros(clp.shape) for label, clp in clps.items()}

    for dataset_label, clp in clps.items():
        full_clp_labels = matrix_provider.get_matrix_container(dataset_label).clp_labels

        if "global_clp_label" in clp.dims:
            full_matrix = matrix_provider.get_full_matrix(dataset_label)
            residual = np.asarray(estimation_provider._residuals[dataset_label])
            dof = max(1, residual.size - full_matrix.shape[1])
            sigma_squared = float(np.dot(residual, residual) / dof)
            cov = sigma_squared * _safe_pinv_xtx(full_matrix)
            linear_variance[dataset_label][:] = np.diag(cov).reshape(clp.shape)
            continue

        global_axis = data_provider.get_global_axis(dataset_label)
        for global_index, global_axis_value in enumerate(global_axis):
            reduced_container = matrix_provider.get_prepared_matrix_container(
                dataset_label, global_index
            )
            residual = np.asarray(estimation_provider._residuals[dataset_label][global_index])
            dof = max(1, residual.size - len(reduced_container.clp_labels))
            sigma_squared = float(np.dot(residual, residual) / dof)
            reduced_cov = sigma_squared * _safe_pinv_xtx(reduced_container.matrix)
            expansion = _expansion_matrix(
                estimation_provider,
                full_clp_labels,
                reduced_container.clp_labels,
                float(global_axis_value),
            )
            full_cov = expansion @ reduced_cov @ expansion.T
            linear_variance[dataset_label][global_index, :] = np.diag(full_cov)

    return linear_variance


def _calculate_linked_linear_variance(
    optimization_group: OptimizationGroup,
    clps: dict[str, xr.DataArray],
) -> dict[str, np.ndarray]:
    data_provider = cast(DataProviderLinked, optimization_group._data_provider)
    matrix_provider = cast(MatrixProviderLinked, optimization_group._matrix_provider)
    estimation_provider = cast(EstimationProviderLinked, optimization_group._estimation_provider)
    linear_variance = {label: np.zeros(clp.shape) for label, clp in clps.items()}

    for aligned_index, aligned_axis_value in enumerate(data_provider.aligned_global_axis):
        group_label = data_provider.get_aligned_group_label(aligned_index)
        aligned_matrix = matrix_provider.get_aligned_matrix_container(aligned_index)
        full_clp_labels = matrix_provider.aligned_full_clp_labels[aligned_index]

        residual = np.asarray(estimation_provider._residuals[aligned_index])
        dof = max(1, residual.size - len(aligned_matrix.clp_labels))
        sigma_squared = float(np.dot(residual, residual) / dof)
        reduced_cov = sigma_squared * _safe_pinv_xtx(aligned_matrix.matrix)
        expansion = _expansion_matrix(
            estimation_provider,
            full_clp_labels,
            aligned_matrix.clp_labels,
            float(aligned_axis_value),
        )
        full_cov = expansion @ reduced_cov @ expansion.T

        for dataset_label in data_provider.group_definitions[group_label]:
            dataset_clp_labels = matrix_provider.get_matrix_container(dataset_label).clp_labels
            global_axis = data_provider.get_global_axis(dataset_label)
            global_position = int(np.abs(global_axis - aligned_axis_value).argmin())
            for clp_position, clp_label in enumerate(dataset_clp_labels):
                full_clp_position = full_clp_labels.index(clp_label)
                linear_variance[dataset_label][global_position, clp_position] = full_cov[
                    full_clp_position,
                    full_clp_position,
                ]

    return linear_variance


def _safe_pinv_xtx(matrix: np.ndarray) -> np.ndarray:
    matrix_np = np.asarray(matrix)
    return np.linalg.pinv(matrix_np.T @ matrix_np)


def _expansion_matrix(
    estimation_provider,
    full_clp_labels: list[str],
    reduced_clp_labels: list[str],
    global_axis_value: float,
) -> np.ndarray:
    expansion = np.zeros((len(full_clp_labels), len(reduced_clp_labels)), dtype=float)
    for reduced_index in range(len(reduced_clp_labels)):
        basis = np.zeros(len(reduced_clp_labels), dtype=float)
        basis[reduced_index] = 1.0
        expansion[:, reduced_index] = estimation_provider.retrieve_clps(
            full_clp_labels,
            reduced_clp_labels,
            basis,
            global_axis_value,
        )
    return expansion


def _flatten_clps(
    clps: dict[str, xr.DataArray],
    expected_slices: dict[str, slice] | None = None,
) -> tuple[np.ndarray, dict[str, slice]]:
    parts: list[np.ndarray] = []
    slices: dict[str, slice] = {}
    cursor = 0
    for dataset_label, clp in clps.items():
        flattened = np.asarray(clp.values).reshape(-1)
        if expected_slices is not None:
            dataset_slice = expected_slices[dataset_label]
            if dataset_slice.stop - dataset_slice.start != flattened.size:
                raise ValueError(
                    "CLP shape changed during finite-difference evaluation for "
                    f"dataset '{dataset_label}'."
                )
            slices[dataset_label] = dataset_slice
        else:
            slices[dataset_label] = slice(cursor, cursor + flattened.size)
        parts.append(flattened)
        cursor += flattened.size
    return np.concatenate(parts), slices
