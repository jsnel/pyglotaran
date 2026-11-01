"""Module containing the estimation provider classes."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from glotaran.builtin.megacomplexes.ctilde import CtildeMegacomplex
from glotaran.builtin.megacomplexes.ctilde.ctilde_megacomplex import is_ctilde_dataset_model
from glotaran.model import DatasetGroup
from glotaran.model import DatasetModel
from glotaran.model import EqualAreaPenalty
from glotaran.model.dataset_model import has_dataset_model_global_model
from glotaran.model.dataset_model import is_dataset_single_amplitude_model
from glotaran.model.item import fill_item
from glotaran.optimization.data_provider import DataProvider
from glotaran.optimization.data_provider import DataProviderLinked
from glotaran.optimization.matrix_provider import MatrixProviderLinked
from glotaran.optimization.matrix_provider import MatrixProviderLinkedFull
from glotaran.optimization.matrix_provider import MatrixProviderUnlinked
from glotaran.optimization.nnls import residual_nnls
from glotaran.optimization.variable_projection import residual_variable_projection

if TYPE_CHECKING:
    from glotaran.typing.types import ArrayLike
SUPPORTED_RESIUDAL_FUNCTIONS = {
    "variable_projection": residual_variable_projection,
    "non_negative_least_squares": residual_nnls,
}


def _calculate_ctilde_residual(
    dataset_model: DatasetModel,
    data_provider: DataProvider,
    matrix_provider: MatrixProviderUnlinked | MatrixProviderLinked,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Calculate direct weighted residuals and identity CLPs for a Ctilde dataset.

    Parameters
    ----------
    dataset_model : DatasetModel
        The direct-residual dataset model.
    data_provider : DataProvider
        The provider containing observations, axes, and dataset weights.
    matrix_provider : MatrixProviderUnlinked | MatrixProviderLinked
        The provider containing current kinetic predictions.

    Returns
    -------
    tuple[np.ndarray, list[np.ndarray]]
        The fixed identity CLPs and weighted residuals by basis coordinate.

    Raises
    ------
    ValueError
        If the basis, time, or scale-list dimensions are incompatible.
    """
    label = dataset_model.label
    matrix = np.asarray(matrix_provider.get_matrix_container(label).matrix).copy()
    global_axis = data_provider.get_global_axis(label)
    if dataset_model.scale is not None:
        scale = dataset_model.scale
        assert not isinstance(scale, str)
        matrix *= float(scale.value)
    if dataset_model.scale_list is not None:
        if len(dataset_model.scale_list) != len(global_axis):
            raise ValueError(
                f"Dataset '{label}': length of 'scale_list' ({len(dataset_model.scale_list)}) "
                f"does not match the number of basis points ({len(global_axis)})."
            )
        scale_values = []
        for scale in dataset_model.scale_list:
            assert not isinstance(scale, str)
            scale_values.append(float(scale.value))
        matrix *= np.asarray(scale_values)[np.newaxis, :]

    weight = data_provider.get_weight(label)
    if weight is not None:
        matrix *= weight

    data = np.asarray(data_provider.get_data(label))
    if data.shape != matrix.shape:
        raise ValueError(
            f"Ctilde dataset '{label}' observations and kinetic predictions must have "
            "matching (time, basis) dimensions."
        )

    clp_labels = matrix_provider.get_matrix_container(label).clp_labels
    if len(global_axis) != len(clp_labels):
        raise ValueError(
            f"Ctilde dataset '{label}' basis coordinates must match its kinetic matrix."
        )
    residual = data - matrix
    return np.eye(len(clp_labels)), [residual[:, index] for index in range(residual.shape[1])]


def _update_ctilde_observations(
    dataset_model: DatasetModel,
    data_provider: DataProvider,
    matrix_provider: MatrixProviderUnlinked | MatrixProviderLinked,
    estimated_clps: dict[str, xr.DataArray],
) -> None:
    """Generate Cnnls observations after estimating the source dataset CLPs.

    Parameters
    ----------
    dataset_model : DatasetModel
        The auxiliary Ctilde dataset model.
    data_provider : DataProvider
        The source and auxiliary observations.
    matrix_provider : MatrixProviderUnlinked | MatrixProviderLinked
        The current source and auxiliary kinetic matrices.
    estimated_clps : dict[str, xr.DataArray]
        The estimated spectra by source dataset.

    Raises
    ------
    ValueError
        If the estimated spectra do not match the reduced kinetic basis.
    """
    ctilde_megacomplex = next(
        megacomplex
        for megacomplex in dataset_model.megacomplex
        if isinstance(megacomplex, CtildeMegacomplex)
    )
    source_label = ctilde_megacomplex.source_dataset
    ctilde_label = dataset_model.label
    source_clps = estimated_clps[source_label]
    basis_labels = matrix_provider.get_matrix_container(ctilde_label).clp_labels
    source_global_dimension = data_provider.get_global_dimension(source_label)
    if "clp_label" not in source_clps.dims or source_global_dimension not in source_clps.dims:
        raise ValueError(f"Ctilde source dataset '{source_label}' must use a spectral CLP model.")
    try:
        etilde = source_clps.sel(clp_label=basis_labels).rename({"clp_label": "basis"})
    except KeyError as exc:
        raise ValueError(
            f"Ctilde basis for dataset '{ctilde_label}' must match source CLP labels "
            f"{basis_labels}."
        ) from exc

    time_axis = data_provider.get_model_axis(ctilde_label)
    basis_axis = data_provider.get_global_axis(ctilde_label)
    ctilde = xr.DataArray(
        matrix_provider.get_matrix_container(ctilde_label).matrix,
        coords={"time": time_axis, "basis": basis_axis},
        dims=("time", "basis"),
    )
    observations = ctilde_megacomplex.calculate_observations(
        data_provider.get_dataset(source_label),
        ctilde,
        etilde,
    )
    data_provider.update_ctilde_data(
        source_label,
        ctilde_label,
        ctilde,
        etilde,
        observations,
    )


class UnsupportedResidualFunctionError(ValueError):
    """Indicates that the residual function is unsupported."""

    def __init__(self, residual_function: str):
        """Initialize an UnsupportedMethodError.

        Parameters
        ----------
        residual_function : str
            The unsupported residual_function.
        """
        super().__init__(
            f"Unknown residual function '{residual_function}', "
            f"supported functions are: {list(SUPPORTED_RESIUDAL_FUNCTIONS.keys())}."
        )


class EstimationProvider:
    """A class to provide estimation for optimization."""

    def __init__(self, dataset_group: DatasetGroup):
        """Initialize an estimation provider for a dataset group.

        Parameters
        ----------
        dataset_group : DatasetGroup
            The dataset group.

        Raises
        ------
        UnsupportedResidualFunctionError
            Raised when residual function of the group dataset group is unsupported.
        """
        self._group = dataset_group
        self._clp_penalty: list[float] = []
        self._clp_penalty_areas: list[dict] = []
        try:
            self._residual_function = SUPPORTED_RESIUDAL_FUNCTIONS[dataset_group.residual_function]
        except KeyError as e:
            raise UnsupportedResidualFunctionError(dataset_group.residual_function) from e

    @property
    def group(self) -> DatasetGroup:
        """Get the dataset group.

        Returns
        -------
        DatasetGroup
            The dataset group.
        """
        return self._group

    def calculate_residual(
        self, matrix: ArrayLike, data: ArrayLike
    ) -> tuple[ArrayLike, ArrayLike]:
        """Calculate the clps and the residual for a matrix and data.

        Parameters
        ----------
        matrix : ArrayLike
            The matrix.
        data : ArrayLike
            The data.

        Returns
        -------
        tuple[ArrayLike, ArrayLike]
            The estimated clp and residual.
        """
        return self._residual_function(matrix, data)

    def retrieve_clps(
        self,
        clp_labels: list[str],
        reduced_clp_labels: list[str],
        reduced_clps: ArrayLike,
        index: int,
    ) -> ArrayLike:
        """Retrieve clp from reduced clp.

        Parameters
        ----------
        clp_labels : list[str]
            The original clp labels.
        reduced_clp_labels : list[str]
            The reduced clp labels.
        reduced_clps : ArrayLike
            The reduced clps.
        index : int
            The index on the global axis.

        Returns
        -------
        ArrayLike
            The retrieved clps.
        """
        model = self.group.model
        parameters = self.group.parameters
        if len(model.clp_relations) == 0 and len(model.clp_constraints) == 0:
            return reduced_clps

        clps = np.zeros(len(clp_labels))

        for i, label in enumerate(reduced_clp_labels):
            idx = clp_labels.index(label)
            clps[idx] = reduced_clps[i]

        for relation in model.clp_relations:
            relation = fill_item(relation, model, parameters)  # type:ignore[arg-type]
            if (
                relation.target in clp_labels
                and relation.applies(index)
                and relation.source in clp_labels
            ):
                source_idx = clp_labels.index(relation.source)
                target_idx = clp_labels.index(relation.target)
                clps[target_idx] = relation.parameter * clps[source_idx]
        return clps

    def get_additional_penalties(self) -> list[float]:
        """Get the additional penalty.

        Returns
        -------
        list[float]
            The additional penalty.
        """
        return self._clp_penalty

    def get_additional_penalty_areas(self) -> list[dict]:
        """Get the area breakdown for each equal-area CLP penalty.

        Returns
        -------
        list[dict]
            One dict per penalty with keys: source, source_intervals, source_area,
            target, target_intervals, target_area, parameter, relative, weight, penalty.
        """
        return self._clp_penalty_areas

    def calculate_clp_penalties(
        self,
        clp_labels: list[list[str]],
        clps: list[np.ndarray],
        global_axis: np.ndarray,
    ) -> tuple[list[float], list[dict]]:
        """Calculate the clp penalty.

        Parameters
        ----------
        clp_labels : list[list[str]]
            The clp labels.
        clps : list[ArrayLike]
            The clps.
        global_axis : ArrayLike
            The global axis.

        Returns
        -------
        tuple[list[float], list[dict]]
            A pair of (penalty_values, area_info).  Each entry in area_info is a dict
            with keys: source, source_intervals, source_area, target, target_intervals,
            target_area, parameter, relative, weight, penalty.
        """
        model = self.group.model
        parameters = self.group.parameters
        penalties = []
        areas = []
        for penalty in model.clp_penalties:
            if not isinstance(penalty, EqualAreaPenalty):
                continue
            penalty = fill_item(penalty, model, parameters)  # type:ignore[arg-type]

            source_area = _get_area(
                penalty.source,
                clp_labels,
                clps,
                penalty.source_intervals,
                global_axis,
            )
            target_area = _get_area(
                penalty.target,
                clp_labels,
                clps,
                penalty.target_intervals,
                global_axis,
            )

            if len(target_area) == 0 and len(source_area) == 0:
                continue
            elif len(target_area) == 0:
                warnings.warn(
                    "Ignoring equal area penalty, target clp " f"{penalty.target} not present."
                )
                continue
            elif len(source_area) == 0:
                warnings.warn(
                    "Ignoring equal area penalty, source clp " f"{penalty.source} not present."
                )
                continue

            source_sum = float(np.sum(source_area))
            target_sum = float(np.sum(target_area))
            parameter_value = float(penalty.parameter)
            target_scaled = parameter_value * target_sum

            if penalty.relative:
                if np.isclose(target_scaled, 0):
                    warnings.warn(
                        "Falling back to absolute equal area penalty in relative mode, "
                        f"parameter * target area is zero for target clp {penalty.target}."
                    )
                    area_penalty = np.abs(source_sum - target_scaled)
                else:
                    area_penalty = source_sum / target_scaled - 1.0
            else:
                area_penalty = np.abs(source_sum - target_scaled)

            penalty_value = float(area_penalty * penalty.weight)

            penalties.append(penalty_value)
            areas.append(
                {
                    "source": penalty.source,
                    "source_intervals": list(penalty.source_intervals),
                    "source_area": source_sum,
                    "target": penalty.target,
                    "target_intervals": list(penalty.target_intervals),
                    "target_area": target_sum,
                    "parameter": parameter_value,
                    "relative": bool(penalty.relative),
                    "weight": float(penalty.weight),
                    "penalty": penalty_value,
                }
            )

        return penalties, areas

    def estimate(self):
        """Calculate the estimation.

        .. # noqa: DAR401
        """
        raise NotImplementedError

    def get_full_penalty(self) -> ArrayLike:
        """Get the full penalty.

        Returns
        -------
        ArrayLike
            The clp penalty.

        .. # noqa: DAR202
        .. # noqa: DAR401
        """
        raise NotImplementedError

    def get_result(
        self,
    ) -> tuple[
        dict[str, xr.DataArray],
        dict[str, xr.DataArray],
    ]:
        """Get the results of the estimation.

        Returns
        -------
        tuple[dict[str, xr.DataArray], dict[str, xr.DataArray]]
            A tuple of the estimated clps and residuals.

        .. # noqa: DAR202
        .. # noqa: DAR401
        """
        raise NotImplementedError


class EstimationProviderUnlinked(EstimationProvider):
    """A class to provide estimation for optimization of an unlinked dataset group."""

    def __init__(
        self,
        dataset_group: DatasetGroup,
        data_provider: DataProvider,
        matrix_provider: MatrixProviderUnlinked,
    ):
        """Initialize an estimation provider for an unlinked dataset group.

        Parameters
        ----------
        dataset_group : DatasetGroup
            The dataset group.
        data_provider : DataProvider
            The data provider.
        matrix_provider : MatrixProviderUnlinked
            The matrix provider.
        """
        super().__init__(dataset_group)
        self._data_provider = data_provider
        self._matrix_provider = matrix_provider
        self._clps: dict[str, list[ArrayLike] | ArrayLike] = {
            label: [] for label in self.group.dataset_models
        }
        self._residuals: dict[str, list[ArrayLike] | ArrayLike] = {
            label: [] for label in self.group.dataset_models
        }
        self._ctilde_clps: dict[str, ArrayLike] = {}
        self._ctilde_residuals: dict[str, list[ArrayLike]] = {}

    def estimate(self):
        """Calculate the estimation."""
        self._clp_penalty.clear()
        self._ctilde_clps.clear()
        self._ctilde_residuals.clear()

        for dataset_model in self.group.dataset_models.values():
            if is_ctilde_dataset_model(dataset_model):
                continue
            if has_dataset_model_global_model(dataset_model):
                self.calculate_full_model_estimation(dataset_model)
            else:
                self.calculate_estimation(dataset_model)

        for dataset_model in self.group.dataset_models.values():
            if not is_ctilde_dataset_model(dataset_model):
                continue
            _update_ctilde_observations(
                dataset_model,
                self._data_provider,
                self._matrix_provider,
                self.get_result()[0],
            )
            self._clps[dataset_model.label], self._residuals[dataset_model.label] = (
                _calculate_ctilde_residual(
                    dataset_model, self._data_provider, self._matrix_provider
                )
            )
            self._ctilde_clps[dataset_model.label] = self._clps[dataset_model.label]
            self._ctilde_residuals[dataset_model.label] = self._residuals[dataset_model.label]

    def get_full_penalty(self) -> ArrayLike:
        """Get the full penalty.

        Returns
        -------
        ArrayLike
            The clp penalty.
        """
        full_penalty = np.concatenate(
            [
                (
                    self._residuals[label]
                    if has_dataset_model_global_model(dataset_model)
                    else np.concatenate(self._residuals[label])
                )
                for label, dataset_model in self.group.dataset_models.items()
            ]
        )
        if len(self._clp_penalty) != 0:
            full_penalty = np.concatenate([full_penalty, self._clp_penalty])
        return full_penalty

    def get_result(
        self,
    ) -> tuple[
        dict[str, list[xr.DataArray]],
        dict[str, list[xr.DataArray]],
    ]:
        """Get the results of the estimation.

        Returns
        -------
        tuple[dict[str, xr.DataArray], dict[str, xr.DataArray]]
            A tuple of the estimated clps and residuals.
        """
        clps, residuals = {}, {}
        for label, dataset_model in self.group.dataset_models.items():
            if is_ctilde_dataset_model(dataset_model) and label not in self._ctilde_clps:
                continue
            model_dimension = self._data_provider.get_model_dimension(label)
            model_axis = self._data_provider.get_model_axis(label)
            global_dimension = self._data_provider.get_global_dimension(label)
            global_axis = self._data_provider.get_global_axis(label)

            if has_dataset_model_global_model(dataset_model):
                residuals[label] = xr.DataArray(
                    np.array(self._residuals[label]).reshape(global_axis.size, model_axis.size).T,
                    coords={global_dimension: global_axis, model_dimension: model_axis},
                    dims=[model_dimension, global_dimension],
                )
                clp_labels = self._matrix_provider.get_matrix_container(label).clp_labels
                global_clp_labels = self._matrix_provider.get_global_matrix_container(
                    label
                ).clp_labels
                clps[label] = xr.DataArray(
                    (
                        np.diag(self._clps[label])
                        if is_dataset_single_amplitude_model(dataset_model)
                        else np.array(self._clps[label]).reshape(
                            (len(global_clp_labels), len(clp_labels))
                        )
                    ),
                    coords={
                        "global_clp_label": global_clp_labels,
                        "clp_label": clp_labels,
                    },
                    dims=["global_clp_label", "clp_label"],
                )

            else:
                residuals[label] = xr.DataArray(
                    np.array(self._residuals[label]).T,
                    coords={global_dimension: global_axis, model_dimension: model_axis},
                    dims=[model_dimension, global_dimension],
                )
                clps[label] = xr.DataArray(
                    self._clps[label],
                    coords=(
                        (global_dimension, global_axis),
                        (
                            "clp_label",
                            self._matrix_provider.get_matrix_container(label).clp_labels,
                        ),
                    ),
                )
        return clps, residuals

    def calculate_full_model_estimation(self, dataset_model: DatasetModel):
        """Calculate the estimation for a dataset with a full model.

        Parameters
        ----------
        dataset_model : DatasetModel
            The dataset model.
        """
        label = dataset_model.label
        full_matrix = self._matrix_provider.get_full_matrix(label)
        data = self._data_provider.get_flattened_data(label)
        self._clps[label], self._residuals[label] = self.calculate_residual(full_matrix, data)

    def calculate_estimation(self, dataset_model: DatasetModel):
        """Calculate the estimation for a dataset.

        Parameters
        ----------
        dataset_model : DatasetModel
            The dataset model.
        """
        label = dataset_model.label
        self._clps[label].clear()
        self._residuals[label].clear()

        global_axis = self._data_provider.get_global_axis(label)
        data = self._data_provider.get_data(label)
        clp_labels = []

        for index, global_index_value in enumerate(global_axis):
            matrix_container = self._matrix_provider.get_prepared_matrix_container(label, index)
            reduced_clps, residual = self.calculate_residual(
                matrix_container.matrix, data[:, index]
            )
            clp_labels.append(self._matrix_provider.get_matrix_container(label).clp_labels)
            clp = self.retrieve_clps(
                clp_labels[index],
                matrix_container.clp_labels,
                reduced_clps,
                global_index_value,
            )

            self._clps[label].append(clp)
            self._residuals[label].append(residual)

        new_penalties, new_areas = self.calculate_clp_penalties(
            clp_labels,
            self._clps[label],
            global_axis,
        )
        self._clp_penalty += new_penalties
        self._clp_penalty_areas += new_areas


class EstimationProviderLinkedFull(EstimationProvider):
    """Estimate one shared set of CLPs for linked full-model datasets."""

    def __init__(
        self,
        dataset_group: DatasetGroup,
        data_provider: DataProvider,
        matrix_provider: MatrixProviderLinkedFull,
    ):
        """Initialize an estimation provider for linked full-model datasets."""
        super().__init__(dataset_group)
        self._data_provider = data_provider
        self._matrix_provider = matrix_provider
        self._clps: dict[str, ArrayLike] = {}
        self._residuals: dict[str, ArrayLike] = {}

    def estimate(self):
        """Estimate linked CLPs and split residuals back by dataset."""
        labels = list(self.group.dataset_models)
        data = [self._data_provider.get_flattened_data(label) for label in labels]
        matrices = [self._matrix_provider.get_aligned_full_matrix(label) for label in labels]
        clps, residuals = self.calculate_residual(np.vstack(matrices), np.concatenate(data))

        offset = 0
        for label, dataset_data in zip(labels, data):
            dataset_clp_indexes = [
                self._matrix_provider.aligned_clp_labels.index(pair)
                for pair in self._matrix_provider.get_reduced_dataset_clp_labels(label)
            ]
            self._clps[label] = self._matrix_provider.expand_dataset_clps(
                label, clps[dataset_clp_indexes]
            )
            self._residuals[label] = residuals[offset : offset + dataset_data.size]
            offset += dataset_data.size

    def get_full_penalty(self) -> ArrayLike:
        """Return the concatenated residuals and any additional penalties.

        Returns
        -------
        ArrayLike
            The residual vector followed by additional penalties.
        """
        full_penalty = np.concatenate(
            [self._residuals[label] for label in self.group.dataset_models]
        )
        if self._clp_penalty:
            full_penalty = np.concatenate([full_penalty, self._clp_penalty])
        return full_penalty

    def get_result(self) -> tuple[dict[str, xr.DataArray], dict[str, xr.DataArray]]:
        """Return per-dataset CLP arrays and residuals.

        Returns
        -------
        tuple[dict[str, xarray.DataArray], dict[str, xarray.DataArray]]
            The CLPs and residuals keyed by dataset label.
        """
        clps: dict[str, xr.DataArray] = {}
        residuals: dict[str, xr.DataArray] = {}
        for label, dataset_model in self.group.dataset_models.items():
            model_clp_labels = self._matrix_provider.get_matrix_container(label).clp_labels
            global_clp_labels = self._matrix_provider.get_global_matrix_container(label).clp_labels
            clp_values = self._clps[label]
            if is_dataset_single_amplitude_model(dataset_model):
                clp_values = np.diag(clp_values)
            else:
                clp_values = clp_values.reshape((len(global_clp_labels), len(model_clp_labels)))

            clps[label] = xr.DataArray(
                clp_values,
                coords={
                    "global_clp_label": global_clp_labels,
                    "clp_label": model_clp_labels,
                },
                dims=["global_clp_label", "clp_label"],
            )

            model_dimension = self._data_provider.get_model_dimension(label)
            model_axis = self._data_provider.get_model_axis(label)
            global_dimension = self._data_provider.get_global_dimension(label)
            global_axis = self._data_provider.get_global_axis(label)
            residuals[label] = xr.DataArray(
                np.asarray(self._residuals[label]).reshape(global_axis.size, model_axis.size).T,
                coords={global_dimension: global_axis, model_dimension: model_axis},
                dims=[model_dimension, global_dimension],
            )
        return clps, residuals


class EstimationProviderLinked(EstimationProvider):
    """A class to provide estimation for optimization of a linked dataset group."""

    def __init__(
        self,
        dataset_group: DatasetGroup,
        data_provider: DataProviderLinked,
        matrix_provider: MatrixProviderLinked,
    ):
        """Initialize an estimation provider for a linked dataset group.

        Parameters
        ----------
        dataset_group : DatasetGroup
            The dataset group.
        data_provider : DataProviderLinked
            The data provider.
        matrix_provider : MatrixProviderLinked
            The matrix provider.
        """
        super().__init__(dataset_group)
        self._data_provider = data_provider
        self._matrix_provider = matrix_provider
        self._clps: list[ArrayLike] = [None] * self._data_provider.aligned_global_axis.size
        self._residuals: list[ArrayLike] = [None] * self._data_provider.aligned_global_axis.size

    def estimate(self):
        """Calculate the estimation."""
        for index, global_index_value in enumerate(self._data_provider.aligned_global_axis):
            matrix_container = self._matrix_provider.get_aligned_matrix_container(index)
            data = self._data_provider.get_aligned_data(index)
            reduced_clps, residual = self.calculate_residual(matrix_container.matrix, data)
            self._clps[index] = self.retrieve_clps(
                self._matrix_provider.aligned_full_clp_labels[index],
                matrix_container.clp_labels,
                reduced_clps,
                global_index_value,
            )
            self._residuals[index] = residual

        self._ctilde_clps: dict[str, np.ndarray] = {}
        self._ctilde_residuals: dict[str, list[np.ndarray]] = {}
        for dataset_model in self.group.dataset_models.values():
            if is_ctilde_dataset_model(dataset_model):
                _update_ctilde_observations(
                    dataset_model,
                    self._data_provider,
                    self._matrix_provider,
                    self.get_result()[0],
                )
                (
                    self._ctilde_clps[dataset_model.label],
                    self._ctilde_residuals[dataset_model.label],
                ) = _calculate_ctilde_residual(
                    dataset_model, self._data_provider, self._matrix_provider
                )

        self._clp_penalty, self._clp_penalty_areas = self.calculate_clp_penalties(
            self._matrix_provider.aligned_full_clp_labels,
            self._clps,
            self._data_provider.aligned_global_axis,
        )

    def get_full_penalty(self) -> ArrayLike:
        """Get the full penalty.

        Returns
        -------
        ArrayLike
            The clp penalty.
        """
        full_penalty_values = [np.concatenate(self._residuals)]
        full_penalty_values.extend(
            np.concatenate(residuals)
            for residuals in getattr(self, "_ctilde_residuals", {}).values()
        )
        full_penalty = np.concatenate((*full_penalty_values, self._clp_penalty))
        return full_penalty

    def get_result(
        self,
    ) -> tuple[
        dict[str, xr.DataArray],
        dict[str, xr.DataArray],
    ]:
        """Get the results of the estimation.

        Returns
        -------
        tuple[dict[str, xr.DataArray], dict[str, xr.DataArray]]
            A tuple of the estimated clps and residuals.
        """
        clps: dict[str, xr.DataArray] = {}
        residuals: dict[str, xr.DataArray] = {}
        for dataset_label in self.group.dataset_models:
            if is_ctilde_dataset_model(self.group.dataset_models[dataset_label]):
                if dataset_label not in self._ctilde_clps:
                    continue
                model_dimension = self._data_provider.get_model_dimension(dataset_label)
                model_axis = self._data_provider.get_model_axis(dataset_label)
                global_dimension = self._data_provider.get_global_dimension(dataset_label)
                global_axis = self._data_provider.get_global_axis(dataset_label)
                clp_labels = self._matrix_provider.get_matrix_container(dataset_label).clp_labels
                clps[dataset_label] = xr.DataArray(
                    self._ctilde_clps[dataset_label],
                    coords={
                        global_dimension: global_axis,
                        "clp_label": clp_labels,
                    },
                    dims=[global_dimension, "clp_label"],
                )
                residuals[dataset_label] = xr.DataArray(
                    np.asarray(self._ctilde_residuals[dataset_label]).T,
                    coords={global_dimension: global_axis, model_dimension: model_axis},
                    dims=[model_dimension, global_dimension],
                )
                continue

            dataset_clps, dataset_residual = [], []
            for index in range(self._data_provider.aligned_global_axis.size):
                group_label = self._data_provider.get_aligned_group_label(index)
                if dataset_label not in group_label:
                    continue

                group_datasets = self._data_provider.group_definitions[group_label]
                dataset_index = group_datasets.index(dataset_label)

                clp_labels = self._matrix_provider.get_matrix_container(dataset_label).clp_labels

                dataset_clps.append(
                    xr.DataArray(
                        [
                            self._clps[index][
                                self._matrix_provider.aligned_full_clp_labels[index].index(label)
                            ]
                            for label in clp_labels
                        ],
                        coords={"clp_label": clp_labels},
                    )
                )

                start = sum(
                    self._data_provider.get_model_axis(label).size
                    for label in group_datasets[:dataset_index]
                )
                end = start + self._data_provider.get_model_axis(dataset_label).size
                dataset_residual.append(self._residuals[index][start:end])

            model_dimension = self._data_provider.get_model_dimension(dataset_label)
            model_axis = self._data_provider.get_model_axis(dataset_label)
            global_dimension = self._data_provider.get_global_dimension(dataset_label)
            global_axis = self._data_provider.get_global_axis(dataset_label)
            clps[dataset_label] = xr.concat(
                dataset_clps,
                dim=global_dimension,
            )
            clps[dataset_label].coords[global_dimension] = global_axis
            residuals[dataset_label] = xr.DataArray(
                np.array(dataset_residual).T,
                coords={global_dimension: global_axis, model_dimension: model_axis},
                dims=[model_dimension, global_dimension],
            )
        return clps, residuals


def _get_area(
    clp_label: str,
    clp_labels: list[str] | list[list[str]],
    clps: list[ArrayLike],
    intervals: list[tuple[float, float]],
    global_axis: ArrayLike,
) -> ArrayLike:
    """Get get slice of a clp on intervals on the global axis.

    Parameters
    ----------
    clp_label : str
        The label of the clp.
    clp_labels: list[str] | list[list[str]]
        The clp labels.
    clps : list[ArrayLike]
        The clps.
    intervals: list[tuple[float, float]]
        The intervals on the global axis.
    global_axis : ArrayLike
        The global axis.

    Returns
    -------
    ArrayLike:
        The concatenated slices.
    """
    area = []

    for interval in intervals:
        if interval[0] > global_axis[-1]:
            continue
        bounded_interval = (
            max(interval[0], np.min(global_axis)),
            min(interval[1], np.max(global_axis)),
        )

        interval_slice = DataProvider.get_axis_slice_from_interval(bounded_interval, global_axis)
        start_idx, end_idx = interval_slice.start, interval_slice.stop

        for i in range(start_idx, end_idx):
            index_clp_labels: list[str] = (
                clp_labels[i]
                if isinstance(clp_labels[0], list)
                else clp_labels  # type:ignore[assignment]
            )
            if clp_label in index_clp_labels:
                area.append(clps[i][index_clp_labels.index(clp_label)])

    return np.asarray(area)  # TODO: normalize for distance on global axis
