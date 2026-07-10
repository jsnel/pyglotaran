"""Module containing the optimization group class."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from glotaran.io.prepare_dataset import add_svd_to_dataset
from glotaran.model import DatasetGroup
from glotaran.model.dataset_model import finalize_dataset_model
from glotaran.model.dataset_model import has_dataset_model_global_model
from glotaran.optimization.data_provider import DataProvider
from glotaran.optimization.data_provider import DataProviderLinked
from glotaran.optimization.estimation_provider import EstimationProvider
from glotaran.optimization.estimation_provider import EstimationProviderLinked
from glotaran.optimization.estimation_provider import EstimationProviderUnlinked
from glotaran.optimization.matrix_provider import MatrixProvider
from glotaran.optimization.matrix_provider import MatrixProviderLinked
from glotaran.optimization.matrix_provider import MatrixProviderUnlinked
from glotaran.parameter import Parameters
from glotaran.project import Scheme

if TYPE_CHECKING:
    from glotaran.typing.types import ArrayLike


class OptimizationGroup:
    """A class to optimize a dataset group."""

    def __init__(
        self,
        scheme: Scheme,
        dataset_group: DatasetGroup,
    ):
        """Initialize an optimization group for a dataset group.

        Parameters
        ----------
        scheme : Scheme
            The optimization scheme.
        dataset_group : DatasetGroup
            The dataset group.
        """
        self._dataset_group = dataset_group
        self._dataset_group.set_parameters(scheme.parameters)
        self._data = scheme.data
        self._add_svd = scheme.add_svd
        link_clp = dataset_group.link_clp
        if link_clp is None:
            link_clp = dataset_group.is_linkable(scheme.parameters, scheme.data)

        if link_clp:
            data_provider = DataProviderLinked(scheme, dataset_group)
            matrix_provider = MatrixProviderLinked(dataset_group, data_provider)
            estimation_provider = EstimationProviderLinked(
                dataset_group, data_provider, matrix_provider
            )
        else:
            data_provider = DataProvider(scheme, dataset_group)  # type:ignore[assignment]
            matrix_provider = MatrixProviderUnlinked(  # type:ignore[assignment]
                self._dataset_group, data_provider
            )
            estimation_provider = EstimationProviderUnlinked(  # type:ignore[assignment]
                dataset_group, data_provider, matrix_provider  # type:ignore[arg-type]
            )

        self._data_provider: DataProvider = data_provider
        self._matrix_provider: MatrixProvider = matrix_provider
        self._estimation_provider: EstimationProvider = estimation_provider

        if self._add_svd:
            for dataset in self._data.values():
                self.add_svd_data(
                    "data",
                    dataset,
                    dataset.data.dims[0],
                    dataset.data.dims[1],
                )

    def calculate(self, parameters: Parameters):
        """Calculate the optimization group data.

        Parameters
        ----------
        parameters : Parameters
            The parameters.
        """
        self._dataset_group.set_parameters(parameters)
        self._matrix_provider.calculate()
        self._estimation_provider.estimate()

    def get_additional_penalties(self) -> list[float]:
        """Get additional penalties.

        Returns
        -------
        list[float]
            The additional penalties.
        """
        return self._estimation_provider.get_additional_penalties()

    def get_additional_parameter_penalties(self) -> list[float]:
        """Get additional parameter penalties.

        Returns
        -------
        list[float]
            The additional parameter penalties.
        """
        return self._estimation_provider.get_additional_parameter_penalties()

    def get_additional_penalty_areas(self) -> list[dict]:
        """Get the area breakdown for each equal-area CLP penalty.

        Returns
        -------
        list[dict]
            One dict per penalty with keys: source, source_intervals, source_area,
            target, target_intervals, target_area, parameter, relative, weight, penalty.
        """
        return self._estimation_provider.get_additional_penalty_areas()

    def get_full_penalty(self) -> ArrayLike:
        """Get the full penalty.

        Returns
        -------
        ArrayLike
            The full penalty.
        """
        return self._estimation_provider.get_full_penalty()

    def add_weight_to_result_data(self, dataset_label: str, result_dataset: xr.Dataset):
        """Add weight to result dataset if dataset is weighted.

        Parameters
        ----------
        dataset_label : str
            The label of the data.
        result_dataset : xr.Dataset
            The label of the data.
        """
        weight = self._data_provider.get_weight(dataset_label)
        if weight is None:
            return
        result_dataset["weighted_residual"] = result_dataset["residual"]
        result_dataset["residual"] = result_dataset["residual"] / weight
        if "weight" not in result_dataset:
            if weight.shape != result_dataset.data.shape:
                weight = weight.T
            result_dataset["weight"] = (result_dataset.data.dims, weight)

    @staticmethod
    def _scale_result_matrix(
        dataset_model,
        matrix: xr.DataArray,
        global_dimension: str,
        global_axis: np.ndarray,
    ) -> xr.DataArray:
        """Scale stored result matrices to match the matrix used in the solve.

        The optimization for non-full models applies ``scale`` and ``scale_list``
        to the prepared matrices before estimating CLPs. The result object should
        expose the same effective matrix so downstream decomposition variables such
        as ``species_concentration`` remain consistent with the fitted data.
        """
        if has_dataset_model_global_model(dataset_model):
            return matrix

        scale = float(dataset_model.scale.value) if dataset_model.scale is not None else 1.0
        scale_list = dataset_model.scale_list

        if scale_list is None:
            return matrix if scale == 1.0 else matrix * scale

        scale_values = np.asarray([float(param.value) for param in scale_list], dtype=np.float64)
        if scale_values.size != global_axis.size:
            raise ValueError(
                f"Dataset '{dataset_model.label}': length of 'scale_list' ({scale_values.size}) "
                f"does not match the number of global axis points ({global_axis.size})."
            )

        scale_da = xr.DataArray(
            scale_values * scale,
            coords={global_dimension: global_axis},
            dims=[global_dimension],
        )

        if global_dimension in matrix.dims:
            return matrix * scale_da

        return matrix.expand_dims({global_dimension: global_axis}) * scale_da

    def create_result_data(
        self,
        clp_standard_error: dict[str, xr.DataArray] | None = None,
    ) -> dict[str, xr.Dataset]:
        """Create resulting datasets.

        Parameters
        ----------
        clp_standard_error : dict[str, xr.DataArray] | None
            Optional CLP standard-error values per dataset.

        Returns
        -------
        dict[str, xr.Dataset]
            The datasets with the results.
        """
        result_datasets = {
            label: data.copy()
            for label, data in self._data.items()
            if label in self._dataset_group.dataset_models.keys()
        }

        global_matrices, matrices = self._matrix_provider.get_result()
        clps, residuals = self._estimation_provider.get_result()

        for label, dataset_model in self._dataset_group.dataset_models.items():
            result_dataset = result_datasets[label]

            model_dimension = self._data_provider.get_model_dimension(label)
            result_dataset.attrs["model_dimension"] = model_dimension
            global_dimension = self._data_provider.get_global_dimension(label)
            result_dataset.attrs["global_dimension"] = global_dimension

            result_dataset["residual"] = residuals[label]
            self.add_weight_to_result_data(label, result_dataset)

            result_dataset["matrix"] = self._scale_result_matrix(
                dataset_model,
                matrices[label],
                global_dimension,
                self._data_provider.get_global_axis(label),
            )
            if label in global_matrices:
                result_dataset["global_matrix"] = global_matrices[label]
            result_dataset["clp"] = clps[label]
            if clp_standard_error is not None and label in clp_standard_error:
                result_dataset["clp_standard_error"] = clp_standard_error[label]
                result_dataset["clp_standard_error"].attrs.update(clp_standard_error[label].attrs)

            if self._add_svd:
                self.add_svd_data("residual", result_dataset, model_dimension, global_dimension)
                if "weighted_residual" in result_dataset:
                    self.add_svd_data(
                        "weighted_residual", result_dataset, model_dimension, global_dimension
                    )

            # Calculate RMS
            size = result_dataset.residual.shape[0] * result_dataset.residual.shape[1]
            result_dataset.attrs["root_mean_square_error"] = np.sqrt(
                (result_dataset.residual**2).sum() / size
            ).data

            result_dataset.attrs["weighted_root_mean_square_error"] = (
                np.sqrt((result_dataset.weighted_residual**2).sum() / size).data
                if "weighted_residual" in result_dataset
                else result_dataset.attrs["root_mean_square_error"]
            )

            _scale = dataset_model.scale
            _scale_list = dataset_model.scale_list
            result_dataset.attrs["dataset_scale"] = (
                _scale.value if _scale is not None else 1  # type:ignore[union-attr]
            )
            result_dataset.attrs["dataset_scale_list"] = (
                [p.value for p in _scale_list]  # type:ignore[union-attr]
                if _scale_list is not None
                else []
            )

            # reconstruct fitted data
            result_dataset["fitted_data"] = result_dataset.data - result_dataset.residual

            finalize_dataset_model(dataset_model, result_dataset)

        return result_datasets

    @staticmethod
    def add_svd_data(name: str, dataset: xr.Dataset, lsv_dim: str, rsv_dim: str):
        """Add the SVD of a data matrix to a dataset.

        Parameters
        ----------
        name : str
            Name of the data matrix.
        dataset : xr.Dataset
            Dataset containing the data, which will be updated with the SVD values.
        lsv_dim : str
            The dimension name of the left singular vectors.
        rsv_dim : str
            The dimension name of the right singular vectors.
        """
        add_svd_to_dataset(
            dataset, name=name, lsv_dim=lsv_dim, rsv_dim=rsv_dim, data_array=dataset[name]
        )

    @property
    def number_of_clps(self) -> int:
        """Return number of conditionally linear parameters.

        Returns
        -------
        int
        """
        return self._matrix_provider.number_of_clps
