import numpy as np
import xarray as xr
from numpy.typing import ArrayLike

from glotaran.model import ExperimentModel
from glotaran.optimization.data import LinkedOptimizationData
from glotaran.optimization.data import OptimizationData
from glotaran.optimization.estimation import OptimizationEstimation
from glotaran.optimization.matrix import OptimizationMatrix
from glotaran.optimization.penalty import calculate_clp_penalties


class OptimizationObjective:
    def __init__(self, model: ExperimentModel):
        self._data = (
            LinkedOptimizationData.from_experiment_model(model)
            if len(model.datasets) > 1
            else OptimizationData(next(iter(model.datasets.values())))
        )
        self._model = model

    def calculate_matrices(self) -> list[OptimizationMatrix]:
        if isinstance(self._data, OptimizationData):
            matrix = OptimizationMatrix.from_data(self._data)
            return [matrix.at_index(i) for i in range(self._data.global_axis.size)]
        return OptimizationMatrix.from_linked_data(self._data)

    def calculate(self) -> ArrayLike:
        matrices = self.calculate_matrices()
        reduced_matrices = [
            matrices[i].reduce(index, self._model.clp_constraints, self._model.clp_relations)
            for i, index in enumerate(self._data.global_axis)
        ]
        estimations = [
            OptimizationEstimation.calculate(matrix.array, data, self._model.residual_function)
            for matrix, data in zip(reduced_matrices, self._data.data_slices)
        ]

        penalties = [e.residual for e in estimations]
        if len(self._model.clp_penalties) > 0:
            estimations = [
                e.resolve_clp(m.clp_labels, r.clp_labels, i, self._model.clp_relations)
                for e, m, r, i in zip(
                    estimations, matrices, reduced_matrices, self._data.global_axis
                )
            ]
            penalties += calculate_clp_penalties(
                matrices, estimations, self._data.global_axis, self._model.clp_penalties
            )
        return np.concatenate(penalties)

    def get_result_dataset(self, data: OptimizationData) -> xr.Dataset:
        dataset = data.model.data.copy()
        dataset.attrs["model_dimension"] = data.model_dimension
        dataset.attrs["global_dimension"] = data.global_dimension

        matrix = OptimizationMatrix.from_data(data)
        matrix_coords = (
            (data.model_dimension, data.model_axis),
            ("clp_label", matrix.clp_labels),
        )
        if matrix.is_index_dependent:
            matrix_coords = ((data.global_dimension, data.global_axis),) + matrix_coords
        dataset["matrix"] = xr.DataArray(matrix.array, coords=matrix_coords)

        matrices = [matrix.at_index(i) for i in range(data.global_axis.size)]
        reduced_matrices = [
            matrices[i].reduce(index, self._model.clp_constraints, self._model.clp_relations)
            for i, index in enumerate(data.global_axis)
        ]
        estimations = [
            OptimizationEstimation.calculate(
                reduced_mat.array, data_slice, self._model.residual_function
            ).resolve_clp(
                mat.clp_labels,
                reduced_mat.clp_labels,
                index,
                self._model.clp_relations,
            )
            for mat, reduced_mat, data_slice, index in zip(
                matrices, reduced_matrices, data.data_slices, data.global_axis
            )
        ]
        dataset["clp"] = xr.DataArray(
            [e.clp for e in estimations],
            coords=((data.global_dimension, data.global_axis), ("clp_label", matrix.clp_labels)),
        )
        dataset["residual"] = xr.DataArray(
            [e.residual for e in estimations],
            coords=(
                (data.global_dimension, data.global_axis),
                (data.model_dimension, data.model_axis),
            ),
        )
        # Calculate RMS
        size = dataset.residual.shape[0] * dataset.residual.shape[1]
        dataset.attrs["root_mean_square_error"] = np.sqrt(
            (dataset.residual**2).sum() / size
        ).data
        return dataset

    def get_result(self) -> dict[str, xr.Dataset]:
        return {
            label: self.get_result_dataset(OptimizationData(data))
            for label, data in self._model.datasets.items()
        }
