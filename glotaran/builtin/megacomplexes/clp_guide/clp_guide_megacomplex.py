from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from glotaran.builtin.megacomplexes.spectral.shape import SpectralShape
from glotaran.model import DatasetModel
from glotaran.model import Megacomplex
from glotaran.model import ModelError
from glotaran.model import ModelItemType
from glotaran.model import item
from glotaran.model import megacomplex

if TYPE_CHECKING:
    from glotaran.typing.types import ArrayLike


@megacomplex(exclusive=True)
class ClpGuideMegacomplex(Megacomplex):
    type: str = "clp-guide"
    target: str

    def calculate_matrix(
        self,
        dataset_model: DatasetModel,
        global_axis: ArrayLike,
        model_axis: ArrayLike,
        **kwargs,
    ):
        clp_label = [self.target]
        matrix = np.ones((1, 1), dtype=np.float64)
        return clp_label, matrix

    def finalize_data(
        self,
        dataset_model: DatasetModel,
        dataset: xr.Dataset,
        is_full_model: bool = False,
        as_global: bool = False,
    ):
        pass


@item
class SpectralModelClpGuideDatasetModel(DatasetModel):
    spectral_axis_inverted: bool = False
    spectral_axis_scale: float = 1


@megacomplex(
    exclusive=True,
    dataset_model_type=SpectralModelClpGuideDatasetModel,
)
class SpectralModelClpGuideMegacomplex(Megacomplex):
    type: str = "spectral-model-clp-guide"
    dimension: str = "time"
    target: str
    shape: dict[str, ModelItemType[SpectralShape]]

    def calculate_matrix(
        self,
        dataset_model: DatasetModel,
        global_axis: ArrayLike,
        model_axis: ArrayLike,
        **kwargs,
    ):
        self._get_target_shape()
        return [self.target], np.ones((model_axis.size, 1), dtype=np.float64)

    def calculate_data(
        self,
        dataset_model: DatasetModel,
        model_axis: ArrayLike,
        spectral_axis: ArrayLike,
    ) -> np.ndarray:
        shape = self._get_target_shape()
        spectral_axis = np.asarray(spectral_axis)
        if dataset_model.spectral_axis_inverted:
            spectral_axis = dataset_model.spectral_axis_scale / spectral_axis
        elif dataset_model.spectral_axis_scale != 1:
            spectral_axis = spectral_axis * dataset_model.spectral_axis_scale
        values = shape.calculate(spectral_axis)
        return np.broadcast_to(values, (model_axis.size, spectral_axis.size)).copy()

    def finalize_data(
        self,
        dataset_model: DatasetModel,
        dataset: xr.Dataset,
        is_full_model: bool = False,
        as_global: bool = False,
    ):
        pass

    def _get_target_shape(self) -> SpectralShape:
        try:
            shape = self.shape[self.target]
        except KeyError as exc:
            raise ModelError(
                f"Spectral model CLP guide target '{self.target}' has no matching shape."
            ) from exc
        if isinstance(shape, str):
            raise ModelError(
                f"Spectral model CLP guide shape for target '{self.target}' was not filled."
            )
        return shape
