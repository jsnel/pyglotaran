"""CLP guide whose guide spectrum is generated from a parametrized spectral shape."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar
from typing import Literal

import numpy as np
import xarray as xr
from pydantic import model_validator

from glotaran.builtin.elements.spectral.element import SpectralDataModel
from glotaran.builtin.elements.spectral.shape import SpectralShape  # noqa: TC001
from glotaran.model.element import Element

if TYPE_CHECKING:
    from glotaran.model.data_model import DataModel
    from glotaran.typing.types import ArrayLike
    from glotaran.typing.types import Self


class SpectralModelClpGuideElement(Element):
    """Guide the CLP ``target`` towards a parametrized spectral shape.

    A dataset using this element does not need measured data. The optimization creates it
    on the spectral axis of the measured datasets and recalculates its values from
    ``shapes[target]`` whenever the parameters change. The element contributes a constant
    matrix column for ``target``, so the residual is ``scale * clp(target) - shape``.

    The data model fields ``spectral_axis_inverted`` and ``spectral_axis_scale`` transform
    the spectral axis before the shape is calculated, for example to specify shape
    parameters in wavenumbers for data on a wavelength axis.
    """

    type: Literal["spectral-model-clp-guide"]  # type:ignore[assignment]
    register_as: ClassVar[str] = "spectral-model-clp-guide"
    is_exclusive: ClassVar[bool] = True
    data_model_type: ClassVar[type[DataModel]] = SpectralDataModel  # type:ignore[valid-type]
    dimension: str = "time"
    target: str
    shapes: dict[str, SpectralShape.get_annotated_type()]  # type:ignore[valid-type]

    @model_validator(mode="after")
    def validate_target_shape(self) -> Self:
        """Ensure that ``shapes`` defines the shape of ``target``."""
        if self.target not in self.shapes:
            msg = f"Spectral model CLP guide target '{self.target}' has no matching shape."
            raise ValueError(msg)
        return self

    def calculate_matrix(
        self,
        model: DataModel,
        global_axis: ArrayLike,
        model_axis: ArrayLike,
        **kwargs: Any,  # noqa: ANN401
    ) -> tuple[list[str], ArrayLike]:
        return [self.target], np.ones((model_axis.size, 1), dtype=np.float64)

    def calculate_data(
        self,
        model: SpectralDataModel,
        model_axis: ArrayLike,
        global_axis: ArrayLike,
    ) -> np.ndarray:
        """Calculate the guide data from the current shape parameters.

        Parameters
        ----------
        model : SpectralDataModel
            The data model, providing the spectral axis transformation.
        model_axis : ArrayLike
            The model axis of the guide dataset.
        global_axis : ArrayLike
            The spectral axis of the guide dataset.

        Returns
        -------
        np.ndarray
            The guide data with shape ``(model_axis.size, global_axis.size)``.
        """
        spectral_axis = np.asarray(global_axis)
        if model.spectral_axis_inverted:
            spectral_axis = model.spectral_axis_scale / spectral_axis
        elif model.spectral_axis_scale != 1:
            spectral_axis = spectral_axis * model.spectral_axis_scale
        values = self.shapes[self.target].calculate(spectral_axis)
        return np.broadcast_to(values, (model_axis.size, spectral_axis.size)).copy()

    def create_result(
        self,
        model: DataModel,
        global_dimension: str,
        model_dimension: str,
        amplitudes: xr.Dataset,
        concentrations: xr.Dataset,
    ) -> xr.Dataset:
        return xr.Dataset()
