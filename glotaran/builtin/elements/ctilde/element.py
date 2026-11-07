"""Regularized concentration observations for a source kinetic dataset."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import ClassVar
from typing import Literal
from typing import cast

import numpy as np
import xarray as xr
from pydantic import Field
from scipy.optimize import nnls

from glotaran.model.element import Element
from glotaran.model.errors import GlotaranModelError

if TYPE_CHECKING:
    from glotaran.model.data_model import DataModel
    from glotaran.typing.types import ArrayLike


class CtildeElement(Element):
    """A generated direct-residual concentration dataset for ``source_dataset``.

    Optional supplied auxiliary data provide the time and basis coordinates and
    weights. Its data values are replaced with generated observations at each
    evaluation; the supplied basis can reorder the reduced source CLP labels.
    """

    type: Literal["ctilde"]  # type:ignore[assignment]
    register_as: ClassVar[str] = "ctilde"
    is_exclusive: ClassVar[bool] = True
    dimension: str = "time"
    source_dataset: str
    regularization_weight: float = Field(default=1.0, ge=0, allow_inf_nan=False)

    def calculate_observations(
        self, source_dataset: xr.Dataset, ctilde: xr.DataArray, etilde: xr.DataArray
    ) -> xr.Dataset:
        """Calculate regularized NNLS concentrations and their spectral residual."""
        data = source_dataset["data"].transpose("spectral", "time")
        try:
            ctilde = ctilde.sel(basis=etilde["basis"], time=data["time"]).transpose(
                "basis", "time"
            )
            etilde = etilde.sel(spectral=data["spectral"]).transpose("spectral", "basis")
        except (KeyError, ValueError) as exc:
            msg = "Ctilde source axes and basis labels must match."
            raise GlotaranModelError(msg) from exc
        values = data.to_numpy()
        spectra = etilde.to_numpy()
        kinetics = ctilde.to_numpy()
        if spectra.shape[0] != values.shape[0] or kinetics.shape != (
            spectra.shape[1],
            values.shape[1],
        ):
            msg = "Ctilde source data, kinetic matrix and spectra have incompatible shapes."
            raise GlotaranModelError(msg)
        if not all(np.isfinite(value).all() for value in (values, spectra, kinetics)):
            msg = "Ctilde source data, kinetic matrix and spectra must be finite."
            raise GlotaranModelError(msg)
        factor = np.sqrt(self.regularization_weight)
        matrix = np.vstack((spectra, factor * np.eye(spectra.shape[1])))
        augmented = np.vstack((values, factor * kinetics))
        concentrations = np.empty((spectra.shape[1], values.shape[1]), dtype=np.float64)
        for index in range(values.shape[1]):
            concentrations[:, index], _ = nnls(matrix, augmented[:, index])
        return xr.Dataset(
            {
                "data": (("time", "basis"), concentrations.T),
                "refresid": (("time", "spectral"), (values - spectra @ concentrations).T),
            },
            coords={"time": data["time"], "basis": etilde["basis"], "spectral": data["spectral"]},
        )

    def calculate_matrix(
        self, model: DataModel, global_axis: ArrayLike, model_axis: ArrayLike, **kwargs: object
    ) -> tuple[list[str], np.ndarray]:
        source_matrix = kwargs.get("source_matrix")
        if source_matrix is None:
            msg = "Ctilde requires its source kinetic matrix."
            raise GlotaranModelError(msg)
        labels, values = cast("tuple[list[str], np.ndarray]", source_matrix)
        basis = [str(value) for value in global_axis]
        if (
            len(set(labels)) != len(labels)
            or len(set(basis)) != len(basis)
            or set(basis) != set(labels)
        ):
            msg = "Ctilde basis must match unique source kinetic labels."
            raise GlotaranModelError(msg)
        values = np.asarray(values)
        if values.shape != (len(model_axis), len(labels)):
            msg = "Ctilde source matrix must have dimensions (time, basis)."
            raise GlotaranModelError(msg)
        return basis, np.take(values, [labels.index(label) for label in basis], axis=1)

    def create_result(
        self,
        model: DataModel,
        global_dimension: str,
        model_dimension: str,
        amplitudes: xr.DataArray,
        concentrations: xr.DataArray,
    ) -> xr.Dataset:
        return xr.Dataset()
