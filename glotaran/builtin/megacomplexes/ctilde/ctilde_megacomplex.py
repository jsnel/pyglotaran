from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr
from scipy.optimize import nnls

from glotaran.model import DatasetModel
from glotaran.model import Megacomplex
from glotaran.model import ModelError
from glotaran.model import megacomplex

if TYPE_CHECKING:
    from glotaran.typing.types import ArrayLike


@megacomplex(exclusive=True)
class CtildeMegacomplex(Megacomplex):
    """Use kinetic concentration predictions as a direct time-by-basis model."""

    dimension: str = "time"
    type: str = "ctilde"
    source_dataset: str
    regularization_weight: float = 1.0

    def calculate_observations(
        self,
        source_dataset: xr.Dataset,
        ctilde: xr.DataArray,
        etilde: xr.DataArray,
    ) -> xr.Dataset:
        """Estimate regularized non-negative concentrations from a source dataset.

        Parameters
        ----------
        source_dataset : xr.Dataset
            The source dataset containing the measured data.
        ctilde : xr.DataArray
            The kinetic concentration matrix with dimensions ``(time, basis)``.
        etilde : xr.DataArray
            The estimated spectra with dimensions ``(spectral, basis)``.

        Returns
        -------
        xr.Dataset
            The regularized non-negative concentrations and their spectral residual.
        """
        if not np.isfinite(self.regularization_weight) or self.regularization_weight < 0:
            raise ModelError("Ctilde regularization_weight must be finite and nonnegative.")

        data = source_dataset["data"].transpose("spectral", "time").to_numpy()
        etilde_values = etilde.transpose("spectral", "basis").to_numpy()
        ctilde_data = ctilde.sel(
            basis=etilde["basis"],
            time=source_dataset["data"]["time"],
        )
        ctilde_values = ctilde_data.transpose("basis", "time").to_numpy()

        if data.shape[0] != etilde_values.shape[0]:
            raise ModelError("Ctilde source data and Etilde must share the spectral axis.")
        if ctilde_values.shape != (etilde_values.shape[1], data.shape[1]):
            raise ModelError(
                "Ctilde source concentrations must have dimensions (basis, time) "
                "matching Etilde and data."
            )
        if not all(
            np.all(np.isfinite(values))
            for values in (data, etilde_values, ctilde_values)
        ):
            raise ModelError("Ctilde source data, kinetic matrix, and spectra must be finite.")

        basis_size = etilde_values.shape[1]
        sqrt_weight = np.sqrt(self.regularization_weight)
        augmented_matrix = np.vstack((etilde_values, sqrt_weight * np.eye(basis_size)))
        augmented_data = np.vstack((data, sqrt_weight * ctilde_values))
        concentrations = np.empty((basis_size, data.shape[1]), dtype=np.float64)

        for time_index in range(data.shape[1]):
            concentrations[:, time_index], _ = nnls(
                augmented_matrix, augmented_data[:, time_index]
            )

        refresid = data - etilde_values @ concentrations
        return xr.Dataset(
            {
                "data": (
                    ("time", "basis"),
                    concentrations.T,
                ),
                "refresid": (
                    ("time", "spectral"),
                    refresid.T,
                ),
            },
            coords={
                "time": source_dataset.coords["time"].values,
                "basis": etilde["basis"].values,
                "spectral": source_dataset.coords["spectral"].values,
            },
        )

    def calculate_matrix(
        self,
        dataset_model: DatasetModel,
        global_axis: ArrayLike,
        model_axis: ArrayLike,
        *,
        source_matrix: tuple[list[str], ArrayLike] | None = None,
        **kwargs,
    ) -> tuple[list[str], ArrayLike]:
        """Return the current kinetic concentrations aligned to the basis coordinate."""
        del kwargs
        if source_matrix is None:
            raise ModelError(
                f"Ctilde megacomplex '{self.label}' requires its source kinetic matrix."
            )

        source_labels, source_values = source_matrix
        basis_labels = [str(label) for label in global_axis]
        if len(set(source_labels)) != len(source_labels):
            raise ModelError("Ctilde source kinetic matrix contains duplicate basis labels.")
        if set(basis_labels) != set(source_labels):
            raise ModelError(
                f"Ctilde basis for dataset '{dataset_model.label}' must match the reduced "
                f"source labels {source_labels}; got {basis_labels}."
            )

        matrix = np.asarray(source_values)
        if matrix.ndim != 2 or matrix.shape != (len(model_axis), len(source_labels)):
            raise ModelError(
                "Ctilde source kinetic matrix must have dimensions (time, basis) matching "
                "the auxiliary dataset."
            )

        column_indexes = [source_labels.index(label) for label in basis_labels]
        return basis_labels, matrix[:, column_indexes]

    def finalize_data(
        self,
        dataset_model: DatasetModel,
        dataset: xr.Dataset,
        is_full_model: bool = False,
        as_global: bool = False,
    ):
        """Leave direct-residual datasets unchanged during result finalization."""
        del dataset_model, dataset, is_full_model, as_global

    def add_result_data(
        self,
        dataset: xr.Dataset,
        source_dataset: xr.Dataset,
        *,
        add_svd: bool,
    ) -> None:
        """Add source spectra and spectral residual diagnostics to a Ctilde result."""
        if "Etilde" not in source_dataset:
            raise ModelError(
                f"Ctilde source dataset '{self.source_dataset}' is missing estimated spectra "
                "'Etilde'."
            )
        if "refresid" not in dataset:
            raise ModelError(
                f"Ctilde dataset '{dataset.attrs.get('source_dataset', self.source_dataset)}' "
                "is missing the residual generated from Cnnls observations."
            )

        source_data = source_dataset["data"].transpose("time", "spectral")
        etilde = source_dataset["Etilde"].transpose("spectral", "basis")
        cnnls = dataset["data"].transpose("time", "basis")
        refresid = dataset["refresid"].transpose("time", "spectral")
        try:
            source_data, etilde, cnnls, refresid = xr.align(
                source_data, etilde, cnnls, refresid, join="exact", copy=False
            )
        except ValueError as exc:
            raise ModelError(
                f"Ctilde source dataset '{self.source_dataset}' axes must match its Cnnls residual."
            ) from exc

        expected_refresid = source_data - xr.dot(cnnls, etilde, dim="basis")
        if not np.allclose(refresid.to_numpy(), expected_refresid.to_numpy()):
            raise ModelError("Ctilde refresid must be computed from the Cnnls concentrations.")

        dataset["Etilde"] = etilde.copy()
        dataset["refresid"] = refresid

        if not add_svd:
            return

        left, singular_values, right = np.linalg.svd(
            dataset["refresid"].to_numpy(), full_matrices=False
        )
        dataset["refresid_left_singular_vectors"] = (
            ("time", "refresid_left_singular_value_index"),
            left,
        )
        dataset["refresid_singular_values"] = (
            "refresid_singular_value_index",
            singular_values,
        )
        dataset["refresid_right_singular_vectors"] = (
            ("spectral", "refresid_right_singular_value_index"),
            right.T,
        )


def is_ctilde_dataset_model(dataset_model: DatasetModel) -> bool:
    """Check whether a dataset model uses a direct Ctilde residual."""
    return any(
        isinstance(megacomplex, CtildeMegacomplex) for megacomplex in dataset_model.megacomplex
    )
