"""Functions for simulating a dataset using a global optimization model."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from glotaran.model.data_model import get_data_model_dimension
from glotaran.model.data_model import is_data_model_single_amplitude
from glotaran.model.data_model import resolve_data_model
from glotaran.model.errors import GlotaranUserError
from glotaran.optimization.matrix import OptimizationMatrix

if TYPE_CHECKING:
    from glotaran.model.data_model import DataModel
    from glotaran.parameter import Parameters
    from glotaran.project.library import ModelLibrary
    from glotaran.typing.types import ArrayLike


def simulate(  # noqa: C901, PLR0912, PLR0915
    model: DataModel,
    library: ModelLibrary,
    parameters: Parameters,
    coordinates: dict[str, ArrayLike],
    clp: xr.DataArray | None = None,
    *,
    noise: bool = False,
    noise_std_dev: float = 1.0,
    noise_seed: int | None = None,
) -> xr.Dataset:
    """Simulate a dataset using a model.

    Parameters
    ----------
    model : Model
        The model containing the dataset model.
    dataset : str
        Label of the dataset to simulate
    parameters : Parameters
        The parameters for the simulation.
    coordinates : dict[str, ArrayLike]
        A dictionary with the coordinates used for simulation (e.g. time, wavelengths, ...).
    clp : xr.DataArray | None
        A matrix with conditionally linear parameters (e.g. spectra, pixel intensity, ...).
        Will be used instead of the dataset's global megacomplexes if not None.
    noise : bool
        Add noise to the simulation.
    noise_std_dev : float
        The standard deviation for noise simulation.
    noise_seed : int | None
        The seed for the noise simulation.

    Returns
    -------
    xr.Dataset
        The simulated dataset.


    Raises
    ------
    ValueError
        Raised if dataset model has no global megacomplex and no clp are provided.
    """
    model = resolve_data_model(model, library, parameters)
    if is_data_model_single_amplitude(model) and not model.global_elements:
        msg = "A single-amplitude simulation requires global elements to be configured."
        raise GlotaranUserError(msg)
    model_dimension = get_data_model_dimension(model)
    model_axis = coordinates[model_dimension]
    global_dimension = next(dim for dim in coordinates if dim != model_dimension)
    global_axis = coordinates[global_dimension]

    if clp is None:
        if not model.global_elements:
            msg = "Cannot simulate dataset without global megacomplexes if no clp are provided."
            raise GlotaranUserError(msg)
        clp_matrix = OptimizationMatrix.from_data_model(
            model, global_axis, model_axis, None, global_matrix=True
        )
        if is_data_model_single_amplitude(model):
            if clp_matrix.is_index_dependent:
                msg = (
                    "Single-amplitude simulation does not support an index-dependent "
                    "global matrix."
                )
                raise GlotaranUserError(msg)
            clp = None
        else:
            clp = xr.DataArray(
                clp_matrix.array,
                coords=((global_dimension, global_axis), ("clp_label", clp_matrix.clp_axis)),
            )

    matrix = OptimizationMatrix.from_data_model(model, global_axis, model_axis, None)
    result = xr.DataArray(
        np.zeros((model_axis.size, global_axis.size)),
        coords=[
            (model_dimension, model_axis),
            (global_dimension, global_axis),
        ],
    )
    global_matrix = (
        OptimizationMatrix.from_data_model(
            model, global_axis, model_axis, None, global_matrix=True
        )
        if model.global_elements
        else None
    )
    if model.scale_list is not None:
        if len(model.scale_list) != global_axis.size:
            msg = (
                f"Dataset scale_list length ({len(model.scale_list)}) does not match the "
                f"number of global axis points ({global_axis.size})."
            )
            raise ValueError(msg)
        scales = np.asarray(
            [float(getattr(value, "value", value)) for value in model.scale_list],
            dtype=float,
        )
    else:
        scales = np.ones(global_axis.size)
    for i in range(global_axis.size):
        if is_data_model_single_amplitude(model):
            assert global_matrix is not None
            if len(global_matrix.clp_axis) != len(matrix.clp_axis):
                msg = (
                    "A single-amplitude simulation requires the model and global matrices "
                    "to have the same number of components."
                )
                raise GlotaranUserError(msg)
            model_matrix = matrix.at_index(i).array
            if global_matrix.is_index_dependent:
                msg = (
                    "Single-amplitude simulation does not support an index-dependent "
                    "global matrix."
                )
                raise GlotaranUserError(msg)
            pair_matrix = model_matrix * global_matrix.array[i, :][np.newaxis, :]
            amplitudes = (
                np.ones(pair_matrix.shape[1])
                if clp is None
                else clp.isel({global_dimension: i}).to_numpy()
            )
            result[:, i] = pair_matrix @ amplitudes
        else:
            assert clp is not None
            result[:, i] = np.dot(
                matrix.at_index(i).array,
                clp.isel({global_dimension: i}).sel({"clp_label": matrix.clp_axis}).to_numpy(),
            )
        result[:, i] *= scales[i]

    if noise and noise_seed is not None:
        rng = np.random.default_rng(noise_seed)
        original_dims = result.dims
        canonical_result = result.transpose(*sorted(original_dims))
        result = xr.DataArray(
            rng.normal(canonical_result.data, noise_std_dev),
            coords=canonical_result.coords,
        ).transpose(*original_dims)

    return result.to_dataset(name="data")
