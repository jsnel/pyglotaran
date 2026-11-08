"""Functions for simulating a dataset using a global optimization model."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from glotaran.model.data_model import get_data_model_dimension
from glotaran.model.data_model import is_data_model_single_amplitude
from glotaran.model.data_model import resolve_data_model
from glotaran.model.errors import GlotaranUserError
from glotaran.optimization.data import resolve_scale_list
from glotaran.optimization.matrix import OptimizationMatrix
from glotaran.optimization.matrix import compose_full_matrix

if TYPE_CHECKING:
    from glotaran.model.data_model import DataModel
    from glotaran.parameter import Parameters
    from glotaran.project.library import ModelLibrary
    from glotaran.typing.types import ArrayLike


def _check_paired_global_matrix(matrix: OptimizationMatrix) -> None:
    if matrix.is_index_dependent:
        msg = "Single-amplitude simulation does not support an index-dependent global matrix."
        raise GlotaranUserError(msg)


def simulate(  # noqa: C901, PLR0912
    model: DataModel,
    library: ModelLibrary,
    parameters: Parameters,
    coordinates: dict[str, ArrayLike],
    clp: xr.DataArray | None = None,
    *,
    noise: bool = False,
    noise_std_dev: float = 1.0,
    noise_seed: int | None = 42,
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
        The seed for the noise simulation. Defaults to 42, so the noise is reproducible.
        Pass None to draw fresh noise on every call.

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
    paired = is_data_model_single_amplitude(model)
    if paired and not model.global_elements:
        msg = "A single-amplitude simulation requires global elements to be configured."
        raise GlotaranUserError(msg)
    model_dimension = get_data_model_dimension(model)
    model_axis = coordinates[model_dimension]
    global_dimension = next(dim for dim in coordinates if dim != model_dimension)
    global_axis = coordinates[global_dimension]

    global_matrix: OptimizationMatrix | None = None
    if clp is None:
        if not model.global_elements:
            msg = "Cannot simulate dataset without global megacomplexes if no clp are provided."
            raise GlotaranUserError(msg)
        global_matrix = OptimizationMatrix.from_data_model(
            model, global_axis, model_axis, None, global_matrix=True
        )
        if paired:
            _check_paired_global_matrix(global_matrix)
        else:
            clp = xr.DataArray(
                global_matrix.array,
                coords=((global_dimension, global_axis), ("clp_label", global_matrix.clp_axis)),
            )

    matrix = OptimizationMatrix.from_data_model(model, global_axis, model_axis, None)
    result = xr.DataArray(
        np.zeros((model_axis.size, global_axis.size)),
        coords=[
            (model_dimension, model_axis),
            (global_dimension, global_axis),
        ],
    )
    if paired:
        if global_matrix is None:
            global_matrix = OptimizationMatrix.from_data_model(
                model, global_axis, model_axis, None, global_matrix=True
            )
        global_matrix = OptimizationMatrix.align_paired_global(model, matrix, global_matrix)
    scales = resolve_scale_list(model.scale_list, global_axis.size)
    if scales is None:
        scales = np.ones(global_axis.size)
    if paired and global_axis.size:
        assert global_matrix is not None
        if len(global_matrix.clp_axis) != len(matrix.clp_axis):
            msg = (
                "A single-amplitude simulation requires the model and global matrices "
                "to have the same number of components."
            )
            raise GlotaranUserError(msg)
        if clp is not None:
            _check_paired_global_matrix(global_matrix)
    for i in range(global_axis.size):
        if paired:
            assert global_matrix is not None
            _, pair_matrix = compose_full_matrix(
                matrix, global_matrix, paired=True, global_index=i
            )
            amplitudes = (
                np.ones(pair_matrix.shape[1])
                if clp is None
                else clp.isel({global_dimension: i}).sel({"clp_label": matrix.clp_axis}).to_numpy()
            )
            result[:, i] = pair_matrix @ amplitudes
        else:
            assert clp is not None
            result[:, i] = np.dot(
                matrix.at_index(i).array,
                clp.isel({global_dimension: i}).sel({"clp_label": matrix.clp_axis}).to_numpy(),
            )
        result[:, i] *= scales[i]

    if noise:
        rng = np.random.default_rng(noise_seed)
        original_dims = result.dims
        canonical_result = result.transpose(*sorted(original_dims))
        result = xr.DataArray(
            rng.normal(canonical_result.data, noise_std_dev),
            coords=canonical_result.coords,
        ).transpose(*original_dims)

    return result.to_dataset(name="data")
