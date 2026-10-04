"""Fits of simulated data for the tests of recording, recompute and export."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

from glotaran.project.scheme import Scheme
from glotaran.testing.simulated_data.sequential_spectral_decay import DATASET
from glotaran.testing.simulated_data.sequential_spectral_decay import SCHEME_DICT
from glotaran.testing.simulated_data.shared_decay import PARAMETERS

if TYPE_CHECKING:
    import xarray as xr

    from glotaran.parameter import Parameters
    from glotaran.project.result import Result


def fit(
    parameters: Parameters = PARAMETERS,
    dataset: xr.Dataset = DATASET,
    scheme_dict: dict[str, Any] = SCHEME_DICT,
    **kwargs: Any,
) -> Result:
    """Fit the sequential decay, by default with one function evaluation and quietly."""
    return Scheme.from_dict(scheme_dict).optimize(
        parameters,
        {"sequential-decay": dataset},
        **{"verbose": False, "maximum_number_function_evaluations": 1} | kwargs,
    )
