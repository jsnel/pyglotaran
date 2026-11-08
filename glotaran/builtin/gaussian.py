"""Gaussian profile calculations shared by activations and spectral elements."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import overload

import numpy as np

if TYPE_CHECKING:
    from glotaran.model.item import ParameterType


def _gaussian_profile(argument: float | np.ndarray) -> float | np.ndarray:
    return np.exp(-np.log(2) * np.square(argument))


@overload
def calculate_skewed_gaussian_component(
    axis: float,
    amplitude: float,
    location: float,
    width: float,
    skewness: float,
) -> float: ...


@overload
def calculate_skewed_gaussian_component(
    axis: np.ndarray,
    amplitude: ParameterType | None,
    location: ParameterType,
    width: ParameterType,
    skewness: ParameterType,
) -> np.ndarray: ...


def calculate_skewed_gaussian_component(
    axis: float | np.ndarray,
    amplitude: ParameterType | None,
    location: ParameterType,
    width: ParameterType,
    skewness: ParameterType,
) -> float | np.ndarray:
    """Evaluate a skewed Gaussian on a scalar coordinate or an array."""
    component: float | np.ndarray
    scalar_coordinate = isinstance(axis, float)
    zero_skew = np.isclose(skewness, 0) if scalar_coordinate else np.allclose(skewness, 0)
    if zero_skew:
        component = _gaussian_profile(2 * (axis - location) / width)  # type:ignore[operator]
    else:
        log_argument = 1 + 2 * skewness * (axis - location) / width  # type:ignore[operator]
        if not scalar_coordinate:
            log_args = np.asarray(log_argument)
            component = np.zeros(log_args.shape)
            valid = log_args > 0
            component[valid] = _gaussian_profile(np.log(log_args[valid]) / skewness)
        elif log_argument > 0:
            component = _gaussian_profile(np.log(log_argument) / skewness)
        else:
            # Scalar dispersion ignores an out-of-domain component before scaling.
            return 0.0
    if amplitude is not None:
        component *= amplitude  # type:ignore[assignment,operator]
    return component
