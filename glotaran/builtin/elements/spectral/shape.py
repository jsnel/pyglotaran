"""This package contains the spectral shape item."""

from __future__ import annotations

from typing import Literal

import numpy as np

from glotaran.model.item import ParameterType
from glotaran.model.item import TypedItem


def _calculate_skewed_gaussian_component(
    axis: np.ndarray,
    amplitude: ParameterType | None,
    location: ParameterType,
    width: ParameterType,
    skewness: ParameterType,
) -> np.ndarray:
    if np.allclose(skewness, 0):
        component = np.exp(
            -np.log(2) * np.square(2 * (axis - location) / width)  # type:ignore[operator]
        )
    else:
        log_args = np.asarray(
            1 + 2 * skewness * (axis - location) / width  # type:ignore[operator]
        )
        component = np.zeros(log_args.shape)
        valid_arg_mask = log_args > 0
        component[valid_arg_mask] = np.exp(
            -np.log(2) * np.square(np.log(log_args[valid_arg_mask]) / skewness)
        )
    if amplitude is not None:
        component *= amplitude
    return component


class SpectralShape(TypedItem):
    pass


class SpectralShapeGaussian(SpectralShape):
    """A Gaussian spectral shape."""

    type: Literal["gaussian"]  # type:ignore[assignment]
    amplitude: ParameterType | None = None
    location: ParameterType
    width: ParameterType

    def calculate(self, axis: np.ndarray) -> np.ndarray:
        r"""Calculate a normal Gaussian shape for a given ``axis``.

        The following equation is used for the calculation:

        .. math::
            f(x, A, x_0, \Delta) = A \exp \left({-
            \frac{
                \log{\left(2 \right)
                \left(2(x - x_{0})\right)^{2}
            }}{\Delta^{2}}}\right)

        The parameters of the equation represent the following attributes of the shape:

        - :math:`x` :       ``axis``

        - :math:`A` :       ``amplitude``

        - :math:`x_0` :     ``location``

        - :math:`\Delta` :  ``width``

        In this formalism, :math:`\Delta` represents the full width at half maximum (FWHM).
        Compared to the more common definition
        :math:`\exp \left(- (x-\mu )^{2}/(2\sigma^{2})\right)`
        we have :math:`\sigma = \Delta/(2\sqrt{2\ln(2)})=\Delta/2.35482`

        Parameters
        ----------
        axis : np.ndarray
            The axis to calculate the shape for.

        Returns
        -------
        np.ndarray
            An array representing a Gaussian shape.
        """
        shape = np.exp(
            -np.log(2) * np.square(2 * (axis - self.location) / self.width)  # type:ignore[operator]
        )
        if self.amplitude is not None:
            shape *= self.amplitude
        return shape


class SpectralShapeSkewedGaussian(SpectralShapeGaussian):
    """A skewed Gaussian spectral shape."""

    type: Literal["skewed-gaussian"]  # type:ignore[assignment]
    skewness: ParameterType

    def calculate(self, axis: np.ndarray) -> np.ndarray:
        r"""Calculate the skewed Gaussian shape for ``axis``.

        The following equation is used for the calculation:

        .. math::
            f(x, x_0, A, \Delta, b) =
            \left\{
                \begin{array}{ll}
                    0                                               & \mbox{if } \theta \leq 0 \\
                    A \exp \left({- \dfrac{\log{\left(2 \right)}
                    \log{\left(\theta(x, x_0, \Delta, b) \right)}^{2}}{b^{2}}}\right)
                                                                    & \mbox{if } \theta > 0
                \end{array}
            \right.

        With:

        .. math::
            \theta(x, x_0, \Delta, b) = \frac{2 b \left(x - x_{0}\right) + \Delta}{\Delta}

        The parameters of the equation represent the following attributes of the shape:

        - :math:`x` :       ``axis``

        - :math:`A` :       ``amplitude``

        - :math:`x_0` :     ``location``

        - :math:`\Delta` :  ``width``

        - :math:`b` :       ``skewness``

        Where :math:`\Delta` represents the full width at half maximum (FWHM),
        see :func:`calculate_gaussian`.

        Note that in the limit of skewness parameter :math:`b` equal to zero
        :math:`f(x, x_0, A, \Delta, b)` simplifies to a normal gaussian
        (since :math:`\lim_{b \to 0} \frac{\ln(1+bx)}{b}=x`),
        see the definition in :func:`SpectralShapeGaussian.calculate`.

        Parameters
        ----------
        axis : np.ndarray
            The axis to calculate the shape for.


        Returns
        -------
        np.ndarray
            An array representing a skewed Gaussian shape.
        """
        return _calculate_skewed_gaussian_component(
            axis, self.amplitude, self.location, self.width, self.skewness
        )


class SpectralShapeSkewedGaussianSum(SpectralShape):
    """A sum of skewed Gaussian spectral shapes."""

    type: Literal["skewed-gaussian-sum"]  # type:ignore[assignment]
    amplitude: list[ParameterType] | None = None
    location: list[ParameterType]
    width: list[ParameterType]
    skewness: list[ParameterType]

    def calculate(self, axis: np.ndarray) -> np.ndarray:
        """Calculate and sum the skewed Gaussian components for ``axis``.

        Each component is calculated as in :meth:`SpectralShapeSkewedGaussian.calculate`.

        Parameters
        ----------
        axis : np.ndarray
            The axis to calculate the shape for.

        Returns
        -------
        np.ndarray
            The sum of all skewed Gaussian components.

        Raises
        ------
        ValueError
            If the parameter lists have different lengths.
        """
        component_count = len(self.location)
        amplitudes: list[ParameterType | None] = (
            list(self.amplitude) if self.amplitude else [None] * component_count
        )
        if len({component_count, len(self.width), len(self.skewness), len(amplitudes)}) != 1:
            msg = (
                "Skewed Gaussian sum parameters amplitude, location, width, and "
                "skewness must have the same number of components."
            )
            raise ValueError(msg)

        return sum(
            (
                _calculate_skewed_gaussian_component(axis, amplitude, location, width, skewness)
                for amplitude, location, width, skewness in zip(
                    amplitudes, self.location, self.width, self.skewness, strict=True
                )
            ),
            start=np.zeros(axis.shape, dtype=float),
        )


class SpectralShapeOne(SpectralShape):
    """A constant spectral shape with value 1."""

    type: Literal["one"]  # type:ignore[assignment]

    def calculate(self, axis: np.ndarray) -> np.ndarray:
        """calculate calculates the shape.

        Parameters
        ----------
        axis: np.ndarray
            The axis to calculate the shape on.

        Returns
        -------
        shape: numpy.ndarray

        """
        return np.ones(axis.shape[0])


class SpectralShapeZero(SpectralShape):
    """A constant spectral shape with value 0."""

    type: Literal["zero"]  # type:ignore[assignment]

    def calculate(self, axis: np.ndarray) -> np.ndarray:
        """calculate calculates the shape.

        Only works after calling ``fill``.

        Parameters
        ----------
        axis: np.ndarray
            The axis to calculate the shape on.

        Returns
        -------
        shape: numpy.ndarray

        """
        return np.zeros(axis.shape[0])
