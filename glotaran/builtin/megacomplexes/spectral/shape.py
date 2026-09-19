"""This package contains the spectral shape item."""

import numpy as np

from glotaran.model import ModelItemTyped
from glotaran.model import ParameterType
from glotaran.model import item


def _as_parameter_list(value):
    return value if isinstance(value, list) else [value]


def _calculate_skewed_gaussian_component(
    axis: np.ndarray,
    amplitude: float | None,
    location: float,
    width: float,
    skewness: float,
) -> np.ndarray:
    if np.allclose(skewness, 0):
        component = np.exp(-np.log(2) * np.square(2 * (axis - location) / width))
    else:
        log_args = 1 + (2 * skewness * (axis - location) / width)
        component = np.zeros(log_args.shape)
        valid_arg_mask = log_args > 0
        component[valid_arg_mask] = np.exp(
            -np.log(2) * np.square(np.log(log_args[valid_arg_mask]) / skewness)
        )
    if amplitude is not None:
        component *= amplitude
    return component


@item
class SpectralShape(ModelItemTyped):
    pass


@item
class SpectralShapeGaussian(SpectralShape):
    """A Gaussian spectral shape"""

    type: str = "gaussian"
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
        shape = np.exp(-np.log(2) * np.square(2 * (axis - self.location) / self.width))
        if self.amplitude is not None:
            shape *= self.amplitude
        return shape


@item
class SpectralShapeSkewedGaussian(SpectralShapeGaussian):
    """A skewed Gaussian spectral shape"""

    type: str = "skewed-gaussian"
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


@item
class SpectralShapeSkewedGaussianSum(SpectralShape):
    """A sum of skewed Gaussian spectral shapes."""

    type: str = "skewed-gaussian-sum"
    amplitude: list[ParameterType] | None = None
    location: list[ParameterType]
    width: list[ParameterType]
    skewness: list[ParameterType]

    def calculate(self, axis: np.ndarray) -> np.ndarray:
        """Calculate and sum the skewed Gaussian components for ``axis``."""
        component_count = len(self.location)
        amplitudes = self.amplitude or [None] * component_count
        if len({component_count, len(self.width), len(self.skewness), len(amplitudes)}) != 1:
            raise ValueError(
                "Skewed Gaussian sum parameters amplitude, location, width, and "
                "skewness must have the same number of components."
            )

        return sum(
            (
                _calculate_skewed_gaussian_component(axis, amplitude, location, width, skewness)
                for amplitude, location, width, skewness in zip(
                    amplitudes, self.location, self.width, self.skewness, strict=True
                )
            ),
            start=np.zeros(axis.shape, dtype=float),
        )


@item
class SpectralShapeOne(SpectralShape):
    """A constant spectral shape with value 1"""

    type: str = "one"

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


@item
class SpectralShapeZero(SpectralShape):
    """A constant spectral shape with value 0"""

    type: str = "zero"

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
