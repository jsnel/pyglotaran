from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from np.typing import ArrayLike

from glotaran.builtin.items.activation.activation import Activation
from glotaran.model import Attribute
from glotaran.model import GlotaranUserError
from glotaran.model import ItemIssue
from glotaran.model import Library
from glotaran.model import ParameterType
from glotaran.parameter import Parameters


class DispersionIssue(ItemIssue):
    def to_string(self) -> str:
        """Get the issue as string.

        Returns
        -------
        str
        """
        return "ActivationError: No dispersion coefficients defined."


class MultiGaussianIssue(ItemIssue):
    def __init__(self, centers: int, widths: int):
        self._centers = centers
        self._widths = widths

    def to_string(self) -> str:
        """Get the issue as string.

        Returns
        -------
        str
        """
        return (
            f"ActivationError: The size of the centers({self._centers}) "
            f"does not match the size of the width({self._widths})"
        )


def validate_multi_gaussian(
    value: list[ParameterType],
    activation: MultiGaussianActivation,
    library: Library,
    parameters: Parameters | None,
) -> list[ItemIssue]:
    issues = []
    if not isinstance(value, list):
        value = [value]
    len_centers = len(value)
    len_widths = len(activation.width) if isinstance(activation.width, list) else 1
    if len_centers - len_widths != 0 and len_centers != 1 and len_widths != 1:
        issues.append(MultiGaussianIssue(len_centers, len_widths))
    return issues


def validate_dispersion(
    value: ParameterType | None,
    activation: MultiGaussianActivation,
    library: Library,
    parameters: Parameters | None,
) -> list[ItemIssue]:
    issues = []
    if value is not None:
        len_centers = len(activation.center_dispersion_coefficients)
        len_widths = len(activation.width_dispersion_coefficients)
        if len_centers + len_widths == 0:
            issues.append(DispersionIssue())
    return issues


@dataclass
class GaussianActivationParameters:

    center: float
    width: float
    scale: float
    backsweep: bool
    backsweep_period: float

    def shift(self, value: float):
        self.center -= value


class MultiGaussianActivation(Activation):
    type: Literal["multi-gaussian"]

    center: list[ParameterType] = Attribute(
        validator=validate_multi_gaussian,  # type:ignore[arg-type]
        description="The center of the gaussian",
    )

    width: list[ParameterType] = Attribute(description="The width of the gaussian.")
    scale: list[ParameterType] | None = Attribute(
        default=None, description="The scales of the gaussians."
    )
    shift: list[ParameterType] | None = Attribute(
        default=None,
        description=(
            "A list parameters which gets subtracted from the centers along the global axis."
        ),
    )

    normalize: bool = Attribute(default=True, description="Whether to normalize the gaussians.")

    backsweep_period: ParameterType | None = Attribute(
        default=None, description="The period of the backsweep in a streak experiment."
    )
    dispersion_center: ParameterType | None = Attribute(
        default=None, validator=validate_dispersion, description="The center of the dispersion."
    )
    center_dispersion_coefficients: list[ParameterType] | None = Attribute(
        default=None, description="The center coefficients of the dispersion."
    )
    width_dispersion_coefficients: list[ParameterType] = Attribute(
        default=None, description="The width coefficients of the dispersion."
    )
    reciproke_global_axis: bool = Attribute(
        default=False,
        description="Set `True` if the global axis is reciproke (e.g. for wavennumbers),",
    )

    def parameters(
        self, global_axis: ArrayLike
    ) -> list[GaussianActivationParameters | list[GaussianActivationParameters]]:
        centers = self.center if isinstance(self.center, list) else [self.center]
        widths = self.width if isinstance(self.width, list) else [self.width]

        len_centers = len(centers)
        len_widths = len(widths)
        nr_gaussians = max(len_centers, len_widths)
        if len_centers != len_widths:
            if len_centers == 1:
                centers = centers * nr_gaussians
            else:
                widths = widths * nr_gaussians

        scales = self.scale or [1.0] * nr_gaussians
        backsweep = self.backsweep_period is not None
        backsweep_period = self.backsweep_period if backsweep else 0

        parameters = [
            GaussianActivationParameters(center, width, backsweep, backsweep_period)
            for center, width in zip(centers, widths)
        ]

        if self.shift is not None and global_axis.size != len(self.shift):
            raise GlotaranUserError(
                f"the number of shifts({len(self.shift)}) does not match "
                f"the size of the global axis({global_axis.size})."
            )


class GaussianActivation(MultiGaussianActivation):
    type: str = "gaussian"
    center: ParameterType
    width: ParameterType
