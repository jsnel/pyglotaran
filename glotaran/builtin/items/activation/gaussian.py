from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from typing import TYPE_CHECKING
from typing import Literal
from typing import overload

import numpy as np
from scipy.interpolate import CubicSpline

from glotaran.builtin.items.activation.activation import Activation
from glotaran.model.errors import GlotaranUserError
from glotaran.model.errors import ItemIssue
from glotaran.model.item import Attribute
from glotaran.model.item import ParameterType

if TYPE_CHECKING:
    from glotaran.parameter import Parameters
    from glotaran.typing.types import ArrayLike


class DispersionIssue(ItemIssue):
    def to_string(self) -> str:
        """Get the issue as string.

        Returns
        -------
        str
        """
        return "ActivationError: No dispersion coefficients defined."


class MultiGaussianIssue(ItemIssue):
    def __init__(self, centers: int, widths: int) -> None:
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


class SkewedGaussianDispersionIssue(ItemIssue):
    """Issue for an incomplete skewed-Gaussian width-dispersion term."""

    def to_string(self) -> str:
        """Get the issue as string."""
        return (
            "ActivationError: The skewed-Gaussian width dispersion requires "
            "amplitude, location, width, and skewness."
        )


def validate_multi_gaussian(
    value: list[ParameterType],
    activation: MultiGaussianActivation,
    parameters: Parameters | None,  # noqa: ARG001
) -> list[ItemIssue]:
    issues: list[ItemIssue] = []
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
    parameters: Parameters | None,  # noqa: ARG001
) -> list[ItemIssue]:
    issues: list[ItemIssue] = []
    if value is not None:
        len_centers = len(activation.center_dispersion_coefficients)
        len_widths = len(activation.width_dispersion_coefficients)
        has_spline = bool(activation.width_dispersion_spline_values)
        has_skewed = any(
            parameter is not None
            for parameter in (
                activation.width_dispersion_skewed_gaussian_amplitude,
                activation.width_dispersion_skewed_gaussian_location,
                activation.width_dispersion_skewed_gaussian_width,
                activation.width_dispersion_skewed_gaussian_skewness,
            )
        )
        if len_centers + len_widths == 0 and not has_spline and not has_skewed:
            issues.append(DispersionIssue())
    return issues


@dataclass
class GaussianActivationParameters:
    center: float
    width: float
    scale: float
    backsweep_period: float

    def shift(self, value: float) -> None:
        self.center -= value

    def disperse(
        self,
        index: float,
        center: float,
        center_coefficients: list[float],
        width_coefficients: list[float],
        *,
        reciproke_global_axis: bool,
    ) -> None:
        distance = (
            (1e3 / index - 1e3 / center) if reciproke_global_axis else (index - center) / 100
        )
        for i, coefficient in enumerate(center_coefficients):
            self.center += coefficient * np.power(distance, i + 1)
        for i, coefficient in enumerate(width_coefficients):
            self.width += coefficient * np.power(distance, i + 1)


class MultiGaussianActivation(Activation):
    type: Literal["multi-gaussian"]  # type:ignore[assignment]

    center: list[ParameterType] = Attribute(
        validator=validate_multi_gaussian,
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

    backsweep: ParameterType | None = Attribute(
        default=None, description="The period of the backsweep in a streak experiment."
    )
    dispersion_center: ParameterType | None = Attribute(
        default=None, validator=validate_dispersion, description="The center of the dispersion."
    )
    center_dispersion_coefficients: list[ParameterType] = Attribute(
        factory=list, description="The center coefficients of the dispersion."
    )
    width_dispersion_coefficients: list[ParameterType] = Attribute(
        factory=list, description="The width coefficients of the dispersion."
    )
    reciproke_global_axis: bool = Attribute(
        default=False,
        description="Set `True` if the global axis is reciproke (e.g. for wavennumbers),",
    )
    # ``model_dispersion_with_wavenumber`` is the v0.7 spelling.  Keep the
    # existing v0.8 spelling above for compatibility while accepting both.
    model_dispersion_with_wavenumber: bool = Attribute(
        default=False,
        description="Interpret the global axis and spline knots as wavenumbers.",
    )
    width_dispersion_spline_knots: list[float] = Attribute(
        factory=list,
        description="Numeric positions of knots for width dispersion.",
    )
    width_dispersion_spline_knots_in_wavelength: bool = Attribute(
        default=False,
        description="Interpret spline knots as wavelengths for a wavenumber axis.",
    )
    width_dispersion_spline_values: list[ParameterType] = Attribute(
        factory=list,
        description="Parameter values of the width-dispersion spline.",
    )
    width_dispersion_skewed_gaussian_amplitude: ParameterType | None = Attribute(
        default=None,
        description="Amplitude of the skewed-Gaussian width-dispersion term.",
    )
    width_dispersion_skewed_gaussian_location: ParameterType | None = Attribute(
        default=None,
        description="Location of the skewed-Gaussian width-dispersion term.",
    )
    width_dispersion_skewed_gaussian_width: ParameterType | None = Attribute(
        default=None,
        description="Width of the skewed-Gaussian width-dispersion term.",
    )
    width_dispersion_skewed_gaussian_skewness: ParameterType | None = Attribute(
        default=None,
        description="Skewness of the skewed-Gaussian width-dispersion term.",
    )

    @property
    def _uses_wavenumber(self) -> bool:
        """Whether dispersion calculations use reciprocal global-axis values."""
        return self.reciproke_global_axis or self.model_dispersion_with_wavenumber

    def _skewed_gaussian_parameters(self) -> tuple[float, float, float, float] | None:
        values = (
            self.width_dispersion_skewed_gaussian_amplitude,
            self.width_dispersion_skewed_gaussian_location,
            self.width_dispersion_skewed_gaussian_width,
            self.width_dispersion_skewed_gaussian_skewness,
        )
        if all(value is None for value in values):
            return None
        if any(value is None for value in values):
            raise GlotaranUserError(SkewedGaussianDispersionIssue().to_string())
        return tuple(float(value) for value in values)  # type: ignore[arg-type]

    def _dispersion_axis_value(self, value: float) -> float:
        return 1e3 / value if self._uses_wavenumber else value

    def _dispersion_distance(self, value: float) -> float:
        if self.dispersion_center is None:
            msg = f"No dispersion center defined for activation '{self.type}'."
            raise GlotaranUserError(msg)
        transformed_value = self._dispersion_axis_value(value)
        transformed_center = self._dispersion_axis_value(float(self.dispersion_center))
        return (
            transformed_value - transformed_center
            if self._uses_wavenumber
            else (transformed_value - transformed_center) / 100
        )

    def _width_spline(self) -> CubicSpline | None:
        if not self.width_dispersion_spline_knots and not self.width_dispersion_spline_values:
            return None
        if len(self.width_dispersion_spline_knots) != len(self.width_dispersion_spline_values):
            msg = "The number of width-dispersion spline knots must match the number of values."
            raise GlotaranUserError(msg)
        if len(self.width_dispersion_spline_knots) < 2:  # noqa: PLR2004
            msg = "A width-dispersion spline requires at least two knots."
            raise GlotaranUserError(msg)
        knots = np.asarray(self.width_dispersion_spline_knots, dtype=np.float64)
        values = np.asarray([float(value) for value in self.width_dispersion_spline_values])
        if self.width_dispersion_spline_knots_in_wavelength and self._uses_wavenumber:
            knots = 1e3 / knots
        order = np.argsort(knots)
        knots = knots[order]
        values = values[order]
        if np.any(np.diff(knots) <= 0):
            msg = "Width-dispersion spline knots must be strictly increasing."
            raise GlotaranUserError(msg)
        return CubicSpline(knots, values, bc_type="natural")

    def _width_dispersion(self, index: float, spline: CubicSpline | None) -> float:
        correction = 0.0
        if spline is not None:
            correction += float(spline(self._dispersion_axis_value(index)))
        skewed_parameters = self._skewed_gaussian_parameters()
        if skewed_parameters is not None:
            amplitude, location, width, skewness = skewed_parameters
            transformed_index = self._dispersion_axis_value(index)
            if np.isclose(skewness, 0.0):
                correction += amplitude * np.exp(
                    -np.log(2) * np.square(2 * (transformed_index - location) / width)
                )
            else:
                log_argument = 1 + 2 * skewness * (transformed_index - location) / width
                if log_argument > 0:
                    correction += amplitude * np.exp(
                        -np.log(2) * np.square(np.log(log_argument) / skewness)
                    )
        return correction

    def calculate_dispersion(self, axis: ArrayLike) -> ArrayLike:
        centers = self.center if isinstance(self.center, list) else [self.center]
        widths = self.width if isinstance(self.width, list) else [self.width]
        nr_gaussians = max(len(centers), len(widths))
        if len(centers) != nr_gaussians:
            centers = centers * nr_gaussians
        center = np.array([float(c) for c in centers], dtype=np.float64)
        if self.dispersion_center is None:
            return np.array([np.full(axis.size, c) for c in center])

        axis_values = np.asarray(axis, dtype=np.float64)
        dispersion_center = float(self.dispersion_center)
        distance = (
            (1e3 / axis_values - 1e3 / dispersion_center)
            if self._uses_wavenumber
            else (axis_values - dispersion_center) / 100
        )
        center_columns = center[:, np.newaxis] * np.ones((nr_gaussians, axis_values.size))
        for i, coefficient in enumerate(self.center_dispersion_coefficients):
            center_columns += float(coefficient) * np.power(distance, i + 1)
        return center_columns

    def is_index_dependent(self) -> bool:
        return (
            self.shift is not None
            or self.dispersion_center is not None
            or bool(self.width_dispersion_spline_values)
            or self._skewed_gaussian_parameters() is not None
        )

    @overload
    def parameters(
        self, global_axis: None = None
    ) -> list[GaussianActivationParameters] | list[list[GaussianActivationParameters]]: ...
    @overload
    def parameters(self, global_axis: ArrayLike) -> list[list[GaussianActivationParameters]]: ...

    def parameters(  # noqa: C901, PLR0912
        self, global_axis: ArrayLike | None = None
    ) -> list[GaussianActivationParameters] | list[list[GaussianActivationParameters]]:
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
        backsweep_period = float(self.backsweep) if self.backsweep is not None else 0

        parameters: list[GaussianActivationParameters] = [
            GaussianActivationParameters(
                float(center),
                float(width),
                float(scale),
                backsweep_period,
            )
            for center, width, scale in zip(centers, widths, scales, strict=False)
        ]

        if global_axis is None or not self.is_index_dependent():
            return parameters

        global_parameters: list[list[GaussianActivationParameters]] = [
            [replace(p) for p in parameters] for _ in global_axis
        ]

        if self.shift is not None:
            if global_axis.size != len(self.shift):
                msg = (
                    f"The number of shifts({len(self.shift)}) does not match "
                    f"the size of the global axis({global_axis.size})."
                )
                raise GlotaranUserError(msg)
            for ps, shift in zip(global_parameters, self.shift, strict=True):
                for p in ps:
                    p.shift(shift)  # type:ignore[arg-type]

        width_spline = self._width_spline()
        if self.dispersion_center is not None:
            for ps, index in zip(global_parameters, global_axis, strict=True):
                for p in ps:
                    p.disperse(
                        index,
                        self.dispersion_center,  # type:ignore[arg-type]
                        self.center_dispersion_coefficients,  # type:ignore[arg-type]
                        self.width_dispersion_coefficients,  # type:ignore[arg-type]
                        reciproke_global_axis=self._uses_wavenumber,
                    )

        if (
            width_spline is not None
            or self.width_dispersion_coefficients
            or self._skewed_gaussian_parameters() is not None
        ):
            for ps, index in zip(global_parameters, global_axis, strict=True):
                width_correction = self._width_dispersion(float(index), width_spline)
                for p in ps:
                    p.width += width_correction

        return global_parameters

    def calculate_function(self, axis: ArrayLike) -> ArrayLike:
        return np.sum(
            [
                p.scale * np.exp(-1 * (axis - p.center) ** 2 / (2 * p.width**2))  # type:ignore[union-attr]
                for p in self.parameters()
            ],
            axis=0,
        )


class GaussianActivation(MultiGaussianActivation):
    type: Literal["gaussian"]  # type:ignore[assignment]
    center: ParameterType  # type:ignore[assignment]
    width: ParameterType  # type:ignore[assignment]


class MultiMultiGaussianActivation(MultiGaussianActivation):
    """Grouped Gaussian activation with relative centers and scales.

    Each inner list is one group. Centers after the first are offsets from the
    group's first center, scales after the first are factors applied to the
    group's first scale, and widths remain absolute.
    """

    type: Literal["multi-multi-gaussian"]  # type:ignore[assignment]
    center: list[list[ParameterType]]
    width: list[list[ParameterType]]
    scale: list[list[ParameterType]] | None = None
    normalize_area: bool = Attribute(
        default=True, description="Normalize the total Gaussian area when enabled."
    )
    normarea: float = Attribute(default=1000.0, description="Target normalized Gaussian area.")

    def _flatten_parameters(self) -> list[GaussianActivationParameters]:
        if len(self.center) != len(self.width):
            msg = "Grouped Gaussian center and width groups must have equal lengths."
            raise GlotaranUserError(msg)
        if self.scale is not None and len(self.center) != len(self.scale):
            msg = "Grouped Gaussian center and scale groups must have equal lengths."
            raise GlotaranUserError(msg)
        flat: list[GaussianActivationParameters] = []
        for group_index, (centers, widths) in enumerate(zip(self.center, self.width, strict=True)):
            if len(centers) != len(widths):
                msg = (
                    f"Grouped Gaussian center and width group {group_index} "
                    "must have equal lengths."
                )
                raise GlotaranUserError(msg)
            group_scales = self.scale[group_index] if self.scale is not None else None
            if group_scales is not None and len(group_scales) != len(centers):
                msg = (
                    f"Grouped Gaussian center and scale group {group_index} "
                    "must have equal lengths."
                )
                raise GlotaranUserError(msg)
            base_center = float(centers[0])
            base_scale = float(group_scales[0]) if group_scales is not None else 1.0
            for component_index, (center, width) in enumerate(zip(centers, widths, strict=True)):
                scale = (
                    base_scale * float(group_scales[component_index])
                    if group_scales is not None and component_index > 0
                    else base_scale
                )
                flat.append(
                    GaussianActivationParameters(
                        base_center + (float(center) if component_index > 0 else 0.0),
                        float(width),
                        scale,
                        float(self.backsweep) if self.backsweep is not None else 0.0,
                    )
                )
        if self.normalize_area and self.normalize:
            total_area = float(np.sum([p.scale * abs(p.width) for p in flat]) * np.sqrt(2 * np.pi))
            if not np.isfinite(self.normarea) or self.normarea <= 0 or total_area <= 0:
                msg = "Grouped Gaussian normalization requires positive finite areas."
                raise GlotaranUserError(msg)
            factor = self.normarea / total_area
            for p in flat:
                p.scale *= factor
        return flat

    def parameters(
        self, global_axis: ArrayLike | None = None
    ) -> list[GaussianActivationParameters] | list[list[GaussianActivationParameters]]:
        parameters = self._flatten_parameters()
        if global_axis is None or not self.is_index_dependent():
            return parameters
        # Resolve grouped components to the existing flat implementation for
        # shifts, polynomial dispersion, splines, and skewed-Gaussian terms.
        flat_model = MultiGaussianActivation(
            type="multi-gaussian",
            compartments=self.compartments,
            center=[p.center for p in parameters],
            width=[p.width for p in parameters],
            scale=[p.scale for p in parameters],
            shift=self.shift,
            normalize=False,
            backsweep=self.backsweep,
            dispersion_center=self.dispersion_center,
            center_dispersion_coefficients=self.center_dispersion_coefficients,
            width_dispersion_coefficients=self.width_dispersion_coefficients,
            reciproke_global_axis=self.reciproke_global_axis,
            model_dispersion_with_wavenumber=self.model_dispersion_with_wavenumber,
            width_dispersion_spline_knots=self.width_dispersion_spline_knots,
            width_dispersion_spline_knots_in_wavelength=self.width_dispersion_spline_knots_in_wavelength,
            width_dispersion_spline_values=self.width_dispersion_spline_values,
            width_dispersion_skewed_gaussian_amplitude=self.width_dispersion_skewed_gaussian_amplitude,
            width_dispersion_skewed_gaussian_location=self.width_dispersion_skewed_gaussian_location,
            width_dispersion_skewed_gaussian_width=self.width_dispersion_skewed_gaussian_width,
            width_dispersion_skewed_gaussian_skewness=self.width_dispersion_skewed_gaussian_skewness,
        )
        return flat_model.parameters(global_axis)


class ConvMultiMultiGaussianActivation(MultiMultiGaussianActivation):
    """Grouped Gaussian activation with per-global-index broadening."""

    type: Literal["conv-multi-multi-gaussian"]  # type:ignore[assignment]
    convwidth: list[ParameterType]

    def is_index_dependent(self) -> bool:
        return True

    def parameters(
        self, global_axis: ArrayLike | None = None
    ) -> list[GaussianActivationParameters] | list[list[GaussianActivationParameters]]:
        if global_axis is None:
            return self._flatten_parameters()
        parameters = super().parameters(global_axis)
        if parameters and not isinstance(parameters[0], list):
            parameters = [[replace(parameter) for parameter in parameters] for _ in global_axis]
        if global_axis.size != len(self.convwidth):
            msg = (
                f"The number of convolution widths({len(self.convwidth)}) does not match "
                f"the size of the global axis({global_axis.size})."
            )
            raise GlotaranUserError(msg)
        for ps, convwidth in zip(parameters, self.convwidth, strict=True):
            width = float(convwidth)
            for p in ps:
                p.width = float(np.sqrt(width**2 + p.width**2))
        return parameters


class NormConvMultiMultiGaussianActivation(ConvMultiMultiGaussianActivation):
    """Convolved grouped Gaussian activation normalized after broadening."""

    type: Literal["norm-conv-multi-multi-gaussian"]  # type:ignore[assignment]

    def parameters(
        self, global_axis: ArrayLike | None = None
    ) -> list[GaussianActivationParameters] | list[list[GaussianActivationParameters]]:
        result = super().parameters(global_axis)
        if global_axis is None or not self.normalize_area or not self.normalize:
            return result
        for parameters in result:
            total_area = float(
                np.sum([p.scale * abs(p.width) for p in parameters]) * np.sqrt(2 * np.pi)
            )
            if not np.isfinite(self.normarea) or self.normarea <= 0 or total_area <= 0:
                msg = "Grouped Gaussian normalization requires positive finite areas."
                raise GlotaranUserError(msg)
            factor = self.normarea / total_area
            for p in parameters:
                p.scale *= factor
        return result
