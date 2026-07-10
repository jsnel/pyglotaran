"""This package contains irf items."""

import numpy as np
from scipy.interpolate import CubicSpline

from glotaran.model import ModelError
from glotaran.model import ModelItemTyped
from glotaran.model import ParameterType
from glotaran.model import attribute
from glotaran.model import item
from glotaran.parameter import Parameter
from glotaran.parameter import Parameters


@item
class Irf(ModelItemTyped):
    """Represents an IRF."""


@item
class IrfMultiGaussian(Irf):
    """
    Represents a gaussian IRF.

    One width and one center is a single gauss.

    One center and multiple widths is a multiple gaussian.

    Multiple center and multiple widths is Double-, Triple- , etc. Gaussian.

    Parameters
    ----------

    label:
        label of the irf
    center:
        one or more center of the irf as parameter indices
    width:
        one or more widths of the gaussian as parameter index
    center_dispersion_coefficients:
        polynomial coefficients for the dispersion of the
        center as list of parameter indices. None for no dispersion.
    width_dispersion_coefficients:
        polynomial coefficients for the dispersion of the
        width as parameter indices. None for no dispersion.

    """

    type: str = "multi-gaussian"

    center: list[ParameterType]
    width: list[ParameterType]
    scale: list[ParameterType] | None = None
    shift: list[ParameterType] | None = None
    normalize: bool = True
    backsweep: bool = False
    backsweep_period: ParameterType | None = None

    def parameter(
        self, global_index: int, global_axis: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, bool, float]:
        """Returns the properties of the irf with shift applied."""

        centers = self.center if isinstance(self.center, list) else [self.center]
        centers = np.asarray([c.value for c in centers])

        widths = self.width if isinstance(self.width, list) else [self.width]
        widths = np.asarray([w.value for w in widths])

        len_centers = len(centers)
        len_widths = len(widths)
        if len_centers != len_widths:
            if min(len_centers, len_widths) != 1:
                raise ModelError(
                    f"len(centers) ({len_centers}) not equal "
                    f"len(widths) ({len_widths}) none of is 1."
                )
            if len_centers == 1:
                centers = np.asarray([centers[0] for _ in range(len_widths)])
            else:
                widths = np.asarray([widths[0] for _ in range(len_centers)])

        scales = self.scale if self.scale is not None else [1.0 for _ in centers]
        scales = scales if isinstance(scales, list) else [scales]
        scales = np.asarray(scales)

        shift = 0
        if self.shift is not None:
            if global_index >= len(self.shift):
                raise ModelError(
                    f"No shift parameter for index {global_index} "
                    f"({global_axis[global_index]}) in irf {self.label}"
                )
            shift = self.shift[global_index]

        backsweep = self.backsweep

        backsweep_period = self.backsweep_period.value if self.backsweep else 0

        return centers, widths, scales, shift, backsweep, backsweep_period

    def calculate(self, index: int, global_axis: np.ndarray, model_axis: np.ndarray) -> np.ndarray:
        centers, widths, scales, _, _, _ = self.parameter(index, global_axis)
        return sum(
            scale * np.exp(-1 * (model_axis - center) ** 2 / (2 * width**2))
            for center, width, scale in zip(centers, widths, scales)
        )

    def is_index_dependent(self):
        return self.shift is not None


@item
class IrfGaussian(IrfMultiGaussian):
    type: str = "gaussian"
    center: ParameterType
    width: ParameterType


@item
class IrfMultiMultiGaussian(IrfMultiGaussian):
    """
    Represents a multi-multi-gaussian IRF.

    center, width, and scale are lists of lists. Each sublist defines a group
    of gaussian shapes. Within each sublist, center parameters are additive
    relative to the first element (center[k] = center[0] + center[k] for k > 0)
    and scale parameters are multiplicative relative to the first element
    (scale[k] = scale[0] * scale[k] for k > 0). Width parameters are always
    absolute (independent).

    The resulting flat list of gaussians is equivalent to the multi-gaussian
    type after applying the transformations.
    """

    type: str = "multi-multi-gaussian"

    center: list[list[ParameterType]]
    width: list[list[ParameterType]]
    scale: list[list[ParameterType]] | None = None
    normalize_area: bool = True
    normarea: float = 1000.0

    def _fill_parameters(self, parameters: Parameters) -> None:
        """Fill nested list parameter attributes from string labels."""

        def _resolve(v: ParameterType) -> Parameter:
            label = v if isinstance(v, str) else v.label
            return parameters.get(label)

        self.center = [[_resolve(v) for v in sublist] for sublist in self.center]
        self.width = [[_resolve(v) for v in sublist] for sublist in self.width]
        if self.scale is not None:
            self.scale = [[_resolve(v) for v in sublist] for sublist in self.scale]

    def _iter_nested_parameter_labels(self):
        """Yield (field_name, label) pairs for nested list parameter attributes."""
        for sublist in self.center:
            for v in sublist:
                yield "center", (v if isinstance(v, str) else v.label)
        for sublist in self.width:
            for v in sublist:
                yield "width", (v if isinstance(v, str) else v.label)
        if self.scale is not None:
            for sublist in self.scale:
                for v in sublist:
                    yield "scale", (v if isinstance(v, str) else v.label)

    def parameter(
        self, global_index: int, global_axis: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, bool, float]:
        """Return IRF parameters with nested list expansion and shift applied.

        Within each sublist:
        - center[k] for k > 0 is added to center[0] (additive)
        - scale[k] for k > 0 is multiplied by scale[0] (multiplicative)
        - width values are absolute (unchanged)
        """
        n_sublists = len(self.center)
        if len(self.width) != n_sublists:
            raise ModelError(
                f"len(width) ({len(self.width)}) must equal len(center) ({n_sublists}) "
                f"for irf '{self.label}'."
            )
        if self.scale is not None and len(self.scale) != n_sublists:
            raise ModelError(
                f"len(scale) ({len(self.scale)}) must equal len(center) ({n_sublists}) "
                f"for irf '{self.label}'."
            )

        flat_centers: list[float] = []
        flat_widths: list[float] = []
        flat_scales: list[float] = []
        sublist_areas: list[float] = []
        gaussian_area_factor = np.sqrt(2 * np.pi)

        for sublist_index in range(n_sublists):
            center_sublist = self.center[sublist_index]
            width_sublist = self.width[sublist_index]
            if len(width_sublist) != len(center_sublist):
                raise ModelError(
                    f"len(width[{sublist_index}]) ({len(width_sublist)}) must equal "
                    f"len(center[{sublist_index}]) ({len(center_sublist)}) for irf '{self.label}'."
                )

            base_center = center_sublist[0].value
            transformed_centers = [base_center]
            transformed_centers.extend(
                base_center + center_value.value for center_value in center_sublist[1:]
            )
            transformed_widths = [w.value for w in width_sublist]

            if self.scale is not None:
                scale_sublist = self.scale[sublist_index]
                if len(scale_sublist) != len(center_sublist):
                    raise ModelError(
                        f"len(scale[{sublist_index}]) ({len(scale_sublist)}) must equal "
                        f"len(center[{sublist_index}]) ({len(center_sublist)}) "
                        f"for irf '{self.label}'."
                    )
                base_scale = scale_sublist[0].value
                transformed_scales = [base_scale]
                transformed_scales.extend(base_scale * s.value for s in scale_sublist[1:])
            else:
                transformed_scales = [1.0] * len(center_sublist)

            flat_centers.extend(transformed_centers)
            flat_widths.extend(transformed_widths)
            flat_scales.extend(transformed_scales)

            if self.normalize_area:
                # Area of sum of Gaussians in one sublist.
                sublist_area = float(
                    np.sum(np.asarray(transformed_scales) * np.abs(np.asarray(transformed_widths)))
                    * gaussian_area_factor
                )
                sublist_areas.append(sublist_area)

        centers = np.asarray(flat_centers)
        widths = np.asarray(flat_widths)
        scales = np.asarray(flat_scales)

        if self.normalize_area and self.normalize:
            total_area = float(np.sum(sublist_areas))
            if not np.isfinite(self.normarea) or self.normarea <= 0:
                raise ModelError(
                    f"Cannot normalize multi-multi-gaussian irf '{self.label}': "
                    f"non-finite or non-positive normarea {self.normarea}."
                )
            if not np.isfinite(total_area) or total_area <= 0:
                raise ModelError(
                    f"Cannot normalize multi-multi-gaussian irf '{self.label}': "
                    f"non-finite or non-positive total area {total_area}."
                )
            scales = scales * (self.normarea / total_area)

        shift = 0
        if self.shift is not None:
            if global_index >= len(self.shift):
                raise ModelError(
                    f"No shift parameter for index {global_index} "
                    f"({global_axis[global_index]}) in irf {self.label}"
                )
            shift = self.shift[global_index]

        backsweep = self.backsweep
        backsweep_period = self.backsweep_period.value if self.backsweep else 0

        return centers, widths, scales, shift, backsweep, backsweep_period


@item
class IrfConvMultiMultiGaussian(IrfMultiMultiGaussian):
    """Represents a convolved multi-multi-gaussian IRF.

    Extends ``type: multi-multi-gaussian`` with a per-index ``convwidth`` parameter list.
    Before the IRF is evaluated, each per-gaussian width is broadened in quadrature::

        cwidth_k = sqrt(convwidth[global_index]^2 + width_k^2)

    This models an additional Gaussian broadening (e.g. a laser pulse or detection
    response) that convolves with the underlying multi-multi-gaussian IRF.

    ``convwidth`` is a list of parameters, one per dataset index (analogous to ``shift``).
    """

    type: str = "conv-multi-multi-gaussian"

    convwidth: list[ParameterType]

    def parameter(
        self, global_index: int, global_axis: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, bool, float]:
        """Return IRF parameters with per-gaussian quadrature width broadening applied.

        Calls the parent ``multi-multi-gaussian`` parameter() method to obtain centers,
        scales, shift, normalization, and base widths, then replaces each width w_k with::

            cwidth_k = sqrt(convwidth[global_index]^2 + w_k^2)
        """
        centers, widths, scales, shift, backsweep, backsweep_period = super().parameter(
            global_index, global_axis
        )
        if global_index >= len(self.convwidth):
            raise ModelError(
                f"No convwidth parameter for index {global_index} "
                f"({global_axis[global_index]}) in irf {self.label}"
            )
        convwidth_val = (
            self.convwidth[global_index].value
            if isinstance(self.convwidth[global_index], Parameter)
            else float(self.convwidth[global_index])
        )
        widths = np.sqrt(convwidth_val**2 + widths**2)
        return centers, widths, scales, shift, backsweep, backsweep_period


@item
class IrfNormConvMultiMultiGaussian(IrfConvMultiMultiGaussian):
    """Represents a convolved multi-multi-gaussian IRF with true-area normalization.

    This type is backward-compatible with ``conv-multi-multi-gaussian`` in terms of
    width broadening, but area normalization is applied using the broadened widths:

        cwidth_k = sqrt(convwidth[global_index]^2 + width_k^2)
        area = sum_k(scale_k * |cwidth_k|) * sqrt(2*pi)

    Existing ``conv-multi-multi-gaussian`` keeps the previous behavior where
    normalization happens in the parent class before convwidth broadening.
    """

    type: str = "norm-conv-multi-multi-gaussian"

    def parameter(
        self, global_index: int, global_axis: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, bool, float]:
        """Return broadened IRF parameters with normalization on true broadened area."""
        centers, widths, scales, shift, backsweep, backsweep_period = super().parameter(
            global_index, global_axis
        )

        if self.normalize_area and self.normalize:
            total_area = float(np.sum(scales * np.abs(widths)) * np.sqrt(2 * np.pi))
            if not np.isfinite(self.normarea) or self.normarea <= 0:
                raise ModelError(
                    f"Cannot normalize norm-conv-multi-multi-gaussian irf '{self.label}': "
                    f"non-finite or non-positive normarea {self.normarea}."
                )
            if not np.isfinite(total_area) or total_area <= 0:
                raise ModelError(
                    f"Cannot normalize norm-conv-multi-multi-gaussian irf '{self.label}': "
                    f"non-finite or non-positive total area {total_area}."
                )
            scales = scales * (self.normarea / total_area)

        return centers, widths, scales, shift, backsweep, backsweep_period


@item
class IrfSpectralMultiGaussian(IrfMultiGaussian):
    """
    Represents a gaussian IRF.

    One width and one center is a single gauss.

    One center and multiple widths is a multiple gaussian.

    Multiple center and multiple widths is Double-, Triple- , etc. Gaussian.

    Parameters
    ----------

    label:
        label of the irf
    center:
        one or more center of the irf as parameter indices
    width:
        one or more widths of the gaussian as parameter index
    center_dispersion_coefficients:
        list of parameters with polynomial coefficients describing
        the dispersion of the irf center location. None for no dispersion.
    width_dispersion_skewed_gaussian_amplitude:
        amplitude of an additional skewed-gaussian correction term for the
        dispersion of the irf width.
    width_dispersion_skewed_gaussian_location:
        location of the skewed-gaussian correction term. Interpreted on the
        spectral axis or wavenumber axis depending on
        ``model_dispersion_with_wavenumber``.
    width_dispersion_skewed_gaussian_width:
        width of the skewed-gaussian correction term.
    width_dispersion_skewed_gaussian_skewness:
        skewness of the skewed-gaussian correction term.
    width_dispersion_coefficients:
        list of parameters with polynomial coefficients describing
        the dispersion of the width of the irf. None for no dispersion.
    width_dispersion_spline_knots:
        spline knot positions on the spectral axis or wavenumber axis depending
        on ``model_dispersion_with_wavenumber``.
    width_dispersion_spline_knots_in_wavelength:
        if ``True``, spline knots are interpreted as wavelength and converted
        internally to wavenumber only when ``model_dispersion_with_wavenumber``
        is ``True``.
    width_dispersion_spline_values:
        width values at the spline knots.

    """

    type: str = "spectral-multi-gaussian"
    dispersion_center: ParameterType | None = None
    center_dispersion_coefficients: list[ParameterType] = attribute(factory=list)
    width_dispersion_coefficients: list[ParameterType] = attribute(factory=list)
    # width_dispersion_spline_knots: list[float] = attribute(factory=list)
    width_dispersion_spline_knots: list[ParameterType] = attribute(factory=list)
    width_dispersion_spline_knots_in_wavelength: bool = False
    width_dispersion_spline_values: list[ParameterType] = attribute(factory=list)
    width_dispersion_skewed_gaussian_amplitude: ParameterType | None = None
    width_dispersion_skewed_gaussian_location: ParameterType | None = None
    width_dispersion_skewed_gaussian_width: ParameterType | None = None
    width_dispersion_skewed_gaussian_skewness: ParameterType | None = None
    model_dispersion_with_wavenumber: bool = False

    def _dispersion_axis_value(self, spectral_index: float) -> float:
        return 1e3 / spectral_index if self.model_dispersion_with_wavenumber else spectral_index

    def _dispersion_distance(self, spectral_index: float) -> float:
        if self.dispersion_center is None:
            raise ModelError(f"No dispersion center defined for irf '{self.label}'")
        transformed_index = self._dispersion_axis_value(spectral_index)
        transformed_center = self._dispersion_axis_value(self.dispersion_center)
        if self.model_dispersion_with_wavenumber:
            return transformed_index - transformed_center
        return (transformed_index - transformed_center) / 100

    def _width_dispersion_skewed_gaussian_parameters(self):
        parameters = (
            self.width_dispersion_skewed_gaussian_amplitude,
            self.width_dispersion_skewed_gaussian_location,
            self.width_dispersion_skewed_gaussian_width,
            self.width_dispersion_skewed_gaussian_skewness,
        )
        if all(parameter is None for parameter in parameters):
            return None
        if any(parameter is None for parameter in parameters):
            raise ModelError(
                "The skewed-gaussian width dispersion term for "
                f"irf '{self.label}' requires amplitude, location, width and skewness."
            )
        return parameters

    def _skewed_gaussian(self, spectral_index: float, parameters) -> float:
        if parameters is None:
            return 0.0

        amplitude, location, width, skewness = parameters
        transformed_index = self._dispersion_axis_value(spectral_index)
        if np.allclose(skewness, 0):
            return amplitude * np.exp(
                -np.log(2) * np.square(2 * (transformed_index - location) / width)
            )

        log_argument = 1 + (2 * skewness * (transformed_index - location) / width)
        if log_argument <= 0:
            return 0.0
        return amplitude * np.exp(-np.log(2) * np.square(np.log(log_argument) / skewness))

    def _evaluate_width_spline(self, spectral_index: float | None) -> float:
        if (
            spectral_index is None
            or not self.width_dispersion_spline_knots
            or not self.width_dispersion_spline_values
        ):
            return 0.0

        if len(self.width_dispersion_spline_knots) != len(self.width_dispersion_spline_values):
            raise ModelError(
                f"Spline definition error in '{self.label}': Number of knots "
                f"({len(self.width_dispersion_spline_knots)}) must match number of values "
                f"({len(self.width_dispersion_spline_values)})."
            )

        knots = np.asarray([float(knot) for knot in self.width_dispersion_spline_knots])
        values = np.asarray([float(value) for value in self.width_dispersion_spline_values])

        if (
            self.width_dispersion_spline_knots_in_wavelength
            and self.model_dispersion_with_wavenumber
        ):
            knots = 1e3 / knots

        order = np.argsort(knots)
        knots = knots[order]
        values = values[order]

        if np.any(np.diff(knots) <= 0):
            raise ModelError(
                f"Spline definition error in '{self.label}': Knot positions must be "
                "strictly increasing after optional unit conversion."
            )

        spline = CubicSpline(knots, values, bc_type="natural")
        return float(spline(self._dispersion_axis_value(spectral_index)))

    def parameter(self, global_index: int, global_axis: np.ndarray):
        """Returns the properties of the irf with shift and dispersion applied."""
        centers, widths, scale, shift, backsweep, backsweep_period = super().parameter(
            global_index, global_axis
        )

        index = global_axis[global_index] if global_index is not None else None

        if len(self.center_dispersion_coefficients) != 0:
            dist = self._dispersion_distance(index)
            for i, disp in enumerate(self.center_dispersion_coefficients):
                centers += disp * np.power(dist, i + 1)

        if len(self.width_dispersion_coefficients) != 0:
            dist = self._dispersion_distance(index)
            for i, disp in enumerate(self.width_dispersion_coefficients):
                widths = widths + disp * np.power(dist, i + 1)

        if self.width_dispersion_spline_values:
            widths += self._evaluate_width_spline(index)

        widths += self._skewed_gaussian(index, self._width_dispersion_skewed_gaussian_parameters())

        return centers, widths, scale, shift, backsweep, backsweep_period

    def calculate_dispersion(self, axis):
        dispersion = []
        for index, _ in enumerate(axis):
            center, _, _, _, _, _ = self.parameter(index, axis)
            dispersion.append(center)
        return np.asarray(dispersion).T

    def is_index_dependent(self):
        return (
            super().is_index_dependent()
            or self.dispersion_center is not None
            or bool(self.width_dispersion_spline_values)
            or self._width_dispersion_skewed_gaussian_parameters() is not None
        )


@item
class IrfSpectralGaussian(IrfSpectralMultiGaussian):
    type: str = "spectral-gaussian"
    center: ParameterType
    width: ParameterType
