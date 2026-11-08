from __future__ import annotations

import numpy as np
import pytest

from glotaran.builtin.items.activation import MultiMultiGaussianActivation

_SHIFT = {"shift": [0.125, -0.25, 0.5]}
_POLYNOMIAL = {
    "dispersion_center": 500.0,
    "center_dispersion_coefficients": [0.25, 0.125],
    "width_dispersion_coefficients": [0.5, 0.25],
}
_SPLINE = {
    "width_dispersion_spline_knots": [400.0, 600.0],
    "width_dispersion_spline_values": [0.125, 0.625],
}
_RECIPROCAL_DISPERSION = (
    _POLYNOMIAL | _SPLINE | {"width_dispersion_spline_knots_in_wavelength": True}
)
_SKEW = {
    "width_dispersion_skewed_gaussian_amplitude": 0.5,
    "width_dispersion_skewed_gaussian_location": 500.0,
    "width_dispersion_skewed_gaussian_width": 100.0,
    "width_dispersion_skewed_gaussian_skewness": 0.5,
}
# At the three wavelengths the skewed Gaussian's log arguments are 0, 1, and 2.
_SKEW_AT_600 = 0.5 * np.exp(-4 * np.log(2.0) ** 3)


@pytest.mark.parametrize(
    ("settings", "center_offsets", "width_offsets"),
    [
        pytest.param(_SHIFT, [-0.125, 0.25, -0.5], [0.0, 0.0, 0.0], id="shift"),
        pytest.param(_POLYNOMIAL, [-0.125, 0.0, 0.375], [-0.25, 0.0, 0.75], id="polynomial"),
        pytest.param(_SPLINE, [0.0, 0.0, 0.0], [0.125, 0.375, 0.625], id="spline"),
        pytest.param(
            _SKEW | {"width_dispersion_skewed_gaussian_skewness": 0.0},
            [0.0, 0.0, 0.0],
            [0.03125, 0.5, 0.03125],
            id="zero-skew",
        ),
        pytest.param(_SKEW, [0.0, 0.0, 0.0], [0.0, 0.5, _SKEW_AT_600], id="nonzero-skew"),
        pytest.param(
            _SHIFT | _POLYNOMIAL | _SPLINE | _SKEW,
            [-0.25, 0.25, -0.125],
            [-0.125, 0.875, 1.375 + _SKEW_AT_600],
            id="combined",
        ),
        pytest.param(
            _RECIPROCAL_DISPERSION | {"reciproke_global_axis": True},
            [0.15625, 0.0, -5 / 72],
            [0.4375, 0.425, 35 / 72],
            id="reciprocal",
        ),
        pytest.param(
            _RECIPROCAL_DISPERSION | {"model_dispersion_with_wavenumber": True},
            [0.15625, 0.0, -5 / 72],
            [0.4375, 0.425, 35 / 72],
            id="wavenumber-alias",
        ),
    ],
)
def test_grouped_gaussian_index_dependence(
    settings: dict[str, bool | float | list[float]],
    center_offsets: list[float],
    width_offsets: list[float],
):
    activation = MultiMultiGaussianActivation(
        type="multi-multi-gaussian",
        center=[[1.25, 0.25], [2.5]],
        width=[[2.0, 4.0], [6.0]],
        scale=[[2.0, 0.5], [4.0]],
        backsweep=7.0,
        normalize=False,
        compartments={},
        **settings,
    )
    parameters = activation.parameters(np.array([400.0, 500.0, 600.0]))
    np.testing.assert_allclose(
        [[p.center for p in row] for row in parameters],
        np.asarray(center_offsets)[:, None] + [1.25, 1.5, 2.5],
        rtol=0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        [[p.width for p in row] for row in parameters],
        np.asarray(width_offsets)[:, None] + [2.0, 4.0, 6.0],
        rtol=0,
        atol=1e-12,
    )
    np.testing.assert_array_equal([[p.scale for p in row] for row in parameters], [[2, 1, 4]] * 3)
    np.testing.assert_array_equal(
        [[p.backsweep_period for p in row] for row in parameters], [[7] * 3] * 3
    )
