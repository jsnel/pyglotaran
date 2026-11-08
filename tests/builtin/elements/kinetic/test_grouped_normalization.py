from __future__ import annotations

import numpy as np
import pytest

from glotaran.builtin.elements.kinetic import KineticElement
from glotaran.builtin.items.activation import ConvMultiMultiGaussianActivation
from glotaran.builtin.items.activation import MultiMultiGaussianActivation
from glotaran.builtin.items.activation import NormConvMultiMultiGaussianActivation


@pytest.mark.parametrize(
    ("activation_type", "activation_class"),
    [
        ("multi-multi-gaussian", MultiMultiGaussianActivation),
        ("conv-multi-multi-gaussian", ConvMultiMultiGaussianActivation),
        ("norm-conv-multi-multi-gaussian", NormConvMultiMultiGaussianActivation),
    ],
)
@pytest.mark.parametrize(
    ("normalize", "normalize_area"), [(False, False), (False, True), (True, False), (True, True)]
)
def test_grouped_kinetic_normalization(
    activation_type: str,
    activation_class: type[MultiMultiGaussianActivation],
    normalize: bool,
    normalize_area: bool,
) -> None:
    element = KineticElement(label="kinetic", type="kinetic", rates={("s1", "s1"): 1.0})
    global_axis = np.array([400.0, 500.0])
    time = np.linspace(-0.1, 3.0, 12)
    rates = np.array([1.0])
    widths = np.array([0.1, 0.2, 0.4])
    scales = np.array([2.0, 1.0, 4.0])
    convwidth = np.array([0.05, 0.2])
    fields = {
        "type": activation_type,
        "compartments": {"s1": 1.0},
        "center": [[0.0, 0.1], [1.0]],
        "width": [[0.1, 0.2], [0.4]],
        "scale": [[2.0, 0.5], [4.0]],
        "normarea": 10.0,
    }
    if activation_class is not MultiMultiGaussianActivation:
        fields["convwidth"] = convwidth.tolist()
    raw = element.calculate_matrix_gaussian_activation(
        activation_class(**fields, normalize=False), global_axis, time, ["s1"], rates
    )
    actual = element.calculate_matrix_gaussian_activation(
        activation_class(**fields, normalize=normalize, normalize_area=normalize_area),
        global_axis,
        time,
        ["s1"],
        rates,
    )

    # The original kinetic code uses the sum of scales unless the IRF normalizes
    # by area. Ordinary convolution uses the area before broadening.
    if not normalize:
        expected = raw
    elif not normalize_area:
        expected = raw / scales.sum()
    else:
        if activation_class is NormConvMultiMultiGaussianActivation:
            widths = np.sqrt(widths**2 + convwidth[:, None] ** 2)
        area = np.sum(widths * scales, axis=-1) * np.sqrt(2 * np.pi)
        expected = raw * (10.0 / area)[..., None, None]
    np.testing.assert_allclose(actual, expected, rtol=1e-12)
