from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from glotaran.builtin.elements.clp_guide import SpectralModelClpGuideElement
from glotaran.builtin.elements.spectral import SpectralDataModel
from glotaran.builtin.elements.spectral import SpectralElement
from glotaran.model.data_model import DataModel
from glotaran.model.item import get_item_issues
from glotaran.parameter import Parameters

GAUSSIAN_SHAPE = {"type": "gaussian", "amplitude": 2.0, "location": 650.0, "width": 20.0}


def test_spectral_model_clp_guide_matrix_and_data():
    element = SpectralModelClpGuideElement(
        type="spectral-model-clp-guide", label="guide", target="s1", shapes={"s1": GAUSSIAN_SHAPE}
    )
    model = SpectralDataModel(elements=[element])
    model_axis = np.array([0.0])
    spectral_axis = np.array([640.0, 650.0, 660.0])

    clp_labels, matrix = element.calculate_matrix(model, spectral_axis, model_axis)
    data = element.calculate_data(model, model_axis, spectral_axis)

    assert clp_labels == ["s1"]
    np.testing.assert_array_equal(matrix, [[1.0]])
    np.testing.assert_allclose(data, [[1.0, 2.0, 1.0]])


def test_spectral_model_clp_guide_applies_spectral_axis_transform():
    element = SpectralModelClpGuideElement(
        type="spectral-model-clp-guide",
        label="guide",
        target="s1",
        shapes={"s1": GAUSSIAN_SHAPE | {"location": 1e7 / 650.0, "width": 100.0}},
    )
    model = SpectralDataModel(
        elements=[element], spectral_axis_inverted=True, spectral_axis_scale=1e7
    )

    generated = element.calculate_data(model, np.array([0.0]), np.array([640.0, 650.0, 660.0]))[0]

    assert generated[1] == pytest.approx(2.0)
    assert generated[1] > generated[0]
    assert generated[1] > generated[2]


def test_spectral_model_clp_guide_requires_target_shape():
    with pytest.raises(ValidationError, match="has no matching shape"):
        SpectralModelClpGuideElement(
            type="spectral-model-clp-guide",
            label="guide",
            target="s2",
            shapes={"s1": GAUSSIAN_SHAPE},
        )


def test_spectral_model_clp_guide_is_exclusive():
    guide = SpectralModelClpGuideElement(
        type="spectral-model-clp-guide", label="guide", target="s1", shapes={"s1": GAUSSIAN_SHAPE}
    )
    spectral = SpectralElement(type="spectral", label="spectral", shapes={"s1": GAUSSIAN_SHAPE})
    data_model = DataModel.create_class_for_elements({type(guide), type(spectral)})(
        elements=[guide, spectral]
    )

    issues = get_item_issues(data_model, Parameters.empty())

    assert any("Exclusive element 'guide'" in issue.to_string() for issue in issues)
