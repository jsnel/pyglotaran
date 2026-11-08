"""Tests for positional element scales with repeated elements."""

from __future__ import annotations

import numpy as np
import pytest

from glotaran.model.data_model import resolve_data_model
from glotaran.model.errors import GlotaranModelError
from glotaran.optimization.matrix import OptimizationMatrix
from glotaran.parameter import Parameters
from glotaran.project.scheme import Scheme


def _scheme() -> Scheme:
    return Scheme.from_dict(
        {
            "library": {
                "artifact": {"type": "coherent-artifact", "order": 1},
                "kinetic": {"type": "kinetic", "rates": {"(s1, s1)": "rate"}},
                "kinetic2": {"type": "kinetic", "rates": {"(s1, s1)": "rate2"}},
            },
            "experiments": {
                "experiment": {
                    "datasets": {
                        "data": {
                            "elements": ["artifact", "artifact", "kinetic"],
                            "element_scale": ["scale.a", "scale.b", "scale.c"],
                            "normalize_element_scales": True,
                            "global_elements": ["kinetic", "kinetic", "kinetic2"],
                            "global_element_scale": ["scale.a", "scale.b", "scale.c"],
                            "normalize_global_element_scales": True,
                            "activations": {
                                "irf": {
                                    "type": "gaussian",
                                    "compartments": {"artifact": 1, "s1": 1},
                                    "not_normalized_compartments": ["artifact"],
                                    "center": 0,
                                    "width": 0.1,
                                }
                            },
                        }
                    }
                }
            },
        }
    )


def _parameters() -> Parameters:
    return Parameters.from_parameter_dict_list(
        [
            {"label": "rate", "value": 0.5},
            {"label": "rate2", "value": 1.1},
            {"label": "scale.a", "value": 1.0},
            {"label": "scale.b", "value": 2.0},
            {"label": "scale.c", "value": 3.0},
        ]
    )


def test_positional_scales_preserve_repeated_artifact_and_resolve_parameters() -> None:
    scheme = _scheme()
    # The native scheme remains serializable and retains the repeated labels.
    scheme = Scheme.from_dict(scheme.model_dump())
    data_model = scheme.experiments["experiment"].datasets["data"]
    assert data_model.elements == ["artifact", "artifact", "kinetic"]
    assert data_model.element_scale == ["scale.a", "scale.b", "scale.c"]
    assert data_model.global_element_scale == ["scale.a", "scale.b", "scale.c"]
    parameters = _parameters()
    axis = np.array([-0.1, 0.0, 0.1])

    def artifact_column() -> np.ndarray:
        resolved = resolve_data_model(data_model, scheme.library, parameters)
        matrix = OptimizationMatrix.from_data_model(resolved, np.array([500.0]), axis, None)
        assert matrix.clp_axis.count("artifact_derivative_0") == 1
        return matrix.array[:, matrix.clp_axis.index("artifact_derivative_0")]

    np.testing.assert_allclose(artifact_column(), 0.5 * np.exp(-(axis**2) / (2 * 0.1**2)))
    parameters.get("scale.a").value = 2.0
    parameters.get("scale.b").value = 4.0
    np.testing.assert_allclose(artifact_column(), (6.0 / 9.0) * np.exp(-(axis**2) / (2 * 0.1**2)))


@pytest.mark.parametrize("is_global", [False, True])
def test_positional_scale_length_is_checked(is_global: bool) -> None:
    scheme = _scheme()
    data_model = scheme.experiments["experiment"].datasets["data"]
    if is_global:
        data_model.global_element_scale = [1.0]
    else:
        data_model.element_scale = [1.0]
    resolved = resolve_data_model(data_model, scheme.library, _parameters())
    with pytest.raises(GlotaranModelError, match=r"number of positional.*scales"):
        OptimizationMatrix.from_data_model(
            resolved, np.array([500.0]), np.array([0.0, 1.0]), None, global_matrix=is_global
        )


def test_positional_global_scales_use_all_repeated_entries() -> None:
    scheme = _scheme()
    model = scheme.experiments["experiment"].datasets["data"]
    parameters = _parameters()
    resolved = resolve_data_model(model, scheme.library, parameters)
    global_axis = np.array([0.0, 0.5, 1.0])
    matrix = OptimizationMatrix.from_data_model(
        resolved, global_axis, np.array([0.0]), None, global_matrix=True
    )
    assert matrix.clp_axis == ["s1"]
    assert resolved.global_elements is not None

    def one_element(index: int) -> np.ndarray:
        single = resolved.model_copy(
            update={
                "global_elements": [resolved.global_elements[index]],
                "global_element_scale": [1.0],
            }
        )
        return OptimizationMatrix.from_data_model(
            single, global_axis, np.array([0.0]), None, global_matrix=True
        ).array

    repeated, distinct = one_element(0), one_element(2)
    assert not np.allclose(repeated, distinct)
    np.testing.assert_allclose(matrix.array, (3.0 * repeated + 3.0 * distinct) / 6.0)

    parameters.get("scale.a").value = 5.0
    changed_model = resolve_data_model(model, scheme.library, parameters)
    changed = OptimizationMatrix.from_data_model(
        changed_model, global_axis, np.array([0.0]), None, global_matrix=True
    )
    np.testing.assert_allclose(changed.array, (7.0 * repeated + 3.0 * distinct) / 10.0)
