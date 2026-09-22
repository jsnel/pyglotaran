"""Independent arithmetic and persistence tests for the port's optimizer options."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

import numpy as np
import pytest

from glotaran.io import load_scheme
from glotaran.io import save_scheme
from glotaran.model.clp_penalties import EqualAreaPenalty
from glotaran.model.parameter_penalties import EqualParameterPenalty
from glotaran.optimization.estimation import OptimizationEstimation
from glotaran.optimization.matrix import OptimizationMatrix
from glotaran.optimization.optimization import Optimization
from glotaran.optimization.penalty import calculate_clp_penalties
from glotaran.optimization.penalty import calculate_clp_penalty_diagnostics
from glotaran.parameter import Parameters
from glotaran.project.library import ModelLibrary
from glotaran.project.scheme import Scheme

if TYPE_CHECKING:
    from pathlib import Path


def make_optimization(source: float = 6, target: float = 2, factor: float = 2, **kwargs: Any):
    parameters = Parameters.from_list(
        [
            ["source", source],
            ["target", target, {"vary": False}],
            ["factor", factor, {"vary": False}],
        ]
    )
    return Optimization(
        models=[],
        parameters=parameters,
        library=ModelLibrary.from_dict({"baseline": {"type": "baseline"}}),
        parameter_penalties=[
            EqualParameterPenalty(
                source="source",
                target="target",
                parameter="factor",
                weight=3,
            )
        ],
        verbose=False,
        raise_exception=True,
        **kwargs,
    )


def test_parameter_penalty_is_in_optimizer_objective():
    optimization = make_optimization()
    np.testing.assert_allclose(optimization.objective_function(np.array([6.0])), [1.5, -1.0])
    parameters, _, info = optimization.run()
    assert parameters.get("source").value == pytest.approx(4.0)
    assert info.cost < 1e-15
    np.testing.assert_allclose(info.additional_parameter_penalty, [0.0, 0.0], atol=1e-7)


def test_parameter_penalty_near_zero_has_constant_length_and_warns_once():
    optimization = make_optimization(source=0)
    with pytest.warns(UserWarning, match="absolute equal parameter"):
        np.testing.assert_allclose(optimization.calculate_penalty(), [-12, -12])
    np.testing.assert_allclose(optimization.calculate_penalty(), [-12, -12])


@pytest.mark.parametrize("scale", [2.0, "jac", [2.0], {"source": 2.0}])
def test_parameter_scaling_optimizes(scale: float | str | list | dict):
    parameters, _, _ = make_optimization(x_scale=scale).run()
    assert parameters.get("source").value == pytest.approx(4.0)


@pytest.mark.parametrize("scale", [0, -1, float("nan"), "invalid", [1, 0]])
def test_invalid_scheme_scaling_is_rejected(scale: float | str | list):
    with pytest.raises(ValueError):
        Scheme.from_dict(
            {"library": {"baseline": {"type": "baseline"}}, "experiments": {}, "x_scale": scale}
        )


@pytest.mark.parametrize("scale", [2.0, "jac", [2.0], {"source": 2.0}])
def test_scheme_penalties_and_scaling_yaml_roundtrip(
    tmp_path: Path, scale: float | str | list | dict
):
    spec = {
        "library": {"baseline": {"type": "baseline"}},
        "experiments": {},
        "x_scale": scale,
        "parameter_penalties": [
            {
                "type": "equal",
                "source": "source",
                "target": "target",
                "parameter": 2.0,
                "weight": 3.0,
            }
        ],
    }
    scheme = Scheme.from_dict(spec)
    path = tmp_path / "scheme.yml"
    save_scheme(scheme, path)
    assert load_scheme(path).model_dump(exclude_unset=True) == scheme.model_dump(
        exclude_unset=True
    )


@pytest.mark.parametrize(("relative", "expected"), [(False, 12.0), (True, 1.5)])
def test_equal_area_relative_arithmetic(relative: bool, expected: float):
    matrices = [OptimizationMatrix(["a", "b"], np.eye(2))] * 2
    estimations = [OptimizationEstimation(np.array([6.0, 2.0]), np.zeros(2))] * 2
    penalty = EqualAreaPenalty(
        type="equal_area", source="a", target="b", parameter=2.0, weight=3.0, relative=relative
    )
    np.testing.assert_allclose(
        calculate_clp_penalties(matrices, estimations, [1, 2], [penalty]), [expected]
    )
    diagnostic = calculate_clp_penalty_diagnostics(matrices, estimations, [1, 2], [penalty])[0]
    assert diagnostic.source_area == 12
    assert diagnostic.target_area == 4
    assert diagnostic.penalty == expected
    assert type(diagnostic).model_validate_json(diagnostic.model_dump_json()) == diagnostic
