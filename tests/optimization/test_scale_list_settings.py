"""Independent arithmetic and persistence tests for the port's optimizer options."""

import numpy as np
import pytest

from glotaran.io import load_scheme
from glotaran.io import save_scheme
from glotaran.model.parameter_penalties import EqualParameterPenalty
from glotaran.optimization.optimization import Optimization
from glotaran.parameter import Parameters
from glotaran.project.library import ModelLibrary
from glotaran.project.scheme import Scheme
from glotaran.model.clp_penalties import EqualAreaPenalty
from glotaran.optimization.estimation import OptimizationEstimation
from glotaran.optimization.matrix import OptimizationMatrix
from glotaran.optimization.penalty import calculate_clp_penalties


def make_optimization(source=6, target=2, factor=2, **kwargs):
    parameters = Parameters.from_list([
        ["source", source], ["target", target, {"vary": False}],
        ["factor", factor, {"vary": False}],
    ])
    return Optimization(
        models=[], parameters=parameters, library=ModelLibrary.from_dict({"baseline": {"type": "baseline"}}),
        parameter_penalties=[EqualParameterPenalty(
            source="source", target="target", parameter="factor", weight=3,
        )], verbose=False, raise_exception=True, **kwargs,
    )


def test_parameter_penalty_is_in_optimizer_objective():
    optimization = make_optimization()
    np.testing.assert_allclose(optimization.objective_function(np.array([6.])), [1.5, -1.])
    parameters, _, info = optimization.run()
    assert parameters.get("source").value == pytest.approx(4.)
    assert info.cost < 1e-15
    np.testing.assert_allclose(info.additional_parameter_penalty, [0., 0.], atol=1e-7)


def test_parameter_penalty_near_zero_has_constant_length_and_warns_once():
    optimization = make_optimization(source=0)
    with pytest.warns(UserWarning, match="absolute equal parameter"):
        np.testing.assert_allclose(optimization.calculate_penalty(), [-12, -12])
    np.testing.assert_allclose(optimization.calculate_penalty(), [-12, -12])


@pytest.mark.parametrize("scale", [2., "jac", [2.], {"source": 2.}])
def test_parameter_scaling_optimizes(scale):
    parameters, _, _ = make_optimization(x_scale=scale).run()
    assert parameters.get("source").value == pytest.approx(4.)


@pytest.mark.parametrize("scale", [0, -1, float("nan"), "invalid", [1, 0]])
def test_invalid_scheme_scaling_is_rejected(scale):
    with pytest.raises(ValueError):
        Scheme.from_dict({"library": {"baseline": {"type": "baseline"}}, "experiments": {}, "x_scale": scale})


@pytest.mark.parametrize("scale", [2., "jac", [2.], {"source": 2.}])
def test_scheme_penalties_and_scaling_yaml_roundtrip(tmp_path, scale):
    spec = {"library": {"baseline": {"type": "baseline"}}, "experiments": {}, "x_scale": scale,
            "parameter_penalties": [{"type": "equal", "source": "source",
                                     "target": "target", "parameter": 2., "weight": 3.}]}
    scheme = Scheme.from_dict(spec)
    path = tmp_path / "scheme.yml"
    save_scheme(scheme, path)
    assert load_scheme(path).model_dump(exclude_unset=True) == scheme.model_dump(exclude_unset=True)


@pytest.mark.parametrize("relative, expected", [(False, 12.), (True, 1.5)])
def test_equal_area_relative_arithmetic(relative, expected):
    matrices = [OptimizationMatrix(["a", "b"], np.eye(2))] * 2
    estimations = [OptimizationEstimation(np.array([6., 2.]), np.zeros(2))] * 2
    penalty = EqualAreaPenalty(type="equal_area", source="a", target="b",
                               parameter=2., weight=3., relative=relative)
    np.testing.assert_allclose(calculate_clp_penalties(matrices, estimations, [1, 2], [penalty]), [expected])
