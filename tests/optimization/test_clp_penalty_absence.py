"""Equal-area penalties with missing CLPs follow the author reference."""

import warnings

import numpy as np
import pytest

from glotaran.model.clp_penalties import EqualAreaPenalty
from glotaran.optimization.estimation import OptimizationEstimation
from glotaran.optimization.matrix import OptimizationMatrix
from glotaran.optimization.penalty import calculate_clp_penalties
from glotaran.optimization.penalty import calculate_clp_penalty_diagnostics


def _inputs(labels, values):
    matrix = OptimizationMatrix(labels, np.eye(len(labels)))
    estimation = OptimizationEstimation(np.asarray(values, dtype=float), np.empty(0))
    return [matrix, matrix], [estimation, estimation], np.asarray([0.0, 1.0])


def _penalty(source="a", target="b", **kwargs):
    return EqualAreaPenalty(
        type="equal_area", source=source, target=target, parameter=2.0, weight=3.0, **kwargs
    )


@pytest.mark.parametrize("relative", [False, True])
@pytest.mark.parametrize(
    ("labels", "values", "message"),
    [(["a"], [4.0], "target clp b"), (["b"], [2.0], "source clp a")],
)
def test_missing_one_clp_skips_penalty_and_diagnostic(labels, values, message, relative):
    matrices, estimations, axis = _inputs(labels, values)
    penalty = _penalty(relative=relative)
    with pytest.warns(UserWarning, match=message):
        assert calculate_clp_penalties(matrices, estimations, axis, [penalty]).size == 0
    with pytest.warns(UserWarning, match=message):
        assert calculate_clp_penalty_diagnostics(matrices, estimations, axis, [penalty]) == []


def test_both_missing_are_silently_skipped_but_zero_values_are_not_absent():
    matrices, estimations, axis = _inputs(["a", "b"], [0.0, 0.0])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert calculate_clp_penalties(matrices, estimations, axis, [_penalty("c", "d")]).size == 0
        np.testing.assert_array_equal(
            calculate_clp_penalties(matrices, estimations, axis, [_penalty()]), [0.0]
        )


def test_out_of_range_interval_is_skipped_and_other_diagnostics_remain_aligned():
    matrices, estimations, axis = _inputs(["a", "b"], [4.0, 1.0])
    skipped = _penalty(source_intervals=(2.0, 3.0))
    valid = _penalty()
    with pytest.warns(UserWarning, match="source clp a"):
        np.testing.assert_array_equal(
            calculate_clp_penalties(matrices, estimations, axis, [skipped, valid]), [12.0]
        )
    with pytest.warns(UserWarning, match="source clp a"):
        diagnostics = calculate_clp_penalty_diagnostics(matrices, estimations, axis, [skipped, valid])
    assert len(diagnostics) == 1
    assert diagnostics[0].source_area == 8.0
    assert diagnostics[0].target_area == 2.0
    assert diagnostics[0].penalty == 12.0
