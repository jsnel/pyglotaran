from __future__ import annotations

from typing import TYPE_CHECKING
from warnings import warn

import numpy as np

from glotaran.model.clp_penalties import EqualAreaPenaltyDiagnostic

if TYPE_CHECKING:
    from glotaran.model.clp_penalties import EqualAreaPenalty
    from glotaran.optimization.estimation import OptimizationEstimation
    from glotaran.optimization.matrix import OptimizationMatrix
    from glotaran.typing.types import ArrayLike


def _get_area(
    label: str,
    intervals: list[tuple[float, float]] | tuple[float, float] | None,
    matrices: list[OptimizationMatrix],
    estimations: list[OptimizationEstimation],
    global_axis: ArrayLike,
) -> ArrayLike:
    global_axis = np.asarray(global_axis)
    intervals = [(-np.inf, np.inf)] if intervals is None else intervals
    intervals = [intervals] if isinstance(intervals, tuple) else intervals
    area = []
    for interval_lower, interval_upper in intervals:
        if interval_lower > global_axis[-1]:
            continue
        lower = max(interval_lower, np.min(global_axis))
        upper = min(interval_upper, np.max(global_axis))
        if lower > upper:
            lower, upper = upper, lower
        start = int(np.argmin(np.abs(global_axis - lower)))
        stop = int(np.argmin(np.abs(global_axis - upper))) + 1
        for matrix, estimation in zip(matrices[start:stop], estimations[start:stop], strict=True):
            if label in matrix.clp_axis:
                area.append(estimation.clp[matrix.clp_axis.index(label)])
    return np.asarray(area)


def calculate_clp_penalties(
    matrices: list[OptimizationMatrix],
    estimations: list[OptimizationEstimation],
    global_axis: ArrayLike,
    penalties: list[EqualAreaPenalty],
) -> ArrayLike:
    """Calculate the clp penalty.

    Parameters
    ----------
    clp_labels : list[list[str]]
        The clp labels.
    clps : list[ArrayLike]
        The clps.
    global_axis : ArrayLike
        The global axis.

    Returns
    -------
    list[float]
        The clp penalty.
    """
    values = []
    for penalty in penalties:
        source = float(
            np.sum(
                _get_area(
                    penalty.source, penalty.source_intervals, matrices, estimations, global_axis
                )
            )
        )
        target = float(penalty.parameter) * float(
            np.sum(
                _get_area(
                    penalty.target, penalty.target_intervals, matrices, estimations, global_axis
                )
            )
        )
        if penalty.relative and np.isclose(target, 0):
            warn(
                "Falling back to absolute equal area penalty near zero target area.", stacklevel=2
            )
        value = (
            (source / target - 1.0)
            if penalty.relative and not np.isclose(target, 0)
            else abs(source - target)
        )
        values.append(value * penalty.weight)
    return np.asarray(values)


def calculate_clp_penalty_diagnostics(
    matrices: list[OptimizationMatrix],
    estimations: list[OptimizationEstimation],
    global_axis: ArrayLike,
    penalties: list[EqualAreaPenalty],
) -> list[EqualAreaPenaltyDiagnostic]:
    """Capture final source and target areas without changing objective state."""
    values = calculate_clp_penalties(matrices, estimations, global_axis, penalties)
    return [
        EqualAreaPenaltyDiagnostic(
            source=penalty.source,
            target=penalty.target,
            source_intervals=penalty.source_intervals,
            target_intervals=penalty.target_intervals,
            source_area=float(
                np.sum(
                    _get_area(
                        penalty.source,
                        penalty.source_intervals,
                        matrices,
                        estimations,
                        global_axis,
                    )
                )
            ),
            target_area=float(
                np.sum(
                    _get_area(
                        penalty.target,
                        penalty.target_intervals,
                        matrices,
                        estimations,
                        global_axis,
                    )
                )
            ),
            parameter=float(penalty.parameter),
            relative=penalty.relative,
            weight=penalty.weight,
            penalty=float(value),
        )
        for penalty, value in zip(penalties, values, strict=True)
    ]
