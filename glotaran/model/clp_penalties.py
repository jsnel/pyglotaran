"""This module contains clp penalty items."""

from __future__ import annotations

from glotaran.model.item import ParameterType
from glotaran.model.item import TypedItem
from glotaran.model.item import item


@item
class ClpPenalty(TypedItem):
    """Baseclass for clp penalties."""


@item
class EqualAreaPenalty(ClpPenalty):
    """Forces the area of 2 clp to be the same.

    An equal area constraint adds a the difference of the sum of a
    compartments in the e matrix in one or more intervals to the scaled sum
    of the e matrix of one or more target compartments to residual. The additional
    residual is scaled with the weight.

    ``relative`` selects the penalty form:

    * ``False`` (default): the additional residual is the absolute difference
      ``source_area - parameter * target_area``, unchanged from prior releases.
    * ``True``: the additional residual is the relative difference
      ``source_area / (parameter * target_area) - 1``.

    Switching ``relative`` changes the optimization landscape and thus the
    optimized parameter values for models that opt in - this is deliberate,
    not a bug. Choosing a ``weight`` that balances this penalty against the
    other residuals is intentionally left to the user; there is no default
    that is correct for every model.
    """

    type: str = "equal_area"
    source: str
    source_intervals: list[tuple[float, float]]
    target: str
    target_intervals: list[tuple[float, float]]
    parameter: ParameterType
    weight: float
    relative: bool = False
