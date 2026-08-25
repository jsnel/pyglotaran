"""This module contains parameter penalty items."""

from __future__ import annotations

from glotaran.model.item import ParameterType
from glotaran.model.item import TypedItem
from glotaran.model.item import item


@item
class ParameterPenalty(TypedItem):
    """Baseclass for parameter penalties."""


@item
class EqualParameterPenalty(ParameterPenalty):
    """Encourage a scaled equality relation between two parameters.

    The additional residual terms are added as
    ``weight * (source / (parameter * target) - 1)`` and
    ``weight * ((parameter * target) / source - 1)``.
    """

    type: str = "equal"
    source: ParameterType
    target: ParameterType
    parameter: ParameterType
    weight: float
