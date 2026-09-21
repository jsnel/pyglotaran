"""Global soft constraints between nonlinear parameters."""

from typing import Literal

from glotaran.model.item import Item
from glotaran.model.item import ParameterType


class EqualParameterPenalty(Item):
    """Add two symmetric ratio residuals for source = parameter * target.

    Near zero, both residuals use the weighted absolute difference instead.
    This preserves the residual vector length but changes its units.
    """

    type: Literal["equal"] = "equal"
    source: ParameterType
    target: ParameterType
    parameter: ParameterType
    weight: float
