"""This package contains the initial concentration item."""

from __future__ import annotations

import numpy as np

from glotaran.model import ItemIssue
from glotaran.model import Model
from glotaran.model import ModelItem
from glotaran.model import ParameterType
from glotaran.model import attribute
from glotaran.model import item
from glotaran.parameter import Parameters


class InitialConcentrationIssue(ItemIssue):
    """Issue for mismatched initial-concentration compartments and parameters."""

    def __init__(
        self,
        label: str,
        compartments: list[str],
        parameters: list[ParameterType],
    ):
        self.label = label
        self.compartments = compartments
        self.parameters = parameters

    def to_string(self) -> str:
        """Get the issue as string."""
        compartment_count = len(self.compartments)
        parameter_count = len(self.parameters)
        compartment_name = "compartment" if compartment_count == 1 else "compartments"
        parameter_name = "parameter" if parameter_count == 1 else "parameters"
        return (
            f"Initial concentration '{self.label}' has {compartment_count} "
            f"{compartment_name} but {parameter_count} {parameter_name}. Expected one "
            f"parameter per compartment. Compartments: {self.compartments}. "
            f"Parameters: {self.parameters}."
        )


def validate_initial_concentration_parameters(
    parameters: list[ParameterType],
    initial_concentration: InitialConcentration,
    model: Model,
    model_parameters: Parameters | None,
) -> list[ItemIssue]:
    """Validate the one-to-one mapping of compartments to parameters."""
    if len(initial_concentration.compartments) != len(parameters):
        return [
            InitialConcentrationIssue(
                initial_concentration.label,
                initial_concentration.compartments,
                parameters,
            )
        ]
    return []


@item
class InitialConcentration(ModelItem):
    """An initial concentration describes the population of the compartments at
    the beginning of an experiment."""

    compartments: list[str]
    parameters: list[ParameterType] = attribute(
        validator=validate_initial_concentration_parameters
    )
    exclude_from_normalize: list[str] = []

    def normalized(self) -> np.ndarray:
        normalized = np.array(self.parameters)
        idx = [c not in self.exclude_from_normalize for c in self.compartments]
        normalized[idx] /= np.sum(normalized[idx])
        return normalized
