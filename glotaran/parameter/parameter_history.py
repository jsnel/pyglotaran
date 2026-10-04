"""The glotaran parameter history package."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from glotaran.parameter.parameters import Parameters
    from glotaran.typing.types import StrOrPath


class ParameterHistory:
    """A history of parameter values, one row per function evaluation.

    The values are the parameter values as the user sees them, not the values the optimizer
    works with (which are logarithms for non-negative parameters).
    """

    def __init__(self) -> None:  # noqa: D107
        self._parameter_labels: list[str] = []
        self._parameters: list[np.ndarray] = []
        self.source_path = "parameter_history.csv"

    @classmethod
    def from_dataframe(cls, history_df: pd.DataFrame) -> ParameterHistory:
        """Create a history from a pandas data frame.

        Parameters
        ----------
        history_df : pd.DataFrame
            The source data frame.

        Returns
        -------
        ParameterHistory
            The created history.
        """
        history = cls()

        history._parameter_labels = history_df.columns

        for parameter_values in history_df.to_numpy():
            history._parameters.append(parameter_values)

        return history

    @classmethod
    def from_csv(cls, path: StrOrPath) -> ParameterHistory:
        """Create a history from a csv file.

        Parameters
        ----------
        path : StrOrPath
            The path to the csv file.

        Returns
        -------
        ParameterHistory
            The created history.
        """
        history_df = pd.read_csv(path)
        return cls.from_dataframe(history_df)

    loader = from_csv

    @property
    def parameter_labels(self) -> list[str]:
        """Return the labels of the parameters in the history.

        Returns
        -------
        list[str]
            A list of parameter labels.
        """
        return self._parameter_labels

    @property
    def parameters(self) -> list[np.ndarray]:
        """Return the parameters in the history.

        Returns
        -------
        list[np.ndarray]
            A list of parameters in the history.
        """
        return self._parameters

    def __len__(self) -> int:
        """Return the number of records in the history."""
        return self.number_of_records

    @property
    def number_of_records(self) -> int:
        """Return the number of records in the history.

        Returns
        -------
        int
            The number of records.
        """
        return len(self._parameters)

    def to_dataframe(self) -> pd.DataFrame:
        """Create a data frame from the history.

        Returns
        -------
        pd.DataFrame
            The created data frame.
        """
        return pd.DataFrame(self._parameters, columns=self.parameter_labels)

    def to_csv(self, path: StrOrPath, delimiter: str = ",") -> None:
        """Write a :class:`ParameterHistory` to a CSV file.

        Parameters
        ----------
        path : StrOrPath
            The path to the CSV file.
        delimiter : str
            The delimiter of the CSV file.
        """
        self.source_path = Path(path).as_posix()
        self.to_dataframe().to_csv(path, sep=delimiter, index=False)

    def append(self, parameters: Parameters, evaluation: int = 0) -> None:
        """Append the values of :class:`Parameters` to the history.

        Parameters
        ----------
        parameters : Parameters
            The parameters to append.
        evaluation: int
            Number of the function evaluation the values belong to, 0 for the initial values.

        Raises
        ------
        ValueError
            Raised if the parameter labels differs from previous.
        """
        parameters.update_parameter_expression()
        parameter_labels = ["evaluation", *(parameter.label for parameter in parameters.all())]
        if len(self._parameter_labels) == 0:
            self._parameter_labels = parameter_labels
        if parameter_labels != self.parameter_labels:
            msg = "Cannot append parameters. Parameter labels do not match existing."
            raise ValueError(msg)

        self._parameters.append(
            np.array([evaluation, *(parameter.value for parameter in parameters.all())])
        )

    def get_parameters(self, index: int) -> np.ndarray:
        """Get parameters for a history index.

        Parameters
        ----------
        index : int
            The history index.

        Returns
        -------
        np.ndarray
            The parameter values at the history index as array.
        """
        return self._parameters[index]
