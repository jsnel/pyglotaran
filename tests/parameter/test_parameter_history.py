from __future__ import annotations

import numpy as np

from glotaran.parameter.parameter_history import ParameterHistory
from glotaran.parameter.parameters import Parameters


def test_parameter_history():
    group0 = Parameters.from_list([["1", 1], ["2", 4]])
    group1 = Parameters.from_list([["1", 2], ["2", 5]])
    group2 = Parameters.from_list([["1", 3], ["2", 6]])

    history = ParameterHistory()

    history.append(group0)

    assert history.parameter_labels == ["evaluation", "1", "2"]

    assert history.number_of_records == 1
    assert all(history.get_parameters(0) == [0, 1, 4])

    history.append(group1, evaluation=1)

    assert history.number_of_records == 2
    assert all(history.get_parameters(1) == [1, 2, 5])

    history.append(group2, evaluation=2)

    assert history.number_of_records == 3
    assert all(history.get_parameters(2) == [2, 3, 6])

    history_df = history.to_dataframe()

    assert all(history_df.columns == history.parameter_labels)
    assert np.all(history_df.to_numpy() == history.parameters)

    group2.set_from_history(history, 0)

    assert group2.get("1") == 1
    assert group2.get("2") == 4


def test_parameter_history_stores_user_values():
    """Non-negative parameters are stored with their value, not the optimizer's logarithm."""
    parameters = Parameters.from_list([["rate", 0.5, {"non-negative": True}]])
    history = ParameterHistory()

    history.append(parameters)
    parameters.get("rate").value = 2
    parameters.set_from_history(history, 0)

    assert history.get_parameters(0)[1] == 0.5
    assert parameters.get("rate").value == 0.5
