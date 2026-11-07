"""Parameter table contracts introduced by the scale_list branch."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from glotaran.io import load_parameters
from glotaran.io import save_parameters
from glotaran.parameter import Parameters

if TYPE_CHECKING:
    from pathlib import Path


def test_duplicate_labels_after_whitespace_normalization():
    with pytest.raises(ValueError, match="Duplicate parameter label"):
        Parameters.from_dataframe(pd.DataFrame({"label": ["a", " a "], "value": [1, 2]}))


def test_unknown_columns_warn_and_derived_columns_are_ignored():
    frame = pd.DataFrame({"label": ["a"], "value": [2], "T-value": [4], "unknown_column": [3]})
    with pytest.warns(UserWarning, match="unknown_column"):
        parameters = Parameters.from_dataframe(frame)
    assert parameters.get("a").value == 2
    assert "unknown_column" in frame  # Loading does not mutate the caller's table.


@pytest.mark.parametrize("suffix", ["csv", "tsv", "xlsx", "ods"])
def test_optimized_table_roundtrip(tmp_path: Path, suffix: str):
    parameters = Parameters.from_list([["free", 4], ["fixed", 3, {"vary": False}]])
    parameters.get("free").standard_error = 2
    parameters.get("fixed").standard_error = 1
    frame = parameters.to_dataframe(as_optimized=True).set_index("label")
    assert frame.loc["free", "T-value"] == 2
    assert np.isnan(frame.loc["fixed", "standard_error"])
    assert np.isnan(frame.loc["fixed", "T-value"])
    assert parameters.get("fixed").standard_error == 1
    path = tmp_path / f"parameters.{suffix}"
    save_parameters(parameters, path, as_optimized=True)
    restored = load_parameters(path)
    assert restored.get("free").value == 4
    assert restored.get("free").standard_error == 2
    assert not restored.get("fixed").vary


def test_near_zero_standard_error_does_not_report_t_value():
    parameters = Parameters.from_list([["a", 4]])
    parameters.get("a").standard_error = 0
    assert np.isnan(parameters.to_dataframe(as_optimized=True)["T-value"].iloc[0])


def test_bounds_validation_reports_fixed_parameters_without_clipping():
    parameters = Parameters.from_list([["a", 4, {"max": 3, "vary": False}]])
    assert "above maximum" in parameters.validate_bounds()[0]
    with pytest.raises(ValueError, match="above maximum"):
        parameters.validate_bounds(raise_exception=True)
    assert parameters.get("a").value == 4
