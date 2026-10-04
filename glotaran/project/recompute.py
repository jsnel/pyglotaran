"""Recompute the results of a recorded fit from its input data."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING
from typing import Any
from warnings import warn

import numpy as np
import pandas as pd

from glotaran.io import load_parameters
from glotaran.io import load_scheme
from glotaran.project.data_summary import compare_data_summaries
from glotaran.project.data_summary import summarize_data
from glotaran.project.record import environment_versions
from glotaran.project.record import find_record
from glotaran.project.record import read_record
from glotaran.project.record import resolve_results_folder
from glotaran.project.result import Reconstruction
from glotaran.utils.io import load_datasets

if TYPE_CHECKING:
    from pathlib import Path

    from glotaran.project.record import RecordReference
    from glotaran.project.result import Result
    from glotaran.typing.types import DatasetMappable

DRIFT_RTOL = 1e-6
"""Relative difference of a recomputed fit statistic from the recorded one that gives a warning.

Provisional, until recompute has been measured across machines and library versions.
"""


def recompute(
    record: str | Path | RecordReference,
    datasets: DatasetMappable,
    *,
    allow_data_mismatch: bool = False,
) -> Result:
    """Recompute the results of a recorded fit from its input data.

    The recorded scheme is evaluated once at the recorded optimized parameters, with the
    recorded optimizer settings. Records contain no data, so the input data of the fit must be
    supplied. The result is not recorded.

    Parameters
    ----------
    record : str | Path | RecordReference
        A record id (``"002"``) or uid in the base folder of records (see
        :func:`~glotaran.project.record.resolve_results_folder`), the path of a record or an
        export, or ``Result.record``.
    datasets : DatasetMappable
        The input data of the fit, by dataset label.
    allow_data_mismatch : bool
        Recompute also if the data differ from the recorded data summary. The result lists
        the differences.

    Returns
    -------
    Result
        The recomputed result. ``Result.reconstruction`` holds the metadata of the original fit
        and of the reconstruction. Its statistics are those of the one evaluation; the
        standard errors are those of the original fit, the covariance matrix is not
        recovered.

    Raises
    ------
    ValueError
        If the data differ from the recorded data summary and ``allow_data_mismatch`` is
        ``False``.
    """
    folder = find_record(record, resolve_results_folder())
    original = read_record(folder)
    datasets = load_datasets(datasets)
    recorded_data = original.get("data", {})
    data_differences = compare_data_summaries(
        recorded_data,
        {
            label: summarize_data(dataset)
            for label, dataset in datasets.items()
            if label in recorded_data
        },
    )
    if data_differences and not allow_data_mismatch:
        msg = (
            f"The data differ from the data of the fit recorded in '{folder}':\n"
            + _format_differences(data_differences)
            + "\nPass 'allow_data_mismatch=True' to recompute anyway."
        )
        raise ValueError(msg)

    result = load_scheme(folder / "scheme.yml").optimize(
        load_parameters(folder / "optimized_parameters.csv"),
        datasets,
        dry_run=True,
        verbose=False,
        **(original.get("optimizer") or {}),
    )
    result.optimization_info.termination_reason = (
        f"Recomputed with one function evaluation from '{folder.name}'."
    )
    result.optimized_parameters.record_uid = original.get("uid")
    drift = _compare_fit_statistics(original.get("summary", {}), result)
    if drifted := {
        name: values
        for name, values in drift.items()
        if values["relative_difference"] is None or values["relative_difference"] > DRIFT_RTOL
    }:
        warn(
            "The recomputed fit statistics differ from the recorded ones:\n"
            + "\n".join(f"  {name}: {_format_drift(values)}" for name, values in drifted.items()),
            stacklevel=2,
        )
    result.reconstruction = Reconstruction(
        original=original,
        original_cost_history=(
            pd.read_csv(folder / "cost_history.csv")["cost"].tolist()
            if (folder / "cost_history.csv").is_file()
            else []
        ),
        created=datetime.now().astimezone().isoformat(timespec="seconds"),
        environment=environment_versions(),
        data_differences=data_differences,
        drift=drift,
    )
    return result


def _compare_fit_statistics(
    summary: dict[str, Any], result: Result
) -> dict[str, dict[str, float | None]]:
    """Compare the cost and the RMSE per dataset with those of the record's summary.

    Statistics without a recorded value are left out.
    """
    pairs = {"cost": (summary.get("cost"), result.optimization_info.cost)}
    for label, optimization_result in result.optimization_results.items():
        recorded = summary.get("datasets", {}).get(label, {})
        for name in ("root_mean_square_error", "weighted_root_mean_square_error"):
            pairs[f"{label} {name}"] = (
                recorded.get(name),
                getattr(optimization_result.meta, name),
            )
    return {
        name: {
            "recorded": recorded,
            "recomputed": recomputed,
            "relative_difference": _relative_difference(recorded, recomputed),
        }
        for name, (recorded, recomputed) in pairs.items()
        if recorded is not None
    }


def _relative_difference(recorded: float, recomputed: float | None) -> float | None:
    if recomputed is None or not np.isfinite(recorded) or not np.isfinite(recomputed):
        return None
    if recorded == recomputed:
        return 0.0
    return float(abs(recomputed - recorded) / max(abs(recorded), abs(recomputed)))


def _format_drift(values: dict[str, float | None]) -> str:
    relative_difference = values["relative_difference"]
    return f"recorded {values['recorded']}, recomputed {values['recomputed']}, " + (
        "not comparable"
        if relative_difference is None
        else f"relative difference {relative_difference:.2e}"
    )


def _format_differences(differences: dict[str, list[str]]) -> str:
    return "\n".join(f"  {label}: {'; '.join(changes)}" for label, changes in differences.items())
