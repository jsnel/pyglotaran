"""Summaries of the input data of a fit, to recognize the data without storing it."""

from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr

DATA_SUMMARY_RTOL = 1e-6
"""Differences of a statistic up to this fraction of the RMS of its array count as equal."""

STATISTICS = ("min", "max", "mean", "rms")


def summarize_data(data: xr.Dataset | xr.DataArray) -> dict[str, Any]:
    """Summarize the input data of one dataset.

    The summary holds the shape by dimension name and the ``min``, ``max``, ``mean`` and
    ``rms`` of the coordinates of the data dimensions and of the variables the optimizer
    reads: ``data`` and, if present, ``weight``. The statistics are calculated in float64
    and do not depend on the dimension order, the dtype or the byte order.

    Parameters
    ----------
    data : xr.Dataset | xr.DataArray
        A dataset with a ``data`` variable, or the data itself.

    Returns
    -------
    dict[str, Any]
        The summary.
    """
    dataset = data.to_dataset(name="data") if isinstance(data, xr.DataArray) else data
    summary: dict[str, Any] = {
        "shape": {str(dim): int(size) for dim, size in dataset["data"].sizes.items()}
    }
    for name in [*dataset["data"].dims, "data", "weight"]:
        if name in dataset and np.issubdtype(dataset[name].dtype, np.number):
            summary[str(name)] = _statistics(dataset[name].to_numpy())
    return summary


def compare_data_summaries(
    recorded: dict[str, dict[str, Any]], current: dict[str, dict[str, Any]]
) -> dict[str, list[str]]:
    """Describe how the data summaries of several datasets differ.

    Parameters
    ----------
    recorded : dict[str, dict[str, Any]]
        The recorded summaries by dataset label.
    current : dict[str, dict[str, Any]]
        The current summaries by dataset label.

    Returns
    -------
    dict[str, list[str]]
        The differences of each dataset that differs, for example
        ``{"ta": ["time max 10 -> 8", "data rms 0.512 -> 0.514 (+0.39 %)"]}``. A dataset that
        is missing from one of the two counts as different. Empty if nothing differs.
    """
    differences = {}
    for label in [*recorded, *(label for label in current if label not in recorded)]:
        if label not in current:
            differences[label] = ["dataset missing"]
        elif label not in recorded:
            differences[label] = ["new dataset"]
        elif dataset_differences := _compare_dataset_summary(recorded[label], current[label]):
            differences[label] = dataset_differences
    return differences


def _compare_dataset_summary(recorded: dict[str, Any], current: dict[str, Any]) -> list[str]:
    differences = []
    if recorded["shape"] != current["shape"]:
        differences.append(
            f"shape {_format_shape(recorded['shape'])} -> {_format_shape(current['shape'])}"
        )
    names = [*recorded, *(name for name in current if name not in recorded)]
    for name in (name for name in names if name != "shape"):
        if name not in current:
            differences.append(f"{name} removed")
        elif name not in recorded:
            differences.append(f"{name} added")
        else:
            tolerance = DATA_SUMMARY_RTOL * max(
                abs(recorded[name]["rms"]), abs(current[name]["rms"])
            )
            differences.extend(
                _describe_difference(
                    name, statistic, recorded[name][statistic], current[name][statistic]
                )
                for statistic in STATISTICS
                if not _equal(recorded[name][statistic], current[name][statistic], tolerance)
            )
    return differences


def _statistics(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "rms": float(np.sqrt(np.mean(values**2))),
    }


def _equal(recorded: float, current: float, tolerance: float) -> bool:
    return (
        recorded == current
        or abs(recorded - current) <= tolerance
        or (np.isnan(recorded) and np.isnan(current))
    )


def _describe_difference(name: str, statistic: str, recorded: float, current: float) -> str:
    description = f"{name} {statistic} {recorded:.6g} -> {current:.6g}"
    if recorded != 0 and np.isfinite(recorded) and np.isfinite(current):
        description += f" ({(current - recorded) / abs(recorded) * 100:+.2g} %)"
    return description


def _format_shape(shape: dict[str, int]) -> str:
    return " x ".join(f"{dim} {size}" for dim, size in shape.items())
