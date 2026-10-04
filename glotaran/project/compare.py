"""List the records of fits and compare fits."""

from __future__ import annotations

import difflib
from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Any

import pandas as pd

from glotaran.builtin.io.yml.utils import write_dict
from glotaran.io import load_parameters
from glotaran.io import load_scheme
from glotaran.project.data_summary import compare_data_summaries
from glotaran.project.data_summary import summarize_data
from glotaran.project.record import find_record
from glotaran.project.record import read_record
from glotaran.project.record import record_folders
from glotaran.project.record import resolve_results_folder
from glotaran.project.record import summarize_result
from glotaran.project.record import try_read_record
from glotaran.project.result import Result
from glotaran.utils.helpers import nan_or_equal

if TYPE_CHECKING:
    from pathlib import Path

    from glotaran.parameter import Parameters
    from glotaran.project.record import RecordReference
    from glotaran.project.scheme import Scheme
    from glotaran.typing.types import StrOrPath

PARAMETER_FIELDS = (
    "value",
    "standard_error",
    "expression",
    "minimum",
    "maximum",
    "non_negative",
    "vary",
)
LIST_COLUMNS = ["id", "created", "source", "name", "scheme_source", "status", "parent"]
LIST_SUMMARY_COLUMNS = ["cost", "root_mean_square_error", "number_of_function_evaluations"]
SUMMARY_STATISTICS = (
    "cost",
    "chi_square",
    "reduced_chi_square",
    "root_mean_square_error",
    "degrees_of_freedom",
    "number_of_function_evaluations",
    "converged",
)


def list_results(
    folder: StrOrPath | None = None,
    *,
    source: str | None = None,
    scheme_source: str | None = None,
    name: str | None = None,
    status: str | None = None,
) -> pd.DataFrame:
    """List the records of fits in a base folder.

    Only numbered folders with a ``record.yml`` are listed, so other output in the folder,
    such as that of a v0.7 project, is ignored. A damaged record is skipped with a warning.

    Parameters
    ----------
    folder : StrOrPath | None
        The base folder, by default the one ``Scheme.optimize`` records to.
    source : str | None
        Only list records whose notebook or script contains this text.
    scheme_source : str | None
        Only list records whose scheme file contains this text.
    name : str | None
        Only list records whose name contains this text.
    status : str | None
        Only list records with this status: ``success``, ``failed``, ``interrupted`` or
        ``incomplete`` (a fit that never returned, for example after a kernel death).

    Returns
    -------
    pd.DataFrame
        One row per record, indexed by record id and sorted by number. Per dataset, the shape
        and the RMS of the data show where the data was replaced. ``parent`` is marked
        ``(missing)`` if its record is gone and ``(replaced)`` if its id now belongs to
        another fit.
    """
    records = {
        record_folder.name: record
        for record_folder in record_folders(resolve_results_folder(folder))
        if (record := try_read_record(record_folder)) is not None
    }
    uids = {record_id: record.get("uid") for record_id, record in records.items()}
    filters = {"source": source, "scheme_source": scheme_source, "name": name}
    rows = []
    for record_id, record in records.items():
        row = {
            "id": record_id,
            "created": record.get("created"),
            "source": record.get("source"),
            "name": record.get("name"),
            "scheme_source": record.get("scheme_source"),
            "status": "incomplete" if record.get("status") == "running" else record.get("status"),
            "parent": _format_parent(record.get("parent"), uids),
        }
        if any(text is not None and text not in str(row[key]) for key, text in filters.items()):
            continue
        if status is not None and row["status"] != status:
            continue
        for label, summary in (record.get("data") or {}).items():
            row[f"{label} shape"] = " x ".join(str(size) for size in summary["shape"].values())
            row[f"{label} rms"] = summary.get("data", {}).get("rms")
        fit_summary = record.get("summary") or {}
        row["cost"] = fit_summary.get("cost")
        row["root_mean_square_error"] = fit_summary.get("root_mean_square_error")
        row["number_of_function_evaluations"] = fit_summary.get("number_of_function_evaluations")
        rows.append(row)
    data_columns = [
        column
        for column in dict.fromkeys(column for row in rows for column in row)
        if column not in LIST_COLUMNS + LIST_SUMMARY_COLUMNS
    ]
    return pd.DataFrame(
        rows, columns=[*LIST_COLUMNS, *data_columns, *LIST_SUMMARY_COLUMNS]
    ).set_index("id")


@dataclass
class ResultComparison:
    """Differences between two fits, see :func:`compare_results`."""

    summary: pd.DataFrame
    """Fit statistics of both fits, with the RMSE per dataset."""
    parameters: pd.DataFrame
    """The parameter fields that differ, one row per parameter and field."""
    data_differences: dict[str, list[str]]
    """Differences of the data summaries, by dataset."""
    scheme_diff: str
    """Unified diff of the schemes, empty if they are equal."""

    def _repr_markdown_(self) -> str:
        """Render the comparison as markdown, for example in a notebook."""
        data_differences = "\n".join(
            f"- {label}: {'; '.join(changes)}" for label, changes in self.data_differences.items()
        )
        return "\n\n".join(
            [
                "### Summary",
                self.summary.to_markdown(),
                "### Parameters",
                self.parameters.to_markdown(index=False) if len(self.parameters) else "Equal.",
                "### Data",
                data_differences or "Equal.",
                "### Scheme",
                f"```diff\n{self.scheme_diff}```" if self.scheme_diff else "Equal.",
            ]
        )

    def __str__(self) -> str:
        """Render the comparison as markdown."""
        return self._repr_markdown_()


def compare_results(
    first: str | Path | RecordReference | Result,
    second: str | Path | RecordReference | Result,
    *,
    folder: StrOrPath | None = None,
) -> ResultComparison:
    """Compare two fits: their statistics, parameters, data and schemes.

    Changes of weights, penalties, scales and constraints appear in the scheme diff or in the
    data differences. Whether a difference of the parameters comes from a change of the data
    or of the model is for you to judge.

    Parameters
    ----------
    first : str | Path | RecordReference | Result
        A record id or uid, the path of a record or an export, ``Result.record`` or a result.
    second : str | Path | RecordReference | Result
        The fit to compare with.
    folder : StrOrPath | None
        The base folder to look up record ids and uids in, by default the one
        ``Scheme.optimize`` records to.

    Returns
    -------
    ResultComparison
        The differences.
    """
    results_folder = resolve_results_folder(folder)
    fits = [_load_fit(fit, results_folder) for fit in (first, second)]
    names = [fit_name for fit_name, *_ in fits]
    if names[0] == names[1]:
        names = ["first", "second"]
    (_, parameters_1, record_1, scheme_1), (_, parameters_2, record_2, scheme_2) = fits
    return ResultComparison(
        summary=_compare_summaries(record_1, record_2, names),
        parameters=_compare_parameters(parameters_1, parameters_2, names),
        data_differences=compare_data_summaries(
            record_1.get("data") or {}, record_2.get("data") or {}
        ),
        scheme_diff="".join(
            difflib.unified_diff(
                scheme_1.splitlines(keepends=True),
                scheme_2.splitlines(keepends=True),
                fromfile=names[0],
                tofile=names[1],
            )
        ),
    )


def _load_fit(
    fit: str | Path | RecordReference | Result, results_folder: Path
) -> tuple[str, Parameters | None, dict[str, Any], str]:
    """Load the name, optimized parameters, record metadata and scheme text of a fit."""
    if isinstance(fit, Result):
        record: dict[str, Any] = {
            "summary": summarize_result(fit),
            "data": {label: summarize_data(data) for label, data in fit.input_data.items()},
        }
        name = fit.record.id if fit.record is not None else "result"
        return name, fit.optimized_parameters, record, _scheme_text(fit.scheme)
    folder = find_record(fit, results_folder)
    record = read_record(folder)
    parameters_file = folder / "optimized_parameters.csv"
    parameters = load_parameters(parameters_file) if parameters_file.is_file() else None
    name = record.get("id") or folder.name
    return name, parameters, record, _scheme_text(load_scheme(folder / "scheme.yml"))


def _scheme_text(scheme: Scheme) -> str:
    """Write a scheme as YAML in a normalized form, independent of the file it came from."""
    return write_dict(scheme.model_dump(exclude_unset=True, mode="json")) or ""


def _compare_summaries(
    record_1: dict[str, Any], record_2: dict[str, Any], names: list[str]
) -> pd.DataFrame:
    summaries = [record.get("summary") or {} for record in (record_1, record_2)]
    rows = {
        statistic: [summary.get(statistic) for summary in summaries]
        for statistic in SUMMARY_STATISTICS
    }
    labels = [*summaries[0].get("datasets", {}), *summaries[1].get("datasets", {})]
    for label in dict.fromkeys(labels):
        for statistic in ("root_mean_square_error", "weighted_root_mean_square_error"):
            rows[f"{label} {statistic}"] = [
                summary.get("datasets", {}).get(label, {}).get(statistic) for summary in summaries
            ]
    return pd.DataFrame.from_dict(rows, orient="index", columns=names)


def _compare_parameters(
    parameters_1: Parameters | None, parameters_2: Parameters | None, names: list[str]
) -> pd.DataFrame:
    fields = [
        {parameter.label: parameter.model_dump() for parameter in parameters.all()}
        if parameters is not None
        else {}
        for parameters in (parameters_1, parameters_2)
    ]
    rows: list[dict[str, Any]] = []
    for label in dict.fromkeys([*fields[0], *fields[1]]):
        values_1, values_2 = fields[0].get(label, {}), fields[1].get(label, {})
        rows.extend(
            {
                "label": label,
                "field": field,
                names[0]: values_1.get(field),
                names[1]: values_2.get(field),
            }
            for field in PARAMETER_FIELDS
            if not nan_or_equal(values_1.get(field), values_2.get(field))
        )
    return pd.DataFrame(rows, columns=["label", "field", *names])


def _format_parent(parent: dict[str, Any] | None, uids: dict[str, str | None]) -> str:
    """Show a parent by id, marked if its record is missing or its id was reused."""
    if not parent:
        return ""
    parent_id = parent.get("id")
    if parent_id is None:
        return "unknown"
    if parent_id not in uids:
        return f"{parent_id} (missing)"
    if uids[parent_id] != parent.get("uid"):
        return f"{parent_id} (replaced)"
    return str(parent_id)
