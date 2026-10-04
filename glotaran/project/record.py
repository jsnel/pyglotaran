"""Records of fits: every fit attempt is written to a numbered record folder.

A record holds what is needed to recompute the results of a fit from its input data, and no
data: the scheme, the initial and optimized parameters, the optimizer settings, a summary of
the input data and of the fit, and the cost of every function evaluation.

A project is a folder whose root contains a ``project.gta`` file. Records go to ``results``
in the project root, or in the working directory outside a project.
"""

from __future__ import annotations

import functools
import os
import platform
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any
from warnings import warn

import numpy as np
import pandas as pd
from ruamel.yaml import YAML

from glotaran.builtin.io.yml.utils import write_dict
from glotaran.io import save_parameters
from glotaran.io import save_scheme
from glotaran.project.data_summary import summarize_data

if TYPE_CHECKING:
    from glotaran.optimization import Optimization
    from glotaran.optimization import OptimizerSettings
    from glotaran.parameter import Parameters
    from glotaran.project.result import Result
    from glotaran.project.scheme import Scheme
    from glotaran.typing.types import StrOrPath

RECORD_SCHEMA_VERSION = 1
PROJECT_FILE = "project.gta"
RECORD_FILE = "record.yml"
EXPORT_FILE = "export.yml"


@dataclass
class RecordSettings:
    """Session settings for recording fits, for example ``record_settings.record = False``.

    ``None`` means not set. Arguments of ``Scheme.optimize`` take precedence over these
    settings, and these settings over the environment variables ``GLOTARAN_RECORD``,
    ``GLOTARAN_RESULTS_FOLDER`` and ``GLOTARAN_RECORD_PARAMETER_HISTORY``. The exception is
    ``GLOTARAN_RECORD=0``, which switches recording off everywhere.
    """

    record: bool | None = None
    """Whether to record fits (default: yes)."""
    output_folder: StrOrPath | None = None
    """Base folder of the records (default: ``results`` in the project root or working
    directory)."""
    record_parameter_history: bool | None = None
    """Whether to record the parameter values of every function evaluation (default: no)."""
    source: str | None = None
    """Notebook or script to record as the source of fits, instead of the detected one."""


record_settings = RecordSettings()


@dataclass(frozen=True)
class RecordReference:
    """Location and identity of a record."""

    path: Path
    """Absolute path of the record folder."""
    id: str
    """Folder number of the record, for example ``"001"``."""
    uid: str
    """Unique id of the fit attempt."""


def find_project_root(start: StrOrPath | None = None) -> Path | None:
    """Find the root of the project that contains a folder.

    Parameters
    ----------
    start : StrOrPath | None
        The folder to start from, by default the working directory.

    Returns
    -------
    Path | None
        The nearest folder, ``start`` or one of its parents, that contains a ``project.gta``
        file, or ``None`` outside a project.
    """
    start = Path.cwd() if start is None else Path(start).resolve()
    return next(
        (folder for folder in (start, *start.parents) if (folder / PROJECT_FILE).is_file()), None
    )


def resolve_results_folder(output_folder: StrOrPath | None = None) -> Path:
    """Resolve the base folder of records to an absolute path.

    Parameters
    ----------
    output_folder : StrOrPath | None
        A folder given for one call.

    Returns
    -------
    Path
        The first that is set: ``output_folder``, ``record_settings.output_folder``, the
        environment variable ``GLOTARAN_RESULTS_FOLDER``, else ``results`` in the project root
        or the working directory.
    """
    folder = (
        output_folder or record_settings.output_folder or os.environ.get("GLOTARAN_RESULTS_FOLDER")
    )
    if folder:
        return Path(folder).resolve()
    return (find_project_root() or Path.cwd()) / "results"


def recording_enabled(*, record: bool | None, output_folder: StrOrPath | None) -> bool:
    """Determine whether a fit is recorded.

    Parameters
    ----------
    record : bool | None
        The ``record`` argument of the call.
    output_folder : StrOrPath | None
        The ``output_folder`` argument of the call, which switches recording on.

    Returns
    -------
    bool
        ``False`` if ``GLOTARAN_RECORD=0``, else the first that is set: ``record``,
        ``output_folder``, ``record_settings.record``, ``GLOTARAN_RECORD``, else ``True``.

    Raises
    ------
    ValueError
        If ``record=False`` is combined with an ``output_folder``.
    """
    if record is False and output_folder is not None:
        msg = "'record=False' cannot be combined with an 'output_folder'."
        raise ValueError(msg)
    environment_setting = _environment_flag("GLOTARAN_RECORD")
    if environment_setting is False:
        return False
    if record is not None:
        return record
    if output_folder is not None:
        return True
    return next(
        (
            setting
            for setting in (record_settings.record, environment_setting)
            if setting is not None
        ),
        True,
    )


def parameter_history_enabled(*, record_parameter_history: bool | None) -> bool:
    """Determine whether the parameter values of every function evaluation are recorded.

    Parameters
    ----------
    record_parameter_history : bool | None
        The ``record_parameter_history`` argument of the call.

    Returns
    -------
    bool
        The first that is set: ``record_parameter_history``,
        ``record_settings.record_parameter_history``, ``GLOTARAN_RECORD_PARAMETER_HISTORY``,
        else ``False``.
    """
    settings = (
        record_parameter_history,
        record_settings.record_parameter_history,
        _environment_flag("GLOTARAN_RECORD_PARAMETER_HISTORY"),
    )
    return next((setting for setting in settings if setting is not None), False)


def detect_source() -> str:
    """Detect the notebook or script that runs the fit.

    Returns
    -------
    str
        ``record_settings.source`` if set, else the path of the VS Code or Jupyter notebook,
        or of the script, that runs the fit, else ``"unknown"`` (for example under nbclient,
        papermill or plain IPython).
    """
    if record_settings.source:
        return record_settings.source
    ipython = sys.modules.get("IPython")
    shell = ipython.get_ipython() if ipython is not None else None
    if shell is not None:
        # Set by the VS Code Jupyter extension and by Jupyter Server.
        notebook = shell.user_ns.get("__vsc_ipynb_file__") or os.environ.get("JPY_SESSION_NAME")
        return str(notebook) if notebook else "unknown"
    script = getattr(sys.modules.get("__main__"), "__file__", None)
    return str(script) if script else "unknown"


@functools.cache
def environment_versions() -> dict[str, str]:
    """Versions of pyglotaran, Python, numpy, scipy, xarray and the installed glotaran plugins.

    Returns
    -------
    dict[str, str]
        Version by package name.
    """
    from glotaran import __version__  # noqa: PLC0415

    versions = {"pyglotaran": __version__, "python": platform.python_version()}
    versions |= {name: metadata.version(name) for name in ("numpy", "scipy", "xarray")}
    for distribution in metadata.distributions():
        name = distribution.metadata["Name"]
        if name != "pyglotaran" and any(
            entry_point.group.startswith("glotaran.plugins")
            for entry_point in distribution.entry_points
        ):
            versions[name] = distribution.version
    return versions


def format_record_id(number: int) -> str:
    """Format a record number as record id (``1`` -> ``"001"``, ``1000`` -> ``"1000"``)."""
    return f"{number:03d}"


def record_folders(results_folder: Path) -> list[Path]:
    """List the folders named by a number in a base folder, sorted by number."""
    if not results_folder.is_dir():
        return []
    folders = [
        folder
        for folder in results_folder.iterdir()
        if folder.is_dir() and folder.name.isascii() and folder.name.isdecimal()
    ]
    return sorted(folders, key=lambda folder: int(folder.name))


def allocate_record_folder(results_folder: Path) -> Path:
    """Create the next record folder in a base folder.

    The number is the largest existing folder number plus one. The folder is created with an
    exclusive ``mkdir``, so concurrent fits get different numbers.

    Parameters
    ----------
    results_folder : Path
        The base folder.

    Returns
    -------
    Path
        The created folder.
    """
    results_folder.mkdir(parents=True, exist_ok=True)
    folders = record_folders(results_folder)
    number = int(folders[-1].name) + 1 if folders else 1
    while True:
        folder = results_folder / format_record_id(number)
        try:
            folder.mkdir()
        except FileExistsError:
            number += 1
        else:
            return folder


def read_record(folder: StrOrPath) -> dict[str, Any]:
    """Read the metadata of a record (``record.yml``) or an export (``export.yml``).

    Parameters
    ----------
    folder : StrOrPath
        The record or export folder.

    Returns
    -------
    dict[str, Any]
        The metadata. Readers ignore unknown fields and treat missing fields as empty.

    Raises
    ------
    ValueError
        If the metadata were written in a newer format than this version of pyglotaran reads.
    """
    folder = Path(folder)
    path = folder / RECORD_FILE if (folder / RECORD_FILE).is_file() else folder / EXPORT_FILE
    record = YAML(typ="safe").load(path.read_text(encoding="utf-8"))
    if record.get("schema_version", 1) > RECORD_SCHEMA_VERSION:
        msg = (
            f"'{path}' has schema version {record['schema_version']}, this version of "
            f"pyglotaran reads up to version {RECORD_SCHEMA_VERSION}. Update pyglotaran."
        )
        raise ValueError(msg)
    return record


def summarize_result(result: Result) -> dict[str, Any]:
    """Summarize the statistics of a successful fit for its record.

    Parameters
    ----------
    result : Result
        The result of the fit.

    Returns
    -------
    dict[str, Any]
        The summary.
    """
    info = result.optimization_info
    return {
        "cost": info.cost,
        "chi_square": info.chi_square,
        "reduced_chi_square": info.reduced_chi_square,
        "root_mean_square_error": info.root_mean_square_error,
        "degrees_of_freedom": info.degrees_of_freedom,
        "number_of_function_evaluations": info.number_of_function_evaluations,
        "termination_reason": info.termination_reason,
        "converged": info.converged,
        "free_parameter_labels": list(info.free_parameter_labels),
        "datasets": {
            label: {
                "root_mean_square_error": optimization_result.meta.root_mean_square_error,
                "weighted_root_mean_square_error": (
                    optimization_result.meta.weighted_root_mean_square_error
                ),
            }
            for label, optimization_result in result.optimization_results.items()
        },
    }


class RecordWriter:
    """Writes the record of one fit: started before the fit and finished after it.

    Write failures give a warning and never replace the outcome of the fit.
    """

    def __init__(self, folder: Path, record: dict[str, Any], *, parameter_history: bool) -> None:
        """Initialize a writer for a started record, use :meth:`start` instead."""
        self._folder = folder
        self._record = record
        self._parameter_history = parameter_history

    @classmethod
    def start(
        cls,
        results_folder: Path,
        *,
        scheme: Scheme,
        initial_parameters: Parameters,
        optimizer_settings: OptimizerSettings,
        name: str | None,
        parameter_history: bool,
    ) -> RecordWriter | None:
        """Create the record folder and write what is known before the fit.

        ``record.yml`` gets the status ``running``, which a fit that never returns to Python
        (for example after a kernel death) keeps.

        Parameters
        ----------
        results_folder : Path
            The base folder of the record.
        scheme : Scheme
            The scheme of the fit, with the input datasets loaded.
        initial_parameters : Parameters
            The initial parameters.
        optimizer_settings : OptimizerSettings
            The optimizer settings.
        name : str | None
            An optional label of the fit.
        parameter_history : bool
            Whether the parameter history is recorded.

        Returns
        -------
        RecordWriter | None
            The writer, or ``None`` if the record could not be created.
        """
        datasets = [
            (label, data_model.data)
            for experiment in scheme.experiments.values()
            for label, data_model in experiment.datasets.items()
        ]
        if len(datasets) > len(dict(datasets)):
            warn(
                "Dataset labels repeat across experiments, so the per-dataset entries of the "
                "record are ambiguous.",
                stacklevel=3,
            )
        try:
            folder = allocate_record_folder(results_folder)
            record = {
                "schema_version": RECORD_SCHEMA_VERSION,
                "id": folder.name,
                "uid": uuid.uuid4().hex,
                "created": datetime.now().astimezone().isoformat(timespec="seconds"),
                "status": "running",
                "error": None,
                "name": name,
                "source": _relative_path(detect_source(), folder),
                "scheme_source": (
                    None
                    if scheme.source_path is None
                    else _relative_path(scheme.source_path, folder)
                ),
                "parent": None,
                "parent_data_changed": [],
                "optimizer": optimizer_settings.model_dump(),
                "summary": {},
                "data": {label: summarize_data(dataset) for label, dataset in datasets},
                "environment": environment_versions(),
            }
            save_scheme(scheme, folder / "scheme.yml", update_source_path=False)
            save_parameters(
                initial_parameters, folder / "initial_parameters.csv", update_source_path=False
            )
            _write_record_file(folder, record)
        except Exception as error:  # noqa: BLE001
            warn(
                f"The fit is not recorded, writing to '{results_folder}' failed: {error}",
                stacklevel=3,
            )
            return None
        return cls(folder, record, parameter_history=parameter_history)

    @property
    def reference(self) -> RecordReference:
        """Location and identity of the record."""
        return RecordReference(path=self._folder, id=self._record["id"], uid=self._record["uid"])

    def finish(
        self,
        optimization: Optimization,
        result: Result | None = None,
        error: BaseException | None = None,
    ) -> RecordReference:
        """Write what is known after the fit and replace ``record.yml``.

        Parameters
        ----------
        optimization : Optimization
            The optimization of the fit.
        result : Result | None
            The result, ``None`` if the fit raised ``error``.
        error : BaseException | None
            The exception the fit raised.

        Returns
        -------
        RecordReference
            Location and identity of the record.
        """
        error = error or optimization.error
        record = self._record
        try:
            if error is None and result is not None:
                record["status"] = "success"
                record["summary"] = summarize_result(result)
            else:
                record["status"] = (
                    "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
                )
                record["error"] = {"type": type(error).__name__, "message": str(error)}
                record["summary"] = {
                    "number_of_function_evaluations": optimization.number_of_evaluations,
                    "cost": None if result is None else result.optimization_info.cost,
                }
            if result is not None:
                optimized_parameters: Parameters | None = result.optimized_parameters
            elif optimization.number_of_evaluations > 0:
                optimized_parameters = optimization.parameters
            else:
                optimized_parameters = None
            if optimized_parameters is not None:
                save_parameters(
                    optimized_parameters,
                    self._folder / "optimized_parameters.csv",
                    update_source_path=False,
                )
            cost_history = optimization.cost_history
            pd.DataFrame(
                {"cost": cost_history},
                index=pd.RangeIndex(1, len(cost_history) + 1, name="evaluation"),
            ).to_csv(self._folder / "cost_history.csv")
            if self._parameter_history:
                optimization.parameter_history.to_csv(self._folder / "parameter_history.csv")
            _write_record_file(self._folder, record)
        except Exception as write_error:  # noqa: BLE001
            warn(
                f"Finishing the record '{self._folder}' failed: {write_error}",
                stacklevel=3,
            )
        return self.reference


def _environment_flag(name: str) -> bool | None:
    value = os.environ.get(name, "").strip().lower()
    if value == "":
        return None
    return value not in {"0", "false", "no", "off"}


def _relative_path(path: StrOrPath, start: Path) -> str:
    """Return ``path`` relative to ``start`` if the file exists and that is possible."""
    if not Path(path).is_file():
        return str(path)
    try:
        return Path(os.path.relpath(Path(path).resolve(), start)).as_posix()
    except ValueError:  # For example another drive on Windows
        return Path(path).resolve().as_posix()


def _write_record_file(folder: Path, record: dict[str, Any]) -> None:
    """Write ``record.yml`` atomically, through a temporary file."""
    temporary_file = folder / f"{RECORD_FILE}.tmp"
    write_dict(to_builtin_types(record), temporary_file)
    temporary_file.replace(folder / RECORD_FILE)


def to_builtin_types(value: Any) -> Any:  # noqa: ANN401
    """Convert numpy scalars in nested dicts and lists to Python types, for YAML."""
    if isinstance(value, dict):
        return {key: to_builtin_types(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [to_builtin_types(item) for item in value]
    return value.item() if isinstance(value, np.generic) else value
