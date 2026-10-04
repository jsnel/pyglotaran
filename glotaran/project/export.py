"""Export a result as a self-contained folder that can be published."""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any
from warnings import warn

from ruamel.yaml import YAML

from glotaran.builtin.io.yml.utils import write_dict
from glotaran.io import SAVING_OPTIONS_DEFAULT
from glotaran.project import record as record_module
from glotaran.project.data_summary import summarize_data
from glotaran.project.record import EXPORT_FILE
from glotaran.project.record import PROJECT_FILE
from glotaran.project.record import RECORD_SCHEMA_VERSION
from glotaran.project.record import detect_source
from glotaran.project.record import environment_versions
from glotaran.project.record import project_folder
from glotaran.project.record import summarize_result
from glotaran.project.record import to_builtin_types
from glotaran.project.record import try_read_record

if TYPE_CHECKING:
    from glotaran.io import SavingOptions
    from glotaran.project.result import Result
    from glotaran.typing.types import StrOrPath


def export(
    result: Result,
    name: StrOrPath = "last_result",
    *,
    input_data: bool = True,
    arrays: bool = True,
    source_files: bool = True,
    overwrite: bool = False,
) -> Path:
    """Export a result as a self-contained folder.

    By default the folder contains everything except the notebook or script:

    - ``export.yml``: the metadata of the fit, as in its record, with the notebook or script
      and the scheme file named by file name only;
    - the scheme and the parameters;
    - the fitted input data, including a ``weight`` of the input datasets;
    - the result arrays, as ``Result.save`` writes them;
    - ``source_files/<dataset label>/``: copies of the files named by the ``source_path`` of
      the datasets that were loaded from a file, with their current contents;
    - ``project.gta``: a copy of the project description, inside a project.

    An export with input data loads with ``load_result``. An export without input data does
    not; it can be recomputed with ``recompute(path, datasets)``.

    Parameters
    ----------
    result : Result
        The result to export. To export an older fit, recompute it first.
    name : StrOrPath
        The export folder. A relative name is resolved in ``exports`` in the project root (the
        folder with ``project.gta``) or the working directory.
    input_data : bool
        Whether to include the input data.
    arrays : bool
        Whether to include the result arrays.
    source_files : bool
        Whether to include the source files of the datasets.
    overwrite : bool
        Whether to replace an existing export of the same name.

    Returns
    -------
    Path
        The export folder.

    Raises
    ------
    FileExistsError
        If the folder exists and ``overwrite`` is ``False``, or if the folder is not an export.
    """
    folder = Path(name) if Path(name).is_absolute() else project_folder() / "exports" / name
    if folder.exists():
        if not overwrite:
            msg = f"'{folder}' exists. Pass 'overwrite=True' to replace it."
            raise FileExistsError(msg)
        if not (folder / EXPORT_FILE).is_file():
            msg = f"'{folder}' is not an export, so it is not replaced."
            raise FileExistsError(msg)
        shutil.rmtree(folder)

    if changed_parameters := result.parameters_changed_since_fit():
        warn(
            "The optimized parameters were changed after the fit, the export contains the "
            f"changed values of: {', '.join(changed_parameters)}",
            stacklevel=2,
        )
    metadata = _fit_metadata(result)
    saving_options: SavingOptions = SAVING_OPTIONS_DEFAULT | {
        "data_filter": (
            set()
            if arrays
            else {"residuals", "fitted_data", "elements", "activations", "fit_decomposition"}
        )
    }
    # Saving points the scheme's source path to the saved scheme, so save a copy.
    result.model_copy(update={"scheme": result.scheme.model_copy()}).save(
        folder, saving_options=saving_options
    )
    if not input_data:
        _remove_input_data(folder)

    metadata |= {
        "schema_version": RECORD_SCHEMA_VERSION,
        "exported": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source": _file_name(metadata.get("source")),
        "scheme_source": _file_name(metadata.get("scheme_source")),
        "changed_parameters": changed_parameters,
    }
    if source_files:
        metadata |= _copy_source_files(result, folder)
    if (project_root := record_module.find_project_root()) is not None:
        shutil.copyfile(project_root / PROJECT_FILE, folder / PROJECT_FILE)
    write_dict(to_builtin_types(metadata), folder / EXPORT_FILE)
    return folder


def _fit_metadata(result: Result) -> dict[str, Any]:
    """Return the metadata of the fit from its record, or as far as the result has it."""
    if result.reconstruction is not None:
        return dict(result.reconstruction.original)
    if result.record is not None and (record := try_read_record(result.record.path)) is not None:
        return record
    return {
        "id": None,
        "uid": None,
        "source": detect_source(),
        "scheme_source": result.scheme.source_path,
        "optimizer": (
            None if result.optimizer_settings is None else result.optimizer_settings.model_dump()
        ),
        "summary": summarize_result(result),
        "data": {label: summarize_data(data) for label, data in result.input_data.items()},
        "environment": environment_versions(),
    }


def _remove_input_data(folder: Path) -> None:
    """Remove the input data files of a saved result and their entries in ``result.yml``."""
    result_file = folder / "result.yml"
    spec = YAML(typ="safe").load(result_file.read_text(encoding="utf-8"))
    for label, optimization_result in spec["optimization_results"].items():
        (folder / "optimization_results" / label / optimization_result["input_data"]).unlink()
        optimization_result["input_data"] = None
    write_dict(spec, result_file)


def _copy_source_files(result: Result, folder: Path) -> dict[str, dict[str, str]]:
    """Copy the files the datasets were loaded from, in their current state.

    Returns
    -------
    dict[str, dict[str, str]]
        The copied files and the files that do not exist, by dataset label.
    """
    copied, missing = {}, {}
    for label, data in result.input_data.items():
        # In-memory data get a made-up ``source_path``, loaded data also an ``io_plugin_name``.
        if "io_plugin_name" not in data.attrs or "source_path" not in data.attrs:
            continue
        source_file = Path(data.attrs["source_path"]).resolve()
        if not source_file.is_file():
            missing[label] = Path(data.attrs["source_path"]).as_posix()
            continue
        target = folder / "source_files" / label / source_file.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_file, target)
        copied[label] = target.relative_to(folder).as_posix()
    if missing:
        warn(
            "Source files of the input data do not exist and are not exported: "
            + ", ".join(f"{label} ({path})" for label, path in missing.items()),
            stacklevel=3,
        )
    return {"source_files": copied, "missing_source_files": missing}


def _file_name(path: StrOrPath | None) -> str | None:
    """Return the file name of a path, so an export contains no local paths."""
    return None if path is None else Path(path).name
