from __future__ import annotations

from collections import ChainMap
from pathlib import Path  # noqa: TC003
from typing import TYPE_CHECKING
from typing import Literal

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

from glotaran.io import load_dataset
from glotaran.model.errors import GlotaranUserError
from glotaran.model.experiment_model import ExperimentModel
from glotaran.optimization import Optimization
from glotaran.optimization import OptimizerSettings
from glotaran.optimization.info import calculate_parameter_errors
from glotaran.project.library import ModelLibrary
from glotaran.project.record import RecordReference
from glotaran.project.record import RecordWriter
from glotaran.project.record import find_record
from glotaran.project.record import parameter_history_enabled
from glotaran.project.record import recording_enabled
from glotaran.project.record import resolve_results_folder
from glotaran.utils.io import load_datasets

if TYPE_CHECKING:
    from collections.abc import Mapping

    import xarray as xr
    from typing_extensions import Self

    from glotaran.parameter import Parameters
    from glotaran.project.result import Result
    from glotaran.typing.types import DatasetMappable
    from glotaran.typing.types import StrOrPath


class Scheme(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    experiments: dict[str, ExperimentModel]
    library: ModelLibrary
    source_path: Path | None = Field(
        default=None,
        description="Path to the source file from which this scheme was loaded.",
        exclude=True,
    )

    @classmethod
    def from_dict(cls, spec: dict, source_path: Path | None = None) -> Self:
        library = ModelLibrary.from_dict(spec["library"])
        experiments = {
            k: ExperimentModel.from_dict(library, e) for k, e in spec["experiments"].items()
        }
        for e in experiments.values():
            for d in e.datasets.values():
                if isinstance(d.data, str):
                    d.data = load_dataset(d.data)
        return cls(experiments=experiments, library=library, source_path=source_path)

    def _load_data(self, datasets: Mapping[str, xr.Dataset]) -> None:
        try:
            for experiment in self.experiments.values():
                for label, data_model in experiment.datasets.items():
                    data_model.data = datasets[label]
        except KeyError as e:
            msg = f"Not data for data model '{label}' provided."
            raise GlotaranUserError(msg) from e

    @property
    def dataset_paths(self) -> dict[str, str]:
        """Paths to all the datasets."""
        return dict(
            ChainMap(*(experiment.dataset_paths for experiment in self.experiments.values()))
        )

    def optimize(
        self,
        parameters: Parameters,
        datasets: DatasetMappable,
        *,
        maximum_number_function_evaluations: int | None = None,
        ftol: float = 1e-8,
        gtol: float = 1e-8,
        xtol: float = 1e-8,
        optimization_method: Literal[
            "TrustRegionReflection",
            "Dogbox",
            "Levenberg-Marquardt",
        ] = "TrustRegionReflection",
        add_svd: bool = True,
        dry_run: bool = False,
        verbose: bool = True,
        raise_exception: bool = False,
        record: bool | None = None,
        output_folder: StrOrPath | None = None,
        record_parameter_history: bool | None = None,
        name: str | None = None,
        parent: str | Path | RecordReference | None = None,
    ) -> Result:
        """Fit the scheme to the datasets.

        Every fit is recorded in a numbered folder (``results/001``, ``results/002``, ...),
        with what is needed to recompute its results from the input data, see
        :mod:`glotaran.project.record`. A dry run is not recorded.

        Parameters
        ----------
        parameters : Parameters
            The initial parameters.
        datasets : DatasetMappable
            The datasets by label, as datasets or paths.
        maximum_number_function_evaluations : int | None
            The maximum number of function evaluations, by default unlimited.
        ftol : float
            Tolerance for termination by the change of the cost.
        gtol : float
            Tolerance for termination by the norm of the gradient.
        xtol : float
            Tolerance for termination by the change of the parameters.
        optimization_method : Literal["TrustRegionReflection", "Dogbox", "Levenberg-Marquardt"]
            The optimization method.
        add_svd : bool
            Unused.
        dry_run : bool
            Evaluate the model once at the initial parameters instead of fitting.
        verbose : bool
            Print the progress of the optimizer and the record folder.
        raise_exception : bool
            Raise an exception of the optimizer instead of warning and returning the result.
        record : bool | None
            Whether to record the fit. By default as set in
            :data:`~glotaran.project.record.record_settings`, else recorded.
        output_folder : StrOrPath | None
            Base folder of the record, which also switches recording on. By default
            ``results`` in the project root (the folder with ``project.gta``) or the working
            directory.
        record_parameter_history : bool | None
            Whether to record the parameter values of every function evaluation.
        name : str | None
            A label for the record.
        parent : str | Path | RecordReference | None
            The record this fit continues from: a record id (``"017"``) or uid in the base
            folder of the record, a record path or ``Result.record``. By default the record of
            the fit that produced ``parameters``, if they are the optimized parameters of a
            recorded fit in this session (or a copy or edit of them).

        Returns
        -------
        Result
            The result of the fit. ``Result.record`` refers to its record.

        Raises
        ------
        FileNotFoundError
            If the ``parent`` record does not exist.
        """
        # Prevent circular import error
        from glotaran.project.result import Result  # noqa: PLC0415

        results_folder = (
            resolve_results_folder(output_folder)
            if recording_enabled(record=record, output_folder=output_folder) and not dry_run
            else None
        )
        parent_folder = (
            None
            if results_folder is None or parent is None
            else find_record(parent, results_folder)
        )
        # The result gets its own copies of the scheme, parameters and data, so later edits
        # of the inputs or later fits do not change it.
        scheme = self.model_copy(deep=True)
        scheme._load_data(  # noqa: SLF001
            {label: dataset.copy(deep=True) for label, dataset in load_datasets(datasets).items()}
        )
        optimizer_settings = OptimizerSettings(
            optimization_method=optimization_method,
            ftol=ftol,
            gtol=gtol,
            xtol=xtol,
            maximum_number_function_evaluations=maximum_number_function_evaluations,
        )
        optimization = Optimization(
            models=list(scheme.experiments.values()),
            parameters=parameters,
            library=scheme.library,
            verbose=verbose,
            raise_exception=raise_exception,
            add_svd=add_svd,
            record_parameter_history=parameter_history_enabled(
                record_parameter_history=record_parameter_history
            ),
            **optimizer_settings.model_dump(),
        )
        writer = (
            None
            if results_folder is None
            else RecordWriter.start(
                results_folder,
                scheme=scheme,
                initial_parameters=parameters,
                optimizer_settings=optimizer_settings,
                name=name,
                parameter_history=parameter_history_enabled(
                    record_parameter_history=record_parameter_history
                ),
                parent_folder=parent_folder,
            )
        )
        try:
            optimized_parameters, optimized_data, optimization_info = (
                optimization.dry_run() if dry_run else optimization.run()
            )
            calculate_parameter_errors(
                optimization_info=optimization_info, parameters=optimized_parameters
            )
            result = Result(
                optimization_results=optimized_data,
                scheme=scheme,
                optimization_info=optimization_info,
                optimizer_settings=optimizer_settings,
                initial_parameters=parameters.copy(),
                optimized_parameters=optimized_parameters,
            )
        except BaseException as error:
            if writer is not None:
                writer.finish(optimization, error=error)
            raise
        if writer is not None:
            result.record = writer.finish(optimization, result=result)
            result.optimized_parameters.record_uid = result.record.uid
            if verbose:
                print(f"Recorded the fit in '{result.record.path}'.")  # noqa: T201
        return result
