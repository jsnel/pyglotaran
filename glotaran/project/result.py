from pathlib import Path
from typing import Any
from typing import Literal

import xarray as xr
from pydantic import BaseModel
from pydantic import Extra

from glotaran.builtin.io.yml.utils import write_dict
from glotaran.io import save_dataset
from glotaran.io import save_parameters
from glotaran.model import ExperimentModel
from glotaran.model import GlotaranUserError
from glotaran.optimization import OptimizationResult
from glotaran.parameter import Parameters


class SavingOptions(BaseModel):
    """A collection of options for result saving."""

    data_filter: list[str] | None = None
    data_format: Literal["nc"] = "nc"
    parameter_format: Literal["csv"] = "csv"


SAVING_OPTIONS_DEFAULT = SavingOptions()


class Result(BaseModel):
    class Config:
        """Config for pydantic.BaseModel."""

        arbitrary_types_allowed = True
        extra = Extra.forbid

    data: dict[str, xr.Dataset]
    experiments: dict[str, ExperimentModel]
    optimization: OptimizationResult
    parameters_intitial: Parameters
    parameters_optimized: Parameters

    def save(self, path: Path, options: SavingOptions = SAVING_OPTIONS_DEFAULT):
        if path.exists() and path.is_file():
            raise GlotaranUserError("Save folder must be a path.")
        path.mkdir()
        result_dict: dict[str, Any] = {"data": {}, "experiments": {}}

        # TODO: Save scheme or experiments
        #  experiment_folder = path / "experiments"
        #  experiment_folder.mkdir()
        #  for label, experiment in self.experiments.items():
        #      experiment_path = experiment_folder / f"{label}.yml"
        #      result_dict["experiments"][label] = experiment_path
        #      write_dict(experiment.dict(), experiment_path)

        data_path = path / "data"
        data_path.mkdir()
        for label, data in self.data.items():
            dataset_path = data_path / f"{label}.{options.data_format}"
            result_dict["data"][label] = str(dataset_path)
            if options.data_filter is not None:
                data = data[options.data_filter]
            save_dataset(data, dataset_path)

        optimization_history_path = path / "optimization_history.csv"
        result_dict["optimization_history"] = str(optimization_history_path)
        self.optimization.optimization_history.to_csv(optimization_history_path)

        parameters_initial_path = path / f"parameters_initial.{options.parameter_format}"
        result_dict["parameters_initial"] = str(parameters_initial_path)
        save_parameters(self.parameters_intitial, parameters_initial_path)

        parameters_optimized_path = path / f"parameters_optimized.{options.parameter_format}"
        result_dict["parameters_optimized"] = str(parameters_optimized_path)
        save_parameters(self.parameters_optimized, parameters_optimized_path)

        result_path = path / "glotaran_result.yml"
        write_dict(result_dict, result_path)
