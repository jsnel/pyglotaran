from __future__ import annotations

from collections import ChainMap
from pathlib import Path  # noqa: TC003
from typing import TYPE_CHECKING
from typing import Literal

import numpy as np
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import field_validator

from glotaran.io import load_dataset
from glotaran.model.errors import GlotaranUserError
from glotaran.model.experiment_model import ExperimentModel
from glotaran.model.parameter_penalties import EqualParameterPenalty  # noqa: TC001
from glotaran.optimization import Optimization
from glotaran.optimization.info import calculate_parameter_errors
from glotaran.project.library import ModelLibrary
from glotaran.utils.io import DatasetMapping
from glotaran.utils.io import load_datasets

if TYPE_CHECKING:
    from typing_extensions import Self

    from glotaran.parameter import Parameters
    from glotaran.project.result import Result
    from glotaran.typing.types import DatasetMappable


class Scheme(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    experiments: dict[str, ExperimentModel]
    library: ModelLibrary
    parameter_penalties: list[EqualParameterPenalty] = Field(default_factory=list)
    x_scale: float | Literal["jac"] | list[float] | dict[str, float] = 1.0
    compute_clp_standard_error: bool = False
    clp_standard_error_finite_difference_relative_step: float = 1e-6

    @field_validator("x_scale")
    @classmethod
    def validate_x_scale(
        cls,
        value: float | Literal["jac"] | list[float] | dict[str, float],
    ) -> float | Literal["jac"] | list[float] | dict[str, float]:
        if value == "jac":
            return value
        values = list(value.values()) if isinstance(value, dict) else value
        if np.any(~np.isfinite(values)) or np.any(np.asarray(values) <= 0):
            msg = "x_scale must contain finite positive values."
            raise ValueError(msg)
        return value

    @field_validator("clp_standard_error_finite_difference_relative_step")
    @classmethod
    def validate_clp_standard_error_step(cls, value: float) -> float:
        if not np.isfinite(value) or value <= 0:
            msg = "clp_standard_error_finite_difference_relative_step must be finite and positive."
            raise ValueError(msg)
        return value

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
        return cls(
            **(spec | {"experiments": experiments, "library": library, "source_path": source_path})
        )

    def _load_data(self, datasets: DatasetMapping) -> None:
        for experiment in self.experiments.values():
            for label, data_model in experiment.datasets.items():
                if label in datasets:
                    data_model.data = datasets[label]
                elif any(
                    hasattr(
                        self.library[element] if isinstance(element, str) else element,
                        "calculate_data",
                    )
                    for element in data_model.elements
                ):
                    # Generated from the parameters during optimization.
                    data_model.data = None
                else:
                    msg = f"Not data for data model '{label}' provided."
                    raise GlotaranUserError(msg)

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
        x_scale: float | Literal["jac"] | list[float] | dict[str, float] | None = None,
        compute_clp_standard_error: bool | None = None,
        clp_standard_error_finite_difference_relative_step: float | None = None,
        optimization_method: Literal[
            "TrustRegionReflection",
            "Dogbox",
            "Levenberg-Marquardt",
        ] = "TrustRegionReflection",
        add_svd: bool = True,
        dry_run: bool = False,
        verbose: bool = True,
        raise_exception: bool = False,
    ) -> Result:
        # Prevent circular import error
        from glotaran.project.result import Result  # noqa: PLC0415

        self._load_data(load_datasets(datasets))
        optimization = Optimization(
            models=list(self.experiments.values()),
            parameters=parameters,
            library=self.library,
            verbose=verbose,
            raise_exception=raise_exception,
            maximum_number_function_evaluations=maximum_number_function_evaluations,
            add_svd=add_svd,
            ftol=ftol,
            gtol=gtol,
            xtol=xtol,
            x_scale=self.x_scale if x_scale is None else self.validate_x_scale(x_scale),
            parameter_penalties=self.parameter_penalties,
            compute_clp_standard_error=(
                self.compute_clp_standard_error
                if compute_clp_standard_error is None
                else compute_clp_standard_error
            ),
            clp_standard_error_finite_difference_relative_step=(
                self.clp_standard_error_finite_difference_relative_step
                if clp_standard_error_finite_difference_relative_step is None
                else self.validate_clp_standard_error_step(
                    clp_standard_error_finite_difference_relative_step
                )
            ),
            optimization_method=optimization_method,
        )
        optimized_parameters, optimized_data, optimization_info = (
            optimization.dry_run() if dry_run else optimization.run()
        )
        calculate_parameter_errors(
            optimization_info=optimization_info, parameters=optimized_parameters
        )
        return Result(
            optimization_results=optimized_data,
            scheme=self,
            optimization_info=optimization_info,
            initial_parameters=parameters,
            optimized_parameters=optimized_parameters,
        )
