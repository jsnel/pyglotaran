from __future__ import annotations

from collections import ChainMap
from pathlib import Path  # noqa: TC003
from typing import TYPE_CHECKING
from typing import Any
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
from glotaran.utils.io import load_datasets

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Self

    import xarray as xr

    from glotaran.parameter import Parameters
    from glotaran.project.result import Result
    from glotaran.typing.types import DatasetMappable


class Scheme(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    experiments: dict[str, ExperimentModel]
    library: ModelLibrary
    parameter_penalties: list[EqualParameterPenalty] = Field(default_factory=list)
    x_scale: float | Literal["jac"] | list[float] | dict[str, float] = 1.0
    add_svd: bool = Field(
        default=True,
        description="Compute optional result-time singular value decompositions.",
    )
    compute_clp_standard_error: bool = False
    clp_standard_error_finite_difference_relative_step: float = 1e-6

    @field_validator("x_scale")
    @classmethod
    def validate_x_scale(
        cls,
        value: float | Literal["jac"] | list[float] | dict[str, float],
    ) -> float | Literal["jac"] | list[float] | dict[str, float]:
        if isinstance(value, str):
            if value == "jac":
                return value
            msg = "x_scale must be 'jac' or contain finite positive values."
            raise ValueError(msg)
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

    def _load_data(self, datasets: Mapping[str, xr.Dataset]) -> None:
        from glotaran.builtin.elements.ctilde import CtildeElement  # noqa: PLC0415

        for experiment in self.experiments.values():
            for label, data_model in experiment.datasets.items():
                if label in datasets:
                    data_model.data = datasets[label]
                elif any(
                    hasattr(
                        self.library[element] if isinstance(element, str) else element,
                        "calculate_data",
                    )
                    or isinstance(
                        self.library[element] if isinstance(element, str) else element,
                        CtildeElement,
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
        ftol: float | None = 1e-8,
        gtol: float | None = 1e-8,
        xtol: float | None = 1e-8,
        x_scale: float | Literal["jac"] | list[float] | dict[str, float] | None = None,
        add_svd: bool | None = None,
        compute_clp_standard_error: bool | None = None,
        clp_standard_error_finite_difference_relative_step: float | None = None,
        optimization_method: Literal[
            "TrustRegionReflection",
            "Dogbox",
            "Levenberg-Marquardt",
        ] = "TrustRegionReflection",
        dry_run: bool = False,
        verbose: bool = True,
        raise_exception: bool = False,
    ) -> Result:
        scheme, optimization = self._prepare_optimization(
            parameters,
            datasets,
            verbose=verbose,
            raise_exception=raise_exception,
            maximum_number_function_evaluations=maximum_number_function_evaluations,
            ftol=ftol,
            gtol=gtol,
            xtol=xtol,
            x_scale=x_scale,
            add_svd=add_svd,
            compute_clp_standard_error=compute_clp_standard_error,
            clp_standard_error_finite_difference_relative_step=(
                clp_standard_error_finite_difference_relative_step
            ),
            optimization_method=optimization_method,
        )
        return scheme._run_optimization(optimization, parameters, dry_run=dry_run)  # noqa: SLF001

    def _prepare_optimization(
        self,
        parameters: Parameters,
        datasets: DatasetMappable,
        **optimization_kwargs: Any,  # noqa: ANN401
    ) -> tuple[Scheme, Optimization]:
        """Create the optimization of a copy of this scheme with copies of the datasets.

        The result holds copies, so that a later change of this scheme, the parameters or the
        caller's data does not change it.

        Parameters
        ----------
        parameters : Parameters
            The initial parameters.
        datasets : DatasetMappable
            The datasets.
        **optimization_kwargs : Any
            Keyword arguments of :class:`Optimization`.

        Returns
        -------
        tuple[Scheme, Optimization]
            The copy of the scheme with the data loaded, and its optimization.
        """
        scheme = self.model_copy(deep=True)
        scheme._load_data(  # noqa: SLF001
            {label: data.copy(deep=True) for label, data in load_datasets(datasets).items()}
        )
        # Project.optimize enters here directly, so defaults must be applied in
        # the common path rather than only in Scheme.optimize.
        x_scale = optimization_kwargs.get("x_scale")
        optimization_kwargs["x_scale"] = (
            scheme.x_scale if x_scale is None else self.validate_x_scale(x_scale)
        )
        optimization_kwargs["parameter_penalties"] = scheme.parameter_penalties
        # Project.optimize also enters through this path. Persist the effective
        # value in optimizer settings so records and exports replay overrides.
        add_svd = optimization_kwargs.get("add_svd")
        optimization_kwargs["add_svd"] = scheme.add_svd if add_svd is None else add_svd
        # Project.optimize enters here directly too. Resolve None as the scheme
        # default and persist explicit per-fit overrides in optimizer settings.
        clp_enabled = optimization_kwargs.get("compute_clp_standard_error")
        optimization_kwargs["compute_clp_standard_error"] = (
            scheme.compute_clp_standard_error if clp_enabled is None else clp_enabled
        )
        step = optimization_kwargs.get("clp_standard_error_finite_difference_relative_step")
        optimization_kwargs["clp_standard_error_finite_difference_relative_step"] = (
            scheme.clp_standard_error_finite_difference_relative_step
            if step is None
            else self.validate_clp_standard_error_step(step)
        )
        optimization = Optimization(
            models=list(scheme.experiments.values()),
            parameters=parameters,
            library=scheme.library,
            **optimization_kwargs,
        )
        return scheme, optimization

    def _run_optimization(
        self, optimization: Optimization, initial_parameters: Parameters, *, dry_run: bool = False
    ) -> Result:
        """Run an optimization created by :meth:`_prepare_optimization` and create its result.

        Parameters
        ----------
        optimization : Optimization
            The optimization of this scheme.
        initial_parameters : Parameters
            The initial parameters, of which the result stores a copy.
        dry_run : bool
            Evaluate the model once at the initial parameters instead of optimizing.

        Returns
        -------
        Result
        """
        # Prevent circular import error
        from glotaran.project.result import Result  # noqa: PLC0415

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
            initial_parameters=initial_parameters.copy(),
            optimized_parameters=optimized_parameters,
            optimizer_settings=optimization.settings,
        )
