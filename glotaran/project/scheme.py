"""The module for :class:``Scheme``."""

from __future__ import annotations
from typing import Literal

from pydantic import BaseModel
from pydantic import Extra

from glotaran.io import load_dataset
from glotaran.model import ExperimentModel
from glotaran.model import Model
from glotaran.optimization import Optimization
from glotaran.parameter import Parameters
from glotaran.project.library import ModelLibrary
from glotaran.project.result import Result


class Scheme(BaseModel):
    class Config:
        """Config for pydantic.BaseModel."""

        arbitrary_types_allowed = True
        extra = Extra.forbid

    experiments: list[ExperimentModel]
    library: dict[str, Model.get_annotated_type()]

    @classmethod
    def from_dict(cls, spec: dict):
        library = ModelLibrary.parse_obj(
            {label: m | {"label": label} for label, m in spec["library"].items()}
        ).__root__
        experiments = [ExperimentModel.from_dict(library, e) for e in spec["experiments"]]
        for e in experiments:
            for d in e.datasets.values():
                if isinstance(d.data, str):
                    d.data = load_dataset(d.data)
        return cls(experiments=experiments, library=library)

    def optimize(
        self,
        parameters: Parameters,
        verbose: bool = True,
        raise_exception: bool = False,
        maximum_number_function_evaluations: int | None = None,
        add_svd: bool = True,
        ftol: float = 1e-8,
        gtol: float = 1e-8,
        xtol: float = 1e-8,
        optimization_method: Literal[
            "TrustRegionReflection",
            "Dogbox",
            "Levenberg-Marquardt",
        ] = "TrustRegionReflection",
    ) -> Result:
        optimized_parameters, optimized_data, optimization_result = Optimization(
            self.experiments,
            parameters,
            library=self.library,
            verbose=verbose,
            raise_exception=raise_exception,
            maximum_number_function_evaluations=maximum_number_function_evaluations,
            add_svd=add_svd,
            ftol=ftol,
            gtol=gtol,
            xtol=xtol,
            optimization_method=optimization_method,
        ).run()
        return Result(
            data=optimized_data,
            experiments=self.experiments,
            optimization=optimization_result,
            parameters_intitial=parameters,
            parameters_optimized=optimized_parameters,
        )
