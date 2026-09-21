from __future__ import annotations

from collections import ChainMap
from typing import TYPE_CHECKING
from typing import Literal
from warnings import warn

import numpy as np
from scipy.optimize import OptimizeResult
from scipy.optimize import least_squares

from glotaran.model.errors import GlotaranModelIssues
from glotaran.model.errors import GlotaranUserError
from glotaran.model.item import resolve_item_parameters
from glotaran.model.parameter_penalties import EqualParameterPenalty
from glotaran.optimization.info import OptimizationInfo
from glotaran.optimization.objective import OptimizationObjective
from glotaran.optimization.objective import OptimizationResult
from glotaran.optimization.optimization_history import OptimizationHistory
from glotaran.parameter import ParameterHistory
from glotaran.parameter import Parameters
from glotaran.utils.tee import TeeContext

if TYPE_CHECKING:
    from glotaran.model.experiment_model import ExperimentModel
    from glotaran.project.library import ModelLibrary
    from glotaran.typing.types import ArrayLike


SUPPORTED_OPTIMIZATION_METHODS = {
    "TrustRegionReflection": "trf",
    "Dogbox": "dogbox",
    "Levenberg-Marquardt": "lm",
}


class UnsupportedMethodError(GlotaranUserError):
    """Indicates that the optimization method is unsupported."""

    def __init__(self, method: str) -> None:
        """Initialize an UnsupportedMethodError.

        Parameters
        ----------
        method : str
            The unsupported method.
        """
        super().__init__(
            f"Unsupported optimization method {method}. "
            f"Supported methods are '{list(SUPPORTED_OPTIMIZATION_METHODS.keys())}'"
        )


class Optimization:
    def __init__(
        self,
        *,
        models: list[ExperimentModel],
        parameters: Parameters,
        library: ModelLibrary,
        verbose: bool = True,
        raise_exception: bool = False,
        maximum_number_function_evaluations: int | None = None,
        add_svd: bool = True,
        ftol: float = 1e-8,
        gtol: float = 1e-8,
        xtol: float = 1e-8,
        x_scale: float | Literal["jac"] | list[float] | dict[str, float] = 1.0,
        parameter_penalties: list[EqualParameterPenalty] | None = None,
        optimization_method: Literal[
            "TrustRegionReflection",
            "Dogbox",
            "Levenberg-Marquardt",
        ] = "TrustRegionReflection",
    ) -> None:
        self._parameters = Parameters.empty()
        models = [
            experiment.resolve(library, self._parameters, initial=parameters)
            for experiment in models
        ]
        self._parameter_penalties = [
            resolve_item_parameters(p, self._parameters, parameters)
            for p in parameter_penalties or []
        ]
        self._warned_penalty_fallback = False
        issues = [issue for experiment in models for issue in experiment.get_issues(parameters)]
        if len(issues) > 0:
            raise GlotaranModelIssues(issues)
        self._objectives = [OptimizationObjective(experiment) for experiment in models]
        self._tee = TeeContext()
        self._verbose = verbose
        self._raise = raise_exception

        self._maximum_number_function_evaluations = maximum_number_function_evaluations
        self._add_svd = add_svd
        self._ftol = ftol
        self._gtol = gtol
        self._xtol = xtol
        self._x_scale = x_scale
        if optimization_method not in SUPPORTED_OPTIMIZATION_METHODS:
            raise UnsupportedMethodError(optimization_method)
        self._optimization_method = SUPPORTED_OPTIMIZATION_METHODS[optimization_method]

        self._parameter_history = ParameterHistory()
        self._parameter_history.append(self._parameters)
        self._free_parameter_labels, _, _, _ = self._parameters.get_label_value_and_bounds_arrays(
            exclude_non_vary=True
        )

    def calculate_parameter_penalties(self) -> np.ndarray:
        residuals = []
        for penalty in self._parameter_penalties:
            source = float(penalty.source)
            target = float(penalty.parameter) * float(penalty.target)
            if np.isclose(source, 0.0) or np.isclose(target, 0.0):
                if not self._warned_penalty_fallback:
                    warn("Falling back to absolute equal parameter penalty near zero.", stacklevel=2)
                    self._warned_penalty_fallback = True
                residuals.extend([penalty.weight * (source - target)] * 2)
            else:
                residuals.extend([penalty.weight * (source / target - 1),
                                  penalty.weight * (target / source - 1)])
        return np.asarray(residuals, dtype=float)

    def calculate_penalty(self) -> np.ndarray:
        return np.concatenate([*[o.calculate() for o in self._objectives],
                               self.calculate_parameter_penalties()])

    def run(self) -> tuple[Parameters, dict[str, OptimizationResult], OptimizationInfo]:
        """Perform the optimization.

        Raises
        ------
        Exception
            Raised if an exception occurs during optimization and raise_exception is `True`.
        """
        (
            _,
            initial_parameter,
            lower_bounds,
            upper_bounds,
        ) = self._parameters.get_label_value_and_bounds_arrays(exclude_non_vary=True)
        x_scale = self._x_scale
        if isinstance(x_scale, dict):
            if set(x_scale) != set(self._free_parameter_labels):
                raise ValueError("x_scale mapping must name every free parameter exactly once.")
            x_scale = [x_scale[label] for label in self._free_parameter_labels]
        if isinstance(x_scale, list) and len(x_scale) != len(self._free_parameter_labels):
            raise ValueError("x_scale list must match the free parameter count.")
        ls_result = None
        termination_reason = ""
        no_free_parameters = initial_parameter.size == 0
        with self._tee:
            if no_free_parameters:
                termination_reason = "No free parameters to optimize."
            else:
                try:
                    verbose = 2 if self._verbose else 0
                    ls_result = least_squares(
                        self.objective_function,
                        initial_parameter,
                        bounds=(lower_bounds, upper_bounds),
                        method=self._optimization_method,
                        max_nfev=self._maximum_number_function_evaluations,
                        verbose=verbose,
                        ftol=self._ftol,
                        gtol=self._gtol,
                        xtol=self._xtol,
                        x_scale=x_scale,
                    )
                    termination_reason = ls_result.message
                # No matter the error we want to behave gracefully
                except Exception as e:
                    if self._raise:
                        raise
                    warn(f"Optimization failed:\n\n{e}", stacklevel=3)
                    termination_reason = str(e)

        if ls_result is not None:
            self._parameters.set_from_label_and_value_arrays(self._free_parameter_labels, ls_result.x)
        # TODO: check how this works for multiple experiments with possible the same dataset name
        penalty = self.calculate_penalty()
        if no_free_parameters:
            ls_result = OptimizeResult(
                x=initial_parameter,
                fun=penalty,
                jac=np.empty((penalty.size, 0)),
                optimality=0.0,
                nfev=1,
                njev=0,
                status=1,
                success=True,
                message=termination_reason,
            )
        results = [o.get_result() for o in self._objectives]
        optimization_results = dict(ChainMap(*[r.optimization_results for r in results]))
        number_of_clps = sum(r.clp_size for r in results)
        additional_penalty = sum(r.additional_penalty for r in results)
        optimization_info = OptimizationInfo.from_least_squares_result(
            ls_result,
            self._parameter_history,
            OptimizationHistory.from_stdout_str(self._tee.read()),
            penalty,
            additional_penalty,
            self._free_parameter_labels,
            termination_reason,
            number_of_clps,
        )
        optimization_info.additional_parameter_penalty = self.calculate_parameter_penalties().tolist()
        return self._parameters, optimization_results, optimization_info

    def dry_run(self) -> tuple[Parameters, dict[str, OptimizationResult], OptimizationInfo]:
        termination_reason = "Dry run."

        penalty = self.calculate_penalty()
        results = [o.get_result() for o in self._objectives]
        data = dict(ChainMap(*[r.optimization_results for r in results]))
        number_of_clps = sum(r.clp_size for r in results)
        additional_penalty = sum(r.additional_penalty for r in results)
        result = OptimizationInfo.from_least_squares_result(
            None,
            self._parameter_history,
            OptimizationHistory.from_stdout_str(self._tee.read()),
            penalty,
            additional_penalty,
            self._free_parameter_labels,
            termination_reason,
            number_of_clps,
        )
        result.additional_parameter_penalty = self.calculate_parameter_penalties().tolist()
        return self._parameters, data, result

    def objective_function(self, parameters: ArrayLike) -> ArrayLike:
        """Calculate the objective for the optimization.

        Parameters
        ----------
        parameters : ArrayLike
            the parameters provided by the optimizer.

        Returns
        -------
        ArrayLike
            The objective for the optimizer.
        """
        self._parameters.set_from_label_and_value_arrays(self._free_parameter_labels, parameters)
        return self.calculate_penalty()
