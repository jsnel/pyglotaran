from __future__ import annotations

from collections import ChainMap
from collections import Counter
from typing import TYPE_CHECKING
from typing import Literal
from warnings import warn

import numpy as np
from scipy.optimize import OptimizeResult
from scipy.optimize import least_squares

from glotaran.model.errors import GlotaranModelIssues
from glotaran.model.errors import GlotaranUserError
from glotaran.model.item import resolve_item_parameters
from glotaran.optimization.clp_standard_error import ClpStandardErrorSettings
from glotaran.optimization.clp_standard_error import calculate_clp_standard_error
from glotaran.optimization.info import OptimizationInfo
from glotaran.optimization.info import OptimizerSettings
from glotaran.optimization.objective import OptimizationObjective
from glotaran.optimization.objective import OptimizationResult
from glotaran.optimization.optimization_history import OptimizationHistory
from glotaran.parameter import ParameterHistory
from glotaran.parameter import Parameters
from glotaran.utils.io import check_file_name_labels
from glotaran.utils.tee import TeeContext

if TYPE_CHECKING:
    from glotaran.model.experiment_model import ExperimentModel
    from glotaran.model.parameter_penalties import EqualParameterPenalty
    from glotaran.optimization.objective import OptimizationObjectiveResult
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
        ftol: float | None = 1e-8,
        gtol: float | None = 1e-8,
        xtol: float | None = 1e-8,
        x_scale: float | Literal["jac"] | list[float] | dict[str, float] = 1.0,
        parameter_penalties: list[EqualParameterPenalty] | None = None,
        compute_clp_standard_error: bool = False,
        clp_standard_error_finite_difference_relative_step: float = 1e-6,
        optimization_method: Literal[
            "TrustRegionReflection",
            "Dogbox",
            "Levenberg-Marquardt",
        ] = "TrustRegionReflection",
    ) -> None:
        label_counts = Counter(label for experiment in models for label in experiment.datasets)
        if repeated := [label for label, count in label_counts.items() if count > 1]:
            msg = (
                f"Dataset labels {repeated} are used in more than one experiment. The results "
                "of a fit are stored by dataset label, so dataset labels must be unique across "
                "experiments."
            )
            raise GlotaranUserError(msg)
        # The result rejects these labels as well, but only after the fit.
        try:
            check_file_name_labels(label_counts, "Dataset")
        except ValueError as error:
            raise GlotaranUserError(str(error)) from error
        self._parameters = Parameters.empty()
        models = [
            experiment.resolve(library, self._parameters, initial=parameters)
            for experiment in models
        ]
        self._parameter_penalties = [
            resolve_item_parameters(p, self._parameters, parameters)
            for p in parameter_penalties or []
        ]
        self._parameters.validate_bounds(raise_exception=True)
        if compute_clp_standard_error:
            unsupported = [
                f"experiment[{experiment_index}]:{dataset_label}"
                for experiment_index, experiment in enumerate(models)
                for dataset_label, dataset in experiment.datasets.items()
                if dataset.residual_function != "variable_projection"
            ]
            if unsupported:
                msg = (
                    "CLP standard errors require variable_projection residuals; "
                    f"unsupported datasets: {unsupported}."
                )
                raise ValueError(msg)
        self._warned_penalty_fallback = False
        issues = [issue for experiment in models for issue in experiment.get_issues(parameters)]
        if len(issues) > 0:
            raise GlotaranModelIssues(issues)
        self._objectives = [OptimizationObjective(experiment) for experiment in models]
        self._tee = TeeContext()
        self._verbose = verbose
        self._raise = raise_exception

        self._compute_clp_standard_error = compute_clp_standard_error
        self._clp_standard_error_settings = ClpStandardErrorSettings(
            relative_step=clp_standard_error_finite_difference_relative_step
        )
        if optimization_method not in SUPPORTED_OPTIMIZATION_METHODS:
            raise UnsupportedMethodError(optimization_method)
        self.settings = OptimizerSettings(
            optimization_method=optimization_method,
            ftol=ftol,
            gtol=gtol,
            xtol=xtol,
            maximum_number_function_evaluations=maximum_number_function_evaluations,
            x_scale=x_scale,
        )

        self.cost_history: list[float] = []
        """Cost of every function evaluation in call order, including Jacobian evaluations."""
        self.error: Exception | None = None
        """Exception of the optimizer, turned into a warning with ``raise_exception=False``."""
        self.converged: bool | None = None
        """SciPy's success flag: whether the optimizer met its tolerances."""
        self.parameter_history = ParameterHistory()
        self.parameter_history.append(self._parameters)
        self._free_parameter_labels, _, _, _ = self._parameters.get_label_value_and_bounds_arrays(
            exclude_non_vary=True
        )

    @property
    def parameters(self) -> Parameters:
        """Parameters of the optimization.

        During a fit the last evaluated ones; after a successful fit the solution, after a failed
        one the last evaluated ones.
        """
        return self._parameters

    def calculate_parameter_penalties(self) -> np.ndarray:
        residuals = []
        for penalty in self._parameter_penalties:
            source = float(penalty.source)
            target = float(penalty.parameter) * float(penalty.target)
            if np.isclose(source, 0.0) or np.isclose(target, 0.0):
                if not self._warned_penalty_fallback:
                    warn(
                        "Falling back to absolute equal parameter penalty near zero.", stacklevel=2
                    )
                    self._warned_penalty_fallback = True
                residuals.extend([penalty.weight * (source - target)] * 2)
            else:
                residuals.extend(
                    [
                        penalty.weight * (source / target - 1),
                        penalty.weight * (target / source - 1),
                    ]
                )
        return np.asarray(residuals, dtype=float)

    def calculate_penalty(self) -> np.ndarray:
        return np.concatenate(
            [*[o.calculate() for o in self._objectives], self.calculate_parameter_penalties()]
        )

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
        x_scale = self.settings.x_scale
        if isinstance(x_scale, dict):
            if set(x_scale) != set(self._free_parameter_labels):
                msg = "x_scale mapping must name every free parameter exactly once."
                raise ValueError(msg)
            x_scale = [x_scale[label] for label in self._free_parameter_labels]
        if isinstance(x_scale, list) and len(x_scale) != len(self._free_parameter_labels):
            msg = "x_scale list must match the free parameter count."
            raise ValueError(msg)
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
                        method=SUPPORTED_OPTIMIZATION_METHODS[self.settings.optimization_method],
                        max_nfev=self.settings.maximum_number_function_evaluations,
                        verbose=verbose,
                        ftol=self.settings.ftol,
                        gtol=self.settings.gtol,
                        xtol=self.settings.xtol,
                        x_scale=x_scale,
                    )
                    termination_reason = ls_result.message
                    self.converged = bool(ls_result.success)
                    # The solution, instead of the last evaluated point (often a
                    # finite-difference Jacobian step), so that the result describes one point
                    self._parameters.set_from_label_and_value_arrays(
                        self._free_parameter_labels, ls_result.x
                    )
                # No matter the error we want to behave gracefully
                except Exception as e:
                    if self._raise:
                        raise
                    warn(f"Optimization failed:\n\n{e}", stacklevel=3)
                    termination_reason = str(e)
                    self.error = e

        try:
            # Without free parameters, this is the only evaluation and is recorded as such.
            penalty = (
                self.objective_function(initial_parameter)
                if no_free_parameters
                else self.calculate_penalty()
            )
        except Exception as e:
            # After an exception in the objective, the evaluation at the same parameters fails
            # again; the error of the optimization is the one to report.
            optimization_error = self.error
            if optimization_error is None:
                raise
            raise optimization_error from e
        if no_free_parameters:
            self.converged = True
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
            self.parameter_history,
            OptimizationHistory.from_stdout_str(self._tee.read()),
            penalty,
            additional_penalty,
            self._free_parameter_labels,
            termination_reason,
            number_of_clps,
            number_of_function_evaluations=len(self.cost_history),
        )
        optimization_info.additional_parameter_penalty = (
            self.calculate_parameter_penalties().tolist()
        )
        optimization_info.additional_penalty_areas = [
            diagnostic
            for objective in self._objectives
            for diagnostic in objective.get_clp_penalty_diagnostics()
        ]
        self._calculate_clp_standard_errors(results, optimization_info)
        return self._parameters, optimization_results, optimization_info

    def _calculate_clp_standard_errors(
        self,
        results: list[OptimizationObjectiveResult],
        optimization_info: OptimizationInfo,
    ) -> None:
        """Calculate and attach CLP uncertainty after a successful fit."""
        if not self._compute_clp_standard_error:
            return
        if (
            optimization_info.covariance_matrix is None
            or optimization_info.root_mean_square_error is None
        ):
            return

        optimization_info.clp_standard_error_method = "linear_plus_nonlinear_propagation"
        optimization_info.clp_standard_error_finite_difference_relative_step = (
            self._clp_standard_error_settings.relative_step
        )
        # ``OptimizationInfo.covariance_matrix`` intentionally remains the unscaled
        # pseudo-inverse used for parameter diagnostics.  CLP propagation needs the
        # residual variance exactly once.
        covariance = (
            optimization_info.root_mean_square_error**2 * optimization_info.covariance_matrix
        )
        for objective, objective_result in zip(self._objectives, results, strict=True):
            standard_errors = calculate_clp_standard_error(
                optimization_objective=objective,
                parameters=self._parameters,
                free_parameter_labels=self._free_parameter_labels,
                covariance_matrix=covariance,
                settings=self._clp_standard_error_settings,
                base_results=objective_result.optimization_results,
                evaluate_at=lambda candidate, current_objective=objective: (
                    self._evaluate_objective_at(current_objective, candidate)
                ),
            )
            for label, error in standard_errors.items():
                result = objective_result.optimization_results[label]
                if result.fit_decomposition is None:
                    continue
                result.fit_decomposition = result.fit_decomposition.model_copy(
                    update={"clp_standard_error": error}
                )

    def _evaluate_objective_at(
        self, objective: OptimizationObjective, candidate: Parameters
    ) -> dict[str, OptimizationResult]:
        """Evaluate an objective at copied parameter values and restore the fit state."""
        original_values = {
            parameter.label: parameter.value for parameter in self._parameters.all()
        }
        try:
            for parameter in candidate.all():
                self._parameters.get(parameter.label).value = parameter.value
            self._parameters.update_parameter_expression()
            return objective.get_result().optimization_results
        finally:
            for label, value in original_values.items():
                self._parameters.get(label).value = value
            self._parameters.update_parameter_expression()

    def dry_run(self) -> tuple[Parameters, dict[str, OptimizationResult], OptimizationInfo]:
        termination_reason = "Dry run."

        penalty = self.calculate_penalty()
        results = [o.get_result() for o in self._objectives]
        data = dict(ChainMap(*[r.optimization_results for r in results]))
        number_of_clps = sum(r.clp_size for r in results)
        additional_penalty = sum(r.additional_penalty for r in results)
        result = OptimizationInfo.from_least_squares_result(
            None,
            self.parameter_history,
            OptimizationHistory.from_stdout_str(self._tee.read()),
            penalty,
            additional_penalty,
            self._free_parameter_labels,
            termination_reason,
            number_of_clps,
            number_of_function_evaluations=1,
            dry_run=True,
        )
        result.additional_parameter_penalty = self.calculate_parameter_penalties().tolist()
        result.additional_penalty_areas = [
            diagnostic
            for objective in self._objectives
            for diagnostic in objective.get_clp_penalty_diagnostics()
        ]
        return self._parameters, data, result

    def objective_function(self, parameters: ArrayLike) -> ArrayLike:
        """Calculate the objective for the optimization.

        Appends the cost to ``cost_history`` and, with ``verbose``, the parameter values to the
        parameter history.

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
        penalty = self.calculate_penalty()
        self.cost_history.append(0.5 * float(np.dot(penalty, penalty)))
        if self._verbose:
            self.parameter_history.append(self._parameters, len(self.cost_history))
        return penalty
