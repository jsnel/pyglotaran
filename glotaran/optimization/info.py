"""The result class for global analysis."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

import numpy as np
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import SerializationInfo
from pydantic import ValidationInfo
from pydantic import computed_field
from pydantic import field_serializer
from pydantic import field_validator
from pydantic import model_validator

from glotaran.optimization.optimization_history import OptimizationHistory  # noqa: TC001
from glotaran.parameter import ParameterHistory  # noqa: TC001
from glotaran.utils.pydantic_serde import deserialize_from_csv
from glotaran.utils.pydantic_serde import serialize_to_csv

if TYPE_CHECKING:
    from scipy.optimize import OptimizeResult
    from typing_extensions import Self

    from glotaran.parameter import Parameters
    from glotaran.typing.types import ArrayLike


class OptimizationInfo(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    """The result of a global analysis."""

    number_of_function_evaluations: int
    """The number of function evaluations."""

    success: bool
    """Indicates if the optimizer returned a result.

    Also ``True`` when the optimizer stopped at the maximum number of function evaluations,
    see :attr:`converged`.
    """

    converged: bool | None = None
    """Whether the optimizer met its convergence tolerances (SciPy's ``success`` flag).

    ``None`` for a dry run, which does not optimize.
    """

    termination_reason: str
    """The reason (message when) the optimizer terminated"""

    free_parameter_labels: list[str]
    """List of labels of the free parameters used in optimization."""

    parameter_history: ParameterHistory
    """The parameter history."""

    optimization_history: OptimizationHistory
    """The optimization history."""

    # The below can be none in case of unsuccessful optimization

    cost: float | None = None
    """The final cost."""

    chi_square: float | None = None
    r"""The chi-square of the optimization.

    :math:`\chi^2 = \sum_i^N [{Residual}_i]^2`."""

    covariance_matrix: np.ndarray | None = None
    """Covariance matrix.

    The rows and columns are corresponding to :attr:`free_parameter_labels`."""

    number_of_clps: int | None = None
    degrees_of_freedom: int | None = None
    """Degrees of freedom in optimization :math:`N - N_{vars}`."""

    jacobian: np.ndarray | list | None = None
    """Modified Jacobian matrix at the solution

    See also: :func:`scipy.optimize.least_squares`
    """
    number_of_data_points: int | None = None
    """Number of data points :math:`N`."""
    number_of_jacobian_evaluations: int | None = None
    """The number of jacobian evaluations."""
    number_of_parameters: int | None = None
    """Number of parameters in optimization :math:`N_{vars}`"""
    optimality: float | None = None
    reduced_chi_square: float | None = None
    r"""The reduced chi-square of the optimization.

    :math:`\chi^2_{red}= {\chi^2} / {(N - N_{vars})}`.
    """
    root_mean_square_error: float | None = None
    r"""
    The root mean square error the optimization.

    :math:`rms = \sqrt{\chi^2_{red}}`
    """
    additional_penalty: float | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def glotaran_version(self) -> str:
        """The glotaran version used to create the result."""
        # Prevent circular import issues
        from glotaran import __version__  # noqa: PLC0415

        return __version__

    @field_serializer("parameter_history", "optimization_history", when_used="json")
    def serialize_data(
        self, value: ParameterHistory | OptimizationHistory, info: SerializationInfo
    ) -> str:
        """Serialize to csv."""
        return serialize_to_csv(value, info)

    @field_validator("parameter_history", "optimization_history", mode="before")
    @classmethod
    def validate_data(
        cls,
        value: Any,  # noqa: ANN401
        info: ValidationInfo,
    ) -> ParameterHistory | OptimizationHistory:
        """Deserialize from csv."""
        return deserialize_from_csv(cls.model_fields[info.field_name].annotation, value, info)  # type: ignore[return-value,arg-type,index]

    @field_serializer("covariance_matrix", "jacobian", when_used="json")
    def exclude_arrays(self, value: Any) -> None:  # noqa: ANN401
        """Exclude numpy arrays in json mode."""

    @model_validator(mode="before")
    @classmethod
    def remove_computed_fields(cls, value: Any) -> Any:  # noqa: ANN401
        """Remove computed fields before validation."""
        if isinstance(value, dict) is True:
            value.pop("glotaran_version", None)
            return value
        return value

    @classmethod
    def from_least_squares_result(
        cls,
        result: OptimizeResult | None,
        parameter_history: ParameterHistory,
        optimization_history: OptimizationHistory,
        penalty: ArrayLike,
        additional_penalty: float,
        free_parameter_labels: list[str],
        termination_reason: str,
        number_of_clps: int,
        number_of_evaluations: int,
    ) -> Self:
        success = result is not None

        result_args = {
            "success": success,
            "converged": bool(result.success) if success else False,  # type:ignore[union-attr]
            "free_parameter_labels": free_parameter_labels,
            "parameter_history": parameter_history,
            "termination_reason": termination_reason,
            "optimization_history": optimization_history,
            "number_of_function_evaluations": result.nfev  # type:ignore[union-attr]
            if success
            else number_of_evaluations,
            "cost": 0.5 * np.dot(penalty, penalty),
        }

        if success:
            result_args |= calculate_residual_statistics(
                result.fun,  # type:ignore[union-attr]
                result.x.size,  # type:ignore[union-attr]
                number_of_clps,
            )
            result_args["additional_penalty"] = additional_penalty
            result_args["number_of_jacobian_evaluations"] = result.njev  # type:ignore[union-attr]
            result_args["optimality"] = float(result.optimality)  # type:ignore[union-attr]
            result_args["jacobian"] = result.jac  # type:ignore[union-attr]
            result_args["covariance_matrix"] = calculate_covariance_matrix_and_standard_errors(
                result_args["jacobian"], result_args["root_mean_square_error"]
            )

        return cls(**result_args)

    @classmethod
    def from_dry_run(
        cls,
        parameter_history: ParameterHistory,
        optimization_history: OptimizationHistory,
        penalty: ArrayLike,
        additional_penalty: float,
        free_parameter_labels: list[str],
        number_of_clps: int,
    ) -> Self:
        """Create the info of a dry run, one function evaluation without optimization.

        The statistics of the residual are calculated as for an optimization. The Jacobian,
        and with it the covariance matrix and the standard errors, are not calculated.

        Parameters
        ----------
        parameter_history : ParameterHistory
            The parameter history.
        optimization_history : OptimizationHistory
            The optimization history.
        penalty : ArrayLike
            The penalty of the evaluation.
        additional_penalty : float
            The additional penalty of the evaluation.
        free_parameter_labels : list[str]
            The labels of the free parameters.
        number_of_clps : int
            The number of conditionally linear parameters.

        Returns
        -------
        Self
            The info of the dry run.
        """
        return cls(
            success=False,
            termination_reason="Dry run.",
            free_parameter_labels=free_parameter_labels,
            parameter_history=parameter_history,
            optimization_history=optimization_history,
            number_of_function_evaluations=1,
            cost=0.5 * np.dot(penalty, penalty),
            additional_penalty=additional_penalty,
            **calculate_residual_statistics(penalty, len(free_parameter_labels), number_of_clps),
        )


def calculate_residual_statistics(
    residual: ArrayLike, number_of_parameters: int, number_of_clps: int
) -> dict[str, Any]:
    """Calculate the statistics of a residual vector.

    Parameters
    ----------
    residual : ArrayLike
        The residual vector, including penalties.
    number_of_parameters : int
        The number of free parameters.
    number_of_clps : int
        The number of conditionally linear parameters.

    Returns
    -------
    dict[str, Any]
        The number of data points, parameters and clps, the degrees of freedom, the
        chi-square, the reduced chi-square and the root mean square error, keyed by the
        names of the :class:`OptimizationInfo` fields.
    """
    degrees_of_freedom = residual.size - number_of_parameters - number_of_clps
    chi_square = float(np.sum(residual**2))
    reduced_chi_square = chi_square / degrees_of_freedom
    return {
        "number_of_data_points": residual.size,
        "number_of_parameters": number_of_parameters,
        "number_of_clps": number_of_clps,
        "degrees_of_freedom": degrees_of_freedom,
        "chi_square": chi_square,
        "reduced_chi_square": reduced_chi_square,
        "root_mean_square_error": float(np.sqrt(reduced_chi_square)),
    }


def calculate_parameter_errors(
    optimization_info: OptimizationInfo, parameters: Parameters
) -> None:
    """Calculate and assign standard errors to parameters in place based on ``optimization_info``.

    This function calculates the standard errors for the free parameters
    based on the provided optimization information and assigns these errors
    directly to the corresponding parameters.

    Parameters
    ----------
    optimization_info : OptimizationInfo
        An object containing the optimization results, including the covariance
        matrix and root mean square error.
    parameters : Parameters
        An object containing the parameters to be updated. The standard errors
        will be assigned to the parameters in place.

    Returns
    -------
    None
    """
    if optimization_info.covariance_matrix is not None:
        standard_errors = optimization_info.root_mean_square_error * np.sqrt(
            np.diag(optimization_info.covariance_matrix)
        )
        for label, error in zip(
            optimization_info.free_parameter_labels, standard_errors, strict=False
        ):
            parameters.get(label).standard_error = error


def calculate_covariance_matrix_and_standard_errors(
    jacobian: ArrayLike,
    root_mean_square_error: float,  # noqa: ARG001
) -> ArrayLike:
    """Calculate the covariance matrix and standard errors of the optimization.

    Parameters
    ----------
    jacobian : ArrayLike
        The jacobian matrix.
    root_mean_square_error : float
        The root mean square error.

    Returns
    -------
    ArrayLike
        The covariance matrix.
    """
    # See PR #706: More robust covariance matrix calculation
    _, jacobian_sv, jacobian_rsv = np.linalg.svd(jacobian, full_matrices=False)
    jacobian_sv_square = jacobian_sv**2
    mask = jacobian_sv_square > np.finfo(float).eps
    return (jacobian_rsv[mask].T / jacobian_sv_square[mask]) @ jacobian_rsv[mask]
