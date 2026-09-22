# CLP uncertainty in staging

The `compute_clp_standard_error` scheme option ports the v0.7 CLP uncertainty feature into the
v0.8 optimizer. It is opt in because it adds one finite difference objective evaluation for every
free nonlinear parameter. The finite difference step defaults to `1e-6` relative to the optimizer
coordinate and can be changed with
`clp_standard_error_finite_difference_relative_step`.

For a variable projection fit, each CLP variance combines two terms:

\[
  \operatorname{Var}(c) = \operatorname{diag}(A^+ \sigma^2 A^{+T})
      + \operatorname{diag}(S\,\operatorname{Cov}(\theta)\,S^T).
\]

The first term uses the weighted and dataset-scaled staging matrix at each global index. Relations
and constraints are expanded back onto the full CLP label axis before the diagonal is selected. For
linked datasets the covariance is calculated on the aligned matrix and then projected to every
dataset's original global index. Global models use the full Cartesian or paired matrix, including
the paired/single-amplitude diagonal mapping.

Staging stores `OptimizationInfo.covariance_matrix` as the unscaled pseudo-inverse of the
nonlinear Jacobian. The optimizer multiplies it by `root_mean_square_error**2` exactly once before
the nonlinear propagation term. Finite differences use the same optimizer coordinates as SciPy:
non-negative parameters are perturbed in log space and then mapped back to physical values. This
fixes a source-branch defect where raw physical values were perturbed against a transformed
covariance matrix.

The resulting `xarray.DataArray` is available as
`result.optimization_results[dataset].fit_decomposition.clp_standard_error`. It has the same
dimensions and coordinates as `fit_decomposition.clp`, and records the method and finite
difference step in its attributes. The optimization info records the method and configured step.

Only `variable_projection` is supported. Enabling this option for a non-negative least-squares
dataset raises a validation error at optimizer construction, rather than returning values with a
linear covariance interpretation that does not apply.

The finite difference evaluator always restores all parameter values in a `finally` block. A failed
perturbed evaluation therefore cannot leave the optimizer at a perturbed state.
