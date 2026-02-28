# CLP Standard Error (CLP-SE) Design Notes

This document explains the design and implementation of CLP standard errors in pyglotaran for developers.

## Goal

Add optional post-fit uncertainty estimates for conditionally linear parameters (CLPs), while preserving current behavior when the feature is not enabled.

The feature is designed to:

- keep optimization behavior unchanged by default,
- reuse existing optimizer covariance information,
- support constraints and relations in CLP reconstruction,
- expose results in a way consistent with existing dataset result variables.

## Scope and Non-Goals

### In scope

- Optional CLP-SE computation after successful optimization.
- Propagation of nonlinear parameter uncertainty into CLPs.
- Linear least-squares uncertainty contribution at fixed nonlinear parameters.
- Result-level metadata and dataset-level `clp_standard_error` output.

### Out of scope (current implementation)

- Analytic (symbolic) CLP sensitivities.
- Broad support for all residual methods.
- Dedicated performance optimizations (parallel finite-difference, caching layers).

## User-Facing API

### Scheme options

Implemented in `glotaran/project/scheme.py`:

- `compute_clp_standard_error: bool = False`
- `clp_standard_error_finite_difference_relative_step: float = 1e-6`

This keeps backward compatibility: existing schemes are unaffected unless explicitly enabling CLP-SE.

### Result metadata

Implemented in `glotaran/project/result.py`:

- `clp_standard_error_method: str | None`
- `clp_standard_error_finite_difference_relative_step: float | None`

These fields document how CLP-SE was computed for a given result.

## Integration Points

### Optimizer orchestration

`glotaran/optimization/optimizer.py` orchestrates CLP-SE post-fit:

1. Run optimization.
2. Compute nonlinear covariance matrix from Jacobian.
3. If CLP-SE is enabled, call CLP-SE core computation per optimization group.
4. Attach CLP-SE arrays to dataset results.

CLP-SE is intentionally computed after optimization convergence to avoid changing the objective and to isolate uncertainty logic from fitting logic.

### Dataset result output

`glotaran/optimization/optimization_group.py` accepts optional CLP-SE arrays and writes:

- `clp_standard_error` with same coordinates and dimensions as `clp`.

This mirrors existing result conventions and simplifies downstream plotting.

## Core Computation

Implemented in `glotaran/optimization/clp_standard_error.py`.

Total CLP variance is modeled as:

- `Var_total(beta) = Var_linear(beta | theta*) + J_beta_theta * Cov(theta) * J_beta_theta^T`

where:

- `theta` are free nonlinear parameters,
- `beta` are CLPs,
- `Cov(theta)` is optimizer covariance,
- `J_beta_theta = d beta / d theta` at the optimum.

### Linear contribution

For each reduced linear problem, covariance is approximated with pseudo-inverse:

- `Cov(beta_reduced) ~= sigma^2 * pinv(X^T X)`

Then mapped to full CLP space using the existing reconstruction path (`retrieve_clps`) so constraints/relations are honored.

### Nonlinear propagated contribution

`J_beta_theta` is approximated using forward finite differences:

- perturb one parameter at a time,
- recompute CLPs,
- estimate `(beta(theta + h) - beta(theta)) / h`.

Step size uses relative scaling:

- `h = relative_step * max(1.0, abs(theta_i))`

This is robust for mixed parameter magnitudes and simple to reason about.

### TIMP parity and sigma^2 scaling

Comparison against TIMP's `getStdErrClp` implementation showed an important scaling requirement:

- both the linear CLP variance term and the nonlinear propagated term must be scaled by residual variance (`sigma^2`).

In TIMP notation, the propagated part uses `G * R_inv * G^T`, where `R_inv` is based on the nonlinear Hessian/Jacobian covariance and `sigma^2` is applied in the final CLP variance expression.

For pyglotaran this implies:

- linear block: `sigma^2 * pinv(X^T X)` (already applied in `clp_standard_error.py`),
- nonlinear block: use `Cov(theta) ~= sigma^2 * (J^T J)^-1` when propagating with finite-difference sensitivities.

An earlier implementation passed an unscaled nonlinear covariance matrix (`(J^T J)^-1`) into CLP propagation. For low-RMSE fits this can inflate CLP standard errors by approximately `1 / RMSE^2`.

The fix is applied in `glotaran/optimization/optimizer.py` before calling `calculate_clp_standard_error(...)`:

- `clp_propagation_covariance = (root_mean_square_error ** 2) * covariance_matrix`

This restores parity with the TIMP formula and keeps CLP-SE magnitudes on the expected scale.

### Residual-method support

Current logic explicitly targets the variable-projection residual path and warns/skips for unsupported residual methods.

Rationale: CLP-SE relies on stable linear-subproblem behavior and reconstruction semantics that are currently best defined for this path.

## Design Choices and Rationale

### 1. Feature-flagged and default-off

Why: avoid regressions and preserve historical optimization behavior/performance unless requested.

### 2. Post-fit architecture

Why: uncertainty computation should be an analysis layer on top of fit results, not part of optimization dynamics.

### 3. Reuse existing providers and reconstruction

Why: matrix reduction, constraints, and relations are already centralized in provider logic. Reusing them reduces duplicate logic and risk.

### 4. Pseudo-inverse for linear covariance

Why: handles rank-deficient/ill-conditioned systems more gracefully than direct inverse.

### 5. Finite-difference sensitivity (v1)

Why: complete, general, and maintainable first implementation with minimal model-specific math.

## Numerical and Performance Considerations

- Finite-difference cost scales with number of free nonlinear parameters.
- Ill-conditioning can increase noise in covariance and sensitivities.
- Relative SE plots (`|SE/CLP|`) can be more interpretable than absolute SE in low-amplitude regions.

Potential future work:

- central finite differences,
- configurable differencing strategy,
- optional parameter batching/parallel FD,
- analytic sensitivities for selected models.

## Testing Strategy

Current coverage includes integration checks in optimization tests for:

- metadata population,
- presence of `clp_standard_error`,
- shape/dimension alignment with `clp`,
- finite numeric values.

Recommended future additions:

- focused unit tests for linear and propagated covariance blocks,
- constraints/relations edge cases,
- linked/unlinked parity tests,
- IO roundtrip tests specific to CLP-SE-heavy results.

## Files of Interest

- `glotaran/project/scheme.py`
- `glotaran/project/result.py`
- `glotaran/optimization/clp_standard_error.py`
- `glotaran/optimization/optimizer.py`
- `glotaran/optimization/optimization_group.py`
- `glotaran/optimization/test/test_optimization.py`
- `docs/source/notebooks/clp_standard_error/visualize_clp_standard_error.ipynb`

## Practical Developer Notes

- When changing CLP reconstruction behavior, verify CLP-SE mapping logic still matches `retrieve_clps` semantics.
- When adding residual functions, decide explicitly whether CLP-SE is supported and test numerical stability.
- Keep method metadata up to date so result provenance remains clear for users and downstream tooling.
