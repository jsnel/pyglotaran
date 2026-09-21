# Optimizer additions

`Scheme.parameter_penalties` holds global equal-parameter soft constraints. Each
adds two residuals, once per objective call across all experiments:
`weight * (source / (parameter * target) - 1)` and its reciprocal form.
The source branch's near-zero absolute-difference fallback is retained, with a
single warning per optimization. Results persist the final residual values in
`optimization_info.additional_parameter_penalty`.

`Scheme.x_scale` is persisted and can be overridden in `Scheme.optimize`.
Supported forms are a positive finite scalar, `jac`, a positive numeric list in
free-parameter order, or a mapping naming every free parameter. The mapping is
recommended when individual scales are needed; it binds by label, unlike the
prototype ndarray. Shape/key errors fail before the optimizer runs.

`EqualAreaPenalty.relative` defaults to false. True uses
`weight * (source_area / (parameter * target_area) - 1)`; near-zero targets use
the absolute difference as in the source. These choices change the objective,
so weights must reflect the desired scientific constraint.

The NNLS iteration allowance is six times the number of columns, matching the
source's accommodation for difficult correlated designs.

Result assembly explicitly restores the optimizer's returned parameter vector
before calculating final residuals. This prevents a finite-difference perturbation
from becoming the saved solution. The exact-solution regression now requires the
initial and final parameters to agree. The YAML writer also leaves enough sequence
indentation to round-trip lists of penalty mappings correctly.

Validation checkpoint: 82 tests pass across optimization, Scheme, and YAML IO,
including direct objective arithmetic, scaling forms, and scheme YAML round trips.
A pre-existing YAML result test compared relative returned paths to absolute paths;
its comparison now resolves paths before comparing, without changing IO behavior.
