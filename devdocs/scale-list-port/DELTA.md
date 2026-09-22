# Endpoint delta and port coverage

This audit compares the source endpoint `84d838991e1d0280954350796037763d4bd612aa`
(`ism200/scale_list`) with `origin/main` at `e6ba6316`, then maps the relevant source behavior
onto v0.8 staging, which started at `origin/staging` `afce5d50`.

The comparison uses the two endpoint trees. `git merge-base origin/main 84d83899` resolves to an
old multi-base history, so `origin/main...84d83899` describes roughly 1,400 commits of shared
repository history and is not a useful feature delta. The relevant v0.7.4 custom commits after the
tree-equivalent reference are the following:

| Source revision | Functionality | Staging coverage |
| --- | --- | --- |
| `7d65e85d`, `c8f38edd`, `9aa731ad`, `1b09ebed` | CLP standard errors, covariance scaling, and single-amplitude crash/count fixes | Ported with staging matrices, relations, links, global paired composition, transformed-coordinate finite differences, and result serialization |
| `2e99eeb5` | Single-amplitude model behavior | Ported through the preferred `global_composition: paired` contract and compatibility `single_amplitude_model` key |
| `b1950161` | `scale_list` and multi/multi-Gaussian IRF composition | Ported; scales are resolved against the global axis and applied in the prepared matrix path |
| `4c8a577e` | TCSPC analysis, relative equal-area penalties, extended Gaussian IRFs, width dispersion, coherent orders 4/5, NNLS iteration | Ported by the corresponding staging element and penalty changes; validation is covered by focused tests |
| `60733c3d`, `c66e9a28` | Parameter penalties and optimizer `x_scale` | Ported with global penalties, mapping/list validation, and optimizer-coordinate scaling |
| `0d0ad12d`, `aee20b18` | Parameter/model validation and initial-concentration checks | Ported into staging validation and bound diagnostics |
| `84d83899` | Hamamatsu ITEX IMG loader | Ported with loader tests |
| `d14bd234`, `f7b6b4a2` | Branch analysis and architecture documentation | Replaced by this endpoint audit plus the focused port documents in this directory |

The v0.8 implementation keeps its result layout and model architecture. `scale_list` is represented
by `DataModel.scale_list`, validated when data are prepared, applied to unlinked and linked fitted
matrices (including full models where the field is declared), and recorded in result metadata.
Full/global models retain their full-model composition semantics while scaling remains an explicit
model field.

The CLP uncertainty port is available with `compute_clp_standard_error: true`. It records
`fit_decomposition.clp_standard_error`, propagates the scaled nonlinear covariance once, handles
relations and constraints, and rejects NNLS opt-in because its CLP solve is not the linear
variable-projection solve assumed by the covariance formula. See
[`CLP_UNCERTAINTY.md`](CLP_UNCERTAINTY.md).

The source prototype's private provider access and unrestored shared-object mutation were not
carried over.
Staging exposes read-only objective data/model accessors, evaluates finite differences in the same
coordinates used by SciPy (including log coordinates for non-negative parameters), and restores
parameter values after every perturbed evaluation. This makes the uncertainty result reviewable and
prevents a failed probe from changing the final fit.

Known architecture differences are intentional: staging uses `DataModel`/`ExperimentModel` and
`OptimizationObjective` instead of the source `DatasetModel`/provider stack; the old v0.7 KMatrix
markdown API has no staging equivalent; and full-model CLP penalties remain outside the existing
global objective. Parameter table reporting uses the staging-native optimized table, and result
saving uses its native saving options to suppress selected fields. These cases are documented
rather than made to look source-identical through post-processing.

## Coverage decisions

| Source behavior | Port decision and evidence |
| --- | --- |
| Per-coordinate `scale_list`, including fitted parameter labels | Native dynamic parameter references; unlinked/linked/full matrices, simulation and result metadata. Linked free-scale recovery prevents accidental numeric caching. See [SCALING.md](SCALING.md). |
| Single-amplitude global models | Native paired column composition, diagonal reported amplitudes and paired CLP/degrees-of-freedom counts; independent synthetic endpoint fit. |
| Equal-parameter soft constraints | Two symmetric ratio residuals, once per overall optimization; fixed-length absolute fallback near zero and final diagnostics. |
| Relative equal-area penalties | Ratio residual with absolute fallback near zero; typed source/target area diagnostics in optimization info. Existing global-objective CLP-penalty restriction remains. |
| Optimizer `x_scale` | Persisted Scheme field plus per-call override; scalar, `jac`, ordered vector, and a validated label mapping. |
| CLP standard errors | Opt-in Scheme settings and fit keyword overrides, native split result persistence, linear plus nonlinear covariance propagation. NNLS is explicitly rejected. See [CLP_UNCERTAINTY.md](CLP_UNCERTAINTY.md). |
| Spectral Gaussian IRF extensions | Existing Gaussian types gain numeric-knot cubic splines, skewed-Gaussian width corrections and the source wavenumber alias; migration maps source spectral type aliases. |
| Multi/multi, convolved and normalized-convolved IRFs | Registered native activation types with nested parameter discovery, relative centers/scales, positional broadening and area normalization. See [IRF.md](IRF.md). |
| Coherent artifacts through order five | Native third/fourth Gaussian derivatives, preserving the previous first three orders. |
| Initial-concentration checks and reporting | Native compartment mapping makes source parallel-list length mismatches structurally unnecessary; external translation validates the lists. Matrix and reporting paths share normalization and preserve exclusions. |
| Parameter table input validation | Duplicate normalized labels rejected, unknown columns warned, caller tables preserved and derived T-values ignored on input. |
| Optimized parameter tables | Optional T-values and blank fixed-parameter errors across CSV/TSV/XLSX/ODS; initial tables retain original values/metadata. |
| Initial-value bounds | Public `Parameters.validate_bounds`, invoked before optimization; diagnostics rather than silently clipping input. |
| Hamamatsu ITEX IMG | Registered native loader with calibration, ROI/reversal and 8/16/32-bit data tests. |
| Suppressing saved datasets | Existing v0.8 `SAVING_OPTIONS_MINIMAL` provides the equivalent behavior; exclusion filters are not replaced by source inclusion-filter semantics. |
| NNLS iteration budget | Source `6 * number_of_columns` maximum iteration budget retained in the native solver. |
| KMatrix markdown changes | Old v0.7 presentation API has no native counterpart; not reintroduced. Numerical kinetic functionality is handled by native elements. |
| Source notebooks, analysis notes, CI and dependency pins | Used as requirements/reference evidence, not copied wholesale into the v0.8 package. Native docs and focused fixtures replace prototype architecture notes. Scientific dependencies are unchanged. |

## Intentional correctness changes

The port keeps numeric spline knots out of parameter resolution, recognizes convolution
width as index-dependent without requiring an unrelated shift, and uses the correct
per-index IRF scale row in kinetic matrices. It restores SciPy's returned optimum before
constructing final results. Uncertainty probes use optimizer coordinates, including log
coordinates, and restore shared parameter values even after exceptions. Native YAML
lists of mappings serialize with valid indentation. These are implementation fixes,
not post-processing to force agreement between non-identifiable parameters.
