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

The source prototype's private provider access and shared-object mutation were not carried over.
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
