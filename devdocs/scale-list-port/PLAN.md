# scale_list functionality port to staging

## Revision contract

- Source: ism200/pyglotaran `scale_list`, fixed at `84d838991e1d0280954350796037763d4bd612aa`.
- Comparison main: `e6ba6316f6bc7365211a841fc417c4a98f65860f`.
- Target base: upstream staging `afce5d503066ed7f3acf4eae0958db0e20bdc326`.
- Delivery branch: `codex/port-scale-list`.

The source includes earlier divergent history. Endpoint behavior and patch equivalence,
not the raw list of unmerged commits, define the relevant delta. Existing source
`devdocs/analysis/scale_list` documents are useful requirements evidence, but describe
an earlier head and are not a substitute for reviewing the fixed source revision.

## Relevant delta inventory

The source adds per-index dataset scales, paired single-amplitude full models,
parameter equality penalties, relative equal-area penalties, optimizer x_scale,
CLP standard errors, extended Gaussian IRFs and width dispersion, initial-concentration
normalization controls, coherent-artifact orders four/five, parameter-table improvements,
and a Hamamatsu ITEX IMG loader. It also changes numerical guards, validation,
result reporting and persistence. The [final inventory](DELTA.md) classifies the relevant behavior and
the [usage guide](USAGE.md) maps it to staging APIs.

Do not transplant v0.7 provider architecture or source CI/dependency configuration.
Keep staging elements, experiment/data models, optimization objectives and typed
results. Compatibility adapters and cross-version evidence belong in the validation
workspace. Scientific bugs in the prototype must be identified explicitly rather
than reproduced to make serialized arrays equal.

## Implementation plan and commit checkpoints

1. Commit this revision contract and plan before implementation. Complete independent
   source-delta and staging-architecture audits using GPT 5.6 Luna at xhigh effort.
2. Port independent IO and parameter-table functionality with focused tests and
   persistence checks. Register the IMG reader using staging packaging.
3. Extend staging model and objective contracts for per-index scaling and paired
   composition. Keep scalar/cartesian defaults unchanged; test matrix construction,
   simulation, optimization, result reconstruction and degrees of freedom together.
4. Port IRF capabilities into staging IRF contracts; verify closed-form physics,
   index dependence, input validation, parameter discovery and YAML round trips.
5. Add optimization settings, penalties and opt-in CLP uncertainty through staging
   APIs. Evaluate penalties once per intended scope, preserve residual length and
   ensure result state corresponds to the final optimizer solution.
6. Run focused suites per checkpoint, staging regression suites, fresh common
   reference/staging notebook validation and semantic comparisons. Run the separate
   report-only runtime benchmark when optimizer runtime paths change. Add dedicated
   source-versus-port numerical evidence for the new functionality.
7. Update the delta/migration documentation, changelog and workspace validation log;
   commit final evidence references and deliver the branch with all remaining limits
   stated precisely. Never commit generated evidence directories.

## Scientific acceptance

Existing validation requires 11/11 notebooks per branch and all 14 result leaves,
with no REGRESSION or BASELINE_FAILURE. Retain documented expected differences and
label-aware fitted-data tolerances. New features require independent arithmetic or
simulation-based tests, not merely reproducing prototype output. Scale/CLP gauge
ambiguities must be documented; paired composition must count only fitted amplitudes.
CLP uncertainty is opt-in and must not mutate the fitted solution. Preserve existing
reference inputs and author notebooks; use fresh timestamped evidence directories.

## Status

Plan committed before implementation as `ded0463b`. Steps 1–5 are implemented in
separate checkpoints, culminating in `d38d973b`. Full core tests and fresh common
notebook acceptance pass; the separate runtime benchmark is completing. See
[EVIDENCE.md](EVIDENCE.md) for exact sources, feature tests and scientific limits.
