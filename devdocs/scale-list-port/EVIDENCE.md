# Port verification

The runtime implementation is pinned at `d38d973bc48501e9ba0a23e893acc8cb5293e1d8`,
based on upstream staging `afce5d503066ed7f3acf4eae0958db0e20bdc326`. Subsequent
port documentation commits do not change the tested implementation.

## Regression and feature tests

- Complete staging suite: 534 passed, 9 expected failures.
- All 47 changed Python files pass Ruff lint and formatting checks.
- Validation workspace suite: 67 passed, 1 skipped.
- Focused coverage includes free-scale recovery with a fixed linked reference,
  scalar/vector composition, paired matrix/CLP counting, independent Gaussian
  and derivative formulas, nested IRF YAML, final parameter restoration, global
  penalties, table and IMG IO, and noisy uncertainty fits plus persistence.
- Ten dedicated CLP uncertainty tests cover unlinked/linked/paired objectives,
  optimizer-coordinate covariance propagation, constructor rejection for NNLS,
  state restoration on an exception, and native Scheme keyword overrides.

The validation workspace contains detailed logs and generated artifacts; these
are intentionally not tracked as core source files.

## Exact source endpoint comparison

`validation/scale_list_port.py` creates analytic Gaussian-convolved decays with
small deterministic noise, independently of either pyglotaran implementation.
It executes the requested source endpoint `84d838991e1d0280954350796037763d4bd612aa`
and this port in their separate installed scientific environments.

| Case | Fitted-data normalized RMS | Evaluations (both) | CLPs (both) |
| --- | ---: | ---: | ---: |
| Paired global model | 3.326e-16 | 4 | 2 |
| Linked scale lists (two datasets) | 0 exactly | 5 | 34 |
| Unlinked scale lists (two datasets) | 0 exactly | 5 | 68 |

Fitted rates also match exactly. The paired comparison additionally reconstructs
fitted data from named matrix/CLP dimensions and retains the raw source fitted
data. In this fixture raw and reconstructed source values agree to 3.326e-16;
it does not reproduce the historical full-model residual-reshape defect.

These fixtures use fixed scale lists. Free-scale fitting is separately tested
with a linked reference, recovering `[2, 3, 4]`. Unconstrained scales can trade
against free amplitudes; this port does not claim identifiability for such models.

Selected source tests themselves give 11 passed and two failures at the exact
source revision, both caused by numeric spline knots being treated as parameter
objects. The native port fixes that defect and verifies numeric knot handling.

Final source comparison artifact (workspace relative):
`validation/runs/scale-list-port-20260922-015900/synthetic-comparison-20260922-023731.json`.
Both captured source trees are clean. Harness SHA-256:
`01f2a4f75cc9d24fbeb28e0907cb76e7361b5cae3e0b48307189f9da7ad4def1`.

## Scientific environment and limits

Source/main uses NumPy 2.2.6, SciPy 1.15.3 and xarray 2025.6.1; staging uses
NumPy 2.0.1, SciPy 1.14.1 and xarray 2024.7.0. No dependency upgrade was part of
this port. Cross-version timing therefore measures these installed stacks and
cannot isolate implementation cost from dependency effects.

CLP uncertainties are local linearized estimates, opt-in, and restricted to
variable projection. They do not establish convergence, identifiability, or
confidence coverage for a singular fit. Full-model CLP penalties remain outside
the pre-existing global objective; the port does not claim to add that feature.
Scale/IRF lists bind by global-coordinate position and must be reordered with
input coordinates. See [usage](USAGE.md) and the individual feature contracts.

## Common notebook and runtime acceptance

Both fresh common runners pass 11/11 notebooks. The manifest-driven comparison
accepts all 14 leaves: 8 PASS, 6 documented EXPECTED_DIFFERENCE, no REGRESSION,
BASELINE_FAILURE or missing artifacts. No tolerances were changed. Source, lock,
examples and result tree hashes were independently checked with zero mismatches.

Workspace artifacts:

- `validation/runs/main/scale-list-20260922-015900/manifest.json`
- `validation/runs/staging/scale-list-20260922-023731/manifest.json`
- `validation/comparisons/v07-v08-scale-list-20260922-023731.json`

Spectral guidance remains 1.25777e-6 against its documented 2e-6 limit; transient
two-dataset drift is 8.81932e-6 against 2e-5. Existing secondary decomposition and
identifiability qualifications still apply. This is acceptance of the established
contract, not an unconditional merge or scientific-convergence sign-off.

The separate standard benchmark completed successfully: 12 workers, 150 timed
samples, 15 fit summaries, `REPORT_ONLY`. Each branch ran one warm-up and five
timed repetitions in fresh processes with one thread; branch order alternated.
Every worker used the same branch-specific core and lockfile hashes.

- Raw: `validation/benchmarks/raw/v07-v08-scale-list-20260922-024526/`
- Report: `validation/benchmarks/v07-v08-scale-list-20260922-024526/runtime.json`
- Manifest SHA-256: `0fa26a041487bf135cd07f0e9cca11ffb02bc8fc5820854b6df838179e23fca2`

Evaluation counts match for 14/15 fits. Spectral guidance uses 23 reference versus
20 staging evaluations and is not an implementation-performance comparison.
Matching-workload staging/main mean-runtime ratios span 0.723–2.840. These are
observations on different installed scientific stacks, not an isolated causal
measurement of this port or a release gate. The full JSON/CSV report preserves
means, standard deviations, individual samples and workload warnings.

After all runs, one mixed newline in the checkout was normalized for Ruff.
The committed file content did not change; the verification artifact
`validation/runs/scale-list-port-20260922-015900/checkout-newline-normalization.json`
records both raw tree hashes and identical canonical content. All 47 changed
Python files pass final lint and formatting checks.
