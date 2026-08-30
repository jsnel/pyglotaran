# Branch analysis: functionality added on `scale_list` relative to `main`

**Date:** 2026-08-30
**Branch head:** `e042a18e` ("architectural improvements")
**Compared against:** `main` at `2dfe53a2` (v0.7.4 + pandas-3 fix + SVD/simulation dimensionality fix)
**Effective delta:** +2,523 / −81 lines across 41 files (11 effective commits; two further
commits on the branch are cherry-picks of fixes already on `main` and are excluded here).

This document records **what** functionality the branch adds, **where** it lives, what its
**API impact** is, and an assessment of the implementation quality. The companion document
[REIMPLEMENTATION_PLAN.md](REIMPLEMENTATION_PLAN.md) specifies how to reintroduce the
functionality properly, independent of the exact state of `main`.

---

## 1. Context: how this branch grew

The branch forked from the v0.7.4 release commit and contains two distinct clusters:

1. **CLP standard-error cluster** (Jan–Mar 2026): CLP standard errors, the
   `single_amplitude_model` flag, and parameter-IO polish (T-values). Mixed authorship;
   this cluster has its own design notes at
   `docs/source/notebooks/clp_standard_error/clp_standard_error_design.md`.
2. **TCSPC experimentation cluster** (Apr–Aug 2026, Ivo van Stokkum): `scale_list`, the
   multi-multi-gaussian IRF family, spline/skewed-gaussian width dispersion, parameter
   penalties, the equal-area penalty rework, `x_scale`, and validation improvements.

The final commit `e042a18e` was itself a cleanup pass responding to an earlier internal
review: it relocated the parameter-penalty computation to the optimizer, removed CSV
re-parsing from `Scheme.validate()`, moved validation checks to their proper layers, and
added changelog discipline for the equal-area work. In other words, part of the cleanup
has already happened; what remains is the harder design work.

**Supersedes.** This document and the companion plan replace the earlier working
documents that circulated during the branch review: `scale_list_architecture_analysis.md`
(revisions of 2026-07-11 and 2026-08-30, plus their HTML renderings), the
`parameter penalty.md` session log, and the two `*.prompt.md` implementation/page
prompts (`multi_multi_gaussian_and_parameter_validation.prompt.md`,
`scale_list_architecture_analysis_page.prompt.md`). All of their still-valid findings
have been re-verified against the code and folded in here; the unique corrections from
the multi-multi-gaussian plan appear as defects 15–17 in §4.2 and as refinements in the
reimplementation plan. The architecture guide at
`devdocs/architecture/v0.7.4/architecture.md` (the consolidated survivor of several
generated variants) is **not** superseded — it describes the codebase itself, not this
branch.

**Current test status (verified on this machine, 2026-08-30):** the branch ships red.
`test_save_model` fails (golden model YAML lacks `parameter_penalties: []`),
`test_save_scheme` fails twice (golden scheme YAML lacks the three new `Scheme` fields),
and both `test_spectral_irf_spline_width_dispersion*` tests fail with
`AttributeError: 'ScalarFloat' object has no attribute 'label'` (see §4.3).

---

## 2. Feature inventory: the what and the where

### 2.1 `scale_list` — per-global-index dataset scaling

**What.** `DatasetModel` gains `scale_list: list[ParameterType] | None` — one scale
parameter per global-axis point. Use case: joint fitting of TCSPC datasets where each
decay curve needs its own scaling factor. During optimization, the prepared matrix at
global index `i` is multiplied by `scale * scale_list[i]` (both the unlinked and the
linked path). Result datasets gain a `dataset_scale_list` attribute, and the stored result
`matrix` is multiplied by the effective scale at result-creation time.

**Where.**
- `glotaran/model/dataset_model.py` — the new attribute
- `glotaran/optimization/matrix_provider.py` — scaling in `MatrixProviderUnlinked.prepare_matrices()` and `MatrixProviderLinked.align_matrices()` loop
- `glotaran/optimization/optimization_group.py` — `_scale_result_matrix()`, `dataset_scale_list` attribute
- `glotaran/optimization/test/suites.py`, `test_optimization.py` — one end-to-end test (linked and unlinked)

**Spec surface added:** `dataset.<label>.scale_list: [param, ...]` in model YAML.

**Assessment.** The need is real and the arithmetic is correct (the end-to-end test
recovers rates and verifies the stored matrix is scaled exactly once). But:

- Length-vs-axis validation is a runtime `ValueError` duplicated in **three** places (two
  providers plus result assembly), because the model layer never sees the data axis. It
  belongs once, at the scheme/validation level where model and data meet.
- The list binds **by position, not by coordinate**. Subsetting or reordering the global
  axis silently changes what `scale_list[i]` means; no validation can catch it.
- Semantics are undefined across execution paths: `simulate()` and the full-model path
  apply neither `scale_list` nor the pre-existing scalar `scale` (the test works around
  this by dividing the simulated data manually).
- Free per-index scales are unidentifiable in general (`s·A·(c/s) = A·c` — any per-index
  scale is absorbed by the freely estimated CLPs) unless a linked dataset or a
  normalization convention pins the gauge. Nothing surfaces this to the user.
- The result `matrix` is now stored pre-scaled — a behavior change that also affects
  existing scalar-`scale` users, undocumented.

### 2.2 `single_amplitude_model` — a structural variant of the full (global) model

**What.** `DatasetModel` gains `single_amplitude_model: bool = False`. For a dataset with
a global megacomplex, the full matrix is built from the *diagonal* pairs only
(`kron(global_matrix[:, i], matrix[:, i])` per column `i` instead of the Cartesian product
of all column pairs), `number_of_clps` becomes `len(global_clp_labels)`, and result CLPs
are reshaped via `np.diag`.

**Where.**
- `glotaran/model/dataset_model.py` — flag + `is_dataset_single_amplitude_model()`
- `glotaran/optimization/matrix_provider.py` — `create_full_matrices()`, `number_of_clps`
- `glotaran/optimization/estimation_provider.py` — `np.diag` reshape in `get_result()`

**Spec surface added:** `dataset.<label>.single_amplitude_model: true`.

**Assessment.** Both sides of the statistics coupling (`number_of_clps` vs matrix width)
were updated consistently — good. But the flag has **no end-to-end optimization test**
(it appears only in spec/serialization comparisons), and a boolean understates what it
is: a different *composition algebra* (paired columns vs Cartesian product) that changes
CLP labels, degrees of freedom, result shape and uncertainty. The `np.diag` reshape is a
presentation repair for a structural concept the code never names.

### 2.3 CLP standard errors — a post-fit uncertainty stage

**What.** Opt-in computation of per-CLP standard errors after a successful fit:
linear variance (from the pseudo-inverse of the reduced design matrices, expanded through
relations/constraints) plus non-linear propagation (finite differences of the CLPs with
respect to the free parameters against `Cov(θ) = RMSE² · (JᵀJ)⁻¹`). Results land as a
`clp_standard_error` variable per dataset (same dims/coords as `clp`) with method
metadata in attrs.

**Where.**
- `glotaran/optimization/clp_standard_error.py` — new module (~293 lines)
- `glotaran/project/scheme.py` — `compute_clp_standard_error: bool = False`,
  `clp_standard_error_finite_difference_relative_step: float = 1e-6`
- `glotaran/project/result.py` — `clp_standard_error_method`,
  `clp_standard_error_finite_difference_relative_step`
- `glotaran/optimization/optimizer.py` — post-fit orchestration in `create_result()`
- `glotaran/optimization/optimization_group.py` — `create_result_data(clp_standard_error=...)`
- Design notes + demo notebook under `docs/source/notebooks/clp_standard_error/`

**Assessment.** The statistical approach is sound, opt-in (default off), documented in
its own design doc, and covered by an end-to-end test. The engineering has two real
problems:

1. **Coupling:** the module reaches into private state of all three provider families
   (`optimization_group._estimation_provider`, `._matrix_provider`, `._data_provider`,
   `estimation_provider._residuals`, `._residual_function`) instead of going through
   `OptimizationGroup` accessors. Any provider refactor breaks it silently.
2. **State mutation:** the finite-difference pass calls
   `optimization_group.calculate(perturbed)` repeatedly and restores the optimum only at
   the end (`optimization_group.calculate(parameters)` as the last statement). An
   exception mid-computation leaves the group — and any result built from it — evaluated
   at a perturbed parameter set.

Also restricted to `variable_projection` (silently skipping with a warning otherwise),
which is fine but should be a validated, documented constraint.

### 2.4 Parameter penalties — `EqualParameterPenalty`

**What.** A new model item category `parameter_penalties` with one concrete type,
`EqualParameterPenalty` (`type: equal`): softly ties two parameters together via two
residual terms `weight * (source/(parameter·target) − 1)` and
`weight * ((parameter·target)/source − 1)`, appended to the objective vector once per
evaluation. Near-zero values fall back to an absolute-difference residual (with a
warning) so the residual-vector length stays constant. Reported in `Result` as
`additional_parameter_penalty` (not persisted) and in the result markdown.

**Where.**
- `glotaran/model/parameter_penalties.py` — new module; exported from `glotaran/model/__init__.py`
- `glotaran/model/model.py` — `parameter_penalties` model attribute
- `glotaran/optimization/optimizer.py` — `calculate_parameter_penalties()`, wiring into `calculate_penalty()` and `create_result()`
- `glotaran/project/result.py` — field + markdown row
- `glotaran/optimization/test/test_penalties.py` — arithmetic test

**Spec surface added:** top-level `parameter_penalties:` block in model YAML.

**Assessment.** The model-layer half is exemplary — a typed item, registered through the
existing `_global_item_attribute` machinery, so parsing, validation, markdown and
serialization all come for free. The computation correctly lives at the optimizer level
(global to the model, not per dataset group; this was fixed in `e042a18e`). Residual
defects:

- **Stale result value:** `create_result()` captures `additional_parameter_penalty`
  (and `additional_penalty`/`additional_penalty_areas`) *before* the final
  `calculate_penalty()` re-evaluates at the optimum and rebinds `_parameter_penalty` —
  so the stored value comes from the last *trial* evaluation, which for a rejected step
  is not the optimum.
- The zero-guard uses `np.isclose(x, 0.0)` — an absolute `atol=1e-8` that will capture
  legitimately small rate constants depending on the time unit; and the fallback residual
  is dimensionally inconsistent with the ratio form (and tends to zero exactly in the
  degenerate regime it guards, giving the optimizer no escape gradient).
- The test exercises the arithmetic but no longer asserts that the terms actually reach
  the optimizer's full penalty vector (the plumbing is untested).

### 2.5 Equal-area penalty rework — `relative` mode + persisted area diagnostics

**What.** `EqualAreaPenalty` gains `relative: bool = False`. When true, the penalty is
`source_area/(parameter·target_area) − 1` instead of the absolute
`|source_area − parameter·target_area|`. Alongside, every equal-area penalty evaluation
records an area-breakdown dict (source/target areas, intervals, parameter, weight,
resulting penalty), surfaced as `OptimizationGroup.get_additional_penalty_areas()` and
**persisted** in `result.yml` as `Result.additional_penalty_areas`.

**Where.**
- `glotaran/model/clp_penalties.py` — `relative` field + docstring
- `glotaran/optimization/estimation_provider.py` — `calculate_clp_penalties()` returns `(penalties, areas)`
- `glotaran/optimization/optimization_group.py`, `optimizer.py`, `project/result.py`
- `changelog.md` — documented under a "Behavior-affecting changes" heading (the only feature with a changelog entry)

**Assessment.** Deliberate, documented, tested (parametrized over `relative`), and the
persistence decision was made explicitly. Weakness: the persisted breakdown is an untyped
`list[list[dict]]` whose schema exists only in a docstring — the moment it is persisted
it becomes a compatibility commitment, and it should be a typed record before that
happens. Also the same capture-before-final-evaluation staleness as §2.4 applies in the
linked path.

### 2.6 `x_scale` — scipy parameter scaling exposed on `Scheme`

**What.** `Scheme` gains `x_scale: float | str | np.ndarray = 1.0`, forwarded verbatim to
`scipy.optimize.least_squares(..., x_scale=...)`.

**Where.** `glotaran/project/scheme.py` (one field), `glotaran/optimization/optimizer.py`
(one argument, via `getattr(self._scheme, "x_scale", 1.0)`).

**Assessment.** The need is genuine (rate constants and IRF widths spanning orders of
magnitude are routine in this domain), but all three declared forms have problems:

- `np.ndarray` **cannot be serialized** — `save_scheme` runs fields through `asdict` into
  ruamel YAML, which raises on ndarrays. A scheme using the array form cannot be saved.
- The array form binds positionally to `Optimizer._free_parameter_labels` — an internal
  ordering the user never declares, cannot see, and that has no length check.
- The string form is unvalidated (scipy accepts exactly `"jac"`; typos surface as scipy
  errors mid-fit).
- The `getattr` defends against absence of a field declared on the very class being read.

No test, no changelog, no docs; one of the causes of the failing `test_save_scheme`.

### 2.7 Multi-multi-gaussian IRF family + core item-system hooks

**What.** Three new IRF types for TCSPC:

- `multi-multi-gaussian` (`IrfMultiMultiGaussian`): `center`/`width`/`scale` become
  `list[list[ParameterType]]` — groups of gaussians where within each sublist centers are
  additive relative to the first element and scales multiplicative; optional area
  normalization to a target `normarea` (default `1000.0`).
- `conv-multi-multi-gaussian`: adds a per-global-index `convwidth` list; widths broadened
  in quadrature `sqrt(convwidth² + width²)` (models e.g. detector response).
- `norm-conv-multi-multi-gaussian`: same, but area-normalizes on the *broadened* widths
  (kept separate to preserve backward compatibility with the middle type's
  normalize-before-broaden behavior).

Because the declarative item system cannot express nested parameter lists, the item
framework itself was extended with two duck-typed hooks:

```python
# glotaran/model/item.py
fill_item():                  if hasattr(item, "_fill_parameters"): item._fill_parameters(parameters)
get_item_parameter_issues():  if hasattr(item, "_iter_nested_parameter_labels"): ...
```

**Where.**
- `glotaran/builtin/megacomplexes/decay/irf.py` — the three classes (~245 lines)
- `glotaran/model/item.py` — the two `hasattr` hooks (**the branch's one core-framework contract change**)
- `glotaran/builtin/megacomplexes/decay/util.py` — normalization special-casing
  (`isinstance(irf, IrfMultiMultiGaussian)` checks) and nested-aware `retrieve_irf()`
- Unit tests for normalization/broadening behavior in `test_spectral_irf.py`

**Assessment.** The IRF mathematics is reasonable and unit-tested. The engineering is
not mergeable as-is:

- The `hasattr` protocol sits in the heart of the declarative→runtime transition. Any
  item anywhere can now intercept parameter filling invisibly to the type system, to
  validation, and to other plugins. It is a workaround for a missing item-system
  capability (see §4.3), not a design.
- A three-level inheritance tower where the grandchild *re-normalizes* what the
  grandparent already normalized (to fix the order of normalization vs broadening) is a
  composition problem solved with subclassing; the type names encode the workaround
  (`norm-conv-multi-multi-gaussian`).
- `isinstance` special-casing of one IRF subclass inside the decay matrix code breaks the
  polymorphism the `Irf` hierarchy exists to provide.
- `normarea: float = 1000.0` is an unexplained magic default interacting confusingly with
  the pre-existing `normalize` flag (`normalize_area` vs `normalize`).
- **`convwidth` does not trigger the index-dependent path.** `IrfConvMultiMultiGaussian`
  inherits `is_index_dependent()` (which checks only `shift`), so with `convwidth` but no
  `shift` the index-independent path calls `irf.parameter(None, global_axis)`, which is
  incompatible with indexing `convwidth[global_index]` (verified: `util.py:170` vs the
  unguarded `global_index >= len(self.convwidth)` comparison).
- **Per-index normalization is silently discarded.** The index-dependent decay matrix
  builder overwrites `irf_scales` on every loop iteration and applies only the **last
  index's** scales to all indices (verified: `util.py` `decay_matrix_implementation_index_dependent`).
  For `norm-conv-multi-multi-gaussian` — whose normalized scale vector varies per index by
  design — the end-to-end matrix is therefore wrong even though the direct
  `parameter(0, ...)` unit tests pass.

### 2.8 Spectral IRF width-dispersion extensions

**What.** `IrfSpectralMultiGaussian` (existing type `spectral-multi-gaussian`) gains two
additional width-dispersion mechanisms on top of the existing polynomial coefficients:

- a **cubic-spline** term: `width_dispersion_spline_knots` (positions),
  `width_dispersion_spline_values` (parameters), `width_dispersion_spline_knots_in_wavelength`;
- a **skewed-gaussian** correction term: four parameters (amplitude, location, width,
  skewness), all-or-nothing validated.

Additionally `dispersion_center` and `center_dispersion_coefficients` were **relaxed from
required to optional** (defaults `None` / `[]`), dispersion-distance computation was
refactored into helpers, and `is_index_dependent()` extended.

**Where.** `glotaran/builtin/megacomplexes/decay/irf.py` (~130 lines),
`test_spectral_irf.py` (extensive closed-form comparison tests),
`test_model_spec.yml`/`test_model_parser.py` (skewed-gaussian spec coverage).

**Assessment.** Well-tested in intent — the tests compare against independently computed
widths — but **the two spline tests ship failing**: the knots are numeric literals in the
YAML while the attribute is declared `list[ParameterType]`, and the item system's fill
machinery handles only label strings or `Parameter` objects, so loading crashes with
`AttributeError: 'ScalarFloat' object has no attribute 'label'`. A commented-out
`list[float]` declaration right above the field shows the author hit the limitation and
worked around it in the wrong direction (verified: both tests red). This is the clearest
evidence for the missing item-system capability described in §4.3. The spline is also
rebuilt (`CubicSpline(...)`) on every `parameter()` call — once per global-axis point per
iteration. The silent relaxation of `dispersion_center` from required to optional changes
validation semantics for an existing public type and deserves its own decision.

### 2.9 Parameter IO and validation improvements

**What.** A collection of quality improvements, largely done properly:

- `Parameters.to_dataframe(as_optimized=...)`: opt-in derived `T-value` column
  (`value/standard_error`), standard errors blanked for non-varied parameters; the
  folder result plugin saves `optimized_parameters.csv` with `as_optimized=True`; csv/
  tsv/xlsx `save_parameters` grew an `as_optimized` keyword (default `False`); loaders
  drop derived columns on read.
- `Parameters.from_dataframe`: **duplicate labels now raise** `ValueError` (previously
  silently overwrote); unknown columns **warn and are dropped** (previously silently
  dropped); `DATAFRAME_COLUMNS`/`DERIVED_PARAMETER_COLUMNS` centralized on `Parameters`.
- New `ParameterBoundsIssue`: initial values outside `[minimum, maximum]` reported
  through `Model.get_issues()` so every validation entry point benefits.
- `Scheme.validate()` is a thin delegate to `Model.validate()` with opt-in
  `raise_exception: bool = False`.

**Where.** `glotaran/parameter/parameters.py`, `glotaran/model/model.py`,
`glotaran/model/item.py` (`ParameterBoundsIssue`), `glotaran/project/scheme.py`,
`glotaran/builtin/io/pandas/{csv,tsv,xlsx}.py`, `glotaran/builtin/io/folder/folder_plugin.py`.

**Assessment.** The right checks in the right layers; the closest thing on the branch to
merge-ready. Residuals: the `T-value` whitelist is case-mismatched
(`DERIVED_PARAMETER_COLUMNS = {"t-value"}` vs the emitted `"T-value"`, so an in-memory
`from_dataframe(to_dataframe(as_optimized=True))` round-trip warns about precisely the
column the whitelist covers); the duplicate-label and bounds error paths have no tests;
and two user-visible behavior changes (duplicates raise, unknown columns warn) are
breaking for somebody's existing files and have no changelog entries. Two scope gaps:
the duplicate check guards only the dataframe boundary, while the other constructors
(`from_list`, `from_dict`, `from_parameter_dict_list`) still collapse duplicate labels
silently; and the bounds check treats all parameters alike, though only `vary: true`
parameters make the optimizer's initial vector infeasible — fixed/expression parameters
outside bounds are a different (definition-level) class of problem.

### 2.10 Small items

| Change | Where | Assessment |
|---|---|---|
| NNLS `maxiter` raised to `6 * n_cols` (scipy default is `3 *`) | `optimization/nnls.py` | Legitimate fix for NNLS non-convergence on hard problems; magic multiplier deserves a comment/constant; has a (mock-based) test. |
| Coherent artifact order extended from ≤3 to ≤5 (4th/5th derivative terms) | `coherent_artifact_megacomplex.py` | Straightforward and low risk; no test for the new orders. |
| Full-model residual reshape fix (`.T.reshape(...)` → `.reshape(...).T`) | `estimation_provider.py` | Genuine bug fix vs main for full-model residual orientation; should land with a regression test, independent of everything else. |
| Compact number formatting in k-matrix markdown | `k_matrix.py` (`format_markdown_number`) | Cosmetic, tested; fine. |
| `data_filter` falsy (e.g. `[]`) now suppresses saving result datasets entirely | `folder_plugin.py` | Useful escape hatch; surprising semantics (empty-list-means-skip); undocumented. |
| `uv.lock` (3 lines) committed | repo root | Debris; declares `requires-python >= 3.13` against the package's `>=3.10,<3.15`. Drop. |

---

## 3. API impact summary

The branch grows every public surface the project has. Each row is a compatibility
commitment if released as-is:

| Surface | Additions | Justified? |
|---|---|---|
| **Model YAML spec** | `dataset.*.scale_list`, `dataset.*.single_amplitude_model`, top-level `parameter_penalties`, `clp_penalties[].relative`, 3 new IRF types, ~10 new attributes on `spectral-multi-gaussian`, `dispersion_center` demoted to optional | Concepts yes; several shapes no (positional lists, boolean for a structural mode, nested lists via hooks) |
| **`Scheme` (= saved scheme schema)** | `compute_clp_standard_error`, `clp_standard_error_finite_difference_relative_step`, `x_scale` | First two yes; `x_scale` needs its type narrowed before it is schema |
| **`Result` (= saved result schema)** | `additional_penalty_areas` (persisted, untyped dicts), `clp_standard_error_method`, `clp_standard_error_finite_difference_relative_step` (persisted), `additional_parameter_penalty` (in-memory) | Yes, but persisted shapes must be typed and versioned first |
| **Result datasets** | `clp_standard_error` variable, `dataset_scale_list` attr, `matrix` now stored pre-scaled (**also changes behavior for existing scalar-`scale` users**) | The matrix change needs an explicit compat decision + changelog |
| **Parameter files** | `T-value` column in saved optimized parameters; duplicate labels raise; unknown columns warn | Yes; needs changelog entries |
| **Plugin interfaces** | `save_parameters(..., as_optimized=...)` on csv/tsv/xlsx plugins | Yes (keyword-only, defaulted) |
| **Core item framework** | `hasattr`-based `_fill_parameters`/`_iter_nested_parameter_labels` protocol in `fill_item()`/`get_item_parameter_issues()` | **No** — see §4.3 |
| **Python API** | `EqualParameterPenalty`, `ParameterPenalty` exported from `glotaran.model`; `Parameters.DATAFRAME_COLUMNS`/`DERIVED_PARAMETER_COLUMNS`; `Model.get_issues` bounds checking; `Scheme.validate(raise_exception=)` | Yes |

---

## 4. Quality assessment

### 4.1 What is genuinely good

- Every feature answers a real analysis need (TCSPC per-curve scaling, uncertainty
  reporting, cross-parameter soft constraints, realistic IRF shapes); this is
  domain-expert-driven scope, not speculation.
- The model-layer work (`EqualParameterPenalty`, `relative` flag) uses the existing item
  machinery correctly and gets parsing/validation/serialization for free.
- The validation relocation (§2.9) put each check in its correct layer, better than a
  naive fix would have.
- Several features have real tests, including closed-form comparisons for the IRF math
  and an end-to-end `scale_list` optimization test.
- CLP-SE has actual design documentation.

### 4.2 Defect list (all verified against the code; test failures verified by running them)

| # | Defect | Where | Severity |
|---|---|---|---|
| 1 | Two spline-dispersion tests fail on model load (`ScalarFloat` has no `.label`) | `irf.py` / `model/item.py` | Blocks merge; symptom of §4.3 |
| 2 | Golden-file drift: `test_save_model`, `test_save_scheme` ×2 fail | yml test data | Blocks merge; trivial fix |
| 3 | `Result.additional_parameter_penalty` (and linked-path `additional_penalty`/`_areas`) captured before the final penalty evaluation → stale, non-optimum values stored (and partly persisted) | `optimizer.py` `create_result()` | Wrong reported numbers |
| 4 | `x_scale: np.ndarray` breaks `save_scheme`; array form positionally bound to invisible internal ordering; string form unvalidated | `scheme.py`/`optimizer.py` | Broken feature arm |
| 5 | `hasattr` protocol added to core `fill_item()` | `model/item.py` | Architecture erosion |
| 6 | `clp_standard_error.py` reads private provider state across module boundaries and mutates group state during finite differences (unsafe on exception) | `optimization/clp_standard_error.py` | Coupling + correctness risk |
| 7 | `scale_list` length validated in 3 places at runtime; no coordinate binding; simulate/full-model silently ignore scales; identifiability unaddressed | provider layer | Spec gap |
| 8 | `np.isclose(x, 0)` absolute tolerance in penalty zero-guards captures small rate constants; fallback residual scale-inconsistent | `optimizer.py`, `estimation_provider.py` | Numeric edge case |
| 9 | Persisted `additional_penalty_areas` is an untyped `list[list[dict]]` schema | `result.py` | Schema commitment without a type |
| 10 | `single_amplitude_model` has no end-to-end test | — | Untested feature |
| 11 | `T-value` whitelist case mismatch (`t-value` vs emitted `T-value`) | `parameters.py` | Warning noise on in-memory round-trip |
| 12 | Behavior changes without changelog: duplicate labels raise, unknown columns warn, result `matrix` pre-scaled, NNLS maxiter, bounds validation | — | Release hygiene |
| 13 | `isinstance(irf, IrfMultiMultiGaussian)` special-casing in decay util; 3-level IRF inheritance with re-normalization override | decay megacomplex | Design smell |
| 14 | `uv.lock` debris with contradictory `requires-python` | repo root | Trivial |
| 15 | `convwidth` without `shift` takes the index-independent path and crashes (`is_index_dependent()` not overridden) | `irf.py`/`util.py` | Broken feature arm |
| 16 | Index-dependent matrix builder keeps only the *last* index's `irf_scales`; per-index normalization (`norm-conv-multi-multi-gaussian`) is wrong end-to-end | `decay/util.py` | Wrong numerics |
| 17 | Pre-existing (also on `main`): `iterate_names_and_labels()` yields a `(name, label)` *tuple* where a label string belongs for already-filled `Parameter` values — must be fixed by any generic traversal rework (§4.3) | `model/item.py` | Latent core bug |

### 4.3 The root cause underneath four features

Four features independently needed to declare a **vector attribute** on a model item and
the item system supports exactly one vector concept — a flat, positional
`list[ParameterType]`. Each feature worked around the same gap differently:

| Feature | Shape needed | Workaround chosen | Consequence |
|---|---|---|---|
| `scale_list` | parameter vector bound to a **data axis** | positional `list[ParameterType]` | triple runtime length checks; silent meaning change on axis reorder |
| Spline knots | **plain numeric** vector | declared `list[ParameterType]` anyway (a `list[float]` attempt is left commented out) | two tests ship failing |
| Multi-multi-gaussian IRFs | **nested** parameter lists | `hasattr` hooks in core `fill_item()` | the branch's one core-contract change |
| `x_scale` | numeric vector bound to **free-parameter order** | raw `np.ndarray` on `Scheme` | unserializable; unvalidated |

This is the highest-leverage finding of the analysis: **one missing abstraction, four
erosions**. Reintroducing these features properly starts with giving the model-item /
spec system first-class vector attribute types (plain numeric vectors, nested parameter
collections, and coordinate- or label-bound parameter vectors). That single design
decision is what [REIMPLEMENTATION_PLAN.md](REIMPLEMENTATION_PLAN.md) Phase 0 addresses;
everything else on the branch is local by comparison.

---

## 5. Conclusion

The branch contains roughly **nine separable features**, all with legitimate scientific
motivation, in three readiness tiers:

- **Nearly ready** (needs only small fixes + tests + changelog): parameter IO &
  validation (§2.9), equal-area `relative` mode (§2.5), parameter penalties (§2.4),
  a narrowed `x_scale` (§2.6), and the small fixes (§2.10).
- **Needs engineering rework, concept sound**: CLP standard errors (§2.3 — decouple and
  de-mutate), `single_amplitude_model` (§2.2 — name the mode, test it end-to-end).
- **Needs design first**: `scale_list` semantics (§2.1), the IRF family and dispersion
  vectors (§2.7/§2.8) — both gated on the vector-attribute abstraction (§4.3).

None of it should be merged from this branch as-is: the branch ships failing tests, one
core-framework erosion, three unversioned persisted-schema additions, and several
undocumented behavior changes. All of it is worth reimplementing.
