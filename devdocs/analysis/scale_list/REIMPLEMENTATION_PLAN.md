# Reimplementation design plan: `scale_list` branch functionality

**Date:** 2026-08-30
**Companion document:** [BRANCH_ANALYSIS.md](BRANCH_ANALYSIS.md) (what the prototype
branch contains and where). This document specifies how to reintroduce that
functionality **properly**.

## How to read this document

This plan is deliberately **not written against the current state of `main`**. The
prototype branch (`scale_list`, head `e042a18e`) is treated as a *reference
implementation and requirements capture*, not as a patch series to be rebased. Every
feature below is specified as: the requirement, the semantic contract (the decisions
that must hold no matter what the code looks like), a design in terms of architectural
*roles*, and acceptance criteria. It can be implemented on v0.7-era `main`, on a
much-changed `staging` branch, or anywhere else that provides the same roles.

The roles, with their v0.7 names for orientation:

| Role | v0.7 incarnation | Notes for other targets |
|---|---|---|
| **Model spec layer** | `glotaran.model` items (`Model`, `DatasetModel`, `Irf`, penalties) | Any declarative model representation (e.g. pydantic-based items on staging, "elements" instead of "megacomplexes") |
| **Parameter store** | `glotaran.parameter.Parameters` / `Parameter` | Any label-addressed parameter container |
| **Validation stage** | `Model.get_issues()` / `Scheme.validate()` | Wherever model + parameters + data first meet before optimization |
| **Optimization engine** | matrix/estimation providers, `OptimizationGroup`, `Optimizer` | Whatever computes matrices, solves CLPs, and drives `least_squares` |
| **Result model & persistence** | `Result` dataclass + `save_result`/`result.yml` + per-dataset xarray outputs | Any result schema with on-disk persistence |
| **IO plugin layer** | `glotaran.io` plugins (csv/tsv/xlsx/yml/folder) | Any format-parsing boundary |

Two rules apply throughout, both learned from the prototype:

1. **Behavior is the spec; the prototype code is not.** Where the prototype's tests
   encode correct numerics (IRF closed-form comparisons, penalty arithmetic, scale
   recovery), port the *tests*; write the implementation fresh.
2. **Nothing ships without its contract:** every user-visible addition lands together
   with validation, serialization round-trip, tests, changelog entry, and docs. Every
   persisted schema addition is typed before it is persisted.

---

## Phase 0 — Foundations (design gates before the gated features)

### F0.1 First-class vector attributes in the model spec layer ◆ design gate

**Problem.** Four prototype features needed a vector-valued item attribute and the item
system supported only one vector concept (a flat, positional list of parameters). The
workarounds were: positional lists with runtime length checks in three places
(`scale_list`), a numeric vector mis-declared as parameters causing two permanently
failing tests (spline knots), duck-typed `hasattr` hooks in the core fill machinery
(nested IRF parameter lists), and an unserializable raw `np.ndarray` (`x_scale`). This
is the single highest-leverage piece of design work; features P3.1 and P3.2 are gated
on it.

**Requirement.** The model spec layer must be able to declare, on any item, attributes
of these kinds — with generic (not per-item) support in conversion, parameter filling,
validation/issue collection, markdown rendering, and serialization:

1. **Numeric vector** — plain numbers (e.g. spline knot positions). Never resolved
   against the parameter store.
2. **Nested parameter collections** — lists of lists of parameters (e.g. grouped IRF
   components). Filling and missing-label detection must traverse the declared
   structure recursively.
3. **Bound parameter vector** — a parameter vector whose elements are bound to
   something *named*, not to list position:
   - bound to a **data-axis coordinate** (e.g. one scale per global-axis point), or
   - bound to **parameter labels** (e.g. per-parameter optimizer scaling).

**Design.**

- Capability discovery must come from **declared attribute metadata** (type
  annotations / field metadata), never from `hasattr` or `isinstance` probing of
  instances. The fill/validation machinery inspects the declaration and dispatches;
  an item cannot silently intercept core behavior.
- For the coordinate-bound kind, the canonical in-memory form is a mapping
  `{coordinate_value: parameter}` (or an ordered pair of arrays with an explicit
  binding declaration). The YAML spec should accept:
  - an explicit mapping (self-describing, robust to axis subsetting/reordering), and
  - a bare list **as sugar**, resolved against the bound axis and rejected at
    validation time if lengths differ — resolution happens exactly once, at the
    stage where model and data meet (validation stage), never inside the
    optimization engine.
- Traversal is selected from the **annotation** (does it contain a parameter reference
  at a leaf?), never from probing runtime values — many ordinary string fields are not
  labels. The generic traversal must serve *every* consumer of parameter references:
  filling, missing-label issues, label iteration, parameter generation, markdown
  rendering, and serialization — partial support is how the prototype's hooks happened.
  While rewriting the label iteration, fix the pre-existing bug (present on `main`)
  where iterating an already-filled item yields a `(name, label)` tuple in place of the
  label string — with its own regression test.
- Length/coordinate validation for bound vectors is implemented **once**, generically,
  at the validation stage. The optimization engine may `assert`, but must not carry
  user-facing validation.
- Serialization: every kind must round-trip through the model/scheme YAML
  representation. Raw `np.ndarray` never appears in a serializable spec object.

**Acceptance criteria.**
- A test item declaring each of the three kinds round-trips YAML → item → filled item →
  YAML, renders in markdown, and reports missing parameter labels through the standard
  issue mechanism.
- A numeric-literal list in YAML loads into a numeric vector without touching the
  parameter store (this is precisely the prototype's failing-test scenario).
- No `hasattr`-style protocol remains in the fill/validation path.

### F0.2 Persisted-schema policy for `Result` additions

**Problem.** The prototype added three persisted result keys, one of them an untyped
`list[list[dict]]` whose schema lives in a docstring, with no versioning and no story
for older readers.

**Requirement / design.**
- Any new persisted result field is a **typed record** (dataclass/attrs/pydantic —
  whatever the target uses), with the type owning the serialized shape.
- The persisted result carries a schema/version marker; loaders ignore unknown keys
  with a warning (forward tolerance) and fill absent keys with defaults (backward
  tolerance). If the target branch already has a versioning mechanism, use it; if not,
  introduce the marker with the first of these features.
- Decision recorded per field: persisted or in-memory-only, and why.

### F0.3 A single "additional penalty" contract in the optimization engine

**Problem.** The prototype has two penalty families computed at two different layers
(equal-area CLP penalties inside per-group estimation; parameter penalties at the
optimizer), with subtly different conventions that had to be reconciled after the fact
(weight vs √weight), and result diagnostics captured at the wrong moment (stale values
from the last *trial* evaluation instead of the optimum).

**Requirement / design.** Define one engine-level contract for anything that appends
extra residual terms to the objective:

- A penalty contributor declares **where it is evaluated** (per dataset group with CLP
  access, or globally with parameter access only) and returns `(residual_terms,
  diagnostics)` where diagnostics is a typed record (F0.2).
- **Constant length:** for a given model, the number of contributed residual terms must
  not depend on parameter values (degenerate cases clamp with a warning, never skip —
  the prototype got this right and it must be preserved).
- **Weight convention:** residual terms are `weight * term`, linearly, everywhere.
  (Contribution to χ² is therefore `(weight * term)²`.) Document this once.
- **Reporting moment:** the result assembly performs the final objective evaluation at
  the optimized parameters *first*, then captures penalty values/diagnostics — a
  defined ordering, tested, so stale-capture bugs are structurally impossible.
- Zero/degeneracy guards use **relative** tolerances scaled to the operands, not
  absolute `atol=1e-8` defaults.

**Acceptance criteria.** A test asserts that (a) penalty terms appear in the final
objective vector exactly once with the documented weight convention, (b) reported
diagnostics match a recomputation from the optimized parameters, and (c) the residual
vector length is constant across parameter values including the degenerate regime.

---

## Phase 1 — Low-risk features (no gate; can start immediately, in any order)

### P1.1 Parameter IO & validation package

**Requirement.** (a) Optimized-parameter exports include a derived `T-value` column
(`value / standard_error`), with standard errors blanked for non-varied parameters;
derived columns are ignored on load. (b) Duplicate parameter labels in a loaded file
are an error. (c) Unknown columns warn instead of vanishing silently. (d) Initial
values outside `[minimum, maximum]` are reported by validation. (e) Scheme-level
validation is advisory by default with opt-in raising.

**Design.** The prototype's *placement* is correct and should be replicated on any
target:
- Derived-column production is an opt-in flag on the dataframe export
  (`as_optimized=...`), threaded through the format plugins as a keyword with default
  `False`; the result-saving path opts in. Loaders drop derived columns.
- Duplicate-label detection is **one shared uniqueness check applied before any
  constructor collapses rows into a label-keyed dictionary** — the dataframe boundary
  covers the file formats, but `from_list`/`from_dict`/`from_parameter_dict_list` can
  silently overwrite duplicates just as well and must go through the same check. Not
  per format plugin, and not anywhere that re-parses files.
- Bounds checking lives in the model/parameter validation stage so every entry point
  benefits — and distinguishes **definition issues** (any parameter outside its bounds)
  from **optimization preflight** (only `vary: true` parameters make the optimizer's
  initial vector infeasible; fixed/expression parameters cannot).
- Raising behavior reuses the target's existing validation convention (one
  `raise_exception`-style contract and error type) rather than introducing a second,
  parallel strict-mode spelling.
- Known-column and derived-column sets are single-sourced constants on the parameter
  store; **comparisons are case-insensitive** (the prototype's `t-value`/`T-value`
  mismatch is the regression test).

**Acceptance criteria.** Tests for: duplicate-label error (message lists each label and
count), unknown-column warning, case-insensitive derived-column acceptance on an
in-memory `from_dataframe(to_dataframe(as_optimized=True))` round-trip, bounds issue
text, and a file round-trip in every supported parameter format. Changelog entries for
both behavior changes (duplicates raise; unknown columns warn).

### P1.2 Equal-area penalty: `relative` mode + typed area diagnostics

**Requirement.** The equal-area CLP penalty gains `relative: bool = False`. Absolute
mode (default) is unchanged: `|source_area − parameter·target_area|`. Relative mode:
`source_area/(parameter·target_area) − 1`, with a clamp-to-absolute fallback (plus
warning) when the denominator is degenerate. Users get a per-penalty diagnostic
breakdown (areas, intervals, parameter, weight, resulting penalty value) on the result.

**Design.**
- `relative` is one boolean on the existing penalty item; spec default preserves
  existing behavior byte-for-byte.
- The diagnostics are a typed record, e.g. `EqualAreaPenaltyDiagnostic(source,
  source_intervals, source_area, target, target_intervals, target_area, parameter,
  relative, weight, penalty)`, produced under the F0.3 contract and persisted per F0.2.
- Behavior-affecting nature of `relative` (it changes the optimization landscape;
  weight choice is the user's) goes in docstring + changelog, as the prototype already
  did — keep that text.

**Acceptance criteria.** Parametrized test over `relative` verifying the penalty value
against hand-computed areas; save/load round-trip of a model with `relative: true`;
result round-trip of the diagnostics; diagnostics recomputed at the optimum match the
stored ones (F0.3c).

### P1.3 Parameter penalties (`EqualParameterPenalty`)

**Requirement.** A new model-level item category `parameter_penalties` for soft
constraints between *parameters* (independent of CLPs). First type `equal`: encourages
`source ≈ parameter · target` via the two symmetric ratio residuals
`weight·(source/(parameter·target) − 1)` and `weight·((parameter·target)/source − 1)`.

**Design.**
- Model layer: a typed item registered through the target's standard item machinery —
  the prototype's model-layer half is the template (it needed zero special cases).
- Engine: a **global** penalty contributor under F0.3 — evaluated once per objective
  evaluation (never per dataset group), from model + parameters only.
- Degenerate guard: relative tolerance; fallback residual keeps vector length constant;
  document that the fallback is scale-inconsistent by nature and warn once per run,
  not per evaluation.
- Result: reported value captured per F0.3's ordering rule; decide persisted vs
  in-memory explicitly (recommendation: in-memory + markdown report only, like
  `additional_penalty`; revisit if users need it on disk).

**Acceptance criteria.** Arithmetic test against hand-computed values; a wiring test
asserting the terms appear in the *optimizer's* final objective vector (the prototype
lost this assertion); an end-to-end fit where the residual count equals
`n_data + n_clp_penalties + 2·n_parameter_penalties`; YAML round-trip; changelog.

### P1.4 `x_scale` — optimizer parameter scaling

**Requirement.** Users can enable scipy's parameter scaling. The immediately safe
subset: `x_scale: float | Literal["jac"]` on the scheme, validated at construction,
serialized like every other scheme field, forwarded to `least_squares`.

**Design.**
- **Do not** expose the raw per-parameter array form on the scheme: it binds to an
  internal free-parameter ordering that users cannot see, and an ndarray does not
  belong in the scheme schema. Narrow the type; validate the string value.
- If per-parameter scaling is wanted (it is scientifically legitimate here), implement
  it as **parameter metadata** — label-keyed, next to `minimum`/`maximum`/`vary` in
  the parameter store and parameter files — assembled by the optimizer into the scipy
  vector in free-parameter order. That is F0.1 kind 3b territory and can follow later
  without breaking the scalar/`"jac"` forms.

**Acceptance criteria.** Scheme YAML round-trip with both forms; invalid string
rejected at validation with a clear message; one fit demonstrating `"jac"` runs;
changelog.

### P1.5 Small independent fixes (each its own tiny PR)

| Item | Contract |
|---|---|
| Full-model residual orientation fix | Regression test that full-model residuals reshape as `(global, model)ᵀ`; pure bug fix. |
| NNLS `maxiter` | Named constant/argument with a comment explaining why scipy's default `3·n` is insufficient for the observed problems; keep the existing mock test, add a real ill-conditioned case if available. |
| Coherent-artifact order 4–5 | Extend limit with the 4th/5th-derivative terms; add a numeric test comparing against analytic Hermite-polynomial forms for orders 4 and 5. |
| Compact k-matrix markdown numbers | Cosmetic; port formatting helper + its tests. |
| Suppress dataset saving via saving options | Give it explicit semantics (e.g. a dedicated `save_data: bool` option) instead of overloading "empty `data_filter` list means skip"; document. |

---

## Phase 2 — Concept sound, engineering rework required

### P2.1 CLP standard errors

**Requirement.** Opt-in post-fit standard errors for conditionally linear parameters:
`Var_total(clp) = Var_linear(clp | θ*) + J_clp,θ · Cov(θ) · J_clp,θᵀ`, where the linear
term comes from the per-index reduced design matrices (expanded through CLP
relations/constraints) and the propagation term from finite differences of the CLPs
w.r.t. the free nonlinear parameters against `Cov(θ) = RMSE²·(JᵀJ)⁻¹`. Output: a
`clp_standard_error` array per dataset with the same dims/coords as `clp`, plus method
metadata on the result. Default off; zero effect on optimization when off.

The prototype's own design notes
(`docs/source/notebooks/clp_standard_error/clp_standard_error_design.md` on the branch)
are a valid statement of the math and user API — carry them forward. The rework is
purely structural:

**Design.**
1. **No private cross-module access.** The computation consumes only a public,
   engine-defined interface: per-index reduced matrices + CLP labels, residuals, the
   relation/constraint expansion, and the aligned/unlinked topology. Whatever the
   target calls its optimization group, that object exposes these as accessors; the
   standard-error module imports no provider internals.
2. **No state mutation.** Replace "recalculate the group at perturbed θ and restore at
   the end" with a **side-effect-free evaluation**: either a pure
   `evaluate_at(parameters) -> snapshot` on the engine (preferred — broadly useful,
   e.g. for future profile-likelihood work), or, minimally, a context manager that
   guarantees restoration on any exit path. An exception mid-computation must leave
   the converged state intact.
3. Applicability is validated, not silently skipped: enabling the feature with an
   unsupported residual function is a validation-stage issue (advisory or error per
   the target's convention), with the runtime warning kept as a backstop.
4. Settings and result metadata as in the prototype (`compute_clp_standard_error`,
   relative FD step, method string); persisted metadata per F0.2. Cost note in docs:
   worst case ≈ one extra Jacobian evaluation.

**Acceptance criteria.** End-to-end test (unlinked + linked) with finite, plausible
errors; a test that a simulated dataset with known noise yields standard errors of the
right magnitude; a test that an exception injected during the FD loop leaves the group
state at the optimum; result round-trip of metadata; docs page from the existing
design notes + notebook.

### P2.2 Single-amplitude full model → an explicit composition mode

**Requirement.** For datasets modeled with both a model-axis and a global-axis matrix,
support a second composition rule: instead of the Cartesian product of component pairs
(one amplitude per pair), the **paired/diagonal** rule — component `i` of the model
matrix couples only with component `i` of the global matrix, one amplitude per pair —
used for e.g. spectrotemporal models with a single amplitude per species.

**Design.**
- Name the concept: a dataset-model attribute such as
  `global_composition: "cartesian" | "paired"` (default `"cartesian"`), replacing the
  prototype's `single_amplitude_model: bool`. A mode with different CLP labels, degrees
  of freedom, and result shape deserves an enum, not a flag riding on a side channel.
- One **central full-matrix builder** owns both modes and validates pairability for
  `"paired"` (equal component counts; defined label pairing). CLP labels for paired
  mode are defined explicitly (e.g. the global component labels), rather than
  materializing a dense `np.diag` matrix to satisfy a Cartesian-shaped result: the
  result CLP container for paired mode is the amplitude vector with its own dimension,
  with the diagonal matrix at most a derived convenience.
- The degrees-of-freedom / `number_of_clps` accounting changes **in the same commit**
  as the matrix shape (the statistics coupling the prototype handled correctly — keep
  a test that locks the two together).

**Acceptance criteria.** Simulate→optimize round-trip parametrized over both modes,
recovering known amplitudes; explicit test for the χ²/DoF statistics in paired mode;
spec round-trip; validation error for non-pairable component spaces; changelog +
docs section explaining when to use which mode.

---

## Phase 3 — Design-gated features (require F0.1)

### P3.1 Per-global-index dataset scaling (`scale_list`) ◆ semantics gate

**Requirement.** When jointly fitting datasets (TCSPC use case), each global-axis point
of a dataset can carry its own scale parameter; the effective matrix at index `i` is
`scale · scale_per_index[i] · A_i` in both linked and unlinked optimization.

**Semantic contract to settle before implementation** (this is the design gate — the
prototype left all four open):

1. **Binding.** The scales are an F0.1 coordinate-bound parameter vector on the
   dataset model (canonical mapping form; list form as validated sugar). Reordering
   or subsetting the global axis must either keep the association (mapping form) or
   fail validation (list form) — never silently rebind.
2. **Identifiability / gauge.** Free per-index scales are absorbed by freely estimated
   CLPs (`s·A·(c/s) = A·c`). The contract: validation emits an issue unless the gauge
   is pinned — at least one scale fixed (`vary: false`), or the dataset's CLPs are
   constrained through linking/guidance spectra. (A normalization convention like
   "mean = 1" may be offered later; the *check* is the requirement now.) Document the
   degeneracy prominently.
3. **Path semantics.** Decide once whether dataset scaling is **generative** (applied
   identically in simulation, fitting, and the full-model path — recommended, and the
   opportunity to fix the pre-existing inconsistency where scalar `scale` is also
   ignored by `simulate()`) or **fit-only normalization** (then name it accordingly
   and *reject* it at validation time in paths that do not honor it). Silent
   ignoring in some paths is not an option.
4. **Result representation.** Recommendation: keep the stored `matrix` raw and expose
   the scaling as a coordinate-aligned data variable (e.g. `dataset_scale` over the
   global dimension, absorbing today's scalar `dataset_scale` attribute as its
   uniform special case) — lossless, backward compatible, and derivable either way.
   If the effective (pre-scaled) matrix is stored instead, that is a documented,
   changelogged behavior change that also affects existing scalar-`scale` users.

**Design.** With F0.1 in place the implementation is small: resolution + length/
coordinate validation happens once at the validation stage; the optimization engine
receives an already-validated per-index scale vector and applies it at matrix
preparation in both providers; result assembly reads the same resolved vector. No
runtime `ValueError`s in providers, no triple duplication.

**Acceptance criteria.** End-to-end recovery test (per-index scales fixed, rates
recovered — port the prototype's test), linked and unlinked; gauge-violation
validation test (all scales free ⇒ issue); coordinate-binding test (axis subset ⇒
mapping still correct / list rejected); simulation honors the chosen path semantics;
result round-trip of the scale representation; changelog.

### P3.2 IRF extensions: grouped gaussians, convolution broadening, width dispersion

**Requirement** (three user-facing capabilities, currently entangled in one inheritance
tower plus ad-hoc attributes):

1. **Grouped gaussian IRFs:** an IRF composed of groups of gaussians; within a group,
   centers are offsets relative to the group's first center and scales are factors
   relative to the group's first scale; optional normalization of the total area to a
   target value.
2. **Per-index convolution broadening:** an additional gaussian broadening (laser
   pulse / detector response) with a per-global-index width, applied in quadrature:
   `w_eff = sqrt(convwidth[i]² + w²)`; normalization policy must be *explicit* about
   whether it uses pre- or post-broadening widths.
3. **Width dispersion extensions** for the spectral IRF: in addition to the existing
   polynomial, (a) a cubic-spline term defined by numeric knot positions + parameter
   values (knots optionally given in wavelength and converted for wavenumber-domain
   models), and (b) a skewed-gaussian correction term (amplitude, location, width,
   skewness — all four or none).

**Design.**
- **Composition over inheritance.** One gaussian-mixture IRF evaluator; the
  capabilities above are declared parts, not subclasses:
  - grouped centers/scales are F0.1 nested parameter collections, expanded to a flat
    mixture by the IRF itself through declared metadata (no core hooks);
  - broadening is a modifier with a per-index bound parameter vector (F0.1 kind 3a,
    same binding concept as `shift` and P3.1);
  - normalization is an explicit step with a declared policy —
    `normalize_area: bool`, `normalization_after_broadening: bool`, target area with a
    justified default — instead of encoding the policy in the *type name*
    (`norm-conv-multi-multi-gaussian`) and re-normalizing in a grandchild override.
  - width dispersion is a list of dispersion terms (polynomial | spline |
    skewed-gaussian) evaluated additively; spline knots are an F0.1 numeric vector
    (this retires the prototype's two permanently-failing tests at the root cause);
    the spline object is constructed once per fill, not once per axis point.
- The decay/matrix code consumes only the evaluator interface (flat centers, widths,
  scales per index + normalization already applied). No `isinstance` checks on IRF
  subtypes outside the IRF module. Two contracts the prototype broke and this design
  must state explicitly: **index-dependence is derived from the declared parts** (a
  per-index broadening or shift vector ⇒ index-dependent — the prototype's `convwidth`
  without `shift` crashes on the index-independent path), and **the matrix builder
  consumes per-index scale vectors** (the prototype's builder keeps only the last
  index's scales, silently voiding per-index normalization).
- **Spec compatibility:** if the prototype's YAML type names are already in use in the
  field (`multi-multi-gaussian`, `conv-multi-multi-gaussian`,
  `norm-conv-multi-multi-gaussian`), keep them loadable as aliases that map onto the
  composed representation, and document the preferred new spelling.
- Decide explicitly whether `dispersion_center` on the spectral IRF is required or
  optional (the prototype relaxed it silently); whichever is chosen, validation states
  which dispersion terms require it.

**Acceptance criteria.** Port the prototype's closed-form tests (group expansion
arithmetic, area normalization with and without broadening, quadrature broadening,
polynomial+spline+skewed-gaussian width composition, wavelength→wavenumber knot
conversion) — they encode the physics and are the best artifact on the branch. Add:
YAML round-trip for every capability including numeric knots; a performance sanity
check that spline construction does not scale with axis length; validation tests for
partial skewed-gaussian parameter sets and non-monotonic knots; a broadening-without-
shift model that builds a matrix end-to-end (the prototype's crash case); and an
**end-to-end** matrix test that per-index normalized scales actually differ per index
in the built matrix (the prototype's unit tests passed while the built matrix was
wrong — direct `parameter()` tests alone are insufficient).

---

## Cross-cutting requirements

**Testing.** Each feature's acceptance criteria above; plus, for every new spec key, a
save→load→save stability test (the prototype's golden-file drift — `test_save_model`,
`test_save_scheme` — must fail *in the same PR* as the schema change, not after merge).

**Compatibility & migration.**
- New model/scheme spec keys: additive with defaults preserving current behavior.
- Persisted result additions: per F0.2 (typed + version-tolerant loaders).
- Behavior changes shipped by this program (duplicate labels raise; unknown parameter
  columns warn; scaling semantics per P3.1 decision 3/4; NNLS iteration limit): each
  gets a changelog entry under a behavior-affecting heading, and where feasible a
  deprecation/warning release before the change becomes an error.

**Documentation.** Each phase-2/3 feature gets a user-docs section with a worked
example (the TCSPC scale/IRF features together form a natural tutorial case); CLP-SE
reuses the branch's design notes and notebook.

**Suggested PR sequence.** P1.5 fixes (immediately) → P1.1 → P1.2 → P1.3 + P1.4 (can
share a PR) → **F0.2/F0.3** (small, unblock correctness of P1.2/P1.3 reporting) →
P2.1 → P2.2 → **F0.1 design RFC + implementation** → P3.1 (after its semantics gate)
→ P3.2. Phase-1 items are independent and parallelizable; nothing in Phase 3 starts
before F0.1 is agreed, and the F0.1 RFC discussion should include whoever maintains
the target branch's item/spec system, since it must decide what else traverses the
same structure (markdown rendering, YAML round-trip, parameter generation, issue
collection).

**Explicitly not carried over from the prototype:** the `hasattr` hooks in the core
item machinery; the raw `np.ndarray` scheme field; per-provider runtime length
validation; the untyped persisted dict schema; `uv.lock`; and the
capture-before-final-evaluation result assembly ordering.
