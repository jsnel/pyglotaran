## Plan: Implement CLP Standard Errors in Pyglotaran

Add optional, full CLP standard error computation to pyglotaran as a post-fit stage. Reuse the existing nonlinear covariance from optimizer Jacobian, compute CLP sensitivity to nonlinear parameters at the optimum, combine base linear LS uncertainty plus propagated nonlinear uncertainty, and store CLP-SE in result datasets with complete tests and IO support. Keep backward compatibility by defaulting this feature off.

**Steps**
1. Phase 1: Configuration and API contract.
Add scheme options in `d:/src/pyglotaran/glotaran/project/scheme.py` for CLP-SE enablement and numerical controls (flag, step strategy, step size, optional parallel FD toggle).

2. Phase 1: Result schema extension. *depends on 1*
Extend `d:/src/pyglotaran/glotaran/project/result.py` with optional CLP-SE fields and method metadata, keeping `None` defaults for compatibility.

3. Phase 2: Capture CLP solve context for uncertainty. *parallel with 2 after 1*
Expose, from providers, the data needed by CLP-SE post-processing: effective reduced matrix `X`, reduced/full CLP labels, reduction mapping induced by constraints/relations, solved CLPs, weighting/scaling context.
Files: `d:/src/pyglotaran/glotaran/optimization/matrix_provider.py`, `d:/src/pyglotaran/glotaran/optimization/estimation_provider.py`.

4. Phase 2: Implement CLP-SE core module. *depends on 3*
Create `d:/src/pyglotaran/glotaran/optimization/clp_standard_error.py` for:
- robust linear term via QR/SVD/pseudo-inverse,
- finite-difference CLP sensitivity `d(beta)/d(theta)` at optimum,
- propagated covariance using nonlinear covariance,
- reduced-to-full CLP propagation honoring constraints/relations,
- rank-deficiency handling and warnings.

5. Phase 3: Wire into optimizer result creation. *depends on 2 and 4*
Integrate post-fit CLP-SE execution in `d:/src/pyglotaran/glotaran/optimization/optimizer.py` after successful optimization and covariance calculation; guard by scheme flag and residual-function support.

6. Phase 3: Surface CLP-SE in dataset results. *depends on 5*
Update `d:/src/pyglotaran/glotaran/optimization/optimization_group.py` to write `clp_standard_error` with coords matching `clp`, plus method/settings attrs.

7. Phase 4: Serialization and load compatibility. *depends on 6*
Update `d:/src/pyglotaran/glotaran/io/` paths to persist and read CLP-SE arrays/metadata without breaking old result files.

8. Phase 4: Testing (unit + integration + IO). *parallelizable after 4/5*
Add:
- `d:/src/pyglotaran/glotaran/optimization/test/test_clp_standard_error_unit.py`
- `d:/src/pyglotaran/glotaran/optimization/test/test_clp_standard_error_integration.py`
- IO roundtrip regression tests for CLP-SE presence/absence.

9. Phase 5: Documentation and operational guidance. *depends on 7 and 8*
Document options, defaults, numerical caveats, runtime expectations, and interpretation in `d:/src/pyglotaran/docs/source/`.

10. Phase 5: Performance guardrails. *depends on 8*
Benchmark representative models and add practical warnings/fallback behavior for expensive FD or ill-conditioned cases.

**Relevant files**
- `d:/src/pyglotaran/glotaran/project/scheme.py` — new CLP-SE options.
- `d:/src/pyglotaran/glotaran/project/result.py` — result payload extension.
- `d:/src/pyglotaran/glotaran/optimization/optimizer.py` — orchestration.
- `d:/src/pyglotaran/glotaran/optimization/optimization_group.py` — dataset output fields.
- `d:/src/pyglotaran/glotaran/optimization/matrix_provider.py` — reduction metadata exposure.
- `d:/src/pyglotaran/glotaran/optimization/estimation_provider.py` — solved CLP state access.
- `d:/src/pyglotaran/glotaran/optimization/variable_projection.py` — reference solver behavior.
- `d:/src/pyglotaran/glotaran/optimization/clp_standard_error.py` — new core implementation.
- `d:/src/pyglotaran/glotaran/optimization/test/test_clp_standard_error_unit.py` — numeric core tests.
- `d:/src/pyglotaran/glotaran/optimization/test/test_clp_standard_error_integration.py` — end-to-end tests.
- `d:/src/pyglotaran/glotaran/io/` — result persistence updates.
- `d:/src/pyglotaran/docs/source/` — user docs.

**Verification**
1. Run unit tests for CLP-SE math and conditioning behavior.
2. Run integration tests across linked/unlinked, index-dependent/global, weights, constraints, relations.
3. Run existing optimization regression tests to confirm no changes when CLP-SE is disabled.
4. Validate save/load roundtrip for results with and without CLP-SE.
5. Cross-check at least one controlled case against TIMP CLP-SE within tolerance.

**Decisions**
- Included: full implementation, result exposure, IO, tests, docs.
- Included: linked and unlinked groups; constraints/relations-aware outputs.
- Included: finite-difference sensitivity as first complete method.
- Excluded (v1): symbolic/analytic `d(beta)/d(theta)` path.
- Excluded (v1): broad support for non-smooth residual modes unless stabilized.

**Further Considerations**
1. Support matrix: start with `variable_projection` only (recommended).
2. FD runtime policy: full-parameter FD first, optionally parallelized later (recommended).
3. Rank deficiency: SVD pseudo-inverse with warnings instead of hard failure (recommended).
