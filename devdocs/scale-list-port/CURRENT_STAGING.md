# scale_list port on current staging

This branch adapts the author's `ism200/scale_list` endpoint `4ffd8ce6` (the
equivalent rebased reference is `fcad091c`) to staging based on `8e725314`.
Earlier feature-focused staging port commits were replayed and adjusted to the
current optimizer, Project, record/export and result APIs. Their order differs
from the author's 0.7 commits where staging dependencies required regrouping;
linked full-model and Ctilde changes are separate later commits. Other historical
port notes describe an earlier staging baseline, not this branch.

The port includes grouped Gaussian activations, coordinate scale lists, paired
and linked full models, CLP uncertainty for supported least-squares models,
global parameter penalties, fit scaling, the spectral-model guide and generated
datasets, and Ctilde/refresid. Project records retain effective fit settings and
generated data can be recomputed from measured inputs.

## Native and migrated scale semantics

Staging continues to interpret element scales literally by default. The author's
newer 0.7.5 source normalizes an explicit ordered scale list by its sum. Converted
models that need this behavior opt in with `normalize_element_scales` and/or
`normalize_global_element_scales`, using a complete element-to-scale mapping.
Paired full models use positional component pairing by default. For a migrated
model where staging resolves identical component labels in a different order,
`align_paired_clp_labels` explicitly selects name-based pairing.

## Supported boundaries

Linked full models reject index-dependent global matrices, CLP constraints and
area penalties, partially overlapping relation intervals, and chained or
duplicate-target CLP relations. These are explicit restrictions; do not infer
support from the ordinary linked-data or unlinked-full paths.

The original also rejects partial relation intervals and mixed full/ordinary
groups. It silently ignores full-model constraints and area penalties; chained
or duplicate-target relations do not produce a reliable expansion. The port
rejects these combinations instead. A single paired full-model experiment with
relations uses the linked-full reduction, including its validation and result
layout. The v0.8 experiment API has no separate legacy `link_clp` switch for
distinguishing the original linked and unlinked single-dataset groups.

Ctilde requires a spectral-CLP source in the same experiment, without IRF
dispersion or spectrally varying relations. An auxiliary dataset contains one
Ctilde element. Supplied auxiliary data provide ordered time/basis coordinates
and weights; their values are replaced by generated regularized NNLS observations.
Basis order may be permuted, but time coordinates must match the source in order.
Ctilde CLP standard errors are not implemented. New Project records preserve
string coordinate labels for recomputation checks; old records cannot
retrospectively verify labels they did not record.

The original imposes the same source-group, dispersion and varying-relation
restrictions. Although it accepts permuted supplied time coordinates, it pairs
them with source-ordered calculated rows; the port rejects that mismatch. Its
generic CLP uncertainty path has no demonstrated Ctilde-specific correctness.
The auxiliary result includes the `refresid` singular values and vectors, using
the staging SVD conventions, as well as the residual itself. Set the scheme's
`add_svd: false` or pass `add_svd=False` to `Scheme.optimize` or
`Project.optimize` to skip this result-time decomposition; the default is
`true`. A per-fit override takes precedence and is saved in the effective
optimizer settings for result persistence, records, exports and recomputation.

The older scheme also computed input, residual and weighted-residual SVDs as
part of optimization. Staging does not currently create those decompositions in
its optimization results, and its input result representation normally retains
only the data and optional weights rather than other supplied input variables.
This setting does not undo SVDs already calculated by input preparation: some
readers with `prepare=True` compute one when loading data, before optimization.
Use those readers' `prepare=False` option when that separate calculation is not
wanted.

## Validation snapshot

A follow-up comparison of the original author's linked-full and Ctilde code
identified the two additional behaviors described above. With the notebook's
one-evaluation budget, the supplied Ctilde case compared 60 arrays across four
measured datasets, four generated Ctilde datasets and eight generated spectral
guides. All comparisons passed, the measured inputs matched exactly, and the
worst measured fitted-data relative norm was `1.0774e-15`. A fresh 16-dataset
linked-ST comparison, with two evaluations and a reference recomputation at
the saved parameters, matched all inputs and had worst fitted-data relative
norm `5.1436e-9`. Neither limited fit demonstrates convergence. Fresh common
runs completed 11/11 notebooks per side; the 14-leaf comparison reported 9
PASS and 5 documented EXPECTED_DIFFERENCE, with no regressions. The reference
core was `fcad091c`, and the staging code was `e4a03e83`. The workspace retains
the manifests, case-study comparisons, source revisions and guarded-run logs.
The full staging core suite reported 773 passed and 9 expected failures; a
separate test confirmed that the new SVD arrays survive result save and reload.
The workspace validation suite reported 81 passed and 1 skipped.
The separate guarded common-example benchmark completed 150 timed samples. Its
evaluation counts matched in 13 of 15 fits; the two mismatches and differences
in Python and scientific dependencies limit performance comparisons. The
benchmark does not measure the linked-ST or Ctilde cases.

At code commit `33ce20fb`, guarded serial checks reported 763 core tests passed
with 9 xfailed, and 75 workspace validation tests passed with 1 skipped. Fresh
common reference and staging runs executed 11 notebooks each; their 14-leaf
semantic comparison reported 9 PASS and 5 documented EXPECTED_DIFFERENCE with no
regressions or missing leaves. The reference was `fcad091c` (0.7.5), not the
workspace's historical pinned 0.7.4. A 16-dataset linked-ST comparison at the
pre-format commit `ca50e8e7` agreed to a worst fitted-data normalized RMS of
5.144e-9 after the author's two-evaluation budget; neither fit had converged.
The separate validation workspace retains the source hashes, manifests, memory
guard logs, reports and report-only runtime benchmark.
