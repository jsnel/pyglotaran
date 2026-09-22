# Scale-list and paired full-model contracts

This note records the staging-native contracts used to port the `scale_list`
and single-amplitude functionality from the v0.7 development branch.

## Per-global-coordinate dataset scales

`DataModel.scale_list` is the canonical runtime field. It is an ordered list
of numeric values or parameter references, with one entry for every coordinate
on the dataset's global axis. Resolution preserves that order and rejects a
list whose length does not equal the global-axis length. `ExperimentModel` also
accepts `scale_list: {dataset_label: [...]}` as a YAML convenience and copies
the resolved list to the corresponding dataset model.

The effective model matrix is multiplied by the scalar dataset scale and then
by the scale at each global coordinate. An index-independent matrix is promoted
to a global-indexed matrix so linked and unlinked datasets use the same
per-index representation. Simulation applies the vector factor to each
generated global slice. Scalar `ExperimentModel.scale` continues to provide
the existing linked-dataset scalar behavior and combines with `scale_list` for
a single dataset as well.

Resolved scale values are retained in optimization result metadata under
`scale_list`, which makes the fitted-data contract inspectable without
reconstructing the input model.

## Paired full-model composition

Full models retain Cartesian composition by default. Set
`global_composition: paired` to compose model and global columns by position,
or use `single_amplitude_model: true` for compatibility with the source branch.
Paired composition requires equal model and global component counts. The
resulting matrix has one column per pair and uses labels of the form
`global_label@model_label`; the result amplitude matrix is diagonal and the
reported CLP size counts only the paired amplitudes.

Index-dependent global matrices remain unsupported for paired composition,
because the paired contract is defined for one global component matrix shared
over the global coordinate. Cartesian composition and all existing scalar
scale behavior remain unchanged when the new fields are omitted.

## Evidence

The focused regression coverage is in:

- `tests/optimization/test_matrix.py` for scale promotion, length validation,
  paired matrix construction, and component-count validation.
- `tests/optimization/test_objective.py` for diagonal paired amplitudes and
  CLP sizing.
- `tests/simulation/test_simulation.py` for per-coordinate scaling and paired
  simulation.

The focused suite currently passes with 46 tests. Per-coordinate scales can be
non-identifiable when the corresponding amplitudes are also unconstrained; the
staging port preserves the requested parameterization and leaves that gauge
choice to the model author.
