# Recording, recomputing and exporting fits

pyglotaran records every fit in a numbered folder. A record holds what is needed to
recompute the result of the fit from its input data, and nothing more, so it stays small.
When a result is worth keeping or sharing, you export it deliberately; an export contains
everything, including the data.

## Records

Every call of `Scheme.optimize` writes a record:

```text
results/001/
  record.yml                # metadata, see below
  scheme.yml                # the scheme of the fit
  initial_parameters.csv
  optimized_parameters.csv
  cost_history.csv          # cost of every function evaluation
  parameter_history.csv     # only with record_parameter_history=True
```

`record.yml` holds the status of the fit (`success`, `failed` or `interrupted`, or `running`
for a fit that never returned, for example after a kernel crash), the notebook or script and
the scheme file it came from, an optional name, the optimizer settings, the fit statistics,
a summary of the input data (shape, and `min`, `max`, `mean` and `rms` of the coordinates,
the data and the weight), the parent fit (see [Continuing a fit](#continuing-a-fit)) and the
versions of pyglotaran, Python, numpy, scipy, xarray and installed plugins.

Records contain no data and no result arrays. `result.record` tells where the record of a
fit is:

```python
result = scheme.optimize(parameters, datasets={"ta": ta}, name="target model B")
result.record  # RecordReference(path=.../results/001, id='001', uid='...')
```

Failed and interrupted fits are recorded too, with the error and the parameters of the last
function evaluation. A fit that fails before the optimization starts (for example because of
an error in the model) and a dry run are not recorded.

## Projects and the results folder

A project is a folder whose root contains a `project.gta` file. pyglotaran looks for
`project.gta` in the working directory and its parents; the nearest one marks the project
root. Records then go to `results` and exports to `exports` in the project root. Without a
`project.gta`, they go to `results` and `exports` in the working directory.

pyglotaran only checks that `project.gta` exists and never writes to it. It is a YAML file
that describes the project and its data for research data management, and every export
contains a copy. All fields are optional; these are the recommended ones:

```yaml
title: Carotenoid dynamics in LH2 complexes
description: >
  Global and target analysis of ...
data_description: >
  Transient absorption, 400-700 nm, pump 800 nm; samples ...; instrument ...
creators:
  - name: ...
    orcid: 0000-0000-0000-0000
    affiliation: ...
keywords: [transient absorption, target analysis]
license: CC-BY-4.0
related_identifiers:
  - doi: 10.xxxx/... # publication, dataset, data management plan
funding: ...
```

A `project.gta` of pyglotaran 0.7 also marks a project root. Check the parents of your
working directory for a forgotten one: it takes the records of every folder below it.

## Settings

Each setting can be given per call, per session or by an environment variable. The first
that is set applies.

| Setting | Per call | Per session | Environment variable |
| --- | --- | --- | --- |
| Record fits | `record=False` | `record_settings.record = False` | `GLOTARAN_RECORD=0` |
| Results folder | `output_folder="my_results"` | `record_settings.output_folder = ...` | `GLOTARAN_RESULTS_FOLDER=...` |
| Parameter history | `record_parameter_history=True` | `record_settings.record_parameter_history = True` | `GLOTARAN_RECORD_PARAMETER_HISTORY=1` |
| Source notebook or script | | `record_settings.source = "analysis.ipynb"` | |

```python
from glotaran.project import record_settings

record_settings.record = False
```

Two exceptions: `GLOTARAN_RECORD=0` switches recording off everywhere, also for calls with
`record=True`, so test suites, continuous integration and documentation builds never write
records. And `output_folder=` switches recording on for that call.

Session settings do not reach other processes. For fits in worker processes (for example a
parameter scan with `joblib` or `multiprocessing`), pass `record=False` per call or set the
environment variable before the workers start.

The notebook or script is detected in VS Code, Jupyter and for scripts. Elsewhere (for
example when notebooks are executed with nbclient or papermill) it is recorded as `unknown`;
set `record_settings.source` to name it.

## Listing and comparing fits

```python
from glotaran.project import compare_results, list_results

list_results()  # the records in the results folder
list_results("D:/archive/lycopene/results", source="lycopene", status="success")
compare_results("001", "002")
```

`list_results` shows one row per record: id, time, source, name, scheme file, status, parent,
and per dataset the shape and the RMS of the data, followed by the cost, RMSE and number of
function evaluations. A change in the shape or RMS columns shows where the data was replaced.

`compare_results` compares two fits, given by record id, record or export path, or as
`Result`: the fit statistics (with the weighted and unweighted RMSE per dataset), the
parameters, the data summaries and the schemes. Changes of weights, penalties, scales and
constraints show up in the scheme diff. Whether a change of the parameters comes from a
change of the data or of the model is for you to judge.

## Continuing a fit

When you start a fit from the optimized parameters of an earlier fit, the record of the new
fit names the earlier one as its parent:

```python
result = scheme.optimize(parameters, datasets={"ta": ta})                       # 001
result2 = scheme.optimize(result.optimized_parameters, datasets={"ta": ta})     # 002, parent 001
```

Copies and edits of the optimized parameters keep this link. Parameters loaded from a file
do not, so after restarting the kernel you name the parent:

```python
scheme.optimize(load_parameters("fit2.csv"), datasets={"ta": ta}, parent="002")
```

The parent is a record id or uid in the results folder, the path of a record or
`result.record`. If the data of the new fit differs from the data of the parent, the record
lists the changed datasets in `parent_data_changed`.

## Recomputing a fit

```python
from glotaran.project import recompute

result = recompute("002", datasets={"ta": ta})
```

`recompute` evaluates the recorded scheme once at the recorded optimized parameters. Because
records contain no data, you supply the input data of the fit. pyglotaran compares it with
the recorded data summary and raises an error that names what changed, for example
`ta: time max 10 -> 8`. Pass `allow_data_mismatch=True` to recompute anyway; the result then
lists the differences.

The recomputed result has the statistics of the one evaluation and the standard errors of
the original fit; the covariance matrix is not recovered. `result.reconstruction` holds the
metadata of the original fit and of the reconstruction. If the recomputed cost or RMSE
differs from the recorded value by more than a relative `1e-6`, you get a warning.

To be able to recompute a fit later, keep its input data, or keep the preprocessing in a
function, notebook or script that produces the input data without fitting. Re-running a
whole notebook re-runs every fit in it and writes a new record for each.

## Exporting a result

```python
from glotaran.project import export

export(result, name="paper_fig3")
export(result, name="paper_fig3", overwrite=True)  # replaces exports/paper_fig3
```

An export is a folder that can be published as is. It contains everything except the
notebook or script:

- `export.yml` with the metadata of the record (the notebook or script and the scheme file
  by file name only, so the export contains no local paths);
- the scheme and the parameters;
- the input data as passed to `optimize`, including a `weight` variable of the datasets;
- the result arrays, as `Result.save` writes them;
- in `source_files`, copies of the files the datasets were loaded from, with their current
  contents;
- a copy of `project.gta`.

`input_data=False`, `arrays=False` and `source_files=False` leave out parts. An export with
input data loads with `load_result` on any machine; one without can be recomputed with
`recompute(path, datasets=...)`. To export an older fit, recompute it first.
