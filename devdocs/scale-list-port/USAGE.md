# Using the port on v0.8 staging

The source branch uses v0.7 `Model`/`Scheme`/megacomplex APIs. This branch keeps
v0.8 `Scheme`, experiment/data models, elements and split result files. Load a
native scheme with `load_scheme`, then use:

```python
scheme.compute_clp_standard_error = True  # also accepted in scheme YAML
result = scheme.optimize(
    parameters, datasets,
    x_scale="jac",
    raise_exception=True,
)
errors = result.optimization_results["sample"].fit_decomposition.clp_standard_error
```

Both CLP uncertainty settings also accept per-call `scheme.optimize` keyword overrides.
CLP standard errors are optional and require variable projection. They combine
conditional linear covariance with nonlinear covariance propagation. They are
local uncertainty estimates, not proof of parameter identifiability or convergence.
The existing optimization defaults are retained when these options are omitted.

## Source-to-staging mapping

| Source v0.7 concept | Native v0.8 port |
| --- | --- |
| dataset `megacomplex` / `global_megacomplex` | `elements` / `global_elements` |
| dataset `single_amplitude_model: true` | preferred `global_composition: paired`; source flag also accepted |
| dataset `scale_list` | dataset `scale_list` (same ordered global-coordinate binding) |
| dataset scalar `scale` | experiment `scale: {dataset: parameter}` |
| global model `parameter_penalties` | top-level scheme `parameter_penalties` |
| equal-area `relative: true` | same field on experiment `clp_penalties` |
| scheme `x_scale` | persisted scheme field or optimize keyword; scalar, `jac`, list or label mapping |
| CLP errors in result dataset | `optimization_results[name].fit_decomposition.clp_standard_error` |
| grouped/convolved IRF items | same type strings under dataset `activations` |
| separate initial-concentration lists | activation `compartments: {species: value}` mapping |
| skip all dataset saving | existing `SAVING_OPTIONS_MINIMAL` (staging filters are exclusions) |
| optimized parameter T-values | `to_dataframe(as_optimized=True)` or table saving with `as_optimized=True` |
| Hamamatsu `.img` loading | `load_dataset("measurement.img")` |
| `spectral-model-clp-guide` megacomplex, `shape: {target: shape_label}` | `spectral-model-clp-guide` element, shape inline under `shapes: {target: {...}}` |
| `skewed-gaussian-sum` shape | same type; `amplitude`, `location`, `width`, `skewness` are equal-length lists |
| guide dataset generated from the model | omit it from the `datasets` passed to `optimize`; it is generated |

Grouped activation type strings are `multi-multi-gaussian`,
`conv-multi-multi-gaussian` and `norm-conv-multi-multi-gaussian`.
Centers, widths and scales are nested lists. Centers after the first are offsets;
scales after the first are relative factors. `convwidth` is an ordered per-global-index
list. `normalize_area` and `normarea` control normalization. The normalized-convolution
type normalizes after broadening; the convolution type preserves pre-broadening
normalization. Numeric spline knot positions are not parameter references.

## Example additions to a native scheme

```yaml
parameter_penalties:
  - type: equal
    source: rates.a
    target: rates.b
    parameter: ratios.ab
    weight: 2.0
x_scale: jac
experiments:
  measured:
    datasets:
      sample:
        elements: [kinetics]
        global_elements: [spectra]
        global_composition: paired
        # One scale per global coordinate, in the data's coordinate order:
        scale_list: [scale.blue, scale.green, scale.red]
        activations:
          excitation:
            type: norm-conv-multi-multi-gaussian
            compartments: {species_a: 1, species_b: 1}
            center: [[irf.center, irf.offset]]
            width: [[irf.width, irf.width_tail]]
            scale: [[1, irf.relative_tail]]
            convwidth: [irf.blue, irf.green, irf.red]
            normalize_area: true
            normarea: 1000
```

This is a fragment: define the named kinetic/spectral elements in `library`, supply
all referenced parameters and use data with three global coordinates. Paired mode
couples columns by position and requires equal counts. Fix a scale or provide a
linked reference dataset when fitting scale lists; unconstrained CLPs can absorb
free scales. Reordering data requires reordering the positional scale/IRF lists.

## Spectral model CLP guides

A `spectral-model-clp-guide` dataset needs no measured data. When its label is missing
from the datasets passed to `optimize`, the optimization creates it with one model-axis
point on the union of the spectral axes of the measured datasets in the same experiment,
and recalculates its values from `shapes[target]` at every evaluation. Its residual is
`scale * clp(target) - shape`, so the experiment `scale` and dataset `weights` set the
guide strength. The result reports the guide data at the optimized parameters.

```yaml
library:
  guide_s1:
    type: spectral-model-clp-guide
    target: s1
    shapes:
      s1:
        type: skewed-gaussian-sum
        amplitude: [shapes.s1.amplitude, shapes.s1v.amplitude]
        location: [shapes.s1.location, shapes.s1v.location]
        width: [shapes.s1.width, shapes.s1v.width]
        skewness: [shapes.s1.skewness, shapes.s1v.skewness]
experiments:
  linked:
    scale: {guide_s1_data: scale.s1}
    datasets:
      guide_s1_data:
        elements: [guide_s1]
        spectral_axis_inverted: true  # shape parameters in cm^-1 for nm data
        spectral_axis_scale: 1.0e7
        weights: [{value: 5000}]
      measured: {elements: [kinetics], activations: {...}}
```

The validation workspace's external `validation.case_studies.migrate.convert_model`
translates supported v0.7 dictionaries without rewriting the v0.8 core into v0.7.
The port's exact revision inventory and scientific evidence are recorded beside
this document and in the validation workspace issue brief.
