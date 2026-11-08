# Usage

```{toctree}
:maxdepth: 1
:glob:

./*
```

## Migrating `scale_list` models to v0.8

Models from the v0.7 `scale_list` branch use megacomplexes and dataset groups.
In v0.8, define elements in the scheme's `library` and group datasets under
`experiments`. Load the scheme with `load_scheme` and fit it with `scheme.optimize`.

| v0.7 `scale_list` concept | v0.8 equivalent |
| --- | --- |
| Dataset `megacomplex` / `global_megacomplex` | Dataset `elements` / `global_elements` |
| Dataset `megacomplex_scale` / `global_megacomplex_scale` | Dataset `element_scale` / `global_element_scale`. Use positional lists for repeated elements. Enable `normalize_element_scales` / `normalize_global_element_scales` to reproduce the source branch's normalization. |
| Dataset `single_amplitude_model: true` | Dataset `global_composition: paired`. The original flag is also accepted. |
| Dataset `scale_list` | Dataset `scale_list`, or experiment `scale_list: {dataset: [...]}`. Use one declaration per dataset. |
| Dataset scalar `scale` | Experiment `scale: {dataset: parameter}`. This also applies to single-dataset and full-model fits. |
| Model `parameter_penalties` | Top-level scheme `parameter_penalties` |
| Equal-area penalty `relative: true` | Same field on experiment `clp_penalties` |
| Scheme `x_scale` | Scheme field or fit keyword. Accepts a scalar, `jac`, a list, or a mapping by free parameter label. |
| CLP errors in the result dataset | `result.optimization_results[name].fit_decomposition.clp_standard_error` |
| Grouped and convolved IRFs | Same type strings under dataset `activations` |
| Initial-concentration lists | Activation `compartments: {species: value}` |
| Skip saving result arrays | Use `SAVING_OPTIONS_MINIMAL` with `Result.save`. |
| Optimized parameter T-values | `to_dataframe(as_optimized=True)` or table saving with `as_optimized=True` |
| Hamamatsu `.img` loading | `load_dataset("measurement.img")` |

Match each `scale_list` and `convwidth` entry to the corresponding global coordinate.
Reorder these lists whenever you reorder the data. When fitting scales that free
CLPs can absorb, fix a scale or include a linked reference dataset.
Paired full models require equal local and global component counts. They pair
columns by position by default. Set `align_paired_clp_labels: true` to pair by
label when both sides have the same unique labels.

Grouped activation types are `multi-multi-gaussian`, `conv-multi-multi-gaussian`,
and `norm-conv-multi-multi-gaussian`. Centers, widths, and scales use nested lists.
Within each group, centers after the first are offsets from the first center, and
scales after the first are factors of the first scale. With `normalize: true`,
`normalize_area: true` selects the target Gaussian area (`normarea`), while
`normalize_area: false` divides the kinetic matrix by the sum of the scales.
The normalized convolution type applies area normalization after broadening.
Spline knot positions are numeric coordinates.

Set `compute_clp_standard_error: true` to estimate uncertainties for
variable-projection experiments. Other residual functions skip the estimate.
Ctilde experiments currently reject CLP uncertainty.
Set `add_svd: false` to omit optional Ctilde residual SVD outputs.
Both options can also be set for an individual fit.
