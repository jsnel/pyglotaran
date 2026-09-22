# IRF and initial-concentration port

This note records the v0.7 `scale_list` behavior that is represented in the
v0.8 activation and element API.

## Width dispersion

`MultiGaussianActivation` and `GaussianActivation` retain the existing v0.8
polynomial center and width dispersion and now also accept:

- `width_dispersion_spline_knots` as numeric knot positions;
- `width_dispersion_spline_values` as parameter labels or values;
- `width_dispersion_spline_knots_in_wavelength` for wavelength knots on a
  reciprocal global axis;
- the four `width_dispersion_skewed_gaussian_*` parameters as an all-or-none
  width correction term.

The v0.7 spelling `model_dispersion_with_wavenumber` is accepted alongside the
existing v0.8 `reciproke_global_axis` spelling. Spline construction occurs once
per activation parameter expansion, while the resulting spline is evaluated at
each global-axis point. Knots are sorted after optional reciprocal conversion
and must be strictly increasing.

`MultiMultiGaussianActivation` expands nested groups into the flat Gaussian
parameters consumed by the kinetic matrix. Centers after the first in a group
are offsets from the first center; scales after the first are factors applied to
the first scale. `ConvMultiMultiGaussianActivation` applies per-global-index
quadrature broadening. `NormConvMultiMultiGaussianActivation` normalizes after
broadening, preserving equal target area at each global index. The model item
resolver must support nested parameter collections for YAML labels.

## Coherent artifact derivatives

`CoherentArtifactElement` now supports orders 1 through 5. Orders 4 and 5 use
the third and fourth Gaussian derivative expressions (Hermite form), and the
existing order 1 through 3 expressions are unchanged.

## Initial concentrations

Kinetic matrix construction and result creation now share one normalization
helper. Activation entries outside the kinetic element still contribute to the
normalization denominator, matching the existing v0.8 behavior; selected
`not_normalized_compartments` remain on their original scale. A numerically near-zero denominator
is rejected with a clear error instead of producing unstable or non-finite matrices.
This guard uses NumPy `isclose(sum, 0)`; normalized input populations should use
a sensible common scale (for example, fractions summing to one). Result
metadata therefore reports the same normalized initial concentrations used by
the fit.

## Focused evidence

The focused tests cover spline dispersion in both axis conventions, skewed
Gaussian correction, grouped area normalization, per-index broadening and
post-broadening normalization, fifth-order coherent artifact derivatives, and
normalized initial-concentration result metadata.
