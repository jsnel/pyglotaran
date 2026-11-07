"""Ctilde direct residuals, evaluated after the measured source CLP solve."""

from __future__ import annotations

import numpy as np
import xarray as xr

from glotaran.builtin.elements.ctilde import CtildeElement
from glotaran.model.errors import GlotaranModelError
from glotaran.optimization.data import LinkedOptimizationData
from glotaran.optimization.data import OptimizationData
from glotaran.optimization.matrix import OptimizationMatrix
from glotaran.optimization.objective import FitDecomposition
from glotaran.optimization.objective import OptimizationObjective
from glotaran.optimization.objective import OptimizationObjectiveResult
from glotaran.optimization.objective import OptimizationResult
from glotaran.optimization.objective import OptimizationResultMetaData
from glotaran.optimization.objective import calculate_root_mean_square_error
from glotaran.plugin_system.base_registry import full_plugin_name


class CtildeObjective:
    """Compose the measured-data objective with generated direct residuals.

    A Ctilde observation is generated from the current source CLP solve at every
    objective evaluation. Ctilde does not add free CLPs of its own.
    """

    def __init__(self, model):
        self._model = model
        self._auxiliary = {}
        self._supplied = {}
        measured = {}
        for label, data_model in model.datasets.items():
            generators = [
                element for element in data_model.elements if isinstance(element, CtildeElement)
            ]
            if generators:
                if (
                    len(generators) != 1
                    or len(data_model.elements) != 1
                    or data_model.global_elements
                ):
                    raise GlotaranModelError(
                        f"Ctilde dataset '{label}' must contain only one Ctilde element."
                    )
                self._auxiliary[label] = (data_model, generators[0])
                if data_model.data is not None:
                    self._supplied[label] = data_model.data.copy(deep=True)
            else:
                measured[label] = data_model
        if not measured:
            raise GlotaranModelError(
                "Ctilde requires a measured source dataset in its experiment."
            )
        for label, (_, element) in self._auxiliary.items():
            if element.source_dataset not in measured:
                raise GlotaranModelError(
                    f"Ctilde dataset '{label}' requires source '{element.source_dataset}' in the same experiment."
                )
            source = measured[element.source_dataset]
            if source.global_elements:
                raise GlotaranModelError(
                    "Ctilde source must use a spectral CLP model, not a full model."
                )
            if not isinstance(source.data, xr.Dataset) or not {"time", "spectral"}.issubset(
                source.data["data"].dims
            ):
                raise GlotaranModelError(
                    "Ctilde source must have time and spectral data dimensions."
                )
        self._inner = OptimizationObjective(model.model_copy(update={"datasets": measured}))

    @property
    def model(self):
        return self._model

    @property
    def data(self):
        return self._inner.data

    def _calculate_auxiliary(self, label, data_model, element, clp):
        source_model = self._model.datasets[element.source_dataset]
        source_data = source_model.data
        time = np.asarray(source_data["time"])
        spectral = np.asarray(source_data["spectral"])
        source_matrix = OptimizationMatrix.from_data_model(source_model, spectral, time, None)
        if source_matrix.is_index_dependent:
            raise GlotaranModelError("Ctilde source cannot have IRF dispersion.")
        reduced = [
            source_matrix.reduce(index, self._model.clp_relations, copy=True) for index in spectral
        ]
        if not reduced:
            raise GlotaranModelError("Ctilde source must have a nonempty spectral axis.")
        basis = reduced[0]
        if any(
            item.clp_axis != basis.clp_axis or not np.allclose(item.array, basis.array)
            for item in reduced[1:]
        ):
            raise GlotaranModelError(
                "Ctilde source relations must be constant over its spectral range."
            )
        basis_labels = list(basis.clp_axis)
        supplied = self._supplied.get(label)
        if supplied is not None:
            if not isinstance(supplied, xr.Dataset) or "data" not in supplied:
                raise GlotaranModelError(f"Ctilde dataset '{label}' must contain data.")
            if set(supplied["data"].dims) != {"time", "basis"}:
                raise GlotaranModelError(
                    f"Ctilde dataset '{label}' must have time and basis dimensions."
                )
            if not np.array_equal(supplied["time"].to_numpy(), time):
                raise GlotaranModelError(
                    f"Ctilde dataset '{label}' time coordinates must match the source in order."
                )
            requested = [str(value) for value in supplied["basis"].to_numpy()]
            if (
                len(set(requested)) != len(requested)
                or len(set(basis_labels)) != len(basis_labels)
                or set(requested) != set(basis_labels)
            ):
                raise GlotaranModelError(
                    f"Ctilde dataset '{label}' basis must match unique source CLP labels."
                )
            basis_labels = requested
        clp_dim = "amplitude_label" if "amplitude_label" in clp.dims else "clp_label"
        try:
            etilde = (
                clp.sel({clp_dim: basis_labels})
                .rename({clp_dim: "basis"})
                .transpose("spectral", "basis")
            )
        except (KeyError, ValueError) as exc:
            raise GlotaranModelError(
                "Ctilde source CLPs must match the reduced kinetic basis."
            ) from exc
        ctilde = xr.DataArray(
            basis.array, dims=("time", "basis"), coords={"time": time, "basis": basis.clp_axis}
        ).sel(basis=basis_labels)
        observed = element.calculate_observations(source_data, ctilde, etilde)
        if supplied is not None and "weight" in supplied:
            observed["weight"] = supplied["weight"]
        data_model.data = observed.assign_attrs(
            generated_by_model=1, source_dataset=element.source_dataset
        )
        auxiliary_data = OptimizationData(data_model, scale=self._model.scale.get(label))
        prediction = ctilde.to_numpy().copy()
        if auxiliary_data.dataset_scale is not None:
            prediction *= auxiliary_data.dataset_scale
        if auxiliary_data.scale_list is not None:
            prediction *= auxiliary_data.scale_list[np.newaxis, :]
        residual = observed["data"].to_numpy() - prediction
        weighted = (
            residual * auxiliary_data.weight if auxiliary_data.weight is not None else residual
        )
        return observed, ctilde, etilde, auxiliary_data, residual, weighted

    def _source_spectra(self):
        """Collect resolved source CLPs without constructing result datasets or SVDs."""
        matrices = self._inner.calculate_matrices()
        reduced = self._inner.calculate_reduced_matrices(matrices)
        estimations = self._inner.resolve_estimations(
            matrices, reduced, self._inner.calculate_estimations(reduced)
        )
        provider = self._inner.data
        labels = {element.source_dataset for _, element in self._auxiliary.values()}
        if isinstance(provider, LinkedOptimizationData):
            return {
                label: self._inner.get_dataset_amplitudes(
                    label,
                    [matrix.clp_axis for matrix in matrices],
                    estimations,
                    OptimizationMatrix.from_data(provider.data[label]).clp_axis,
                    provider.data[label].global_dimension,
                    provider.data[label].global_axis,
                )
                for label in labels
            }
        label = next(iter(labels))
        matrix = matrices[0]
        return {
            label: xr.DataArray(
                [estimation.clp for estimation in estimations],
                dims=(provider.global_dimension, "amplitude_label"),
                coords={
                    provider.global_dimension: provider.global_axis,
                    "amplitude_label": matrix.clp_axis,
                },
            )
        }

    def calculate(self):
        base = self._inner.calculate()
        source_spectra = self._source_spectra()
        penalties = [base]
        for label, (model, element) in self._auxiliary.items():
            *_, weighted = self._calculate_auxiliary(
                label, model, element, source_spectra[element.source_dataset]
            )
            penalties.append(weighted.T.ravel())
        return np.concatenate(penalties)

    def get_result(self):
        result = self._inner.get_result()
        for label, (model, element) in self._auxiliary.items():
            source_result = result.optimization_results[element.source_dataset]
            observed, ctilde, etilde, data, residual, weighted = self._calculate_auxiliary(
                label, model, element, source_result.fit_decomposition.clp
            )
            basis = observed["basis"]
            residual_array = xr.DataArray(
                residual, dims=("time", "basis"), coords={"time": observed["time"], "basis": basis}
            )
            identity = xr.DataArray(
                np.eye(basis.size),
                dims=("basis", "amplitude_label"),
                coords={"basis": basis, "amplitude_label": basis.to_numpy()},
            )
            matrix = ctilde.rename({"basis": "amplitude_label"})
            auxiliary_element = xr.Dataset(
                {"Ctilde": ctilde, "Etilde": etilde, "refresid": observed["refresid"]}
            )
            auxiliary_element.attrs["element_uid"] = full_plugin_name(element)
            source_result.elements[element.label] = xr.Dataset(
                {"Ctilde": ctilde, "Etilde": etilde},
                attrs={"element_uid": full_plugin_name(element)},
            )
            result.optimization_results[label] = OptimizationResult(
                input_data=observed["data"],
                residuals=residual_array,
                elements={element.label: auxiliary_element},
                fit_decomposition=FitDecomposition(clp=identity, matrix=matrix),
                meta=OptimizationResultMetaData(
                    global_dimension="basis",
                    model_dimension="time",
                    root_mean_square_error=calculate_root_mean_square_error(residual_array),
                    weighted_root_mean_square_error=float(
                        np.linalg.norm(weighted) / np.sqrt(weighted.size)
                    ),
                    scale=data.dataset_scale if data.dataset_scale is not None else 1,
                    scale_list=data.scale_list.tolist() if data.scale_list is not None else None,
                ),
            )
        return OptimizationObjectiveResult(
            optimization_results=result.optimization_results,
            clp_size=result.clp_size,
            additional_penalty=result.additional_penalty,
        )

    def get_clp_penalty_diagnostics(self):
        return self._inner.get_clp_penalty_diagnostics()
