"""Ctilde direct residuals, evaluated after the measured source CLP solve."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import TypeIs
from typing import cast

import numpy as np
import xarray as xr

from glotaran.builtin.elements.ctilde import CtildeElement
from glotaran.io.prepare_dataset import add_svd_to_dataset
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

if TYPE_CHECKING:
    from glotaran.model.clp_penalties import EqualAreaPenaltyDiagnostic
    from glotaran.model.data_model import DataModel
    from glotaran.model.element import Element
    from glotaran.model.experiment_model import ExperimentModel
    from glotaran.optimization.estimation import OptimizationEstimation


def is_ctilde_element(element: Element | str) -> TypeIs[CtildeElement]:
    """Identify an element that generates Ctilde observations."""
    return isinstance(element, CtildeElement)


def is_ctilde_experiment(model: ExperimentModel) -> bool:
    """Return whether an experiment needs a Ctilde objective."""
    return any(
        is_ctilde_element(element)
        for dataset in model.datasets.values()
        for element in dataset.elements
    )


class CtildeObjective:
    """Compose the measured-data objective with generated direct residuals.

    A Ctilde observation is generated from the current source CLP solve at every
    objective evaluation. Ctilde does not add free CLPs of its own.
    """

    def __init__(self, model: ExperimentModel, *, add_svd: bool = True) -> None:
        self._model = model
        self._add_svd = add_svd
        self._auxiliary = {}
        self._supplied = {}
        self._supplied_basis: dict[str, list[str]] = {}
        measured = {}
        for label, data_model in model.datasets.items():
            generators = [element for element in data_model.elements if is_ctilde_element(element)]
            if generators:
                if (
                    len(generators) != 1
                    or len(data_model.elements) != 1
                    or data_model.global_elements
                ):
                    msg = f"Ctilde dataset '{label}' must contain only one Ctilde element."
                    raise GlotaranModelError(msg)
                self._auxiliary[label] = (data_model, generators[0])
                if data_model.data is not None:
                    self._supplied[label] = cast("xr.Dataset", data_model.data).copy(deep=True)
            else:
                measured[label] = data_model
        if not measured:
            msg = "Ctilde requires a measured source dataset in its experiment."
            raise GlotaranModelError(msg)
        for label, (_, element) in self._auxiliary.items():
            if element.source_dataset not in measured:
                msg = (
                    f"Ctilde dataset '{label}' requires source '{element.source_dataset}' "
                    "in the same experiment."
                )
                raise GlotaranModelError(msg)
            self._validate_source(label, measured[element.source_dataset])
        self._inner = OptimizationObjective(model.model_copy(update={"datasets": measured}))

    def _validate_source(self, label: str, source: DataModel) -> None:
        if source.global_elements:
            msg = "Ctilde source must use a spectral CLP model, not a full model."
            raise GlotaranModelError(msg)
        if not isinstance(source.data, xr.Dataset) or not {"time", "spectral"}.issubset(
            source.data["data"].dims
        ):
            msg = "Ctilde source must have time and spectral data dimensions."
            raise GlotaranModelError(msg)
        if source.data["spectral"].size == 0:
            msg = "Ctilde source must have a nonempty spectral axis."
            raise GlotaranModelError(msg)
        if label in self._supplied:
            self._supplied_basis[label] = self._validate_supplied_axes(
                label, self._supplied[label], np.asarray(source.data["time"])
            )

    @property
    def model(self) -> ExperimentModel:
        return self._model

    @property
    def data(self) -> OptimizationData | LinkedOptimizationData:
        return cast("OptimizationData | LinkedOptimizationData", self._inner.data)

    def _calculate_source_matrices(self) -> dict[str, OptimizationMatrix]:
        """Build each unweighted source basis once at the current parameters."""
        matrices: dict[str, OptimizationMatrix] = {}
        for _, element in self._auxiliary.values():
            label = element.source_dataset
            if label not in matrices:
                model = self._model.datasets[label]
                data = cast("xr.Dataset", model.data)
                matrices[label] = OptimizationMatrix.from_data_model(
                    model, np.asarray(data["spectral"]), np.asarray(data["time"]), None
                )
        return matrices

    def _calculate_auxiliary(
        self,
        label: str,
        data_model: DataModel,
        element: CtildeElement,
        clp: xr.DataArray,
        source_matrix: OptimizationMatrix,
    ) -> tuple[xr.Dataset, xr.DataArray, xr.DataArray, OptimizationData, np.ndarray, np.ndarray]:
        source_data = cast("xr.Dataset", self._model.datasets[element.source_dataset].data)
        time = np.asarray(source_data["time"])
        spectral = np.asarray(source_data["spectral"])
        if source_matrix.is_index_dependent:
            msg = "Ctilde source cannot have IRF dispersion."
            raise GlotaranModelError(msg)
        reduced = [
            source_matrix.reduce(index, self._model.clp_relations, copy=True) for index in spectral
        ]
        basis = reduced[0]
        if any(
            item.clp_axis != basis.clp_axis or not np.allclose(item.array, basis.array)
            for item in reduced[1:]
        ):
            msg = "Ctilde source relations must be constant over its spectral range."
            raise GlotaranModelError(msg)
        supplied = self._supplied.get(label)
        basis_labels = self._supplied_basis.get(label, basis.clp_axis)
        if supplied is not None and (
            len(set(basis.clp_axis)) != len(basis.clp_axis)
            or set(basis_labels) != set(basis.clp_axis)
        ):
            msg = f"Ctilde dataset '{label}' basis must match unique source CLP labels."
            raise GlotaranModelError(msg)
        clp_dim = "amplitude_label" if "amplitude_label" in clp.dims else "clp_label"
        try:
            etilde = (
                clp.sel({clp_dim: basis_labels})
                .rename({clp_dim: "basis"})
                .transpose("spectral", "basis")
            )
        except (KeyError, ValueError) as exc:
            msg = "Ctilde source CLPs must match the reduced kinetic basis."
            raise GlotaranModelError(msg) from exc
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

    @staticmethod
    def _validate_supplied_axes(label: str, supplied: xr.Dataset, time: np.ndarray) -> list[str]:
        if not isinstance(supplied, xr.Dataset) or "data" not in supplied:
            msg = f"Ctilde dataset '{label}' must contain data."
            raise GlotaranModelError(msg)
        if set(supplied["data"].dims) != {"time", "basis"}:
            msg = f"Ctilde dataset '{label}' must have time and basis dimensions."
            raise GlotaranModelError(msg)
        if not np.array_equal(supplied["time"].to_numpy(), time):
            msg = f"Ctilde dataset '{label}' time coordinates must match the source in order."
            raise GlotaranModelError(msg)
        requested = [str(value) for value in supplied["basis"].to_numpy()]
        if len(set(requested)) != len(requested):
            msg = f"Ctilde dataset '{label}' basis must match unique source CLP labels."
            raise GlotaranModelError(msg)
        return requested

    def _source_spectra(
        self,
        matrices: list[OptimizationMatrix],
        estimations: list[OptimizationEstimation],
        source_matrices: dict[str, OptimizationMatrix],
    ) -> dict[str, xr.DataArray]:
        """Collect resolved source CLPs without constructing result datasets or SVDs."""
        provider = self.data
        if isinstance(provider, LinkedOptimizationData):
            amplitude_axes = [matrix.clp_axis for matrix in matrices]
            return {
                label: self._inner.get_dataset_amplitudes(
                    label,
                    amplitude_axes,
                    estimations,
                    source_matrix.clp_axis,
                    provider.data[label].global_dimension,
                    provider.data[label].global_axis,
                )
                for label, source_matrix in source_matrices.items()
            }
        label = next(iter(source_matrices))
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

    def calculate(self) -> np.ndarray:
        matrices, estimations, penalties = self._inner.calculate_linear_fit(resolve_clps=True)
        source_matrices = self._calculate_source_matrices()
        source_spectra = self._source_spectra(matrices, estimations, source_matrices)
        for label, (model, element) in self._auxiliary.items():
            *_, weighted = self._calculate_auxiliary(
                label,
                model,
                element,
                source_spectra[element.source_dataset],
                source_matrices[element.source_dataset],
            )
            penalties.append(weighted.T.ravel())
        return np.concatenate(penalties)

    def get_result(self) -> OptimizationObjectiveResult:
        result = self._inner.get_result()
        source_matrices = self._calculate_source_matrices()
        for label, (model, element) in self._auxiliary.items():
            source_result = result.optimization_results[element.source_dataset]
            observed, ctilde, etilde, data, residual, weighted = self._calculate_auxiliary(
                label,
                model,
                element,
                cast("FitDecomposition", source_result.fit_decomposition).clp,
                source_matrices[element.source_dataset],
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
            if data.dataset_scale is not None:
                matrix = matrix * data.dataset_scale
            if data.scale_list is not None:
                matrix = matrix * xr.DataArray(
                    data.scale_list,
                    dims=("amplitude_label",),
                    coords={"amplitude_label": basis.to_numpy()},
                )
            auxiliary_element = xr.Dataset(
                {"Ctilde": ctilde, "Etilde": etilde, "refresid": observed["refresid"]}
            )
            if self._add_svd:
                add_svd_to_dataset(auxiliary_element, name="refresid")
            auxiliary_element.attrs["element_uid"] = full_plugin_name(element)
            source_result.elements[element.label] = xr.Dataset(
                {"Ctilde": ctilde, "Etilde": etilde},
                attrs={"element_uid": full_plugin_name(element)},
            )
            result.optimization_results[label] = OptimizationResult(
                input_data=observed["data"],
                supplied_input_data=self._supplied.get(label),
                input_data_is_generated=True,
                supplied_input_data_was_provided=label in self._supplied,
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

    def get_clp_penalty_diagnostics(self) -> list[EqualAreaPenaltyDiagnostic]:
        return self._inner.get_clp_penalty_diagnostics()
