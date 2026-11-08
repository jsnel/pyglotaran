"""Linked full-model variable projection with a shared set of CLPs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from glotaran.model.data_model import is_data_model_single_amplitude
from glotaran.optimization.data import LinkedFullOptimizationData
from glotaran.optimization.data import OptimizationData
from glotaran.optimization.estimation import OptimizationEstimation
from glotaran.optimization.matrix import OptimizationMatrix
from glotaran.optimization.matrix import compose_full_matrix
from glotaran.optimization.objective import FitDecomposition
from glotaran.optimization.objective import OptimizationObjectiveResult
from glotaran.optimization.objective import OptimizationResult
from glotaran.optimization.objective import create_input_data

if TYPE_CHECKING:
    from glotaran.model.clp_relation import ClpRelation
    from glotaran.optimization.objective import OptimizationObjective


_MAX_RELATION_COEFFICIENTS = 4096


@dataclass
class _RelationExpansion:
    """Record which original columns remain and how to recover removed columns."""

    kept: list[int]
    relations: list[tuple[int, int, float]]

    def reduce(self, matrix: np.ndarray) -> np.ndarray:
        column_count = matrix.shape[1]
        if column_count * len(self.kept) <= _MAX_RELATION_COEFFICIENTS:
            # Preserve the BLAS accumulation order for sensitive finite differences.
            # The coefficient matrix is bounded and is only needed for actual relations.
            positions = {index: position for position, index in enumerate(self.kept)}
            coefficients = np.zeros((column_count, len(self.kept)), order="F")
            coefficients[self.kept, np.arange(len(self.kept))] = 1.0
            for source, target, coefficient in self.relations:
                coefficients[target, positions[source]] = coefficient
            return matrix @ coefficients
        # Match the promotion of the previous float64 expansion matrix.
        matrix = matrix.astype(np.result_type(matrix.dtype, np.float64), copy=False)
        for source, target, coefficient in self.relations:
            matrix[:, source] += coefficient * matrix[:, target]
        return matrix[:, self.kept]

    def expand(self, values: np.ndarray, *, variance: bool = False) -> np.ndarray:
        """Restore removed coefficients or marginal variances from their sole source."""
        expanded = np.empty(
            len(self.kept) + len(self.relations), dtype=np.result_type(values.dtype, np.float64)
        )
        expanded[self.kept] = values
        for source, target, coefficient in self.relations:
            value = coefficient * expanded[source]
            expanded[target] = coefficient * value if variance else value
        return expanded


@dataclass
class FullDataset:
    data: OptimizationData
    local: OptimizationMatrix
    global_matrix: OptimizationMatrix
    pairs: list[tuple[str, str]]
    reduced_pairs: list[tuple[str, str]]
    expansion: _RelationExpansion | None
    matrix: np.ndarray


def _prepare_dataset(data: OptimizationData, relations: list[ClpRelation]) -> FullDataset:
    local = OptimizationMatrix.from_data(data, apply_weight=False)
    global_matrix = OptimizationMatrix.from_data(data, apply_weight=False, global_matrix=True)
    if global_matrix.is_index_dependent:
        msg = "Index dependent global matrices are not supported"
        raise ValueError(msg)
    if local.constraints or global_matrix.constraints:
        msg = "CLP constraints on linked full models are not supported"
        raise ValueError(msg)
    paired = is_data_model_single_amplitude(data.model)
    if paired and len(global_matrix.clp_axis) != len(local.clp_axis):
        msg = "Single-amplitude models require equal component counts"
        raise ValueError(msg)
    if paired:
        global_matrix = OptimizationMatrix.align_paired_global(data.model, local, global_matrix)
    pairs, matrix = compose_full_matrix(local, global_matrix, paired=paired)
    if len(set(pairs)) != len(pairs):
        msg = "Linked full models require unique CLP label pairs"
        raise ValueError(msg)
    if data.flat_weight is not None:
        matrix *= data.flat_weight[:, None]
    expansion = _relation_expansion(pairs, relations, data.global_axis, paired=paired)
    reduced_pairs = pairs
    if expansion is not None:
        matrix = expansion.reduce(matrix)
        reduced_pairs = [pairs[index] for index in expansion.kept]
    return FullDataset(data, local, global_matrix, pairs, reduced_pairs, expansion, matrix)


def _relation_expansion(
    pairs: list[tuple[str, str]],
    relations: list[ClpRelation],
    global_axis: np.ndarray,
    *,
    paired: bool,
) -> _RelationExpansion | None:
    if not paired:
        return None
    applicable_relations = []
    for relation in relations:
        source, target = (relation.source, relation.source), (relation.target, relation.target)
        if source not in pairs or target not in pairs:
            continue
        applies = [relation.applies(index) for index in global_axis]
        if any(applies) and not all(applies):
            msg = "CLP relation intervals partially overlapping a full-model axis are unsupported"
            raise ValueError(msg)
        if not any(applies):
            continue
        applicable_relations.append(relation)
    if not applicable_relations:
        return None
    targets = [relation.target for relation in applicable_relations]
    sources = [relation.source for relation in applicable_relations]
    if len(targets) != len(set(targets)) or set(targets) & set(sources):
        msg = "Chained or multiple CLP relations are unsupported"
        raise ValueError(msg)
    indices = []
    for relation in applicable_relations:
        source, target = (relation.source, relation.source), (relation.target, relation.target)
        source_index, target_index = pairs.index(source), pairs.index(target)
        coefficient = float(getattr(relation.parameter, "value", relation.parameter))
        indices.append((source_index, target_index, coefficient))
    removed = {target for _, target, _ in indices}
    return _RelationExpansion([i for i in range(len(pairs)) if i not in removed], indices)


def build_linked_full(
    objective: OptimizationObjective,
) -> tuple[dict[str, FullDataset], list[tuple[str, str]], np.ndarray, np.ndarray]:
    """Build one common full-model linear system at the current parameters."""
    data_provider = objective.data
    assert isinstance(data_provider, LinkedFullOptimizationData)
    data_provider.update_generated_data()
    datasets = {
        label: _prepare_dataset(data, objective.model.clp_relations)
        for label, data in data_provider.data.items()
    }
    aligned = list(
        dict.fromkeys(pair for dataset in datasets.values() for pair in dataset.reduced_pairs)
    )
    matrices, vectors = [], []
    for dataset in datasets.values():
        aligned_matrix = np.zeros((dataset.matrix.shape[0], len(aligned)))
        aligned_matrix[:, [aligned.index(pair) for pair in dataset.reduced_pairs]] = dataset.matrix
        matrices.append(aligned_matrix)
        assert dataset.data.flat_data is not None
        vectors.append(dataset.data.flat_data)
    return datasets, aligned, np.vstack(matrices), np.concatenate(vectors)


def calculate_linked_full(
    objective: OptimizationObjective, *, result: bool
) -> np.ndarray | OptimizationObjectiveResult:
    """Solve a shared full-model CLP system and optionally build native results."""
    datasets, aligned, matrix, vector = build_linked_full(objective)
    estimation = OptimizationEstimation.calculate(
        matrix, vector, objective.model.residual_function
    )
    if not result:
        return estimation.residual
    results = {}
    offset = 0
    for label, dataset in datasets.items():
        data, local, global_matrix = dataset.data, dataset.local, dataset.global_matrix
        assert data.flat_data is not None
        size = data.flat_data.size
        values = estimation.clp[[aligned.index(pair) for pair in dataset.reduced_pairs]]
        if dataset.expansion is not None:
            values = dataset.expansion.expand(values)
        shape = (len(global_matrix.clp_axis), len(local.clp_axis))
        clp_values = (
            np.diag(values)
            if is_data_model_single_amplitude(data.model)
            else values.reshape(shape)
        )
        clp = xr.DataArray(
            clp_values,
            coords={"global_clp_label": global_matrix.clp_axis, "clp_label": local.clp_axis},
            dims=["global_clp_label", "clp_label"],
        )
        model_dim, global_dim = data.model_dimension, data.global_dimension
        model_axis, global_axis = data.model_axis, data.global_axis
        local_array = local.to_data_array(global_dim, global_axis, model_dim, model_axis)
        global_array = global_matrix.to_data_array(model_dim, model_axis, global_dim, global_axis)
        result_dataset = objective.create_result_dataset(data)
        result_dataset["residual"] = xr.DataArray(
            estimation.residual[offset : offset + size]
            .reshape(global_axis.size, model_axis.size)
            .T,
            coords={model_dim: model_axis, global_dim: global_axis},
            dims=[model_dim, global_dim],
        )
        data.unweight_result_dataset(result_dataset)
        results[label] = OptimizationResult(
            input_data=create_input_data(data, result_dataset),
            residuals=result_dataset.residual,
            elements={
                label: xr.Dataset(
                    {
                        "amplitudes": clp,
                        "global_concentrations": global_array,
                        "model_concentrations": local_array,
                    }
                )
            },
            fit_decomposition=FitDecomposition(clp=clp, matrix=local_array),
            meta=objective.create_result_metadata(label, data, result_dataset),
        )
        offset += size
    return OptimizationObjectiveResult(
        optimization_results=results, clp_size=len(aligned), additional_penalty=0
    )
