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
from glotaran.optimization.objective import FitDecomposition
from glotaran.optimization.objective import OptimizationObjectiveResult
from glotaran.optimization.objective import OptimizationResult
from glotaran.optimization.objective import create_input_data

if TYPE_CHECKING:
    from glotaran.model.clp_relation import ClpRelation
    from glotaran.optimization.objective import OptimizationObjective


@dataclass
class FullDataset:
    data: OptimizationData
    local: OptimizationMatrix
    global_matrix: OptimizationMatrix
    pairs: list[tuple[str, str]]
    reduced_pairs: list[tuple[str, str]]
    expansion: np.ndarray
    matrix: np.ndarray


def _normalized_matrix(data: OptimizationData, *, global_matrix: bool) -> OptimizationMatrix:
    """Build a result matrix with native or explicitly normalized scales."""
    return OptimizationMatrix.from_data(data, apply_weight=False, global_matrix=global_matrix)


def _prepare_dataset(data: OptimizationData, relations: list[ClpRelation]) -> FullDataset:
    local = _normalized_matrix(data, global_matrix=False)
    global_matrix = _normalized_matrix(data, global_matrix=True)
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
    pairs = (
        list(zip(global_matrix.clp_axis, local.clp_axis, strict=True))
        if paired
        else [(g, m) for g in global_matrix.clp_axis for m in local.clp_axis]
    )
    if len(set(pairs)) != len(pairs):
        msg = "Linked full models require unique CLP label pairs"
        raise ValueError(msg)
    local_values = (
        local.array
        if local.is_index_dependent
        else np.broadcast_to(local.array, (len(data.global_axis), *local.array.shape))
    )
    if paired:
        matrix = (local_values * global_matrix.array[:, None, :]).reshape(-1, len(pairs))
    else:
        matrix = np.einsum("gmi,gj->gmji", local_values, global_matrix.array).reshape(
            -1, len(pairs)
        )
    if data.flat_weight is not None:
        matrix *= data.flat_weight[:, None]
    expansion, reduced_pairs = _relation_expansion(
        pairs, relations, data.global_axis, paired=paired
    )
    return FullDataset(
        data, local, global_matrix, pairs, reduced_pairs, expansion, matrix @ expansion
    )


def _relation_expansion(
    pairs: list[tuple[str, str]],
    relations: list[ClpRelation],
    global_axis: np.ndarray,
    *,
    paired: bool,
) -> tuple[np.ndarray, list[tuple[str, str]]]:
    expansion = np.eye(len(pairs))
    removed = []
    applicable_relations = []
    for relation in relations if paired else []:
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
    targets = [relation.target for relation in applicable_relations]
    sources = [relation.source for relation in applicable_relations]
    if len(targets) != len(set(targets)) or set(targets) & set(sources):
        msg = "Chained or multiple CLP relations are unsupported"
        raise ValueError(msg)
    for relation in applicable_relations:
        source, target = (relation.source, relation.source), (relation.target, relation.target)
        source_index, target_index = pairs.index(source), pairs.index(target)
        expansion[target_index, source_index] = float(
            getattr(relation.parameter, "value", relation.parameter)
        )
        removed.append(target_index)
    keep = [i for i in range(len(pairs)) if i not in removed]
    expansion = expansion[:, keep]
    return expansion, [pairs[i] for i in keep]


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
        values = (
            dataset.expansion
            @ estimation.clp[[aligned.index(pair) for pair in dataset.reduced_pairs]]
        )
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
