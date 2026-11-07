from copy import deepcopy

import numpy as np
import pytest
import xarray as xr

from glotaran.model.clp_relation import ClpRelation
from glotaran.model.experiment_model import ExperimentModel
from glotaran.builtin.elements.spectral.shape import SpectralShapeSkewedGaussianSum
from glotaran.optimization.objective import OptimizationObjective
from tests.optimization.data import TestDataModelGlobal


def paired(value):
    model = deepcopy(TestDataModelGlobal)
    model.global_composition = 'paired'
    model.elements[0].compartments = ['s1', 's2']
    model.global_elements[0].compartments = ['s1', 's2']
    model.data['data'].values[:] = value
    return model


def test_linked_shared_paired_amplitudes_relations_and_dataset_scaling():
    # Matrix columns both equal 5*4. Fit s1=2, s2=3*s1=6, so the
    # first dataset is 160 and the second, scaled by 3, is 480.
    one, two = paired(160), paired(480)
    model = ExperimentModel(datasets={'one': one, 'two': two},
                            scale={'one': 1, 'two': 3},
                            clp_relations=[ClpRelation(source='s1', target='s2', parameter=3)])
    objective = OptimizationObjective(model)
    np.testing.assert_allclose(objective.calculate(), 0, atol=1e-10)
    result = objective.get_result()
    assert result.clp_size == 1
    for label in ('one', 'two'):
        amplitudes = result.optimization_results[label].elements[label].amplitudes
        np.testing.assert_allclose(amplitudes.values, [[2, 0], [0, 6]], atol=1e-10)


def test_global_component_is_used_for_linked_pair():
    one, two = paired(160), paired(160)
    for model in (one, two):
        model.global_elements[0].value = 0
    objective = OptimizationObjective(ExperimentModel(datasets={'one': one, 'two': two}))
    assert np.linalg.norm(objective.calculate()) > 100


def test_partial_relation_interval_is_rejected():
    one, two = paired(160), paired(160)
    relation = ClpRelation(source='s1', target='s2', parameter=3, interval=(0, 3))
    objective = OptimizationObjective(ExperimentModel(datasets={'one': one, 'two': two}, clp_relations=[relation]))
    with pytest.raises(ValueError, match='partially overlapping'):
        objective.calculate()


def test_weighted_residual_order_and_result_are_consistent():
    one, two = paired(160), paired(480)
    one.data['data'].values[0, 0] += 7
    one.data['weight'] = xr.ones_like(one.data.data) * 2
    two.data['weight'] = xr.ones_like(two.data.data) * 3
    model = ExperimentModel(datasets={'one': one, 'two': two}, scale={'one': 1, 'two': 3},
                            clp_relations=[ClpRelation(source='s1', target='s2', parameter=3)])
    obj = OptimizationObjective(model)
    residual = obj.calculate()
    results = obj.get_result().optimization_results
    expected = np.concatenate([
        (results[label].residuals.values * weight).T.ravel()
        for label, weight in [('one', 2), ('two', 3)]
    ])
    np.testing.assert_allclose(residual, expected, atol=1e-10)


def test_chained_relations_are_rejected():
    one, two = paired(160), paired(160)
    relations = [ClpRelation(source='s1', target='s2', parameter=3),
                 ClpRelation(source='s2', target='s1', parameter=2)]
    obj = OptimizationObjective(ExperimentModel(datasets={'one': one, 'two': two}, clp_relations=relations))
    with pytest.raises(ValueError, match='Chained or multiple'):
        obj.calculate()


def test_reordered_global_labels_are_paired_by_name():
    one, two = paired(160), paired(160)
    for model in (one, two):
        model.global_elements[0].compartments = ['s2', 's1']
    obj = OptimizationObjective(ExperimentModel(datasets={'one': one, 'two': two},
        clp_relations=[ClpRelation(source='s1', target='s2', parameter=3)]))
    result = obj.get_result()
    amplitudes = result.optimization_results['one'].elements['one'].amplitudes
    assert amplitudes.global_clp_label.values.tolist() == ['s1', 's2']
    np.testing.assert_allclose(amplitudes.values, [[2, 0], [0, 6]], atol=1e-10)


def test_nonmatching_labels_keep_positional_pairing():
    one, two = paired(160), paired(160)
    two.global_elements[0].compartments = ['s1', 'different']
    obj = OptimizationObjective(ExperimentModel(datasets={'one': one, 'two': two}))
    result = obj.get_result()
    assert result.clp_size == 3
    assert result.optimization_results['two'].elements['two'].amplitudes.global_clp_label.values.tolist() == ['s1', 'different']


def test_cartesian_full_models_share_clps():
    one, two = paired(80), paired(240)
    for model in (one, two):
        model.global_composition = 'cartesian'
    objective = OptimizationObjective(ExperimentModel(datasets={'one': one, 'two': two},
                                                      scale={'one': 1, 'two': 3}))
    np.testing.assert_allclose(objective.calculate(), 0, atol=1e-10)
    result = objective.get_result()
    assert result.clp_size == 4
    for label in ('one', 'two'):
        assert result.optimization_results[label].elements[label].amplitudes.shape == (2, 2)


def test_mixed_full_and_ordinary_models_are_rejected():
    one, two = paired(160), paired(160)
    two.global_elements = None
    with pytest.raises(ValueError, match='cannot be mixed'):
        OptimizationObjective(ExperimentModel(datasets={'one': one, 'two': two}))


def test_mixed_full_model_modes_are_rejected():
    one, two = paired(160), paired(160)
    two.global_composition = 'cartesian'
    with pytest.raises(ValueError, match='same composition mode'):
        OptimizationObjective(ExperimentModel(datasets={'one': one, 'two': two}))


def test_full_link_does_not_require_aligned_global_grids():
    one, two = paired(160), paired(160)
    two.data = two.data.assign_coords(global_dim=[0, 3, 7, 11])
    objective = OptimizationObjective(ExperimentModel(datasets={'one': one, 'two': two},
                                                      clp_link_tolerance=2))
    np.testing.assert_allclose(objective.calculate(), 0, atol=1e-10)


def test_fitted_data_is_input_minus_residual():
    one, two = paired(160), paired(480)
    one.data['data'].values[0, 0] += 7
    model = ExperimentModel(datasets={'one': one, 'two': two}, scale={'one': 1, 'two': 3},
                            clp_relations=[ClpRelation(source='s1', target='s2', parameter=3)])
    results = OptimizationObjective(model).get_result().optimization_results
    for result in results.values():
        xr.testing.assert_allclose(result.fitted_data, result.input_data - result.residuals)


def test_linked_full_linear_uncertainty_includes_shared_solve():
    from glotaran.optimization.clp_standard_error import _calculate_linear_variance

    one, two = paired(160), paired(480)
    one.data['data'].values[0, 0] += 7
    model = ExperimentModel(datasets={'one': one, 'two': two}, scale={'one': 1, 'two': 3},
                            clp_relations=[ClpRelation(source='s1', target='s2', parameter=3)])
    objective = OptimizationObjective(model)
    results = objective.get_result().optimization_results
    clps = {label: result.fit_decomposition.clp for label, result in results.items()}
    variance = _calculate_linear_variance(objective, clps)
    assert variance['one'][0, 0] > 0
    assert variance['one'][1, 1] == pytest.approx(9 * variance['one'][0, 0])
    np.testing.assert_array_equal(variance['one'], variance['two'])


def test_global_element_scales_are_normalized_each_evaluation():
    one, two = paired(160), paired(160)
    for model in (one, two):
        second = deepcopy(model.global_elements[0])
        second.label = 'second'
        model.global_elements.append(second)
        model.global_element_scale = {model.global_elements[0].label: 1, 'second': 3}
    obj = OptimizationObjective(ExperimentModel(datasets={'one': one, 'two': two},
        clp_relations=[ClpRelation(source='s1', target='s2', parameter=3)]))
    amplitudes = obj.get_result().optimization_results['one'].elements['one'].amplitudes
    np.testing.assert_allclose(amplitudes.values, [[2, 0], [0, 6]], atol=1e-10)


def test_global_scale_normalization_uses_current_values():
    one, two = paired(160), paired(160)
    for model in (one, two):
        second = deepcopy(model.global_elements[0])
        second.label = 'second'
        second.value = 8
        model.global_elements.append(second)
        model.global_element_scale = {model.global_elements[0].label: 1, 'second': 3}
    obj = OptimizationObjective(ExperimentModel(datasets={'one': one, 'two': two},
        clp_relations=[ClpRelation(source='s1', target='s2', parameter=3)]))
    first = obj.get_result().optimization_results['one'].elements['one'].amplitudes.values[0, 0]
    for model in obj.model.datasets.values():
        model.global_element_scale[model.global_elements[0].label] = 3
        model.global_element_scale['second'] = 1
    second = obj.get_result().optimization_results['one'].elements['one'].amplitudes.values[0, 0]
    assert first == pytest.approx(160 / (5 * 7 * 4))
    assert second == pytest.approx(160 / (5 * 5 * 4))


def test_skewed_gaussian_sum_first_amplitude_scales_full_sum():
    axis = np.array([0., 1., 2.])
    shape = SpectralShapeSkewedGaussianSum(type='skewed-gaussian-sum',
        amplitude=[2., 3.], location=[0., 1.], width=[2., 2.], skewness=[0., 0.])
    g0 = np.exp(-np.log(2) * (axis - 0.) ** 2)
    g1 = np.exp(-np.log(2) * (axis - 1.) ** 2)
    np.testing.assert_allclose(shape.calculate(axis), 2 * (g0 + 3 * g1))
