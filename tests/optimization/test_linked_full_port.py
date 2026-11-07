from copy import deepcopy

import numpy as np
import pytest
import xarray as xr

from glotaran.model.clp_relation import ClpRelation
from glotaran.model.experiment_model import ExperimentModel
from glotaran.builtin.elements.spectral.shape import SpectralShapeSkewedGaussianSum
from glotaran.optimization.objective import OptimizationObjective
from glotaran.optimization.data import OptimizationData
from glotaran.optimization.matrix import OptimizationMatrix
from glotaran.parameter import Parameter
from tests.optimization.data import TestDataModelGlobal


def paired(value):
    model = deepcopy(TestDataModelGlobal)
    model.global_composition = 'paired'
    model.elements[0].compartments = ['s1', 's2']
    model.global_elements[0].compartments = ['s1', 's2']
    model.data['data'].values[:] = value
    return model


def reordered_exponential_model():
    from tests.optimization.elements import TestElementExponential

    model = paired(0)
    model.elements = [TestElementExponential(
        type='test-element-exponential', label='local', dimension='model_dim',
        is_index_dependent=False, compartments=['s1', 's2'], rates=[0.1, 0.4])]
    model.global_elements = [TestElementExponential(
        type='test-element-exponential', label='global', dimension='global_dim',
        is_index_dependent=False, compartments=['s2', 's1'], rates=[0.3, 0.7])]
    model.element_scale = None
    model.global_element_scale = None
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
        model.align_paired_clp_labels = True
    obj = OptimizationObjective(ExperimentModel(datasets={'one': one, 'two': two},
        clp_relations=[ClpRelation(source='s1', target='s2', parameter=3)]))
    result = obj.get_result()
    amplitudes = result.optimization_results['one'].elements['one'].amplitudes
    assert amplitudes.global_clp_label.values.tolist() == ['s1', 's2']
    np.testing.assert_allclose(amplitudes.values, [[2, 0], [0, 6]], atol=1e-10)


@pytest.mark.parametrize('by_name', [False, True])
@pytest.mark.parametrize('linked', [False, True])
def test_pairing_order_is_explicit_and_consistent_for_single_and_linked(by_name, linked):
    model = reordered_exponential_model()
    model.align_paired_clp_labels = by_name
    x, y = model.data.model_dim.values, model.data.global_dim.values
    global_rates = [0.7, 0.3] if by_name else [0.3, 0.7]
    expected = (2 * np.outer(np.exp(-0.1 * x), np.exp(-global_rates[0] * y)) +
                3 * np.outer(np.exp(-0.4 * x), np.exp(-global_rates[1] * y)))
    model.data['data'].values[:] = expected
    datasets = {'one': model}
    if linked:
        datasets['two'] = deepcopy(model)
    objective = OptimizationObjective(ExperimentModel(datasets=datasets))
    result = objective.get_result()
    assert result.clp_size == 2
    for output in result.optimization_results.values():
        amplitudes = next(iter(output.elements.values())).amplitudes
        assert amplitudes.global_clp_label.values.tolist() == (['s1', 's2'] if by_name else ['s2', 's1'])
        np.testing.assert_allclose(amplitudes.values, [[2, 0], [0, 3]], atol=1e-9)
        np.testing.assert_allclose(output.fitted_data.values, expected, atol=1e-9)


@pytest.mark.parametrize('by_name', [False, True])
def test_simulation_uses_the_same_explicit_pairing(by_name):
    from glotaran.simulation import simulate

    model = reordered_exponential_model()
    model.align_paired_clp_labels = by_name
    x, y = model.data.model_dim.values, model.data.global_dim.values
    global_rates = [0.7, 0.3] if by_name else [0.3, 0.7]
    expected = (np.outer(np.exp(-0.1 * x), np.exp(-global_rates[0] * y)) +
                np.outer(np.exp(-0.4 * x), np.exp(-global_rates[1] * y)))
    output = simulate(model, {}, None, {'model_dim': x, 'global_dim': y})
    np.testing.assert_allclose(output.data.values, expected, atol=1e-9)


def test_name_alignment_rejects_unmatched_labels():
    one = paired(160)
    one.global_elements[0].compartments = ['s1', 'different']
    one.align_paired_clp_labels = True
    with pytest.raises(Exception, match='matching unique'):
        OptimizationObjective(ExperimentModel(datasets={'one': one})).calculate()


def test_single_global_result_uses_experiment_nnls():
    model = paired(-10)
    # Dataset defaults to variable_projection; the experiment's NNLS controls
    # the objective and must control the reported solution as well.
    objective = OptimizationObjective(ExperimentModel(
        datasets={'one': model}, residual_function='non_negative_least_squares'))
    residual = objective.calculate()
    result = objective.get_result().optimization_results['one']
    np.testing.assert_allclose(residual, result.residuals.values.T.ravel(), atol=1e-10)
    np.testing.assert_allclose(result.fitted_data, 0, atol=1e-10)


@pytest.mark.parametrize(('composition', 'model_count', 'global_count', 'expected'), [
    ('cartesian', 3, 4, 12), ('cartesian', 1, 1, 1), ('paired', 3, 3, 3),
])
def test_single_full_model_counts_fitted_clps(composition, model_count, global_count, expected):
    model = paired(10)
    model.global_composition = composition
    model.elements[0].compartments = [f'm{i}' for i in range(model_count)]
    model.global_elements[0].compartments = [f'g{i}' for i in range(global_count)]
    result = OptimizationObjective(ExperimentModel(datasets={'one': model})).get_result()
    assert result.clp_size == expected


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
        model.normalize_global_element_scales = True
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
        model.normalize_global_element_scales = True
    obj = OptimizationObjective(ExperimentModel(datasets={'one': one, 'two': two},
        clp_relations=[ClpRelation(source='s1', target='s2', parameter=3)]))
    first = obj.get_result().optimization_results['one'].elements['one'].amplitudes.values[0, 0]
    for model in obj.model.datasets.values():
        model.global_element_scale[model.global_elements[0].label] = 3
        model.global_element_scale['second'] = 1
    second = obj.get_result().optimization_results['one'].elements['one'].amplitudes.values[0, 0]
    assert first == pytest.approx(160 / (5 * 7 * 4))
    assert second == pytest.approx(160 / (5 * 5 * 4))


def test_native_scalar_element_scale_remains_literal_by_default():
    model = paired(0)
    model.element_scale = {model.elements[0].label: 3}
    matrix = OptimizationMatrix.from_data(OptimizationData(model))
    np.testing.assert_allclose(matrix.array, 15)


def test_native_partial_element_scale_keeps_unlisted_default_one():
    model = paired(0)
    other = deepcopy(model.elements[0])
    other.label, other.value = 'other', 8
    model.elements.append(other)
    model.element_scale = {model.elements[0].label: 3}
    matrix = OptimizationMatrix.from_data(OptimizationData(model))
    np.testing.assert_allclose(matrix.array, 23)


def test_opt_in_local_normalization_preserves_dataset_scalar_and_scale_list():
    model = paired(0)
    other = deepcopy(model.elements[0])
    other.label, other.value = 'other', 8
    model.elements.append(other)
    model.element_scale = {model.elements[0].label: 1, 'other': 3}
    model.normalize_element_scales = True
    model.scale_list = [2, 3, 4, 5]
    matrix = OptimizationMatrix.from_data(OptimizationData(model, scale=10))
    np.testing.assert_allclose(matrix.array[:, 0, 0], np.array([2, 3, 4, 5]) * 72.5)


def test_opt_in_scale_normalization_follows_live_parameters():
    model = paired(0)
    other = deepcopy(model.elements[0])
    other.label, other.value = 'other', 8
    model.elements.append(other)
    first = Parameter(label='first', value=1)
    second = Parameter(label='second', value=3)
    model.element_scale = {model.elements[0].label: first, 'other': second}
    model.normalize_element_scales = True
    data = OptimizationData(model)
    np.testing.assert_allclose(OptimizationMatrix.from_data(data).array, 7.25)
    first.value, second.value = 3, 1
    np.testing.assert_allclose(OptimizationMatrix.from_data(data).array, 5.75)


@pytest.mark.parametrize('values', [(0, 0), (float('inf'), 1), (float('nan'), 1)])
def test_opt_in_normalization_rejects_zero_or_nonfinite_sum(values):
    model = paired(0)
    other = deepcopy(model.elements[0])
    other.label = 'other'
    model.elements.append(other)
    model.element_scale = {model.elements[0].label: values[0], 'other': values[1]}
    model.normalize_element_scales = True
    with pytest.raises(ValueError, match='finite and nonzero'):
        OptimizationMatrix.from_data(OptimizationData(model))


def test_opt_in_normalization_rejects_partial_mapping():
    model = paired(0)
    other = deepcopy(model.elements[0])
    other.label = 'other'
    model.elements.append(other)
    model.element_scale = {model.elements[0].label: 3}
    model.normalize_element_scales = True
    with pytest.raises(ValueError, match='every element'):
        OptimizationMatrix.from_data(OptimizationData(model))


def test_skewed_gaussian_sum_first_amplitude_scales_full_sum():
    axis = np.array([0., 1., 2.])
    shape = SpectralShapeSkewedGaussianSum(type='skewed-gaussian-sum',
        amplitude=[2., 3.], location=[0., 1.], width=[2., 2.], skewness=[0., 0.])
    g0 = np.exp(-np.log(2) * (axis - 0.) ** 2)
    g1 = np.exp(-np.log(2) * (axis - 1.) ** 2)
    np.testing.assert_allclose(shape.calculate(axis), 2 * (g0 + 3 * g1))
