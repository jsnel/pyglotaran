import warnings
from textwrap import dedent

import numpy as np
import pytest
from attrs import evolve
from scipy.interpolate import CubicSpline

from glotaran.io import load_model
from glotaran.io import load_parameters
from glotaran.model import fill_item
from glotaran.optimization.optimize import optimize
from glotaran.project import Scheme
from glotaran.simulation import simulate

MODEL_BASE = """\
default_megacomplex: decay
dataset:
    dataset1:
        megacomplex: [mc1]
        initial_concentration: j1
        irf: irf1
initial_concentration:
    j1:
        compartments: [s1]
        parameters: [j.1]
megacomplex:
    mc1:
        k_matrix: [k1]
    mc2:
        type: spectral
        shape:
            s1: sh1
k_matrix:
    k1:
        matrix:
            (s1, s1): kinetic.1
shape:
    sh1:
        type: one
"""
MODEL_NO_IRF_DISPERSION = f"""\
{MODEL_BASE}
irf:
    irf1:
        type: gaussian
        center: irf.center
        width: irf.width
"""
MODEL_SIMPLE_IRF_DISPERSION = f"""\
{MODEL_BASE}
irf:
    irf1:
        type: spectral-gaussian
        center: irf.center
        width: irf.width
        dispersion_center: irf.dispersion_center
        center_dispersion_coefficients: [irf.cdc1]
"""
MODEL_MULTI_IRF_DISPERSION = f"""\
{MODEL_BASE}
irf:
    irf1:
        type: spectral-multi-gaussian
        center: [irf.center]
        width: [irf.width]
        dispersion_center: irf.dispersion_center
        center_dispersion_coefficients: [irf.cdc1, irf.cdc2]
        width_dispersion_coefficients: [irf.wdc1]
"""

MODEL_SKEWED_GAUSSIAN_IRF_WIDTH_DISPERSION = f"""\
{MODEL_BASE}
irf:
    irf1:
        type: spectral-gaussian
        center: irf.center
        width: irf.width
        dispersion_center: irf.dispersion_center
        width_dispersion_coefficients: [irf.wdc1]
        width_dispersion_skewed_gaussian_amplitude: irf.sga
        width_dispersion_skewed_gaussian_location: irf.sgl
        width_dispersion_skewed_gaussian_width: irf.sgw
        width_dispersion_skewed_gaussian_skewness: irf.sgs
"""

MODEL_SKEWED_GAUSSIAN_IRF_WIDTH_DISPERSION_WAVENUMBER = f"""\
{MODEL_BASE}
irf:
    irf1:
        type: spectral-gaussian
        center: irf.center
        width: irf.width
        dispersion_center: irf.dispersion_center
        width_dispersion_skewed_gaussian_amplitude: irf.sga
        width_dispersion_skewed_gaussian_location: irf.sgl
        width_dispersion_skewed_gaussian_width: irf.sgw
        width_dispersion_skewed_gaussian_skewness: irf.sgs
        model_dispersion_with_wavenumber: true
"""

MODEL_SPLINE_IRF_WIDTH_DISPERSION = f"""\
{MODEL_BASE}
irf:
    irf1:
        type: spectral-gaussian
        center: irf.center
        width: irf.width
        dispersion_center: irf.dispersion_center
        width_dispersion_spline_knots: [300.0, 400.0, 500.0]
        width_dispersion_spline_values: [irf.wds1, irf.wds2, irf.wds3]
"""

MODEL_SPLINE_IRF_WIDTH_DISPERSION_WAVENUMBER_FROM_WAVELENGTH_KNOTS = f"""\
{MODEL_BASE}
irf:
    irf1:
        type: spectral-gaussian
        center: irf.center
        width: irf.width
        dispersion_center: irf.dispersion_center
        model_dispersion_with_wavenumber: true
        width_dispersion_spline_knots_in_wavelength: true
        width_dispersion_spline_knots: [300.0, 400.0, 500.0]
        width_dispersion_spline_values: [irf.wds1, irf.wds2, irf.wds3]
"""

MODEL_MULTIPULSE_IRF_DISPERSION = f"""\
{MODEL_BASE}
irf:
    irf1:
        type: spectral-multi-gaussian
        center: [irf.center1, irf.center2]
        width: [irf.width]
        dispersion_center: irf.dispersion_center
        center_dispersion_coefficients: [irf.cdc1, irf.cdc2, irf.cdc3]
"""

PARAMETERS_BASE = """\
j:
    - ['1', 1, {'vary': False, 'non-negative': False}]
kinetic:
    - ['1', 0.5, {'non-negative': False}]
"""

MODEL_MULTI_MULTI_GAUSSIAN_IRF = f"""\
{MODEL_BASE}
irf:
    irf1:
        type: multi-multi-gaussian
        center: [[irf.center1, irf.center2], [irf.center3]]
        width: [[irf.width1, irf.width2], [irf.width3]]
        scale: [[irf.scale1, irf.scale2], [irf.scale3]]
"""

MODEL_MULTI_MULTI_GAUSSIAN_IRF_CUSTOM_NORMAREA = f"""\
{MODEL_BASE}
irf:
    irf1:
        type: multi-multi-gaussian
        normarea: 250
        center: [[irf.center1, irf.center2], [irf.center3]]
        width: [[irf.width1, irf.width2], [irf.width3]]
        scale: [[irf.scale1, irf.scale2], [irf.scale3]]
"""

MODEL_MULTI_MULTI_GAUSSIAN_IRF_NO_NORM = f"""\
{MODEL_BASE}
irf:
    irf1:
        type: multi-multi-gaussian
        normalize: false
        center: [[irf.center1, irf.center2], [irf.center3]]
        width: [[irf.width1, irf.width2], [irf.width3]]
        scale: [[irf.scale1, irf.scale2], [irf.scale3]]
"""

MODEL_MULTI_MULTI_GAUSSIAN_IRF_LEGACY_NORM = f"""\
{MODEL_BASE}
irf:
    irf1:
        type: multi-multi-gaussian
        normalize_area: false
        center: [[irf.center1, irf.center2], [irf.center3]]
        width: [[irf.width1, irf.width2], [irf.width3]]
        scale: [[irf.scale1, irf.scale2], [irf.scale3]]
"""

MODEL_CONV_MULTI_MULTI_GAUSSIAN_IRF = f"""\
{MODEL_BASE}
irf:
    irf1:
        type: conv-multi-multi-gaussian
        convwidth: [irf.convwidth]
        center: [[irf.center1, irf.center2], [irf.center3]]
        width: [[irf.width1, irf.width2], [irf.width3]]
        scale: [[irf.scale1, irf.scale2], [irf.scale3]]
"""

MODEL_NORM_CONV_MULTI_MULTI_GAUSSIAN_IRF = f"""\
{MODEL_BASE}
irf:
    irf1:
        type: norm-conv-multi-multi-gaussian
        convwidth: [irf.convwidth]
        center: [[irf.center1, irf.center2], [irf.center3]]
        width: [[irf.width1, irf.width2], [irf.width3]]
        scale: [[irf.scale1, irf.scale2], [irf.scale3]]
"""

PARAMETERS_CONV_MULTI_MULTI_GAUSSIAN_IRF = f"""\
{PARAMETERS_BASE}
irf:
    - ["center1", 0.1]
    - ["center2", 0.2]
    - ["center3", 0.4]
    - ["width1", 0.10]
    - ["width2", 0.20]
    - ["width3", 0.40]
    - ["scale1", 2.0]
    - ["scale2", 0.5]
    - ["scale3", 4.0]
    - ["convwidth", 0.05]
"""

PARAMETERS_MULTI_MULTI_GAUSSIAN_IRF = f"""\
{PARAMETERS_BASE}
irf:
    - ["center1", 0.1]
    - ["center2", 0.2]
    - ["center3", 0.4]
    - ["width1", 0.10]
    - ["width2", 0.20]
    - ["width3", 0.40]
    - ["scale1", 2.0]
    - ["scale2", 0.5]
    - ["scale3", 4.0]
"""

PARAMETERS_NO_IRF_DISPERSION = f"""\
{PARAMETERS_BASE}
irf:
    - ['center', 0.3]
    - ['width', 0.1]
"""

PARAMETERS_SIMPLE_IRF_DISPERSION = f"""\
{PARAMETERS_BASE}
irf:
    - ['center', 0.3]
    - ['width', 0.1]
    - ['dispersion_center', 400, {{'vary': False}}]
    - ['cdc1', 0.5]
"""

# What is this?
PARAMETERS_MULTI_IRF_DISPERSION = f"""\
{PARAMETERS_BASE}
irf:
    - ["center", 0.3]
    - ["width", 0.1]
    - ["dispersion_center", 400, {{"vary": False}}]
    - ["cdc1", 0.1]
    - ["cdc2", 0.01]
    - ["wdc1", 0.025]
"""

PARAMETERS_SKEWED_GAUSSIAN_IRF_WIDTH_DISPERSION = f"""\
{PARAMETERS_BASE}
irf:
    - ["center", 0.3]
    - ["width", 0.1]
    - ["dispersion_center", 400, {{"vary": False}}]
    - ["wdc1", 0.01]
    - ["sga", 0.04]
    - ["sgl", 410, {{"vary": False}}]
    - ["sgw", 25]
    - ["sgs", 0.35]
"""

PARAMETERS_SKEWED_GAUSSIAN_IRF_WIDTH_DISPERSION_WAVENUMBER = f"""\
{PARAMETERS_BASE}
irf:
    - ["center", 0.3]
    - ["width", 0.1]
    - ["dispersion_center", 400, {{"vary": False}}]
    - ["sga", 0.04]
    - ["sgl", 2.5, {{"vary": False}}]
    - ["sgw", 0.12]
    - ["sgs", -0.25]
"""

PARAMETERS_SPLINE_IRF_WIDTH_DISPERSION = f"""\
{PARAMETERS_BASE}
irf:
    - ["center", 0.3]
    - ["width", 0.1]
    - ["dispersion_center", 400, {{"vary": False}}]
    - ["wds1", 0.02]
    - ["wds2", 0.03]
    - ["wds3", 0.05]
"""

PARAMETERS_MULTIPULSE_IRF_DISPERSION = f"""\
{PARAMETERS_BASE}
irf:
    - ["center1", 0.3]
    - ["center2", 0.4]
    - ['width', 0.1]
    - ['dispersion_center', 400, {{'vary': False}}]
    - ["cdc1", 0.5]
    - ["cdc2", 0.1]
    - ["cdc3", -0.01]
"""


def _time_axis():
    time_p1 = np.linspace(-1, 1, 20, endpoint=False)
    time_p2 = np.linspace(1, 2, 10, endpoint=False)
    time_p3 = np.geomspace(2, 20, num=20)
    return np.array(np.concatenate([time_p1, time_p2, time_p3]))


def _spectral_axis():
    return np.linspace(300, 500, 3)


def _calculate_irf_position(
    index,
    center,
    dispersion_center=None,
    center_dispersion_coefficients=None,
    model_dispersion_with_wavenumber=False,
):
    if center_dispersion_coefficients is None:
        center_dispersion_coefficients = []
    if dispersion_center is not None:
        if model_dispersion_with_wavenumber:
            distance = 1e3 / index - 1e3 / dispersion_center
        else:
            distance = (index - dispersion_center) / 100
        for i, coefficient in enumerate(center_dispersion_coefficients):
            center += coefficient * np.power(distance, i + 1)
    return center


def _calculate_irf_width(
    index,
    irf_width,
    dispersion_center=None,
    width_dispersion_coefficients=None,
    skewed_gaussian_parameters=None,
    model_dispersion_with_wavenumber=False,
    width_dispersion_spline_knots=None,
    width_dispersion_spline_values=None,
    width_dispersion_spline_knots_in_wavelength=False,
):
    if width_dispersion_coefficients is None:
        width_dispersion_coefficients = []
    if dispersion_center is not None:
        if model_dispersion_with_wavenumber:
            distance = 1e3 / index - 1e3 / dispersion_center
        else:
            distance = (index - dispersion_center) / 100
        for i, coefficient in enumerate(width_dispersion_coefficients):
            irf_width += coefficient * np.power(distance, i + 1)
    if width_dispersion_spline_knots and width_dispersion_spline_values:
        transformed_index = 1e3 / index if model_dispersion_with_wavenumber else index
        knots = np.asarray([float(knot) for knot in width_dispersion_spline_knots])
        values = np.asarray([float(value) for value in width_dispersion_spline_values])
        if width_dispersion_spline_knots_in_wavelength and model_dispersion_with_wavenumber:
            knots = 1e3 / knots
        order = np.argsort(knots)
        knots = knots[order]
        values = values[order]
        irf_width += float(
            CubicSpline(knots, values, bc_type="natural")(transformed_index)
        )
    if skewed_gaussian_parameters is not None:
        amplitude, location, skew_width, skewness = skewed_gaussian_parameters
        transformed_index = 1e3 / index if model_dispersion_with_wavenumber else index
        if np.allclose(skewness, 0):
            irf_width += amplitude * np.exp(
                -np.log(2) * np.square(2 * (transformed_index - location) / skew_width)
            )
        else:
            log_argument = 1 + (2 * skewness * (transformed_index - location) / skew_width)
            if log_argument > 0:
                irf_width += amplitude * np.exp(
                    -np.log(2) * np.square(np.log(log_argument) / skewness)
                )
    return irf_width


class NoIrfDispersion:
    model = load_model(MODEL_NO_IRF_DISPERSION, format_name="yml_str")
    parameters = load_parameters(PARAMETERS_NO_IRF_DISPERSION, format_name="yml_str")
    axis = {"time": _time_axis(), "spectral": _spectral_axis()}


class SimpleIrfDispersion:
    model = load_model(MODEL_SIMPLE_IRF_DISPERSION, format_name="yml_str")
    parameters = load_parameters(PARAMETERS_SIMPLE_IRF_DISPERSION, format_name="yml_str")
    axis = {"time": _time_axis(), "spectral": _spectral_axis()}


class MultiIrfDispersion:
    model = load_model(MODEL_MULTI_IRF_DISPERSION, format_name="yml_str")
    parameters = load_parameters(PARAMETERS_MULTI_IRF_DISPERSION, format_name="yml_str")
    axis = {"time": _time_axis(), "spectral": _spectral_axis()}


class MultiCenterIrfDispersion:
    model = load_model(MODEL_MULTIPULSE_IRF_DISPERSION, format_name="yml_str")
    parameters = load_parameters(PARAMETERS_MULTIPULSE_IRF_DISPERSION, format_name="yml_str")
    axis = {"time": _time_axis(), "spectral": _spectral_axis()}


class SkewedGaussianIrfWidthDispersion:
    model = load_model(MODEL_SKEWED_GAUSSIAN_IRF_WIDTH_DISPERSION, format_name="yml_str")
    parameters = load_parameters(
        PARAMETERS_SKEWED_GAUSSIAN_IRF_WIDTH_DISPERSION, format_name="yml_str"
    )
    axis = {"time": _time_axis(), "spectral": _spectral_axis()}


@pytest.mark.parametrize(
    "suite",
    [
        NoIrfDispersion,
        SimpleIrfDispersion,
        MultiIrfDispersion,
        MultiCenterIrfDispersion,
    ],
)
def test_spectral_irf(suite):
    model = suite.model
    assert model.valid(), model.validate()

    parameters = suite.parameters
    assert model.valid(parameters), model.validate(parameters)

    sim_model = evolve(model)
    sim_model.dataset["dataset1"].global_megacomplex = ["mc2"]
    print(sim_model)
    dataset = simulate(sim_model, "dataset1", parameters, suite.axis)

    assert dataset.data.shape == (suite.axis["time"].size, suite.axis["spectral"].size)

    data = {"dataset1": dataset}

    scheme = Scheme(
        model=model,
        parameters=parameters,
        data=data,
        maximum_number_function_evaluations=20,
    )
    result = optimize(scheme)

    for param in result.optimized_parameters.all():
        assert np.allclose(param.value, parameters.get(param.label).value, rtol=1e-1)

    resultdata = result.data["dataset1"]

    # print(resultdata)
    assert np.array_equal(dataset["time"], resultdata["time"])
    assert np.array_equal(dataset["spectral"], resultdata["spectral"])
    assert dataset.data.shape == resultdata.data.shape
    assert dataset.data.shape == resultdata.fitted_data.shape
    # assert np.allclose(dataset.data, resultdata.fitted_data, atol=1e-14)

    fit_data_max_at_start = resultdata.fitted_data.isel(spectral=0).argmax(axis=0)
    fit_data_max_at_end = resultdata.fitted_data.isel(spectral=-1).argmax(axis=0)

    if suite is NoIrfDispersion:
        assert "center_dispersion_1" not in resultdata
        assert fit_data_max_at_start == fit_data_max_at_end
    else:
        assert "center_dispersion_1" in resultdata
        assert fit_data_max_at_start != fit_data_max_at_end
        if abs(fit_data_max_at_start - fit_data_max_at_end) < 3:
            warnings.warn(
                dedent(
                    """
                    Bad test, one of the following could be the case:
                    - dispersion too small
                    - spectral window to small
                    - time resolution (around the maximum of the IRF) too low"
                    """
                )
            )

        for x in suite.axis["spectral"]:
            # calculated irf location
            irf = fill_item(suite.model.irf["irf1"], suite.model, result.optimized_parameters)
            model_irf_center = irf.center
            model_dispersion_center = irf.dispersion_center
            model_center_dispersion_coefficients = irf.center_dispersion_coefficients
            calc_irf_location_at_x = _calculate_irf_position(
                x,
                model_irf_center,
                model_dispersion_center,
                model_center_dispersion_coefficients,
                irf.model_dispersion_with_wavenumber,
            )
            # fitted irf location
            fitted_irf_loc_at_x = resultdata["irf_center_location"].sel(spectral=x)
            assert np.allclose(calc_irf_location_at_x, fitted_irf_loc_at_x.values), dedent(
                f"""
                Error in {suite.__name__} comparing irf_center_location,
                - diff={calc_irf_location_at_x-fitted_irf_loc_at_x.values}
                """
            )


@pytest.mark.parametrize(
    "model_text, parameter_text, spectral_axis",
    [
        (
            MODEL_SKEWED_GAUSSIAN_IRF_WIDTH_DISPERSION,
            PARAMETERS_SKEWED_GAUSSIAN_IRF_WIDTH_DISPERSION,
            np.asarray([380.0, 400.0, 430.0]),
        ),
        (
            MODEL_SKEWED_GAUSSIAN_IRF_WIDTH_DISPERSION_WAVENUMBER,
            PARAMETERS_SKEWED_GAUSSIAN_IRF_WIDTH_DISPERSION_WAVENUMBER,
            np.asarray([380.0, 400.0, 430.0]),
        ),
    ],
)
def test_spectral_irf_skewed_gaussian_width_dispersion(model_text, parameter_text, spectral_axis):
    model = load_model(model_text, format_name="yml_str")
    parameters = load_parameters(parameter_text, format_name="yml_str")

    irf = fill_item(model.irf["irf1"], model, parameters)
    widths = np.asarray(
        [irf.parameter(index, spectral_axis)[1][0] for index, _ in enumerate(spectral_axis)]
    )

    expected = np.asarray(
        [
            _calculate_irf_width(
                spectral_value,
                irf.width,
                irf.dispersion_center,
                irf.width_dispersion_coefficients,
                (
                    irf.width_dispersion_skewed_gaussian_amplitude,
                    irf.width_dispersion_skewed_gaussian_location,
                    irf.width_dispersion_skewed_gaussian_width,
                    irf.width_dispersion_skewed_gaussian_skewness,
                ),
                irf.model_dispersion_with_wavenumber,
            )
            for spectral_value in spectral_axis
        ]
    )

    assert np.allclose(widths, expected)


def test_spectral_irf_spline_width_dispersion():
    model = load_model(MODEL_SPLINE_IRF_WIDTH_DISPERSION, format_name="yml_str")
    parameters = load_parameters(PARAMETERS_SPLINE_IRF_WIDTH_DISPERSION, format_name="yml_str")

    irf = fill_item(model.irf["irf1"], model, parameters)
    spectral_axis = np.asarray([320.0, 400.0, 470.0])
    widths = np.asarray(
        [irf.parameter(index, spectral_axis)[1][0] for index, _ in enumerate(spectral_axis)]
    )

    expected = np.asarray(
        [
            _calculate_irf_width(
                spectral_value,
                irf.width,
                irf.dispersion_center,
                irf.width_dispersion_coefficients,
                model_dispersion_with_wavenumber=irf.model_dispersion_with_wavenumber,
                width_dispersion_spline_knots=irf.width_dispersion_spline_knots,
                width_dispersion_spline_values=irf.width_dispersion_spline_values,
            )
            for spectral_value in spectral_axis
        ]
    )

    assert irf.is_index_dependent()
    assert np.allclose(widths, expected)


def test_spectral_irf_spline_width_dispersion_wavenumber_from_wavelength_knots():
    model = load_model(
        MODEL_SPLINE_IRF_WIDTH_DISPERSION_WAVENUMBER_FROM_WAVELENGTH_KNOTS,
        format_name="yml_str",
    )
    parameters = load_parameters(PARAMETERS_SPLINE_IRF_WIDTH_DISPERSION, format_name="yml_str")

    irf = fill_item(model.irf["irf1"], model, parameters)
    spectral_axis = np.asarray([320.0, 400.0, 470.0])
    widths = np.asarray(
        [irf.parameter(index, spectral_axis)[1][0] for index, _ in enumerate(spectral_axis)]
    )

    expected = np.asarray(
        [
            _calculate_irf_width(
                spectral_value,
                irf.width,
                irf.dispersion_center,
                irf.width_dispersion_coefficients,
                model_dispersion_with_wavenumber=irf.model_dispersion_with_wavenumber,
                width_dispersion_spline_knots=irf.width_dispersion_spline_knots,
                width_dispersion_spline_values=irf.width_dispersion_spline_values,
                width_dispersion_spline_knots_in_wavelength=(
                    irf.width_dispersion_spline_knots_in_wavelength
                ),
            )
            for spectral_value in spectral_axis
        ]
    )

    assert irf.is_index_dependent()
    assert np.allclose(widths, expected)


def test_spectral_irf_with_width_skewed_gaussian_dispersion_runs():
    model = SkewedGaussianIrfWidthDispersion.model
    parameters = SkewedGaussianIrfWidthDispersion.parameters
    axis = SkewedGaussianIrfWidthDispersion.axis

    assert model.valid(), model.validate()
    assert model.valid(parameters), model.validate(parameters)

    sim_model = evolve(model)
    sim_model.dataset["dataset1"].global_megacomplex = ["mc2"]
    dataset = simulate(sim_model, "dataset1", parameters, axis)

    scheme = Scheme(
        model=model,
        parameters=parameters,
        data={"dataset1": dataset},
        maximum_number_function_evaluations=5,
    )
    result = optimize(scheme)

    assert dataset.data.shape == result.data["dataset1"].fitted_data.shape


def test_multi_multi_gaussian_irf_area_normalization():
    model = load_model(MODEL_MULTI_MULTI_GAUSSIAN_IRF, format_name="yml_str")
    parameters = load_parameters(PARAMETERS_MULTI_MULTI_GAUSSIAN_IRF, format_name="yml_str")
    irf = fill_item(model.irf["irf1"], model, parameters)

    centers, widths, scales, *_ = irf.parameter(0, np.asarray([400.0]))

    expected_scales_before_norm = np.asarray([2.0, 1.0, 4.0])
    expected_widths = np.asarray([0.10, 0.20, 0.40])
    expected_total_area = np.sum(expected_scales_before_norm * expected_widths) * np.sqrt(
        2 * np.pi
    )
    expected_normarea = 1000.0

    assert np.allclose(widths, expected_widths)
    assert irf.normarea == expected_normarea
    assert np.allclose(
        scales, expected_scales_before_norm * (expected_normarea / expected_total_area)
    )
    assert np.allclose(np.sum(scales * np.abs(widths)) * np.sqrt(2 * np.pi), expected_normarea)
    assert len(centers) == len(widths) == len(scales)


def test_multi_multi_gaussian_irf_custom_normarea_from_yaml():
    model = load_model(MODEL_MULTI_MULTI_GAUSSIAN_IRF_CUSTOM_NORMAREA, format_name="yml_str")
    parameters = load_parameters(PARAMETERS_MULTI_MULTI_GAUSSIAN_IRF, format_name="yml_str")
    irf = fill_item(model.irf["irf1"], model, parameters)

    _, widths, scales, *_ = irf.parameter(0, np.asarray([400.0]))

    expected_scales_before_norm = np.asarray([2.0, 1.0, 4.0])
    expected_total_area = np.sum(expected_scales_before_norm * widths) * np.sqrt(2 * np.pi)

    assert irf.normarea == 250
    assert np.allclose(scales, expected_scales_before_norm * (irf.normarea / expected_total_area))
    assert np.allclose(np.sum(scales * np.abs(widths)) * np.sqrt(2 * np.pi), irf.normarea)


def test_multi_multi_gaussian_irf_without_normalization():
    model = load_model(MODEL_MULTI_MULTI_GAUSSIAN_IRF_NO_NORM, format_name="yml_str")
    parameters = load_parameters(PARAMETERS_MULTI_MULTI_GAUSSIAN_IRF, format_name="yml_str")
    irf = fill_item(model.irf["irf1"], model, parameters)

    _, widths, scales, *_ = irf.parameter(0, np.asarray([400.0]))

    assert np.allclose(widths, np.asarray([0.10, 0.20, 0.40]))
    assert np.allclose(scales, np.asarray([2.0, 1.0, 4.0]))


def test_multi_multi_gaussian_irf_legacy_normalization():
    """normalize_area=False keeps the legacy transformed scales unchanged."""
    model = load_model(MODEL_MULTI_MULTI_GAUSSIAN_IRF_LEGACY_NORM, format_name="yml_str")
    parameters = load_parameters(PARAMETERS_MULTI_MULTI_GAUSSIAN_IRF, format_name="yml_str")
    irf = fill_item(model.irf["irf1"], model, parameters)

    _, widths, scales, *_ = irf.parameter(0, np.asarray([400.0]))

    assert np.allclose(widths, np.asarray([0.10, 0.20, 0.40]))
    assert np.allclose(scales, np.asarray([2.0, 1.0, 4.0]))


def test_conv_multi_multi_gaussian_irf_width_broadening():
    """conv-multi-multi-gaussian returns widths broadened in quadrature with convwidth."""
    model = load_model(MODEL_CONV_MULTI_MULTI_GAUSSIAN_IRF, format_name="yml_str")
    parameters = load_parameters(PARAMETERS_CONV_MULTI_MULTI_GAUSSIAN_IRF, format_name="yml_str")
    irf = fill_item(model.irf["irf1"], model, parameters)

    _, widths_conv, scales_conv, *_ = irf.parameter(0, np.asarray([400.0]))

    # Base widths from the parent multi-multi-gaussian (absolute, not relative)
    base_widths = np.asarray([0.10, 0.20, 0.40])
    convwidth_val = 0.05  # convwidth[0]
    expected_widths = np.sqrt(convwidth_val**2 + base_widths**2)

    assert np.allclose(widths_conv, expected_widths)
    # Scales are normalized using the *base* widths (normalization happens before broadening)
    # so the area after broadening is larger than normarea; just check shape and positivity.
    assert len(scales_conv) == 3
    assert np.all(scales_conv > 0)
    assert np.sum(scales_conv * np.abs(widths_conv)) * np.sqrt(2 * np.pi) > irf.normarea


def test_norm_conv_multi_multi_gaussian_irf_true_area_normalization():
    """norm-conv-multi-multi-gaussian normalizes using broadened (true) widths."""
    model = load_model(MODEL_NORM_CONV_MULTI_MULTI_GAUSSIAN_IRF, format_name="yml_str")
    parameters = load_parameters(PARAMETERS_CONV_MULTI_MULTI_GAUSSIAN_IRF, format_name="yml_str")
    irf = fill_item(model.irf["irf1"], model, parameters)

    _, widths_conv, scales_conv, *_ = irf.parameter(0, np.asarray([400.0]))

    base_widths = np.asarray([0.10, 0.20, 0.40])
    convwidth_val = 0.05
    expected_widths = np.sqrt(convwidth_val**2 + base_widths**2)

    assert np.allclose(widths_conv, expected_widths)
    assert np.allclose(
        np.sum(scales_conv * np.abs(widths_conv)) * np.sqrt(2 * np.pi), irf.normarea
    )
