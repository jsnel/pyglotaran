from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from glotaran.builtin.items.activation import GaussianActivation
from glotaran.builtin.items.activation import MultiGaussianActivation
from glotaran.builtin.items.activation import MultiMultiGaussianActivation
from glotaran.builtin.items.activation import NormConvMultiMultiGaussianActivation
from glotaran.builtin.items.activation.data_model import ActivationDataModel
from glotaran.model.errors import GlotaranUserError


def test_gaussian_activation():
    activation = GaussianActivation(
        type="gaussian",
        center=1,
        width=10,
        backsweep=2,
        compartments={},
    )
    parameter = activation.parameters(np.array([0]))
    assert len(parameter) == 1
    assert parameter[0].backsweep_period == 2
    assert parameter[0].center == 1
    assert parameter[0].width == 10


def test_multi_gaussian_activation():
    activation = MultiGaussianActivation(
        type="multi-gaussian",
        center=[1, 2],
        width=[10, 20],
        compartments={},
    )
    parameter = activation.parameters(np.array([0]))
    assert len(parameter) == 2
    assert parameter[0].backsweep_period == 0
    assert parameter[0].center == 1
    assert parameter[0].width == 10
    assert parameter[1].center == 2
    assert parameter[1].width == 20


def test_gaussian_activation_shift():
    activation = GaussianActivation(
        type="gaussian",
        center=1,
        width=10,
        shift=[0.5, 1],
        compartments={},
    )
    parameter = activation.parameters(np.array([0, 1]))
    assert len(parameter) == 2
    assert parameter[0][0].center == 0.5
    assert parameter[1][0].center == 0
    with pytest.raises(GlotaranUserError):
        activation.parameters(np.array([0, 1, 2]))


def test_gaussian_activation_dispersion():
    activation = GaussianActivation(
        type="gaussian",
        center=1,
        width=10,
        dispersion_center=100,
        center_dispersion_coefficients=[2],
        width_dispersion_coefficients=[2],
        compartments={},
    )
    parameter = activation.parameters(np.array([0, 100]))
    assert len(parameter) == 2
    assert parameter[0][0].center == -1
    assert parameter[0][0].width == 8
    assert parameter[1][0].center == 1
    assert parameter[1][0].width == 10


def test_gaussian_activation_dispersion_reciproke():
    activation = GaussianActivation(
        type="gaussian",
        center=1,
        width=10,
        dispersion_center=200,
        center_dispersion_coefficients=[2],
        width_dispersion_coefficients=[2],
        reciproke_global_axis=True,
        compartments={},
    )
    parameter = activation.parameters(np.array([100, 200]))
    assert len(parameter) == 2
    assert parameter[0][0].center == 11
    assert parameter[0][0].width == 20
    assert parameter[1][0].center == 1
    assert parameter[1][0].width == 10


def test_gaussian_activation_spline_width_dispersion():
    activation = GaussianActivation(
        type="gaussian",
        center=1,
        width=10,
        width_dispersion_spline_knots=[300.0, 400.0, 500.0],
        width_dispersion_spline_values=[1.0, 2.0, 3.0],
        compartments={},
    )
    parameters = activation.parameters(np.array([300.0, 400.0, 500.0]))
    assert [p[0].width for p in parameters] == pytest.approx([11.0, 12.0, 13.0])
    assert activation.is_index_dependent()


def test_gaussian_activation_spline_width_dispersion_from_wavelength_knots():
    activation = GaussianActivation(
        type="gaussian",
        center=1,
        width=10,
        model_dispersion_with_wavenumber=True,
        width_dispersion_spline_knots=[400.0, 500.0, 600.0],
        width_dispersion_spline_knots_in_wavelength=True,
        width_dispersion_spline_values=[1.0, 2.0, 3.0],
        compartments={},
    )
    axis = np.array([400.0, 500.0, 600.0])
    parameters = activation.parameters(axis)
    assert [p[0].width for p in parameters] == pytest.approx([11.0, 12.0, 13.0])


def test_gaussian_activation_skewed_width_dispersion():
    activation = GaussianActivation(
        type="gaussian",
        center=1,
        width=10,
        width_dispersion_skewed_gaussian_amplitude=2.0,
        width_dispersion_skewed_gaussian_location=400.0,
        width_dispersion_skewed_gaussian_width=100.0,
        width_dispersion_skewed_gaussian_skewness=0.0,
        compartments={},
    )
    parameters = activation.parameters(np.array([400.0]))
    assert parameters[0][0].width == pytest.approx(12.0)


def test_multi_multi_gaussian_activation_group_expansion_and_area_normalization():
    activation = MultiMultiGaussianActivation(
        type="multi-multi-gaussian",
        center=[[0.0, 0.1], [1.0]],
        width=[[0.1, 0.2], [0.4]],
        scale=[[2.0, 0.5], [4.0]],
        compartments={},
    )
    parameters = activation.parameters()
    assert [p.center for p in parameters] == pytest.approx([0.0, 0.1, 1.0])
    assert [p.scale for p in parameters] == pytest.approx([398.9422804, 199.4711402, 797.8845608])
    assert np.sum([p.scale * abs(p.width) for p in parameters]) * np.sqrt(
        2 * np.pi
    ) == pytest.approx(1000.0)


def test_norm_conv_multi_multi_gaussian_uses_per_index_broadening_and_normalization():
    activation = NormConvMultiMultiGaussianActivation(
        type="norm-conv-multi-multi-gaussian",
        center=[[0.0]],
        width=[[0.1]],
        scale=[[1.0]],
        convwidth=[0.05, 0.2],
        compartments={},
    )
    parameters = activation.parameters(np.array([0.0, 1.0]))
    assert parameters[0][0].width == pytest.approx(np.sqrt(0.1**2 + 0.05**2))
    assert parameters[1][0].width == pytest.approx(np.sqrt(0.1**2 + 0.2**2))
    for per_index in parameters:
        normalized_area = np.sum([p.scale * abs(p.width) for p in per_index]) * np.sqrt(2 * np.pi)
        assert normalized_area == pytest.approx(1000.0)
    assert parameters[0][0].scale != pytest.approx(parameters[1][0].scale)


def test_create_result_gaussian_activation():
    """Test create_result with GaussianActivation produces expected dimensions."""
    model = ActivationDataModel(
        elements=[],
        activations={
            "irf": GaussianActivation(
                type="gaussian",
                center=0.1,
                width=0.1,
                compartments={},
            )
        },
    )

    global_axis = np.linspace(400, 600, 5)
    model_axis = np.linspace(-1, 1, 20)

    amplitudes = xr.DataArray(
        np.zeros((5, 3)),
        coords={"wavelength": global_axis, "species": ["s1", "s2", "s3"]},
        dims=["wavelength", "species"],
    )

    concentrations = xr.DataArray(
        np.zeros((20, 3)),
        coords={"time": model_axis, "species": ["s1", "s2", "s3"]},
        dims=["time", "species"],
    )

    result = model.create_result(
        model=model,
        global_dimension="wavelength",
        model_dimension="time",
        amplitudes=amplitudes,
        concentrations=concentrations,
    )

    assert "irf" in result
    assert "trace" in result["irf"]
    assert "shift" in result["irf"]
    assert "center" in result["irf"]
    assert result["irf"]["trace"].dims == ("time",)
    assert result["irf"]["trace"].shape == (model_axis.size,)
    assert result["irf"]["shift"].dims == ("wavelength",)
    assert result["irf"]["shift"].shape == (global_axis.size,)
    assert result["irf"]["center"].dims == ("component_index", "wavelength")
    assert result["irf"]["center"].shape == (1, global_axis.size)
    assert np.allclose(result["irf"]["center"][0], 0.1)


def test_create_result_multi_gaussian_activation():
    """Test create_result with MultiGaussianActivation produces expected dimensions."""
    model = ActivationDataModel(
        elements=[],
        activations={
            "irf": MultiGaussianActivation(
                type="multi-gaussian",
                center=[0.1, 0.5],
                width=[0.1, 0.2],
                compartments={},
            )
        },
    )

    global_axis = np.linspace(400, 600, 5)
    model_axis = np.linspace(-1, 1, 20)

    amplitudes = xr.DataArray(
        np.zeros((5, 3)),
        coords={"wavelength": global_axis, "species": ["s1", "s2", "s3"]},
        dims=["wavelength", "species"],
    )

    concentrations = xr.DataArray(
        np.zeros((20, 3)),
        coords={"time": model_axis, "species": ["s1", "s2", "s3"]},
        dims=["time", "species"],
    )

    result = model.create_result(
        model=model,
        global_dimension="wavelength",
        model_dimension="time",
        amplitudes=amplitudes,
        concentrations=concentrations,
    )

    assert "irf" in result
    assert "trace" in result["irf"]
    assert "shift" in result["irf"]
    assert "center" in result["irf"]
    assert result["irf"]["trace"].dims == ("time",)
    assert result["irf"]["trace"].shape == (model_axis.size,)
    assert result["irf"]["shift"].dims == ("wavelength",)
    assert result["irf"]["shift"].shape == (global_axis.size,)
    assert result["irf"]["center"].dims == ("component_index", "wavelength")
    assert result["irf"]["center"].shape == (2, global_axis.size)
    assert np.allclose(result["irf"]["center"][0], 0.1)
    assert np.allclose(result["irf"]["center"][1], 0.5)
