import numpy as np
import pytest
import xarray as xr

from glotaran.builtin.megacomplexes.ctilde import CtildeMegacomplex
from glotaran.builtin.megacomplexes.decay import DecayParallelMegacomplex
from glotaran.model import Model
from glotaran.optimization.optimize import optimize
from glotaran.parameter import Parameters
from glotaran.project import Scheme


@pytest.mark.parametrize("link_clp", [False, True])
def test_ctilde_dataset_uses_direct_weighted_residual_and_source_relations(link_clp):
    model = Model.create_class_from_megacomplexes([DecayParallelMegacomplex, CtildeMegacomplex])(
        dataset_groups={
            "streaknpq4": {
                "residual_function": "non_negative_least_squares",
                "link_clp": link_clp,
            }
        },
        megacomplex={
            "decay": {
                "type": "decay-parallel",
                "compartments": ["a", "b"],
                "rates": ["rate.a", "rate.b"],
            },
            "ctilde": {
                "type": "ctilde",
                "source_dataset": "source",
                "regularization_weight": 1.0,
            },
        },
        dataset={
            "source": {"group": "streaknpq4", "megacomplex": ["decay"]},
            "source_Ctilde": {"group": "streaknpq4", "megacomplex": ["ctilde"]},
        },
        clp_relations=[{"source": "a", "target": "b", "parameter": "relation.factor"}],
        weights=[{"datasets": ["source_Ctilde"], "value": 2.0}],
    )
    parameters = Parameters.from_list([["rate.a", 0.4], ["rate.b", 1.5], ["relation.factor", 1.0]])

    time = np.arange(0.0, 5.0)
    spectral = np.array([500.0, 600.0])
    reference_ctilde = 0.5 * (np.exp(-0.4 * time) + np.exp(-1.5 * time))
    etilde = np.array([[1.0], [2.0]])
    source_data = xr.Dataset(
        {
            "data": (
                ("time", "spectral"),
                reference_ctilde[:, None] * etilde[:, 0][None, :],
            )
        },
        coords={"time": time, "spectral": spectral},
    )
    scheme = Scheme(
        model=model,
        parameters=parameters,
        data={"source": source_data},
    )

    result = optimize(scheme, verbose=False, raise_exception=True)
    ctilde_result = result.data["source_Ctilde"]

    assert "Ctilde" in result.data["source"]
    assert "Etilde" in result.data["source"]
    np.testing.assert_allclose(
        result.data["source"]["Ctilde"].values,
        reference_ctilde[:, None],
        atol=1e-10,
    )
    np.testing.assert_allclose(
        result.data["source"]["Etilde"].values,
        etilde,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        ctilde_result.data.values,
        reference_ctilde[:, None],
        atol=1e-10,
    )
    np.testing.assert_allclose(
        ctilde_result.residual.values,
        ctilde_result.data.values - ctilde_result.matrix.values,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        ctilde_result.weighted_residual.values,
        2.0 * ctilde_result.residual.values,
        atol=1e-10,
    )
    assert ctilde_result.matrix.clp_label.values.tolist() == ["a"]
    assert ctilde_result.Etilde.dims == ("spectral", "basis")
    np.testing.assert_allclose(ctilde_result.Etilde.values, etilde)

    expected_refresid = source_data["data"].values - (
        etilde @ ctilde_result.data.values.T
    ).T
    assert ctilde_result.refresid.dims == ("time", "spectral")
    np.testing.assert_allclose(ctilde_result.refresid.values, expected_refresid, atol=1e-10)

    left, singular_values, right = np.linalg.svd(
        ctilde_result.data.values, full_matrices=False
    )
    np.testing.assert_allclose(ctilde_result.data_left_singular_vectors.values, left)
    np.testing.assert_allclose(ctilde_result.data_singular_values.values, singular_values)
    np.testing.assert_allclose(ctilde_result.data_right_singular_vectors.values, right.T)

    left, singular_values, right = np.linalg.svd(expected_refresid, full_matrices=False)
    assert ctilde_result.refresid_left_singular_vectors.dims == (
        "time",
        "refresid_left_singular_value_index",
    )
    assert ctilde_result.refresid_singular_values.dims == ("refresid_singular_value_index",)
    assert ctilde_result.refresid_right_singular_vectors.dims == (
        "spectral",
        "refresid_right_singular_value_index",
    )
    np.testing.assert_allclose(ctilde_result.refresid_left_singular_vectors.values, left)
    np.testing.assert_allclose(ctilde_result.refresid_singular_values.values, singular_values)
    np.testing.assert_allclose(ctilde_result.refresid_right_singular_vectors.values, right.T)


def test_ctilde_observations_calculate_refresid_from_cnnls_not_kinetic_concentrations():
    megacomplex = CtildeMegacomplex(
        label="ctilde",
        source_dataset="source",
        regularization_weight=1.0,
    )
    time = np.array([0.0, 1.0])
    spectral = np.array([500.0, 600.0])
    basis = np.array(["s1"])
    source_dataset = xr.Dataset(
        {"data": (("time", "spectral"), np.array([[3.0, 0.0], [0.0, 5.0]]))},
        coords={"time": time, "spectral": spectral},
    )
    ctilde = xr.DataArray(
        np.array([[5.0], [5.0]]),
        dims=("time", "basis"),
        coords={"time": time, "basis": basis},
    )
    etilde = xr.DataArray(
        np.array([[1.0], [2.0]]),
        dims=("spectral", "basis"),
        coords={"spectral": spectral, "basis": basis},
    )

    observations = megacomplex.calculate_observations(source_dataset, ctilde, etilde)
    expected_refresid = source_dataset["data"].values - (
        etilde.values @ observations["data"].values.T
    ).T
    residual_from_kinetic = source_dataset["data"].values - (
        etilde.values @ ctilde.values.T
    ).T

    np.testing.assert_allclose(observations["refresid"].values, expected_refresid)
    assert not np.allclose(observations["refresid"].values, residual_from_kinetic)
