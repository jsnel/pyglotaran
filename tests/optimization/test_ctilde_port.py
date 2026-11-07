"""Behavioral coverage of generated Ctilde observations and direct residuals."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import xarray as xr

from glotaran.builtin.elements.ctilde import CtildeElement
from glotaran.io import load_result
from glotaran.model.errors import GlotaranModelError
from glotaran.model.errors import GlotaranUserError
from glotaran.model.experiment_model import ExperimentModel
from glotaran.optimization.optimization import Optimization
from glotaran.parameter import Parameters
from glotaran.project import Project
from glotaran.project.record import read_record_file
from glotaran.project.scheme import Scheme

if TYPE_CHECKING:
    from pathlib import Path


def test_observations_refresid_uses_estimated_concentrations():
    element = CtildeElement(type="ctilde", label="ctilde", source_dataset="source")
    time = np.array([0.0, 1.0])
    spectral = np.array([500.0, 600.0])
    source = xr.Dataset(
        {"data": (("time", "spectral"), [[3.0, 0.0], [0.0, 5.0]])},
        coords={"time": time, "spectral": spectral},
    )
    kinetic = xr.DataArray(
        [[5.0], [5.0]], dims=("time", "basis"), coords={"time": time, "basis": ["s1"]}
    )
    spectra = xr.DataArray(
        [[1.0], [2.0]], dims=("spectral", "basis"), coords={"spectral": spectral, "basis": ["s1"]}
    )
    observed = element.calculate_observations(source, kinetic, spectra)
    expected = source["data"] - xr.dot(observed["data"], spectra, dim="basis")
    xr.testing.assert_allclose(observed.refresid, expected)
    assert not np.allclose(
        observed.refresid, source["data"] - xr.dot(kinetic, spectra, dim="basis")
    )


def make_scheme(relation_interval: list[list[float]] | None = None):
    relation = {"source": "a", "target": "b", "parameter": "relation.factor"}
    if relation_interval is not None:
        relation["interval"] = relation_interval
    return Scheme.from_dict(
        {
            "library": {
                "decay": {"type": "kinetic", "rates": {"(a, a)": "rates.a", "(b, b)": "rates.b"}},
                "ctilde_model": {
                    "type": "ctilde",
                    "source_dataset": "source",
                    "regularization_weight": 1.0,
                },
            },
            "experiments": {
                "one": {
                    "residual_function": "non_negative_least_squares",
                    "clp_relations": [relation],
                    "datasets": {
                        "source": {
                            "elements": ["decay"],
                            "activations": {
                                "irf": {"type": "instant", "compartments": {"a": 1.0, "b": 1.0}}
                            },
                        },
                        "auxiliary": {"elements": ["ctilde_model"], "weights": [{"value": 2.0}]},
                    },
                }
            },
        }
    )


def input_data():
    time = np.arange(5.0)
    spectral = np.array([500.0, 600.0])
    kinetic = 0.5 * (np.exp(-0.4 * time) + np.exp(-1.5 * time))
    return xr.Dataset(
        {"data": (("time", "spectral"), kinetic[:, None] * [[1.0, 2.0]])},
        coords={"time": time, "spectral": spectral},
    )


def params():
    return Parameters.from_list(
        [
            ["rates.a", 0.4, {"vary": False}],
            ["rates.b", 1.5, {"vary": False}],
            ["relation.factor", 1.0, {"vary": False}],
        ]
    )


def test_generated_direct_residual_uses_source_relations_and_weights():
    result = make_scheme().optimize(
        params(), {"source": input_data()}, dry_run=True, verbose=False, raise_exception=True
    )
    source = input_data()
    auxiliary = result.optimization_results["auxiliary"]
    assert set(result.optimization_results) == {"source", "auxiliary"}
    details = auxiliary.elements["ctilde_model"]
    source_details = result.optimization_results["source"].elements["ctilde_model"]
    xr.testing.assert_equal(source_details.Ctilde, details.Ctilde)
    xr.testing.assert_equal(source_details.Etilde, details.Etilde)
    expected = 0.5 * (np.exp(-0.4 * source.time) + np.exp(-1.5 * source.time))
    np.testing.assert_allclose(auxiliary.input_data.values[:, 0], expected, atol=1e-9)
    np.testing.assert_allclose(auxiliary.residuals.values, 0, atol=1e-9)
    np.testing.assert_allclose(details.Ctilde.values[:, 0], expected, atol=1e-9)
    np.testing.assert_allclose(details.Etilde.values[:, 0], [1.0, 2.0], atol=1e-9)
    expected_refresid = source.data - xr.dot(auxiliary.input_data, details.Etilde, dim="basis")
    xr.testing.assert_allclose(details.refresid, expected_refresid)
    np.testing.assert_array_equal(auxiliary.fit_decomposition.clp.values, [[1.0]])


def test_ctilde_source_respects_normalized_element_scale():
    reference = make_scheme().optimize(
        params(), {"source": input_data()}, dry_run=True, verbose=False, raise_exception=True
    )
    scaled_scheme = make_scheme()
    source = scaled_scheme.experiments["one"].datasets["source"]
    source.element_scale = {"decay": 3.0}
    source.normalize_element_scales = True
    scaled = scaled_scheme.optimize(
        params(), {"source": input_data()}, dry_run=True, verbose=False, raise_exception=True
    )
    expected = reference.optimization_results["auxiliary"]
    actual = scaled.optimization_results["auxiliary"]
    xr.testing.assert_allclose(actual.input_data, expected.input_data)
    xr.testing.assert_allclose(actual.residuals, expected.residuals)
    for name in ("Ctilde", "Etilde", "refresid"):
        xr.testing.assert_allclose(
            actual.elements["ctilde_model"][name], expected.elements["ctilde_model"][name]
        )


def test_auxiliary_does_not_add_free_clps():
    scheme = make_scheme()
    scheme._load_data({"source": input_data()})
    optimization = Optimization(
        models=list(scheme.experiments.values()),
        parameters=params(),
        library=scheme.library,
        verbose=False,
        raise_exception=True,
    )
    objective = optimization._objectives[0]
    assert objective.get_result().clp_size == objective._inner.get_result().clp_size == 2


def test_weighted_direct_residual_updates_with_parameters():
    scheme = make_scheme()
    noisy = input_data()
    noisy["data"][2, 0] += 0.25
    scheme._load_data({"source": noisy})
    optimization = Optimization(
        models=list(scheme.experiments.values()),
        parameters=params(),
        library=scheme.library,
        verbose=False,
        raise_exception=True,
    )
    objective = optimization._objectives[0]
    before = objective.calculate()
    first = objective.get_result().optimization_results["auxiliary"]
    np.testing.assert_allclose(
        before[-first.residuals.size :], 2 * first.residuals.values.T.ravel(), atol=1e-12
    )
    assert np.linalg.norm(first.residuals) > 0
    optimization._parameters.get("rates.a").value = 0.8
    after = objective.calculate()
    second = objective.get_result().optimization_results["auxiliary"]
    assert after.size == before.size
    assert not np.allclose(
        first.elements["ctilde_model"].Ctilde, second.elements["ctilde_model"].Ctilde
    )
    assert not np.allclose(first.input_data, second.input_data)


def test_source_missing_is_rejected():
    with pytest.raises(Exception, match="source"):
        make_scheme().optimize(params(), {}, dry_run=True, verbose=False, raise_exception=True)


def supplied_ctilde(basis: tuple[str, str] = ("b", "a")):
    return xr.Dataset(
        {
            "data": (("time", "basis"), np.full((5, 2), 42.0)),
            "weight": (("time", "basis"), np.tile([3.0, 5.0], (5, 1))),
        },
        coords={"time": np.arange(5.0), "basis": list(basis)},
    )


def supplied_ctilde_scheme():
    scheme = make_scheme()
    scheme.experiments["one"].clp_relations = []
    return scheme


def test_supplied_ctilde_basis_and_weights_are_used_but_values_are_generated():
    scheme = supplied_ctilde_scheme()
    source = input_data()
    source["data"][2, 0] += 0.25
    supplied = supplied_ctilde()
    scheme._load_data({"source": source, "auxiliary": supplied})
    optimization = Optimization(
        models=list(scheme.experiments.values()),
        parameters=params(),
        library=scheme.library,
        verbose=False,
        raise_exception=True,
    )
    objective = optimization._objectives[0]
    residual = objective.calculate()
    auxiliary = objective.get_result().optimization_results["auxiliary"]
    assert auxiliary.input_data.basis.values.tolist() == ["b", "a"]
    assert not np.allclose(auxiliary.input_data, supplied.data)
    assert np.linalg.norm(auxiliary.residuals) > 0
    np.testing.assert_allclose(
        residual[-auxiliary.residuals.size :],
        (auxiliary.residuals * supplied.weight).values.T.ravel(),
        atol=1e-12,
    )
    natural = (
        supplied_ctilde_scheme()
        .optimize(params(), {"source": source}, dry_run=True, verbose=False, raise_exception=True)
        .optimization_results["auxiliary"]
    )
    xr.testing.assert_allclose(auxiliary.input_data.sel(basis=["a", "b"]), natural.input_data)


def test_supplied_ctilde_project_record_and_recompute(tmp_path: Path):
    project = Project.start(tmp_path)
    datasets = {"source": input_data(), "auxiliary": supplied_ctilde()}
    result = project.optimize(
        supplied_ctilde_scheme(), params(), datasets, verbose=False, raise_exception=True
    )
    record = read_record_file(result.record.path)
    assert set(record["data"]) == {"source", "auxiliary"}
    assert "weight" in record["data"]["auxiliary"]
    recomputed = project.recompute(result.record, datasets)
    expected = result.optimization_results["auxiliary"]
    actual = recomputed.optimization_results["auxiliary"]
    xr.testing.assert_allclose(expected.input_data, actual.input_data)
    xr.testing.assert_allclose(expected.residuals, actual.residuals)
    altered = dict(datasets)
    altered["auxiliary"] = datasets["auxiliary"].assign_coords(basis=["a", "b"])
    with pytest.raises(GlotaranUserError, match="basis coordinate labels changed"):
        project.recompute(result.record, altered)


@pytest.mark.parametrize("axis", ["time", "basis", "duplicate"])
def test_supplied_ctilde_mismatched_axes_are_rejected(axis: str):
    supplied = supplied_ctilde()
    if axis == "time":
        supplied = supplied.assign_coords(time=np.arange(5.0)[::-1])
    elif axis == "basis":
        supplied = supplied.assign_coords(basis=["a", "unknown"])
    else:
        supplied = supplied.assign_coords(basis=["a", "a"])
    with pytest.raises(GlotaranModelError, match=r"Ctilde dataset 'auxiliary'.*(time|basis)"):
        supplied_ctilde_scheme().optimize(
            params(),
            {"source": input_data(), "auxiliary": supplied},
            dry_run=True,
            verbose=False,
            raise_exception=True,
        )


def test_source_in_other_experiment_is_rejected():
    scheme = make_scheme()
    experiment = scheme.experiments["one"]
    scheme.experiments["other"] = ExperimentModel(
        datasets={"other-source": experiment.datasets["source"]}
    )
    scheme.library["ctilde_model"].source_dataset = "other-source"
    with pytest.raises(Exception, match="same experiment"):
        scheme.optimize(
            params(),
            {"source": input_data(), "other-source": input_data()},
            dry_run=True,
            verbose=False,
            raise_exception=True,
        )


def test_ctilde_uncertainty_is_rejected():
    scheme = make_scheme()
    scheme.experiments["one"].residual_function = "variable_projection"
    with pytest.raises(ValueError, match=r"CLP standard errors.*Ctilde"):
        scheme.optimize(
            params(),
            {"source": input_data()},
            compute_clp_standard_error=True,
            dry_run=True,
            verbose=False,
            raise_exception=True,
        )


def test_ctilde_diagnostics_are_saved_and_reloaded(tmp_path: Path):
    result = make_scheme().optimize(
        params(), {"source": input_data()}, dry_run=True, verbose=False, raise_exception=True
    )
    result.save(tmp_path / "result")
    loaded = load_result(tmp_path / "result")
    expected = result.optimization_results["auxiliary"].elements["ctilde_model"]
    actual = loaded.optimization_results["auxiliary"].elements["ctilde_model"]
    for name in ("Ctilde", "Etilde", "refresid"):
        xr.testing.assert_allclose(actual[name], expected[name])


def test_project_records_and_recomputes_without_generated_input(tmp_path: Path):
    project = Project.start(tmp_path)
    source = input_data()
    result = project.optimize(
        make_scheme(), params(), {"source": source}, verbose=False, raise_exception=True
    )
    assert result.record is not None
    record = read_record_file(result.record.path)
    assert set(record["data"]) == {"source"}
    recomputed = project.recompute(result.record, {"source": source})
    expected = result.optimization_results["auxiliary"]
    actual = recomputed.optimization_results["auxiliary"]
    xr.testing.assert_allclose(expected.input_data, actual.input_data)
    xr.testing.assert_allclose(expected.residuals, actual.residuals)
    for name in ("Ctilde", "Etilde", "refresid"):
        xr.testing.assert_allclose(
            expected.elements["ctilde_model"][name], actual.elements["ctilde_model"][name]
        )
    export = project.export(result, "ctilde")
    from_export = project.recompute(export, {"source": source})
    xr.testing.assert_allclose(
        expected.input_data, from_export.optimization_results["auxiliary"].input_data
    )


def test_spectrally_varying_relations_are_rejected():
    with pytest.raises(Exception, match="relations must be constant"):
        make_scheme([[490.0, 550.0]]).optimize(
            params(), {"source": input_data()}, dry_run=True, verbose=False, raise_exception=True
        )
