"""Behavioral coverage of generated Ctilde observations and direct residuals."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from glotaran.builtin.elements.ctilde import CtildeElement
from glotaran.builtin.io.yml.utils import load_dict
from glotaran.builtin.io.yml.utils import write_dict
from glotaran.io import load_result
from glotaran.io.interface import SAVING_OPTIONS_MINIMAL
from glotaran.model.clp_relation import ClpRelation
from glotaran.model.errors import GlotaranModelError
from glotaran.model.errors import GlotaranUserError
from glotaran.model.experiment_model import ExperimentModel
from glotaran.optimization.optimization import Optimization
from glotaran.parameter import Parameters
from glotaran.project import Project
from glotaran.project.record import read_record_file
from glotaran.project.scheme import Scheme

REFRESID_SVD = {
    "refresid_left_singular_vectors",
    "refresid_singular_values",
    "refresid_right_singular_vectors",
}


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


def test_refresid_svd_reconstructs_spectral_residual():
    source = input_data()
    source["data"][2, 0] += 0.25
    result = make_scheme().optimize(
        params(), {"source": source}, dry_run=True, verbose=False, raise_exception=True
    )
    details = result.optimization_results["auxiliary"].elements["ctilde_model"]
    left = details.refresid_left_singular_vectors
    values = details.refresid_singular_values
    right = details.refresid_right_singular_vectors
    assert left.dims == ("time", "left_singular_value_index")
    assert values.dims == ("singular_value_index",)
    assert right.dims == ("spectral", "right_singular_value_index")
    np.testing.assert_allclose(
        (left.values * values.values) @ right.values.T, details.refresid.values, atol=1e-12
    )


def test_disabling_svd_skips_calculation_without_changing_fit(monkeypatch: pytest.MonkeyPatch):
    source = input_data()
    source["data"][2, 0] += 0.25
    reference = make_scheme().optimize(
        params(), {"source": source}, dry_run=True, verbose=False, raise_exception=True
    )

    def fail_svd(*args: object, **kwargs: object) -> None:
        raise AssertionError("SVD calculation must not run when add_svd is false")

    monkeypatch.setattr("glotaran.optimization.ctilde.add_svd_to_dataset", fail_svd)
    result = make_scheme().optimize(
        params(),
        {"source": source},
        add_svd=False,
        dry_run=True,
        verbose=False,
        raise_exception=True,
    )
    assert result.optimizer_settings.add_svd is False
    for label in ("source", "auxiliary"):
        actual = result.optimization_results[label]
        expected = reference.optimization_results[label]
        xr.testing.assert_allclose(actual.input_data, expected.input_data)
        xr.testing.assert_allclose(actual.fitted_data, expected.fitted_data)
        xr.testing.assert_allclose(actual.residuals, expected.residuals)
    actual = result.optimization_results["auxiliary"].elements["ctilde_model"]
    expected = reference.optimization_results["auxiliary"].elements["ctilde_model"]
    assert REFRESID_SVD.isdisjoint(actual.data_vars)
    assert REFRESID_SVD.issubset(expected.data_vars)
    for name in ("Ctilde", "Etilde", "refresid"):
        xr.testing.assert_allclose(actual[name], expected[name])
    assert result.optimization_info.cost == pytest.approx(reference.optimization_info.cost)


@pytest.mark.parametrize(
    ("scheme_value", "override", "expected"),
    [(True, None, True), (False, None, False), (True, False, False), (False, True, True)],
)
def test_svd_scheme_default_and_per_fit_override(
    scheme_value: bool, override: bool | None, expected: bool
):
    scheme = make_scheme()
    scheme.add_svd = scheme_value
    result = scheme.optimize(
        params(),
        {"source": input_data()},
        add_svd=override,
        dry_run=True,
        verbose=False,
        raise_exception=True,
    )
    details = result.optimization_results["auxiliary"].elements["ctilde_model"]
    assert (
        REFRESID_SVD.issubset(details.data_vars)
        if expected
        else REFRESID_SVD.isdisjoint(details.data_vars)
    )
    assert result.optimizer_settings.add_svd is expected
    assert scheme.add_svd is scheme_value


@pytest.mark.parametrize("via_project", [False, True])
@pytest.mark.parametrize("invalid", ["false", 0])
def test_svd_override_requires_bool(tmp_path: Path, via_project: bool, invalid: str | int):
    scheme = make_scheme()
    data = {"source": input_data()}
    if via_project:
        with pytest.raises(TypeError, match="add_svd must be a boolean"):
            Project.start(tmp_path).optimize(
                scheme, params(), data, add_svd=invalid, verbose=False
            )
    else:
        with pytest.raises(TypeError, match="add_svd must be a boolean"):
            scheme.optimize(params(), data, add_svd=invalid, verbose=False)


def test_disabled_svd_result_save_and_reload(tmp_path: Path):
    result = make_scheme().optimize(
        params(),
        {"source": input_data()},
        add_svd=False,
        dry_run=True,
        verbose=False,
        raise_exception=True,
    )
    result.save(tmp_path / "result")
    restored = load_result(tmp_path / "result")
    assert restored.optimizer_settings.add_svd is False
    actual = restored.optimization_results["auxiliary"].elements["ctilde_model"]
    expected = result.optimization_results["auxiliary"].elements["ctilde_model"]
    assert REFRESID_SVD.isdisjoint(actual.data_vars)
    for name in ("Ctilde", "Etilde", "refresid"):
        xr.testing.assert_allclose(actual[name], expected[name])


def test_old_result_without_svd_setting_loads_with_original_default(tmp_path: Path):
    result = make_scheme().optimize(
        params(), {"source": input_data()}, dry_run=True, verbose=False, raise_exception=True
    )
    result.save(tmp_path / "result")
    result_file = tmp_path / "result" / "result.yml"
    serialized = load_dict(result_file, is_file=True)
    serialized["optimizer_settings"].pop("add_svd", None)
    write_dict(serialized, file_name=result_file)
    restored = load_result(tmp_path / "result")
    assert restored.optimizer_settings.add_svd is True
    assert REFRESID_SVD.issubset(
        restored.optimization_results["auxiliary"].elements["ctilde_model"].data_vars
    )


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
    xr.testing.assert_identical(auxiliary.supplied_input_data, supplied)
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


@pytest.mark.parametrize(
    ("scalar", "scale_list"),
    [(1.0, None), (2.0, None), (1.0, [3.0, 5.0]), (2.0, [3.0, 5.0])],
)
def test_scaled_ctilde_decomposition_reconstructs_fitted_data(
    scalar: float, scale_list: list[float] | None
):
    scheme = supplied_ctilde_scheme()
    experiment = scheme.experiments["one"]
    experiment.scale["auxiliary"] = scalar
    if scale_list is not None:
        experiment.scale_list["auxiliary"] = scale_list
    result = scheme.optimize(
        params(),
        {"source": input_data(), "auxiliary": supplied_ctilde()},
        add_svd=False,
        dry_run=True,
        verbose=False,
        raise_exception=True,
    )
    auxiliary = result.optimization_results["auxiliary"]
    decomposition = auxiliary.fit_decomposition
    assert decomposition is not None
    assert auxiliary.fitted_data is not None
    reconstructed = xr.dot(decomposition.matrix, decomposition.clp, dim="amplitude_label")
    xr.testing.assert_allclose(reconstructed.transpose("time", "basis"), auxiliary.fitted_data)
    expected = auxiliary.elements["ctilde_model"].Ctilde * scalar
    if scale_list is not None:
        expected = expected * xr.DataArray(
            scale_list, dims=("basis",), coords={"basis": expected.basis}
        )
    xr.testing.assert_allclose(decomposition.matrix.rename({"amplitude_label": "basis"}), expected)


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
    exported = project.export(result, "supplied-ctilde")
    exported_recomputed = project.recompute(exported, datasets)
    xr.testing.assert_allclose(
        exported_recomputed.optimization_results["auxiliary"].fitted_data, expected.fitted_data
    )
    with pytest.raises(GlotaranUserError, match="basis coordinate labels changed"):
        project.recompute(exported, altered)
    changed_weights = dict(datasets)
    changed_weights["auxiliary"] = datasets["auxiliary"].copy(deep=True)
    changed_weights["auxiliary"]["weight"] *= 2
    with pytest.raises(GlotaranUserError, match="weight"):
        project.recompute(exported, changed_weights)

    restored = load_result(exported)
    xr.testing.assert_allclose(
        restored.optimization_results["auxiliary"].supplied_input_data, datasets["auxiliary"]
    )
    assert restored.optimization_results["auxiliary"].input_data_is_generated
    second_export = project.export(restored, "restored-ctilde")
    project.recompute(second_export, datasets)

    saved = tmp_path / "saved-result"
    result.save(saved)
    loaded = load_result(saved)
    xr.testing.assert_allclose(
        loaded.optimization_results["auxiliary"].supplied_input_data, datasets["auxiliary"]
    )
    serialized = load_dict(saved / "result.yml", is_file=True)
    paths = type(result).extract_paths_from_serialization(saved / "result.yml", serialized)
    supplied_path = saved / "optimization_results" / "auxiliary" / "supplied_input_data.nc"
    assert supplied_path.resolve() in {Path(path).resolve() for path in paths}


def test_filtered_supplied_ctilde_cannot_be_silently_exported(tmp_path: Path):
    project = Project.start(tmp_path)
    datasets = {"source": input_data(), "auxiliary": supplied_ctilde()}
    result = project.optimize(
        supplied_ctilde_scheme(), params(), datasets, verbose=False, raise_exception=True
    )
    original_attrs = {
        label: dict(optimization_result.input_data.attrs)
        for label, optimization_result in result.optimization_results.items()
    }
    supplied_attrs = dict(result.optimization_results["auxiliary"].supplied_input_data.attrs)
    minimal_export = project.export(
        result, "minimal-export", saving_options=SAVING_OPTIONS_MINIMAL
    )
    for label, optimization_result in result.optimization_results.items():
        assert optimization_result.input_data.attrs == original_attrs[label]
    assert result.optimization_results["auxiliary"].supplied_input_data.attrs == supplied_attrs
    project.recompute(minimal_export, datasets)
    minimal = tmp_path / "minimal-result"
    result.save(minimal, saving_options=SAVING_OPTIONS_MINIMAL)
    restored = load_result(minimal)
    auxiliary = restored.optimization_results["auxiliary"]
    assert auxiliary.input_data_is_generated
    assert auxiliary.supplied_input_data_was_provided
    assert auxiliary.supplied_input_data is None
    with pytest.raises(GlotaranUserError, match=r"supplied input data.*auxiliary"):
        project.export(restored, "incomplete")

    complete = tmp_path / "complete-result"
    result.save(complete)
    complete_result = load_result(complete)
    export = project.export(complete_result, "complete")
    project.recompute(export, datasets)
    changed = dict(datasets)
    changed["auxiliary"] = datasets["auxiliary"].copy(deep=True)
    changed["auxiliary"]["weight"] *= 2
    with pytest.raises(GlotaranUserError, match="weight"):
        project.recompute(export, changed)


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
    for name in (
        "Ctilde",
        "Etilde",
        "refresid",
        "refresid_singular_values",
        "refresid_left_singular_vectors",
        "refresid_right_singular_vectors",
    ):
        xr.testing.assert_allclose(actual[name], expected[name])


@pytest.mark.parametrize(("scheme_value", "override"), [(True, False), (False, True)])
def test_project_records_ctilde_svd_override_and_recomputes(
    tmp_path: Path, scheme_value: bool, override: bool
):
    project = Project.start(tmp_path)
    source = input_data()
    scheme = make_scheme()
    scheme.add_svd = scheme_value
    result = project.optimize(
        scheme,
        params(),
        {"source": source},
        add_svd=override,
        verbose=False,
        raise_exception=True,
    )
    assert result.record is not None
    assert result.optimizer_settings.add_svd is override
    record = read_record_file(result.record.path)
    assert record["optimizer"]["add_svd"] is override
    expected = result.optimization_results["auxiliary"].elements["ctilde_model"]
    assert (
        REFRESID_SVD.issubset(expected.data_vars)
        if override
        else REFRESID_SVD.isdisjoint(expected.data_vars)
    )

    recomputed = project.recompute(result.record, {"source": source})
    assert recomputed.optimizer_settings.add_svd is override
    actual = recomputed.optimization_results["auxiliary"].elements["ctilde_model"]
    assert (
        REFRESID_SVD.issubset(actual.data_vars)
        if override
        else REFRESID_SVD.isdisjoint(actual.data_vars)
    )
    xr.testing.assert_allclose(actual.refresid, expected.refresid)

    exported = project.export(result, "ctilde-svd")
    restored = load_result(exported)
    assert restored.optimizer_settings.add_svd is override
    exported_recomputation = project.recompute(exported, {"source": source})
    assert exported_recomputation.optimizer_settings.add_svd is override
    exported_details = exported_recomputation.optimization_results["auxiliary"].elements[
        "ctilde_model"
    ]
    assert (
        REFRESID_SVD.issubset(exported_details.data_vars)
        if override
        else REFRESID_SVD.isdisjoint(exported_details.data_vars)
    )


def test_project_records_and_recomputes_without_generated_input(tmp_path: Path):
    project = Project.start(tmp_path)
    source = input_data()
    # Metadata alone must not make a supplied input disappear from the export summary.
    source.attrs["generated_by_model"] = 1
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
    metadata = load_dict(export / "export.yml", is_file=True)
    assert set(metadata["data"]) == {"source"}
    from_export = project.recompute(export, {"source": source})
    xr.testing.assert_allclose(
        expected.input_data, from_export.optimization_results["auxiliary"].input_data
    )
    changed_source = source.copy(deep=True)
    changed_source["data"] *= 2
    with pytest.raises(GlotaranUserError, match="source: data"):
        project.recompute(export, {"source": changed_source})


def test_spectrally_varying_relations_are_rejected():
    with pytest.raises(Exception, match="relations must be constant"):
        make_scheme([[490.0, 550.0]]).optimize(
            params(), {"source": input_data()}, dry_run=True, verbose=False, raise_exception=True
        )


def test_ctilde_rechecks_initially_equal_relation_coefficients_after_parameter_update():
    scheme = make_scheme()
    experiment = scheme.experiments["one"]
    experiment.clp_relations = [
        ClpRelation(source="a", target="b", parameter="relation.low", interval=(490.0, 550.0)),
        ClpRelation(source="a", target="b", parameter="relation.high", interval=(551.0, 610.0)),
    ]
    initial = Parameters.from_list(
        [
            ["rates.a", 0.4, {"vary": False}],
            ["rates.b", 1.5, {"vary": False}],
            ["relation.low", 1.0],
            ["relation.high", 1.0],
        ]
    )
    source = input_data()
    scheme._load_data({"source": source})
    optimization = Optimization(
        models=[experiment],
        parameters=initial,
        library=scheme.library,
        verbose=False,
        raise_exception=True,
    )

    _, results, _ = optimization.dry_run()
    details = results["auxiliary"].elements["ctilde_model"]
    assert details.basis.values.tolist() == ["a"]
    expected = 0.5 * (np.exp(-0.4 * source.time) + np.exp(-1.5 * source.time))
    np.testing.assert_allclose(details.Ctilde.sel(basis="a"), expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(details.Etilde.sel(basis="a"), [1.0, 2.0], rtol=1e-12, atol=1e-12)

    # The labels are unchanged, but the kinetic basis now differs between wavelengths.
    optimization.parameters.get("relation.high").value = 2.0
    with pytest.raises(GlotaranModelError, match="relations must be constant"):
        optimization.calculate_penalty()
