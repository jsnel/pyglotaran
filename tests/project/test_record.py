from __future__ import annotations

import sys
import threading
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import Any

import pandas as pd
import pytest

from glotaran.io import load_parameters
from glotaran.optimization.objective import OptimizationObjective
from glotaran.project import record as record_module
from glotaran.project.record import RecordSettings
from glotaran.project.record import allocate_record_folder
from glotaran.project.record import detect_source
from glotaran.project.record import environment_versions
from glotaran.project.record import find_project_root
from glotaran.project.record import read_record
from glotaran.project.record import recording_enabled
from glotaran.project.record import resolve_results_folder
from glotaran.project.scheme import Scheme
from glotaran.testing.simulated_data.sequential_spectral_decay import DATASET
from glotaran.testing.simulated_data.sequential_spectral_decay import SCHEME_DICT
from glotaran.testing.simulated_data.shared_decay import PARAMETERS

if TYPE_CHECKING:
    from pathlib import Path

    from glotaran.project.result import Result


@pytest.fixture(autouse=True)
def recording(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Record in a temporary working directory, with default settings.

    Project roots outside the temporary folder (for example a ``project.gta`` in the home
    folder) are ignored.
    """
    monkeypatch.chdir(tmp_path)
    for name in [
        "GLOTARAN_RECORD",
        "GLOTARAN_RESULTS_FOLDER",
        "GLOTARAN_RECORD_PARAMETER_HISTORY",
    ]:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(record_module, "record_settings", RecordSettings())
    find_root = record_module.find_project_root

    def find_root_in_tmp_path(start: Path | None = None) -> Path | None:
        root = find_root(start)
        return root if root is not None and root.is_relative_to(tmp_path) else None

    monkeypatch.setattr(record_module, "find_project_root", find_root_in_tmp_path)
    return tmp_path


def fit(**kwargs: Any) -> Result:
    """Fit the sequential decay with one function evaluation."""
    return Scheme.from_dict(SCHEME_DICT).optimize(
        PARAMETERS,
        {"sequential-decay": DATASET},
        **{"verbose": False, "maximum_number_function_evaluations": 1} | kwargs,
    )


def test_find_project_root(tmp_path: Path):
    """The nearest folder with a ``project.gta``, including a v0.7 one, is the root."""
    (tmp_path / "project.gta").write_text("version: 0.7.4\n")
    subfolder = tmp_path / "analysis" / "notebooks"
    subfolder.mkdir(parents=True)

    assert find_project_root(subfolder) == tmp_path
    assert find_project_root(tmp_path) == tmp_path

    (tmp_path / "analysis" / "project.gta").write_text("title: Nested project\n")
    assert find_project_root(subfolder) == tmp_path / "analysis"


def test_record_folder_defaults_to_project_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    assert resolve_results_folder() == tmp_path / "results"

    (tmp_path / "project.gta").touch()
    subfolder = tmp_path / "notebooks"
    subfolder.mkdir()
    monkeypatch.chdir(subfolder)
    assert resolve_results_folder() == tmp_path / "results"


def test_record_folder_precedence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Per call before session before environment variable before default."""
    monkeypatch.setenv("GLOTARAN_RESULTS_FOLDER", "environment")
    assert resolve_results_folder() == tmp_path / "environment"
    record_module.record_settings.output_folder = "session"
    assert resolve_results_folder() == tmp_path / "session"
    assert resolve_results_folder("call") == tmp_path / "call"


def test_recording_switches(monkeypatch: pytest.MonkeyPatch):
    """Per call before session before environment variable, except that GLOTARAN_RECORD=0 wins."""
    assert recording_enabled(record=None, output_folder=None) is True
    monkeypatch.setenv("GLOTARAN_RECORD", "1")
    record_module.record_settings.record = False
    assert recording_enabled(record=None, output_folder=None) is False
    assert recording_enabled(record=True, output_folder=None) is True
    assert recording_enabled(record=None, output_folder="folder") is True

    monkeypatch.setenv("GLOTARAN_RECORD", "0")
    record_module.record_settings.record = True
    assert recording_enabled(record=True, output_folder="folder") is False

    with pytest.raises(ValueError, match="record=False"):
        recording_enabled(record=False, output_folder="folder")


def test_fit_is_recorded(recording: Path):
    result = fit(name="first try")

    assert result.record is not None
    assert result.record.path == recording / "results" / "001"
    assert result.record.id == "001"
    assert sorted(path.name for path in result.record.path.iterdir()) == [
        "cost_history.csv",
        "initial_parameters.csv",
        "optimized_parameters.csv",
        "record.yml",
        "scheme.yml",
    ]
    record = read_record(result.record.path)
    assert record["uid"] == result.record.uid
    assert record["status"] == "success"
    assert record["error"] is None
    assert record["name"] == "first try"
    assert record["scheme_source"] is None
    assert record["optimizer"]["maximum_number_function_evaluations"] == 1
    assert record["summary"]["cost"] == result.optimization_info.cost
    assert record["summary"]["converged"] is False
    assert record["summary"]["datasets"]["sequential-decay"]["root_mean_square_error"] == (
        result.optimization_results["sequential-decay"].meta.root_mean_square_error
    )
    assert record["data"]["sequential-decay"]["shape"] == {"time": 2100, "spectral": 72}
    assert record["environment"]["pyglotaran"] == environment_versions()["pyglotaran"]
    assert load_parameters(result.record.path / "optimized_parameters.csv") == (
        result.optimized_parameters
    )
    cost_history = pd.read_csv(result.record.path / "cost_history.csv", index_col="evaluation")
    assert cost_history.index[0] == 1
    assert len(cost_history) > result.optimization_info.number_of_function_evaluations

    assert fit().record.id == "002"


def test_record_parameter_history(recording: Path):
    record_module.record_settings.record_parameter_history = True
    result = fit()

    history = pd.read_csv(result.record.path / "parameter_history.csv")
    cost_history = pd.read_csv(result.record.path / "cost_history.csv")
    assert history["evaluation"].tolist() == [0, *cost_history["evaluation"]]
    assert result.optimization_info.parameter_history.number_of_records == len(history)


def test_record_numbering(recording: Path):
    results_folder = recording / "results"
    (results_folder / "not_a_record").mkdir(parents=True)
    (results_folder / "999").mkdir()

    assert allocate_record_folder(results_folder).name == "1000"
    assert allocate_record_folder(results_folder).name == "1001"
    (results_folder / "1001").rmdir()
    assert allocate_record_folder(results_folder).name == "1001"


def test_record_folders_are_unique_for_concurrent_fits(recording: Path):
    folders: list[Path] = []

    def allocate():
        folders.extend(allocate_record_folder(recording / "results") for _ in range(5))

    threads = [threading.Thread(target=allocate) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(set(folders)) == 40


def test_no_record(recording: Path, monkeypatch: pytest.MonkeyPatch):
    """Switched off per call, per environment and for dry runs."""
    assert fit(record=False).record is None
    assert fit(dry_run=True).record is None
    monkeypatch.setenv("GLOTARAN_RECORD", "0")
    assert fit(record=True).record is None
    assert not (recording / "results").exists()


def test_record_in_output_folder(recording: Path):
    record_module.record_settings.record = False
    result = fit(output_folder="elsewhere")
    assert result.record.path == recording / "elsewhere" / "001"


def test_unwritable_record_folder_warns(recording: Path):
    (recording / "results").write_text("A file where the record folder should be")

    with pytest.warns(UserWarning, match="The fit is not recorded"):
        result = fit()

    assert result.record is None
    assert result.optimization_info.success


def test_record_is_running_during_the_fit(recording: Path, monkeypatch: pytest.MonkeyPatch):
    statuses = []
    calculate = OptimizationObjective.calculate

    def calculate_and_read_status(self: OptimizationObjective):
        statuses.append(read_record(recording / "results" / "001")["status"])
        return calculate(self)

    monkeypatch.setattr(OptimizationObjective, "calculate", calculate_and_read_status)
    fit()

    assert set(statuses) == {"running"}
    assert read_record(recording / "results" / "001")["status"] == "success"


def failing_calculate(error: BaseException, failing_calls: range):
    """Return an objective calculation that raises ``error`` in the calls ``failing_calls``."""
    calculate = OptimizationObjective.calculate
    calls = []

    def calculate_or_raise(self: OptimizationObjective):
        calls.append(1)
        if len(calls) in failing_calls:
            raise error
        return calculate(self)

    return calculate_or_raise


@pytest.mark.parametrize(
    ("error", "status"),
    [(ValueError("bad model"), "failed"), (KeyboardInterrupt(), "interrupted")],
)
def test_record_failed_fit(
    recording: Path, monkeypatch: pytest.MonkeyPatch, error: BaseException, status: str
):
    monkeypatch.setattr(
        OptimizationObjective, "calculate", failing_calculate(error, range(4, 100))
    )

    with pytest.raises(type(error)):
        fit(raise_exception=True)

    folder = recording / "results" / "001"
    record = read_record(folder)
    assert record["status"] == status
    assert record["error"]["type"] == type(error).__name__
    assert record["summary"] == {"number_of_function_evaluations": 4, "cost": None}
    assert len(pd.read_csv(folder / "cost_history.csv")) == 3
    # The last evaluated parameters, those that raised, without standard errors.
    optimized_parameters = load_parameters(folder / "optimized_parameters.csv")
    assert all(pd.isna(parameter.standard_error) for parameter in optimized_parameters.all())


def test_record_failed_fit_without_exception(recording: Path, monkeypatch: pytest.MonkeyPatch):
    """With ``raise_exception=False`` the fit returns a result and the record says failed."""
    monkeypatch.setattr(
        OptimizationObjective, "calculate", failing_calculate(ValueError("bad step"), range(2, 3))
    )

    with pytest.warns(UserWarning, match="Optimization failed"):
        result = fit()

    record = read_record(result.record.path)
    assert record["status"] == "failed"
    assert record["error"] == {"type": "ValueError", "message": "bad step"}
    assert record["summary"]["cost"] == result.optimization_info.cost


def test_record_write_failure_keeps_the_fit_error(
    recording: Path, monkeypatch: pytest.MonkeyPatch
):
    def failing_to_csv(*args: Any, **kwargs: Any):
        raise OSError("disk full")

    def calculate_and_fail_writing(self: OptimizationObjective):
        # The record was started, finishing it fails.
        monkeypatch.setattr(pd.DataFrame, "to_csv", failing_to_csv)
        raise ValueError("bad model")

    monkeypatch.setattr(OptimizationObjective, "calculate", calculate_and_fail_writing)

    with (
        pytest.warns(UserWarning, match="disk full"),
        pytest.raises(ValueError, match="bad model"),
    ):
        fit(raise_exception=True)

    assert read_record(recording / "results" / "001")["status"] == "running"


def test_verbose_fit_prints_the_record_folder(recording: Path, capsys: pytest.CaptureFixture):
    fit(verbose=True)
    assert str(recording / "results" / "001") in capsys.readouterr().out


def test_detect_source(monkeypatch: pytest.MonkeyPatch):
    shell = SimpleNamespace(user_ns={"__vsc_ipynb_file__": "C:/analysis/fit.ipynb"})
    monkeypatch.setitem(sys.modules, "IPython", SimpleNamespace(get_ipython=lambda: shell))
    assert detect_source() == "C:/analysis/fit.ipynb"

    shell.user_ns = {}
    monkeypatch.setenv("JPY_SESSION_NAME", "/home/user/fit.ipynb")
    assert detect_source() == "/home/user/fit.ipynb"

    monkeypatch.delenv("JPY_SESSION_NAME")
    assert detect_source() == "unknown"

    monkeypatch.setitem(sys.modules, "IPython", SimpleNamespace(get_ipython=lambda: None))
    monkeypatch.setitem(sys.modules, "__main__", SimpleNamespace(__file__="fit.py"))
    assert detect_source() == "fit.py"

    monkeypatch.setitem(sys.modules, "__main__", SimpleNamespace())
    assert detect_source() == "unknown"

    record_module.record_settings.source = "my_analysis.ipynb"
    assert detect_source() == "my_analysis.ipynb"


def test_record_source_is_relative_to_the_record(recording: Path):
    (recording / "analysis.py").touch()
    record_module.record_settings.source = str(recording / "analysis.py")

    assert read_record(fit().record.path)["source"] == "../../analysis.py"
