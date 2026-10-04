"""Fixtures for the tests of the project package."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from glotaran.project import record as record_module
from glotaran.project.record import RecordSettings

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
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
