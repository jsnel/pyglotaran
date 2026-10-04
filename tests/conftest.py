"""Configuration of the test suite."""

from __future__ import annotations

import os

# Fits in the test suite must not write records into the working directory.
# Tests of the recording remove this variable with ``monkeypatch``.
os.environ["GLOTARAN_RECORD"] = "0"
