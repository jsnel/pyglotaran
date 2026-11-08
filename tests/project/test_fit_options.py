from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

import pytest

from glotaran.optimization import Optimization
from glotaran.project import Project
from glotaran.testing.simulated_data.shared_decay import PARAMETERS
from tests.project.conftest import LABEL

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    import xarray as xr

    from glotaran.project import Scheme


@pytest.fixture(params=["scheme", "project", "optimization"])
def fit_with_options(
    request: pytest.FixtureRequest, scheme: Scheme, data: xr.Dataset, tmp_path: Path
) -> Callable[..., None]:
    def fit(**options: Any) -> None:
        options = {"maximum_number_function_evaluations": 1, "verbose": False} | options
        if request.param == "scheme":
            scheme.optimize(PARAMETERS, {LABEL: data}, **options)
        elif request.param == "project":
            Project.start(tmp_path).optimize(scheme, PARAMETERS, {LABEL: data}, **options)
        else:
            models = list(scheme.experiments.values())
            for model in models:
                model.datasets[LABEL].data = data
            Optimization(models=models, library=scheme.library, parameters=PARAMETERS, **options)

    return fit


@pytest.mark.parametrize(
    ("scale", "match"),
    [
        (0.0, "finite positive values"),
        ([1.0], "match the free parameter count"),
        ({"unknown": 1.0}, "name every free parameter"),
    ],
)
def test_invalid_fit_x_scale(
    fit_with_options: Callable[..., None],
    scale: float | list[float] | dict[str, float],
    match: str,
):
    with pytest.raises(ValueError, match=match):
        fit_with_options(x_scale=scale)
