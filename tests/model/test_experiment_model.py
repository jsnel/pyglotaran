from __future__ import annotations

import pytest

from glotaran.model.data_model import DataModel
from glotaran.model.errors import GlotaranUserError
from glotaran.model.experiment_model import ExperimentModel
from glotaran.parameter import Parameters
from glotaran.project.library import ModelLibrary
from tests.model.test_data_model import MockDataModel
from tests.model.test_data_model import MockElementWithDataModel
from tests.model.test_data_model import MockElementWithItem


def test_experiment_model_from_dict():
    library = {
        "m1": MockElementWithDataModel(label="m1", type="mock-w-datamodel"),
        "m2": MockElementWithItem(label="m2", type="mock-w-item"),
    }

    experiment_model = ExperimentModel.from_dict(
        library,
        {
            "datasets": {
                "d1": {"elements": ["m1"]},
                "d2": {
                    "elements": ["m2"],
                    "weights": [{"value": 1}, {"value": 2, "global_interval": (5, 6)}],
                },
                "d3": {"elements": ["m2"], "global_elements": ["m1"]},
            },
            "clp_penalties": [
                {
                    "type": "equal_area",
                    "source": "s",
                    "source_intervals": (1, 2),
                    "target": "t",
                    "target_intervals": [(1, 2)],
                    "parameter": "p",
                    "weight": 1,
                }
            ],
            "clp_relations": [
                {
                    "source": "s",
                    "target": "t",
                    "interval": [(1, 2)],
                    "parameter": "p",
                },
            ],
        },
    )

    d1 = experiment_model.datasets["d1"]
    assert isinstance(d1, MockDataModel)

    d2 = experiment_model.datasets["d2"]
    assert isinstance(d2, DataModel)
    assert not isinstance(d2, MockDataModel)

    d3 = experiment_model.datasets["d3"]
    assert isinstance(d3, MockDataModel)


@pytest.mark.parametrize("dataset_scales", [[2.0], []])
def test_resolve_rejects_duplicate_scale_list_declarations(dataset_scales: list[float]):
    experiment = ExperimentModel(
        datasets={"sample": DataModel(elements=[], scale_list=dataset_scales)},
        scale_list={"sample": [2.0]},
    )
    library = ModelLibrary(root={})
    parameters = Parameters.empty()

    with pytest.raises(GlotaranUserError, match=r"Dataset 'sample'.*scale_list.*both"):
        experiment.resolve(library, parameters, initial=parameters)


def test_resolve_keeps_experiment_scale_list_only_in_resolved_dataset():
    experiment = ExperimentModel(
        datasets={"sample": DataModel(elements=[])}, scale_list={"sample": [2.0]}
    )
    library = ModelLibrary(root={})
    parameters = Parameters.empty()

    resolved = experiment.resolve(library, parameters, initial=parameters)
    resolved_again = resolved.resolve(library, parameters, initial=parameters)

    assert resolved_again.datasets["sample"].scale_list == [2.0]
    assert experiment.datasets["sample"].scale_list is None
    assert experiment.scale_list == {"sample": [2.0]}
