"""This module contains the data model."""

from __future__ import annotations

from collections.abc import Generator
from typing import Any
from typing import Literal
from uuid import uuid4

import xarray as xr
from pydantic import Field
from pydantic import create_model

from glotaran.model.element import Element
from glotaran.model.errors import GlotaranModelError
from glotaran.model.errors import GlotaranUserError
from glotaran.model.errors import ItemIssue
from glotaran.model.item import Attribute
from glotaran.model.item import Item
from glotaran.model.item import ParameterType
from glotaran.model.item import resolve_item_parameters
from glotaran.model.weight import Weight
from glotaran.parameter import Parameter
from glotaran.parameter import Parameters


class ExclusiveModelIssue(ItemIssue):
    """Issue for exclusive elements."""

    def __init__(self, label: str, element_type: str, is_global: bool):
        """Create an ExclusiveModelIssue.

        Parameters
        ----------
        label : str
            The element label.
        element_type : str
            The element type.
        is_global : bool
            Whether the element is global.
        """
        self._label = label
        self._type = element_type
        self._is_global = is_global

    def to_string(self) -> str:
        """Get the issue as string.

        Returns
        -------
        str
        """
        return (
            f"Exclusive {'global ' if self._is_global else ''}element '{self._label}' of "
            f"type '{self._type}' cannot be combined with other elements."
        )


class UniqueModelIssue(ItemIssue):
    """Issue for unique elements."""

    def __init__(self, label: str, element_type: str, is_global: bool):
        """Create a UniqueModelIssue.

        Parameters
        ----------
        label : str
            The element label.
        element_type : str
            The element type.
        is_global : bool
            Whether the element is global.
        """
        self._label = label
        self._type = element_type
        self._is_global = is_global

    def to_string(self):
        """Get the issue as string.

        Returns
        -------
        str
        """
        return (
            f"Unique {'global ' if self._is_global else ''}element '{self._label}' of "
            f"type '{self._type}' can only be used once per dataset."
        )


def get_element_issues(value: list[str | Element] | None, is_global: bool) -> list[ItemIssue]:
    """Get issues for elements.

    Parameters
    ----------
    value: list[str | Element] | None
        A list of elements.
    element: Element
        The element.
    is_global: bool
        Whether the elements are global.

    Returns
    -------
    list[ItemIssue]
    """
    issues: list[ItemIssue] = []

    if value is not None:
        elements = [v for v in value if isinstance(v, Element)]
        for element in elements:
            element_type = element.__class__
            if element_type.is_exclusive and len(elements) > 1:
                issues.append(ExclusiveModelIssue(element.label, element.type, is_global))
            if (
                element_type.is_unique
                and len([m for m in elements if m.__class__ is element_type]) > 1
            ):
                issues.append(UniqueModelIssue(element.label, element.type, is_global))
    return issues


def validate_elements(
    value: list[str | Element],
    data_model: DataModel,
    parameters: Parameters | None,
) -> list[ItemIssue]:
    """Get issues for dataset model elements.

    Parameters
    ----------
    value: list[str | Element]
        A list of elements.
    dataset_model: DatasetModel
        The dataset model.
    element: Element
        The element.
    parameters: Parameters | None,
        The parameters.

    Returns
    -------
    list[ItemIssue]
    """
    return get_element_issues(value, False)


def validate_global_elements(
    value: list[str | Element] | None,
    data_model: DataModel,
    parameters: Parameters | None,
) -> list[ItemIssue]:
    """Get issues for dataset model global elements.

    Parameters
    ----------
    value: list[str | Element] | None
        A list of elements.
    dataset_model: DatasetModel
        The dataset model.
    element: Element
        The element.
    parameters: Parameters | None,
        The parameters.

    Returns
    -------
    list[ItemIssue]
    """
    return get_element_issues(value, True)


class DataModel(Item):
    """A model for datasets."""

    data: str | xr.Dataset | None = None
    extra_data: str | xr.Dataset | None = None
    elements: list[Element | str] = Attribute(
        description="The elements contributing to this dataset.",
        validator=validate_elements,  # type:ignore[arg-type]
    )
    element_scale: list[ParameterType] | None = None
    global_elements: list[Element | str] | None = Attribute(
        default=None,
        description="The global elements contributing to this dataset.",
        validator=validate_global_elements,  # type:ignore[arg-type]
    )
    global_element_scale: list[ParameterType] | None = None
    residual_function: Literal["variable_projection", "non_negative_least_squares"] = Attribute(
        default="variable_projection", description="The residual function to use."
    )
    weights: list[Weight] = Field(default_factory=list)

    @classmethod
    def from_dict(cls, library: dict[str, Element], model_dict: dict[str, Any]) -> DataModel:
        data_model_cls_name = f"GlotaranDataModel_{str(uuid4()).replace('-','_')}"
        element_labels = model_dict.get("elements", []) + model_dict.get("global_elements", [])
        if len(element_labels) == 0:
            raise GlotaranModelError("No element defined for dataset")
        elements = {type(library[label]) for label in element_labels}
        data_models = [
            m.data_model_type for m in filter(lambda m: m.data_model_type is not None, elements)
        ] + [DataModel]
        return create_model(data_model_cls_name, __base__=tuple(data_models))(**model_dict)


def is_data_model_global(data_model: DataModel) -> bool:
    """Check if a data model can model the global dimension.

    Parameters
    ----------
    data_model: DataModel
        The data model.

    Returns
    -------
    bool
    """
    return data_model.global_elements is not None and len(data_model.global_elements) != 0


def get_data_model_dimension(data_model: DataModel) -> str:
    """Get the data model's model dimension.

    Parameters
    ----------
    data_model: DataModel
        The data model.

    Returns
    -------
    str

    Raises
    ------
    ValueError
        Raised if the data model does not have elements or if it is not filled.
    """
    if len(data_model.elements) == 0:
        raise GlotaranModelError(f"No elements set for data model '{data_model.label}'.")
    if any(isinstance(m, str) for m in data_model.elements):
        raise GlotaranUserError(f"Data model '{data_model.label}' was not resolved.")
    model_dimension: str = data_model.elements[0].dimension  # type:ignore[union-attr, assignment]
    if any(
        model_dimension != m.dimension  # type:ignore[union-attr]
        for m in data_model.elements
    ):
        raise GlotaranModelError("Model dimensions do not match for data model.")
    if model_dimension is None:
        raise GlotaranModelError("No models dimensions defined for data model.")
    return model_dimension


def iterate_data_model_elements(
    data_model: DataModel,
) -> Generator[tuple[Parameter | str | None, Element | str], None, None]:
    """Iterate the data model's elements.

    Parameters
    ----------
    data_model: DataModel
        The data model.

    Yields
    ------
    tuple[Parameter | str | None, Element | str]
        A scale and elements.
    """
    for i, element in enumerate(data_model.elements):
        scale = data_model.elements_scale[i] if data_model.element_scale is not None else None
        yield scale, element


def iterate_data_model_global_elements(
    data_model: DataModel,
) -> Generator[tuple[Parameter | str | None, Element | str], None, None]:
    """Iterate the data model's global elements.

    Parameters
    ----------
    data_model: DataModel
        The data model.

    Yields
    ------
    tuple[Parameter | str | None, Element | str]
        A scale and element.
    """
    if data_model.global_elements is None:
        return
    for i, element in enumerate(data_model.global_elements):
        scale = (
            data_model.global_element_scale[i]
            if data_model.global_element_scale is not None
            else None
        )
        yield scale, element


def resolve_data_model(
    model: DataModel,
    library: dict[str, Element],
    parameters: Parameters,
    initial: Parameters | None = None,
) -> DataModel:
    model = model.copy()
    model.elements = [library[m] if isinstance(m, str) else m for m in model.elements]
    if model.global_elements is not None:
        model.global_elements = [
            library[m] if isinstance(m, str) else m for m in model.global_elements
        ]
    return resolve_item_parameters(model, parameters, initial)


def finalize_data_model(data_model: DataModel, data: xr.Dataset):
    """Finalize a data by applying all model finalize methods.

    Parameters
    ----------
    data_model: DataModel
        The data model.
    data: xr.Dataset
        The data.
    """
    is_full_model = is_data_model_global(data_model)
    for model in data_model.models:
        model.finalize_data(  # type:ignore[union-attr]
            data_model, data, is_full_model=is_full_model
        )
    if is_full_model and data_model.global_models is not None:
        for model in data_model.global_models:
            model.finalize_data(  # type:ignore[union-attr]
                data_model, data, is_full_model=is_full_model, as_global=True
            )
