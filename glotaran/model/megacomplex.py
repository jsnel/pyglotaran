"""This module contains the megacomplex."""
from __future__ import annotations

import abc
from typing import TYPE_CHECKING
from typing import ClassVar

import xarray as xr

from glotaran.model.item import LibraryItemTyped
from glotaran.plugin_system.megacomplex_registration import register_megacomplex

if TYPE_CHECKING:

    from glotaran.model.data_model import DataModel


class Megacomplex(LibraryItemTyped, abc.ABC):  # type:ignore[misc]
    """A base class for megacomplex models.

    Subclasses must overwrite :method:`glotaran.model.Megacomplex.calculate_matrix`.
    """

    data_model_type: ClassVar[type | None] = None
    is_exclusive: ClassVar[bool] = False
    is_unique: ClassVar[bool] = False
    register_as: ClassVar[str | None] = None

    dimension: str | None = None

    def __init_subclass__(cls):
        """Register the megacomplex if necessary."""
        super().__init_subclass__()
        if cls.register_as is not None:
            register_megacomplex(cls.register_as, cls)

    @abc.abstractmethod
    def calculate_matrix(
        self,
<<<<<<< HEAD
        dataset_model: DatasetModel,
        global_axis: ArrayLike,
        model_axis: ArrayLike,
=======
        model: DataModel,
        global_axis: np.typing.ArrayLike,
        model_axis: np.typing.ArrayLike,
>>>>>>> b223d79b (Remove old model and items.)
        **kwargs,
    ) -> tuple[list[str], ArrayLike]:
        """Calculate the megacomplex matrix.

        Parameters
        ----------
<<<<<<< HEAD
        dataset_model: DatasetModel
            The dataset model.
        global_axis: ArrayLike
=======
        data_model: DataModel
            The data model.
        global_axis: np.typing.ArrayLike
>>>>>>> b223d79b (Remove old model and items.)
            The global axis.
        model_axis: ArrayLike
            The model axis.
        **kwargs
            Additional arguments.

        Returns
        -------
        tuple[list[str], ArrayLike]:
            The clp labels and the matrix.

        .. # noqa: DAR202
        """
        pass

    def finalize_data(
        self,
        model: DataModel,
        data: xr.Dataset,
        is_full_model: bool = False,
        as_global: bool = False,
    ):
        """Finalize the result data.

        Parameters
        ----------
        data_model: DataModel
            The data model.
        data: xr.Dataset
            The data.
        is_full_model: bool
            Whether the model is a full model.
        as_global: bool
            Whether megacomplex is calculated as global megacomplex.
        """
        pass
