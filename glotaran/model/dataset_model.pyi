from __future__ import annotations

from typing import Any
from typing import Generator
from typing import Hashable

import numpy as np
import xarray as xr

from glotaran.model.megacomplex import Megacomplex
from glotaran.model.model import Model
from glotaran.parameter import Parameter

def create_dataset_model_type(properties: dict[str, Any]) -> type[DatasetModel]: ...

class DatasetModel:

    label: str
    megacomplex: list[str]
    megacomplex_scale: list[Parameter] | None
    global_megacomplex: list[str]
    global_megacomplex_scale: list[Parameter] | None
    scale: Parameter | None
    _coords: dict[Hashable, np.ndarray]
    def iterate_megacomplexes(
        self,
    ) -> Generator[tuple[Parameter | int | None, Megacomplex | str], None, None]: ...
    def iterate_global_megacomplexes(
        self,
    ) -> Generator[tuple[Parameter | int | None, Megacomplex | str], None, None]: ...
    def get_model_dimension(self) -> str: ...
    def finalize_data(self, dataset: xr.Dataset) -> None: ...
    def overwrite_model_dimension(self, model_dimension: str) -> None: ...
    def get_global_dimension(self) -> str: ...
    def overwrite_global_dimension(self, global_dimension: str) -> None: ...
    def swap_dimensions(self) -> None: ...
    def set_data(self, dataset: xr.Dataset) -> DatasetModel: ...
    def get_data(self) -> np.ndarray: ...
    def get_weight(self) -> np.ndarray | None: ...
    def index_dependent(self) -> bool: ...
    def overwrite_index_dependent(self, index_dependent: bool): ...
    def has_global_model(self) -> bool: ...
    def set_coordinates(self, coords: dict[str, np.ndarray]): ...
    def get_coordinates(self) -> dict[Hashable, np.ndarray]: ...
    def get_model_axis(self) -> np.ndarray: ...
    def get_global_axis(self) -> np.ndarray: ...
    def ensure_unique_megacomplexes(self, model: Model) -> list[str]: ...
