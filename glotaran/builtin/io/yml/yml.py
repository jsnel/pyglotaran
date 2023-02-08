"""Module containing the YAML Data and Project IO plugins."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

from glotaran.builtin.io.yml.utils import load_dict
from glotaran.io import ProjectIoInterface
from glotaran.io import register_project_io
from glotaran.parameter import Parameters
from glotaran.project import Result
from glotaran.project import Scheme
from glotaran.project.dataclass_helpers import fromdict
from glotaran.utils.sanitize import sanitize_yaml

if TYPE_CHECKING:
    from typing import Any


@register_project_io(["yml", "yaml", "yml_str"])
class YmlProjectIo(ProjectIoInterface):
    """Plugin for YAML project io."""

    def load_parameters(self, file_name: str) -> Parameters:
        """Load :class:`Parameters` instance from the specification defined in ``file_name``.

        Parameters
        ----------
        file_name: str
            File containing the parameter specification.

        Returns
        -------
        Parameters
        """  # noqa:  D414
        spec = self._load_yml(file_name)

        if isinstance(spec, list):
            return Parameters.from_list(spec)
        else:
            return Parameters.from_dict(spec)

    def load_scheme(self, file_name: str) -> Scheme:
        """Load :class:`Scheme` instance from the specification defined in ``file_name``.

        Parameters
        ----------
        file_name: str
            File containing the scheme specification.

        Returns
        -------
        Scheme
        """
        spec = sanitize_yaml(self._load_yml(file_name), do_values=True)
        return Scheme.from_dict(spec)

    #  def save_scheme(self, scheme: Scheme, file_name: str):
    #      """Write a :class:`Scheme` instance to a specification file ``file_name``.
    #
    #      Parameters
    #      ----------
    #      scheme: Scheme
    #          :class:`Scheme` instance to save to file.
    #      file_name: str
    #          Path to the file to write the scheme specification to.
    #      """
    #      scheme_dict = asdict(scheme, folder=Path(file_name).parent)
    #      write_dict(scheme_dict, file_name=file_name)

    def load_result(self, result_path: str) -> Result:
        """Create a :class:`Result` instance from the specs defined in a file.

        Parameters
        ----------
        result_path : str
            Path containing the result data.

        Returns
        -------
        Result
            :class:`Result` instance created from the saved format.
        """
        result_file_path = Path(result_path)
        if result_file_path.suffix not in [".yml", ".yaml"]:
            result_file_path = result_file_path / "result.yml"
        spec = self._load_yml(result_file_path.as_posix())
        if "number_of_data_points" in spec:
            spec["number_of_residuals"] = spec.pop("number_of_data_points")
        if "number_of_parameters" in spec:
            spec["number_of_free_parameters"] = spec.pop("number_of_parameters")
        return fromdict(Result, spec, folder=result_file_path.parent)

    #  def save_result(
    #      self,
    #      result: Result,
    #      result_path: str,
    #      saving_options: SavingOptions = SAVING_OPTIONS_DEFAULT,
    #  ) -> list[str]:
    #      """Write a :class:`Result` instance to a specification file and data files.
    #
    #      Returns a list with paths of all saved items.
    #      The following files are saved if not configured otherwise:
    #      * ``result.md``: The result with the model formatted as markdown text.
    #      * ``result.yml``: Yaml spec file of the result
    #      * ``model.yml``: Model spec file.
    #      * ``scheme.yml``: Scheme spec file.
    #      * ``initial_parameters.csv``: Initially used parameters.
    #      * ``optimized_parameters.csv``: The optimized parameter as csv file.
    #      * ``parameter_history.csv``: Parameter changes over the optimization
    #      * ``optimization_history.csv``: Parsed table printed by the SciPy optimizer
    #      * ``{dataset_label}.nc``: The result data for each dataset as NetCDF file.
    #
    #      Parameters
    #      ----------
    #      result: Result
    #          :class:`Result` instance to write.
    #      result_path: str
    #          Path to write the result data to.
    #      saving_options: SavingOptions
    #          Options for saving the the result.
    #
    #      Returns
    #      -------
    #      list[str]
    #          List of file paths which were created.
    #      """
    #      result_folder = Path(result_path).parent
    #      paths = save_result(
    #          result,
    #          result_folder,
    #          format_name="folder",
    #          saving_options=saving_options,
    #          allow_overwrite=True,
    #          used_inside_of_plugin=True,
    #      )
    #
    #      model_path = result_folder / "model.yml"
    #      save_model(result.scheme.model, model_path, allow_overwrite=True)
    #      paths.append(model_path.as_posix())
    #
    #      scheme_path = result_folder / "scheme.yml"
    #      save_scheme(result.scheme, scheme_path, allow_overwrite=True)
    #      paths.append(scheme_path.as_posix())
    #
    #      result_dict = asdict(result, folder=result_folder)
    #      write_dict(result_dict, file_name=result_path)
    #      paths.append(result_path)
    #
    #      return paths

    def _load_yml(self, file_name: str) -> dict[str, Any]:
        return load_dict(file_name, self.format != "yml_str")
