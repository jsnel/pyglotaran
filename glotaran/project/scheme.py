"""The module for :class:``Scheme``."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING

from glotaran.io import load_scheme
from glotaran.model import Model
from glotaran.parameter import Parameters
from glotaran.project.dataclass_helpers import file_loadable_field
from glotaran.project.dataclass_helpers import init_file_loadable_fields
from glotaran.utils.io import DatasetMapping
from glotaran.utils.ipython import MarkdownStr
from typing import Union
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Mapping
    from typing import Literal

    import xarray as xr

    from glotaran.typing import StrOrPath


@dataclass
class Scheme:
    """A scheme is a collection of a model, parameters and a dataset.

    A scheme also holds options for optimization.
    """

    model: Model = file_loadable_field(Model)  # type:ignore[type-var]
    parameters: Parameters = file_loadable_field(Parameters)  # type:ignore[type-var]
    data: Mapping[str, xr.Dataset] = file_loadable_field(
        DatasetMapping, is_wrapper_class=True
    )  # type:ignore[type-var]

    clp_link_tolerance: float = 0.0
    clp_link_method: Literal["nearest", "backward", "forward"] = "nearest"
    compute_clp_standard_error: bool = False
    clp_standard_error_finite_difference_relative_step: float = 1e-6

    maximum_number_function_evaluations: int | None = None
    add_svd: bool = True
    ftol: float = 1e-8
    gtol: float = 1e-8
    xtol: float = 1e-8
    x_scale: Union[float, str, np.ndarray] = 1.0
    optimization_method: Literal[
        "TrustRegionReflection",
        "Dogbox",
        "Levenberg-Marquardt",
    ] = "TrustRegionReflection"
    result_path: str | None = None
    source_path: StrOrPath = field(
        default="scheme.yml", init=False, repr=False, metadata={"exclude_from_dict": True}
    )
    loader: Callable[[StrOrPath], Scheme] = field(
        default=load_scheme, init=False, repr=False, metadata={"exclude_from_dict": True}
    )

    def __post_init__(self):
        """Override attributes after initialization."""
        init_file_loadable_fields(self)

    def validate(self) -> MarkdownStr:
        """Return a string listing all problems in the model and missing parameters.

        In addition to the model/parameter cross-reference check performed by
        ``Model.validate``, this method also detects:

        * **Double declarations** – the same label appearing more than once in
          the source CSV file (silently overwritten during loading, the second
          value wins without warning).
        * **Initial guess outside declared bounds** – any parameter whose
          ``value`` is strictly below its ``minimum`` or above its ``maximum``
          (which would raise a ``ValueError`` inside ``optimize()``).

        Returns
        -------
        MarkdownStr
            A user-friendly string when no problems are found.

        Raises
        ------
        ValueError
            Raised when any parameter problem is detected (double declaration
            or initial guess outside bounds), so that execution stops before
            ``optimize()`` is called.
        """
        import csv
        import math
        import pathlib
        from collections import Counter

        result = str(self.model.validate(self.parameters))
        parameter_issues: list[str] = []

        # --- Double declaration: re-read the source CSV to count raw labels ---
        source_path = getattr(self.parameters, "source_path", None)
        if source_path is not None:
            csv_path = pathlib.Path(source_path)
            if csv_path.suffix.lower() == ".csv":
                try:
                    with open(csv_path, newline="", encoding="utf-8") as fh:
                        raw_labels = [
                            row["label"].strip()
                            for row in csv.DictReader(fh)
                            if row.get("label", "").strip()
                        ]
                    for lbl, cnt in sorted(Counter(raw_labels).items()):
                        if cnt > 1:
                            parameter_issues.append(
                                f"Parameter '{lbl}' is declared {cnt} times" f" in '{csv_path}'"
                            )
                except (OSError, KeyError):
                    pass  # file not accessible – skip duplicate check

        # --- Out-of-bounds initial guess ---
        for param in self.parameters.all():
            val = param.value
            if val is None or (isinstance(val, float) and math.isnan(val)):
                continue
            mn = param.minimum if not math.isinf(param.minimum) else None
            mx = param.maximum if not math.isinf(param.maximum) else None
            if mn is not None and val < mn:
                parameter_issues.append(
                    f"Parameter '{param.label}' initial value {val:.6g}"
                    f" is below its minimum {mn:.6g}"
                )
            if mx is not None and val > mx:
                parameter_issues.append(
                    f"Parameter '{param.label}' initial value {val:.6g}"
                    f" exceeds its maximum {mx:.6g}"
                )

        if parameter_issues:
            n = len(parameter_issues)
            problems = "\n".join(f"  * {issue}" for issue in parameter_issues)
            raise ValueError(
                f"Scheme.validate() found {n} parameter"
                f" problem{'s' if n > 1 else ''} -"
                f" fix before calling optimize():\n{problems}"
            )

        return MarkdownStr(result)

    def valid(self) -> bool:
        """Check if there are no problems with the model or the parameters.

        Returns
        -------
        bool
            Whether the scheme is valid.
        """
        return self.model.valid(self.parameters)

    def markdown(self):
        """Format the :class:`Scheme` as markdown string.

        Returns
        -------
        MarkdownStr
            The scheme as markdown string.
        """
        model_markdown_str = self.model.markdown(parameters=self.parameters)

        markdown_str = "\n\n__Scheme__\n\n"
        markdown_str += (
            "* *maximum_number_function_evaluations*: "
            f"{self.maximum_number_function_evaluations}\n"
        )
        markdown_str += f"* *clp_link_tolerance*: {self.clp_link_tolerance}\n"
        markdown_str += f"* *compute_clp_standard_error*: {self.compute_clp_standard_error}\n"

        return model_markdown_str + MarkdownStr(markdown_str)

    def _repr_markdown_(self) -> str:
        """Return a markdown representation str.

        Special method used by ``ipython`` to render markdown.

        Returns
        -------
        str
            The scheme as markdown string.
        """
        return str(self.markdown())

    def __str__(self) -> str:
        """Representation used by print and str."""
        return str(self.markdown())
