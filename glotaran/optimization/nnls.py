"""Module for residual calculation with the non-negative least-squares method."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import nnls

if TYPE_CHECKING:
    from glotaran.typing.types import ArrayLike


def residual_nnls(matrix: ArrayLike, data: ArrayLike) -> tuple[ArrayLike, ArrayLike]:
    """Calculate the conditionally linear parameters and residual with the NNLS method.

    NNLS stands for 'non-negative least-squares'.

    Parameters
    ----------
    matrix : ArrayLike
        The model matrix.
    data : ArrayLike
        The data to analyze.

    Returns
    -------
    tuple[ArrayLike, ArrayLike]
        The clps and the residual.
    """
    # Difficult correlated designs can exceed SciPy's default 3 * n iterations.
    clp, _ = nnls(matrix, data, maxiter=6 * np.asarray(matrix).shape[1])
    residual = data - np.dot(matrix, clp)
    return clp, residual
