from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr
from scipy.special import erf

from glotaran.builtin.megacomplexes.decay.decay_parallel_megacomplex import DecayDatasetModel
from glotaran.builtin.megacomplexes.decay.irf import IrfMultiGaussian
from glotaran.model import DatasetModel
from glotaran.model import ItemIssue
from glotaran.model import Megacomplex
from glotaran.model import Model
from glotaran.model import ParameterType
from glotaran.model import attribute
from glotaran.model import item
from glotaran.model import megacomplex
from glotaran.parameter import Parameters
from glotaran.builtin.megacomplexes.decay.decay_matrix_gaussian_irf import calculate_decay_matrix_gaussian_irf_on_index

if TYPE_CHECKING:
    from glotaran.typing.types import ArrayLike


def index_dependent(*args, **kwargs):
    return True


class OscillationParameterIssue(ItemIssue):
    def __init__(self, label: str, len_labels: int, len_frequencies: int, len_rates: int):
        self._label = label
        self._len_labels = len_labels
        self._len_frequencies = len_frequencies
        self._len_rates = len_rates

    def to_string(self) -> str:
        return (
            f"Size of labels ({self.len_labels}), frequencies ({self.len_frequencies}) "
            f"and rates ({self.len_rates}) does not match for damped oscillation "
            f"megacomplex '{self.label}'."
        )


def validate_pfid_parameter(
    labels: list[str],
    pfid: PFIDMegacomplex,
    model: Model,
    parameters: Parameters | None,
) -> list[ItemIssue]:
    issues = []

    len_labels, len_frequencies, len_rates = (
        len(pfid.labels),
        len(pfid.frequencies),
        len(pfid.rates),
    )

    if len({len_labels, len_frequencies, len_rates}) > 1:
        issues.append(
            OscillationParameterIssue(pfid.label, len_labels, len_frequencies, len_rates)
        )

    return issues
@item
class PFIDDatasetModel(DatasetModel):
    spectral_axis_inverted: bool = False
    spectral_axis_scale: float = 1

@megacomplex(dataset_model_type=PFIDDatasetModel)
class PFIDMegacomplex(Megacomplex):
    dimension: str = "time"
    type: str = "pfid"
    labels: list[str] = attribute(validator=validate_pfid_parameter)
    frequencies: list[ParameterType]  # omega_a
    rates: list[ParameterType]  # 1/T2
    alpha: list[ParameterType]
    kappa: list[ParameterType]


    def calculate_matrix(
        self,
        dataset_model: DatasetModel,
        global_axis: ArrayLike,
        model_axis: ArrayLike,
        **kwargs,
    ):
        clp_label = [f"{label}_pfid" for label in self.labels]

        frequencies = np.array(self.frequencies)
        rates = np.array(self.rates)
        alpha = np.array(self.alpha)
        kappa = np.array(self.kappa)

        if dataset_model.spectral_axis_inverted:
            frequencies = dataset_model.spectral_axis_scale / frequencies
        elif dataset_model.spectral_axis_scale != 1:
            frequencies = frequencies * dataset_model.spectral_axis_scale

        irf = dataset_model.irf
        matrix_shape = (
            (global_axis.size, model_axis.size, len(clp_label))
            if index_dependent(dataset_model)
            else (model_axis.size, len(clp_label))
        )
        # matrix = np.ones(matrix_shape, dtype=np.float64)
        matrix = np.zeros(matrix_shape, dtype=np.float64)

        if irf is None:
            raise ValueError("IRF is required for PFID megacomplex")
        elif isinstance(irf, IrfMultiGaussian):
            if index_dependent(dataset_model):
                for i in range(global_axis.size):
                    calculate_pfid_matrix_gaussian_irf_on_index(
                        matrix[i], frequencies, rates, alpha, kappa, irf, i, global_axis, model_axis
                    )
            else:
                calculate_pfid_matrix_gaussian_irf_on_index(
                    matrix, frequencies, rates, alpha, kappa, irf, None, global_axis, model_axis
                )

        return clp_label, matrix

    def finalize_data(
        self,
        dataset_model: DatasetModel,
        dataset: xr.Dataset,
        is_full_model: bool = False,
        as_global: bool = False,
    ):
        if is_full_model:
            return

        megacomplexes = (
            dataset_model.global_megacomplex if is_full_model else dataset_model.megacomplex
        )
        unique = len([m for m in megacomplexes if isinstance(m, PFIDMegacomplex)]) < 2

        prefix = "pfid" if unique else f"{self.label}_pfid"

        dataset.coords[f"{prefix}"] = self.labels
        dataset.coords[f"{prefix}_frequency"] = (prefix, self.frequencies)
        dataset.coords[f"{prefix}_rate"] = (prefix, self.rates)

        model_dimension = dataset.attrs["model_dimension"]
        global_dimension = dataset.attrs["global_dimension"]
        dim1 = dataset.coords[global_dimension].size
        dim2 = len(self.labels)
        pfid = np.zeros((dim1, dim2), dtype=np.float64)
        for i, label in enumerate(self.labels):
            pfid[:, i] = dataset.clp.sel(clp_label=f"{label}_pfid")

        dataset[f"{prefix}_associated_spectra"] = (
            (global_dimension, prefix),
            pfid,
        )

        # always index dependent
        dataset[f"{prefix}_associated_concentration"] = (
            (
                global_dimension,
                model_dimension,
                prefix,
            ),
            dataset.matrix.sel(clp_label=[f"{label}_pfid" for label in self.labels]).values,
        )


def calculate_pfid_matrix_gaussian_irf_on_index(
    matrix: ArrayLike,
    frequencies: ArrayLike,
    rates: ArrayLike,
    alpha: ArrayLike,
    kappa: ArrayLike,
    irf: IrfMultiGaussian,
    global_index: int | None,
    global_axis: ArrayLike,
    model_axis: ArrayLike,
):
    centers, widths, scales, shift, _, _ = irf.parameter(global_index, global_axis)
    for center, width, scale in zip(centers, widths, scales):
        matrix += calculate_pfid_matrix_gaussian_irf(
            frequencies,
            rates,
            alpha,
            kappa,
            model_axis,
            center,
            width,
            shift,
            scale,
            global_axis[global_index],
        )
    matrix /= np.sum(scales)


def calculate_pfid_matrix_gaussian_irf(
    frequencies: np.ndarray,
    rates: np.ndarray,
    alpha: np.ndarray,
    kappa: np.ndarray,
    model_axis: np.ndarray,
    center: float,
    width: float,
    shift: float,
    scale: float,
    global_axis_value: float,
):
    """Calculate the damped oscillation matrix taking into account a gaussian irf

    Parameters
    ----------
    frequencies : np.ndarray
        an array of frequencies in THz, one per oscillation
    rates : np.ndarray
        an array of dephasing rates (negative), one per oscillation
    alpha: np.ndarray
        attempt: alpha[0] is a shift of the frequencies to mimic inhomogeneous broadening
    kappa : np.ndarray
        an array of decay rates (positive), only the first is used
    model_axis : np.ndarray
        the model axis (time)
    center : float
        the center of the gaussian IRF
    width : float
        the width (σ) parameter of the the IRF
    shift : float
        a shift parameter per item on the global axis
    scale : float
        the scale parameter to scale the matrix by

    Returns
    -------
    np.ndarray
        An array of the real and imaginary part of the oscillation matrix,
        the shape being (len(model_axis), len(frequencies)).
    """
    shifted_axis = model_axis - center - shift
    # For calculations using the negative rates we use the time axis
    # from the beginning up to 5 σ from the irf center
    # try 20231223
    left_shifted_axis_indices = np.where(shifted_axis < 5 * width)[0]
    # modify this to enable positive time response according to Hamm 1995, eq.1.6
    # left_shifted_axis_indices = np.where(shifted_axis < 0)[0]
    left_shifted_axis = shifted_axis[left_shifted_axis_indices]
    neg_idx = np.where(rates < 0)[0]
    # For calculations using the positive rates axis we use the time axis
    # from 5 σ before the irf center until the end
    # right_shifted_axis_indices = np.where(shifted_axis > -5 * width)[0]
    # modify this to enable positive time response according to Hamm 1995, eq.1.6
    right_shifted_axis_indices = np.where(shifted_axis > 0)[0]
    right_shifted_axis = shifted_axis[right_shifted_axis_indices]
    pos_idx = np.where(rates >= 0)[0]

    # c multiply by 0.03 to convert wavenumber (cm-1) to frequency (THz)
    # where 0.03 is the product of speed of light 3*10**10 cm/s and time-unit ps (10^-12)
    # we postpone the conversion because the global axis is
    # always expected to be in cm-1 for relevant experiments
    frequency_diff = (global_axis_value - frequencies) * 0.03 * 2 * np.pi
    # 20230925
    # attempt: alpha[0] is a shift of the frequencies to mimic inhomogeneous broadening
    frequency_diff2 = (global_axis_value - frequencies-alpha[0]) * 0.03 * 2 * np.pi
    # frequency_diff = alpha[0] * (global_axis_value - frequencies) * 0.03 * 2 * np.pi
    d = width**2
    k = rates + 1j * frequency_diff
    dk = k * d
    k2 = rates + 1j * frequency_diff2
    dk2 = k2 * d
    sqwidth = np.sqrt(2) * width

    a = np.zeros((len(model_axis), len(rates)), dtype=np.complex128)
    a2 = np.zeros((len(model_axis), len(rates)), dtype=np.complex128)
    # a[np.ix_(right_shifted_axis_indices, pos_idx)] = np.exp(
    #     (-1 * right_shifted_axis[:, None] + 0.5 * dk[pos_idx]) * k[pos_idx]
    # )

    # a[np.ix_(left_shifted_axis_indices, neg_idx)] = np.exp(
    # try 20231223
    # a = np.exp((-1 * shifted_axis[:, None] + 0.5 * dk[:]) * k[:])
    # a2 = np.exp((-1 * shifted_axis[:, None] + 0.5 * dk2[:]) * k2[:])
    a[np.ix_(left_shifted_axis_indices, neg_idx)] = np.exp((-1 * left_shifted_axis[:, None] + 0.5 * dk[:]) * k[:])
    a2[np.ix_(left_shifted_axis_indices, neg_idx)] = np.exp((-1 * left_shifted_axis[:, None] + 0.5 * dk2[:]) * k2[:])

    b = np.zeros((len(model_axis), len(rates)), dtype=np.complex128)
    b2 = np.zeros((len(model_axis), len(rates)), dtype=np.complex128)
    # b[np.ix_(right_shifted_axis_indices, pos_idx)] = 1 + erf(
    #     (right_shifted_axis[:, None] - dk[pos_idx]) / sqwidth
    # )
    # For negative rates we flip the sign of the `erf` by using `-sqwidth` in lieu of `sqwidth`
    # b[np.ix_(left_shifted_axis_indices, neg_idx)] = 1 + erf(
    # b[np.ix_(True, neg_idx)] = 1 + erf(
    # b[np.ix_(neg_idx)] = 1 + erf(
    b = 1 + erf((shifted_axis[:, None] - dk[:]) / -sqwidth)
    b2 = 1 + erf((shifted_axis[:, None] - dk2[:]) / -sqwidth)
    # c = np.zeros((len(model_axis), len(rates)), dtype=np.complex128)
    c = np.zeros((len(model_axis), len(rates)), dtype=np.float64)
    # this c term describes a nondecaying excited state following Hamm 1995
    # c = 1 - erf((shifted_axis[:, None]) / -sqwidth)
    # this c term describes an excited state decaying with rate kd(ecay)
    # temporarily kd 15, to be parameterized
    # kd=np.zeros((len(rates)), dtype=np.complex128)+15.
    kd=np.zeros((len(rates)), dtype=np.float64)+kappa[0]
    # TODO we need to call calculate_decay_matrix_gaussian_irf_on_index to avoid overflows
    calculate_decay_matrix_gaussian_irf_on_index(matrix=c,
    rates=kd,
    times=model_axis,
    centers=np.array([center]),
    widths=np.array([width]),
    scales=np.array([1.]),
    backsweep=False,
    backsweep_period=1.
    )
    # dkd = kd * d
    # c1 = np.zeros((len(model_axis), len(rates)), dtype=np.complex128)
    # c2 = np.zeros((len(model_axis), len(rates)), dtype=np.complex128)
    # c1 = np.exp((1 * shifted_axis[:, None] + 0.5 * dkd[:]) * kd[:])
    # c2 = 1 - erf((shifted_axis[:, None]+dkd[:]) / sqwidth)
    # c = c1 * c2
    # added a minus to facilitate the NNLS fit of the instantaneous bleach
    osc = -(a * b + c) * scale
    osc2 = -(a2 * b2 + c) * scale
    # output = np.zeros((len(model_axis), len(rates)), dtype=np.float64)
    output = (osc.real * rates - frequency_diff * osc.imag) / (rates**2 + frequency_diff**2)
    output = output+(osc2.real * rates - frequency_diff2 * osc2.imag) / (rates**2 + frequency_diff2**2)
    # minus to mimic two bands alpha apart with opposite sign or derivative when alpha is small
    # output = output-(osc2.real * rates - frequency_diff2 * osc2.imag) / (rates**2 + frequency_diff2**2)
    # output = (osc.real * alpha[0] * rates - frequency_diff * osc.imag) / (
    #     (alpha[0] * rates) ** 2 + frequency_diff**2
    # )
    # here we must put a constant in the numerator for positive time
    # output[np.ix_(right_shifted_axis_indices, neg_idx)] = -scale/(rates**2 + frequency_diff**2)
    return output
