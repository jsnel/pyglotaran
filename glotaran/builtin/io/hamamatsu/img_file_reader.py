"""Data IO plugin for Hamamatsu ITEX ``*.img`` streak-camera files."""

from __future__ import annotations

import re
import struct
from pathlib import Path
from typing import BinaryIO

import numpy as np
import xarray as xr

from glotaran.io import DataIoInterface
from glotaran.io import register_data_io
from glotaran.io.prepare_dataset import prepare_time_trace_dataset

_HEADER_SIZE = 64
_MAGIC = b"IM"
_FILE_TYPE_TO_DTYPE = {
    0: np.dtype("<u1"),
    2: np.dtype("<u2"),
    3: np.dtype("<u4"),
}
_FIXED_CALIBRATION_LENGTH = {
    "*": 1024,
    "+": 1280,
}


def _read_exact(file: BinaryIO, size: int, description: str) -> bytes:
    """Read exactly ``size`` bytes or fail with a format-specific error.

    Parameters
    ----------
    file : BinaryIO
        Open binary file to read from.
    size : int
        Number of bytes to read.
    description : str
        Description used in the truncation error message.

    Returns
    -------
    bytes
        The requested bytes.

    Raises
    ------
    ValueError
        If fewer than ``size`` bytes are available.
    """
    data = file.read(size)
    if len(data) != size:
        raise ValueError(f"Truncated Hamamatsu IMG file while reading {description}.")
    return data


def _extract_scaling_reference(comment: str, axis: str) -> str | None:
    """Extract an X or Y calibration-table reference from the ITEX comment.

    Parameters
    ----------
    comment : str
        ITEX comment block.
    axis : str
        Axis identifier, either ``"X"`` or ``"Y"``.

    Returns
    -------
    str | None
        Calibration-table reference, or ``None`` when no reference is present.
    """
    key = f"Scaling{axis}ScalingFile"
    match = re.search(
        rf"{re.escape(key)}\s*=\s*(?:\"(?P<quoted>[^\"]*)\"|(?P<bare>[*+]\d+|#\d+,\d+|Other))",
        comment,
        flags=re.IGNORECASE,
    )
    if match is None:
        return None
    return (match.group("quoted") or match.group("bare")).strip()


def _parse_scaling_reference(reference: str | None) -> tuple[int, int] | None:
    """Translate an ITEX calibration reference into byte offset and table length.

    Parameters
    ----------
    reference : str | None
        ITEX calibration reference using ``*offset``, ``+offset``,
        ``#offset,count``, or ``Other`` encoding.

    Returns
    -------
    tuple[int, int] | None
        Byte offset and calibration-table length, or ``None`` for missing or
        ``Other`` calibration.

    Raises
    ------
    ValueError
        If the reference uses an invalid or unsupported encoding.
    """
    if reference is None or reference.lower().startswith("other"):
        return None

    if reference[0] in _FIXED_CALIBRATION_LENGTH:
        try:
            return int(reference[1:]), _FIXED_CALIBRATION_LENGTH[reference[0]]
        except ValueError as error:
            raise ValueError(f"Invalid Hamamatsu calibration reference: {reference!r}") from error

    if reference.startswith("#"):
        try:
            position, length = reference[1:].split(",", maxsplit=1)
            return int(position), int(length)
        except ValueError as error:
            raise ValueError(f"Invalid Hamamatsu calibration reference: {reference!r}") from error

    raise ValueError(f"Unsupported Hamamatsu calibration reference: {reference!r}")


def _read_calibration(
    file: BinaryIO,
    reference: str | None,
    *,
    axis_length: int,
    image_offset: int,
) -> np.ndarray:
    """Read and crop an axis calibration table, or return pixel indices.

    Parameters
    ----------
    file : BinaryIO
        Open binary file containing the calibration table.
    reference : str | None
        ITEX calibration-table reference.
    axis_length : int
        Number of calibration values required for the image axis.
    image_offset : int
        Offset of the image region within a full calibration table.

    Returns
    -------
    np.ndarray
        Calibration values for the image axis, or pixel indices when no
        calibration is available.

    Raises
    ------
    ValueError
        If the calibration table is truncated or incompatible with the image
        axis and offset.
    """
    parsed_reference = _parse_scaling_reference(reference)
    if parsed_reference is None:
        return np.arange(axis_length, dtype=float)

    position, calibration_length = parsed_reference
    file.seek(position)
    calibration_bytes = _read_exact(
        file,
        calibration_length * np.dtype("<f4").itemsize,
        "axis calibration",
    )
    calibration = np.frombuffer(calibration_bytes, dtype="<f4").astype(float)

    if calibration_length == axis_length:
        return calibration

    start = image_offset
    stop = start + axis_length
    if start >= 0 and stop <= calibration_length:
        return calibration[start:stop]

    raise ValueError(
        "Hamamatsu IMG calibration length does not match the image axis "
        f"({calibration_length} calibration values for {axis_length} pixels, "
        f"offset {image_offset})."
    )


@register_data_io("img")
class HamamatsuImgDataIo(DataIoInterface):
    """Read Hamamatsu HPD-TA/HiPic ITEX ``*.img`` streak-camera data."""

    def load_dataset(self, file_name: str, *, prepare: bool = True) -> xr.Dataset | xr.DataArray:
        """Read a Hamamatsu ``*.img`` file as time-resolved spectral data.

        The returned data has dimensions ``("time", "spectral")``. Calibrated
        wavelength and time axes are read from the ITEX comment block when present;
        otherwise integer pixel indices are used. Descending wavelength calibration
        is reversed together with the data so the spectral axis is ascending.

        Parameters
        ----------
        file_name : str
            Path to a Hamamatsu ITEX ``*.img`` file.
        prepare : bool
            If ``True`` (default), prepare the data for global analysis by adding
            the standard pyglotaran SVD variables.

        Returns
        -------
        xr.Dataset | xr.DataArray
            Prepared dataset when ``prepare=True``; otherwise the raw DataArray.

        Raises
        ------
        NotImplementedError
            If the IMG file uses compressed data.
        ValueError
            If the file header is invalid, the data type is unsupported, or the
            file is truncated.
        """
        path = Path(file_name)
        with path.open("rb") as file:
            header = _read_exact(file, _HEADER_SIZE, "header")
            (
                magic,
                comment_length,
                width,
                height,
                x_offset,
                y_offset,
                file_type,
            ) = struct.unpack_from("<2s6h", header)

            if magic != _MAGIC:
                raise ValueError(f"File {file_name!r} is not a Hamamatsu ITEX IMG file.")
            if comment_length < 0 or width <= 0 or height <= 0:
                raise ValueError("Invalid Hamamatsu IMG header dimensions or comment length.")
            if file_type == 1:
                raise NotImplementedError("Compressed Hamamatsu IMG files are not supported.")
            if file_type not in _FILE_TYPE_TO_DTYPE:
                raise ValueError(f"Unsupported Hamamatsu IMG file type: {file_type}.")

            comment = _read_exact(file, comment_length, "comment").decode(
                "utf-8", errors="replace"
            )

            dtype = _FILE_TYPE_TO_DTYPE[file_type]
            pixel_count = width * height
            pixel_bytes = _read_exact(file, pixel_count * dtype.itemsize, "image data")
            data = (
                np.frombuffer(pixel_bytes, dtype=dtype, count=pixel_count)
                .reshape(height, width)
                .copy()
            )

            spectral = _read_calibration(
                file,
                _extract_scaling_reference(comment, "X"),
                axis_length=width,
                image_offset=x_offset,
            )
            time = _read_calibration(
                file,
                _extract_scaling_reference(comment, "Y"),
                axis_length=height,
                image_offset=y_offset,
            )

        if spectral.size > 1 and spectral[0] > spectral[1]:
            spectral = spectral[::-1].copy()
            data = data[:, ::-1].copy()

        dataset = xr.DataArray(
            data,
            coords=[("time", time), ("spectral", spectral)],
            name="data",
        )
        return prepare_time_trace_dataset(dataset) if prepare else dataset
