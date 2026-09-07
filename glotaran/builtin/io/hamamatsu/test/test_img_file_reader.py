from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from glotaran.builtin.io.hamamatsu.img_file_reader import HamamatsuImgDataIo
from glotaran.builtin.io.hamamatsu.img_file_reader import _parse_scaling_reference


def _write_img(
    path: Path,
    data: np.ndarray,
    *,
    file_type: int = 2,
    spectral: np.ndarray | None = None,
    time: np.ndarray | None = None,
    spectral_reference: str = "#",
    time_reference: str = "#",
    x_offset: int = 0,
    y_offset: int = 0,
) -> None:
    dtype = {0: np.dtype("<u1"), 2: np.dtype("<u2"), 3: np.dtype("<u4")}[file_type]
    height, width = data.shape

    x_position = 64 + data.size * dtype.itemsize + 4096
    y_position = x_position + (spectral.nbytes if spectral is not None else 0) + 4096
    scaling_entries = []
    if spectral is not None:
        if spectral_reference == "#":
            reference = f"#{x_position},{len(spectral):04d}"
        else:
            reference = f"{spectral_reference}{x_position}"
        scaling_entries.append(f'ScalingXScalingFile="{reference}"')
    if time is not None:
        if time_reference == "#":
            reference = f"#{y_position},{len(time):04d}"
        else:
            reference = f"{time_reference}{y_position}"
        scaling_entries.append(f'ScalingYScalingFile="{reference}"')
    comment = ("[Scaling]\r\n" + ",".join(scaling_entries)).encode("utf-8")

    header = bytearray(64)
    struct.pack_into(
        "<2s6h",
        header,
        0,
        b"IM",
        len(comment),
        width,
        height,
        x_offset,
        y_offset,
        file_type,
    )

    with path.open("wb") as file:
        file.write(header)
        file.write(comment)
        file.write(np.asarray(data, dtype=dtype).tobytes(order="C"))
        if spectral is not None:
            file.seek(x_position)
            file.write(np.asarray(spectral, dtype="<f4").tobytes())
        if time is not None:
            file.seek(y_position)
            file.write(np.asarray(time, dtype="<f4").tobytes())


@pytest.mark.parametrize("file_type", [0, 2, 3])
def test_load_img_pixel_axes(tmp_path: Path, file_type: int):
    path = tmp_path / "test.img"
    data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint32)
    _write_img(path, data, file_type=file_type)

    dataset = HamamatsuImgDataIo("img").load_dataset(path.as_posix(), prepare=False)

    assert isinstance(dataset, xr.DataArray)
    np.testing.assert_array_equal(dataset.data, data)
    np.testing.assert_array_equal(dataset.time, [0.0, 1.0])
    np.testing.assert_array_equal(dataset.spectral, [0.0, 1.0, 2.0])


def test_load_img_calibrated_axes_and_reverse_wavelength(tmp_path: Path):
    path = tmp_path / "calibrated.img"
    data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint16)
    spectral = np.array([600.0, 500.0, 400.0], dtype=float)
    time = np.array([-1.0, 2.0], dtype=float)
    _write_img(path, data, spectral=spectral, time=time)

    dataset = HamamatsuImgDataIo("img").load_dataset(path.as_posix(), prepare=False)

    np.testing.assert_allclose(dataset.spectral, [400.0, 500.0, 600.0])
    np.testing.assert_allclose(dataset.time, time)
    np.testing.assert_array_equal(dataset.data, data[:, ::-1])


def test_reject_non_img_file(tmp_path: Path):
    path = tmp_path / "not-an-img.img"
    path.write_bytes(b"not an IMG file" + bytes(64))

    with pytest.raises(ValueError, match="not a Hamamatsu ITEX IMG file"):
        HamamatsuImgDataIo("img").load_dataset(path.as_posix(), prepare=False)


@pytest.mark.parametrize(
    "reference, expected",
    [
        ("*123", (123, 1024)),
        ("+456", (456, 1280)),
        ("#789,0672", (789, 672)),
        ("Other", None),
    ],
)
def test_parse_scaling_reference(reference: str, expected: tuple[int, int] | None):
    assert _parse_scaling_reference(reference) == expected


def test_load_img_preserves_unsigned_16_bit_counts(tmp_path: Path):
    path = tmp_path / "unsigned.img"
    data = np.array([[0, 32768, 65535]], dtype=np.uint16)
    _write_img(path, data, file_type=2)

    dataset = HamamatsuImgDataIo(path.as_posix()).load_dataset(path.as_posix(), prepare=False)

    np.testing.assert_array_equal(dataset.data, data)


@pytest.mark.parametrize(
    "file_type, shape, spectral_reference, time_reference, x_offset, y_offset",
    [
        pytest.param(2, (1021, 530), None, None, 341, 0, id="16-bit-pixel-axes"),
        pytest.param(2, (1024, 1024), "*", "*", 0, 0, id="16-bit-full-calibration"),
        pytest.param(3, (1020, 1024), "*", "#", 0, 2, id="32-bit-cropped-calibration"),
        pytest.param(3, (1020, 1024), "+", "#", 100, 2, id="32-bit-plus-calibration"),
    ],
)
def test_load_simulated_measurement_variants(
    tmp_path: Path,
    file_type: int,
    shape: tuple[int, int],
    spectral_reference: str | None,
    time_reference: str | None,
    x_offset: int,
    y_offset: int,
):
    height, width = shape
    data = np.arange(height * width, dtype=np.uint32).reshape(shape)
    if file_type == 2:
        data %= 65536

    spectral = None
    if spectral_reference is not None:
        spectral_length = {"*": 1024, "+": 1280, "#": width}[spectral_reference]
        spectral = np.linspace(500.0, 800.0, spectral_length)

    time = None
    if time_reference is not None:
        time_length = {"*": 1024, "+": 1280, "#": height + y_offset}[time_reference]
        time = np.linspace(0.0, 800.0, time_length)

    path = tmp_path / "measurement.img"
    _write_img(
        path,
        data,
        file_type=file_type,
        spectral=spectral,
        time=time,
        spectral_reference=spectral_reference or "#",
        time_reference=time_reference or "#",
        x_offset=x_offset,
        y_offset=y_offset,
    )

    dataset = HamamatsuImgDataIo("img").load_dataset(path.as_posix(), prepare=False)

    assert isinstance(dataset, xr.DataArray)
    assert dataset.shape == shape
    assert dataset.dtype == np.dtype({0: "uint8", 2: "uint16", 3: "uint32"}[file_type])
    assert np.all(np.isfinite(dataset.time))
    assert np.all(np.isfinite(dataset.spectral))
    assert np.all(np.diff(dataset.time) > 0)
    assert np.all(np.diff(dataset.spectral) > 0)
    if spectral is None:
        np.testing.assert_array_equal(dataset.spectral, np.arange(width))
    else:
        np.testing.assert_allclose(dataset.spectral, spectral[x_offset : x_offset + width])
    if time is None:
        np.testing.assert_array_equal(dataset.time, np.arange(height))
    else:
        np.testing.assert_allclose(dataset.time, time[y_offset : y_offset + height])
