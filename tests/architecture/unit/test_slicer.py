"""Tests for xbar value-domain slicers."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from neurox.architecture.unit.cim.slicer import DirectSlicer, SerialSlicer, SimpleSlicer
from neurox.common.encoding import Encoding, create_transcoder


def _decode_serial_slices(slices: torch.Tensor, weights: tuple[int, ...]) -> torch.Tensor:
    slice_weights = torch.tensor(weights, dtype=slices.dtype, device=slices.device)
    return (slices.squeeze(-1) * slice_weights).sum(dim=-1)


def _decode_simple_slices(slices: torch.Tensor, *, digit_radix: int, weights: tuple[int, ...]) -> torch.Tensor:
    digit_weights = torch.tensor(
        [digit_radix**i for i in range(slices.shape[-1])],
        dtype=slices.dtype,
        device=slices.device,
    )
    slice_weights = torch.tensor(weights, dtype=slices.dtype, device=slices.device)
    per_slice = (slices * digit_weights).sum(dim=-1)
    return (per_slice * slice_weights).sum(dim=-1)


def test_serial_slicer_contract() -> None:
    slicer = SerialSlicer(slice_num=3, digit_radix=4)
    assert slicer.value_range == (0, 4**3 - 1)
    assert slicer.slice_radix == 4
    assert slicer.slice_weights == (1, 4, 16)


def test_serial_slicer_shape_and_roundtrip(device: torch.device) -> None:
    slicer = SerialSlicer(slice_num=3, digit_radix=4)
    x = torch.arange(0, 4**3, dtype=torch.int32, device=device).reshape(8, 8)
    sliced = slicer.slice(x)
    assert sliced.shape == (8, 8, 3, 1)
    decoded = _decode_serial_slices(sliced, slicer.slice_weights)
    assert torch.equal(decoded, x)


@pytest.mark.parametrize(
    ("slice_num", "digit_radix"),
    [
        (1, 2),
        (2, 3),
        (4, 2),
    ],
)
def test_serial_slicer_roundtrip_for_geometry_cases(slice_num: int, digit_radix: int, device: torch.device) -> None:
    slicer = SerialSlicer(slice_num=slice_num, digit_radix=digit_radix)
    lo, hi = slicer.value_range
    x = torch.arange(lo, hi + 1, dtype=torch.int32, device=device)
    sliced = slicer.slice(x)
    assert sliced.shape == (hi - lo + 1, slice_num, 1)
    decoded = _decode_serial_slices(sliced, slicer.slice_weights)
    assert torch.equal(decoded, x)


@pytest.mark.parametrize(
    "encoding",
    [Encoding.TRUE_FORM, Encoding.COMPLEMENT, Encoding.CANONICAL],
)
def test_simple_slicer_value_range_delegates_to_full_length_transcoder(encoding: Encoding) -> None:
    slicer = SimpleSlicer(slice_num=2, digit_count=3, digit_radix=2, encoding=encoding)
    transcoder = create_transcoder(encoding=encoding, radix=2, digit_count=6)
    assert slicer.value_range == transcoder.value_range


def test_simple_slicer_contract() -> None:
    slicer = SimpleSlicer(slice_num=2, digit_count=3, digit_radix=2, encoding=Encoding.TRUE_FORM)
    assert slicer.value_range == (-63, 63)
    assert slicer.slice_radix == 8
    assert slicer.slice_weights == (1, 8)


def test_simple_slicer_shape_and_roundtrip(device: torch.device) -> None:
    slicer = SimpleSlicer(slice_num=2, digit_count=3, digit_radix=2, encoding=Encoding.TRUE_FORM)
    lo, hi = slicer.value_range
    w = torch.arange(lo, hi + 1, dtype=torch.int32, device=device)
    sliced = slicer.slice(w)
    assert sliced.shape == (hi - lo + 1, 2, 3)
    decoded = _decode_simple_slices(sliced, digit_radix=2, weights=slicer.slice_weights)
    assert torch.equal(decoded, w)


@pytest.mark.parametrize(
    ("encoding", "slice_num", "digit_count", "digit_radix"),
    [
        (Encoding.TRUE_FORM, 1, 1, 4),
        (Encoding.TRUE_FORM, 3, 1, 4),
        (Encoding.TRUE_FORM, 2, 3, 2),
        (Encoding.COMPLEMENT, 2, 3, 2),
        (Encoding.COMPLEMENT, 3, 2, 3),
        (Encoding.CANONICAL, 2, 3, 2),
        (Encoding.CANONICAL, 2, 2, 4),
    ],
)
def test_simple_slicer_roundtrip_for_encoding_and_geometry_cases(
    encoding: Encoding,
    slice_num: int,
    digit_count: int,
    digit_radix: int,
    device: torch.device,
) -> None:
    slicer = SimpleSlicer(
        slice_num=slice_num,
        digit_count=digit_count,
        digit_radix=digit_radix,
        encoding=encoding,
    )
    lo, hi = slicer.value_range
    values = torch.arange(lo, hi + 1, dtype=torch.int32, device=device)
    sliced = slicer.slice(values)
    assert sliced.shape == (hi - lo + 1, slice_num, digit_count)
    decoded = _decode_simple_slices(sliced, digit_radix=digit_radix, weights=slicer.slice_weights)
    assert torch.equal(decoded, values)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"slice_num": 0, "digit_count": 3, "digit_radix": 2, "encoding": Encoding.TRUE_FORM},
        {"slice_num": 1, "digit_count": 0, "digit_radix": 2, "encoding": Encoding.TRUE_FORM},
        {"slice_num": 1, "digit_count": 3, "digit_radix": 1, "encoding": Encoding.TRUE_FORM},
    ],
)
def test_simple_slicer_rejects_invalid_geometry(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        SimpleSlicer(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"slice_num": 0, "digit_radix": 2},
        {"slice_num": 1, "digit_radix": 1},
    ],
)
def test_serial_slicer_rejects_invalid_geometry(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        SerialSlicer(**kwargs)


@pytest.mark.parametrize("value_range", [(-8, 7), (0, 15)])
def test_direct_slicer_contract(value_range: tuple[int, int]) -> None:
    slicer = DirectSlicer(value_range=value_range)
    lo, hi = value_range
    assert slicer.value_range == value_range
    assert slicer.slice_radix == hi - lo + 1
    assert slicer.slice_weights == (1,)


@pytest.mark.parametrize("dtype", [torch.int8, torch.int32, torch.int64])
def test_direct_slicer_slice_is_identity_with_structural_axes(dtype: torch.dtype, device: torch.device) -> None:
    slicer = DirectSlicer(value_range=(-8, 7))
    x = torch.arange(-8, 8, dtype=dtype, device=device).reshape(4, 4)
    sliced = slicer.slice(x)
    assert sliced.shape == (4, 4, 1, 1)
    assert sliced.dtype == dtype
    assert torch.equal(sliced.squeeze(-1).squeeze(-1), x)


@pytest.mark.parametrize("value_range", [(-8, 7), (0, 15)])
def test_direct_slicer_weighted_sum_reconstruction(value_range: tuple[int, int], device: torch.device) -> None:
    slicer = DirectSlicer(value_range=value_range)
    lo, hi = value_range
    x = torch.arange(lo, hi + 1, dtype=torch.int32, device=device).reshape(2, 2, 4)
    sliced = slicer.slice(x)
    weights = torch.tensor(slicer.slice_weights, dtype=sliced.dtype, device=device)
    reconstructed = (sliced.squeeze(-1) * weights).sum(dim=-1)
    assert torch.equal(reconstructed, x)


@pytest.mark.parametrize("value_range", [(7, -8), (0, 0), (5, 5)])
def test_direct_slicer_rejects_invalid_value_range(value_range: tuple[int, int]) -> None:
    with pytest.raises(ValueError):
        DirectSlicer(value_range=value_range)


@pytest.mark.parametrize("value", [-9, 8])
def test_direct_slicer_does_not_scan_runtime_values(value: int, device: torch.device) -> None:
    slicer = DirectSlicer(value_range=(-8, 7))
    x = torch.tensor([0, value], dtype=torch.int32, device=device)
    sliced = slicer.slice(x)
    assert torch.equal(sliced[..., 0, 0], x)
