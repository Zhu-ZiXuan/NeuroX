"""Tests for xbar value-domain slicers."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from neurox.architecture.unit.cim.slicer import DirectSlicer, SerialSlicer, SimpleSlicer
from neurox.common.encoding import Encoding, create_transcoder


def _decode_serial_slices(slices: torch.Tensor, weights: tuple[int, ...]) -> torch.Tensor:
    slice_weights = torch.tensor(weights, dtype=slices.dtype, device=slices.device)
    return (slices * slice_weights).sum(dim=-1)


def _decode_simple_slices(slices: torch.Tensor, *, weights: tuple[int, ...]) -> torch.Tensor:
    slice_weights = torch.tensor(weights, dtype=slices.dtype, device=slices.device)
    return (slices * slice_weights).sum(dim=-1)


def test_serial_slicer_contract() -> None:
    slicer = SerialSlicer(slice_num=3, digit_radix=4)
    assert slicer.value_range == (0, 4**3 - 1)
    assert slicer.slice_radix == 4
    assert slicer.slice_weights == (1, 4, 16)


def test_serial_slicer_shape_and_roundtrip(device: torch.device) -> None:
    slicer = SerialSlicer(slice_num=3, digit_radix=4)
    x = torch.arange(0, 4**3, dtype=torch.int32, device=device).reshape(8, 8)
    sliced = slicer.slice(x)
    assert sliced.shape == (8, 8, 3)
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
    assert sliced.shape == (hi - lo + 1, slice_num)
    decoded = _decode_serial_slices(sliced, slicer.slice_weights)
    assert torch.equal(decoded, x)


@pytest.mark.parametrize(
    "encoding",
    [Encoding.TRUE_FORM, Encoding.COMPLEMENT, Encoding.CANONICAL],
)
def test_simple_slicer_value_range_delegates_to_full_length_transcoder(encoding: Encoding) -> None:
    slicer = SimpleSlicer(
        slice_num=2,
        slice_value_range=(-7, 7),
        encoding=encoding,
    )
    transcoder = create_transcoder(encoding=encoding, radix=8, digit_count=2)
    assert slicer.value_range == transcoder.value_range


def test_simple_slicer_contract() -> None:
    slicer = SimpleSlicer(
        slice_num=2,
        slice_value_range=(-7, 7),
        encoding=Encoding.TRUE_FORM,
    )
    assert slicer.value_range == (-63, 63)
    assert slicer.slice_radix == 8
    assert slicer.slice_weights == (1, 8)


def test_simple_slicer_shape_and_roundtrip(device: torch.device) -> None:
    slicer = SimpleSlicer(
        slice_num=2,
        slice_value_range=(-7, 7),
        encoding=Encoding.TRUE_FORM,
    )
    lo, hi = slicer.value_range
    w = torch.arange(lo, hi + 1, dtype=torch.int32, device=device)
    sliced = slicer.slice(w)
    assert sliced.shape == (hi - lo + 1, 2)
    decoded = _decode_simple_slices(sliced, weights=slicer.slice_weights)
    assert torch.equal(decoded, w)


@pytest.mark.parametrize(
    ("encoding", "slice_num", "slice_value_range"),
    [
        (Encoding.TRUE_FORM, 1, (-3, 3)),
        (Encoding.TRUE_FORM, 3, (-3, 3)),
        (Encoding.TRUE_FORM, 2, (-7, 7)),
        (Encoding.COMPLEMENT, 2, (-7, 7)),
        (Encoding.COMPLEMENT, 3, (-4, 3)),
        (Encoding.CANONICAL, 2, (-7, 7)),
        (Encoding.CANONICAL, 2, (-15, 15)),
    ],
)
def test_simple_slicer_roundtrip_for_encoding_and_geometry_cases(
    encoding: Encoding,
    slice_num: int,
    slice_value_range: tuple[int, int],
    device: torch.device,
) -> None:
    slicer = SimpleSlicer(
        slice_num=slice_num,
        slice_value_range=slice_value_range,
        encoding=encoding,
    )
    lo, hi = slicer.value_range
    values = torch.arange(lo, hi + 1, dtype=torch.int32, device=device)
    sliced = slicer.slice(values)
    assert sliced.shape == (hi - lo + 1, slice_num)
    decoded = _decode_simple_slices(sliced, weights=slicer.slice_weights)
    assert torch.equal(decoded, values)


def test_simple_slicer_unsigned_macro_range(device: torch.device) -> None:
    slicer = SimpleSlicer(
        slice_num=3,
        slice_value_range=(0, 7),
        encoding=Encoding.TRUE_FORM,
    )
    assert slicer.value_range == (0, 8**3 - 1)
    values = torch.arange(0, 8**3, dtype=torch.int32, device=device)
    sliced = slicer.slice(values)
    assert sliced.min().item() == 0
    assert sliced.max().item() == 7
    assert torch.equal(
        _decode_simple_slices(sliced, weights=slicer.slice_weights),
        values,
    )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (
            {"slice_num": 0, "slice_value_range": (-3, 3), "encoding": Encoding.TRUE_FORM},
            r"require: slice_num \(0\) >= 1",
        ),
        (
            {"slice_num": 1, "slice_value_range": (0, 0), "encoding": Encoding.TRUE_FORM},
            r"require: slice_value_range upper bound \(0\) >= 1",
        ),
        (
            {"slice_num": 1, "slice_value_range": (-2, 3), "encoding": Encoding.TRUE_FORM},
            r"cannot represent slice_value_range \(-2, 3\)",
        ),
        (
            {"slice_num": 1, "slice_value_range": (0, 3), "encoding": Encoding.COMPLEMENT},
            "an unsigned slice_value_range requires true-form encoding",
        ),
    ],
)
def test_simple_slicer_rejects_invalid_geometry(kwargs: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        SimpleSlicer(**kwargs)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"slice_num": 0, "digit_radix": 2}, r"require: slice_num \(0\) >= 1"),
        ({"slice_num": 1, "digit_radix": 1}, r"require: digit_radix \(1\) >= 2"),
    ],
)
def test_serial_slicer_rejects_invalid_geometry(kwargs: dict[str, int], match: str) -> None:
    with pytest.raises(ValueError, match=match):
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
    assert sliced.shape == (4, 4, 1)
    assert sliced.dtype == dtype
    assert torch.equal(sliced.squeeze(-1), x)


@pytest.mark.parametrize("value_range", [(-8, 7), (0, 15)])
def test_direct_slicer_weighted_sum_reconstruction(value_range: tuple[int, int], device: torch.device) -> None:
    slicer = DirectSlicer(value_range=value_range)
    lo, hi = value_range
    x = torch.arange(lo, hi + 1, dtype=torch.int32, device=device).reshape(2, 2, 4)
    sliced = slicer.slice(x)
    weights = torch.tensor(slicer.slice_weights, dtype=sliced.dtype, device=device)
    reconstructed = (sliced * weights).sum(dim=-1)
    assert torch.equal(reconstructed, x)


@pytest.mark.parametrize("value_range", [(7, -8), (0, 0), (5, 5)])
def test_direct_slicer_rejects_invalid_value_range(value_range: tuple[int, int]) -> None:
    with pytest.raises(ValueError, match="require: value_range lo"):
        DirectSlicer(value_range=value_range)


@pytest.mark.parametrize("value", [-9, 8])
def test_direct_slicer_does_not_scan_runtime_values(value: int, device: torch.device) -> None:
    slicer = DirectSlicer(value_range=(-8, 7))
    x = torch.tensor([0, value], dtype=torch.int32, device=device)
    sliced = slicer.slice(x)
    assert torch.equal(sliced[..., 0], x)
