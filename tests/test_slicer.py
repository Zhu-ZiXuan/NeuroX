"""Tests for xbar value-domain slicers."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from neurox.common.encoding import Encoding, Transcoder
from neurox.macro.xbar.slicer import SerialSlicer, SimpleSlicer


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


def test_serial_slicer_shape_and_roundtrip() -> None:
    slicer = SerialSlicer(slice_num=3, digit_radix=4)
    x = torch.arange(0, 4**3, dtype=torch.int32).reshape(8, 8)
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
def test_serial_slicer_roundtrip_for_geometry_cases(slice_num: int, digit_radix: int) -> None:
    slicer = SerialSlicer(slice_num=slice_num, digit_radix=digit_radix)
    lo, hi = slicer.value_range
    x = torch.arange(lo, hi + 1, dtype=torch.int32)
    sliced = slicer.slice(x)
    assert sliced.shape == (hi - lo + 1, slice_num, 1)
    decoded = _decode_serial_slices(sliced, slicer.slice_weights)
    assert torch.equal(decoded, x)


@pytest.mark.parametrize(
    "encoding",
    ["true_form", "complement", "canonical"],
)
def test_simple_slicer_value_range_delegates_to_full_length_transcoder(encoding: Encoding) -> None:
    slicer = SimpleSlicer(slice_num=2, digit_count=3, digit_radix=2, encoding=encoding)
    transcoder = Transcoder.create(encoding, radix=2, digit_count=6)
    assert slicer.value_range == transcoder.value_range


def test_simple_slicer_contract() -> None:
    slicer = SimpleSlicer(slice_num=2, digit_count=3, digit_radix=2, encoding="true_form")
    assert slicer.value_range == (-63, 63)
    assert slicer.slice_radix == 8
    assert slicer.slice_weights == (1, 8)


def test_simple_slicer_shape_and_roundtrip() -> None:
    slicer = SimpleSlicer(slice_num=2, digit_count=3, digit_radix=2, encoding="true_form")
    lo, hi = slicer.value_range
    w = torch.arange(lo, hi + 1, dtype=torch.int32)
    sliced = slicer.slice(w)
    assert sliced.shape == (hi - lo + 1, 2, 3)
    decoded = _decode_simple_slices(sliced, digit_radix=2, weights=slicer.slice_weights)
    assert torch.equal(decoded, w)


@pytest.mark.parametrize(
    ("encoding", "slice_num", "digit_count", "digit_radix"),
    [
        ("true_form", 1, 1, 4),
        ("true_form", 3, 1, 4),
        ("true_form", 2, 3, 2),
        ("complement", 2, 3, 2),
        ("complement", 3, 2, 3),
        ("canonical", 2, 3, 2),
        ("canonical", 2, 2, 4),
    ],
)
def test_simple_slicer_roundtrip_for_encoding_and_geometry_cases(
    encoding: Encoding,
    slice_num: int,
    digit_count: int,
    digit_radix: int,
) -> None:
    slicer = SimpleSlicer(
        slice_num=slice_num,
        digit_count=digit_count,
        digit_radix=digit_radix,
        encoding=encoding,
    )
    lo, hi = slicer.value_range
    values = torch.arange(lo, hi + 1, dtype=torch.int32)
    sliced = slicer.slice(values)
    assert sliced.shape == (hi - lo + 1, slice_num, digit_count)
    decoded = _decode_simple_slices(sliced, digit_radix=digit_radix, weights=slicer.slice_weights)
    assert torch.equal(decoded, values)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"slice_num": 0, "digit_count": 3, "digit_radix": 2, "encoding": "true_form"},
        {"slice_num": 1, "digit_count": 0, "digit_radix": 2, "encoding": "true_form"},
        {"slice_num": 1, "digit_count": 3, "digit_radix": 1, "encoding": "true_form"},
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
