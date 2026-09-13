"""Tests for macro value-domain slicers."""

from __future__ import annotations

import pytest
import torch

from neurox.architecture.unit.cim.slicer import SimpleSlicer
from neurox.encoding import Encoding


def _decode_simple_slices(slices: torch.Tensor, *, weights: tuple[int, ...]) -> torch.Tensor:
    slice_weights = torch.tensor(weights, dtype=slices.dtype, device=slices.device)
    return (slices * slice_weights).sum(dim=-1)


@pytest.mark.parametrize(
    ("encoding", "slice_num", "slice_value_range"),
    [
        (Encoding.TRUE_FORM, 3, (-3, 3)),
        (Encoding.COMPLEMENT, 3, (-4, 3)),
        (Encoding.CANONICAL, 2, (-7, 7)),
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
