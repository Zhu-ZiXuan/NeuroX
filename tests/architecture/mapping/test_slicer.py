"""Tests for macro value-domain slicers."""

from __future__ import annotations

import pytest
import torch

from neurox.architecture.mapping.slicer import DirectSlicer, SimpleSlicer
from neurox.encoding import Encoding


@pytest.mark.parametrize(
    ("encoding", "slice_num", "slice_value_range", "expected_radix"),
    [
        (Encoding.UNSIGNED, 3, (0, 7), 8),
        (Encoding.UNSIGNED, 2, (-3, 3), 4),
        (Encoding.TRUE_FORM, 3, (-3, 3), 4),
        (Encoding.COMPLEMENT, 3, (-3, 3), 4),
        (Encoding.CANONICAL, 2, (-7, 7), 8),
        (Encoding.UNSIGNED, 2, (-4, 3), 4),
        (Encoding.TRUE_FORM, 2, (-4, 3), 4),
        (Encoding.COMPLEMENT, 3, (-4, 3), 4),
        (Encoding.CANONICAL, 2, (-4, 3), 4),
        (Encoding.COMPLEMENT, 1, (-4, 3), 8),
        (Encoding.COMPLEMENT, 1, (-3, 3), 7),
        (Encoding.COMPLEMENT, 1, (-1, 0), 2),
        (Encoding.UNSIGNED, 2, (-2, 7), 8),
        (Encoding.TRUE_FORM, 2, (-2, 7), 3),
        (Encoding.CANONICAL, 2, (-2, 7), 3),
        (Encoding.COMPLEMENT, 2, (-2, 7), 5),
        (Encoding.UNSIGNED, 2, (-7, 2), 3),
        (Encoding.TRUE_FORM, 2, (-7, 2), 3),
        (Encoding.CANONICAL, 2, (-7, 2), 3),
        (Encoding.COMPLEMENT, 2, (-7, 2), 3),
        (Encoding.COMPLEMENT, 1, (-7, 2), 6),
        (Encoding.COMPLEMENT, 1, (-7, 0), 2),
    ],
)
def test_simple_slicer_roundtrip_for_encoding_and_geometry_cases(
    encoding: Encoding,
    slice_num: int,
    slice_value_range: tuple[int, int],
    expected_radix: int,
    device: torch.device,
) -> None:
    slicer = SimpleSlicer(
        slice_num=slice_num,
        slice_value_range=slice_value_range,
        encoding=encoding,
    )
    assert slicer.slice_radix == expected_radix
    lo, hi = slicer.value_range
    values = torch.arange(lo, hi + 1, dtype=torch.int32, device=device)
    sliced = slicer.slice(values)
    assert sliced.shape == (hi - lo + 1, slice_num)
    assert sliced.min() >= slice_value_range[0]
    assert sliced.max() <= slice_value_range[1]
    decoded = slicer.recover(sliced, dim=-1)
    assert torch.equal(decoded, values)
    outside = torch.tensor([lo - 1, hi + 1, lo - 100, hi + 100], dtype=values.dtype, device=device)
    outside_sliced = slicer.slice(outside)
    assert outside_sliced.min() >= slice_value_range[0]
    assert outside_sliced.max() <= slice_value_range[1]


@pytest.mark.parametrize("dim", [0, 1, -1])
@pytest.mark.parametrize("direct", [False, True])
def test_recover_recombines_linear_results(dim: int, direct: bool, device: torch.device) -> None:
    slicer = (
        DirectSlicer(value_range=(-7, 7))
        if direct
        else SimpleSlicer(slice_num=3, slice_value_range=(-3, 3), encoding=Encoding.TRUE_FORM)
    )
    values = torch.tensor([[-5, 2, 7], [1, -3, 4]], dtype=torch.int32, device=device)
    # Shape: [vector, input, slice] -> [vector, slice]
    partial = slicer.slice(values).sum(dim=-2, dtype=values.dtype)
    # Results may exceed the range of a single slice after a linear operation.
    partial = partial.unsqueeze(-1).movedim(1, dim)
    actual = slicer.recover(partial, dim=dim)
    torch.testing.assert_close(actual, values.sum(dim=-1, keepdim=True, dtype=values.dtype))
