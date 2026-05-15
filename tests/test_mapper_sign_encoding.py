"""Tests for the tiler / slicer / mapper trio.

Covers:

* :class:`SerialSlicer` — radix-r decomposition with structural
  ``digit_num = 1`` (activation path).
* :class:`SimpleSlicer` — slice-first-then-digitize, configurable
  slice / digit encodings (weight path).
* :class:`SimpleTiler` — N/K right-pad-and-unflatten with separate
  ``make_w_plan`` / ``make_x_plan`` factories.
* :class:`SimpleMapper` — composes tiler + slicers and produces
  the macro canonical layout via ``map_x`` / ``map_w``.  Every
  runtime method takes the full xbar capability kwarg set; the
  mapper internally translates those caps into the unified
  ``Slicer`` 3-kwarg contract so a concrete slicer is fully
  interchangeable from the mapper's point of view.

Tests pass each argument explicitly (no ``**kwargs`` splats) to
mirror the production contract: every cross-class call documents
its full argument list inline.  ``value_range`` is the slicer's
*output*, never an input.
"""

from __future__ import annotations

import pytest
import torch

from neurox.mapper.transcoder import SignedDigitTranscoder
from neurox.mapper.xbar import (
    SerialSlicer,
    SimpleMapper,
    SimpleSlicer,
    SimpleTiler,
    SlicingResult,
    TilePlan,
    WMappingResult,
    XMappingResult,
)

# --- SignedDigitTranscoder.value_range -----------------------------------


def test_transcoder_value_range_symmetric() -> None:
    t = SignedDigitTranscoder("true_form", radix=2, digit_num=3)
    # r=2, D=3 -> envelope (-(2^3 - 1), 2^3 - 1) = (-7, 7).
    assert t.value_range() == (-7, 7)


# --- SerialSlicer ---------------------------------------------------------


def test_serial_slicer_value_range_and_radix() -> None:
    s = SerialSlicer(slice_num=4, encoding="true_form")
    # digit_range = (0, 1) -> radix = 2 -> Sa=4 -> range = (0, 2^4 - 1) = (0, 15).
    assert s.value_range(digit_count=1, digit_radix=2, digit_range=(0, 1)) == (0, 15)
    assert s.slice_radix(digit_count=1, digit_radix=2, digit_range=(0, 1)) == 2


def test_serial_slicer_rejects_signed_grid() -> None:
    s = SerialSlicer(slice_num=4, encoding="true_form")
    with pytest.raises(ValueError, match="digit_range lo"):
        s.value_range(digit_count=1, digit_radix=3, digit_range=(-1, 1))


def test_serial_slicer_rejects_inner_digit_axis() -> None:
    s = SerialSlicer(slice_num=4, encoding="true_form")
    with pytest.raises(ValueError, match="digit_count"):
        s.value_range(digit_count=2, digit_radix=2, digit_range=(0, 1))


def test_serial_slicer_rejects_mismatched_radix() -> None:
    s = SerialSlicer(slice_num=4, encoding="true_form")
    with pytest.raises(ValueError, match="digit_radix"):
        s.value_range(digit_count=1, digit_radix=4, digit_range=(0, 1))


def test_serial_slicer_slice_num_zero_rejected() -> None:
    with pytest.raises(ValueError, match="slice_num"):
        SerialSlicer(slice_num=0, encoding="true_form")


def test_serial_slicer_output_shape() -> None:
    s = SerialSlicer(slice_num=3, encoding="true_form")
    x = torch.randint(0, 4, (5, 8), dtype=torch.int32)
    out = s.slice(x, digit_count=1, digit_radix=4, digit_range=(0, 3))
    assert isinstance(out, SlicingResult)
    # Trailing-2: [slice_num=3, digit_num=1].
    assert out.values.shape == (5, 8, 3, 1)
    # slice_weights = [r^0, r^1, r^2] = [1, 4, 16].
    assert torch.equal(out.slice_weights.cpu(), torch.tensor([1, 4, 16], dtype=out.values.dtype))
    # digit_weights = [1] (digit_num = 1).
    assert torch.equal(out.digit_weights.cpu(), torch.tensor([1], dtype=out.values.dtype))
    assert out.value_range == (0, 4**3 - 1)


def test_serial_slicer_decode_roundtrip() -> None:
    torch.manual_seed(0)
    s = SerialSlicer(slice_num=3, encoding="true_form")
    x = torch.randint(0, 4, (8, 8), dtype=torch.int32)
    out = s.slice(x, digit_count=1, digit_radix=4, digit_range=(0, 3))
    # Decode = collapse slice axis with positional weights, then squeeze digit_num=1.
    decoded = (out.values.squeeze(-1) * out.slice_weights).sum(dim=-1)
    assert torch.equal(decoded, x)


# --- SimpleSlicer ---------------------------------------------------------


def test_simple_slicer_value_range_and_radix() -> None:
    s = SimpleSlicer(slice_num=1, encoding="true_form")
    # r=2, D=3 -> slice_radix = 2^3 = 8.  Sw=1 -> range = (-(8-1), 8-1) = (-7, 7).
    assert s.value_range(digit_count=3, digit_radix=2, digit_range=(-1, 1)) == (-7, 7)
    assert s.slice_radix(digit_count=3, digit_radix=2, digit_range=(-1, 1)) == 8


def test_simple_slicer_value_range_multi_slice() -> None:
    s2 = SimpleSlicer(slice_num=2, encoding="true_form")
    # slice_radix=8, Sw=2 -> (-8^2 - 1, 8^2 - 1) = (-63, 63).
    assert s2.value_range(digit_count=3, digit_radix=2, digit_range=(-1, 1)) == (-63, 63)
    s3 = SimpleSlicer(slice_num=3, encoding="true_form")
    assert s3.value_range(digit_count=3, digit_radix=2, digit_range=(-1, 1)) == (-511, 511)


def test_simple_slicer_output_shape() -> None:
    s = SimpleSlicer(slice_num=2, encoding="true_form")
    w = torch.randint(-30, 31, (8, 8), dtype=torch.int32)
    out = s.slice(w, digit_count=3, digit_radix=2, digit_range=(-1, 1))
    # Trailing-2: [slice_num=2, digit_num=3].
    assert out.values.shape == (8, 8, 2, 3)
    # slice_weights = [1, slice_radix=8].
    assert torch.equal(out.slice_weights.cpu(), torch.tensor([1, 8], dtype=out.values.dtype))
    # digit_weights = [1, 2, 4].
    assert torch.equal(out.digit_weights.cpu(), torch.tensor([1, 2, 4], dtype=out.values.dtype))


def test_simple_slicer_multi_slice_roundtrip() -> None:
    torch.manual_seed(0)
    s = SimpleSlicer(slice_num=2, encoding="true_form")
    w = torch.randint(-63, 64, (8, 8), dtype=torch.int32)
    out = s.slice(w, digit_count=3, digit_radix=2, digit_range=(-1, 1))
    # Recombine: collapse digit axis, then slice axis.
    per_slice = (out.values * out.digit_weights).sum(dim=-1)
    decoded = (per_slice * out.slice_weights).sum(dim=-1)
    assert torch.equal(decoded, w)


def test_simple_slicer_slice_num_zero_rejected() -> None:
    with pytest.raises(ValueError, match="slice_num"):
        SimpleSlicer(slice_num=0, encoding="true_form")


# --- SimpleTiler ----------------------------------------------------------


def test_simple_tiler_w_plan_no_padding() -> None:
    t = SimpleTiler()
    plan = t.make_w_plan(n=8, k=8, col_num=8, row_num=8)
    assert isinstance(plan, TilePlan)
    assert plan.logical_out_dim == 8
    assert plan.logical_in_dim == 8
    assert plan.row_tile_num == 1
    assert plan.col_tile_num == 1
    assert plan.data_num == 8
    assert plan.row_num == 8
    assert plan.n_pad == 0
    assert plan.k_pad == 0


def test_simple_tiler_w_plan_with_padding() -> None:
    t = SimpleTiler()
    plan = t.make_w_plan(n=10, k=20, col_num=8, row_num=8)
    assert plan.row_tile_num == 2
    assert plan.col_tile_num == 3
    assert plan.n_pad == 6  # 2*8 - 10
    assert plan.k_pad == 4  # 3*8 - 20


def test_simple_tiler_x_plan_only_carries_k_geometry() -> None:
    t = SimpleTiler()
    plan = t.make_x_plan(k=20, row_num=8)
    # K-only geometry is non-zero; N-side fields are placeholders 0.
    assert plan.col_tile_num == 3
    assert plan.row_num == 8
    assert plan.k_pad == 4
    assert plan.logical_in_dim == 20
    assert plan.logical_out_dim == 0
    assert plan.row_tile_num == 0
    assert plan.data_num == 0
    assert plan.n_pad == 0


def test_simple_tiler_w_plan_rejects_zero_n() -> None:
    t = SimpleTiler()
    with pytest.raises(ValueError, match=r"n"):
        t.make_w_plan(n=0, k=8, col_num=8, row_num=8)


def test_simple_tiler_tile_w_shape() -> None:
    t = SimpleTiler()
    plan = t.make_w_plan(n=8, k=8, col_num=8, row_num=8)
    # Input: [N=8, K=8, slice_num=2, digit_num=3].
    w = torch.zeros((8, 8, 2, 3), dtype=torch.int32)
    out = t.tile_w(w, plan=plan)
    # Output: [Tr=1, data_num=8, Tc=1, row_num=8, slice_num=2, digit_num=3].
    assert out.shape == (1, 8, 1, 8, 2, 3)


def test_simple_tiler_tile_x_shape() -> None:
    t = SimpleTiler()
    plan = t.make_x_plan(k=8, row_num=8)
    # Input: [M=5, K=8, slice_num=3, digit_num=1].
    x = torch.zeros((5, 8, 3, 1), dtype=torch.int32)
    out = t.tile_x(x, plan=plan)
    # Output: [M=5, Tc=1, row_num=8, slice_num=3, digit_num=1].
    assert out.shape == (5, 1, 8, 3, 1)


# --- SimpleMapper ---------------------------------------------------------


def _make_mapper(*, x_slice_num: int = 4, w_slice_num: int = 1) -> SimpleMapper:
    return SimpleMapper(
        tiler=SimpleTiler(),
        x_slicer=SerialSlicer(slice_num=x_slice_num, encoding="true_form"),
        w_slicer=SimpleSlicer(slice_num=w_slice_num, encoding="true_form"),
    )


def test_simple_mapper_value_ranges() -> None:
    m = _make_mapper(x_slice_num=4, w_slice_num=1)
    # x: radix=2, Sa=4 -> (0, 15).
    assert m.x_value_range(
        x_range=(0, 1),
        col_num=8,
        row_num=8,
        w_digit_count=3,
        w_digit_radix=2,
        w_digit_range=(-1, 1),
    ) == (0, 15)
    # w: r=2, D=3, Sw=1 -> (-7, 7).
    assert m.w_value_range(
        x_range=(0, 1),
        col_num=8,
        row_num=8,
        w_digit_count=3,
        w_digit_radix=2,
        w_digit_range=(-1, 1),
    ) == (-7, 7)


def test_simple_mapper_slice_radixes() -> None:
    m = _make_mapper(x_slice_num=4, w_slice_num=1)
    assert m.x_slice_radix(
        x_range=(0, 1),
        col_num=8,
        row_num=8,
        w_digit_count=3,
        w_digit_radix=2,
        w_digit_range=(-1, 1),
    ) == 2
    assert m.w_slice_radix(
        x_range=(0, 1),
        col_num=8,
        row_num=8,
        w_digit_count=3,
        w_digit_radix=2,
        w_digit_range=(-1, 1),
    ) == 8


def test_simple_mapper_map_x_shape() -> None:
    m = _make_mapper(x_slice_num=3)
    x = torch.randint(0, 4, (5, 8), dtype=torch.int32)
    out = m.map_x(
        x,
        x_range=(0, 3),
        col_num=8,
        row_num=8,
        w_digit_count=3,
        w_digit_radix=2,
        w_digit_range=(-1, 1),
    )
    assert isinstance(out, XMappingResult)
    # Layout [M, Tc, Tr=1, Sa, Sw=1, row_num].
    M, _Tc, Tr, Sa, Sw, row_num = out.x_xbar.shape
    assert M == 5
    assert Tr == 1
    assert Sa == 3
    assert Sw == 1
    assert row_num == 8
    # slice_weights = [1, 4, 16] (Sa positional weights at radix 4).
    assert torch.equal(out.slice_weights.cpu(), torch.tensor([1, 4, 16], dtype=out.x_xbar.dtype))
    assert out.logical_batch_shape == ()


def test_simple_mapper_map_x_preserves_leading_batch() -> None:
    m = _make_mapper()
    x = torch.randint(0, 2, (2, 3, 5, 8), dtype=torch.int32)
    out = m.map_x(
        x,
        x_range=(0, 1),
        col_num=8,
        row_num=8,
        w_digit_count=3,
        w_digit_radix=2,
        w_digit_range=(-1, 1),
    )
    assert out.logical_batch_shape == (2, 3)


def test_simple_mapper_map_w_shape_single_slice() -> None:
    m = _make_mapper(w_slice_num=1)
    w = torch.randint(-3, 4, (8, 8), dtype=torch.int32)
    out = m.map_w(
        w,
        x_range=(0, 1),
        col_num=8,
        row_num=8,
        w_digit_count=3,
        w_digit_radix=2,
        w_digit_range=(-1, 1),
    )
    assert isinstance(out, WMappingResult)
    # Layout [M=1, Tc, Tr, Sa=1, Sw, data_num, digit_num, row_num].
    M, Tc, Tr, Sa, Sw, data_num, digit_num, row_num = out.w_xbar.shape
    assert M == 1
    assert Tc == 1
    assert Tr == 1
    assert Sa == 1
    assert Sw == 1
    assert data_num == 8
    assert digit_num == 3
    assert row_num == 8
    assert out.slice_weights.shape == (1,)
    assert out.slice_weights.item() == 1
    assert out.logical_out_dim == 8
    assert out.row_tile_num == 1
    assert out.col_tile_num == 1


def test_simple_mapper_map_w_shape_multi_slice() -> None:
    m = _make_mapper(w_slice_num=2)
    w = torch.randint(-30, 31, (8, 8), dtype=torch.int32)
    out = m.map_w(
        w,
        x_range=(0, 1),
        col_num=8,
        row_num=8,
        w_digit_count=3,
        w_digit_radix=2,
        w_digit_range=(-1, 1),
    )
    M, Tc, Tr, Sa, Sw, data_num, digit_num, row_num = out.w_xbar.shape
    assert (M, Tc, Tr, Sa, Sw, data_num, digit_num, row_num) == (1, 1, 1, 1, 2, 8, 3, 8)
    assert torch.equal(out.slice_weights.cpu(), torch.tensor([1, 8], dtype=out.w_xbar.dtype))


def test_simple_mapper_map_w_handles_padding() -> None:
    m = _make_mapper(w_slice_num=1)
    w = torch.randint(-3, 4, (10, 20), dtype=torch.int32)
    out = m.map_w(
        w,
        x_range=(0, 1),
        col_num=8,
        row_num=8,
        w_digit_count=3,
        w_digit_radix=2,
        w_digit_range=(-1, 1),
    )
    _, _, _, _, _, data_num, digit_num, row_num = out.w_xbar.shape
    assert data_num == 8
    assert digit_num == 3
    assert row_num == 8
    assert out.logical_out_dim == 10
    assert out.row_tile_num == 2
    assert out.col_tile_num == 3


def test_simple_mapper_map_w_decode_roundtrip_single_slice() -> None:
    torch.manual_seed(0)
    m = _make_mapper(w_slice_num=1)
    w = torch.randint(-7, 8, (8, 8), dtype=torch.int32)
    out = m.map_w(
        w,
        x_range=(0, 1),
        col_num=8,
        row_num=8,
        w_digit_count=3,
        w_digit_radix=2,
        w_digit_range=(-1, 1),
    )
    # w_xbar layout: [M=1, Tc=1, Tr=1, Sa=1, Sw=1, data_num=8, D=3, row_num=8].
    transcoder = SignedDigitTranscoder("true_form", radix=2, digit_num=3)
    decoded = transcoder.decode(out.w_xbar, dim=-2)
    while decoded.ndim > 2:
        decoded = decoded.squeeze(0)
    assert torch.equal(decoded, w)


def test_simple_mapper_map_w_decode_roundtrip_multi_slice() -> None:
    torch.manual_seed(0)
    m = _make_mapper(w_slice_num=2)
    w = torch.randint(-63, 64, (8, 8), dtype=torch.int32)
    out = m.map_w(
        w,
        x_range=(0, 1),
        col_num=8,
        row_num=8,
        w_digit_count=3,
        w_digit_radix=2,
        w_digit_range=(-1, 1),
    )
    transcoder = SignedDigitTranscoder("true_form", radix=2, digit_num=3)
    per_slice = transcoder.decode(out.w_xbar, dim=-2)
    weights = out.slice_weights.view(-1, 1, 1).to(per_slice.dtype)
    combined = (per_slice * weights).sum(dim=-3)
    while combined.ndim > 2:
        combined = combined.squeeze(0)
    assert torch.equal(combined, w)


def test_simple_mapper_method_ignores_unused_caps() -> None:
    """Vary an unused cap on a method; the result is unchanged.

    Demonstrates that the macro can pass the full kwarg set without
    a concrete mapper accidentally coupling to caps it doesn't need.
    """
    m = _make_mapper(x_slice_num=3, w_slice_num=1)
    a = m.x_value_range(
        x_range=(0, 3),
        col_num=8,
        row_num=8,
        w_digit_count=3,
        w_digit_radix=2,
        w_digit_range=(-1, 1),
    )
    # Vary the weight-side caps + col_num (all irrelevant for x_value_range).
    b = m.x_value_range(
        x_range=(0, 3),
        col_num=4,
        row_num=8,
        w_digit_count=2,
        w_digit_radix=3,
        w_digit_range=(-2, 2),
    )
    assert a == b


# --- IdealMacro (explicit value_range constructor) -----------------------


def test_ideal_macro_publishes_supplied_ranges() -> None:
    from neurox.macro.ideal import IdealMacro

    m = IdealMacro(x_value_range=(-7, 7), w_value_range=(-3, 3))
    assert m.x_value_range == (-7, 7)
    assert m.w_value_range == (-3, 3)
    assert m.output_rescale_factor == 1.0
