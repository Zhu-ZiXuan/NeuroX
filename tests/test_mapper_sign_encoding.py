"""Tests for the value-domain mapper primitives and the xbar macro pipeline."""

from __future__ import annotations

import pytest
import torch

from neurox.digital import (
    AccumulatorConfig,
    RequantizerConfig,
    ShiftAdderConfig,
)
from neurox.macro.xbar import (
    InterXbarSliceMacro,
    InterXbarSliceMacroConfig,
    IntraXbarSliceMacro,
    IntraXbarSliceMacroConfig,
    XbarMacro,
)
from neurox.mapper.transcoder import (
    CanonicalTranscoder,
    ComplementTranscoder,
    Transcoder,
    TrueFormTranscoder,
)
from neurox.mapper.xbar.slicer import (
    SerialSlicer,
    SimpleSlicer,
)
from neurox.xbar import IdealXbar, XbarConfig, XbarRescaleEntry

# --- Transcoder value_range per encoding ---------------------------------


def test_true_form_value_range_symmetric() -> None:
    t = TrueFormTranscoder(radix=2, digit_num=3)
    # r=2, D=3 -> envelope (-(2^3 - 1), 2^3 - 1) = (-7, 7).
    assert t.value_range == (-7, 7)


def test_complement_value_range_asymmetric() -> None:
    # r=2, D=3 -> [-1*4, 1*4 - 1] = [-4, 3].
    assert ComplementTranscoder(radix=2, digit_num=3).value_range == (-4, 3)
    # r=4, D=4 -> [-2*64, 2*64 - 1] = [-128, 127].
    assert ComplementTranscoder(radix=4, digit_num=4).value_range == (-128, 127)
    # r=3, D=2 (odd radix) -> [-1*3, 2*3 - 1] = [-3, 5].
    assert ComplementTranscoder(radix=3, digit_num=2).value_range == (-3, 5)


def test_canonical_value_range_tighter_than_true_form() -> None:
    # r=4, D=4 -> M = 3*64 + 3*4 = 204; true-form bound would be 255.
    assert CanonicalTranscoder(radix=4, digit_num=4).value_range == (-204, 204)
    # r=2, D=3 -> M = 1*4 + 1*1 = 5; true-form bound would be 7.
    assert CanonicalTranscoder(radix=2, digit_num=3).value_range == (-5, 5)


def test_transcoder_create_dispatches_on_encoding() -> None:
    assert isinstance(Transcoder.create("true_form", radix=2, digit_num=3), TrueFormTranscoder)
    assert isinstance(Transcoder.create("complement", radix=2, digit_num=3), ComplementTranscoder)
    assert isinstance(Transcoder.create("canonical", radix=2, digit_num=3), CanonicalTranscoder)


# --- SerialSlicer ---------------------------------------------------------


def test_serial_slicer_value_range_and_radix() -> None:
    s = SerialSlicer(slice_num=4, digit_radix=2)
    # r=2, Sa=4 -> value_range = (0, 2^4 - 1) = (0, 15); slice_radix = 2.
    assert s.value_range == (0, 15)
    assert s.slice_radix == 2


def test_serial_slicer_slice_num_zero_rejected() -> None:
    with pytest.raises(ValueError, match="slice_num"):
        SerialSlicer(slice_num=0, digit_radix=2)


def test_serial_slicer_digit_radix_too_small_rejected() -> None:
    with pytest.raises(ValueError, match="digit_radix"):
        SerialSlicer(slice_num=4, digit_radix=1)


def test_serial_slicer_output_shape() -> None:
    s = SerialSlicer(slice_num=3, digit_radix=4)
    x = torch.randint(0, 4, (5, 8), dtype=torch.int32)
    out = s.slice(x)
    # Trailing-2: [slice_num=3, digit_num=1].
    assert out.shape == (5, 8, 3, 1)
    assert s.value_range == (0, 4**3 - 1)
    assert s.slice_radix == 4
    assert s.slice_weights == (1, 4, 16)


def test_serial_slicer_decode_roundtrip() -> None:
    torch.manual_seed(0)
    s = SerialSlicer(slice_num=3, digit_radix=4)
    x = torch.randint(0, 4, (8, 8), dtype=torch.int32)
    out = s.slice(x)
    weights = torch.tensor(s.slice_weights, dtype=out.dtype)
    decoded = (out.squeeze(-1) * weights).sum(dim=-1)
    assert torch.equal(decoded, x)


# --- SimpleSlicer ---------------------------------------------------------


def test_simple_slicer_value_range_and_radix() -> None:
    s = SimpleSlicer(slice_num=1, digit_count=3, digit_radix=2, encoding="true_form")
    # r=2, D=3 -> slice_radix = 2^3 = 8.  Sw=1 -> value_range = (-(8-1), 8-1) = (-7, 7).
    assert s.value_range == (-7, 7)
    assert s.slice_radix == 8


def test_simple_slicer_value_range_multi_slice() -> None:
    s2 = SimpleSlicer(slice_num=2, digit_count=3, digit_radix=2, encoding="true_form")
    # slice_radix=8, Sw=2 -> (-8^2 - 1, 8^2 - 1) = (-63, 63).
    assert s2.value_range == (-63, 63)
    s3 = SimpleSlicer(slice_num=3, digit_count=3, digit_radix=2, encoding="true_form")
    assert s3.value_range == (-511, 511)


def test_simple_slicer_output_shape() -> None:
    s = SimpleSlicer(slice_num=2, digit_count=3, digit_radix=2, encoding="true_form")
    w = torch.randint(-30, 31, (8, 8), dtype=torch.int32)
    out = s.slice(w)
    # Trailing-2: [slice_num=2, digit_num=3].
    assert out.shape == (8, 8, 2, 3)
    assert s.slice_radix == 8
    assert s.slice_weights == (1, 8)


def test_simple_slicer_multi_slice_roundtrip() -> None:
    torch.manual_seed(0)
    s = SimpleSlicer(slice_num=2, digit_count=3, digit_radix=2, encoding="true_form")
    w = torch.randint(-63, 64, (8, 8), dtype=torch.int32)
    out = s.slice(w)
    # Inner-digit weights stay test-local (no slicer surface for them);
    # outer slice weights come from the slicer.
    digit_weights = torch.tensor([2**i for i in range(out.shape[-1])], dtype=out.dtype)
    slice_weights = torch.tensor(s.slice_weights, dtype=out.dtype)
    per_slice = (out * digit_weights).sum(dim=-1)
    decoded = (per_slice * slice_weights).sum(dim=-1)
    assert torch.equal(decoded, w)


def test_simple_slicer_slice_num_zero_rejected() -> None:
    with pytest.raises(ValueError, match="slice_num"):
        SimpleSlicer(slice_num=0, digit_count=3, digit_radix=2, encoding="true_form")


def test_simple_slicer_digit_count_zero_rejected() -> None:
    with pytest.raises(ValueError, match="digit_count"):
        SimpleSlicer(slice_num=1, digit_count=0, digit_radix=2, encoding="true_form")


def test_simple_slicer_digit_radix_too_small_rejected() -> None:
    with pytest.raises(ValueError, match="digit_radix"):
        SimpleSlicer(slice_num=1, digit_count=3, digit_radix=1, encoding="true_form")


# --- chunk_pad_along (geometric primitive) -------------------------------


def test_chunk_pad_no_padding() -> None:
    t = torch.arange(16.0)
    out = XbarMacro.chunk_pad_along(t, axis=0, chunk_size=4)
    # n=16, chunk=4 -> (4, 4).
    assert out.shape == (4, 4)
    assert torch.equal(out.flatten(), t)


def test_chunk_pad_with_padding() -> None:
    t = torch.arange(13.0)
    out = XbarMacro.chunk_pad_along(t, axis=0, chunk_size=16)
    # n=13, chunk=16 -> (1, 16) with 3 trailing zeros.
    assert out.shape == (1, 16)
    assert torch.equal(out[0, :13], t)
    assert torch.equal(out[0, 13:], torch.zeros(3))


def test_chunk_pad_negative_axis() -> None:
    t = torch.arange(60.0).reshape(3, 4, 5)
    out = XbarMacro.chunk_pad_along(t, axis=-1, chunk_size=3)
    # axis=-1, size=5 pads to 6, then (2, 3) inserted at axis=-1.
    assert out.shape == (3, 4, 2, 3)


def test_chunk_pad_rejects_bad_chunk_size() -> None:
    t = torch.arange(8.0)
    with pytest.raises(ValueError, match="chunk_size"):
        XbarMacro.chunk_pad_along(t, axis=0, chunk_size=0)


def test_chunk_pad_rejects_out_of_range_axis() -> None:
    t = torch.arange(8.0)
    with pytest.raises(ValueError, match="axis"):
        XbarMacro.chunk_pad_along(t, axis=3, chunk_size=4)


# --- End-to-end mode consistency (13 / 3 / 16 case) ----------------------

# 13 logical weights, Sw=3 slices, col_num=16 — the canonical case from
# the architecture discussion.  IntraXbar packs 5 weights × 3 slices per
# xbar with 1 idle col; InterXbar uses one Sw plane per xbar with 3 idle
# cols.  Both must reproduce torch.matmul exactly.


@pytest.fixture
def small_ideal_xbar() -> IdealXbar:
    cfg = XbarConfig(
        col_num=16, row_num=16, adc_mode=0, adc_bits=0,
        output_rescale_factors=(XbarRescaleEntry(adc_mode=0, adc_bits=0, rf=1.0),),
    )
    xbar = IdealXbar(
        cfg, x_range=(0, 1),
        w_digit_count=1, w_digit_radix=4, w_digit_range=(-3, 3),
    )
    xbar.eval()
    return xbar


def _make_macro_configs(*, w_slice_num: int, x_slice_num: int) -> dict[str, object]:
    return dict(
        w_slice_num=w_slice_num, x_slice_num=x_slice_num,
        w_encoding="true_form",
        col_accumulator_cfg=AccumulatorConfig(bit_width=32),
        sa_shift_adder_cfg=ShiftAdderConfig(bit_width=32),
        sw_shift_adder_cfg=ShiftAdderConfig(bit_width=32),
        requantizer_cfg=RequantizerConfig(bit_width=32),
    )


def test_inter_xbar_matches_torch_matmul(small_ideal_xbar: IdealXbar) -> None:
    torch.manual_seed(0)
    macro = InterXbarSliceMacro(
        cfg=InterXbarSliceMacroConfig(**_make_macro_configs(w_slice_num=3, x_slice_num=4)),
        xbar=small_ideal_xbar,
    )
    macro.eval()
    n, k, m = 13, 20, 8
    w = torch.randint(-63, 64, (n, k), dtype=torch.int32)
    x = torch.randint(0, 16, (m, k), dtype=torch.int32)
    macro.fabricate(w)
    mult = torch.ones(n, dtype=torch.int32)
    shift = torch.zeros(n, dtype=torch.int32)
    y = macro.matmul(x, w, None, mult, shift, None)
    y_ref = (x.float() @ w.float().T).to(torch.int32)
    assert torch.equal(y, y_ref)


def test_intra_xbar_matches_torch_matmul(small_ideal_xbar: IdealXbar) -> None:
    torch.manual_seed(0)
    macro = IntraXbarSliceMacro(
        cfg=IntraXbarSliceMacroConfig(**_make_macro_configs(w_slice_num=3, x_slice_num=4)),
        xbar=small_ideal_xbar,
    )
    macro.eval()
    # 5 weights per xbar, 1 idle col, 3 Tr tiles for N=13.
    assert macro._weights_per_xbar == 5
    assert macro._used_data_num == 15
    assert macro._idle_per_xbar == 1
    n, k, m = 13, 20, 8
    w = torch.randint(-63, 64, (n, k), dtype=torch.int32)
    x = torch.randint(0, 16, (m, k), dtype=torch.int32)
    macro.fabricate(w)
    mult = torch.ones(n, dtype=torch.int32)
    shift = torch.zeros(n, dtype=torch.int32)
    y = macro.matmul(x, w, None, mult, shift, None)
    y_ref = (x.float() @ w.float().T).to(torch.int32)
    assert torch.equal(y, y_ref)


def test_inter_and_intra_xbar_agree(small_ideal_xbar: IdealXbar) -> None:
    """Cross-mode consistency: same w/x, same int output."""
    torch.manual_seed(0)
    cfg_kwargs = _make_macro_configs(w_slice_num=3, x_slice_num=4)
    inter = InterXbarSliceMacro(
        cfg=InterXbarSliceMacroConfig(**cfg_kwargs),
        xbar=small_ideal_xbar,
    )
    inter.eval()
    intra = IntraXbarSliceMacro(
        cfg=IntraXbarSliceMacroConfig(**cfg_kwargs),
        # Fresh xbar so intra's fabricate does not stomp inter's.
        xbar=IdealXbar(
            XbarConfig(
                col_num=16, row_num=16, adc_mode=0, adc_bits=0,
                output_rescale_factors=(XbarRescaleEntry(adc_mode=0, adc_bits=0, rf=1.0),),
            ),
            x_range=(0, 1),
            w_digit_count=1, w_digit_radix=4, w_digit_range=(-3, 3),
        ),
    )
    intra.eval()

    n, k, m = 13, 20, 8
    w = torch.randint(-63, 64, (n, k), dtype=torch.int32)
    x = torch.randint(0, 16, (m, k), dtype=torch.int32)
    inter.fabricate(w)
    intra.fabricate(w)
    mult = torch.ones(n, dtype=torch.int32)
    shift = torch.zeros(n, dtype=torch.int32)
    y_inter = inter.matmul(x, w, None, mult, shift, None)
    y_intra = intra.matmul(x, w, None, mult, shift, None)
    assert torch.equal(y_inter, y_intra)
