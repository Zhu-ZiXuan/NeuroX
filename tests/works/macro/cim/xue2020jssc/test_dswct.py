"""CPU-only law tests for the scheme-local DSWCT place-value combiner.

Covers `neurox.works.macro.cim.xue2020jssc.dswct.Dswct` standalone on a
tiny hand-built witness config (no macro, no solve):

  * shape law — `inst_count` derives from the `(gn, 2)` fabrication shape
    (one bank per (IO, polarity)) and the static area / leakage seats scale
    with it; the forward output drops exactly the `w_digit` axis,
  * value law — the forward output equals the same computation done inline:
    LSB-first digit-ratio weighting then sum over `w_digit`; no profiler
    required and no events emitted outside one,
  * billing law — the recorded dynamic energy equals the hand-computed formula
    on a tiny witness: rail `VDD * |I_WDL leg| * window` summed over every
    (plane, slot, lane, digit) leg — the per-bit DIAGONAL window rides the
    leading batch — plus the `c_load * VDD**2` cap event per (slot x plane)
    per bank; negative leg currents bill by `|I|`,
  * construction / call guards — a polarity axis that is not 2, a non-1-D
    ratio buffer, and a trailing shape mismatch all raise.

All witness numbers are arbitrary small values; the assertions constrain how
the output and the billed energy MOVE, not what they are.
"""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from neurox import Profiler, Reporter, stamp_names
from neurox.works.macro.cim.xue2020jssc.dswct import Dswct, DswctConfig, DswctPolicy

# --- Tiny witness geometry ---
_GN = 2  # CIM-IO count
_POL = 2  # polarity pair
_W_DIGIT = 2  # place-value legs per bank
_SERIAL = 3  # column-MUX slots
_DTYPE = torch.float64

_VDD__V = 1.2  # non-unity so a dropped rail factor is caught
_DIGIT_RATIOS = (0.25, 0.5)  # LSB-first (MSB anchor 0.5, radix 2)


def _build_config(*, c_load__fF: float = 0.0) -> DswctConfig:
    return DswctConfig(
        area_per_inst__um2=2.0,
        leakage_per_inst__uW=3.0,
        c_load__fF=c_load__fF,
    )


def _build_dswct(
    *,
    c_load__fF: float = 0.0,
    inst_shape: tuple[int, ...] = (_GN, _POL),
    digit_ratios: tuple[float, ...] = _DIGIT_RATIOS,
) -> Dswct:
    dswct = Dswct(
        config=_build_config(c_load__fF=c_load__fF),
        policy=DswctPolicy(),
        inst_shape=inst_shape,
        digit_ratios=torch.tensor(digit_ratios, dtype=_DTYPE),
        vdd__V=_VDD__V,
    )
    dswct.eval()
    dswct.fabricate()
    stamp_names(dswct)  # the standalone module is its own root, named ""
    return dswct


def _i_dl(*leading: int) -> Tensor:
    """Seeded signed witness currents."""
    gen = torch.Generator().manual_seed(0)
    return torch.randn((*leading, _SERIAL, _GN, _POL, _W_DIGIT), dtype=_DTYPE, generator=gen)


def _inline_i_wdl(i_dl__uA: Tensor, digit_ratios: tuple[float, ...] = _DIGIT_RATIOS) -> Tensor:
    """The forward computation done inline: digit-ratio weighting + digit sum."""
    ratios = torch.tensor(digit_ratios, dtype=i_dl__uA.dtype)
    return (i_dl__uA * ratios).sum(dim=-1)


# ---------------------------------------------------------------------------
# Shape law
# ---------------------------------------------------------------------------


def test_inst_count_and_static_seats_derive_from_config() -> None:
    """One bank per (IO, polarity): inst_count = gn * 2 and statics scale with it."""
    dswct = _build_dswct()
    assert dswct.inst_shape == (_GN, _POL)
    assert dswct.inst_count == _GN * _POL
    assert dswct.digit_num == _W_DIGIT
    config = dswct.config
    assert dswct.area__um2 == pytest.approx(config.area_per_inst__um2 * _GN * _POL)
    assert dswct.leakage__uW == pytest.approx(config.leakage_per_inst__uW * _GN * _POL)


def test_forward_drops_exactly_the_digit_axis() -> None:
    """Output shape is the input shape with the trailing `w_digit` axis reduced."""
    dswct = _build_dswct()
    for leading in ((), (4,), (2, 3)):
        i_dl = _i_dl(*leading)
        out = dswct(i_dl, window__ns=1.0)
        assert tuple(out.shape) == (*leading, _SERIAL, _GN, _POL)


def test_fabrication_prefix_rides_inst_shape() -> None:
    """A fabrication prefix multiplies inst_count; the bank trailing stays (gn, 2)."""
    dswct = _build_dswct(inst_shape=(5, _GN, _POL))
    assert dswct.inst_count == 5 * _GN * _POL


# ---------------------------------------------------------------------------
# Value law
# ---------------------------------------------------------------------------


def test_forward_equals_inline_computation() -> None:
    """Forward == LSB-first digit-ratio weighting + sum over w_digit, exactly."""
    dswct = _build_dswct()
    i_dl = _i_dl(2)
    out = dswct(i_dl, window__ns=1.0)
    assert torch.equal(out, _inline_i_wdl(i_dl))


def test_forward_outside_profiler_emits_nothing_and_matches() -> None:
    """No active profiler: forward still runs and a later profiled run sees only its own records."""
    dswct = _build_dswct()
    i_dl = _i_dl()
    out_plain = dswct(i_dl, window__ns=2.0)
    with Profiler() as prof:
        out_profiled = dswct(i_dl, window__ns=2.0)
    assert torch.equal(out_plain, out_profiled)
    assert len(prof.records) == 1


# ---------------------------------------------------------------------------
# Billing law
# ---------------------------------------------------------------------------


def test_rail_billing_is_vdd_abs_i_wdl_window() -> None:
    """Scalar window: E = VDD * sum |i_dl * ratio| * window over all legs (signed input)."""
    dswct = _build_dswct()
    i_dl = _i_dl(2)  # signed entries: the |I| convention is load-bearing
    window__ns = 4.0
    with Profiler() as prof:
        dswct(i_dl, window__ns=window__ns)
    ratios = torch.tensor(_DIGIT_RATIOS, dtype=_DTYPE)
    expect__fJ = float(_VDD__V * (i_dl * ratios).abs().sum() * window__ns)
    assert Reporter(dswct).total_dynamic_energy__fJ(prof) == pytest.approx(expect__fJ, rel=1e-12)


def test_per_bit_diagonal_window_rides_the_leading_batch() -> None:
    """Per-plane window tensor: each leading plane bills against ITS window only."""
    dswct = _build_dswct()
    x_bits = 2
    i_dl = _i_dl(x_bits)  # leading = the WL bit-plane axis
    window__ns = torch.tensor((2.0, 5.0), dtype=_DTYPE)
    with Profiler() as prof:
        dswct(i_dl, window__ns=window__ns)
    ratios = torch.tensor(_DIGIT_RATIOS, dtype=_DTYPE)
    per_plane = (i_dl * ratios).abs().sum(dim=(-4, -3, -2, -1))
    expect__fJ = float(_VDD__V * (per_plane * window__ns).sum())
    assert Reporter(dswct).total_dynamic_energy__fJ(prof) == pytest.approx(expect__fJ, rel=1e-12)


def test_cap_event_per_slot_plane_bank() -> None:
    """c_load * VDD**2 fires once per (slot x plane) per bank on top of the rail term."""
    c_load__fF = 0.5
    dswct = _build_dswct(c_load__fF=c_load__fF)
    plane_num = 2
    i_dl = _i_dl(plane_num)
    window__ns = 4.0
    with Profiler() as prof:
        dswct(i_dl, window__ns=window__ns)
    ratios = torch.tensor(_DIGIT_RATIOS, dtype=_DTYPE)
    rail__fJ = float(_VDD__V * (i_dl * ratios).abs().sum() * window__ns)
    cap__fJ = c_load__fF * _VDD__V**2 * (plane_num * _SERIAL * _GN * _POL)
    assert Reporter(dswct).total_dynamic_energy__fJ(prof) == pytest.approx(rail__fJ + cap__fJ, rel=1e-12)


def test_zero_c_load_bills_rail_only() -> None:
    """The shipped c_load = 0.0 leaves the billed energy exactly the rail term."""
    i_dl = _i_dl(2)
    window__ns = 3.0
    dswct = _build_dswct(c_load__fF=0.0)
    with Profiler() as prof_zero:
        dswct(i_dl, window__ns=window__ns)
    ratios = torch.tensor(_DIGIT_RATIOS, dtype=_DTYPE)
    expect__fJ = float(_VDD__V * (i_dl * ratios).abs().sum() * window__ns)
    assert Reporter(dswct).total_dynamic_energy__fJ(prof_zero) == pytest.approx(expect__fJ, rel=1e-12)


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------


def test_construction_guards() -> None:
    with pytest.raises(ValueError, match="inst_shape"):
        _build_dswct(inst_shape=(_GN, 3))  # polarity axis must be 2
    with pytest.raises(ValueError, match="inst_shape"):
        _build_dswct(inst_shape=(2,))  # missing (gn, 2) trailing
    with pytest.raises(ValueError, match="digit_ratios"):
        Dswct(
            config=_build_config(),
            policy=DswctPolicy(),
            inst_shape=(_GN, _POL),
            digit_ratios=torch.ones((2, 2), dtype=_DTYPE),  # not 1-D
            vdd__V=_VDD__V,
        )
    with pytest.raises(ValueError, match="c_load__fF"):
        _build_config(c_load__fF=-1.0)


def test_forward_trailing_shape_guards() -> None:
    dswct = _build_dswct()
    with pytest.raises(ValueError, match="forward"):
        dswct(torch.zeros((_SERIAL, _GN, _POL), dtype=_DTYPE), window__ns=1.0)  # ndim < 4
    with pytest.raises(ValueError, match="forward"):
        # wrong digit count
        dswct(torch.zeros((_SERIAL, _GN, _POL, _W_DIGIT + 1), dtype=_DTYPE), window__ns=1.0)
    with pytest.raises(ValueError, match="forward"):
        # lane axes swapped against inst_shape trailing
        dswct(torch.zeros((_SERIAL, _POL, _GN + 1, _W_DIGIT), dtype=_DTYPE), window__ns=1.0)
