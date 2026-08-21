"""Laws for the scheme-local PN-ISUB subtractor module.

Hand-built tiny witness, eager, CPU. Three laws:

  * SHAPE LAW: the module fabricates at the config-derived CIM-IO count
    `gn = output_num // mux_factor` (no magic numbers) and its static
    area / leakage totals scale with `inst_count`.
  * VALUE LAW: `forward` equals the inline subtraction — magnitude
    `|I_P - I_N|` and sign `I_N > I_P` (a tie is non-negative) —
    and is billing-independent (identical with and without a profiler).
  * BILLING LAW: one un-channelled dynamic event per forward whose total is
    `sum(VDD * window * (I_P + I_N + I_SUB)) + e_per_op * entry_count`
    — the three rail branches over the injected window plus the comparator
    decision constant once per (slot, IO) entry.
"""

from __future__ import annotations

import pytest
import torch

from neurox import Profiler, Reporter, stamp_names
from neurox.works.macro.cim.xue2020jssc.pn_isub import PnIsub, PnIsubConfig, PnIsubPolicy

_DTYPE = torch.float64

# --- Tiny witness geometry (mirrors the scheme witness in miniature) ---
_OUTPUT_NUM = 4
_MUX_FACTOR = 2  # serial slot count; gn = output_num // mux_factor = 2

# --- Witness physics knobs (small explicit values, no code defaults) ---
_VDD__V = 0.9
_WINDOW__ns = 3.5
_E_PER_OP__fJ = 1.25
_AREA_PER_INST__um2 = 2.0
_LEAKAGE_PER_INST__uW = 3.0


def _config(*, e_per_op__fJ: float = _E_PER_OP__fJ) -> PnIsubConfig:
    return PnIsubConfig(
        e_per_op__fJ=e_per_op__fJ,
        area_per_inst__um2=_AREA_PER_INST__um2,
        leakage_per_inst__uW=_LEAKAGE_PER_INST__uW,
    )


def _build(*, gn: int) -> PnIsub:
    module = PnIsub(
        config=_config(),
        policy=PnIsubPolicy(),
        inst_shape=(gn,),
        vdd__V=_VDD__V,
    )
    module.eval()
    module.fabricate()
    stamp_names(module)  # the standalone module is its own root, named ""
    return module


def _lane_currents(gn: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Asymmetric polarity currents covering P>N, N>P, and a tie."""
    i_p = torch.tensor([[[6.0, 1.5], [2.0, 4.0]]], dtype=_DTYPE)
    i_n = torch.tensor([[[2.5, 5.0], [2.0, 0.5]]], dtype=_DTYPE)
    assert i_p.shape == (1, _MUX_FACTOR, gn)
    return i_p, i_n


# ---------------------------------------------------------------------------
# Shape law
# ---------------------------------------------------------------------------


def test_inst_count_derived_from_config_geometry() -> None:
    """inst_shape = (gn,) with gn = output_num // mux_factor; static PPA scales with inst_count."""
    gn = _OUTPUT_NUM // _MUX_FACTOR
    module = _build(gn=gn)
    assert module.inst_shape == (gn,)
    assert module.inst_count == _OUTPUT_NUM // _MUX_FACTOR
    assert module.area__um2 == pytest.approx(_AREA_PER_INST__um2 * gn)
    assert module.leakage__uW == pytest.approx(_LEAKAGE_PER_INST__uW * gn)


# ---------------------------------------------------------------------------
# Value law
# ---------------------------------------------------------------------------


def test_forward_equals_inline_subtraction() -> None:
    """Magnitude = |I_P - I_N|, sign = (I_N > I_P); a tie decides non-negative."""
    gn = _OUTPUT_NUM // _MUX_FACTOR
    module = _build(gn=gn)
    i_p, i_n = _lane_currents(gn)

    i_sub_abs, sign = module(i_p, i_n, window__ns=_WINDOW__ns)

    assert torch.equal(i_sub_abs, (i_p - i_n).abs())
    assert torch.equal(sign, i_n > i_p)
    # The tie entry (I_P == I_N) yields zero magnitude and a non-negative sign.
    assert i_sub_abs[0, 1, 0] == 0.0
    assert not bool(sign[0, 1, 0])

    # Billing-independent values: identical under an active profiler.
    with Profiler(), torch.no_grad():
        i_sub_abs_prof, sign_prof = module(i_p, i_n, window__ns=_WINDOW__ns)
    assert torch.equal(i_sub_abs_prof, i_sub_abs)
    assert torch.equal(sign_prof, sign)


# ---------------------------------------------------------------------------
# Billing law
# ---------------------------------------------------------------------------


def test_dynamic_energy_equals_rail_branches_plus_per_op() -> None:
    """E = sum(VDD * window * (I_P + I_N + I_SUB)) + e_per_op per (slot, IO) entry."""
    gn = _OUTPUT_NUM // _MUX_FACTOR
    module = _build(gn=gn)
    i_p, i_n = _lane_currents(gn)

    with Profiler() as prof, torch.no_grad():
        module(i_p, i_n, window__ns=_WINDOW__ns)

    i_sub_abs = (i_p - i_n).abs()
    conduction__fJ = float((_VDD__V * _WINDOW__ns * (i_p + i_n + i_sub_abs)).sum())
    entry_count = i_p.numel()  # once per output-code sign decision: per (slot, IO) entry
    expected__fJ = conduction__fJ + _E_PER_OP__fJ * entry_count

    reporter = Reporter(module)
    assert reporter.total_dynamic_energy__fJ(prof) == pytest.approx(expected__fJ, rel=1e-12)
    # One un-channelled record per forward — the module's own report row.
    assert len(prof.records) == 1
    assert prof.records[0].channel is None
    assert reporter.by_name(prof) == {"": pytest.approx(expected__fJ, rel=1e-12)}
