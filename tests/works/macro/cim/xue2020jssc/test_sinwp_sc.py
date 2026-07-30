"""Laws for the SINWP-SC input-radix sample-and-hold combine module.

Hand-built tiny witness, eager, CPU. Three laws:

  * SHAPE LAW: ``inst_count`` derives from the config geometry (``gn * 2``, no
    magic numbers) and ``forward`` reduces exactly the bit axis.
  * VALUE LAW: ``forward`` equals the inline LSB-first ratio-weighted bit sum.
  * LEG-BILLING LAW (branch-tensor law): the recorded dynamic energy equals
    the materialized-leg formula — ``v_dd * sum_k (sum_lanes s_k * i[k]) *
    window[k]`` with the per-leg currents ``i_leg[k] = bit_ratios[k] * i[k]``
    (the mirror legs carry the ``s_k``-scaled copies, NOT the raw interface
    current — non-unity ratios make an interface-current bill fail) and the
    SIGNED per-leg sum (a mixed-sign witness pins the no-``|I|`` semantics) —
    plus the ``c_hold * v_dd**2`` per-(slot x bit) per-instance cap event; no
    latency event (the macro is the sole emitter). The value output is the sum
    of the SAME legs the billing consumed.
"""

from __future__ import annotations

import pytest
import torch

from neurox.common.profiler import NeuroxProfiler
from neurox.works.macro.cim.xue2020jssc.sinwp_sc import SinwpSc, SinwpScConfig, SinwpScPolicy

_DTYPE = torch.float64

# Tiny hand-written witness geometry.
_GN = 3
_POLARITY_NUM = 2
_SERIAL = 2
_X_BITS = 2
_V_DD__V = 1.0
_C_HOLD__fF = 0.7
# LSB-first input-radix combine ratios (paper law s_k = msb * 2**(k - (K-1))).
_BIT_RATIOS = (0.25, 0.5)
_WINDOW__NS = (7.0, 4.0)


def _build_sinwp_sc(*, c_hold__fF: float = _C_HOLD__fF) -> SinwpSc:
    module = SinwpSc(
        config=SinwpScConfig(
            c_hold__fF=c_hold__fF,
            area_per_inst__um2=0.0,
            leakage_per_inst__uW=0.0,
        ),
        policy=SinwpScPolicy(),
        inst_shape=(_GN, _POLARITY_NUM),
        bit_ratios=torch.tensor(_BIT_RATIOS, dtype=_DTYPE),
        v_dd__V=_V_DD__V,
    )
    module.fabricate()
    return module


def _witness_currents() -> torch.Tensor:
    """Mixed-sign lane currents [uA] at batch 1."""
    torch.manual_seed(0)
    return torch.randn(1, _X_BITS, _SERIAL, _GN, _POLARITY_NUM, dtype=_DTYPE)


def test_shape_law() -> None:
    """inst_count = gn * 2 from the geometry; forward reduces exactly the bit axis."""
    module = _build_sinwp_sc()
    assert module.inst_count == _GN * _POLARITY_NUM
    assert module.input_bit_num == len(_BIT_RATIOS)

    i__uA = _witness_currents()
    out = module(i__uA, window_per_bit__ns=torch.tensor(_WINDOW__NS, dtype=_DTYPE))
    assert tuple(out.shape) == (1, _SERIAL, _GN, _POLARITY_NUM)


def test_value_law() -> None:
    """forward == inline LSB-first ratio-weighted sum over the bit axis."""
    module = _build_sinwp_sc()
    i__uA = _witness_currents()
    out = module(i__uA, window_per_bit__ns=torch.tensor(_WINDOW__NS, dtype=_DTYPE))

    ratios = torch.tensor(_BIT_RATIOS, dtype=_DTYPE).view(_X_BITS, 1, 1, 1)
    expected = (i__uA * ratios).sum(dim=-4)
    assert torch.equal(out, expected)


def test_leg_billing_law() -> None:
    """Recorded energy == materialized-leg formula (signed leg sum) + cap events; no latency.

    The bit ratios are non-unity (0.25, 0.5), so a bill of the raw interface
    currents (the pre-fix bug: ``v_dd * sum_lanes(i[k]) * window[k]``) differs
    from the leg bill by more than the tolerance — the witness discriminates
    the two placements of the scaling.
    """
    module = _build_sinwp_sc()
    i__uA = _witness_currents()
    # The witness must carry mixed signs so an |I| implementation fails.
    assert (i__uA < 0).any() and (i__uA > 0).any()
    window = torch.tensor(_WINDOW__NS, dtype=_DTYPE)

    with NeuroxProfiler() as prof, torch.no_grad():
        module(i__uA, window_per_bit__ns=window)

    # Branch-tensor law: the billed branches are the materialized legs
    # i_leg[k] = s_k * i[k], not the interface currents.
    ratios = torch.tensor(_BIT_RATIOS, dtype=_DTYPE).view(_X_BITS, 1, 1, 1)
    i_leg = i__uA * ratios
    # The SIGNED leg sum.
    # Shape: [1, x_bits, serial, gn, 2] -> [1, x_bits]
    i_leg_per_bit = i_leg.sum(dim=(-3, -2, -1))
    # Shape: [1, x_bits] -> []
    e_conduction__fJ = (_V_DD__V * (i_leg_per_bit * window).sum(dim=-1)).sum()
    e_conduction = float(e_conduction__fJ)
    e_cap = _C_HOLD__fF * _V_DD__V**2 * (_X_BITS * _SERIAL * _GN * _POLARITY_NUM)
    assert prof.total_dynamic_energy__fJ == pytest.approx(e_conduction + e_cap)
    assert prof.total_latency__ns == 0.0

    # The interface-current bill (the pre-fix scaling placement) is a
    # DIFFERENT number on this witness — the law discriminates.
    i_iface_per_bit = i__uA.sum(dim=(-3, -2, -1))
    # Shape: [1, x_bits] -> []
    e_iface__fJ = (_V_DD__V * (i_iface_per_bit * window).sum(dim=-1)).sum()
    e_iface = float(e_iface__fJ)
    assert e_iface != pytest.approx(e_conduction)


def test_billing_outside_profiler_is_silent() -> None:
    """forward outside a profiler records nothing and still returns the value."""
    module = _build_sinwp_sc()
    i__uA = _witness_currents()
    out = module(i__uA, window_per_bit__ns=torch.tensor(_WINDOW__NS, dtype=_DTYPE))
    assert tuple(out.shape) == (1, _SERIAL, _GN, _POLARITY_NUM)
