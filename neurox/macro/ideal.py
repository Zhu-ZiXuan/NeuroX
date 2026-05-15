"""Macro-level ideal baseline: ``IdealMacro``.

``IdealMacro`` is the macro-level counterpart to ``Xbar1T1R`` configured
with an ideal switch (the xbar-level baseline in
:mod:`neurox.xbar.xbar_1t1r`).  Where an ideal-switch ``Xbar1T1R``
plugs into ``XbarMacro`` to validate the macro's reshape / tiling /
aggregation logic, ``IdealMacro`` *replaces* the entire ``XbarMacro`` —
no tiling, no sign-split, no transcoding, no ADC clipping — with a single
exact-integer matmul + integer requantize.

It satisfies ``NeuroxMacroQuantMatMul`` and exposes the same
``output_rescale_factor`` (always ``1.0``), so it drops into operator
layers (``QuantLinear`` / ``QuantConv2d``) wherever a real macro would.
Two intended uses:

1. **Operator wiring check.**  Build a ``QuantLinear`` / ``QuantConv2d``
   around a ``IdealMacro`` and confirm the operator-replacement flow
   (state-dict load, ``fabricate``, forward) yields finite, correctly-
   shaped output.  Any failure here is an operator-side bug — not a
   macro or crossbar bug.

2. **Macro accuracy reference.**  Run the same crafted state through
   ``IdealMacro`` and through a real ``XbarMacro``.  The gap measures the
   real macro's combined ADC + tiling + circuit error.

``IdealMacro`` reports zero PPA metrics — it is not intended for any
hardware-cost analysis.

Range surface (§11 of ``temp/mapping.md``)
------------------------------------------
``IdealMacro`` receives ``x_value_range`` / ``w_value_range`` directly
as explicit tuples, matching what ``XbarMacro.x_value_range`` /
``XbarMacro.w_value_range`` publish via their mapper.  Both macro
families therefore expose the **same** range surface to operators —
the ideal twin is a drop-in baseline, not a parallel API.
"""

import torch
import torch.nn as nn
from torch import Tensor


class IdealMacro(nn.Module):
    """Macro-level ideal baseline: exact integer matmul + integer requantize.

    Satisfies ``NeuroxMacroQuantMatMul`` with no tiling, no radix
    decomposition, no sign-split, and no analog effects.  The ``matmul``
    method computes a plain ``torch.matmul`` in int64, optionally adds
    ``bias``, applies the ``Requantizer``-style multiply-shift, and adds
    the output zero-point — matching the integer arithmetic that
    ``XbarMacro.matmul`` performs after all crossbar-specific stages.

    Use as the macro-level reference when:

    * verifying the operator-replacement flow (see module docstring), or
    * measuring how much accuracy a real ``XbarMacro`` loses relative to
      a perfectly-accurate integer matmul on the same input + state.

    For the xbar-level analogue (a perfect crossbar plugged into a real
    ``XbarMacro``), use ``neurox.xbar.xbar_1t1r.Xbar1T1R`` configured
    with an ideal switch (``g_on = inf``, ``g_off = 0``).

    Args:
        x_value_range: Inclusive ``(lo, hi)`` algorithm-side integer
            activation range.  Mirrors the value ``XbarMacro``
            publishes from its mapper.
        w_value_range: Inclusive ``(lo, hi)`` algorithm-side integer
            weight range.  Mirrors the value ``XbarMacro`` publishes.
    """

    def __init__(
        self,
        *,
        x_value_range: tuple[int, int],
        w_value_range: tuple[int, int],
    ) -> None:
        super().__init__()
        self._x_value_range = x_value_range
        self._w_value_range = w_value_range

    @property
    def w_value_range(self) -> tuple[int, int]:
        return self._w_value_range

    @property
    def x_value_range(self) -> tuple[int, int]:
        return self._x_value_range

    @property
    def output_rescale_factor(self) -> float:
        return 1.0

    def fabricate(self, weight: Tensor) -> None:
        """No-op; IdealMacro holds no physical state."""

    @torch.no_grad()
    @torch.compile
    def matmul(
        self,
        input: Tensor,
        weight: Tensor,
        bias: Tensor | None,
        rescale_multiplier: Tensor,
        rescale_rshift: Tensor,
        output_zero_point: Tensor | None,
    ) -> Tensor:
        """Integer matmul with fixed-point requantization using pure PyTorch.

        Macro-level ``@torch.compile(dynamic=True)`` entry, matching the
        compile policy on ``XbarMacro.matmul``.  IdealMacro has no
        physical cost and emits no profiler events.
        """
        x_dtype = input.dtype

        y = torch.matmul(input.to(torch.float32), weight.to(torch.float32).transpose(-2, -1))
        y = torch.round(y).to(torch.int32)

        if bias is not None:
            y = y + bias.to(torch.int32).unsqueeze(-2)

        m = rescale_multiplier.to(torch.int32).unsqueeze(-2)
        s = rescale_rshift.to(torch.int32).unsqueeze(-2)
        y = (y.to(torch.int32) * m) >> s

        if output_zero_point is not None:
            y = y + output_zero_point.to(torch.int32)

        return y.to(x_dtype)
