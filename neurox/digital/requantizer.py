"""Fixed-point requantizer for CiM MAC output rescaling.

After accumulation and digit recombination the partial-product tensor is in an
intermediate scale that differs from the target output scale.  This module
rescales it using the standard integer-arithmetic requantization formula

    y = (x * multiplier) >> rshift  [+ output_zero_point]

where ``multiplier`` is an int32 fixed-point scale factor and ``rshift`` is a
power-of-two right shift.  This matches the requantization scheme used by
quantization-aware training frameworks (e.g. PyTorch/FBGEMM, TFLite) and maps
directly to a multiply-high + shift sequence on hardware.

An optional ``output_zero_point`` offsets the result to the target quantization
grid.  PPA metrics are tracked per-instance; physical instance count is set
externally by the parent macro at fabricate time via
``ProfiledModule._record_inst_count``.

Stochastic rounding
-------------------
Set ``RequantizerConfig.stochastic`` to ``True`` to force unbiased
stochastic rounding on the right-shift step; ``False`` to disable;
``None`` (default) lets the calling ``nn.Module``'s ``training`` flag
choose — stochastic in train mode, deterministic in eval.  The
canonical pattern is implemented in
:func:`neurox.common.quant.stochastic_floor_div`.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from neurox.common.quant import stochastic_floor_div
from neurox.profiler import ProfiledModule


@dataclass(frozen=True)
class RequantizerConfig:
    """Immutable configuration for a Requantizer instance.

    Attributes:
        bit_width: Nominal output bit width (informational; the caller clips
            to this range after requantization if needed).
        energy_per_op__fJ: Dynamic energy consumed per output element (fJ).
        latency_per_op__ns: Critical-path latency per operation (ns).
        leakage_per_inst__uW: Static leakage power per instance (uW).
        area_per_inst__um2: Silicon area per instance (um^2).
    """

    bit_width: int

    energy_per_op__fJ: float = 0.0

    latency_per_op__ns: float = 0.0
    leakage_per_inst__uW: float = 0.0
    area_per_inst__um2: float = 0.0


class Requantizer(nn.Module, ProfiledModule):
    """Multiply-shift requantizer that maps accumulated MACs to output scale.

    Implements ``y = (x * multiplier) >> rshift [+ output_zero_point]``.
    The multiply-then-right-shift sequence models the integer multiply-high
    unit and barrel shifter found in dedicated CiM peripheral circuits or
    general-purpose digital post-processing cores.

    When ``module.training`` (or the ``stochastic`` init override) selects
    stochastic rounding, the right-shift adds a uniform
    ``[0, 1 << rshift)`` jitter before shifting — unbiased and
    ``torch.compile``-safe.

    Args:
        config: Immutable cost / bit-width configuration.
        name: Hierarchical instance name used by the profiler.
        stochastic: Per-instance switch for stochastic rounding on the
            right-shift.  ``None`` (default) follows
            ``module.training``; ``True`` / ``False`` force on / off.
            Kept as an init arg, not a config field, so the same
            cost-model config can be reused across train and eval
            macros.
    """

    def __init__(self, config: RequantizerConfig, *, name: str = "", stochastic: bool | None = None) -> None:
        nn.Module.__init__(self)
        ProfiledModule.__init__(self, name)
        self.config = config
        self.stochastic: bool | None = stochastic

    @property
    def area_per_inst__um2(self) -> float:
        """Area per instance in um2."""
        return self.config.area_per_inst__um2

    @property
    def leakage_per_inst__uW(self) -> float:
        """Leakage per instance in uW."""
        return self.config.leakage_per_inst__uW

    @property
    def latency_per_op__ns(self) -> float:
        """Latency per op in ns."""
        return self.config.latency_per_op__ns

    def fabricate(self) -> None:
        """No-op: Requantizer's instance count is set by the parent macro."""
        return None

    def operate(self, x: Tensor, multiplier: Tensor, rshift: Tensor, output_zero_point: Tensor | None) -> Tensor:
        """Rescale accumulated partial products to the target output scale.

        Computes ``y = (x * multiplier) >> rshift`` and optionally adds
        ``output_zero_point`` to shift onto the target quantization grid.
        Dynamic energy and latency emit through the profiler side channel.

        Args:
            x: Accumulated integer MAC result tensor.
            multiplier: Per-channel or scalar int32 fixed-point scale factor,
                broadcastable to ``x``.
            rshift: Per-channel or scalar right-shift amount (non-negative),
                broadcastable to ``x``.
            output_zero_point: Zero-point offset added after the shift.
                Pass ``None`` to skip (e.g. for symmetric quantization).

        Returns:
            Rescaled integer tensor.
        """
        y = stochastic_floor_div(
            x * multiplier,
            rshift,
            training=self.training,
            override=self.stochastic,
        )
        if output_zero_point is not None:
            y = y + output_zero_point

        dynamic_energy__fJ = torch.full_like(y, self.config.energy_per_op__fJ, dtype=torch.float32)
        self._log_dynamic(dynamic_energy__fJ, self.config.latency_per_op__ns)
        return y
