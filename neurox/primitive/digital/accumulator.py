"""Digital modular-arithmetic accumulator over an integer-tensor axis.

See also:
    docs/reference/primitive/digital/accumulator.md
"""

import torch
from torch import Tensor

from .base import DigitalBase, DigitalConfig, DigitalPolicy


class AccumulatorConfig(DigitalConfig):
    """Immutable configuration for an Accumulator instance.

    Attributes:
        bit_width: Signed output bit width; result wraps modulo ``2^bit_width``
            into ``[-2^(bw-1), 2^(bw-1) - 1]``.
        energy_per_op__fJ: Dynamic energy consumed per operand element folded
            into the sum.
        latency_per_op__ns: Reduction window of one accumulate.
    """

    bit_width: int

    energy_per_op__fJ: float
    latency_per_op__ns: float

    def validate(self) -> None:
        super().validate()

        # --- Arithmetic ---

        self._require_pos(self.bit_width, "bit_width")

        # --- PPA ---

        self._require_non_neg(self.energy_per_op__fJ, "energy_per_op__fJ")
        self._require_non_neg(self.latency_per_op__ns, "latency_per_op__ns")


class Accumulator(DigitalBase[AccumulatorConfig]):
    """Modular adder-tree that sums an integer tensor along one axis.

    Args:
        config: Accumulator configuration.
        policy: Digital execution policy.
        inst_shape: Per-instance fabrication shape.
    """

    def __init__(
        self,
        *,
        config: AccumulatorConfig,
        policy: DigitalPolicy,
        inst_shape: tuple[int, ...],
    ) -> None:
        super().__init__(config=config, policy=policy, inst_shape=inst_shape)

    @property
    def _area_per_inst__um2(self) -> float:
        return self.config.area_per_inst__um2

    @property
    def _leakage_per_inst__uW(self) -> float:
        return self.config.leakage_per_inst__uW

    def accumulate(self, x: Tensor, dim: int) -> Tensor:
        """Sum ``x`` along ``dim`` and wrap into the signed ``bit_width`` range.

        Dynamic energy is billed against the pre-reduction operand ``x``: one
        adder evaluation is spent per operand folded in, so the switching count
        follows the reduced extent, which the result no longer carries.

        Args:
            x: Integer-valued input tensor.
            dim: Axis along which to reduce.

        Returns:
            Modular-wrapped sum with ``dim`` reduced.
        """
        bw = self.config.bit_width
        half = 1 << (bw - 1)
        full = 1 << bw
        y = (x.sum(dim) + half) % full - half

        if self._is_dynamic_energy_profile_active():
            # The engine positions this block's own inst_shape space axes
            # inside x, split from the batch by the reduced axis and any
            # further engine axes rather than held as one leading block;
            # billing the full pre-reduction operand covers them along with
            # the rest. A flat per-op lump: the expanded constant holds no
            # storage, and the energy dtype is the constant's rather than the
            # integer operand's.
            # Shape: [] -> [*x.shape]
            e_op__fJ = torch.full((), self.config.energy_per_op__fJ, dtype=torch.float32, device=x.device)
            self._record_dynamic_energy(e_op__fJ.expand(x.shape))
        return y
