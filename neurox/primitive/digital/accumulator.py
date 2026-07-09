"""Digital modular-arithmetic accumulator over an integer-tensor axis.

See also:
    docs/reference/primitive/digital/accumulator.md
"""

from dataclasses import dataclass

import torch
from torch import Tensor

from neurox.primitive.circuit import CircuitConfig

from .base import DigitalCircuit


@dataclass(frozen=True)
class AccumulatorConfig(CircuitConfig):
    """Immutable configuration for an Accumulator instance.

    Attributes:
        bit_width: Signed output bit width; result wraps modulo ``2^bit_width``
            into ``[-2^(bw-1), 2^(bw-1) - 1]``.
        energy_per_op__fJ: Dynamic energy consumed per output element.
        latency_per_op__ns: Per-output-element latency; multiplied
            by the runtime serial-op count at logging time.
    """

    bit_width: int

    energy_per_op__fJ: float
    latency_per_op__ns: float

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        self.validate_arithmetic()
        self.validate_ppa()

    def validate_arithmetic(self) -> None:
        self._require_pos(self.bit_width, "bit_width")

    def validate_ppa(self) -> None:
        super().validate_ppa()
        self._require_non_neg(self.energy_per_op__fJ, "energy_per_op__fJ")
        self._require_non_neg(self.latency_per_op__ns, "latency_per_op__ns")


class Accumulator(DigitalCircuit[AccumulatorConfig]):
    """Modular adder-tree that sums an integer tensor along one axis.

    Models a hardware adder tree with a fixed output register of ``bit_width``
    bits.  Overflow wraps via two's-complement modular arithmetic, matching
    the behavior of a synthesized ripple-carry or carry-save adder tree with
    no saturation logic.
    """

    def __init__(
        self,
        *,
        config: AccumulatorConfig,
        name: str,
        inst_shape: tuple[int, ...],
    ) -> None:
        super().__init__(config=config, name=name, inst_shape=inst_shape)

    def operate(self, x: Tensor, dim: int) -> Tensor:
        """Sum ``x`` along ``dim`` and wrap into the signed ``bit_width`` range.

        Dynamic energy and latency emit through the profiler side
        channel.

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

        # Each tree adder produces one output element. Serial via the
        # position-invariant numel rule (reduced dim is already gone
        # from y so the divisor is just inst_count).
        serial_op_count = -(-y.numel() // max(self.inst_count, 1))  # ceil(numel / inst); empty -> 0
        dynamic_energy__fJ = torch.full_like(y, self.config.energy_per_op__fJ, dtype=torch.float32)
        latency__ns = torch.tensor(
            self.config.latency_per_op__ns * serial_op_count,
            device=y.device,
            dtype=dynamic_energy__fJ.dtype,
        )
        self._log_dynamic_energy(dynamic_energy__fJ)
        self._log_latency(latency__ns)
        return y
