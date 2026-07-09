"""Shift-adder for multi-digit partial-product recombination.

See also:
    docs/reference/primitive/digital/shift_adder.md
"""

from dataclasses import dataclass

import torch
from torch import Tensor

from neurox.primitive.circuit import CircuitConfig

from .base import DigitalCircuit


@dataclass(frozen=True)
class ShiftAdderConfig(CircuitConfig):
    """Immutable configuration for a ShiftAdder instance.

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


class ShiftAdder(DigitalCircuit[ShiftAdderConfig]):
    """Weighted positional-sum unit for digit recombination."""

    def __init__(
        self,
        *,
        config: ShiftAdderConfig,
        name: str,
        inst_shape: tuple[int, ...],
    ) -> None:
        super().__init__(config=config, name=name, inst_shape=inst_shape)

    def operate(self, x: Tensor, scale: int, dim: int, init_val: Tensor | None) -> Tensor:
        """Compute the radix-weighted digit sum and wrap to ``bit_width`` bits.

        Args:
            x: Integer digit tensor; size of ``dim`` is the digit count.
            scale: Radix of the digit representation.
            dim: Axis indexing the digit positions.
            init_val: Optional partial-sum tensor added after the modular wrap,
                broadcastable to the output shape.

        Returns:
            Recombined sum with ``dim`` reduced.
        """
        bw = self.config.bit_width
        half = 1 << (bw - 1)
        full = 1 << bw
        scales = torch.tensor([scale**i for i in range(x.size(dim))], device=x.device, dtype=x.dtype)

        shape = [1] * x.ndim
        shape[dim] = x.size(dim)
        y = ((x * scales.view(*shape)).sum(dim=dim) + half) % full - half

        if init_val is not None:
            y = y + init_val

        # Each shift-adder produces one output element. Serial via the
        # position-invariant numel rule (reduced digit dim is already
        # gone from y so the divisor is just inst_count).
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
