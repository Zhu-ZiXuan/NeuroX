"""Digital modular-arithmetic accumulator for a time-serial operand stream.

See also:
    docs/reference/primitive/digital/serial_accumulator.md
"""

import torch
from torch import Tensor

from .accumulator import Accumulator


class SerialAccumulator(Accumulator):
    """Modular accumulator for a time-serial operand axis."""

    def operate(self, x: Tensor, dim: int) -> Tensor:
        """Sum ``x`` along ``dim`` and wrap into the signed ``bit_width`` range.

        Args:
            x: Integer-valued input tensor; the ``dim`` axis is a time-serial
                operand stream.
            dim: Axis along which to reduce.

        Returns:
            Modular-wrapped sum with ``dim`` reduced.
        """
        bw = self.config.bit_width
        half = 1 << (bw - 1)
        full = 1 << bw
        y = (x.sum(dim) + half) % full - half

        serial_op_count = -(-x.numel() // max(self.inst_count, 1))  # ceil(numel / inst); empty -> 0
        dynamic_energy__fJ = torch.full_like(x, self.config.energy_per_op__fJ, dtype=torch.float32)
        latency__ns = torch.tensor(
            self.config.latency_per_op__ns * serial_op_count,
            device=y.device,
            dtype=dynamic_energy__fJ.dtype,
        )
        self._log_dynamic_energy(dynamic_energy__fJ)
        self._log_latency(latency__ns)
        return y
