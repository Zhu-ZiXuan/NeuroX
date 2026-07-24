"""Digital modular-arithmetic accumulator for a time-serial operand stream.

See also:
    docs/reference/primitive/digital/serial_accumulator.md
"""

import torch
from torch import Tensor

from .accumulator import Accumulator


class SerialAccumulator(Accumulator):
    """Modular accumulator for a time-serial operand axis."""

    def accumulate(self, x: Tensor, dim: int) -> Tensor:
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

        serial_round_count = self._count_serial_rounds(x.numel())
        latency__ns = self._latency_per_op__ns * serial_round_count
        if self._is_dynamic_energy_profile_active():
            self._record_dynamic_energy(torch.full_like(x, self.config.energy_per_op__fJ, dtype=torch.float32))
        self._record_latency(latency__ns)
        return y
