"""Element-wise integer adder with energy accounting.

See Also:
    docs/reference/primitive/digital/adder.md
"""

import torch
from torch import Tensor

from .base import DigitalBase, DigitalConfig, DigitalPolicy


class AdderConfig(DigitalConfig):
    bit_width: int
    """Nominal output bit width; sizes the PPA, no wrap is applied."""

    energy_per_op__fJ: float
    """Dynamic energy per output element."""
    latency_per_op__ns: float
    """Combinational window of one add."""

    def validate(self) -> None:
        super().validate()

        # --- Arithmetic ---

        self._require_pos(self.bit_width, "bit_width")

        # --- PPA ---

        self._require_non_neg(self.energy_per_op__fJ, "energy_per_op__fJ")
        self._require_non_neg(self.latency_per_op__ns, "latency_per_op__ns")


class Adder(DigitalBase[AdderConfig]):
    """Element-wise integer adder without saturation or wrapping."""

    def __init__(
        self,
        *,
        config: AdderConfig,
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

    def add(self, a: Tensor, b: Tensor) -> Tensor:
        """Add two integer tensors element-wise.

        `b` broadcasts against `a`.

        Returns:
            `a + b`, unwrapped and unsaturated.
        """
        y = a + b
        if self._is_dynamic_energy_profile_active():
            # Shape: [] -> [*y.shape]
            e_op__fJ = torch.full((), self.config.energy_per_op__fJ, dtype=torch.float32, device=y.device)
            self._record_dynamic_energy(e_op__fJ.expand(y.shape))
        return y
