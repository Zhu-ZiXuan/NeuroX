"""1-D interconnect wire model.

See also:
    docs/dev/modules/device/wire.md
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from neurox.common.validate import ValidateMixin


@dataclass(frozen=True)
class WireConfig(ValidateMixin):
    """Per-length wire capacitance configuration.

    Attributes:
        c__fF_per_um: Wire capacitance per unit length [fF/um].
    """

    c__fF_per_um: float

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        self._require_pos(self.c__fF_per_um, "c__fF_per_um")


class Wire(nn.Module):
    """Stateful wire discretization with per-segment resistance."""

    segment_r__MOhm: Tensor
    segment_g__uS: Tensor

    def __init__(
        self,
        *,
        cfg: WireConfig,
        T__K: float,
        dtype: torch.dtype,
    ) -> None:
        """Construct one wire model.

        Args:
            cfg: Wire configuration.
            T__K: Operating temperature [K].
            dtype: Tensor dtype for internal buffers.
        """
        super().__init__()
        self.cfg = cfg
        self.dtype = dtype
        self.T__K = T__K
        for buf_name in ("segment_r__MOhm", "segment_g__uS"):
            self.register_buffer(buf_name, torch.empty(0, dtype=dtype), persistent=False)

    @property
    def c__fF_per_um(self) -> float:
        """Wire capacitance per unit length [fF/um]."""
        return self.cfg.c__fF_per_um

    def fabricate(self, segment_r__MOhm: Tensor) -> None:
        """Bind a per-segment resistance tensor and derive conductance.

        Args:
            segment_r__MOhm: 1-D segment resistance tensor [MOhm].
        """
        segment_r = segment_r__MOhm.to(dtype=self.dtype, device=self.segment_r__MOhm.device)
        if segment_r.ndim != 1:
            raise ValueError(f"require: segment_r__MOhm.ndim ({segment_r.ndim}) == 1")
        num_nodes = segment_r.numel()
        if not (num_nodes >= 1):
            raise ValueError(f"require: segment_r__MOhm.numel ({num_nodes}) >= 1")
        if not bool((segment_r > 0.0).all()):
            raise ValueError(f"require: segment_r__MOhm > 0.0 (got min {segment_r.min().item()})")

        segment_g = 1.0 / segment_r

        self.register_buffer("segment_r__MOhm", segment_r, persistent=False)
        self.register_buffer("segment_g__uS", segment_g, persistent=False)
