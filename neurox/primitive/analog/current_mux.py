"""Ideal current multiplexer — single-ended N:1 time-share transport block.

See Also:
    docs/reference/primitive/analog/current_mux.md
"""

from __future__ import annotations

import torch
from torch import Tensor

from neurox.common.module import ConfigBase, NonProfileModule, PolicyBase


class ImuxConfig(ConfigBase):
    # === Multiplexing ===

    mux_ratio: int
    """N in the N:1 ratio of inputs to each output lane."""
    mux_gain: float
    """Matched transport gain shared by every lane."""

    def validate(self) -> None:

        # --- Multiplexing ---

        self._require_pos(self.mux_ratio, "mux_ratio")
        self._require_pos(self.mux_gain, "mux_gain")


class ImuxPolicy(PolicyBase):
    pass


_Config = ImuxConfig
_Policy = ImuxPolicy


class Imux(NonProfileModule):
    """Apply an ideal current transport gain to already selected input currents.

    `transport` is elementwise and preserves layout. The caller selects channels
    and schedules their reuse; `mux_ratio` records the intended multiplexing
    geometry but performs no indexing or time loop. This implementation needs no
    fabrication or programming and reports no independent PPA or latency.

    Args:
        config: Hardware configuration.
        policy: Run policy matching `config`.
        inst_shape: Positive physical instance extents; singletons allow
            broadcasting.
        dtype: Retained argument; transport uses normal tensor type promotion.
    """

    config: _Config
    policy: _Policy

    def __init__(
        self,
        *,
        config: _Config,
        policy: _Policy,
        inst_shape: tuple[int, ...],
        dtype: torch.dtype,
    ) -> None:
        super().__init__(config=config, policy=policy, inst_shape=inst_shape)

    @torch.no_grad()
    def transport(self, i__uA: Tensor) -> Tensor:
        """Apply the mux transport gain elementwise.

        Args:
            i__uA: Single-ended input currents.

        Returns:
            Gained currents, one value per `i__uA` element.
        """
        return self.config.mux_gain * i__uA
