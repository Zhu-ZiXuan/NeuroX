"""Ideal current multiplexer — single-ended N:1 time-share transport block.

See also:
    docs/reference/primitive/analog/current_mux.md
"""

from dataclasses import dataclass
from typing import ClassVar

import torch
from torch import Tensor

from neurox.primitive.analog.base import AnalogBase, AnalogConfig, AnalogPolicy


@dataclass(frozen=True, kw_only=True)
class CurrentMuxConfig(AnalogConfig):
    """Immutable configuration for :class:`CurrentMux`.

    Attributes:
        select_num: Design N of the N:1 fan-in — the number of columns
            sharing one lane. Cross-checked by the caller against its
            reference group size; design-only, with no effect on the
            transport.
        mux_gain: Scalar matched transport gain (copy/transport factor).
    """

    # --- Fan-in (design only) ---
    select_num: int

    # --- Gain ---
    mux_gain: float

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        self.validate_fan_in()
        self.validate_gain()

    def validate_fan_in(self) -> None:
        self._require_pos(self.select_num, "select_num")

    def validate_gain(self) -> None:
        self._require_pos(self.mux_gain, "mux_gain")


@dataclass(frozen=True)
class CurrentMuxPolicy(AnalogPolicy):
    """Abstract marker for CurrentMux nonideality policy — no sources."""


class CurrentMux(AnalogBase[CurrentMuxConfig, CurrentMuxPolicy]):
    """Ideal N:1 time-share current mux — identity·gain transport.

    Args:
        config: Concrete configuration dataclass.
        policy: Per-source nonideality enable flags.
        inst_shape: Per-instance fabrication shape.
        dtype: Tensor dtype for internal buffers.
        T__K: Operating temperature.
    """

    # Non-reporter: embedded primitive whose static PPA rolls up to the owning block.
    is_profile_target: ClassVar[bool] = False

    def __init__(
        self,
        *,
        config: CurrentMuxConfig,
        policy: CurrentMuxPolicy,
        inst_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
    ) -> None:
        super().__init__(config=config, policy=policy, inst_shape=inst_shape)
        self.dtype = dtype
        self.T__K = T__K

    def _sample_fabricate_mismatch(self) -> None:
        pass  # ideal identity-gain transport: no static mismatch

    def transport(self, i__uA: Tensor) -> Tensor:
        """Transport one current through the shared lane at the configured gain.

        Args:
            i__uA: Per-column input current.

        Returns:
            Lane output current ``mux_gain * i__uA``.
        """
        return self.config.mux_gain * i__uA
