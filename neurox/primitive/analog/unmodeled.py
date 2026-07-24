"""Unmodeled circuit block — static PPA seat only, no functional model.

See also:
    docs/reference/primitive/analog/unmodeled.md
"""

from dataclasses import dataclass

import torch

from neurox.primitive.analog.base import AnalogBase, AnalogConfig, AnalogPolicy


@dataclass(frozen=True, kw_only=True)
class UnmodeledBlockConfig(AnalogConfig):
    """Immutable configuration for :class:`UnmodeledBlock`.

    Attributes:
        area_per_inst__um2: Silicon area per fabricated instance.
        leakage_per_inst__uW: Static leakage per instance; carries the
            block's whole standing bias power.
    """

    # --- Static PPA ---
    area_per_inst__um2: float
    leakage_per_inst__uW: float

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        self.validate_ppa()

    def validate_ppa(self) -> None:
        self._require_non_neg(self.area_per_inst__um2, "area_per_inst__um2")
        self._require_non_neg(self.leakage_per_inst__uW, "leakage_per_inst__uW")


@dataclass(frozen=True)
class UnmodeledBlockPolicy(AnalogPolicy):
    """Abstract marker for UnmodeledBlock nonideality policy — no sources."""


class UnmodeledBlock(AnalogBase[UnmodeledBlockConfig, UnmodeledBlockPolicy]):
    """A real circuit block modeled only as static PPA — functionally unmodeled.

    It carries silicon area plus the block's whole standing bias power (folded
    into ``leakage_per_inst__uW``) as a profiler-visible static seat, and does
    nothing else: no functional method, no forward, no state, no nonideality.
    Its per-op dynamic energy, if any, is not billed here — a composing macro
    that knows the block's activity bills it through a profiler channel keyed to
    this block's role.

    Use it to reserve the PPA seat of a block whose internal transfer is out of
    scope (control logic, a fixed bias network) while keeping its power on the
    ledger.

    Args:
        config: Concrete configuration dataclass.
        policy: Per-source nonideality enable flags.
        inst_shape: Per-instance fabrication shape.
        dtype: Tensor dtype for internal buffers.
        T__K: Operating temperature.
    """

    def __init__(
        self,
        *,
        config: UnmodeledBlockConfig,
        policy: UnmodeledBlockPolicy,
        inst_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
    ) -> None:
        super().__init__(config=config, policy=policy, inst_shape=inst_shape)
        self._area_per_inst__um2 = config.area_per_inst__um2
        self._leakage_per_inst__uW = config.leakage_per_inst__uW
        self.dtype = dtype
        self.T__K = T__K

    def _sample_fabricate_mismatch(self) -> None:
        pass  # functionally unmodeled: no static mismatch state
