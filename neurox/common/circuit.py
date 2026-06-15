"""Common base for every electrical circuit module.

See also:
    docs/internals/config_and_construction.md
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Generic, TypeVar

import torch.nn as nn

from .mixin import FabricateMixin, ProfileMixin, ValidateMixin


@dataclass(frozen=True)
class CircuitConfig(ValidateMixin):
    """Static PPA fields common to every electrical circuit's config.

    Devices (RRAM, NMOS, ...) are physical primitives whose contribution
    rolls up into the owning circuit's PPA and do NOT use this base.

    Per-op latency is NOT a base field: leaves with a dynamic profile
    model compute the value in their primary method and emit it via
    ``_log_latency(latency)`` (paired with
    ``_log_dynamic_energy(dynamic_energy__fJ)`` when the leaf also
    emits dynamic energy) — fixed-latency leaves declare
    ``latency_per_op__ns`` on their own config; parametric leaves
    (e.g. SAR ADC) derive it from runtime parameters. Leaves without
    a dynamic model emit nothing and carry no ``latency_per_op__ns``.

    Attributes:
        area_per_inst__um2: Silicon area per fabricated instance [um²].
        leakage_per_inst__uW: Static leakage per instance [uW].
    """

    area_per_inst__um2: float
    leakage_per_inst__uW: float

    def validate_ppa(self) -> None:
        self._require_nonneg(self.area_per_inst__um2, "area_per_inst__um2")
        self._require_nonneg(self.leakage_per_inst__uW, "leakage_per_inst__uW")


ConfigT = TypeVar("ConfigT", bound=CircuitConfig)


class CircuitBase(FabricateMixin, ProfileMixin, nn.Module, Generic[ConfigT]):
    """Common base for every electrical circuit module.

    Composes four orthogonal capabilities:
      - ``FabricateMixin``: fab-cascade hook
      - ``ProfileMixin``: profile event emission
      - ``nn.Module``: PyTorch hook
      - typed static-PPA surface backed by ``self.config``

    Subclasses parameterise the config type via the generic argument
    (e.g. ``Driver(CircuitBase[DriverConfig])``); mypy narrows
    ``self.config`` accordingly so ``self.config.drive_value`` etc. type
    correctly without per-subclass forward declaration.

    Per-instance static accessors (area / leakage) read from
    ``self.config``. Derived inst-aggregate properties
    (``inst_count`` / ``inst_area__um2`` / ``inst_leakage__uW``) compute
    on access — no cache. Per-op latency is NOT a base concern: a leaf
    that has a dynamic profile model assembles its own ``latency__ns``
    tensor in the primary method and emits it via
    ``_log_latency(latency)`` (paired with
    ``_log_dynamic_energy(dynamic_energy__fJ)`` when the leaf also
    emits energy) — fixed-latency leaves read
    ``self.config.latency_per_op__ns``; parametric leaves derive it
    from runtime parameters. Dynamics-less leaves emit nothing.
    """

    config: ConfigT
    _inst_shape: tuple[int, ...]

    def __init__(self, *, config: ConfigT, name: str, inst_shape: tuple[int, ...]) -> None:
        """Initialise nn.Module + ProfileMixin and bind config / inst_shape.

        Args:
            config: Concrete configuration dataclass.
            name: Hierarchical instance name used by the profiler.
            inst_shape: Per-instance fabrication shape.
        """
        nn.Module.__init__(self)
        ProfileMixin.__init__(self, name)
        self.config = config
        self._inst_shape = inst_shape

    # --- Per-instance accessors (from config) ---

    @property
    def area_per_inst__um2(self) -> float:
        """Silicon area per instance [um²]."""
        return self.config.area_per_inst__um2

    @property
    def leakage_per_inst__uW(self) -> float:
        """Static leakage per instance [uW]."""
        return self.config.leakage_per_inst__uW

    # --- Inst-level aggregates (derived) ---

    @property
    def inst_shape(self) -> tuple[int, ...]:
        """Per-instance fabrication shape declared at construction."""
        return self._inst_shape

    @property
    def inst_count(self) -> int:
        """Total fabrication instance count — ``prod(self._inst_shape)``."""
        return math.prod(self._inst_shape)

    @property
    def inst_area__um2(self) -> float:
        """Silicon area for the full fabrication multiplicity [um²]."""
        return self.area_per_inst__um2 * self.inst_count

    @property
    def inst_leakage__uW(self) -> float:
        """Static leakage for the full fabrication multiplicity [uW]."""
        return self.leakage_per_inst__uW * self.inst_count
