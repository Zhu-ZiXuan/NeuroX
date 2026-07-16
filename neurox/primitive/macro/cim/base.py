"""Abstract physical-crossbar primitive.

See also:
    docs/reference/primitive/macro/cim/README.md
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from neurox.common import ConfigBase, ModuleBase, PolicyBase
from neurox.common.mixin import RegistryMixin
from neurox.primitive.analog.adc_common import AdcOperationPoint

if TYPE_CHECKING:
    from .ideal import IdealCimMacro


@dataclass(frozen=True)
class CimMacroConfig(ConfigBase):
    """Geometry and PPA shared by every xbar tile.

    Attributes:
        area_per_inst__um2: Silicon area per fabricated instance.
        leakage_per_inst__uW: Static leakage per instance.
        col_num: Number of columns per tile (cells aggregating to
            one output).
        row_num: Number of rows per tile (cells sharing one input).
    """

    area_per_inst__um2: float
    leakage_per_inst__uW: float
    col_num: int
    row_num: int

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        self.validate_geometry()
        self.validate_ppa()

    def validate_geometry(self) -> None:
        # IR-drop solvers assume at least two nodes per wire.
        if not (self.col_num > 1):
            raise ValueError(f"require: col_num ({self.col_num}) > 1")
        if not (self.row_num > 1):
            raise ValueError(f"require: row_num ({self.row_num}) > 1")

    def validate_ppa(self) -> None:
        self._require_non_neg(self.area_per_inst__um2, "area_per_inst__um2")
        self._require_non_neg(self.leakage_per_inst__uW, "leakage_per_inst__uW")

    def validate_value_grid(self) -> None:
        """Reject degenerate single-point ranges that collapse rescale math."""
        if hasattr(self, "x_range"):
            x_lo, x_hi = self.x_range
            if x_lo == 0 and x_hi == 0:
                raise ValueError("require: x_range cannot be (0, 0) — collapses rescale math")
        if hasattr(self, "w_digit_range"):
            d_lo, d_hi = self.w_digit_range
            if d_lo == 0 and d_hi == 0:
                raise ValueError("require: w_digit_range cannot be (0, 0) — collapses rescale math")


@dataclass(frozen=True)
class CimMacroPolicy(PolicyBase):
    """Abstract marker base for CimMacro-family nonideality policies."""


class CimMacro(
    ModuleBase[CimMacroConfig, CimMacroPolicy],
    RegistryMixin[type["CimMacroConfig"], "CimMacro"],
):
    """Abstract base class for a physical crossbar tile.

    Args:
        config: Concrete configuration dataclass.
        policy: Composite nonideality policy.
        inst_shape: Per-instance multiplicity prefix; trailing
            ``(col_num, w_digit_count, row_num)`` is derived from config.
        dtype: Tensor dtype for internal buffers.
        T__K: Operating temperature.
    """

    policy: CimMacroPolicy
    T__K: float
    dtype: torch.dtype

    def __init__(
        self,
        *,
        config: CimMacroConfig,
        policy: CimMacroPolicy,
        inst_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
    ) -> None:
        super().__init__(config=config, policy=policy, inst_shape=inst_shape)
        self.T__K = T__K
        self.dtype = dtype

        self.col_num = config.col_num
        self.row_num = config.row_num

    @property
    def _w_layout_shape(self) -> tuple[int, ...]:
        """Full digit-tensor shape ``(*inst_shape, col_num, w_digit_count, row_num)``."""
        return (*self._inst_shape, self.col_num, self.w_digit_count, self.row_num)

    @classmethod
    def from_config(
        cls,
        *,
        config: CimMacroConfig,
        policy: CimMacroPolicy,
        inst_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
    ) -> CimMacro:
        """Build the concrete impl registered for ``type(config)``."""
        impl = cls._lookup_impl(type(config))
        return impl(
            config=config,
            policy=policy,
            inst_shape=inst_shape,
            dtype=dtype,
            T__K=T__K,
        )

    def _sample_fabricate_mismatch(self) -> None:
        pass  # container: cell / peripheral mismatch is sampled through the cascade

    # ----- Value-domain semantics (abstract) -----

    @property
    @abstractmethod
    def x_range(self) -> tuple[int, int]:
        """Inclusive single-cycle integer input range the tile accepts."""
        raise NotImplementedError

    @property
    @abstractmethod
    def w_digit_count(self) -> int:
        """Number of digits per ``w`` inside this tile."""
        raise NotImplementedError

    @property
    @abstractmethod
    def w_digit_radix(self) -> int:
        """Positional base ``r`` of the in-tile digit combination."""
        raise NotImplementedError

    @property
    @abstractmethod
    def w_digit_range(self) -> tuple[int, int]:
        """Inclusive integer range a single digit cell can carry physically.

        Set by the array structure and the per-cell device-state count.
        """
        raise NotImplementedError

    # ----- ADC operating-point surface -----

    @property
    @abstractmethod
    def adc_mode_num(self) -> int:
        """Number of supported ADC operating points; valid ``adc_mode`` values are ``[0, adc_mode_num)``."""
        raise NotImplementedError

    @property
    @abstractmethod
    def adc_max_bits(self) -> int:
        """Maximum supported ``adc_bits`` value."""
        raise NotImplementedError

    @abstractmethod
    def adc_rescale_factor(self, adc_operation_point: AdcOperationPoint) -> float:
        """Rescale factor for ``adc_operation_point``; raises ``KeyError`` if uncalibrated."""
        raise NotImplementedError

    # ----- Lifecycle -----

    @abstractmethod
    def program(self, w: Tensor) -> None:
        """Write the tile's owned device buffers from one xbar-native digit tensor.

        Args:
            w: Integer digit tensor whose shape matches
                ``self._w_layout_shape = (*inst_shape, col_num, w_digit_count, row_num)``.
                Entries must lie in :attr:`w_digit_range`.
        """
        raise NotImplementedError

    @abstractmethod
    def vec_mat_mul(self, x: Tensor, *, adc_operation_point: AdcOperationPoint) -> Tensor:
        """Run one analog VMM through the tile.

        Args:
            x: Activation tensor with primitive trailing ``[row_num]``.
                Entries must lie in :attr:`x_range`; leading dims are
                broadcast-only.
            adc_operation_point: Runtime ADC operating point.

        Returns:
            ADC-code tensor with primitive trailing ``[col_num]``.
        """
        raise NotImplementedError

    def to_ideal(self) -> IdealCimMacro:
        """Return the lossless ideal twin of this tile.

        The twin inherits this tile's per-instance multiplicity and ADC
        operating-point metadata.
        """
        # Local import — the ``ideal`` module imports from this file,
        # so the symbol is only safe to resolve at call time.
        from .ideal import IdealCimMacro, IdealCimMacroConfig, IdealCimMacroPolicy

        base_kwargs = {f.name: getattr(self.config, f.name) for f in fields(CimMacroConfig)}
        ideal_config = IdealCimMacroConfig(
            **base_kwargs,
            x_range=self.x_range,
            w_digit_count=self.w_digit_count,
            w_digit_radix=self.w_digit_radix,
            w_digit_range=self.w_digit_range,
            adc_mode_num=self.adc_mode_num,
            adc_max_bits=self.adc_max_bits,
        )
        return IdealCimMacro(
            config=ideal_config,
            policy=IdealCimMacroPolicy(),
            inst_shape=self._inst_shape,
            dtype=self.dtype,
            T__K=self.T__K,
        )
