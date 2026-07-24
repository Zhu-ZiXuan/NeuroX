"""Abstract CIM-macro primitive.

See also:
    docs/internals/primitive/macro/cim/base.md
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import fields
from typing import TYPE_CHECKING, Generic, TypeVar

import torch
from torch import Tensor

from neurox.common import ConfigBase, ModuleBase, PolicyBase
from neurox.common.mixin import RegistryMixin

if TYPE_CHECKING:
    from .ideal import IdealCimMacro


class CimMacroConfig(ConfigBase, ABC):
    """Geometry and PPA shared by every CIM macro.

    Attributes:
        area_per_inst__um2: Silicon area per fabricated instance.
        leakage_per_inst__uW: Static leakage per instance.
        col_num: Number of columns per tile (cells aggregating to
            one output).
        row_num: Number of rows per tile (cells sharing one input).
        active_row_num: Maximum rows simultaneously activated per
            conversion; sets the per-conversion analog dot-product
            dynamic range and therefore the ADC calibration.
    """

    area_per_inst__um2: float
    leakage_per_inst__uW: float
    col_num: int
    row_num: int
    active_row_num: int

    def validate(self) -> None:
        # --- Geometry ---

        # Physical array solvers require at least two nodes per wire.
        if not (self.col_num > 1):
            raise ValueError(f"require: col_num ({self.col_num}) > 1")
        if not (self.row_num > 1):
            raise ValueError(f"require: row_num ({self.row_num}) > 1")
        if not (1 <= self.active_row_num <= self.row_num):
            raise ValueError(f"require: 1 <= active_row_num ({self.active_row_num}) <= row_num ({self.row_num})")

        # --- PPA ---

        self._require_non_neg(self.area_per_inst__um2, "area_per_inst__um2")
        self._require_non_neg(self.leakage_per_inst__uW, "leakage_per_inst__uW")


class CimMacroPolicy(PolicyBase, ABC):
    """Abstract marker base for CimMacro-family nonideality policies."""


ConfigT = TypeVar("ConfigT", bound=CimMacroConfig)
PolicyT = TypeVar("PolicyT", bound=CimMacroPolicy)


class CimMacro(
    ModuleBase[ConfigT, PolicyT],
    RegistryMixin[type["CimMacroConfig"], "CimMacro"],
    Generic[ConfigT, PolicyT],
    ABC,
):
    """Abstract base class for a CIM macro.

    Args:
        config: Concrete configuration dataclass.
        policy: Composite nonideality policy.
        inst_shape: Per-instance multiplicity prefix; trailing
            ``(col_num, w_digit_count, row_num)`` is derived from config.
        dtype: Tensor dtype for internal buffers.
        T__K: Operating temperature.
    """

    def __init__(
        self,
        *,
        config: ConfigT,
        policy: PolicyT,
        inst_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
    ) -> None:
        super().__init__(config=config, policy=policy, inst_shape=inst_shape)
        self._T__K = T__K
        self._dtype = dtype

        self.col_num = config.col_num
        self.row_num = config.row_num

    @property
    def max_active_rows(self) -> int:
        """Maximum simultaneously active word lines per conversion."""
        return self.config.active_row_num

    @property
    def w_layout_shape(self) -> tuple[int, ...]:
        """Full digit-tensor shape ``(*inst_shape, col_num, w_digit_count, row_num)``."""
        return (*self.inst_shape, self.col_num, self.w_digit_count, self.row_num)

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
        """Build the implementation registered for ``type(config)``.

        Args:
            config: Concrete configuration dataclass.
            policy: Composite nonideality policy.
            inst_shape: Per-instance multiplicity prefix.
            dtype: Tensor dtype for internal buffers.
            T__K: Operating temperature.

        Returns:
            Registered CIM macro implementation.
        """
        impl = cls._lookup_impl(type(config))
        return impl(
            config=config,
            policy=policy,
            inst_shape=inst_shape,
            dtype=dtype,
            T__K=T__K,
        )

    def _sample_fabricate_mismatch(self) -> None:
        pass

    @staticmethod
    def _split_col_lanes(t: Tensor, *, col_per_lane: int) -> Tensor:
        """Split the trailing column axis into ``(lane_num, col_per_lane)``.

        Requires exact divisibility.
        """
        if t.shape[-1] % col_per_lane != 0:
            raise ValueError(f"require: trailing col axis ({t.shape[-1]}) % col_per_lane ({col_per_lane}) == 0")
        split: Tensor = t.unflatten(-1, (-1, col_per_lane))
        return split

    @property
    @abstractmethod
    def x_value_range(self) -> tuple[int, int]:
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
    def w_digit_value_range(self) -> tuple[int, int]:
        """Inclusive integer range a single digit cell can carry physically.

        Set by the array structure and the per-cell device-state count.
        """
        raise NotImplementedError

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
    def adc_rescale_factor(self, *, adc_mode: int, adc_bits: int) -> float:
        """Rescale factor for ``(adc_mode, adc_bits)``; raises ``KeyError`` if uncalibrated.

        Args:
            adc_mode: ADC operating-point index selecting the reference
                row / tap set; valid values are ``[0, adc_mode_num)``.
            adc_bits: ADC resolution [bits] the conversion runs at.
        """
        raise NotImplementedError

    @abstractmethod
    def program(self, w: Tensor) -> None:
        """Program the macro from an array-native digit tensor.

        Args:
            w: Integer digit tensor whose shape matches
                ``self.w_layout_shape = (*inst_shape, col_num, w_digit_count, row_num)``.
                Entries must lie in :attr:`w_digit_value_range`.
        """
        raise NotImplementedError

    @abstractmethod
    def vec_mat_mul(self, x: Tensor, *, adc_mode: int, adc_bits: int) -> Tensor:
        """Run one conversion per word-line plane.

        Args:
            x: WL plane tensor with primitive trailing ``[row_num]``;
                leading axes are broadcast batch dimensions. At most
                :attr:`max_active_rows` rows may be nonzero per plane.
                Entries must lie in :attr:`x_value_range`.
            adc_mode: ADC operating-point index selecting the reference
                row / tap set; valid values are ``[0, adc_mode_num)``.
            adc_bits: ADC resolution [bits] the conversion runs at.

        Returns:
            ADC-code tensor with the same leading dimensions and trailing
            ``[col_num]``.
        """
        raise NotImplementedError

    def to_ideal(self) -> IdealCimMacro:
        """Return the lossless ideal twin of this tile.

        The twin inherits this tile's per-instance multiplicity and ADC
        operating-point metadata.
        """
        # Resolve lazily to avoid the module import cycle.
        from .ideal import IdealCimMacro, IdealCimMacroConfig, IdealCimMacroPolicy

        base_kwargs = {f.name: getattr(self.config, f.name) for f in fields(CimMacroConfig)}
        ideal_config = IdealCimMacroConfig(
            **base_kwargs,
            x_value_range=self.x_value_range,
            w_digit_count=self.w_digit_count,
            w_digit_radix=self.w_digit_radix,
            w_digit_value_range=self.w_digit_value_range,
            adc_mode_num=self.adc_mode_num,
            adc_max_bits=self.adc_max_bits,
        )
        return IdealCimMacro(
            config=ideal_config,
            policy=IdealCimMacroPolicy(),
            inst_shape=self.inst_shape,
            dtype=self._dtype,
            T__K=self._T__K,
        )
