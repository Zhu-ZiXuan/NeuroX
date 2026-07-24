"""Abstract physical-crossbar primitive.

See also:
    docs/internals/primitive/macro/cim/base.md
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Generic, TypeVar

import torch
from torch import Tensor

from neurox.common import ConfigBase, ModuleBase, PolicyBase
from neurox.common.mixin import RegistryMixin

if TYPE_CHECKING:
    from .ideal import IdealCimMacro


@dataclass(frozen=True)
class CimMacroConfig(ConfigBase, ABC):
    """Geometry and PPA shared by every xbar tile.

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
        if not (1 <= self.active_row_num <= self.row_num):
            raise ValueError(f"require: 1 <= active_row_num ({self.active_row_num}) <= row_num ({self.row_num})")
        # No divisibility guard here: a tile may have any row_num / active_row_num
        # ratio. The engine covers every row with a ceil number of sub-phases (the
        # last block partially active). Uniform row-blocking — where a divisor is
        # required so every sub-phase reads the same dot-product dynamic range — is
        # an operator-layer contract enforced by the consuming unit (LinearCimUnit).

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
    """Abstract base class for a physical crossbar tile.

    Args:
        config: Concrete configuration dataclass.
        policy: Composite nonideality policy.
        inst_shape: Per-instance multiplicity prefix; trailing
            ``(col_num, w_digit_count, row_num)`` is derived from config.
        dtype: Tensor dtype for internal buffers.
        T__K: Operating temperature.
    """

    T__K: float
    dtype: torch.dtype

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
        self.T__K = T__K
        self.dtype = dtype

        self.col_num = config.col_num
        self.row_num = config.row_num

    @property
    def max_active_rows(self) -> int:
        """Maximum simultaneously active word lines per conversion; the single sub-phase query for upper layers."""
        return self.config.active_row_num

    @property
    def _w_layout_shape(self) -> tuple[int, ...]:
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

    # ----- Intra-tile serialization helpers -----
    # Serial-vs-parallel semantics live in tensor shape: axes matching a
    # stage's fabricated ``inst_shape`` are parallel circuit copies; every
    # other axis is time-serial on that hardware.

    @staticmethod
    def _split_col_lanes(t: Tensor, *, col_per_lane: int) -> Tensor:
        """Split the trailing column axis into ``(lane_num, col_per_lane)``.

        The lane axis aligns with fabricated instance axes (parallel
        circuit copies); the trailing axis is time-serial on each lane,
        with ``lane = col // col_per_lane``. Requires exact divisibility.
        """
        if t.shape[-1] % col_per_lane != 0:
            raise ValueError(f"require: trailing col axis ({t.shape[-1]}) % col_per_lane ({col_per_lane}) == 0")
        return t.unflatten(-1, (-1, col_per_lane))

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
    def adc_rescale_factor(self, *, adc_mode: int, adc_bits: int) -> float:
        """Rescale factor for ``(adc_mode, adc_bits)``; raises ``KeyError`` if uncalibrated.

        Args:
            adc_mode: ADC operating-point index selecting the reference
                row / tap set; valid values are ``[0, adc_mode_num)``.
            adc_bits: ADC resolution [bits] the conversion runs at.
        """
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
    def vec_mat_mul(self, x: Tensor, *, adc_mode: int, adc_bits: int) -> Tensor:
        """Run one independent analog conversion per WL plane through the tile.

        The macro boundary is digital -> DAC -> analog -> ADC -> digital,
        encapsulating the minimal analog chain.

        Args:
            x: WL plane tensor with primitive trailing ``[row_num]``;
                every leading axis is anonymous broadcast batch — the
                macro never inspects, reorders, or reduces leading axes.
                Rows outside the caller's active window (at most
                :attr:`max_active_rows` live rows per plane) must arrive
                zeroed (WL off). Entries must lie in :attr:`x_range`.
            adc_mode: ADC operating-point index selecting the reference
                row / tap set; valid values are ``[0, adc_mode_num)``.
            adc_bits: ADC resolution [bits] the conversion runs at.

        Returns:
            ADC-code tensor with the same leading order and primitive
            trailing ``[col_num]``. The macro performs no phase
            expansion, no trailing movedim, and no accumulation; any
            routing or accumulation of leading axes is the caller's
            digital-domain decision.
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
            inst_shape=self.inst_shape,
            dtype=self.dtype,
            T__K=self.T__K,
        )
