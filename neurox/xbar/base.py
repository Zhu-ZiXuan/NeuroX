"""Abstract physical-crossbar primitive.

See also:
    docs/dev/modules/xbar/base.md
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch import Tensor

from neurox.analog.adc import AdcCalibrationRecord, AdcOperationPoint
from neurox.common.mixin import FabricateMixin, ProfileMixin, RegistryMixin, ValidateMixin

if TYPE_CHECKING:
    from .ideal import IdealXbar


@dataclass(frozen=True)
class XbarConfig(ValidateMixin):
    """Geometry, runtime operating point and PPA for one xbar tile.

    Attributes:
        col_num: Number of columns per tile (cells aggregating to
            one output).
        row_num: Number of rows per tile (cells sharing one input).
        adc_calibration: Externally-calibrated ``(adc_mode, adc_bits) →
            rescale_factor`` records; ``M_ideal ≈ code · rescale_factor``
            (quantize is ``code = floor(M_ideal / rescale_factor)``).
            Lists the set of ADC operating points the tile supports.
        latency_per_op__ns: Array read latency per op [ns].
        leakage_per_inst__uW: Static leakage per tile instance [uW].
        area_per_inst__um2: Silicon area per tile instance [μm²].
    """

    col_num: int
    row_num: int

    adc_calibration: tuple[AdcCalibrationRecord, ...]

    latency_per_op__ns: float
    leakage_per_inst__uW: float
    area_per_inst__um2: float

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        self.validate_geometry()
        self.validate_adc_calibration()
        self.validate_ppa()

    def validate_geometry(self) -> None:
        # IR-drop solvers assume at least two nodes per wire.
        if not (self.col_num > 1):
            raise ValueError(f"require: col_num ({self.col_num}) > 1")
        if not (self.row_num > 1):
            raise ValueError(f"require: row_num ({self.row_num}) > 1")

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

    def validate_adc_calibration(self) -> None:
        if len(self.adc_calibration) == 0:
            raise ValueError("require: adc_calibration must contain at least one entry")
        seen: set[tuple[int, int]] = set()
        for entry in self.adc_calibration:
            key = (entry.adc_mode, entry.adc_bits)
            if key in seen:
                raise ValueError(
                    f"adc_calibration has duplicate (adc_mode, adc_bits)={key}"
                )
            seen.add(key)
            if not (entry.rescale_factor > 0.0):
                raise ValueError(
                    f"require: rescale_factor ({entry.rescale_factor}) > 0 for "
                    f"(adc_mode={entry.adc_mode}, adc_bits={entry.adc_bits})"
                )

    def validate_ppa(self) -> None:
        self._require_nonneg(self.area_per_inst__um2, "area_per_inst__um2")
        self._require_nonneg(self.leakage_per_inst__uW, "leakage_per_inst__uW")
        self._require_nonneg(self.latency_per_op__ns, "latency_per_op__ns")


@dataclass(frozen=True)
class XbarPolicy:
    """Abstract marker base for Xbar-family nonideality policies."""


class Xbar(FabricateMixin, nn.Module, ProfileMixin, RegistryMixin[type["XbarConfig"], "Xbar"], ABC):
    """Abstract base class for a physical crossbar tile.

    Args:
        config: Concrete configuration dataclass.
        policy: Composite nonideality policy.
        name: Hierarchical instance name used by the profiler.
        inst_shape: Per-instance multiplicity prefix; trailing
            ``(col_num, w_digit_count, row_num)`` is derived from config.
        dtype: Tensor dtype for internal buffers.
        T__K: Operating temperature [K].
    """

    config: XbarConfig
    policy: XbarPolicy
    T__K: float
    dtype: torch.dtype

    def __init__(
        self,
        *,
        config: XbarConfig,
        policy: XbarPolicy,
        name: str,
        inst_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
    ) -> None:
        nn.Module.__init__(self)
        ProfileMixin.__init__(self, name)
        self.config = config
        self.policy = policy
        self.T__K = T__K
        self.dtype = dtype

        self._inst_shape = inst_shape

        self.col_num = config.col_num
        self.row_num = config.row_num
        self._rescale_lut: dict[AdcOperationPoint, float] = {
            AdcOperationPoint(adc_mode=e.adc_mode, adc_bits=e.adc_bits): e.rescale_factor
            for e in config.adc_calibration
        }

        # `_log_static` is called by the concrete subclass at the end of its
        # ``__init__`` — base does not call to avoid double-recording.

    @property
    def _w_layout_shape(self) -> tuple[int, ...]:
        """Full digit-tensor shape ``(*inst_shape, col_num, w_digit_count, row_num)``."""
        return (*self._inst_shape, self.col_num, self.w_digit_count, self.row_num)

    @classmethod
    def from_config(
        cls,
        *,
        config: XbarConfig,
        policy: XbarPolicy,
        name: str,
        inst_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
    ) -> Xbar:
        """Build the concrete impl registered for ``type(config)``."""
        impl = cls._lookup_impl(type(config))
        return impl(
            config=config,
            policy=policy,
            name=name,
            inst_shape=inst_shape,
            dtype=dtype,
            T__K=T__K,
        )

    # ----- PPA properties (delegated to the immutable config) -----

    @property
    def area_per_inst__um2(self) -> float:
        """Silicon area per instance [um^2]."""
        return self.config.area_per_inst__um2

    @property
    def leakage_per_inst__uW(self) -> float:
        """Static leakage per instance [uW]."""
        return self.config.leakage_per_inst__uW

    @property
    def latency_per_op__ns(self) -> float:
        """Latency per op [ns]."""
        return self.config.latency_per_op__ns

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
        """Number of ADC operating points the tile supports."""
        raise NotImplementedError

    @property
    @abstractmethod
    def adc_max_bits(self) -> int:
        """Maximum ``adc_bits`` value the tile's ADC supports."""
        raise NotImplementedError

    def adc_rescale_factor(self, adc_operation_point: AdcOperationPoint) -> float:
        """Recovery-side multiplier for ``adc_operation_point``: ``M_ideal ≈ code · rescale_factor``.

        The relationship is strictly proportional (no intercept) by design —
        differential ADC zeroes ``v_diff`` at ``v_data == v_ref``, so chip
        offsets show up as noise, not as a constant bias.

        Raises:
            KeyError: When ``adc_operation_point`` is absent from the
                calibrated LUT. The error message lists the available
                operating points so the caller can spot misconfigured
                ``adc_calibration`` tables at a glance.
        """
        try:
            return self._rescale_lut[adc_operation_point]
        except KeyError:
            available = sorted(
                (op.adc_mode, op.adc_bits) for op in self._rescale_lut
            )
            raise KeyError(
                f"{adc_operation_point} not in adc_calibration; "
                f"available (adc_mode, adc_bits): {available}"
            ) from None

    # ----- Lifecycle -----

    @abstractmethod
    def program(self, w: Tensor) -> None:
        """Write the tile's owned device buffers from a digit tensor.

        Args:
            w: Integer digit tensor whose shape matches
                :attr:`_w_layout_shape` —
                ``(*inst_shape, col_num, w_digit_count, row_num)``.
                Entries must lie in :attr:`w_digit_range`.
        """
        raise NotImplementedError

    @abstractmethod
    def vec_mat_mul(self, x: Tensor, *, adc_operation_point: AdcOperationPoint) -> Tensor:
        """Run one analog VMM through the tile.

        Args:
            x: Activation tensor with primitive trailing
                ``[row_num]``. Entries must lie in :attr:`x_range`;
                leading dims are broadcast-only.
            adc_operation_point: Runtime ADC operating point.

        Returns:
            Output tensor with primitive trailing ``[col_num]``.
        """
        raise NotImplementedError

    def to_ideal(self) -> IdealXbar:
        """Return the lossless :class:`IdealXbar` counterpart of this tile.

        The new ``IdealXbar`` inherits this tile's per-instance
        multiplicity and ADC operating-point metadata. ``IdealXbar.to_ideal``
        overrides this to ``return self``.
        """
        # Local import — the ``ideal`` module imports from this file,
        # so the symbol is only safe to resolve at call time.
        from .ideal import IdealXbar, IdealXbarConfig, IdealXbarPolicy

        base_kwargs = {f.name: getattr(self.config, f.name) for f in fields(XbarConfig)}
        ideal_config = IdealXbarConfig(
            **base_kwargs,
            x_range=self.x_range,
            w_digit_count=self.w_digit_count,
            w_digit_radix=self.w_digit_radix,
            w_digit_range=self.w_digit_range,
            adc_mode_num=self.adc_mode_num,
            adc_max_bits=self.adc_max_bits,
        )
        return IdealXbar(
            config=ideal_config,
            policy=IdealXbarPolicy(),
            name=self.qualified_name,
            inst_shape=self._inst_shape,
            dtype=self.dtype,
            T__K=self.T__K,
        )
