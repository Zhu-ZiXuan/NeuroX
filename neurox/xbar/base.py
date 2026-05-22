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

from neurox.common.mixin import FabricateMixin, ProfileMixin, RegistryMixin, ValidateMixin

if TYPE_CHECKING:
    from .ideal import IdealXbar


@dataclass(frozen=True)
class XbarRescaleEntry(ValidateMixin):
    """One row of the ``(adc_mode, adc_bits) -> rf`` lookup table.

    Attributes:
        adc_mode: ADC operating-point index (e.g. multi-V_ref SAR
            reference selection).
        adc_bits: Active ADC bit width.
        rf: Rescale factor such that ``floor(M_ideal * rf) == code``
            where ``M_ideal`` is the ideal tile-level output value.
    """

    adc_mode: int
    adc_bits: int
    rf: float

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        self._require_nonneg(self.adc_mode, "adc_mode")
        self._require_nonneg(self.adc_bits, "adc_bits")


@dataclass(frozen=True)
class XbarConfig(ValidateMixin):
    """Geometry, runtime operating point and PPA for one xbar tile.

    Attributes:
        col_num: Number of columns per tile (cells aggregating to
            one output).
        row_num: Number of rows per tile (cells sharing one input).
        adc_mode: Runtime ADC operating-point index; also selects
            the active entry in :attr:`output_rescale_factors`.
        adc_bits: Runtime ADC bit width; second key for the rescale
            lookup.
        output_rescale_factors: Externally-calibrated ``(adc_mode,
            adc_bits) -> rf`` entries; ``floor(M_ideal · rf) == code``.
        latency_per_op__ns: Array read latency per op [ns].
        leakage_per_inst__uW: Static leakage per tile instance [uW].
        area_per_inst__um2: Silicon area per tile instance [μm²].
    """

    col_num: int
    row_num: int

    adc_mode: int
    adc_bits: int
    output_rescale_factors: tuple[XbarRescaleEntry, ...]

    latency_per_op__ns: float
    leakage_per_inst__uW: float
    area_per_inst__um2: float

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        self.validate_geometry()
        self.validate_adc_mode()
        self.validate_ppa()

    def validate_geometry(self) -> None:
        # IR-drop solvers assume at least two nodes per wire.
        if not (self.col_num > 1):
            raise ValueError(f"require: col_num ({self.col_num}) > 1")
        if not (self.row_num > 1):
            raise ValueError(f"require: row_num ({self.row_num}) > 1")

    def validate_adc_mode(self) -> None:
        self._require_nonneg(self.adc_mode, "adc_mode")
        self._require_nonneg(self.adc_bits, "adc_bits")

    def validate_ppa(self) -> None:
        self._require_nonneg(self.area_per_inst__um2, "area_per_inst__um2")
        self._require_nonneg(self.leakage_per_inst__uW, "leakage_per_inst__uW")
        self._require_nonneg(self.latency_per_op__ns, "latency_per_op__ns")


class Xbar(FabricateMixin, nn.Module, ProfileMixin, RegistryMixin[type["XbarConfig"], "Xbar"], ABC):
    """Abstract base class for a physical crossbar tile.

    Args:
        cfg: Tile geometry, runtime ADC operating point, and PPA.
        name: Hierarchical instance name used by the profiler.
        w_layout_shape: Full xbar-native digit-tensor shape
            ``(*prefix, data_num, digit_num, row_num)`` the tile will
            receive in :meth:`program`. Per-instance multiplicity is
            ``w_layout_shape[:-3]``.
        dtype: Tensor dtype for internal buffers.
        T__K: Operating temperature [K].
    """

    cfg: XbarConfig
    T__K: float
    dtype: torch.dtype

    def __init__(
        self,
        *,
        cfg: XbarConfig,
        name: str,
        w_layout_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
    ) -> None:
        nn.Module.__init__(self)
        ProfileMixin.__init__(self, name)
        self.cfg = cfg
        self.T__K = T__K
        self.dtype = dtype

        if len(w_layout_shape) < 3:
            raise ValueError(
                f"w_layout_shape must have at least 3 trailing dims (data_num, digit_num, row_num); got {w_layout_shape}"
            )
        self._w_layout_shape = tuple(w_layout_shape)
        self._inst_shape = self._w_layout_shape[:-3]

        self.col_num = cfg.col_num
        self.row_num = cfg.row_num
        self._adc_mode = cfg.adc_mode
        self._adc_bits = cfg.adc_bits
        self._rescale_lut = {(e.adc_mode, e.adc_bits): e.rf for e in cfg.output_rescale_factors}

        # `_log_static` is called by the concrete subclass at the end of its
        # ``__init__`` — base does not call to avoid double-recording.

    @classmethod
    def from_config(
        cls,
        *,
        cfg: XbarConfig,
        name: str,
        w_layout_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
    ) -> Xbar:
        """Build the concrete impl registered for ``type(cfg)``."""
        impl = cls._lookup_impl(type(cfg))
        return impl(cfg=cfg, name=name, w_layout_shape=w_layout_shape, dtype=dtype, T__K=T__K)

    # ----- PPA properties (delegated to the immutable config) -----

    @property
    def area_per_inst__um2(self) -> float:
        """Area per instance in um2."""
        return self.cfg.area_per_inst__um2

    @property
    def leakage_per_inst__uW(self) -> float:
        """Leakage per instance in uW."""
        return self.cfg.leakage_per_inst__uW

    @property
    def latency_per_op__ns(self) -> float:
        """Latency per op in ns."""
        return self.cfg.latency_per_op__ns

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

    # ----- Output rescale -----

    @property
    def output_rescale_factor(self) -> float:
        """Rescale factor for the active ``(adc_mode, adc_bits)``.

        Raises:
            KeyError: When no entry matches the active operating point.
        """
        key = (self._adc_mode, self._adc_bits)
        return self._rescale_lut[key]

    # ----- Lifecycle -----

    @abstractmethod
    def program(self, w: Tensor) -> None:
        """Write the tile's owned device buffers from a digit tensor.

        Args:
            w: Integer digit tensor whose shape matches
                :attr:`_w_layout_shape` —
                ``(*prefix, data_num, digit_num, row_num)`` where
                ``digit_num == self.w_digit_count``. Entries must lie
                in :attr:`w_digit_range`.
        """
        raise NotImplementedError

    @abstractmethod
    def vec_mat_mul(self, x: Tensor) -> Tensor:
        """Run one analog VMM through the tile.

        Args:
            x: Activation tensor with primitive trailing
                ``[row_num]``. Entries must lie in :attr:`x_range`;
                leading dims are broadcast-only.

        Returns:
            Output tensor with primitive trailing ``[data_num]``.
        """
        raise NotImplementedError

    def to_ideal(self) -> IdealXbar:
        """Return the lossless :class:`IdealXbar` counterpart of this tile.

        The new ``IdealXbar`` inherits this tile's per-instance
        multiplicity; the primitive trailing dims of its
        ``w_layout_shape`` are ``(col_num, w_digit_count, row_num)``.
        ``IdealXbar.to_ideal`` overrides this to ``return self``.
        """
        # Local import — the ``ideal`` module imports from this file,
        # so the symbol is only safe to resolve at call time.
        from .ideal import IdealXbar, IdealXbarConfig

        base_kwargs = {f.name: getattr(self.cfg, f.name) for f in fields(XbarConfig)}
        ideal_cfg = IdealXbarConfig(
            **base_kwargs,
            x_range=self.x_range,
            w_digit_count=self.w_digit_count,
            w_digit_radix=self.w_digit_radix,
            w_digit_range=self.w_digit_range,
        )
        ideal_layout_shape = (*self._inst_shape, self.cfg.col_num, self.w_digit_count, self.cfg.row_num)
        return IdealXbar(
            cfg=ideal_cfg,
            name=self.qualified_name,
            w_layout_shape=ideal_layout_shape,
            dtype=self.dtype,
            T__K=self.T__K,
        )
