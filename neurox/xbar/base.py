"""Abstract physical-crossbar primitive.

See also:
    docs/dev/modules/xbar/base.md
"""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import torch.nn as nn
from torch import Tensor

from neurox.common.validate import ValidateMixin
from neurox.profiler import ProfiledModule


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

    adc_mode: int = 0
    adc_bits: int = 0
    output_rescale_factors: tuple[XbarRescaleEntry, ...] = field(default_factory=tuple)

    latency_per_op__ns: float = 0.0
    leakage_per_inst__uW: float = 0.0
    area_per_inst__um2: float = 0.0

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


class Xbar(nn.Module, ProfiledModule, ABC):
    """Abstract base class for a physical crossbar tile.

    Args:
        cfg: Tile geometry, runtime ADC operating point, and PPA.
        name: Hierarchical instance name used by the profiler.
    """

    def __init__(self, cfg: XbarConfig, *, name: str = "") -> None:
        nn.Module.__init__(self)
        ProfiledModule.__init__(self, name)
        self.config = cfg

        self.col_num = cfg.col_num
        self.row_num = cfg.row_num
        self._adc_mode: int = cfg.adc_mode
        self._adc_bits: int = cfg.adc_bits
        self._rescale_lut: dict[tuple[int, int], float] = {
            (e.adc_mode, e.adc_bits): e.rf for e in cfg.output_rescale_factors
        }

    # ----- PPA properties (delegated to the immutable config) -----

    @property
    def area_per_inst__um2(self) -> float:
        """Area per instance in um2."""
        return self.config.area_per_inst__um2

    @property
    def leakage_per_inst__uW(self) -> float:
        """Leakage per instance in uW."""
        return self.config.leakage_per_inst__uW

    @property
    def latency_per_op__ns(self) -> float:
        """Latency per op in ns."""
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
    def fabricate(self, w: Tensor) -> None:
        """Program the tile's owned device buffers from a digit tensor.

        Args:
            w: Integer digit tensor with primitive trailing
                ``[data_num, digit_num, row_num]`` where
                ``digit_num == self.w_digit_count``. Entries must lie
                in :attr:`w_digit_range`; leading dims are
                broadcast-only.
        """
        raise NotImplementedError

    def _record_xbar_inst_count(self, w: Tensor) -> None:
        """Record the profiler instance count from a fabricated digit tensor."""
        leading = w.shape[:-3] if w.ndim >= 3 else ()
        self._record_inst_count(math.prod(leading))

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
