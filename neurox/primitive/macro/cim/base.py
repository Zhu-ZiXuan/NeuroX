"""Abstract CIM-macro primitive.

See Also:
    docs/reference/primitive/macro/cim/family.md
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, final

import torch
from torch import Tensor

from neurox.common.encoding import Encoding, Transcoder
from neurox.common.module import ConfigBase, ModuleBase, PolicyBase
from neurox.common.registry_mixin import RegistryMixin

if TYPE_CHECKING:
    from .ideal import IdealCimMacro


class CimMacroQuantizationScheme(Enum):
    """Macro output quantization scheme."""

    ZERO_POINT = "zero_point"
    SIGN_MAGNITUDE = "sign_magnitude"


class CimMacroConfig(ConfigBase, ABC):
    input_num: int
    """Logical input capacity fixed by the macro hardware."""

    lane_num: int
    """Parallel readout-circuit groups along the logical output direction."""

    scan_num: int
    """Serial output positions served by each readout group."""

    max_active_num: int
    """Maximum number of input positions one conversion may select; positions
    outside the selected set must be zero."""

    rescale_factors: tuple[float, ...]
    """MAC units represented by one output code at `adc_bits`, indexed by
    `quantization_mode`."""

    area_per_inst__um2: float
    """Macro-owned area per instance, excluding profiled child modules."""

    leakage_per_inst__uW: float
    """Macro-owned leakage per instance, excluding profiled child modules."""

    @property
    @abstractmethod
    def w_digit_n(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def w_digit_r(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def w_enc(self) -> Encoding:
        raise NotImplementedError

    @property
    @abstractmethod
    def x_digit_n(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def x_digit_r(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def x_enc(self) -> Encoding:
        raise NotImplementedError

    @property
    @abstractmethod
    def quant_scheme(self) -> CimMacroQuantizationScheme:
        raise NotImplementedError

    @property
    @final
    def output_num(self) -> int:
        """Logical output capacity fixed by the readout geometry."""
        return self.lane_num * self.scan_num

    def validate(self) -> None:

        # --- Port geometry ---

        self._require_pos(self.input_num, "input_num")
        self._require_pos(self.lane_num, "lane_num")
        self._require_pos(self.scan_num, "scan_num")
        self._require_pos(self.max_active_num, "max_active_num")
        self._require_le(self.max_active_num, "max_active_num", self.input_num)

        # --- Digit geometry ---

        self._require_pos(self.w_digit_n, "w_digit_n")
        self._require_ge(self.w_digit_r, "w_digit_r", 2)
        self._require_pos(self.x_digit_n, "x_digit_n")
        self._require_ge(self.x_digit_r, "x_digit_r", 2)

        # --- Output scales ---

        self._require_non_empty(self.rescale_factors, "rescale_factors")
        for index, factor in enumerate(self.rescale_factors):
            self._require_pos(factor, f"rescale_factors[{index}]")

        # --- PPA ---

        self._require_non_neg(self.area_per_inst__um2, "area_per_inst__um2")
        self._require_non_neg(self.leakage_per_inst__uW, "leakage_per_inst__uW")


class CimMacroPolicy(PolicyBase, ABC):
    pass


_Config = CimMacroConfig
_Policy = CimMacroPolicy


class CimMacro(
    ModuleBase,
    RegistryMixin["_Config", "_Policy", "CimMacro"],
    ABC,
):
    """Abstract base class for a CIM macro.

    The macro closes the analog domain: analog signals and analog
    non-idealities live inside it and never cross above it, so what its
    interface carries is integer codes and physical configuration.

    Args:
        inst_shape: Per-instance multiplicity prefix.
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
        T__K: float,
    ) -> None:
        super().__init__(config=config, policy=policy, inst_shape=inst_shape)
        self._w_digit_num = config.w_digit_n
        self._w_digit_radix = config.w_digit_r
        self._x_digit_num = config.x_digit_n
        self._x_digit_radix = config.x_digit_r
        self._w_transcoder = Transcoder.from_encoding(
            encoding=config.w_enc,
            radix=self._w_digit_radix,
            digit_count=self._w_digit_num,
        )
        self._x_transcoder = Transcoder.from_encoding(
            encoding=config.x_enc,
            radix=self._x_digit_radix,
            digit_count=self._x_digit_num,
        )
        self._T__K = T__K
        self._dtype = dtype

    @property
    @final
    def input_num(self) -> int:
        return self.config.input_num

    @property
    @final
    def output_num(self) -> int:
        return self.config.output_num

    @property
    @final
    def lane_num(self) -> int:
        """Parallel readout-circuit groups."""
        return self.config.lane_num

    @property
    @final
    def scan_num(self) -> int:
        """Output positions serialized onto each readout lane."""
        return self.config.scan_num

    @property
    def max_active_num(self) -> int:
        """Maximum number of input positions selected by one conversion."""
        return self.config.max_active_num

    @property
    @final
    def _area_per_inst__um2(self) -> float:
        return self.config.area_per_inst__um2

    @property
    @final
    def _leakage_per_inst__uW(self) -> float:
        return self.config.leakage_per_inst__uW

    @classmethod
    def from_config(
        cls,
        *,
        config: _Config,
        policy: _Policy,
        inst_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
    ) -> CimMacro:
        """Build the implementation registered for the config-policy pair.

        Returns:
            Registered CIM macro implementation.
        """
        impl = cls._lookup_impl(config=config, policy=policy)
        return impl(
            config=config,
            policy=policy,
            inst_shape=inst_shape,
            dtype=dtype,
            T__K=T__K,
        )

    @property
    def x_value_range(self) -> tuple[int, int]:
        """Inclusive single-cycle integer input range the macro accepts."""
        return self._x_transcoder.value_range

    @property
    def w_value_range(self) -> tuple[int, int]:
        """Inclusive integer weight range the macro can program directly."""
        return self._w_transcoder.value_range

    @property
    @abstractmethod
    def adc_bits(self) -> int:
        """Maximum selectable ADC resolution."""
        raise NotImplementedError

    @final
    def _check_quantization_mode(self, quantization_mode: int) -> None:
        mode_num = len(self.config.rescale_factors)
        if not (0 <= quantization_mode < mode_num):
            raise ValueError(f"require: quantization_mode ({quantization_mode}) in [0, {mode_num})")

    @final
    def _check_adc_active_bits(self, adc_active_bits: int | None) -> None:
        if adc_active_bits is not None and not (1 <= adc_active_bits <= self.adc_bits):
            raise ValueError(f"require: adc_active_bits ({adc_active_bits}) in [1, adc_bits ({self.adc_bits})]")

    @final
    def _resolve_adc_active_bits(self, adc_active_bits: int | None) -> int:
        return self.adc_bits if adc_active_bits is None else adc_active_bits

    @abstractmethod
    def latency__ns(self, *, adc_active_bits: int | None) -> float:
        """Circuit latency of one complete `vec_mat_mul` call [ns].

        Args:
            adc_active_bits: Active ADC resolution in `[1, adc_bits]`; `None`
                requests this macro's highest available precision.
        """
        raise NotImplementedError

    @abstractmethod
    def program(self, w: Tensor) -> None:
        """Program the macro from a logical weight matrix.

        Args:
            w: Integer weight tensor matching the configured logical matrix
                geometry. Entries must lie in `w_value_range`.
                Shape: `[*inst_shape, input, output]`.
        """
        raise NotImplementedError

    @final
    @torch.no_grad()
    def vec_mat_mul(
        self,
        x: Tensor,
        *,
        quantization_mode: int,
        adc_active_bits: int | None,
    ) -> Tensor:
        """Run one conversion per word-line plane.

        Args:
            x: Logical input tensor whose leading axes end with the complete
                `inst_shape`-aligned block. At most `max_active_num` positions
                may be selected per conversion; unselected positions must be
                zero. Entries must lie in `x_value_range`.
                Shape: `[..., input]`.
            quantization_mode: Index selecting one reference operating point
                and its calibrated output scale.
            adc_active_bits: Active ADC resolution in `[1, adc_bits]`; `None`
                requests this macro's highest available precision.

        Returns:
            Final macro output-code tensor retaining the input's aligned leading axes.
            Shape: `[..., output]`.
        """
        self._check_quantization_mode(quantization_mode)
        self._check_adc_active_bits(adc_active_bits)
        if x.ndim == 0 or x.shape[-1] != self.input_num:
            raise ValueError(
                f"require: x.shape[-1] ({x.shape[-1] if x.ndim else None}) == input_num ({self.input_num})"
            )
        output = self._vec_mat_mul_impl(
            x,
            quantization_mode=quantization_mode,
            adc_active_bits=adc_active_bits,
        )
        expected_shape = (self.lane_num, self.scan_num)
        if tuple(output.shape[-2:]) != expected_shape:
            raise ValueError(
                f"require: vec_mat_mul implementation output trailing shape {expected_shape}; got {tuple(output.shape)}"
            )
        return output.flatten(-2)

    @abstractmethod
    def _vec_mat_mul_impl(
        self,
        x: Tensor,
        *,
        quantization_mode: int,
        adc_active_bits: int | None,
    ) -> Tensor:
        """Return the canonical readout layout before output flattening.

        Returns:
            Macro output codes with the readout axes kept separate.
            Shape: `[..., lane, scan]`.
        """
        raise NotImplementedError

    def rescale_factor(
        self,
        *,
        quantization_mode: int,
        adc_active_bits: int | None,
    ) -> float:
        """Return the MAC units represented by one output code.

        A mode supplies the full-resolution factor. Dropping one ADC decision
        bit doubles the MAC interval represented by one final output code.

        Args:
            quantization_mode: Index selecting the calibrated output scale.
            adc_active_bits: Active ADC resolution in `[1, adc_bits]`; `None`
                requests this macro's highest available precision.

        Returns:
            The selected-resolution rescale factor `r_b`.

        Raises:
            ValueError: Resolution is outside `[1, adc_bits]`.
        """
        self._check_quantization_mode(quantization_mode)
        self._check_adc_active_bits(adc_active_bits)
        adc_active_bits = self._resolve_adc_active_bits(adc_active_bits)
        factor = self.config.rescale_factors[quantization_mode]
        return factor * float(1 << (self.adc_bits - adc_active_bits))

    def to_ideal(self) -> IdealCimMacro:
        """Return an ideal twin using the calibrated output scales.

        The twin inherits this macro's logical geometry, instance multiplicity,
        value domains, mode scales and quantization scheme.
        """
        # Local import — the `ideal` module imports from this file, so the
        # symbols are only safe to resolve at call time.
        from .ideal import IdealCimMacro, IdealCimMacroConfig, IdealCimMacroPolicy

        config = IdealCimMacroConfig(
            input_num=self.config.input_num,
            lane_num=self.config.lane_num,
            scan_num=self.config.scan_num,
            max_active_num=self.config.max_active_num,
            rescale_factors=self.config.rescale_factors,
            area_per_inst__um2=self.config.area_per_inst__um2,
            leakage_per_inst__uW=self.config.leakage_per_inst__uW,
            w_digit_num=self.config.w_digit_n,
            w_digit_radix=self.config.w_digit_r,
            w_encoding=self.config.w_enc,
            x_digit_num=self.config.x_digit_n,
            x_digit_radix=self.config.x_digit_r,
            x_encoding=self.config.x_enc,
            quantization_scheme=self.config.quant_scheme,
            x_value_range=self.x_value_range,
            w_value_range=self.w_value_range,
            adc_bits=self.adc_bits,
        )
        return IdealCimMacro(
            config=config,
            policy=IdealCimMacroPolicy(),
            inst_shape=self.inst_shape,
            dtype=self._dtype,
            T__K=self._T__K,
        )
