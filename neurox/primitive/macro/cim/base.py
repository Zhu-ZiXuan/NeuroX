"""Abstract CIM-macro primitive.

See Also:
    docs/reference/primitive/macro/cim/family.md
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, final, overload

import torch
from torch import Tensor

from neurox.common.module import ConfigBase, PolicyBase, ProfileModule
from neurox.common.registry_mixin import RegistryMixin
from neurox.common.torch_compat import torch_assert_async
from neurox.encoding import Encoding, Transcoder

if TYPE_CHECKING:
    from .ideal import IdealCimMacro


class CimMacroQuantizationScheme(Enum):
    ZERO_POINT = "zero_point"
    SIGN_MAGNITUDE = "sign_magnitude"


class CimMacroConfig(ConfigBase, ABC, base_only=True):
    # === Port geometry ===

    input_num: int
    """Logical input capacity fixed by the macro hardware."""
    lane_num: int
    """Parallel readout-circuit groups along the logical output direction."""
    scan_num: int
    """Serial output positions served by each readout group."""
    max_active_num: int
    """Maximum number of input positions one conversion may select; positions
    outside the selected set must be zero."""

    # === Output scales ===

    rescale_factors: tuple[float, ...]
    """MAC units represented by one output code at `adc_bits`, indexed by
    `quantization_mode`."""

    # === Static PPA ===

    area_per_inst__um2: float
    leakage_per_inst__uW: float

    # === Public API ===

    @property
    @final
    def output_num(self) -> int:
        """Logical output capacity fixed by the readout geometry."""
        return self.lane_num * self.scan_num

    # === Required by base class ===

    def validate(self) -> None:
        super().validate()

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

        # --- Static PPA ---

        self._require_non_neg(self.area_per_inst__um2, "area_per_inst__um2")
        self._require_non_neg(self.leakage_per_inst__uW, "leakage_per_inst__uW")

    # === For subclass to implement or override ===

    @property
    @abstractmethod
    def supports_signed_weights(self) -> bool:
        """Native signed-weight capability, independent of encoding and value range."""
        raise NotImplementedError

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


class CimMacroPolicy(PolicyBase, base_only=True):
    pass


_Config = CimMacroConfig
_Policy = CimMacroPolicy


class CimMacro(ProfileModule, RegistryMixin[_Config, _Policy], ABC, base_only=True):
    """CIM interface carrying integer codes and physical configuration.

    Logical outputs enumerate all lanes of one scan before the next scan.
    Programming and output recovery follow this order; concrete implementations
    map lane/scan positions to their own physical cell layouts.

    The caller supplies valid output counts from the unpadded geometry of the
    weight block selected for each access, consistently for execution and
    timing. A valid weight vector counts even when all its values are zero.
    """

    config: _Config
    policy: _Policy

    def __init__(
        self,
        *,
        config: _Config,
        policy: _Policy,
        inst_shape: tuple[int, ...],
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__(config=config, policy=policy, inst_shape=inst_shape)
        self._w_digit_num = config.w_digit_n
        self._w_digit_radix = config.w_digit_r
        self._x_digit_num = config.x_digit_n
        self._x_digit_radix = config.x_digit_r
        self._w_transcoder = Transcoder.from_encoding(
            config.w_enc, radix=self._w_digit_radix, digit_count=self._w_digit_num
        )
        self._x_transcoder = Transcoder.from_encoding(
            config.x_enc, radix=self._x_digit_radix, digit_count=self._x_digit_num
        )
        self._dtype = dtype

    # === Public API ===

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
        return self.config.lane_num

    @property
    @final
    def scan_num(self) -> int:
        return self.config.scan_num

    @property
    def max_active_num(self) -> int:
        return self.config.max_active_num

    @classmethod
    def from_config(
        cls,
        *,
        config: _Config,
        policy: _Policy,
        inst_shape: tuple[int, ...],
        dtype: torch.dtype = torch.float32,
    ) -> CimMacro:
        """Build the implementation registered for the config-policy pair."""
        impl = cls._lookup_impl(config=config, policy=policy)
        return impl(
            config=config,
            policy=policy,
            inst_shape=inst_shape,
            dtype=dtype,
        )

    @property
    def x_value_range(self) -> tuple[int, int]:
        """Inclusive single-cycle integer input range the macro accepts."""
        return self._x_transcoder.value_range

    @property
    def w_value_range(self) -> tuple[int, int]:
        """Inclusive integer weight range the macro can program directly."""
        return self._w_transcoder.value_range

    @final
    @torch.no_grad()
    @torch.compile(dynamic=False, fullgraph=True)
    def vec_mat_mul(
        self,
        x: Tensor,
        *,
        quantization_mode: int,
        adc_active_bits: int | None = None,
        effective_output_num: Tensor | None = None,
    ) -> Tensor:
        """Run one conversion per word-line plane.

        Args:
            x: Integer logical input tensor whose leading axes end with the complete
                `inst_shape`-aligned block. At most `max_active_num` positions
                may be selected per conversion; unselected positions must be
                zero. Entries must lie in `x_value_range`.
                Shape: `[..., input]`.
            effective_output_num: Valid output width of the weight block used
                by this access, excluding padding and including output slices.
                Dtype is `torch.int64`; counts match the programmed placement
                and broadcast to the input's aligned leading shape without enlarging it.
                `None` assumes every output is valid. Padding output codes are zero.
                A zero count requires all corresponding input entries to be zero.
                Shape: `[...]`.
            quantization_mode: Index selecting one reference operating point
                and its calibrated output scale.
            adc_active_bits: Active ADC resolution in `[1, adc_bits]`; `None`
                requests this macro's highest available precision.

        Returns:
            Final integer macro output-code tensor retaining the input's aligned leading axes.
            Shape: `[..., output]`.

        Raises:
            ValueError: Input geometry, conversion settings, or output shape is invalid.
        """
        self._check_quantization_mode(quantization_mode)
        self._check_adc_active_bits(adc_active_bits)
        self._check_effective_output_num(effective_output_num)
        if x.ndim == 0 or x.shape[-1] != self.input_num:
            raise ValueError(
                f"require: x.shape[-1] ({x.shape[-1] if x.ndim else None}) == input_num ({self.input_num})"
            )
        leading_shape = torch.broadcast_shapes(x.shape[:-1], self.inst_shape)
        if effective_output_num is not None:
            effective_output_num_broadcast_shape = torch.broadcast_shapes(effective_output_num.shape, leading_shape)
            if effective_output_num_broadcast_shape != leading_shape:
                raise ValueError("effective_output_num must broadcast to x's leading shape without expanding it")
            torch_assert_async(
                torch.all((effective_output_num != 0) | (x == 0).all(dim=-1)),
                "zero effective_output_num requires zero input",
            )
        phase_mask = None if effective_output_num is None else self._get_phase_mask(effective_output_num)
        output = self._vec_mat_mul_impl(
            x,
            leading_shape=leading_shape,
            quantization_mode=quantization_mode,
            adc_active_bits=adc_active_bits,
            phase_mask=phase_mask,
        )
        expected_shape = (*leading_shape, self.output_num)
        if tuple(output.shape) != expected_shape:
            raise ValueError(
                f"require: vec_mat_mul implementation output shape {expected_shape}; got {tuple(output.shape)}"
            )
        if effective_output_num is not None:
            logical_indices = torch.arange(self.output_num, device=output.device)
            output = output.where(logical_indices < effective_output_num.unsqueeze(-1), 0)
        return output

    @overload
    def latency__ns(self, *, adc_active_bits: int | None = None, effective_output_num: int | None = None) -> float: ...

    @overload
    def latency__ns(self, *, adc_active_bits: int | None = None, effective_output_num: Tensor) -> Tensor: ...

    @final
    @torch.no_grad()
    def latency__ns(
        self,
        *,
        adc_active_bits: int | None = None,
        effective_output_num: int | Tensor | None = None,
    ) -> float | Tensor:
        """Return operation duration from the longest active lane schedule.

        Args:
            adc_active_bits: Active ADC resolution; `None` uses the maximum.
            effective_output_num: Valid output width of the selected weight
                block. A Python integer describes a static count; tensor entries
                must have dtype `torch.int64` and describe distinct mapped accesses.
                A scalar tensor applies uniformly. `None` assumes every output is valid.
                Shape: `[...]`.

        Returns:
            A float for an integer or `None`; otherwise a Tensor with the count tensor's
            shape and device, including zero durations.
        """
        self._check_adc_active_bits(adc_active_bits)
        per_scan__ns = self._latency_per_scan__ns(adc_active_bits=adc_active_bits)
        if effective_output_num is None:
            return self.scan_num * per_scan__ns
        # Static-count timing specializes on the Python type without reading tensor values.
        if isinstance(effective_output_num, int):
            if not 0 <= effective_output_num <= self.output_num:
                raise ValueError("invalid effective_output_num")
            return -(-effective_output_num // self.lane_num) * per_scan__ns
        self._check_effective_output_num(effective_output_num)
        effective_scan_num = self._effective_scan_num(effective_output_num)
        return effective_scan_num * per_scan__ns

    def rescale_factor(
        self,
        *,
        quantization_mode: int,
        adc_active_bits: int | None = None,
    ) -> float:
        """Return the MAC units represented by one output code.

        A mode supplies the full-resolution factor. Dropping one ADC decision
        bit doubles the MAC interval represented by one final output code.

        Args:
            quantization_mode: Index selecting the calibrated output scale.
            adc_active_bits: Active ADC resolution in `[1, adc_bits]`; `None`
                requests this macro's highest available precision.

        Raises:
            ValueError: Resolution is outside `[1, adc_bits]`.
        """
        self._check_quantization_mode(quantization_mode)
        self._check_adc_active_bits(adc_active_bits)
        adc_active_bits = self._resolve_adc_active_bits(adc_active_bits)
        factor = self.config.rescale_factors[quantization_mode]
        return factor * (1 << (self.adc_bits - adc_active_bits))

    @final
    def to_ideal(self) -> IdealCimMacro:
        """Return an ideal twin using the calibrated output scales.

        The twin inherits this macro's logical geometry, instance multiplicity,
        value domains, signed-weight capability, mode scales, quantization scheme,
        and current temperature.
        """
        # Resolve the ideal subclass after its base finishes importing.
        from .ideal import IdealCimMacro, IdealCimMacroConfig, IdealCimMacroPolicy

        ideal_config = IdealCimMacroConfig(
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
            w_signed=self.config.supports_signed_weights,
            x_digit_num=self.config.x_digit_n,
            x_digit_radix=self.config.x_digit_r,
            x_encoding=self.config.x_enc,
            quantization_scheme=self.config.quant_scheme,
            x_value_range=self.x_value_range,
            w_value_range=self.w_value_range,
            adc_bits=self.adc_bits,
        )
        ideal = IdealCimMacro(
            config=ideal_config,
            policy=IdealCimMacroPolicy(),
            inst_shape=self.inst_shape,
            dtype=self._dtype,
        )
        ideal.set_temperature(self.T__K)
        return ideal

    # === Required by base class ===

    @property
    @final
    def _area_per_inst__um2(self) -> float:
        return self.config.area_per_inst__um2

    @property
    @final
    def _leakage_per_inst__uW(self) -> float:
        return self.config.leakage_per_inst__uW

    # === For subclass to implement or override ===

    @property
    @abstractmethod
    def adc_bits(self) -> int:
        """Maximum selectable ADC resolution."""
        raise NotImplementedError

    @abstractmethod
    def _latency_per_scan__ns(self, *, adc_active_bits: int | None) -> float:
        """Return the circuit duration of one scan, including input-digit phases."""
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

    @abstractmethod
    def _vec_mat_mul_impl(
        self,
        x: Tensor,
        *,
        leading_shape: tuple[int, ...],
        quantization_mode: int,
        adc_active_bits: int | None,
        phase_mask: Tensor | None,
    ) -> Tensor:
        """Restore physical readouts to logical output order.

        Args:
            x: Input data retaining its original broadcast layout.
                Shape: `[..., input]`.
            leading_shape: Complete caller and instance shape after broadcasting
                the input's leading shape with `inst_shape`.
            phase_mask: Broadcastable scan/lane activity; `None` enables every
                phase. Gates drive commands and circuit events, not solved
                currents or their energy.
                Shape: `[..., scan, lane]`.

        Returns:
            Codes ordered along the logical output axis used by `program`.
            The base class zero-fills positions beyond the valid logical prefix.
            Shape: `[..., output_num]`.
        """
        raise NotImplementedError

    # === Tools for subclass and internal use ===

    @final
    def _effective_scan_num(self, effective_output_num: Tensor) -> Tensor:
        return (effective_output_num + self.lane_num - 1) // self.lane_num

    @final
    def _get_phase_mask(self, effective_output_num: Tensor) -> Tensor:
        lanes = torch.arange(self.lane_num, device=effective_output_num.device)
        scans = torch.arange(self.scan_num, device=effective_output_num.device)
        whole_scans = effective_output_num // self.lane_num
        remaining_outputs = effective_output_num % self.lane_num
        # Shape: [...] -> [..., lane]
        lane_counts = whole_scans.unsqueeze(-1) + (lanes < remaining_outputs.unsqueeze(-1))
        # Shape: [scan, lane=1] < [..., scan=1, lane] -> [..., scan, lane]
        return scans.unsqueeze(-1) < lane_counts.unsqueeze(-2)

    @final
    def _check_effective_output_num(self, effective_output_num: Tensor | None) -> None:
        if effective_output_num is None:
            return
        if effective_output_num.dtype != torch.int64:
            raise TypeError("effective_output_num tensor must have dtype torch.int64")
        torch_assert_async(
            ((effective_output_num >= 0) & (effective_output_num <= self.output_num)).all(),
            "invalid effective_output_num",
        )

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
