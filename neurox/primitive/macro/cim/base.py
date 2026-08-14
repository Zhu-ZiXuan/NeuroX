"""Abstract CIM-macro primitive.

See Also:
    docs/internals/primitive/macro/cim/base.md
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from neurox.common import ConfigBase, ModuleBase, PolicyBase, RegistryMixin, ValidateMixin

if TYPE_CHECKING:
    from .ideal import IdealCimMacro


def validate_quantization_input_range(code_range: tuple[int, int]) -> None:
    """Require one canonical quantization window.

    Two window shapes are legal, and only these two put the zero point on a
    bin edge at every bit width: unsigned `[0, upper]` with `upper >= 1`, and
    mid-zero `[-m, m - 1]` with `m >= 1`. A symmetric window `[-n, n]` holds
    zero strictly inside a bin and its threshold ladders do not nest across
    bit widths, so it is rejected rather than canonicalized.

    Args:
        code_range: Inclusive integer bounds `(lower, upper)` in MAC units.

    Raises:
        ValueError: The pair is not one of the two canonical shapes.
    """
    lower, upper = code_range
    if lower == 0 and upper >= 1:
        return
    if lower < 0 and upper == -lower - 1:
        return
    raise ValueError(
        f"require: quantization_input_range ({lower}, {upper}) is canonical — "
        f"unsigned [0, upper] with upper >= 1, or mid-zero [-m, m - 1]"
    )


def map_magnitude_input_code(code: Tensor, *, code_range: tuple[int, int]) -> tuple[Tensor, tuple[int, int]]:
    """Map quantization input codes onto a sign-magnitude ADC input grid.

    Args:
        code: Exact integer MAC-unit codes entering the quantizer.
        code_range: The mode's canonical quantization input range.

    Returns:
        The magnitudes the converter discriminates, and the inclusive ADC
        input code range `(0, upper)` its taps cover. A mid-zero window's
        bottom value has magnitude `upper + 1`, one step outside the returned
        range: the circuit resolves no tap there.
    """
    validate_quantization_input_range(code_range)
    _, upper = code_range
    return code.abs(), (0, upper)


def map_zero_point_input_code(code: Tensor, *, code_range: tuple[int, int]) -> tuple[Tensor, tuple[int, int]]:
    """Map quantization input codes onto a zero-point ADC input grid.

    Args:
        code: Exact integer MAC-unit codes entering the quantizer.
        code_range: The mode's canonical quantization input range.

    Returns:
        The offset codes `code - lower`, and the inclusive ADC input code
        range `(0, upper - lower)` they span. The map is the identity for an
        unsigned window.
    """
    validate_quantization_input_range(code_range)
    lower, upper = code_range
    return code - lower, (0, upper - lower)


@dataclass(frozen=True)
class CimMacroMode(ValidateMixin):
    """One quantization operating point of a physical CIM macro."""

    quantization_input_range: tuple[int, int]
    """Canonical inclusive MAC-unit window `(lower, upper)` the mode converts."""
    adc_input_code_range: tuple[int, int]
    """Inclusive range of the ADC input codes the mode's converter
    discriminates — circuit knowledge calibrated per mode, not a restatement
    of the conversion window."""
    max_bits_rescale_factor: float
    """This mode's output code at `adc_max_bits`, in ideal-macro codes."""

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        validate_quantization_input_range(self.quantization_input_range)
        lower, upper = self.adc_input_code_range
        self._require_non_neg(lower, "adc_input_code_range lower")
        self._require_gt(upper, "adc_input_code_range upper", lower)
        self._require_pos(self.max_bits_rescale_factor, "max_bits_rescale_factor")


class CimMacroConfig(ConfigBase, ABC):
    """PPA and activation limit shared by every CIM macro."""

    area_per_inst__um2: float
    """Silicon area of one fabricated instance."""
    leakage_per_inst__uW: float
    """Static leakage power of one fabricated instance."""
    max_active_num: int
    """Maximum number of input positions one conversion may select; positions
    outside the selected set must be zero."""

    def validate(self) -> None:

        # --- Activation limit ---

        self._require_pos(self.max_active_num, "max_active_num")

        # --- PPA ---

        self._require_non_neg(self.area_per_inst__um2, "area_per_inst__um2")
        self._require_non_neg(self.leakage_per_inst__uW, "leakage_per_inst__uW")


class CimMacroPolicy(PolicyBase, ABC):
    """Abstract marker base for CimMacro-family nonideality policies."""


class CimMacro[ConfigT: CimMacroConfig, PolicyT: CimMacroPolicy](
    ModuleBase[ConfigT, PolicyT],
    RegistryMixin["CimMacroConfig", "CimMacroPolicy", "CimMacro[CimMacroConfig, CimMacroPolicy]"],
    ABC,
):
    """Abstract base class for a CIM macro.

    Args:
        config: PPA, activation limit and scheme-specific parameters.
        policy: Composite nonideality policy of the scheme's parts.
        input_num: Logical input-vector length selected by the owner.
        output_num: Logical output-vector length selected by the owner.
        inst_shape: Per-instance multiplicity prefix.
        dtype: Tensor dtype for internal buffers.
        T__K: Operating temperature.
    """

    def __init__(
        self,
        *,
        config: ConfigT,
        policy: PolicyT,
        input_num: int,
        output_num: int,
        inst_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
    ) -> None:
        super().__init__(config=config, policy=policy, inst_shape=inst_shape)
        if input_num < 1:
            raise ValueError(f"require: input_num ({input_num}) >= 1")
        if output_num < 1:
            raise ValueError(f"require: output_num ({output_num}) >= 1")
        self._logical_shape = (input_num, output_num)
        self._T__K = T__K
        self._dtype = dtype

    @property
    def max_active_num(self) -> int:
        """Maximum number of input positions selected by one conversion."""
        return self.config.max_active_num

    @classmethod
    def from_config(
        cls,
        *,
        config: CimMacroConfig,
        policy: CimMacroPolicy,
        input_num: int,
        output_num: int,
        inst_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
    ) -> CimMacro[CimMacroConfig, CimMacroPolicy]:
        """Build the implementation registered for the config-policy pair.

        Args:
            config: Config whose type selects the implementation.
            policy: Policy whose type selects the implementation.
            input_num: Logical input-vector length selected by the owner.
            output_num: Logical output-vector length selected by the owner.
            inst_shape: Per-instance multiplicity prefix.
            dtype: Tensor dtype for internal buffers.
            T__K: Operating temperature.

        Returns:
            Registered CIM macro implementation.
        """
        impl = cls._lookup_neurox_module(config=config, policy=policy)
        return impl(
            config=config,
            policy=policy,
            input_num=input_num,
            output_num=output_num,
            inst_shape=inst_shape,
            dtype=dtype,
            T__K=T__K,
        )

    def _sample_fabricate_mismatch(self) -> None:
        pass

    @staticmethod
    def _split_col_lanes(t: Tensor, *, col_per_lane: int) -> Tensor:
        """Split the trailing column axis into a lane grid.

        Requires exact divisibility.

        Args:
            t: Tensor whose trailing axis enumerates columns.
                Shape: `[..., lane_num · col_per_lane]`.
            col_per_lane: Columns sharing one lane.

        Returns:
            The same values regrouped, lane axis ahead of the in-lane
            position.
            Shape: `[..., lane_num, col_per_lane]`.
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
    def w_value_range(self) -> tuple[int, int]:
        """Inclusive integer weight range the macro can program directly."""
        raise NotImplementedError

    @property
    @abstractmethod
    def quantization_input_ranges(self) -> tuple[tuple[int, int], ...]:
        """Canonical conversion window per mode, in MAC units.

        The tuple position is the `quantization_mode` index and the tuple
        length is the mode count.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def adc_max_bits(self) -> int:
        """Maximum supported `adc_bits` value."""
        raise NotImplementedError

    @abstractmethod
    def _max_bits_rescale_factor(self, quantization_mode: int) -> float:
        """Return the rescale factor of one mode at the maximum bit width.

        Args:
            quantization_mode: Mode index in
                `[0, len(quantization_input_ranges))`.
        """
        raise NotImplementedError

    @abstractmethod
    def map_quantization_input_code(self, code: Tensor, *, quantization_mode: int) -> tuple[Tensor, tuple[int, int]]:
        """Map exact MAC-unit codes onto this macro's ADC input code grid.

        The grid is the axis the macro's converter discriminates on: a
        magnitude for a sign-magnitude scheme, a zero-point offset for an
        offset scheme.

        Args:
            code: Exact integer plane dots — the quantizer's input codes.
            quantization_mode: Mode index in
                `[0, len(quantization_input_ranges))`.

        Returns:
            The mapped input codes and their inclusive range.
        """
        raise NotImplementedError

    @abstractmethod
    def latency__ns(self, *, adc_bits: int | None) -> float:
        """Duration of one `vec_mat_mul` call [ns].

        A macro's access time is fixed by its own schedule except for the
        readout, whose window follows the requested resolution; every other
        child settles inside a window the macro already owns and is never
        summed in.

        Args:
            adc_bits: Conversion resolution [bits] in `[1, adc_max_bits]`,
                or `None` for the lossless oracle.

        Returns:
            Duration of one conversion per word-line plane.
        """
        raise NotImplementedError

    @abstractmethod
    def program(self, w: Tensor) -> None:
        """Program the macro from a logical weight matrix.

        Args:
            w: Integer weight tensor matching the logical matrix geometry
                supplied at construction. Entries must lie in
                `w_value_range`.
                Shape: `[*inst_shape, input_num, output_num]`.
        """
        raise NotImplementedError

    @abstractmethod
    def vec_mat_mul(self, x: Tensor, *, quantization_mode: int, adc_bits: int | None) -> Tensor:
        """Run one conversion per word-line plane.

        Args:
            x: Logical input tensor; leading axes are broadcast batch
                dimensions. At most `max_active_num` positions may be
                selected per conversion; unselected positions must be zero.
                Entries must lie in `x_value_range`.
                Shape: `[..., input_num]`.
            quantization_mode: Mode index in
                `[0, len(quantization_input_ranges))`; selects the
                conversion window and its reference taps.
            adc_bits: Conversion resolution [bits] in `[1, adc_max_bits]`,
                or `None` for the lossless oracle.

        Returns:
            Output-code tensor whose leading axes broadcast the input's
            against `inst_shape`.
            Shape: `[..., output_num]`.
        """
        raise NotImplementedError

    def rescale_factor(self, *, quantization_mode: int, adc_bits: int | None) -> float:
        """Return this macro's output code expressed in ideal-macro codes.

        One bit-width law holds for every macro, `r_b = r_B · 2^(B - b)` with
        `B = adc_max_bits`, since dropping a bit doubles what one code
        carries; a concrete class supplies `r_B` alone. Dequantization into
        MAC units belongs to the algorithm side, which owns the window step.

        Args:
            quantization_mode: Mode index in
                `[0, len(quantization_input_ranges))`.
            adc_bits: Conversion resolution [bits] in `[1, adc_max_bits]`,
                or `None` for the lossless oracle, which is outside the
                bit-width chain.

        Returns:
            `1.0` for the lossless oracle; otherwise `r_b`.

        Raises:
            ValueError: Resolution is neither `None` nor in
                `[1, adc_max_bits]`.
        """
        if adc_bits is None:
            return 1.0
        max_bits = self.adc_max_bits
        if not (1 <= adc_bits <= max_bits):
            raise ValueError(
                f"require: adc_bits ({adc_bits}) in [1, adc_max_bits ({max_bits})] or None for the lossless oracle"
            )
        return self._max_bits_rescale_factor(quantization_mode) * float(1 << (max_bits - adc_bits))

    def to_ideal(self) -> IdealCimMacro:
        """Return an ideal twin quantizing over the published windows.

        The twin inherits this macro's logical geometry, instance
        multiplicity, value domains, published quantization windows and
        `adc_max_bits`. A macro whose encoding resolves a wider signed code
        range than its converter bit width overrides this to publish that
        wider width.
        """
        # Local import — the `ideal` module imports from this file, so the
        # symbols are only safe to resolve at call time.
        from .ideal import IdealCimMacro, IdealCimMacroConfig, IdealCimMacroPolicy

        input_num, output_num = self._logical_shape
        config = IdealCimMacroConfig(
            **{f.name: getattr(self.config, f.name) for f in fields(CimMacroConfig)},
            x_value_range=self.x_value_range,
            w_value_range=self.w_value_range,
            quantization_input_ranges=self.quantization_input_ranges,
            adc_max_bits=self.adc_max_bits,
        )
        return IdealCimMacro(
            config=config,
            policy=IdealCimMacroPolicy(),
            input_num=input_num,
            output_num=output_num,
            inst_shape=self.inst_shape,
            dtype=self._dtype,
            T__K=self._T__K,
        )
