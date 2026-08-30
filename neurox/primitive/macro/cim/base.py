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

from neurox.common import ConfigBase, ModuleBase, PolicyBase, RegistryMixin

if TYPE_CHECKING:
    from .ideal import IdealCimMacro


class CimMacroQuantizationScheme(Enum):
    """Macro output quantization scheme."""

    ZERO_POINT = "zero_point"
    SIGN_MAGNITUDE = "sign_magnitude"


class CimMacroConfig(ConfigBase, ABC):
    rescale_factors: tuple[float, ...]
    """MAC units represented by one output code at `adc_bits`, indexed by
    `quantization_mode`."""

    area_per_inst__um2: float
    """Macro-owned area per instance, excluding profiled child modules."""

    leakage_per_inst__uW: float
    """Macro-owned leakage per instance, excluding profiled child modules."""

    max_active_num: int
    """Maximum number of input positions one conversion may select; positions
    outside the selected set must be zero."""

    def validate(self) -> None:

        # --- Output scales ---

        self._require_non_empty(self.rescale_factors, "rescale_factors")
        for index, factor in enumerate(self.rescale_factors):
            self._require_pos(factor, f"rescale_factors[{index}]")

        # --- Activation limit ---

        self._require_pos(self.max_active_num, "max_active_num")

        # --- PPA ---

        self._require_non_neg(self.area_per_inst__um2, "area_per_inst__um2")
        self._require_non_neg(self.leakage_per_inst__uW, "leakage_per_inst__uW")


class CimMacroPolicy(PolicyBase, ABC):
    pass


class CimMacro[ConfigT: CimMacroConfig, PolicyT: CimMacroPolicy](
    ModuleBase[ConfigT, PolicyT],
    RegistryMixin["CimMacroConfig", "CimMacroPolicy", "CimMacro[CimMacroConfig, CimMacroPolicy]"],
    ABC,
):
    """Abstract base class for a CIM macro.

    The macro closes the analog domain: analog signals and analog
    non-idealities live inside it and never cross above it, so what its
    interface carries is integer codes and physical configuration.

    Args:
        input_num: Logical input-vector length.
        output_num: Logical output-vector length.
        inst_shape: Per-instance multiplicity prefix.
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
        config: CimMacroConfig,
        policy: CimMacroPolicy,
        input_num: int,
        output_num: int,
        inst_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
    ) -> CimMacro[CimMacroConfig, CimMacroPolicy]:
        """Build the implementation registered for the config-policy pair.

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

    @property
    @abstractmethod
    def x_value_range(self) -> tuple[int, int]:
        """Inclusive single-cycle integer input range the macro accepts."""
        raise NotImplementedError

    @property
    @abstractmethod
    def w_value_range(self) -> tuple[int, int]:
        """Inclusive integer weight range the macro can program directly."""
        raise NotImplementedError

    @property
    @abstractmethod
    def adc_bits(self) -> int:
        """Maximum selectable ADC resolution."""
        raise NotImplementedError

    @property
    @abstractmethod
    def _quantization_scheme(self) -> CimMacroQuantizationScheme:
        raise NotImplementedError

    @final
    def _check_quantization_mode(self, quantization_mode: int) -> None:
        mode_num = len(self.config.rescale_factors)
        if not (0 <= quantization_mode < mode_num):
            raise ValueError(f"require: quantization_mode ({quantization_mode}) in [0, {mode_num})")

    @final
    def _check_adc_active_bits(self, adc_active_bits: int) -> None:
        if not (1 <= adc_active_bits <= self.adc_bits):
            raise ValueError(f"require: adc_active_bits ({adc_active_bits}) in [1, adc_bits ({self.adc_bits})]")

    @abstractmethod
    def restore_adc_layout(self, value: Tensor) -> Tensor:
        """Restore a converter-aligned tensor to the logical output layout.

        Args:
            value: One value per physical conversion position, in the layout
                used by the concrete macro's converter call.

        Returns:
            The same values arranged like `vec_mat_mul` output.
                Shape: `[..., output_num]`.
        """
        raise NotImplementedError

    @abstractmethod
    def initiation_interval__ns(self, *, adc_active_bits: int) -> float:
        """Scheduled interval occupied by one `vec_mat_mul` call [ns].

        Args:
            adc_active_bits: Active ADC resolution in `[1, adc_bits]`.
        """
        raise NotImplementedError

    @abstractmethod
    def program(self, w: Tensor) -> None:
        """Program the macro from a logical weight matrix.

        Args:
            w: Integer weight tensor matching the logical matrix geometry
                supplied at construction. Entries must lie in `w_value_range`.
                Shape: `[*inst_shape, input_num, output_num]`.
        """
        raise NotImplementedError

    @abstractmethod
    def vec_mat_mul(self, x: Tensor, *, quantization_mode: int, adc_active_bits: int) -> Tensor:
        """Run one conversion per word-line plane.

        Args:
            x: Logical input tensor whose leading axes end with the complete
                `inst_shape`-aligned block. At most `max_active_num` positions
                may be selected per conversion; unselected positions must be
                zero. Entries must lie in `x_value_range`.
                Shape: `[..., input_num]`.
            quantization_mode: Index selecting one reference operating point
                and its calibrated output scale.
            adc_active_bits: Active ADC resolution in `[1, adc_bits]`.

        Returns:
            Final macro output-code tensor retaining the input's aligned leading axes.
            Shape: `[..., output_num]`.
        """
        raise NotImplementedError

    def rescale_factor(self, *, quantization_mode: int, adc_active_bits: int) -> float:
        """Return the MAC units represented by one output code.

        A mode supplies the full-resolution factor. Dropping one ADC decision
        bit doubles the MAC interval represented by one final output code.

        Args:
            quantization_mode: Index selecting the calibrated output scale.
            adc_active_bits: Active ADC resolution in `[1, adc_bits]`.

        Returns:
            The selected-resolution rescale factor `r_b`.

        Raises:
            ValueError: Resolution is outside `[1, adc_bits]`.
        """
        self._check_quantization_mode(quantization_mode)
        self._check_adc_active_bits(adc_active_bits)
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

        input_num, output_num = self._logical_shape
        config = IdealCimMacroConfig(
            rescale_factors=self.config.rescale_factors,
            area_per_inst__um2=self.config.area_per_inst__um2,
            leakage_per_inst__uW=self.config.leakage_per_inst__uW,
            max_active_num=self.config.max_active_num,
            x_value_range=self.x_value_range,
            w_value_range=self.w_value_range,
            adc_bits=self.adc_bits,
            quantization_scheme=self._quantization_scheme,
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
