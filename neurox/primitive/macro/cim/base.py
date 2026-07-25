"""Abstract CIM-macro primitive.

See also:
    docs/internals/primitive/macro/cim/base.md
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, TypeVar

import torch
from torch import Tensor

from neurox.common import ConfigBase, ModuleBase, PolicyBase
from neurox.common.mixin import RegistryMixin

if TYPE_CHECKING:
    from .ideal import IdealCimMacro


class CimMacroConfig(ConfigBase, ABC):
    """PPA and activation limit shared by every CIM macro.

    Attributes:
        area_per_inst__um2: Silicon area per fabricated instance.
        leakage_per_inst__uW: Static leakage per instance.
        max_active_num: Maximum number of input positions selected by one
            conversion. Positions outside the selected set are forced to zero.
    """

    area_per_inst__um2: float
    leakage_per_inst__uW: float
    max_active_num: int

    def validate(self) -> None:
        # --- Activation limit ---

        self._require_pos(self.max_active_num, "max_active_num")

        # --- PPA ---

        self._require_non_neg(self.area_per_inst__um2, "area_per_inst__um2")
        self._require_non_neg(self.leakage_per_inst__uW, "leakage_per_inst__uW")


class CimMacroPolicy(PolicyBase, ABC):
    """Abstract marker base for CimMacro-family nonideality policies."""


ConfigT = TypeVar("ConfigT", bound=CimMacroConfig)
PolicyT = TypeVar("PolicyT", bound=CimMacroPolicy)


class CimMacro(
    ModuleBase[ConfigT, PolicyT],
    RegistryMixin["CimMacroConfig", "CimMacroPolicy", "CimMacro"],
    Generic[ConfigT, PolicyT],
    ABC,
):
    """Abstract base class for a CIM macro.

    Args:
        config: Concrete configuration dataclass.
        policy: Composite nonideality policy.
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
    ) -> CimMacro:
        """Build the implementation registered for the config-policy pair.

        Args:
            config: Concrete configuration dataclass.
            policy: Composite nonideality policy.
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
    def w_value_range(self) -> tuple[int, int]:
        """Inclusive integer weight range the macro can program directly."""
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
        """Program the macro from a logical weight matrix.

        Args:
            w: Integer weight tensor whose shape matches
                the logical matrix geometry supplied at construction.
                Entries must lie in :attr:`w_value_range`.
        """
        raise NotImplementedError

    @abstractmethod
    def vec_mat_mul(self, x: Tensor, *, adc_mode: int, adc_bits: int) -> Tensor:
        """Run one conversion per word-line plane.

        Args:
            x: Logical input tensor with primitive trailing ``[input_num]``;
                leading axes are broadcast batch dimensions. At most
                :attr:`max_active_num` positions may be selected per
                conversion; unselected positions must be zero.
                Entries must lie in :attr:`x_value_range`.
            adc_mode: ADC operating-point index selecting the reference
                row / tap set; valid values are ``[0, adc_mode_num)``.
            adc_bits: ADC resolution [bits] the conversion runs at.

        Returns:
            ADC-code tensor with the same leading dimensions and trailing
            ``[output_num]``.
        """
        raise NotImplementedError

    def to_ideal(self) -> IdealCimMacro:
        """Return a lossless ideal twin with the same logical interface."""
        from .ideal import IdealCimMacro, IdealCimMacroConfig, IdealCimMacroPolicy

        input_num, output_num = self._logical_shape
        ideal_config = IdealCimMacroConfig(
            area_per_inst__um2=self.config.area_per_inst__um2,
            leakage_per_inst__uW=self.config.leakage_per_inst__uW,
            max_active_num=self.config.max_active_num,
            x_value_range=self.x_value_range,
            w_value_range=self.w_value_range,
            adc_mode_num=self.adc_mode_num,
            adc_max_bits=self.adc_max_bits,
        )
        return IdealCimMacro(
            config=ideal_config,
            policy=IdealCimMacroPolicy(),
            input_num=input_num,
            output_num=output_num,
            inst_shape=self.inst_shape,
            dtype=self._dtype,
            T__K=self._T__K,
        )
