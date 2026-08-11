"""Weight-slice layout and inverse digital aggregation for CIM engines.

See also:
    docs/internals/architecture/unit/cim/engine/weight_slice.md
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar, Generic, TypeVar

import torch.nn.functional as F
from torch import Tensor

from neurox.architecture.unit.cim.slicer import DirectSlicer, SimpleSlicer, Slicer
from neurox.common import ConfigBase, ModuleBase, PolicyBase
from neurox.common.encoding import Encoding
from neurox.common.mixin import RegistryMixin
from neurox.primitive.digital import DigitalPolicy, ShiftAdder, ShiftAdderConfig


class WeightSliceStageConfig(ConfigBase, ABC):
    """Abstract configuration root for weight-slice layouts."""

    @abstractmethod
    def layout_geometry(self, *, output_num: int) -> tuple[int, int]:
        """Return ``(logical outputs per block, physical macro planes)``."""
        raise NotImplementedError


class WeightSliceStagePolicy(PolicyBase, ABC):
    """Abstract policy root for weight-slice layouts."""


ConfigT = TypeVar("ConfigT", bound=WeightSliceStageConfig)
PolicyT = TypeVar("PolicyT", bound=WeightSliceStagePolicy)


class WeightSliceStage(
    ModuleBase[ConfigT, PolicyT],
    RegistryMixin["WeightSliceStageConfig", "WeightSliceStagePolicy", "WeightSliceStage"],
    Generic[ConfigT, PolicyT],
    ABC,
):
    """Pair program-time weight slicing with output-side reconstruction."""

    is_profile_target: ClassVar[bool] = False

    #: Sw-axis reconstruction block, or ``None`` for a layout with no
    #: arithmetic between the macro and the logical output (the single
    #: structural plane already is the result).
    shift_adder: ShiftAdder | None

    def __init__(
        self,
        *,
        config: ConfigT,
        policy: PolicyT,
        macro_w_value_range: tuple[int, int],
        output_num: int,
        w_parallel_size: int,
        macro_group_num: int,
    ) -> None:
        super().__init__(config=config, policy=policy, inst_shape=())
        self._output_num = output_num
        self._slicer = self._build_slicer(macro_w_value_range)
        self.shift_adder = None

    @classmethod
    def from_config(
        cls,
        *,
        config: WeightSliceStageConfig,
        policy: WeightSliceStagePolicy,
        macro_w_value_range: tuple[int, int],
        output_num: int,
        w_parallel_size: int,
        macro_group_num: int,
    ) -> WeightSliceStage:
        """Build the weight-slice layout selected by config and policy types."""
        impl = cls._lookup_neurox_module(config=config, policy=policy)
        return impl(
            config=config,
            policy=policy,
            macro_w_value_range=macro_w_value_range,
            output_num=output_num,
            w_parallel_size=w_parallel_size,
            macro_group_num=macro_group_num,
        )

    def _sample_fabricate_mismatch(self) -> None:
        pass

    @abstractmethod
    def _build_slicer(self, macro_w_value_range: tuple[int, int]) -> Slicer:
        """Construct the slicer defining this stage's logical weight domain."""
        raise NotImplementedError

    @property
    def value_range(self) -> tuple[int, int]:
        return self._slicer.value_range

    @property
    def aggregated_output_num(self) -> int:
        """Output elements one Sw reconstruction leaves per macro group."""
        return self._output_num

    def slice(self, weight: Tensor) -> Tensor:
        """Append the logical Sw axis to a weight tensor."""
        return self._slicer.slice(weight)

    @abstractmethod
    def arrange_weight(self, weight: Tensor) -> Tensor:
        """Map canonical geometric blocks into the macro-facing weight layout.

        Returns:
            Weight codes in macro-facing order.
            Shape: ``[..., Sw, Tc, G, D, L, output_num]``.
        """
        raise NotImplementedError

    @abstractmethod
    def aggregate(self, code: Tensor) -> Tensor:
        """Reconstruct the logical output represented by the Sw layout."""
        raise NotImplementedError


class DirectWeightSliceStageConfig(WeightSliceStageConfig):
    """Configuration for :class:`DirectWeightSliceStage`."""

    def layout_geometry(self, *, output_num: int) -> tuple[int, int]:
        return output_num, 1


class DirectWeightSliceStagePolicy(WeightSliceStagePolicy):
    """Policy for :class:`DirectWeightSliceStage`."""


@WeightSliceStage.register_neurox_module(
    config_type=DirectWeightSliceStageConfig,
    policy_type=DirectWeightSliceStagePolicy,
)
class DirectWeightSliceStage(WeightSliceStage[DirectWeightSliceStageConfig, DirectWeightSliceStagePolicy]):
    """Identity weight layout with one structural Sw plane."""

    def _build_slicer(self, macro_w_value_range: tuple[int, int]) -> Slicer:
        return DirectSlicer(value_range=macro_w_value_range)

    def arrange_weight(self, weight: Tensor) -> Tensor:
        # Shape: [..., D, G, Q, Tc, L, Sw=1] -> [..., Sw=1, Tc, G, D, L, output_num]
        b = weight.ndim - 6
        return weight.permute([*range(b), b + 5, b + 3, b + 1, b, b + 4, b + 2])

    def aggregate(self, code: Tensor) -> Tensor:
        # Shape: [..., Sa, Sw=1, G, output_num] -> [..., Sa, G, output_num]
        return code.squeeze(-3)


class InterWeightSliceStageConfig(WeightSliceStageConfig):
    """Configuration for :class:`InterWeightSliceStage`.

    Attributes:
        w_slice_num: Number of Macro-level weight slices.
        w_encoding: Positional encoding across weight slices.
        shift_adder_config: Sw-axis shift-adder configuration.
    """

    w_slice_num: int
    w_encoding: Encoding
    shift_adder_config: ShiftAdderConfig

    def validate(self) -> None:
        self._require_pos(self.w_slice_num, "w_slice_num")

    def layout_geometry(self, *, output_num: int) -> tuple[int, int]:
        return output_num, self.w_slice_num


class InterWeightSliceStagePolicy(WeightSliceStagePolicy):
    """Policy for :class:`InterWeightSliceStage`."""


@WeightSliceStage.register_neurox_module(
    config_type=InterWeightSliceStageConfig,
    policy_type=InterWeightSliceStagePolicy,
)
class InterWeightSliceStage(WeightSliceStage[InterWeightSliceStageConfig, InterWeightSliceStagePolicy]):
    """Place Sw slices on separate Macro planes."""

    def __init__(
        self,
        *,
        config: InterWeightSliceStageConfig,
        policy: InterWeightSliceStagePolicy,
        macro_w_value_range: tuple[int, int],
        output_num: int,
        w_parallel_size: int,
        macro_group_num: int,
    ) -> None:
        super().__init__(
            config=config,
            policy=policy,
            macro_w_value_range=macro_w_value_range,
            output_num=output_num,
            w_parallel_size=w_parallel_size,
            macro_group_num=macro_group_num,
        )
        self.shift_adder = ShiftAdder(
            config=config.shift_adder_config,
            policy=DigitalPolicy(),
            inst_shape=(w_parallel_size, macro_group_num),
            scale=self._slicer.slice_radix,
            digit_count=config.w_slice_num,
        )

    def _build_slicer(self, macro_w_value_range: tuple[int, int]) -> Slicer:
        return SimpleSlicer(
            slice_num=self.config.w_slice_num,
            slice_value_range=macro_w_value_range,
            encoding=self.config.w_encoding,
        )

    def arrange_weight(self, weight: Tensor) -> Tensor:
        # Shape: [..., D, G, Q, Tc, L, Sw] -> [..., Sw, Tc, G, D, L, output_num]
        b = weight.ndim - 6
        return weight.permute([*range(b), b + 5, b + 3, b + 1, b, b + 4, b + 2])

    def aggregate(self, code: Tensor) -> Tensor:
        # Shape: [..., Sa, Sw, G, output_num] -> [..., Sa, G, output_num]
        return self.shift_adder.shift_add(code, dim=-3, init_val=None)


class IntraWeightSliceStageConfig(WeightSliceStageConfig):
    """Configuration for :class:`IntraWeightSliceStage`.

    Attributes:
        w_slice_num: Number of Macro-level weight slices.
        w_encoding: Positional encoding across weight slices.
        shift_adder_config: Sw-axis shift-adder configuration.
    """

    w_slice_num: int
    w_encoding: Encoding
    shift_adder_config: ShiftAdderConfig

    def validate(self) -> None:
        self._require_pos(self.w_slice_num, "w_slice_num")

    def layout_geometry(self, *, output_num: int) -> tuple[int, int]:
        if self.w_slice_num > output_num:
            raise ValueError(f"require: w_slice_num ({self.w_slice_num}) <= output_num ({output_num})")
        return output_num // self.w_slice_num, 1


class IntraWeightSliceStagePolicy(WeightSliceStagePolicy):
    """Policy for :class:`IntraWeightSliceStage`."""


@WeightSliceStage.register_neurox_module(
    config_type=IntraWeightSliceStageConfig,
    policy_type=IntraWeightSliceStagePolicy,
)
class IntraWeightSliceStage(WeightSliceStage[IntraWeightSliceStageConfig, IntraWeightSliceStagePolicy]):
    """Place Sw slices on adjacent output ports of one Macro."""

    def __init__(
        self,
        *,
        config: IntraWeightSliceStageConfig,
        policy: IntraWeightSliceStagePolicy,
        macro_w_value_range: tuple[int, int],
        output_num: int,
        w_parallel_size: int,
        macro_group_num: int,
    ) -> None:
        super().__init__(
            config=config,
            policy=policy,
            macro_w_value_range=macro_w_value_range,
            output_num=output_num,
            w_parallel_size=w_parallel_size,
            macro_group_num=macro_group_num,
        )
        self._weights_per_macro = output_num // config.w_slice_num
        self._used_output_num = self._weights_per_macro * config.w_slice_num
        self.shift_adder = ShiftAdder(
            config=config.shift_adder_config,
            policy=DigitalPolicy(),
            inst_shape=(w_parallel_size, macro_group_num),
            scale=self._slicer.slice_radix,
            digit_count=config.w_slice_num,
        )

    @property
    def aggregated_output_num(self) -> int:
        """Logical weights one macro's ports hold — ``output_num // w_slice_num``."""
        return self._weights_per_macro

    def _build_slicer(self, macro_w_value_range: tuple[int, int]) -> Slicer:
        return SimpleSlicer(
            slice_num=self.config.w_slice_num,
            slice_value_range=macro_w_value_range,
            encoding=self.config.w_encoding,
        )

    def arrange_weight(self, weight: Tensor) -> Tensor:
        # Shape: [..., D, G, Q, Tc, L, Sw] -> [..., Tc, G, D, L, Q, Sw]
        b = weight.ndim - 6
        arranged = weight.permute([*range(b), b + 3, b + 1, b, b + 4, b + 2, b + 5])
        # Shape: [..., Tc, G, D, L, Q, Sw] -> [..., Tc, G, D, L, Q*Sw]
        arranged = arranged.flatten(start_dim=-2, end_dim=-1)
        # Shape: [..., Tc, G, D, L, Q*Sw] -> [..., Tc, G, D, L, output_num]
        arranged = F.pad(arranged, (0, self._output_num - arranged.shape[-1]))
        # Shape: [..., Tc, G, D, L, output_num] -> [..., Sw=1, Tc, G, D, L, output_num]
        return arranged.unsqueeze(b)

    def aggregate(self, code: Tensor) -> Tensor:
        # Shape: [..., Sa, Sw=1, G, output_num] -> [..., Sa, G, output_num]
        code = code.squeeze(-3)
        # Shape: [..., Sa, G, output_num] -> [..., Sa, G, Q*Sw]
        code = code[..., : self._used_output_num]
        # Shape: [..., Sa, G, Q*Sw] -> [..., Sa, G, Q, Sw]
        code = code.unflatten(-1, (self._weights_per_macro, self.config.w_slice_num))
        # Shape: [..., Sa, G, Q, Sw] -> [..., Sa, G, Q]
        return self.shift_adder.shift_add(code, dim=-1, init_val=None)
