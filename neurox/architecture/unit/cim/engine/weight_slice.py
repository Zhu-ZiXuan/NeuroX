"""Weight-slice layout and inverse digital aggregation.

See Also:
    docs/reference/architecture/unit/cim/engine/weight_slice.md
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

import torch.nn.functional as F
from torch import Tensor

from neurox.architecture.unit.cim.slicer import DirectSlicer, SimpleSlicer, Slicer
from neurox.common.module import ConfigBase, ModuleBase, PolicyBase
from neurox.common.registry_mixin import RegistryMixin
from neurox.encoding import Encoding
from neurox.primitive.digital import DigitalPolicy, ShiftAdder, ShiftAdderConfig


class WeightSliceStageConfig(ConfigBase, ABC):
    @abstractmethod
    def layout_geometry(self, *, output_num: int) -> tuple[int, int]:
        """Return `(logical outputs per block, physical macro planes)`."""
        raise NotImplementedError


class WeightSliceStagePolicy(PolicyBase, ABC):
    pass


class WeightSliceStage(
    ModuleBase,
    RegistryMixin[
        "WeightSliceStageConfig",
        "WeightSliceStagePolicy",
        "WeightSliceStage",
    ],
    ABC,
):
    """Pair program-time weight slicing with output-side reconstruction."""

    is_profile_target: ClassVar[bool] = False

    config: WeightSliceStageConfig
    policy: WeightSliceStagePolicy

    shift_adder: ShiftAdder | None
    """Sw-axis reconstruction block; `None` where the single structural plane already is the result."""

    def __init__(
        self,
        *,
        config: WeightSliceStageConfig,
        policy: WeightSliceStagePolicy,
        macro_w_value_range: tuple[int, int],
        output_num: int,
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
        macro_group_num: int,
    ) -> WeightSliceStage:
        """Build the weight-slice layout selected by config and policy types."""
        impl = cls._lookup_impl(config=config, policy=policy)
        return impl(
            config=config,
            policy=policy,
            macro_w_value_range=macro_w_value_range,
            output_num=output_num,
            macro_group_num=macro_group_num,
        )

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
            Shape: `[Sw, Tc, G, D, L, output]`.
        """
        raise NotImplementedError

    @abstractmethod
    def aggregate(self, code: Tensor) -> Tensor:
        """Reconstruct the logical output represented by the Sw layout."""
        raise NotImplementedError


class DirectWeightSliceStageConfig(WeightSliceStageConfig):
    def layout_geometry(self, *, output_num: int) -> tuple[int, int]:
        return output_num, 1


class DirectWeightSliceStagePolicy(WeightSliceStagePolicy):
    pass


@WeightSliceStage.register_impl(
    config_type=DirectWeightSliceStageConfig,
    policy_type=DirectWeightSliceStagePolicy,
)
class DirectWeightSliceStage(WeightSliceStage):
    """Identity weight layout with one structural Sw plane."""

    config: DirectWeightSliceStageConfig
    policy: DirectWeightSliceStagePolicy

    def __init__(
        self,
        *,
        config: DirectWeightSliceStageConfig,
        policy: DirectWeightSliceStagePolicy,
        macro_w_value_range: tuple[int, int],
        output_num: int,
        macro_group_num: int,
    ) -> None:
        super().__init__(
            config=config,
            policy=policy,
            macro_w_value_range=macro_w_value_range,
            output_num=output_num,
            macro_group_num=macro_group_num,
        )

    def _build_slicer(self, macro_w_value_range: tuple[int, int]) -> Slicer:
        return DirectSlicer(value_range=macro_w_value_range)

    def arrange_weight(self, weight: Tensor) -> Tensor:
        # Shape: [D, G, Q, Tc, L, Sw=1] -> [Sw=1, Tc, G, D, L, output]
        return weight.permute([5, 3, 1, 0, 4, 2])

    def aggregate(self, code: Tensor) -> Tensor:
        # Shape: [..., D, M, Sx, Sw=1, G, output] -> [..., D, M, Sx, G, output]
        return code.squeeze(-3)


class InterWeightSliceStageConfig(WeightSliceStageConfig):
    w_slice_num: int
    """Macro-level slices one logical weight is split into — the Sw axis."""
    w_encoding: Encoding
    """Positional encoding recombining the slices."""
    shift_adder_config: ShiftAdderConfig
    """Shift-adder recombining the Sw axis."""

    def validate(self) -> None:
        self._require_pos(self.w_slice_num, "w_slice_num")

    def layout_geometry(self, *, output_num: int) -> tuple[int, int]:
        return output_num, self.w_slice_num


class InterWeightSliceStagePolicy(WeightSliceStagePolicy):
    pass


@WeightSliceStage.register_impl(
    config_type=InterWeightSliceStageConfig,
    policy_type=InterWeightSliceStagePolicy,
)
class InterWeightSliceStage(WeightSliceStage):
    """Place Sw slices on separate Macro planes."""

    config: InterWeightSliceStageConfig
    policy: InterWeightSliceStagePolicy

    shift_adder: ShiftAdder
    """Sw-axis reconstruction block, always present in this layout."""

    def __init__(
        self,
        *,
        config: InterWeightSliceStageConfig,
        policy: InterWeightSliceStagePolicy,
        macro_w_value_range: tuple[int, int],
        output_num: int,
        macro_group_num: int,
    ) -> None:
        super().__init__(
            config=config,
            policy=policy,
            macro_w_value_range=macro_w_value_range,
            output_num=output_num,
            macro_group_num=macro_group_num,
        )
        self.shift_adder = ShiftAdder(
            config=config.shift_adder_config,
            policy=DigitalPolicy(),
            # Shape: [G]
            inst_shape=(macro_group_num,),
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
        # Shape: [D, G, Q, Tc, L, Sw] -> [Sw, Tc, G, D, L, output]
        return weight.permute([5, 3, 1, 0, 4, 2])

    def aggregate(self, code: Tensor) -> Tensor:
        # Shape: [..., D, M, Sx, Sw, G, output] -> [..., D, M, Sx, G, output]
        return self.shift_adder.shift_add(code, dim=-3, init_val=None)


class IntraWeightSliceStageConfig(WeightSliceStageConfig):
    w_slice_num: int
    """Macro-level slices one logical weight is split into — the Sw axis."""
    w_encoding: Encoding
    """Positional encoding recombining the slices."""
    shift_adder_config: ShiftAdderConfig
    """Shift-adder recombining the Sw axis."""

    def validate(self) -> None:
        self._require_pos(self.w_slice_num, "w_slice_num")

    def layout_geometry(self, *, output_num: int) -> tuple[int, int]:
        if self.w_slice_num > output_num:
            raise ValueError(f"require: w_slice_num ({self.w_slice_num}) <= output_num ({output_num})")
        return output_num // self.w_slice_num, 1


class IntraWeightSliceStagePolicy(WeightSliceStagePolicy):
    pass


@WeightSliceStage.register_impl(
    config_type=IntraWeightSliceStageConfig,
    policy_type=IntraWeightSliceStagePolicy,
)
class IntraWeightSliceStage(WeightSliceStage):
    """Place Sw slices on adjacent output ports of one Macro."""

    config: IntraWeightSliceStageConfig
    policy: IntraWeightSliceStagePolicy

    shift_adder: ShiftAdder
    """Sw-axis reconstruction block, always present in this layout."""

    def __init__(
        self,
        *,
        config: IntraWeightSliceStageConfig,
        policy: IntraWeightSliceStagePolicy,
        macro_w_value_range: tuple[int, int],
        output_num: int,
        macro_group_num: int,
    ) -> None:
        super().__init__(
            config=config,
            policy=policy,
            macro_w_value_range=macro_w_value_range,
            output_num=output_num,
            macro_group_num=macro_group_num,
        )
        self._weights_per_macro = output_num // config.w_slice_num
        self._used_output_num = self._weights_per_macro * config.w_slice_num
        self.shift_adder = ShiftAdder(
            config=config.shift_adder_config,
            policy=DigitalPolicy(),
            # Shape: [G]
            inst_shape=(macro_group_num,),
            scale=self._slicer.slice_radix,
            digit_count=config.w_slice_num,
        )

    @property
    def aggregated_output_num(self) -> int:
        """Logical weights one macro's ports hold — `output_num // w_slice_num`."""
        return self._weights_per_macro

    def _build_slicer(self, macro_w_value_range: tuple[int, int]) -> Slicer:
        return SimpleSlicer(
            slice_num=self.config.w_slice_num,
            slice_value_range=macro_w_value_range,
            encoding=self.config.w_encoding,
        )

    def arrange_weight(self, weight: Tensor) -> Tensor:
        # Shape: [D, G, Q, Tc, L, Sw] -> [Tc, G, D, L, Q, Sw]
        arranged = weight.permute([3, 1, 0, 4, 2, 5])
        # Shape: [Tc, G, D, L, Q, Sw] -> [Tc, G, D, L, Q*Sw]
        arranged = arranged.flatten(start_dim=-2, end_dim=-1)
        # Shape: [Tc, G, D, L, Q*Sw] -> [Tc, G, D, L, output]
        arranged = F.pad(arranged, (0, self._output_num - arranged.shape[-1]))
        # Shape: [Tc, G, D, L, output] -> [Sw=1, Tc, G, D, L, output]
        return arranged.unsqueeze(0)

    def aggregate(self, code: Tensor) -> Tensor:
        # Shape: [..., D, M, Sx, Sw=1, G, output] -> [..., D, M, Sx, G, output]
        code = code.squeeze(-3)
        # Shape: [..., D, M, Sx, G, output] -> [..., D, M, Sx, G, Q*Sw]
        code = code[..., : self._used_output_num]
        # Shape: [..., D, M, Sx, G, Q*Sw] -> [..., D, M, Sx, G, Q, Sw]
        code = code.unflatten(-1, (self._weights_per_macro, self.config.w_slice_num))
        # Shape: [..., D, M, Sx, G, Q, Sw] -> [..., D, M, Sx, G, Q]
        return self.shift_adder.shift_add(code, dim=-1, init_val=None)
