"""Input-slice serialization and inverse digital aggregation for CIM engines.

See also:
    docs/reference/architecture/unit/cim/engine/x_slice.md
    docs/internals/architecture/unit/cim/engine/x_slice.md
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar, Generic, TypeVar

from torch import Tensor

from neurox.architecture.unit.cim.slicer import DirectSlicer, SerialSlicer, Slicer
from neurox.common import ConfigBase, ModuleBase, PolicyBase, RegistryMixin
from neurox.primitive.digital import DigitalPolicy, ShiftAdder, ShiftAdderConfig


class XSliceStageConfig(ConfigBase, ABC):
    """Abstract configuration root for input-slice serialization."""


class XSliceStagePolicy(PolicyBase, ABC):
    """Abstract policy root for input-slice serialization."""


ConfigT = TypeVar("ConfigT", bound=XSliceStageConfig, covariant=True)
PolicyT = TypeVar("PolicyT", bound=XSliceStagePolicy, covariant=True)


class XSliceStage(
    ModuleBase[ConfigT, PolicyT],
    RegistryMixin["XSliceStageConfig", "XSliceStagePolicy", "XSliceStage[XSliceStageConfig, XSliceStagePolicy]"],
    Generic[ConfigT, PolicyT],
    ABC,
):
    """Pair input slicing with Sa-axis digital reconstruction."""

    is_profile_target: ClassVar[bool] = False

    shift_adder: ShiftAdder | None
    """Sa-axis reconstruction block; `None` where the single structural cycle already is the result."""

    def __init__(
        self,
        *,
        config: ConfigT,
        policy: PolicyT,
        macro_x_value_range: tuple[int, int],
        w_parallel_size: int,
        macro_group_num: int,
    ) -> None:
        super().__init__(config=config, policy=policy, inst_shape=())
        self._slicer = self._build_slicer(macro_x_value_range)
        self.shift_adder = None

    @classmethod
    def from_config(
        cls,
        *,
        config: XSliceStageConfig,
        policy: XSliceStagePolicy,
        macro_x_value_range: tuple[int, int],
        w_parallel_size: int,
        macro_group_num: int,
    ) -> XSliceStage[XSliceStageConfig, XSliceStagePolicy]:
        """Build the input-slice stage selected by config and policy types."""
        impl = cls._lookup_neurox_module(config=config, policy=policy)
        return impl(
            config=config,
            policy=policy,
            macro_x_value_range=macro_x_value_range,
            w_parallel_size=w_parallel_size,
            macro_group_num=macro_group_num,
        )

    def _sample_fabricate_mismatch(self) -> None:
        pass

    @abstractmethod
    def _build_slicer(self, macro_x_value_range: tuple[int, int]) -> Slicer:
        """Construct the slicer defining this stage's logical input domain."""
        raise NotImplementedError

    @property
    def value_range(self) -> tuple[int, int]:
        return self._slicer.value_range

    @property
    @abstractmethod
    def slice_num(self) -> int:
        """Successive input cycles one logical input is serialized into — the Sa axis."""
        raise NotImplementedError

    def slice(self, x: Tensor) -> Tensor:
        """Append the Sa axis to a logical input tensor."""
        return self._slicer.slice(x)

    @abstractmethod
    def aggregate(self, code: Tensor) -> Tensor:
        """Reconstruct the logical input precision represented by Sa."""
        raise NotImplementedError


class DirectXSliceStageConfig(XSliceStageConfig):
    """Configuration for `DirectXSliceStage`."""


class DirectXSliceStagePolicy(XSliceStagePolicy):
    """Policy for `DirectXSliceStage`."""


@XSliceStage.register_neurox_module(
    config_type=DirectXSliceStageConfig,
    policy_type=DirectXSliceStagePolicy,
)
class DirectXSliceStage(XSliceStage[DirectXSliceStageConfig, DirectXSliceStagePolicy]):
    """Identity input serialization with one structural Sa step."""

    def _build_slicer(self, macro_x_value_range: tuple[int, int]) -> Slicer:
        return DirectSlicer(value_range=macro_x_value_range)

    @property
    def slice_num(self) -> int:
        return 1

    def aggregate(self, code: Tensor) -> Tensor:
        # Shape: [..., M, Sa=1, G, Q] -> [..., M, G, Q]
        return code.squeeze(-3)


class SerialXSliceStageConfig(XSliceStageConfig):
    """Configuration for `SerialXSliceStage`."""

    x_slice_num: int
    """Serial cycles one logical input is split into — the Sa axis."""
    shift_adder_config: ShiftAdderConfig
    """Shift-adder recombining the Sa axis."""

    def validate(self) -> None:
        self._require_pos(self.x_slice_num, "x_slice_num")


class SerialXSliceStagePolicy(XSliceStagePolicy):
    """Policy for `SerialXSliceStage`."""


@XSliceStage.register_neurox_module(
    config_type=SerialXSliceStageConfig,
    policy_type=SerialXSliceStagePolicy,
)
class SerialXSliceStage(XSliceStage[SerialXSliceStageConfig, SerialXSliceStagePolicy]):
    """Serialize logical inputs into radix-weighted Macro input cycles."""

    shift_adder: ShiftAdder
    """Sa-axis reconstruction block, always present in this layout."""

    def __init__(
        self,
        *,
        config: SerialXSliceStageConfig,
        policy: SerialXSliceStagePolicy,
        macro_x_value_range: tuple[int, int],
        w_parallel_size: int,
        macro_group_num: int,
    ) -> None:
        super().__init__(
            config=config,
            policy=policy,
            macro_x_value_range=macro_x_value_range,
            w_parallel_size=w_parallel_size,
            macro_group_num=macro_group_num,
        )
        self.shift_adder = ShiftAdder(
            config=config.shift_adder_config,
            policy=DigitalPolicy(),
            inst_shape=(w_parallel_size, macro_group_num),
            scale=self._slicer.slice_radix,
            digit_count=config.x_slice_num,
        )

    @property
    def slice_num(self) -> int:
        return self.config.x_slice_num

    def _build_slicer(self, macro_x_value_range: tuple[int, int]) -> Slicer:
        lo, hi = macro_x_value_range
        return SerialSlicer(
            slice_num=self.config.x_slice_num,
            digit_radix=hi - lo + 1,
        )

    def aggregate(self, code: Tensor) -> Tensor:
        # Shape: [..., M, Sa, G, Q] -> [..., M, G, Q]
        return self.shift_adder.shift_add(code, dim=-3, init_val=None)
