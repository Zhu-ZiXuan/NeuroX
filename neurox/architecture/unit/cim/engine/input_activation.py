"""Input-phase scheduling and P-axis aggregation.

See Also:
    docs/reference/architecture/unit/cim/engine/input_activation.md
"""

from __future__ import annotations

from typing import ClassVar

import torch
from torch import Tensor

from neurox.architecture.unit.matmul_mapping import make_activation_group_mask, make_input_activation_plan
from neurox.common import ConfigBase, ModuleBase, PolicyBase
from neurox.primitive.digital import AccumulatorConfig, DigitalPolicy, SerialAccumulator


class InputActivationStageConfig(ConfigBase):
    phase_accumulator_config: AccumulatorConfig
    """Accumulator folding the P axis."""


class InputActivationStagePolicy(PolicyBase):
    pass


class InputActivationStage(ModuleBase[InputActivationStageConfig, InputActivationStagePolicy]):
    """Pair max-active input scheduling with P-axis aggregation."""

    is_profile_target: ClassVar[bool] = False

    # === Functional buffers ===

    _active_input_mask: Tensor  # Shape: [P, L]

    def __init__(
        self,
        *,
        config: InputActivationStageConfig,
        policy: InputActivationStagePolicy,
        input_block_size: int,
        max_active_num: int,
        macro_plane_num: int,
        input_tile_num: int,
        macro_group_num: int,
        macro_inst_rank: int,
    ) -> None:
        super().__init__(config=config, policy=policy, inst_shape=())
        activation_plan = make_input_activation_plan(
            input_block_size=input_block_size,
            active_input_limit=max_active_num,
        )
        self._input_phase_num = activation_plan.activation_group_num
        self._input_phase_dim = -(macro_inst_rank + 2)
        self._register_nonpersistent_buffer(
            "_active_input_mask",
            make_activation_group_mask(activation=activation_plan),
        )
        self.phase_accumulator = SerialAccumulator(
            config=config.phase_accumulator_config,
            policy=DigitalPolicy(),
            inst_shape=(
                macro_plane_num,
                input_tile_num,
                macro_group_num,
            ),
        )

    @property
    def input_phase_num(self) -> int:
        """Successive activations one input block is driven as — the P axis."""
        return self._input_phase_num

    def unroll_input_phases(self, x: Tensor) -> Tensor:
        """Split one local input block into CIM input phases.

        The input-phase axis `P` serializes word-line planes, one plane per
        macro access. The macro contract knows no phase — planes in, codes out
        — so `P` reaches it as one more broadcast leading axis.
        """
        # Shape: [P, L] -> [..., P, L]
        mask = self._active_input_mask.reshape(
            *(1,) * (x.ndim - 1),
            self._input_phase_num,
            x.shape[-1],
        )
        # Shape: [..., M, Sx, Sw, Tc, G, L] -> [..., M, Sx, Sw, Tc, G, P, L]
        return torch.where(mask, x.unsqueeze(-2), x.new_zeros(()))

    def accumulate_phases(self, code: Tensor) -> Tensor:
        """Accumulate P-axis codes produced by one logical input block."""
        return self.phase_accumulator.accumulate(code, dim=self._input_phase_dim)
