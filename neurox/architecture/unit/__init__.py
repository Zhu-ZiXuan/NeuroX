"""User-facing compute units that implement neural-network operators."""

from . import cim, ideal
from .base import UnitBase
from .conv2d import Conv2dUnit
from .ideal import (
    IdealConv2dUnit,
    IdealConv2dUnitConfig,
    IdealConv2dUnitPolicy,
    IdealLinearUnit,
    IdealLinearUnitConfig,
    IdealLinearUnitPolicy,
)
from .linear import LinearUnit
from .matmul_mapping import (
    BlockSlotRouting,
    InputActivationPlan,
    MatmulPlacementPlan,
    make_activation_group_mask,
    make_block_slot_routing,
    make_input_activation_plan,
    make_matmul_placement_plan,
)

__all__ = [
    "cim",
    "ideal",
    "BlockSlotRouting",
    "Conv2dUnit",
    "IdealConv2dUnit",
    "IdealConv2dUnitConfig",
    "IdealConv2dUnitPolicy",
    "IdealLinearUnit",
    "IdealLinearUnitConfig",
    "IdealLinearUnitPolicy",
    "InputActivationPlan",
    "LinearUnit",
    "MatmulPlacementPlan",
    "UnitBase",
    "make_activation_group_mask",
    "make_block_slot_routing",
    "make_input_activation_plan",
    "make_matmul_placement_plan",
]
