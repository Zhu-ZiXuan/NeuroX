"""Ideal current subtractor — magnitude-and-sign current-difference primitive.

See also:
    docs/reference/primitive/analog/current_subtractor.md
"""

from dataclasses import dataclass
from typing import ClassVar

import torch
from torch import Tensor

from neurox.primitive.analog.base import AnalogBase, AnalogConfig, AnalogPolicy
from neurox.primitive.nonideality import apply_gaussian


@dataclass(frozen=True, kw_only=True)
class CurrentSubtractorConfig(AnalogConfig):
    """Immutable configuration for :class:`CurrentSubtractor`.

    Attributes:
        gain: Dimensionless nominal current gain of the subtraction stage.
        mismatch_sigma_relative: Relative σ of the static per-instance
            subtracted-leg ratio mismatch [dimensionless]; ``0`` leaves the
            exact unit ratio.
        offset_sigma__uA: σ of the static per-instance sign-comparator
            input-referred current offset [uA]; ``0`` leaves a zero offset.
    """

    # --- Subtraction gain ---
    gain: float

    # --- Mismatch ---
    mismatch_sigma_relative: float

    # --- Sign-comparator offset ---
    offset_sigma__uA: float

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        self.validate_gain()
        self.validate_sigmas()

    def validate_gain(self) -> None:
        self._require_pos(self.gain, "gain")

    def validate_sigmas(self) -> None:
        self._require_non_neg(self.mismatch_sigma_relative, "mismatch_sigma_relative")
        self._require_non_neg(self.offset_sigma__uA, "offset_sigma__uA")


@dataclass(frozen=True)
class CurrentSubtractorPolicy(AnalogPolicy):
    """Per-source toggles selecting which CurrentSubtractor nonidealities are active.

    Attributes:
        mismatch: Apply the static per-instance subtracted-leg ratio mismatch
            with σ ``mismatch_sigma_relative``; ``False`` leaves the exact unit
            ratio.
        offset: Apply the static per-instance sign-comparator input-referred
            current offset with σ ``offset_sigma__uA``; ``False`` leaves a zero
            offset.
    """

    mismatch: bool
    offset: bool


class CurrentSubtractor(AnalogBase[CurrentSubtractorConfig, CurrentSubtractorPolicy]):
    """Single-ended current subtractor — magnitude-and-sign difference primitive.

    Emits the gained magnitude of the current difference plus a direction bit:
    ``i_diff = gain·|i_a - ratio·i_b + offset|`` with ``sign`` set where the
    subtracted ``i_b`` leg dominates. The subtracted leg carries a static
    per-instance multiplicative ratio mismatch (relative σ
    ``mismatch_sigma_relative``) and the sign comparator a static per-instance
    additive input-referred offset (σ ``offset_sigma__uA``), both fixed at
    fabrication and off unless their policy toggle is on.

    Args:
        config: Concrete configuration dataclass.
        policy: Per-source nonideality enable flags.
        inst_shape: Per-instance fabrication shape.
        dtype: Tensor dtype for internal buffers.
        T__K: Operating temperature.
    """

    # Non-reporter: embedded primitive whose static PPA rolls up to the owning block.
    reports_static_ppa: ClassVar[bool] = False

    ratio_mismatch: Tensor
    offset__uA: Tensor

    def __init__(
        self,
        *,
        config: CurrentSubtractorConfig,
        policy: CurrentSubtractorPolicy,
        inst_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
    ) -> None:
        super().__init__(config=config, policy=policy, inst_shape=inst_shape)
        self.dtype = dtype
        self.T__K = T__K

        # Held static per-instance subtracted-leg ratio multiplier; unit ratio
        # when the mismatch is off.
        self.register_buffer(
            "ratio_mismatch",
            torch.ones(inst_shape, dtype=dtype),
            persistent=False,
        )
        # Held static per-instance sign-comparator input-referred offset [uA];
        # zero when the offset is off.
        self.register_buffer(
            "offset__uA",
            torch.zeros(inst_shape, dtype=dtype),
            persistent=False,
        )

    def _sample_fabricate_mismatch(self) -> None:
        """Sample the static per-instance subtractor mismatch over ``inst_shape``.

        Draws each element's fixed mismatch once: the subtracted-leg ratio, a
        multiplicative ``1 + N(0, mismatch_sigma_relative)``, and the
        sign-comparator input-referred offset, an additive
        ``N(0, offset_sigma__uA)``. Each toggle off leaves the identity — unit
        ratio, zero offset.
        """
        self.ratio_mismatch = apply_gaussian(
            torch.ones_like(self.ratio_mismatch),
            self.config.mismatch_sigma_relative,
            enabled=self.policy.mismatch,
        )
        self.offset__uA = apply_gaussian(
            torch.zeros_like(self.offset__uA),
            self.config.offset_sigma__uA,
            enabled=self.policy.offset,
        )

    def subtract(self, i_a__uA: Tensor, i_b__uA: Tensor) -> tuple[Tensor, Tensor]:
        """Emit the gained magnitude and sign of the two legs' current difference.

        Scales the subtracted ``i_b`` leg by the static per-instance ratio (unit
        when ``mismatch`` is off) and adds the static sign-comparator offset
        (zero when ``offset`` is off) before taking the gained absolute
        difference.

        Args:
            i_a__uA: Added leg current, shape ``(*leading, *inst_shape)``.
            i_b__uA: Subtracted leg current, same shape as ``i_a__uA``.

        Returns:
            Tuple ``(i_diff__uA, sign)`` of the non-negative magnitude
            ``gain * |i_a - ratio_mismatch * i_b + offset|`` and the boolean
            direction bit ``sign`` (``True`` where the subtracted leg dominates),
            both shaped like the inputs.
        """
        delta__uA = i_a__uA - self.ratio_mismatch * i_b__uA + self.offset__uA
        i_diff__uA = self.config.gain * delta__uA.abs()  # physical unipolar magnitude
        sign = delta__uA < 0  # True => i_b leg dominates
        return i_diff__uA, sign
