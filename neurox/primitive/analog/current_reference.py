"""Multi-output current reference source — PPA + state, no compute.

See Also:
    docs/reference/primitive/analog/current_reference.md
    docs/system_design/physical_state.md
"""

import torch
from torch import Tensor

from neurox.primitive.nonideality import apply_relative_gaussian

from .base import AnalogBase, AnalogConfig, AnalogPolicy


class IrefConfig(AnalogConfig):
    i_refs__uA: tuple[tuple[float, ...], ...]
    """Nominal reference-current taps, 2-D `[mode][tap]`. Modes have equal
    length and non-negative values; ordering within a mode is not enforced,
    because what a mode means is the consumer's knowledge — a decision ladder
    must ascend, a bank of bias taps need not."""
    tolerance_sigma_relative: float
    """Relative per-instance initial-accuracy σ [dimensionless], applied
    multiplicatively at fabricate time; 0 leaves the exact nominal taps, and a
    zero tap stays exact at any σ."""
    area_per_inst__um2: float
    leakage_per_inst__uW: float
    """Carries all static power, including the always-on bias network that
    generates the references."""

    @property
    def mode_num(self) -> int:
        """Number of quasi-statically selectable tap modes."""
        return len(self.i_refs__uA)

    @property
    def tap_num(self) -> int:
        """Number of taps per mode."""
        return len(self.i_refs__uA[0])

    def validate(self) -> None:

        # --- Reference bank ---

        self._require_non_empty(self.i_refs__uA, "i_refs__uA")
        first_taps = self.i_refs__uA[0]
        for mode, taps in enumerate(self.i_refs__uA):
            self._require_non_empty(taps, f"i_refs__uA[{mode}]")
            self._require_same_len(taps, f"i_refs__uA[{mode}]", first_taps, "i_refs__uA[0]")
            for tap, value in enumerate(taps):
                self._require_non_neg(value, f"i_refs__uA[{mode}][{tap}]")

        # --- Tolerance and PPA ---

        self._require_non_neg(self.tolerance_sigma_relative, "tolerance_sigma_relative")
        self._require_non_neg(self.area_per_inst__um2, "area_per_inst__um2")
        self._require_non_neg(self.leakage_per_inst__uW, "leakage_per_inst__uW")


class IrefPolicy(AnalogPolicy):
    tolerance: bool
    """Apply the per-instance initial-accuracy spread
    `tolerance_sigma_relative` at fabricate time."""


class Iref(AnalogBase[IrefConfig, IrefPolicy]):
    """Fabricate-only multi-output current reference with static tolerance.

    A pure identity source: one static `[mode][tap]` bank per physical
    instance, sampled once at fabricate time and read back through
    `i_out__uA`. The source has no forward path and no per-call noise; its
    identity is shared and never resampled.
    """

    # === Nominal buffers ===

    _nominal_i_refs__uA: Tensor  # Shape: [mode_num, tap_num]

    # === Fabricated state ===

    _i_refs__uA: Tensor  # Shape: [*inst_shape, mode_num, tap_num]

    def __init__(
        self,
        *,
        config: IrefConfig,
        policy: IrefPolicy,
        inst_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
    ) -> None:
        super().__init__(config=config, policy=policy, inst_shape=inst_shape)
        self._register_fabrication_buffers(dtype=dtype)

    @property
    def _area_per_inst__um2(self) -> float:
        return self.config.area_per_inst__um2

    @property
    def _leakage_per_inst__uW(self) -> float:
        return self.config.leakage_per_inst__uW

    def _register_fabrication_buffers(self, *, dtype: torch.dtype) -> None:
        self.register_buffer(
            "_nominal_i_refs__uA",
            torch.tensor(self.config.i_refs__uA, dtype=dtype),
            persistent=False,
        )

    @property
    def mode_num(self) -> int:
        return self.config.mode_num

    @property
    def tap_num(self) -> int:
        return self.config.tap_num

    def _sample_fabrication_variation(self) -> None:
        # Clone before expanding: with the tolerance off the fabricated state stays a
        # view of this tensor, which must not alias the registered nominal buffer.
        i_refs__uA = self._nominal_i_refs__uA.clone().expand(*self.inst_shape, self.mode_num, self.tap_num)
        self._i_refs__uA = apply_relative_gaussian(
            i_refs__uA,
            self.config.tolerance_sigma_relative,
            enabled=self.policy.tolerance,
        )

    @property
    def i_out__uA(self) -> Tensor:
        """Fabricated reference-current bank, post static tolerance.

        Read-only view over the fabricated buffer, valid after `fabricate()`:
        one physical identity per instance. Mode selection, call-shape
        broadcasting, and per-access dynamic noise lie outside this source.
        Shape: `[*inst_shape, mode_num, tap_num]`.
        """
        return self._i_refs__uA
