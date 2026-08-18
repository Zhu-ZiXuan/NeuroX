"""Multi-output voltage reference source — PPA + state, no compute.

See Also:
    docs/reference/primitive/analog/voltage_reference.md
    docs/system_design/physical_state.md
"""

import torch
from torch import Tensor

from neurox.primitive.nonideality import apply_relative_gaussian

from .base import AnalogBase, AnalogConfig, AnalogPolicy


class VrefConfig(AnalogConfig):
    """Immutable configuration for `Vref`."""

    v_refs__V: tuple[tuple[float, ...], ...]
    """Nominal reference-voltage taps, 2-D `[mode][tap]`. Modes have equal
    length and non-negative values; ordering within a mode is not enforced,
    because what a mode means is the consumer's knowledge. A single-tap
    single-mode bank is the degenerate `[[v]]`."""
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
        return len(self.v_refs__V)

    @property
    def tap_num(self) -> int:
        """Number of taps per mode."""
        return len(self.v_refs__V[0])

    def validate(self) -> None:

        # --- Reference bank ---

        self._require_min_length(self.v_refs__V, 1, "v_refs__V")
        tap_num = self.tap_num
        for mode, taps in enumerate(self.v_refs__V):
            self._require_min_length(taps, 1, f"v_refs__V[{mode}]")
            if len(taps) != tap_num:
                raise ValueError(
                    f"require: equal tap lengths in v_refs__V; mode {mode} has {len(taps)} tap(s), mode 0 has {tap_num}"
                )
            for tap, value in enumerate(taps):
                self._require_non_neg(value, f"v_refs__V[{mode}][{tap}]")

        # --- Tolerance and PPA ---

        self._require_non_neg(self.tolerance_sigma_relative, "tolerance_sigma_relative")
        self._require_non_neg(self.area_per_inst__um2, "area_per_inst__um2")
        self._require_non_neg(self.leakage_per_inst__uW, "leakage_per_inst__uW")


class VrefPolicy(AnalogPolicy):
    """Per-source toggle selecting whether the Vref tolerance is active."""

    tolerance: bool
    """Apply the per-instance initial-accuracy spread
    `tolerance_sigma_relative` at fabricate time."""


class Vref(AnalogBase[VrefConfig, VrefPolicy]):
    """Fabricate-only multi-output voltage reference with static tolerance.

    A pure identity source: one static `[mode][tap]` bank per physical
    instance, sampled once at fabricate time and read back through `v_out__V`.
    No forward path and no per-call noise — dynamic per-access variation is a
    consuming driver's own law, not this source's; the source's identity is
    shared and never resampled.
    """

    # === Nominal buffers ===

    _nominal_v_refs__V: Tensor  # Shape: [mode_num, tap_num]

    # === Fabricated state ===

    _v_refs__V: Tensor  # Shape: [*inst_shape, mode_num, tap_num]

    def __init__(
        self,
        *,
        config: VrefConfig,
        policy: VrefPolicy,
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
            "_nominal_v_refs__V",
            torch.tensor(self.config.v_refs__V, dtype=dtype),
            persistent=False,
        )

    @property
    def mode_num(self) -> int:
        return self.config.mode_num

    @property
    def tap_num(self) -> int:
        return self.config.tap_num

    def _sample_fabricate_mismatch(self) -> None:
        # Clone before expanding: with the tolerance off the fabricated state stays a
        # view of this tensor, which must not alias the registered nominal buffer.
        v_refs__V = self._nominal_v_refs__V.clone().expand(*self.inst_shape, self.mode_num, self.tap_num)
        self._v_refs__V = apply_relative_gaussian(
            v_refs__V,
            self.config.tolerance_sigma_relative,
            enabled=self.policy.tolerance,
        )

    @property
    def v_out__V(self) -> Tensor:
        """Fabricated reference-voltage bank, post static tolerance.

        Read-only view over the fabricated buffer, valid after `fabricate()`:
        one physical identity per instance. A consumer selects its mode and
        broadcasts the result onto its own call shape by view; that broadcast,
        and any per-access dynamic noise on top of it, is the consuming
        driver's concern.
        Shape: `[*inst_shape, mode_num, tap_num]`.
        """
        return self._v_refs__V
