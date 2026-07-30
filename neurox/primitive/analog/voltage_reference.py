"""Multi-output voltage reference source — PPA + state, no compute.

See also:
    docs/reference/primitive/analog/voltage_reference.md
"""

from dataclasses import dataclass

import torch
from torch import Tensor

from neurox.primitive.nonideality import apply_relative_gaussian

from .base import AnalogBase, AnalogConfig, AnalogPolicy


class VrefConfig(AnalogConfig):
    """Immutable configuration for :class:`Vref`.

    Attributes:
        v_refs__V: Nominal reference-voltage taps, 2-D ``[mode][tap]``.
            Modes have equal length and non-negative values; ordering
            within a mode is not enforced, because what a mode means is
            the consumer's knowledge. A single-tap single-mode bank is
            the degenerate ``[[v]]``. A zero tap remains exact under
            relative noise.
        tolerance_sigma_relative: Relative per-instance initial-accuracy
            σ [dimensionless], applied multiplicatively at fabricate
            time; ``0`` leaves the exact nominal taps.
        noise_sigma_relative: Relative per-call noise σ
            [dimensionless], applied multiplicatively at snapshot time;
            ``0`` leaves the taps noise-free.
        area_per_inst__um2: Silicon area per fabricated instance.
        leakage_per_inst__uW: Static leakage per instance; carries
            all static power, including the always-on bias network that
            generates the references.
    """

    v_refs__V: tuple[tuple[float, ...], ...]
    tolerance_sigma_relative: float
    noise_sigma_relative: float
    area_per_inst__um2: float
    leakage_per_inst__uW: float

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

        # --- Noise and PPA ---

        self._require_non_neg(self.tolerance_sigma_relative, "tolerance_sigma_relative")
        self._require_non_neg(self.noise_sigma_relative, "noise_sigma_relative")
        self._require_non_neg(self.area_per_inst__um2, "area_per_inst__um2")
        self._require_non_neg(self.leakage_per_inst__uW, "leakage_per_inst__uW")


class VrefPolicy(AnalogPolicy):
    """Per-source toggles selecting which Vref nonidealities are active.

    Attributes:
        tolerance: Apply the per-instance initial-accuracy spread
            ``tolerance_sigma_relative`` at fabricate time.
        noise: Apply the per-call noise ``noise_sigma_relative`` at
            snapshot time.
    """

    tolerance: bool
    noise: bool


@dataclass(frozen=True)
class VrefSnap:
    """One sampled reference snap.

    Attributes:
        v_refs__V: Actual reference-voltage taps of the selected mode,
            post tolerance + noise, at the requested call shape. The mode
            axis is resolved away by :meth:`Vref.snapshot`.
            Shape: ``[..., tap_num]``.
    """

    v_refs__V: Tensor


class Vref(AnalogBase[VrefConfig, VrefPolicy]):
    """Multi-output voltage reference with static tolerance and runtime noise.

    Args:
        config: Concrete configuration dataclass.
        policy: Per-source nonideality enable flags.
        inst_shape: Per-instance fabrication shape.
        dtype: Tensor dtype for internal buffers.
        T__K: Operating temperature.
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
        """Register immutable tensors used as fabrication sources."""
        self.register_buffer(
            "_nominal_v_refs__V",
            torch.tensor(self.config.v_refs__V, dtype=dtype),
            persistent=False,
        )

    @property
    def mode_num(self) -> int:
        """Number of quasi-statically selectable tap modes."""
        return self.config.mode_num

    @property
    def tap_num(self) -> int:
        """Number of taps per mode."""
        return self.config.tap_num

    def _sample_fabricate_mismatch(self) -> None:
        v_refs__V = self._nominal_v_refs__V.clone().expand(*self.inst_shape, self.mode_num, self.tap_num)
        self._v_refs__V = apply_relative_gaussian(
            v_refs__V,
            self.config.tolerance_sigma_relative,
            enabled=self.policy.tolerance,
        )

    def snapshot(self, *, mode: int, shape: tuple[int, ...]) -> VrefSnap:
        """Select one mode and sample it with per-call noise.

        The mode selects the taps here rather than in the caller, so the
        bank layout stays private. The static mismatch drawn at fabricate
        time spans ``inst_shape`` alone, while this per-call noise spans
        the full ``shape``: the fabricated taps are one physical source,
        but every position of a call is a distinct instant or a distinct
        point along the distribution net, each carrying its own draw.

        Args:
            mode: Mode index into the ``[mode][tap]`` bank. Range checking
                belongs to the consumer that owns the mode set.
            shape: Full output shape, ending in ``tap_num``.

        Returns:
            Per-call snap carrying the actual reference-voltage taps.
        """
        # Shape: [*inst_shape, mode_num, tap_num] -> [..., *inst_shape, tap_num]
        v_refs__V = self._v_refs__V[..., mode, :].expand(shape)
        v_refs__V = apply_relative_gaussian(
            v_refs__V,
            self.config.noise_sigma_relative,
            enabled=self.policy.noise,
        )
        return VrefSnap(v_refs__V=v_refs__V)
