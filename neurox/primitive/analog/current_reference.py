"""Multi-output current reference source — PPA + state, no compute.

See also:
    docs/reference/primitive/analog/current_reference.md
"""

from dataclasses import dataclass

import torch
from torch import Tensor

from neurox.primitive.nonideality import apply_relative_gaussian

from .base import AnalogBase, AnalogConfig, AnalogPolicy


class IrefConfig(AnalogConfig):
    """Immutable configuration for :class:`Iref`.

    Attributes:
        i_refs__uA: Nominal reference-current taps, 2-D ``[mode][tap]``.
            Modes have equal length and non-negative values; ordering
            within a mode is not enforced, because what a mode means is
            the consumer's knowledge — a decision ladder must ascend, a
            bank of bias taps need not. A zero tap remains exact under
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

    i_refs__uA: tuple[tuple[float, ...], ...]
    tolerance_sigma_relative: float
    noise_sigma_relative: float
    area_per_inst__um2: float
    leakage_per_inst__uW: float

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

        self._require_min_length(self.i_refs__uA, 1, "i_refs__uA")
        tap_num = self.tap_num
        for mode, taps in enumerate(self.i_refs__uA):
            self._require_min_length(taps, 1, f"i_refs__uA[{mode}]")
            if len(taps) != tap_num:
                raise ValueError(
                    f"require: equal tap lengths in i_refs__uA; mode {mode} has {len(taps)} tap(s), mode 0 has {tap_num}"
                )
            for tap, value in enumerate(taps):
                self._require_non_neg(value, f"i_refs__uA[{mode}][{tap}]")

        # --- Noise and PPA ---

        self._require_non_neg(self.tolerance_sigma_relative, "tolerance_sigma_relative")
        self._require_non_neg(self.noise_sigma_relative, "noise_sigma_relative")
        self._require_non_neg(self.area_per_inst__um2, "area_per_inst__um2")
        self._require_non_neg(self.leakage_per_inst__uW, "leakage_per_inst__uW")


class IrefPolicy(AnalogPolicy):
    """Per-source toggles selecting which Iref nonidealities are active.

    Attributes:
        tolerance: Apply the per-instance initial-accuracy spread
            ``tolerance_sigma_relative`` at fabricate time.
        noise: Apply the per-call noise ``noise_sigma_relative`` at
            snapshot time.
    """

    tolerance: bool
    noise: bool


@dataclass(frozen=True)
class IrefSnap:
    """One sampled reference snap.

    Attributes:
        i_refs__uA: Actual reference-current taps of the selected mode,
            post tolerance + noise, at the requested call shape. The mode
            axis is resolved away by :meth:`Iref.snapshot`.
            Shape: ``[..., tap_num]``.
    """

    i_refs__uA: Tensor


class Iref(AnalogBase[IrefConfig, IrefPolicy]):
    """Multi-output current reference with static tolerance and runtime noise.

    Args:
        config: Concrete configuration dataclass.
        policy: Per-source nonideality enable flags.
        inst_shape: Per-instance fabrication shape.
        dtype: Tensor dtype for internal buffers.
        T__K: Operating temperature.
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
        """Register immutable tensors used as fabrication sources."""
        self.register_buffer(
            "_nominal_i_refs__uA",
            torch.tensor(self.config.i_refs__uA, dtype=dtype),
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
        i_refs__uA = self._nominal_i_refs__uA.clone().expand(*self.inst_shape, self.mode_num, self.tap_num)
        self._i_refs__uA = apply_relative_gaussian(
            i_refs__uA,
            self.config.tolerance_sigma_relative,
            enabled=self.policy.tolerance,
        )

    def snapshot(self, *, mode: int, shape: tuple[int, ...]) -> IrefSnap:
        """Select one mode and sample it with per-call noise.

        The mode selects the taps here rather than in the caller, so the
        bank layout stays private and no operating-mode identity travels
        further downstream. The static mismatch drawn at fabricate time
        spans ``inst_shape`` alone, while this per-call noise spans the
        full ``shape``: the fabricated taps are one physical source, but
        every position of a call is a distinct instant or a distinct
        mirrored branch, each carrying its own draw.

        Args:
            mode: Mode index into the ``[mode][tap]`` bank. Range checking
                belongs to the consumer that owns the mode set.
            shape: Full output shape, ending in ``tap_num``.

        Returns:
            Per-call snap carrying the actual reference-current taps.
        """
        # Shape: [*inst_shape, mode_num, tap_num] -> [..., *inst_shape, tap_num]
        i_refs__uA = self._i_refs__uA[..., mode, :].expand(shape)
        i_refs__uA = apply_relative_gaussian(
            i_refs__uA,
            self.config.noise_sigma_relative,
            enabled=self.policy.noise,
        )
        return IrefSnap(i_refs__uA=i_refs__uA)
