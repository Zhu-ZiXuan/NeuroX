"""Multi-output current reference source — PPA + state, no compute.

See also:
    docs/reference/primitive/analog/current_reference.md
"""

from dataclasses import dataclass

import torch
from torch import Tensor

from neurox.primitive.analog.base import AnalogBase, AnalogConfig, AnalogPolicy


class CurrentReferenceConfig(AnalogConfig):
    """Immutable configuration for :class:`CurrentReference`.

    Attributes:
        i_refs__uA: Nominal reference-current taps, 2-D ``[mode][tap]``.
            Rows have equal length and strictly increasing non-negative
            values. A zero tap remains exact under relative noise.
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
        """Number of quasi-statically selectable tap rows."""
        return len(self.i_refs__uA)

    @property
    def tap_num(self) -> int:
        """Number of taps per mode row (equal across rows)."""
        return len(self.i_refs__uA[0])

    def validate(self) -> None:
        self.validate_taps()
        self.validate_noise()
        self.validate_ppa()

    def validate_taps(self) -> None:
        self._require_min_length(self.i_refs__uA, 1, "i_refs__uA")
        tap_num = len(self.i_refs__uA[0])
        for m, row in enumerate(self.i_refs__uA):
            self._require_min_length(row, 1, f"i_refs__uA[{m}]")
            if len(row) != tap_num:
                raise ValueError(
                    f"require: equal row lengths in i_refs__uA; row {m} has {len(row)} tap(s), row 0 has {tap_num}"
                )
            self._require_increasing(row, f"i_refs__uA[{m}]")
            for t, v in enumerate(row):
                self._require_non_neg(v, f"i_refs__uA[{m}][{t}]")

    def validate_noise(self) -> None:
        self._require_non_neg(self.tolerance_sigma_relative, "tolerance_sigma_relative")
        self._require_non_neg(self.noise_sigma_relative, "noise_sigma_relative")

    def validate_ppa(self) -> None:
        self._require_non_neg(self.area_per_inst__um2, "area_per_inst__um2")
        self._require_non_neg(self.leakage_per_inst__uW, "leakage_per_inst__uW")


class CurrentReferencePolicy(AnalogPolicy):
    """Per-source toggles selecting which CurrentReference nonidealities are active.

    Attributes:
        tolerance: Apply the per-instance initial-accuracy spread
            ``tolerance_sigma_relative`` at fabricate time.
        noise: Apply the per-call noise ``noise_sigma_relative`` at
            snapshot time.
    """

    tolerance: bool
    noise: bool


@dataclass(frozen=True)
class CurrentReferenceSnap:
    """One sampled reference snap.

    Attributes:
        i_refs__uA: Actual reference-current taps, post
            tolerance + noise, shape ``(*inst_shape, mode_num, tap_num)``.
    """

    i_refs__uA: Tensor


class CurrentReference(AnalogBase[CurrentReferenceConfig, CurrentReferencePolicy]):
    """Multi-output current reference with static tolerance and runtime noise.

    Args:
        config: Concrete configuration dataclass.
        policy: Per-source nonideality enable flags.
        inst_shape: Per-instance fabrication shape.
        dtype: Tensor dtype for internal buffers.
        T__K: Operating temperature.
    """

    # --- Fabrication source buffers ---

    nominal_i_refs__uA: Tensor

    def __init__(
        self,
        *,
        config: CurrentReferenceConfig,
        policy: CurrentReferencePolicy,
        inst_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
    ) -> None:
        super().__init__(config=config, policy=policy, inst_shape=inst_shape)
        self._area_per_inst__um2 = config.area_per_inst__um2
        self._leakage_per_inst__uW = config.leakage_per_inst__uW
        self._register_fabrication_buffers(dtype=dtype)

    def _register_fabrication_buffers(self, *, dtype: torch.dtype) -> None:
        """Register immutable tensors used as fabrication sources."""
        self.register_buffer(
            "nominal_i_refs__uA",
            torch.tensor(self.config.i_refs__uA, dtype=dtype),
            persistent=False,
        )

    @property
    def mode_num(self) -> int:
        """Number of quasi-statically selectable tap rows."""
        return self.config.mode_num

    @property
    def tap_num(self) -> int:
        """Number of taps per mode row."""
        return self.config.tap_num

    def _sample_fabricate_mismatch(self) -> None:
        base = self.nominal_i_refs__uA.expand(*self.inst_shape, self.mode_num, self.tap_num)
        if self.policy.tolerance:
            self.i_refs__uA = base * (1.0 + torch.randn_like(base) * self.config.tolerance_sigma_relative)
        else:
            self.i_refs__uA = base.clone()

    def snapshot(self, *, shape: tuple[int, ...] = ()) -> CurrentReferenceSnap:
        """Sample reference taps with per-call noise.

        Args:
            shape: Output shape ending in ``(mode_num, tap_num)``.
                An empty tuple preserves the fabricated shape.

        Returns:
            Per-call snap carrying the actual reference-current taps.
        """
        base = self.i_refs__uA
        view = base.expand(shape) if shape else base
        view = (
            view * (1.0 + torch.randn_like(view) * self.config.noise_sigma_relative)
            if self.policy.noise
            else view.clone()
        )
        return CurrentReferenceSnap(i_refs__uA=view)

    def i_ref__uA(self, snap: CurrentReferenceSnap) -> Tensor:
        """Read all reference-current taps from a snap.

        Args:
            snap: Per-call snap returned by :meth:`snapshot`.

        Returns:
            Reference-current taps, shape ``(*inst_shape, mode_num, tap_num)``.
        """
        return snap.i_refs__uA
