"""Multi-output current reference source — PPA + state, no compute.

See also:
    docs/reference/primitive/analog/current_reference.md
"""

from dataclasses import dataclass

import torch
from torch import Tensor

from neurox.primitive.analog.base import AnalogBase, AnalogConfig, AnalogPolicy


@dataclass(frozen=True, kw_only=True)
class CurrentReferenceConfig(AnalogConfig):
    """Immutable configuration for :class:`CurrentReference`.

    Attributes:
        i_refs__uA: Nominal reference-current taps, 2-D ``[mode][tap]``.
            One module holds ``mode_num`` quasi-statically selectable tap
            rows of ``tap_num`` taps each; every row is strictly
            increasing and all rows have equal length. Taps are
            non-negative; 0 uA denotes a ground/rail reference (relative
            noise * 0 == 0, so a 0 tap stays stable and exact). A nested
            TOML array loads straight into this tuple-of-tuples.
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

    # --- Reference taps, [mode][tap] ---
    i_refs__uA: tuple[tuple[float, ...], ...]

    # --- Initial accuracy (fabricate-time, per-instance) ---
    tolerance_sigma_relative: float

    # --- Runtime noise (per-call) ---
    noise_sigma_relative: float

    # --- Static PPA ---
    area_per_inst__um2: float
    leakage_per_inst__uW: float

    def __post_init__(self) -> None:
        self.validate()

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


@dataclass(frozen=True)
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
    """Multi-output current reference source — PPA + state, no compute.

    A behavioural reference: it holds a ``[mode_num, tap_num]`` bank of
    nominal current-tap rows and exists to (1) carry the reference's
    static PPA — silicon area plus the always-on bias power folded into
    ``leakage_per_inst__uW`` — and (2) hand downstream blocks the actual
    tap values through a per-call snap, read back through
    :meth:`i_ref__uA`. Mode selection is quasi-static: a consumer indexes
    one mode row and holds it across conversions, so switching modes
    costs no per-conversion energy and the module emits neither dynamic
    energy nor latency — its entire hardware cost is static. It performs
    no transport, copy, or solve.

    Two nonidealities perturb the taps. The per-instance initial
    accuracy is a static spread sampled once at ``fabricate`` time
    (``tolerance``); per-call noise is resampled every ``snapshot``
    (``noise``). Both are relative (multiplicative), so a single σ
    applies uniformly across taps of differing magnitude.

    Args:
        config: Concrete configuration dataclass.
        policy: Per-source nonideality enable flags.
        inst_shape: Per-instance fabrication shape.
        dtype: Tensor dtype for internal buffers.
        T__K: Operating temperature.
    """

    nominal_i_refs__uA: Tensor
    i_refs__uA: Tensor

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
        self.dtype = dtype
        self.T__K = T__K

        # Nominal tap bank, [mode_num, tap_num].
        nominal_i_refs__uA = torch.tensor(config.i_refs__uA, dtype=dtype)
        self.register_buffer("nominal_i_refs__uA", nominal_i_refs__uA, persistent=False)
        # Actual per-instance taps before any fabricate() call: the
        # broadcast nominal. fabricate() resamples the static tolerance.
        self.register_buffer(
            "i_refs__uA",
            nominal_i_refs__uA.expand(*inst_shape, self.mode_num, self.tap_num).clone(),
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
        """Resample the per-instance initial-accuracy spread at ``(*inst_shape, mode_num, tap_num)``."""
        base = self.nominal_i_refs__uA.expand(*self.inst_shape, self.mode_num, self.tap_num)
        if self.policy.tolerance:
            self.i_refs__uA = base * (1.0 + torch.randn_like(base) * self.config.tolerance_sigma_relative)
        else:
            self.i_refs__uA = base.clone()

    def snapshot(self, *, shape: tuple[int, ...] = ()) -> CurrentReferenceSnap:
        """Sample one per-call reference snap.

        Reads the fabricated per-instance taps, expands them to the
        requested ``shape``, and applies the per-call relative noise
        (gated by the ``noise`` policy). Expanding before the noise draw
        lets a noise-enabled snap sample independently at every position
        of ``shape`` instead of broadcasting a single draw.

        Args:
            shape: Full requested snap shape, with intrinsic trailing
                ``(mode_num, tap_num)``; the caller passes
                ``(*inst_shape, mode_num, tap_num)``. Empty leaves the
                taps at their fabricated ``(*inst_shape, mode_num,
                tap_num)`` shape.

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
        """Read all reference-current taps from a per-call snap.

        The encapsulated read path: returns the full tap bank so a
        consumer indexes its quasi-static mode row, selects a tap, and
        broadcasts it onto its own grid. Pairs with :meth:`snapshot`.

        Args:
            snap: Per-call snap returned by :meth:`snapshot`.

        Returns:
            Reference-current taps, shape ``(*inst_shape, mode_num, tap_num)``.
        """
        return snap.i_refs__uA
