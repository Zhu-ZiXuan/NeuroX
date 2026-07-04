"""Multi-output current reference source — PPA + state, no compute.

See also:
    docs/reference/analog/current_reference.md
"""

from dataclasses import dataclass

import torch
from torch import Tensor

from neurox.common.circuit import CircuitBase, CircuitConfig


@dataclass(frozen=True, kw_only=True)
class CurrentReferenceConfig(CircuitConfig):
    """Immutable configuration for :class:`CurrentReference`.

    Attributes:
        i_refs__uA: Nominal reference-current taps. One module
            sources ``len(i_refs__uA)`` independent taps; the taps are
            unordered. Each tap is non-negative; 0 uA denotes a
            ground/rail reference (relative noise * 0 == 0, so a 0 tap
            stays stable and exact). A TOML array loads straight into this
            tuple.
        tolerance_sigma_relative: Relative per-instance initial-accuracy
            σ [dimensionless], applied multiplicatively at fabricate
            time and gated by the ``tolerance`` policy; ``0`` leaves the
            exact nominal taps.
        noise_sigma_relative: Relative per-call noise σ
            [dimensionless], applied multiplicatively at snapshot time
            and gated by the ``noise`` policy; ``0`` leaves the taps
            noise-free.
        area_per_inst__um2: Silicon area per fabricated instance.
        leakage_per_inst__uW: Static leakage per instance; carries
            all static power, including the always-on bias network that
            generates the references.
    """

    # --- Reference taps ---
    i_refs__uA: tuple[float, ...]

    # --- Initial accuracy (fabricate-time, per-instance) ---
    tolerance_sigma_relative: float

    # --- Runtime noise (per-call) ---
    noise_sigma_relative: float

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        self.validate_taps()
        self.validate_noise()
        self.validate_ppa()

    def validate_taps(self) -> None:
        self._require_min_length(self.i_refs__uA, 1, "i_refs__uA")
        for i, v in enumerate(self.i_refs__uA):
            self._require_nonneg(v, f"i_refs__uA[{i}]")

    def validate_noise(self) -> None:
        self._require_nonneg(self.tolerance_sigma_relative, "tolerance_sigma_relative")
        self._require_nonneg(self.noise_sigma_relative, "noise_sigma_relative")


@dataclass(frozen=True)
class CurrentReferencePolicy:
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
            tolerance + noise, shape ``(*inst_shape, num_refs)``.
    """

    i_refs__uA: Tensor


class CurrentReference(CircuitBase[CurrentReferenceConfig]):
    """Multi-output current reference source — PPA + state, no compute.

    A behavioural reference: it sources one or more nominal current taps
    and exists to (1) carry the reference's static PPA — silicon area
    plus the always-on bias power folded into ``leakage_per_inst__uW`` —
    and (2) hand downstream blocks the actual tap values through a
    per-call snap, read back through :meth:`i_ref__uA`.
    It performs no transport, copy, or solve, and emits neither dynamic
    energy nor latency: its entire hardware cost is static and is
    collected by the profiler's static walk over ``CircuitBase``.

    The bias power that generates the reference currents is static and
    is folded into ``leakage_per_inst__uW`` — it is not derived from the
    tap values, so the current and voltage references share one
    PPA stance.

    Two nonidealities perturb the taps. The per-instance initial
    accuracy is a static spread sampled once at ``fabricate`` time
    (``tolerance``); per-call noise is resampled every ``snapshot``
    (``noise``). Both are relative (multiplicative), so a single σ
    applies uniformly across taps of differing magnitude.

    Args:
        config: Concrete configuration dataclass.
        policy: Per-source nonideality enable flags.
        name: Hierarchical instance name used by the profiler.
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
        name: str,
        inst_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
    ) -> None:
        super().__init__(config=config, name=name, inst_shape=inst_shape)
        self.policy = policy
        self.dtype = dtype
        self.T__K = T__K

        nominal_i_refs__uA = torch.tensor(config.i_refs__uA, dtype=dtype)
        self.register_buffer("nominal_i_refs__uA", nominal_i_refs__uA, persistent=False)
        # Actual per-instance taps before any fabricate() call: the
        # broadcast nominal. fabricate() resamples the static tolerance.
        self.register_buffer(
            "i_refs__uA",
            nominal_i_refs__uA.expand(*inst_shape, self.num_refs).clone(),
            persistent=False,
        )

    @property
    def num_refs(self) -> int:
        """Number of reference taps sourced by this module."""
        return len(self.config.i_refs__uA)

    def _sample_fabricate_mismatch(self) -> None:
        """Resample the per-instance initial-accuracy spread at ``(*inst_shape, num_refs)``."""
        base = self.nominal_i_refs__uA.expand(*self._inst_shape, self.num_refs)
        if self.policy.tolerance:
            self.i_refs__uA = base * (1.0 + torch.randn_like(base) * self.config.tolerance_sigma_relative)
        else:
            self.i_refs__uA = base.clone()

    def snapshot(self) -> CurrentReferenceSnap:
        """Sample one per-call reference snap.

        Reads the fabricated per-instance taps and applies the per-call
        relative noise (gated by the ``noise`` policy). No external
        shape: a reference's output is intrinsically ``(*inst_shape,
        num_refs)`` — a consumer picks a tap and broadcasts it onto its
        own grid.

        Returns:
            Per-call snap carrying the actual reference-current taps.
        """
        i = self.i_refs__uA
        i = i * (1.0 + torch.randn_like(i) * self.config.noise_sigma_relative) if self.policy.noise else i.clone()
        return CurrentReferenceSnap(i_refs__uA=i)

    def i_ref__uA(self, snap: CurrentReferenceSnap) -> Tensor:
        """Read all reference-current taps from a per-call snap.

        The encapsulated read path: returns every tap (count is
        ``num_refs``) so a consumer selects one by index and broadcasts
        it onto its own grid. Reading through the snap rather than the
        post-fabricate buffer guarantees the per-call ``noise`` is
        included. Pairs with :meth:`snapshot`.

        Args:
            snap: Per-call snap returned by :meth:`snapshot`.

        Returns:
            Reference-current taps, shape ``(*inst_shape, num_refs)``.
        """
        return snap.i_refs__uA
