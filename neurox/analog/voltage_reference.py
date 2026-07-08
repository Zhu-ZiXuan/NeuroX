"""Multi-output voltage reference source — PPA + state, no compute.

See also:
    docs/reference/analog/voltage_reference.md
"""

from dataclasses import dataclass

import torch
from torch import Tensor

from neurox.common.circuit import CircuitBase, CircuitConfig


@dataclass(frozen=True, kw_only=True)
class VoltageReferenceConfig(CircuitConfig):
    """Immutable configuration for :class:`VoltageReference`.

    Attributes:
        v_refs__V: Nominal reference-voltage taps. One module
            sources ``len(v_refs__V)`` independent taps; the taps are
            unordered (unlike an ADC's ordered mode anchors). Each tap is
            non-negative; 0 V denotes a ground/rail reference (relative
            noise * 0 == 0, so a 0 tap stays stable and exact). A TOML
            array loads straight into this tuple.
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

    # --- Reference taps ---
    v_refs__V: tuple[float, ...]

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
        self._require_min_length(self.v_refs__V, 1, "v_refs__V")
        for i, v in enumerate(self.v_refs__V):
            self._require_non_neg(v, f"v_refs__V[{i}]")

    def validate_noise(self) -> None:
        self._require_non_neg(self.tolerance_sigma_relative, "tolerance_sigma_relative")
        self._require_non_neg(self.noise_sigma_relative, "noise_sigma_relative")


@dataclass(frozen=True)
class VoltageReferencePolicy:
    """Per-source toggles selecting which VoltageReference nonidealities are active.

    Attributes:
        tolerance: Apply the per-instance initial-accuracy spread
            ``tolerance_sigma_relative`` at fabricate time.
        noise: Apply the per-call noise ``noise_sigma_relative`` at
            snapshot time.
    """

    tolerance: bool
    noise: bool


@dataclass(frozen=True)
class VoltageReferenceSnap:
    """One sampled reference snap.

    Attributes:
        v_refs__V: Actual reference-voltage taps, post
            tolerance + noise, shape ``(*inst_shape, num_refs)``.
    """

    v_refs__V: Tensor


class VoltageReference(CircuitBase[VoltageReferenceConfig]):
    """Multi-output voltage reference source — PPA + state, no compute.

    A behavioural reference: it sources one or more nominal voltage taps
    and exists to (1) carry the reference's static PPA — silicon area
    plus the always-on bias power folded into ``leakage_per_inst__uW`` —
    and (2) hand downstream blocks the actual tap values through a
    per-call snap, read back through :meth:`v_ref__V`.
    It performs no transport, copy, or solve, and emits neither dynamic
    energy nor latency: its entire hardware cost is static and is
    collected by the profiler's static walk over ``CircuitBase``.

    Two nonidealities perturb the taps. The per-instance initial
    accuracy is a static spread sampled once at ``fabricate`` time
    (``tolerance``); per-call noise is resampled every ``snapshot``
    (``noise``). Both are relative (multiplicative), so a single σ
    applies uniformly across taps of differing magnitude.

    The block exposes no ``v_supply__V`` field, no bias current, and no
    ``solve_dc`` — the same behavioural-source stance as
    :class:`~neurox.analog.VoltageDriver`.

    Args:
        config: Concrete configuration dataclass.
        policy: Per-source nonideality enable flags.
        name: Hierarchical instance name used by the profiler.
        inst_shape: Per-instance fabrication shape.
        dtype: Tensor dtype for internal buffers.
        T__K: Operating temperature.
    """

    nominal_v_refs__V: Tensor
    v_refs__V: Tensor

    def __init__(
        self,
        *,
        config: VoltageReferenceConfig,
        policy: VoltageReferencePolicy,
        name: str,
        inst_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
    ) -> None:
        super().__init__(config=config, name=name, inst_shape=inst_shape)
        self.policy = policy
        self.dtype = dtype
        self.T__K = T__K

        nominal_v_refs__V = torch.tensor(config.v_refs__V, dtype=dtype)
        self.register_buffer("nominal_v_refs__V", nominal_v_refs__V, persistent=False)
        # Actual per-instance taps before any fabricate() call: the
        # broadcast nominal. fabricate() resamples the static tolerance.
        self.register_buffer(
            "v_refs__V",
            nominal_v_refs__V.expand(*inst_shape, self.num_refs).clone(),
            persistent=False,
        )

    @property
    def num_refs(self) -> int:
        """Number of reference taps sourced by this module."""
        return len(self.config.v_refs__V)

    def _sample_fabricate_mismatch(self) -> None:
        """Resample the per-instance initial-accuracy spread at ``(*inst_shape, num_refs)``."""
        base = self.nominal_v_refs__V.expand(*self._inst_shape, self.num_refs)
        if self.policy.tolerance:
            self.v_refs__V = base * (1.0 + torch.randn_like(base) * self.config.tolerance_sigma_relative)
        else:
            self.v_refs__V = base.clone()

    def snapshot(self) -> VoltageReferenceSnap:
        """Sample one per-call reference snap.

        Reads the fabricated per-instance taps and applies the per-call
        relative noise (gated by the ``noise`` policy). No external
        shape: a reference's output is intrinsically ``(*inst_shape,
        num_refs)`` — a consumer picks a tap and broadcasts it onto its
        own grid.

        Returns:
            Per-call snap carrying the actual reference-voltage taps.
        """
        v = self.v_refs__V
        v = v * (1.0 + torch.randn_like(v) * self.config.noise_sigma_relative) if self.policy.noise else v.clone()
        return VoltageReferenceSnap(v_refs__V=v)

    def v_ref__V(self, snap: VoltageReferenceSnap) -> Tensor:
        """Read all reference-voltage taps from a per-call snap.

        The encapsulated read path: returns every tap (count is
        ``num_refs``) so a consumer selects one by index and broadcasts
        it onto its own grid. Reading through the snap rather than the
        post-fabricate buffer guarantees the per-call ``noise`` is
        included. Pairs with :meth:`snapshot`.

        Args:
            snap: Per-call snap returned by :meth:`snapshot`.

        Returns:
            Reference-voltage taps, shape ``(*inst_shape, num_refs)``.
        """
        return snap.v_refs__V
