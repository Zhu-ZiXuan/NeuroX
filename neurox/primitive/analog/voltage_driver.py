"""Generic Thevenin voltage-source clamp-driver model.

See also:
    docs/reference/primitive/analog/voltage_driver.md
    docs/internals/primitive/analog/voltage_driver.md
"""

from dataclasses import dataclass

import torch
from torch import Tensor

from neurox.common import TensorGroupMixin
from neurox.primitive.nonideality import apply_gaussian

from .base import AnalogBase, AnalogConfig, AnalogPolicy


class VoltageDriverConfig(AnalogConfig):
    """Immutable configuration for `VoltageDriver`."""

    r_out__MOhm: float
    """Series output resistance — the constant clamp slope dVclamp/dI; 0
    recovers the ideal voltage-source limit."""
    offset_sigma__V: float
    """σ of the static systematic per-instance offset on the reference."""
    thermal_sigma__V: float
    """σ of the per-solve Gaussian thermal noise on the reference."""
    energy_per_op__fJ: float
    """Per-column-op interface energy, e.g. one full C·V² interface-node
    precharge cycle; a physical zero is a legitimate value."""
    area_per_inst__um2: float
    leakage_per_inst__uW: float
    """Carries all static power, including any internal amplifier / bias."""

    def validate(self) -> None:

        # --- Source and noise ---

        self._require_non_neg(self.r_out__MOhm, "r_out__MOhm")
        self._require_non_neg(self.offset_sigma__V, "offset_sigma__V")
        self._require_non_neg(self.thermal_sigma__V, "thermal_sigma__V")

        # --- PPA ---

        self._require_non_neg(self.energy_per_op__fJ, "energy_per_op__fJ")
        self._require_non_neg(self.area_per_inst__um2, "area_per_inst__um2")
        self._require_non_neg(self.leakage_per_inst__uW, "leakage_per_inst__uW")


class VoltageDriverPolicy(AnalogPolicy):
    """Per-source toggles selecting which clamp nonidealities are active."""

    offset: bool
    """Apply the static systematic per-instance offset `offset_sigma__V`."""
    thermal: bool
    """Apply the per-solve thermal noise `thermal_sigma__V`."""


@dataclass(frozen=True)
class VoltageDriverSnap(TensorGroupMixin):
    """One sampled clamp snap."""

    v_ref__V: Tensor
    """NOMINAL reference clamp voltage — the ideal value, carrying no offset
    or thermal draw.
    Shape: `[..., *inst_shape]`."""
    v_perturb__V: Tensor
    """Driver-owned perturbation on top of the nominal reference — the static
    per-instance offset plus the per-call thermal draw, whichever the policy
    enables; exactly zero under an all-off policy.
    Shape: `[..., *inst_shape]`."""
    r_out__MOhm: Tensor
    """Series output resistance — a frozen constant slope broadcast to the
    call shape.
    Shape: `[..., *inst_shape]`."""


class VoltageDriver(AnalogBase[VoltageDriverConfig, VoltageDriverPolicy]):
    """Generic Thevenin voltage-source clamp driver.

    The clamp follows `v_clamp = v_ref + v_perturb - i_port * r_out`, where
    `v_perturb` is this driver's own offset / thermal perturbation on top of
    the nominal reference it is handed. A zero output resistance represents an
    ideal voltage source.

    Args:
        config: Concrete configuration dataclass.
        policy: Per-source nonideality enable flags.
        inst_shape: Per-instance fabrication shape.
        dtype: Tensor dtype for internal buffers.
        T__K: Operating temperature.
    """

    # === Circuit constant buffers ===

    _frozen_r_out__MOhm: Tensor  # Shape: []

    # === Nominal buffers ===

    _nominal_offset__V: Tensor  # Shape: []

    # === Fabricated state ===

    _offset__V: Tensor  # Shape: [*inst_shape]

    def __init__(
        self,
        *,
        config: VoltageDriverConfig,
        policy: VoltageDriverPolicy,
        inst_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
    ) -> None:
        super().__init__(config=config, policy=policy, inst_shape=inst_shape)

        self.register_buffer(
            "_frozen_r_out__MOhm",
            torch.tensor(config.r_out__MOhm, dtype=dtype),
            persistent=False,
        )
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
            "_nominal_offset__V",
            torch.zeros((), dtype=dtype),
            persistent=False,
        )

    def _sample_fabricate_mismatch(self) -> None:
        self._offset__V = apply_gaussian(
            self._nominal_offset__V.clone().expand(self.inst_shape),
            self.config.offset_sigma__V,
            enabled=self.policy.offset,
        )

    def snapshot(
        self,
        *,
        v_ref__V: Tensor,
        shape: tuple[int, ...],
    ) -> VoltageDriverSnap:
        """Sample the driver's static state and per-call noise at `shape`.

        The reference and the fabricated static buffer both expand onto
        `shape`, and the thermal draw takes a fresh sample per position of it.
        The returned snap's `v_ref__V` stays the NOMINAL reference — every
        perturbation accumulates instead in `v_perturb__V`, which an all-off
        policy leaves exactly zero, so the clamp reduces to the nominal
        reference bit-exactly.

        Sampling only: no energy is billed here, because the drive is billed
        at the converged port state a snapshot cannot see.

        Args:
            v_ref__V: Injected reference / zero-current clamp voltage — the
                Thevenin open-circuit voltage, broadcastable to `shape`.
            shape: Full per-call shape to expand the reference and the
                fabricated offset onto and to draw the thermal noise at.
                Shape: `[..., *inst_shape]`.

        Returns:
            Per-call snap of the fabricated state.
        """
        v_ref__V = v_ref__V.expand(shape)
        v_perturb__V = self._offset__V.expand(shape) if self.policy.offset else self._nominal_offset__V.expand(shape)
        v_perturb__V = apply_gaussian(v_perturb__V, self.config.thermal_sigma__V, enabled=self.policy.thermal)
        # The slope is one number for every position, and the expand is the
        # stride-0 view that says so without storing it.
        # Shape: [] -> [..., *inst_shape]
        return VoltageDriverSnap(
            v_ref__V=v_ref__V,
            v_perturb__V=v_perturb__V,
            r_out__MOhm=self._frozen_r_out__MOhm.expand(shape),
        )

    def drive(self, i_port__uA: Tensor, v_clamp__V: Tensor) -> Tensor:
        """Deliver the clamp at the converged port state and bill the drive.

        The port state is the settled pair `(i_port__uA, v_clamp__V)`: the
        Thevenin drop `i_port * r_out` is already inside the clamp node, so the
        delivered voltage is that node itself and no snap is needed.

        Call once per access, on the port state at the caller's full leading:
        the drive is billed per driven position, and only that layout states
        which positions those are.

        Args:
            i_port__uA: Converged port current [uA] — one instance per column,
                so the column axis is last.
                Shape: `[*caller_leading, ...]`.
            v_clamp__V: Converged clamp voltage [V] at the same layout.
                Shape: `[*caller_leading, ...]`.

        Returns:
            Delivered clamp voltage [V] — the terminal voltage this driver
            holds at `i_port__uA`.
            Shape: `[*caller_leading, ...]`.
        """
        # A flat per-port-op lump: the expanded constant holds no storage, so no
        # energy tensor is materialized, and the energy dtype comes from the constant
        # rather than from the port current.
        # Shape: [] -> [*i_port__uA.shape]
        e_op__fJ = torch.full((), self.config.energy_per_op__fJ, dtype=torch.float32, device=i_port__uA.device)
        self._record_dynamic_energy(e_op__fJ.expand(i_port__uA.shape))
        return v_clamp__V

    def solve_clamp(
        self,
        i_port__uA: Tensor,
        snap: VoltageDriverSnap,
        *,
        v_clamp_init__V: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        """Solve the Thevenin clamp at the present port current.

        Args:
            i_port__uA: Port current.
            snap: Sampled clamp state the solve reads.
            v_clamp_init__V: Optional warm-start hint. Accepted and ignored —
                the clamp is closed-form.

        Returns:
            Tuple `(v_clamp__V, dVclamp_dI__MOhm)`, where `v_clamp__V =
            snap.v_ref__V + snap.v_perturb__V - i_port__uA * snap.r_out__MOhm`
            and `dVclamp_dI__MOhm = snap.r_out__MOhm` broadcast to
            `i_port__uA`.
        """
        v_clamp__V = snap.v_ref__V + snap.v_perturb__V - i_port__uA * snap.r_out__MOhm
        dVclamp_dI__MOhm = snap.r_out__MOhm.expand_as(i_port__uA)
        return v_clamp__V, dVclamp_dI__MOhm
