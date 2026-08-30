"""Generic Thevenin voltage-source clamp-driver model.

See Also:
    docs/reference/primitive/analog/voltage_driver.md
    docs/system_design/ppa_accounting.md
"""

import torch
from torch import Tensor

from neurox.common import ConfigBase, DcopBase, ModuleBase, PolicyBase, SnapBase
from neurox.primitive.nonideality import apply_gaussian


class VoltageDriverConfig(ConfigBase):
    r_out__MOhm: float
    """Series output resistance — its NEGATIVE is the constant clamp slope
    ∂V_clamp/∂I; 0 recovers the ideal voltage-source limit."""
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


class VoltageDriverPolicy(PolicyBase):
    offset: bool
    """Apply the static systematic per-instance offset `offset_sigma__V`."""
    thermal: bool
    """Apply the per-solve thermal noise `thermal_sigma__V`."""


class VoltageDriverDcop(DcopBase):
    v_clamp__V: Tensor
    """Clamp voltage held at the evaluated port current."""
    dvclamp_di__MOhm: Tensor
    """Clamp slope against the port current — the NEGATED constant series
    output resistance, broadcast to the port current; ≤ 0 for r_out ≥ 0, and
    exactly 0 in the ideal-source limit."""


class VoltageDriverSnap(SnapBase):
    v_ref__V: Tensor
    """NOMINAL reference clamp voltage — the ideal value, carrying no offset
    or thermal draw."""
    v_perturb__V: Tensor
    """Driver-owned perturbation on top of the nominal reference — the static
    per-instance offset plus the per-call thermal draw, whichever the policy
    enables; exactly zero under an all-off policy."""
    r_out__MOhm: Tensor
    """Series output resistance — a frozen constant broadcast to the call
    shape."""


class VoltageDriver(ModuleBase[VoltageDriverConfig, VoltageDriverPolicy]):
    """Generic Thevenin voltage-source clamp driver.

    The clamp follows `v_clamp = v_ref + v_perturb - i_port * r_out`, where
    `v_perturb` is this driver's own offset / thermal perturbation on top of
    the nominal reference it is handed. A zero output resistance represents an
    ideal voltage source.
    """

    # === Functional buffers ===

    _r_out__MOhm: Tensor  # Shape: []
    _energy_per_op__fJ: Tensor  # Shape: []

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

        self._register_nonpersistent_buffer(
            "_r_out__MOhm",
            torch.tensor(config.r_out__MOhm, dtype=dtype),
        )
        self._register_nonpersistent_buffer(
            "_energy_per_op__fJ",
            torch.tensor(config.energy_per_op__fJ, dtype=dtype),
        )
        self._register_fabrication_buffers(dtype=dtype)

    @property
    def _area_per_inst__um2(self) -> float:
        return self.config.area_per_inst__um2

    @property
    def _leakage_per_inst__uW(self) -> float:
        return self.config.leakage_per_inst__uW

    def _register_fabrication_buffers(self, *, dtype: torch.dtype) -> None:
        self._register_nonpersistent_buffer(
            "_nominal_offset__V",
            torch.zeros((), dtype=dtype),
        )

    def _sample_fabrication_variation(self) -> None:
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

        `shape` also fixes the noise extent: the thermal draw covers exactly
        the positions it spans, so a shape short of the real access count
        shares one sample across accesses that are physically distinct.

        Args:
            v_ref__V: Injected reference / zero-current clamp voltage — the
                Thevenin open-circuit voltage, broadcastable to `shape`.
            shape: Full per-call shape to expand the reference and the
                fabricated offset onto and to draw the thermal noise at.

        Returns:
            Per-call snap of the fabricated state.
        """
        v_ref__V = v_ref__V.expand(shape)
        v_perturb__V = self._offset__V.expand(shape) if self.policy.offset else self._nominal_offset__V.expand(shape)
        v_perturb__V = apply_gaussian(v_perturb__V, self.config.thermal_sigma__V, enabled=self.policy.thermal)
        # The slope is one number for every position, and the expand is the
        # stride-0 view that says so without storing it.
        return VoltageDriverSnap(
            v_ref__V=v_ref__V,
            v_perturb__V=v_perturb__V,
            r_out__MOhm=self._r_out__MOhm.expand(shape),
        )

    def drive(self, *, i_port__uA: Tensor) -> None:
        """Record one access at the converged port-current layout."""
        if self._is_dynamic_energy_profile_active():
            self._record_dynamic_energy(self._energy_per_op__fJ.expand(i_port__uA.shape))

    def solve_dc(
        self,
        i_port__uA: Tensor,
        snap: VoltageDriverSnap,
        *,
        v_clamp_init__V: Tensor | None,
    ) -> VoltageDriverDcop:
        """Solve the Thevenin clamp at the present port current.

        Args:
            i_port__uA: Port current.
            snap: Sampled clamp state the solve reads.
            v_clamp_init__V: Optional warm-start hint. Accepted and ignored —
                the clamp is closed-form.

        Returns:
            Clamp state, where `v_clamp__V = snap.v_ref__V +
            snap.v_perturb__V - i_port__uA * snap.r_out__MOhm` and
            `dvclamp_di__MOhm = -snap.r_out__MOhm` broadcast to `i_port__uA`
            — the derivative of that map, which the series drop makes
            non-positive.
        """
        return VoltageDriverDcop(
            v_clamp__V=snap.v_ref__V + snap.v_perturb__V - i_port__uA * snap.r_out__MOhm,
            dvclamp_di__MOhm=-snap.r_out__MOhm.expand_as(i_port__uA),
        )
