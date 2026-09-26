"""Generic Thevenin voltage-source clamp-driver model.

See Also:
    docs/reference/primitive/analog/voltage_driver.md
    docs/system_design/ppa_accounting.md
"""

from __future__ import annotations

import torch
from torch import Tensor

from neurox.common.module import ConfigBase, DcopBase, PolicyBase, ProfileModule, SnapBase
from neurox.primitive.nonideality import apply_gaussian


class VoltageDriverConfig(ConfigBase):
    # === Output impedance ===

    r_out__MOhm: float
    """Series output resistance — its NEGATIVE is the constant port-voltage slope
    ∂V_port/∂I; 0 recovers the ideal voltage-source limit."""

    # === Nonidealities ===

    offset_sigma__V: float
    """σ of the static systematic per-instance offset on the reference."""
    thermal_sigma__V: float
    """σ of the per-solve Gaussian thermal noise on the reference."""

    # === Static PPA ===

    area_per_inst__um2: float
    leakage_per_inst__uW: float

    # === Dynamic energy ===

    energy_per_op__fJ: float
    """Per-column-op interface energy, e.g. one full C·V² interface-node
    precharge cycle; a physical zero is a legitimate value."""

    def validate(self) -> None:
        super().validate()

        # --- Output impedance ---

        self._require_non_neg(self.r_out__MOhm, "r_out__MOhm")

        # --- Nonidealities ---

        self._require_non_neg(self.offset_sigma__V, "offset_sigma__V")
        self._require_non_neg(self.thermal_sigma__V, "thermal_sigma__V")

        # --- Static PPA ---

        self._require_non_neg(self.area_per_inst__um2, "area_per_inst__um2")
        self._require_non_neg(self.leakage_per_inst__uW, "leakage_per_inst__uW")

        # --- Dynamic energy ---

        self._require_non_neg(self.energy_per_op__fJ, "energy_per_op__fJ")


class VoltageDriverPolicy(PolicyBase):
    offset: bool
    """Apply the static systematic per-instance offset `offset_sigma__V`."""
    thermal: bool
    """Apply the per-solve thermal noise `thermal_sigma__V`."""


class VoltageDriverDcop(DcopBase):
    v_port__V: Tensor
    """Port voltage held at the evaluated port current."""
    dvport_di__MOhm: Tensor
    """Port-voltage slope against the port current — the NEGATED constant series
    output resistance, broadcast to the port current; ≤ 0 for r_out ≥ 0, and
    exactly 0 in the ideal-source limit."""


class VoltageDriverSnap(SnapBase):
    v_open__V: Tensor
    """Zero-load output voltage, including sampled offset and thermal noise."""
    r_out__MOhm: Tensor
    """Series output resistance — a frozen constant broadcast to the call
    shape."""


_Config = VoltageDriverConfig
_Policy = VoltageDriverPolicy
_Dcop = VoltageDriverDcop
_Snap = VoltageDriverSnap


class VoltageDriver(ProfileModule):
    """Generic Thevenin voltage-source clamp driver.

    The port voltage follows `v_port = v_ref + v_perturb - i_port * r_out`,
    where `v_perturb` is this driver's own offset / thermal perturbation on top
    of the nominal reference it is handed. A zero output resistance represents
    an ideal voltage source.

    Place and fabricate before taking a snapshot when offset is enabled. Supply
    the nominal reference and full access layout to `snapshot`, then reuse that
    snapshot while solving port currents. `solve_dc` neither samples variation
    nor bills energy. Call `drive` once for each modeled enabled boundary event;
    its current argument determines the event layout, not the energy magnitude.
    There is no headroom clamp or current-dependent output resistance.

    Args:
        config: Hardware configuration.
        policy: Run policy matching `config`.
        inst_shape: Positive physical instance extents; singletons allow
            broadcasting.
        dtype: Electrical tensor dtype.
    """

    config: _Config
    policy: _Policy

    # === Functional buffers ===

    _r_out__MOhm: Tensor  # Shape: []

    # === Nominal buffers ===

    _nominal_offset__V: Tensor  # Shape: []

    # === Fabricated state ===

    _offset__V: Tensor  # Shape: [*inst_shape]

    def __init__(
        self,
        *,
        config: _Config,
        policy: _Policy,
        inst_shape: tuple[int, ...],
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__(config=config, policy=policy, inst_shape=inst_shape)

        self._register_nonpersistent_buffer(
            "_r_out__MOhm",
            torch.tensor(config.r_out__MOhm, dtype=dtype),
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
            sigma=self.config.offset_sigma__V,
            enabled=self.policy.offset,
        )

    @torch.no_grad()
    def snapshot(
        self,
        v_ref__V: Tensor,
        *,
        shape: tuple[int, ...],
    ) -> _Snap:
        """Sample the driver's static state and per-call noise at `shape`.

        The zero-load output combines the nominal reference with enabled static
        offset and thermal noise. Sampling bills no energy; `drive` bills at
        the converged port state.

        Args:
            v_ref__V: Nominal reference voltage before driver offset and noise,
                broadcastable to `shape`.
            shape: Full per-call layout for expanded reference and offset,
                with a fresh thermal draw per position. Include every distinct
                access to avoid sharing its noise sample.

        Returns:
            Per-call snap of the fabricated state.
        """
        v_ref__V = v_ref__V.expand(shape)
        v_perturb__V = self._offset__V.expand(shape) if self.policy.offset else self._nominal_offset__V.expand(shape)
        v_perturb__V = apply_gaussian(v_perturb__V, sigma=self.config.thermal_sigma__V, enabled=self.policy.thermal)
        return _Snap(
            v_open__V=v_ref__V + v_perturb__V,
            r_out__MOhm=self._r_out__MOhm.expand(shape),
        )

    @torch.no_grad()
    def drive(self, i_port__uA: Tensor, *, enable: Tensor | None = None) -> None:
        """Record enabled boundary accesses at the port-current layout.

        `enable` selects startup events, independently of the current
        amplitude. `None` enables all events. Conduction energy belongs to
        the circuit supplying the port.
        """
        if self._is_profiler_active():
            # Mask presence specializes during tracing; only masked costs depend on its device.
            if enable is None:
                energy__fJ = torch.full((), self.config.energy_per_op__fJ, dtype=torch.float32)
            else:
                energy__fJ = enable.to(dtype=torch.float32) * self.config.energy_per_op__fJ
            self._record_dynamic_energy(energy__fJ.expand(i_port__uA.shape))

    @torch.no_grad()
    def solve_dc(
        self,
        i_port__uA: Tensor,
        *,
        snap: _Snap,
        v_port_init__V: Tensor | None = None,
    ) -> _Dcop:
        """Solve the Thevenin driver's port voltage at the present port current.

        Args:
            i_port__uA: Current sourced from the driver into its load.
            snap: Held driver open-circuit voltage and output resistance.
            v_port_init__V: Optional initial port voltage for a warm start.
                Accepted and ignored — this driver is closed-form.

        Returns:
            Port state, where `v_port__V = snap.v_open__V - i_port__uA *
            snap.r_out__MOhm` and `dvport_di__MOhm = -snap.r_out__MOhm`
            broadcast to `i_port__uA` — the derivative of that map, which the
            series drop makes non-positive.
        """
        return _Dcop(
            v_port__V=snap.v_open__V - i_port__uA * snap.r_out__MOhm,
            dvport_di__MOhm=-snap.r_out__MOhm.expand_as(i_port__uA),
        )
