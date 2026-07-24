"""Generic Thevenin voltage-source clamp-driver model.

See also:
    docs/reference/primitive/analog/voltage_driver.md
"""

from dataclasses import dataclass

import torch
from torch import Tensor

from neurox.primitive.nonideality import apply_gaussian

from .base import AnalogBase, AnalogConfig, AnalogPolicy


class VoltageDriverConfig(AnalogConfig):
    """Immutable configuration for :class:`VoltageDriver`.

    Attributes:
        r_out__MOhm: Series output resistance — the constant clamp
            slope ``dVclamp/dI``. ``r_out = 0`` recovers the ideal
            voltage-source limit.
        offset_sigma__V: σ of the static systematic per-instance offset
            on the reference.
        thermal_sigma__V: σ of the per-solve Gaussian thermal noise
            on the reference.
        energy_per_op__fJ: Per-column-op interface energy, e.g. one full
            ``C·V²`` interface-node precharge cycle; a physical zero is
            a legitimate value.
        area_per_inst__um2: Silicon area per fabricated instance.
        leakage_per_inst__uW: Static leakage per instance; carries
            all static power, including any internal amplifier / bias.
    """

    r_out__MOhm: float
    offset_sigma__V: float
    thermal_sigma__V: float
    energy_per_op__fJ: float
    area_per_inst__um2: float
    leakage_per_inst__uW: float

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
    """Per-source toggles selecting which clamp nonidealities are active.

    Attributes:
        offset: Apply the static systematic per-instance offset
            (``offset_sigma__V``).
        thermal: Apply the per-solve thermal noise (``thermal_sigma__V``).
    """

    offset: bool
    thermal: bool


@dataclass(frozen=True)
class VoltageDriverSnap:
    """One sampled clamp snap.

    Attributes:
        v_ref__V: Reference clamp voltage, post offset + thermal,
            broadcast to the per-call shape.
        r_out__MOhm: Series output resistance — a 0-d frozen
            constant slope.
    """

    v_ref__V: Tensor
    r_out__MOhm: Tensor


class VoltageDriver(AnalogBase[VoltageDriverConfig, VoltageDriverPolicy]):
    """Generic Thevenin voltage-source clamp driver.

    The clamp follows ``v_clamp = v_ref - i_port * r_out``. A zero output
    resistance represents an ideal voltage source.

    Args:
        config: Concrete configuration dataclass.
        policy: Per-source nonideality enable flags.
        inst_shape: Per-instance fabrication shape.
        dtype: Tensor dtype for internal buffers.
        T__K: Operating temperature.
    """

    # --- Immutable model buffers ---

    _frozen_r_out__MOhm: Tensor

    # --- Fabrication source buffers ---

    _nominal_offset__V: Tensor

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

        self._area_per_inst__um2 = config.area_per_inst__um2
        self._leakage_per_inst__uW = config.leakage_per_inst__uW

        self.register_buffer(
            "_frozen_r_out__MOhm",
            torch.tensor(config.r_out__MOhm, dtype=dtype),
            persistent=False,
        )
        self._register_fabrication_buffers(dtype=dtype)

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
        multi_coords: tuple[Tensor, ...] | None,
    ) -> VoltageDriverSnap:
        """Sample the driver state and runtime noise over ``shape``.

        Args:
            v_ref__V: Injected reference / zero-current clamp voltage
                — the Thevenin open-circuit voltage. A scalar or
                instance-shaped tensor that broadcasts onto ``shape``.
            shape: Per-call broadcast shape; the snap fills tensor
                fields at this shape.
            multi_coords: Advanced-index tuple selecting a chunk's
                positions from the broadcast view; ``None`` returns the
                full view.

        Returns:
            Per-call snap of the fabricated state.
        """
        v_view = v_ref__V.expand(shape) if shape else v_ref__V
        v = (v_view if multi_coords is None else v_view[multi_coords]).clone()
        if self.policy.offset:
            offset_view = self._offset__V.expand(shape) if shape else self._offset__V
            v = v + (offset_view if multi_coords is None else offset_view[multi_coords])
        v = apply_gaussian(v, self.config.thermal_sigma__V, enabled=self.policy.thermal)
        if self._is_dynamic_energy_profile_active():
            self._record_dynamic_energy(torch.full_like(v, self.config.energy_per_op__fJ, dtype=torch.float32))
        return VoltageDriverSnap(v_ref__V=v, r_out__MOhm=self._frozen_r_out__MOhm)

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
            snap: Snap returned by :meth:`snapshot`.
            v_clamp_init__V: Optional warm-start hint. Accepted and
                ignored — the clamp is closed-form.

        Returns:
            Tuple ``(v_clamp__V, dVclamp_dI__MOhm)``, where
            ``v_clamp__V = snap.v_ref__V - i_port__uA * snap.r_out__MOhm``
            and ``dVclamp_dI__MOhm = snap.r_out__MOhm`` broadcast to
            ``i_port__uA``.
        """
        v_clamp__V = snap.v_ref__V - i_port__uA * snap.r_out__MOhm
        dVclamp_dI__MOhm = snap.r_out__MOhm.expand_as(i_port__uA)
        return v_clamp__V, dVclamp_dI__MOhm
