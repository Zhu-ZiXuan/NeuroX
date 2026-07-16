"""Generic Thevenin voltage-source clamp-driver model.

See also:
    docs/reference/primitive/analog/voltage_driver.md
"""

from dataclasses import dataclass

import torch
from torch import Tensor

from neurox.primitive.analog.base import AnalogBase, AnalogConfig, AnalogPolicy
from neurox.primitive.nonideality import apply_gaussian


@dataclass(frozen=True, kw_only=True)
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
        area_per_inst__um2: Silicon area per fabricated instance.
        leakage_per_inst__uW: Static leakage per instance; carries
            all static power, including any internal amplifier / bias.
    """

    # --- Series output resistance (constant clamp slope) ---
    r_out__MOhm: float

    # --- Systematic offset ---
    offset_sigma__V: float

    # --- Thermal noise ---
    thermal_sigma__V: float

    # --- Static PPA ---
    area_per_inst__um2: float
    leakage_per_inst__uW: float

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        self.validate_source()
        self.validate_noise()
        self.validate_ppa()

    def validate_source(self) -> None:
        self._require_non_neg(self.r_out__MOhm, "r_out__MOhm")

    def validate_noise(self) -> None:
        self._require_non_neg(self.offset_sigma__V, "offset_sigma__V")
        self._require_non_neg(self.thermal_sigma__V, "thermal_sigma__V")

    def validate_ppa(self) -> None:
        self._require_non_neg(self.area_per_inst__um2, "area_per_inst__um2")
        self._require_non_neg(self.leakage_per_inst__uW, "leakage_per_inst__uW")


@dataclass(frozen=True)
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

    A design-agnostic boundary clamp: a reference voltage source
    ``v_ref`` in series with a constant output resistance ``r_out``,
    holding a port near ``v_ref`` and drooping linearly with the current
    it sources or sinks. The clamp transfer is the closed-form Thevenin
    map ``v_clamp = v_ref - i_port * r_out``; ``r_out = 0`` recovers the
    ideal voltage source (flat clamp), and a finite ``r_out`` is the
    physical series impedance the consuming solver sees as the clamp
    slope ``dVclamp/dI``.

    This block carries only its own static power, folded into
    ``leakage_per_inst__uW`` (including any internal amplifier or bias
    network).

    It satisfies the structural ``ClampDriver`` role (``snapshot``,
    ``solve_clamp``) without inheriting the protocol; the reference
    voltage is injected per call into :meth:`snapshot`.

    Args:
        config: Concrete configuration dataclass.
        policy: Per-source nonideality enable flags.
        inst_shape: Per-instance fabrication shape.
        dtype: Tensor dtype for internal buffers.
        T__K: Operating temperature.
    """

    frozen_r_out__MOhm: Tensor
    offset__V: Tensor

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

        self.dtype = dtype
        self.T__K = T__K

        self.register_buffer(
            "frozen_r_out__MOhm",
            torch.tensor(config.r_out__MOhm, dtype=dtype),
            persistent=False,
        )
        # Held static systematic per-instance reference offset; zero when the
        # offset is off.
        self.register_buffer(
            "offset__V",
            torch.zeros(inst_shape, dtype=dtype),
            persistent=False,
        )

    def _sample_fabricate_mismatch(self) -> None:
        """Sample the static systematic per-instance offset over ``inst_shape``.

        Additive zero-mean Gaussian with σ ``offset_sigma__V``; ``offset`` off
        holds a zero offset.
        """
        if self.policy.offset:
            self.offset__V = torch.randn_like(self.offset__V) * self.config.offset_sigma__V
        else:
            self.offset__V = torch.zeros_like(self.offset__V)

    # --- Snapshot + clamp solve ---

    def snapshot(
        self,
        *,
        v_ref__V: Tensor,
        shape: tuple[int, ...],
        multi_coords: tuple[Tensor, ...] | None,
    ) -> VoltageDriverSnap:
        """Sample one per-call runtime snap over ``shape``.

        Adds the static systematic per-instance offset to the injected
        reference, then the per-solve thermal fluctuation. The offset shares
        the per-instance ``inst_shape`` and is broadcast to ``shape`` and
        chunk-selected by ``multi_coords`` in lockstep with the reference.

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
            offset_view = self.offset__V.expand(shape) if shape else self.offset__V
            v = v + (offset_view if multi_coords is None else offset_view[multi_coords])
        v = apply_gaussian(v, self.config.thermal_sigma__V, enabled=self.policy.thermal)
        return VoltageDriverSnap(v_ref__V=v, r_out__MOhm=self.frozen_r_out__MOhm)

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
