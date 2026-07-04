"""Generic linear transimpedance amplifier used as a BL clamp driver.

See also:
    docs/reference/analog/tia/general.md
"""

from dataclasses import dataclass

import torch
from torch import Tensor

from .base import TIA, TIAConfig, TIAPolicy, TIASnap


@dataclass(frozen=True, kw_only=True)
class GeneralTIAConfig(TIAConfig):
    """Configuration for :class:`GeneralTIA`.

    Attributes:
        input_impedance__MOhm: Thevenin small-signal input impedance
            ``Z_in`` at the clamp node.
        load_resistance__MOhm: Conversion-stage equivalent load
            resistance ``R_load`` — the transimpedance gain.
    """

    input_impedance__MOhm: float
    load_resistance__MOhm: float

    def validate(self) -> None:
        super().validate()
        self._require_nonneg(self.input_impedance__MOhm, "input_impedance__MOhm")
        self._require_pos(self.load_resistance__MOhm, "load_resistance__MOhm")


@dataclass(frozen=True)
class GeneralTIAPolicy(TIAPolicy):
    """Nonideality policy for GeneralTIA — ideal, no toggles."""


@dataclass(frozen=True)
class GeneralTIASnap(TIASnap):
    """Per-call GeneralTIA snap.

    Attributes:
        v_ref__V: Injected reference clamp voltage, broadcast to the
            per-call shape.
    """

    v_ref__V: Tensor


@dataclass(frozen=True)
class GeneralTIADCOP:
    """DC operating-point result of :meth:`GeneralTIA.solve_dc`.

    Attributes:
        v_clamp__V: Clamp-node voltage at the operating point.
        v_out__V: Transimpedance output voltage.
        dVclamp_dI__MOhm: ``∂v_clamp / ∂i_port``.
        dVout_dI__MOhm: ``∂v_out / ∂i_port``.
    """

    v_clamp__V: Tensor
    v_out__V: Tensor
    dVclamp_dI__MOhm: Tensor
    dVout_dI__MOhm: Tensor


@TIA.register_key(GeneralTIAConfig)
class GeneralTIA(TIA[GeneralTIASnap]):
    """Linear (no-Newton) TIA clamp driver.

    A Thevenin-input + resistive-transimpedance model: the clamp node
    sits at ``v_ref`` plus an ``input_impedance`` drop, and the output
    is ``v_ref`` plus the ``load_resistance`` conversion of the port
    current. Exact and loop-free — the linear sibling of
    :class:`OpAmpTIA`.
    """

    config: GeneralTIAConfig

    def __init__(
        self,
        *,
        config: GeneralTIAConfig,
        policy: GeneralTIAPolicy,
        name: str,
        inst_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
    ) -> None:
        super().__init__(
            config=config,
            policy=policy,
            name=name,
            inst_shape=inst_shape,
            dtype=dtype,
            T__K=T__K,
        )

        self.policy = policy
        self.dtype = dtype
        self.T__K = T__K

    def _sample_fabricate_mismatch(self) -> None:
        pass  # ideal linear TIA: no nonideality to fabricate

    # --- snapshot ---

    def snapshot(
        self,
        *,
        v_ref__V: Tensor,
        shape: tuple[int, ...],
        multi_coords: tuple[Tensor, ...] | None,
    ) -> GeneralTIASnap:
        """Store the injected reference into the per-call snap.

        Args:
            v_ref__V: Injected reference clamp voltage. A scalar or
                instance-shaped tensor that broadcasts onto ``shape``.
            shape: Per-call broadcast shape; the snap fills the reference
                field at this shape.
            multi_coords: Advanced-index tuple selecting a chunk's
                positions from the broadcast view; ``None`` returns the
                full view.

        Returns:
            A :class:`GeneralTIASnap` carrying the injected reference.
        """
        v_view = v_ref__V.expand(shape) if shape else v_ref__V
        v = v_view if multi_coords is None else v_view[multi_coords]
        return GeneralTIASnap(v_ref__V=v)

    # --- forward path ---

    def solve_dc(
        self,
        i_port__uA: Tensor,
        snap: GeneralTIASnap,
        *,
        v_clamp_init__V: Tensor | None,
    ) -> GeneralTIADCOP:
        """Solve the linear GeneralTIA at one port current.

        Args:
            i_port__uA: Port-output current; positive = sourcing.
            snap: Per-call snap from :meth:`snapshot`.
            v_clamp_init__V: Ignored — the model is exact, no warm start.

        Returns:
            ``v_clamp``, ``v_out`` and the constant small-signal
            sensitivities ``∂v_clamp/∂i_port`` (``Z_in``) and
            ``∂v_out/∂i_port`` (``R_load``) ([V/uA] = [MOhm]).
        """
        del v_clamp_init__V  # linear/exact: no warm start
        v_ref = snap.v_ref__V
        z_in__MOhm = self.config.input_impedance__MOhm
        r_load__MOhm = self.config.load_resistance__MOhm

        v_clamp__V = v_ref + i_port__uA * z_in__MOhm
        v_out__V = v_ref + i_port__uA * r_load__MOhm
        dVclamp_dI__MOhm = torch.full_like(i_port__uA, z_in__MOhm)
        dVout_dI__MOhm = torch.full_like(i_port__uA, r_load__MOhm)

        return GeneralTIADCOP(
            v_clamp__V=v_clamp__V,
            v_out__V=v_out__V,
            dVclamp_dI__MOhm=dVclamp_dI__MOhm,
            dVout_dI__MOhm=dVout_dI__MOhm,
        )

    def solve_clamp(
        self,
        i_port__uA: Tensor,
        snap: GeneralTIASnap,
        *,
        v_clamp_init__V: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        """Boundary-clamp wrapper around :meth:`solve_dc`.

        Args:
            i_port__uA: Port-output current; see :meth:`solve_dc`.
            snap: Per-call snap from :meth:`snapshot`.
            v_clamp_init__V: Ignored — see :meth:`solve_dc`.

        Returns:
            ``(v_clamp__V, dVclamp_dI__MOhm)``.
        """
        dc = self.solve_dc(i_port__uA, snap, v_clamp_init__V=v_clamp_init__V)
        return dc.v_clamp__V, dc.dVclamp_dI__MOhm

    # --- energy model ---

    def dynamic_energy__fJ(
        self,
        i_port__uA: Tensor,
        dcop: GeneralTIADCOP,
        *,
        read_pulse__ns: float,
    ) -> Tensor:
        """Transimpedance-resistor dissipation over one read window.

        Pure compute — the caller logs the returned term.

        Args:
            i_port__uA: Port-output current.
            dcop: Operating point from :meth:`solve_dc` (unused; the
                dissipation depends only on ``i_port`` and ``R_load``).
            read_pulse__ns: Read-window width.

        Returns:
            ``i_port**2 · R_load · read_pulse``
            (uA**2 · MOhm · ns = fJ).
        """
        del dcop  # dissipation is i_port**2 * R_load; DCOP not needed
        return i_port__uA**2 * self.config.load_resistance__MOhm * read_pulse__ns
