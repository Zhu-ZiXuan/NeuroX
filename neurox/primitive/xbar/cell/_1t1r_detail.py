"""Detailed 1T1R cell — nonlinear device models condensed by a per-cell Newton.

See also:
    docs/reference/primitive/xbar/cell/_1t1r/cell_detail.md
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import ClassVar, Self

import torch
from torch import Tensor

from neurox.common.prober import Prober
from neurox.primitive.device import (
    MosfetConfig,
    MosfetPolicy,
    MosfetSnap,
    Nmos,
    Rram,
    RramConfig,
    RramPolicy,
    RramSnap,
)

from ._1t1r import (
    XbarCell1t1r,
    XbarCell1t1rConfig,
    XbarCell1t1rDcop,
    XbarCell1t1rPolicy,
    XbarCell1t1rSnap,
)

# ---------------------------------------------------------------------------
# Observation side-channel payload
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class XbarCell1t1rDetailObservation:
    """Per-cell access-node KCL residual of a detailed 1T1R branch solve.

    Built at the converged ``V_X`` of a :meth:`XbarCell1t1rDetail.solve_dc`
    call and submitted to :class:`XbarCell1t1rDetailProber` only when a prober
    is active.

    Attributes:
        cell__uA: ``|I_NMOS - I_RRAM|`` per cell at the condensed ``V_X``.
            Shape: ``[..., col, row]``.
    """

    cell__uA: Tensor

    def detach(self) -> Self:
        """Return an equivalent payload with the tensor field detached."""
        return replace(self, cell__uA=self.cell__uA.detach())


class XbarCell1t1rDetailProber(Prober[XbarCell1t1rDetailObservation]):
    """Capture point for the detailed 1T1R cell's access-node observation link.

    :class:`XbarCell1t1rDetail` emits a :class:`XbarCell1t1rDetailObservation`
    — the access-node KCL residual ``|I_NMOS - I_RRAM|`` at the converged
    ``V_X`` — once per :meth:`XbarCell1t1rDetail.solve_dc` call when a prober
    is active.
    """

    _active_stack: ClassVar[list[Prober[XbarCell1t1rDetailObservation]]] = []

    @classmethod
    def _stack(cls) -> list[Prober[XbarCell1t1rDetailObservation]]:
        """Return this observation link's active-prober stack."""
        return cls._active_stack


# ---------------------------------------------------------------------------
# Config / policy / result containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class XbarCell1t1rDetailConfig(XbarCell1t1rConfig):
    """Physical knobs for the detailed (nonlinear-device) 1T1R cell.

    Attributes:
        rram_config: RRAM storage-device configuration.
        nmos_config: Access-NMOS configuration.
        state_to_g_map__uS: State-index to target-conductance lookup
            table. Strictly increasing; endpoints must lie inside
            ``[rram_config.g_min__uS, rram_g_max__uS]``.
        access_nmos_W__um: Access-NMOS width.
        access_nmos_L__um: Access-NMOS length.
        rram_g_max__uS: Maximum programmable RRAM conductance.
        n_newton: Number of unrolled per-cell Newton steps on ``V_X``
            after the Pade current-divider seed.
    """

    rram_config: RramConfig
    nmos_config: MosfetConfig

    state_to_g_map__uS: tuple[float, ...]

    access_nmos_W__um: float
    access_nmos_L__um: float

    rram_g_max__uS: float

    n_newton: int

    def validate(self) -> None:
        super().validate()
        self.validate_access_nmos()
        self.validate_rram_window()
        self.validate_state_map()
        self.validate_solver()

    def validate_access_nmos(self) -> None:
        self._require_pos(self.access_nmos_W__um, "access_nmos_W__um")
        self._require_pos(self.access_nmos_L__um, "access_nmos_L__um")

    def validate_rram_window(self) -> None:
        if not (self.rram_g_max__uS > self.rram_config.g_min__uS):
            raise ValueError(
                f"require: rram_g_max__uS ({self.rram_g_max__uS}) > rram_config.g_min__uS ({self.rram_config.g_min__uS})"
            )

    def validate_state_map(self) -> None:
        self._require_min_length(self.state_to_g_map__uS, 2, "state_to_g_map__uS")
        self._require_increasing(self.state_to_g_map__uS, "state_to_g_map__uS")
        if self.state_to_g_map__uS[0] < self.rram_config.g_min__uS:
            raise ValueError(
                f"require: state_to_g_map__uS[0] ({self.state_to_g_map__uS[0]}) >= "
                f"rram_config.g_min__uS ({self.rram_config.g_min__uS})"
            )
        if self.state_to_g_map__uS[-1] > self.rram_g_max__uS:
            raise ValueError(
                f"require: state_to_g_map__uS[-1] ({self.state_to_g_map__uS[-1]}) <= "
                f"rram_g_max__uS ({self.rram_g_max__uS})"
            )

    def validate_solver(self) -> None:
        self._require_pos(self.n_newton, "n_newton")


@dataclass(frozen=True)
class XbarCell1t1rDetailPolicy(XbarCell1t1rPolicy):
    """Composite nonideality policy for the detailed 1T1R cell.

    Attributes:
        rram_policy: RRAM storage-device nonideality policy.
        nmos_policy: Access-NMOS nonideality policy.
    """

    rram_policy: RramPolicy
    nmos_policy: MosfetPolicy


@dataclass(frozen=True, kw_only=True)
class XbarCell1t1rDetailSnap(XbarCell1t1rSnap):
    """Per-call snap of a detailed 1T1R cell's fabricated state.

    Attributes:
        rram: RRAM read-conductance snap.
        nmos: Access-NMOS parameter snap.
    """

    rram: RramSnap
    nmos: MosfetSnap


# ---------------------------------------------------------------------------
# Cell
# ---------------------------------------------------------------------------


@XbarCell1t1r.register_key(XbarCell1t1rDetailConfig)
class XbarCell1t1rDetail(XbarCell1t1r[XbarCell1t1rDetailConfig, XbarCell1t1rDetailPolicy, XbarCell1t1rDetailSnap]):
    """Series access-NMOS + RRAM 1T1R cell with a condensed BL-to-SL branch.

    Emits the converged-``V_X`` :class:`XbarCell1t1rDetailObservation` to
    :class:`XbarCell1t1rDetailProber` once per :meth:`solve_dc` call, but only
    when a prober is active.
    """

    state_to_g_map__uS: Tensor
    rram: Rram
    nmos: Nmos

    def __init__(
        self,
        *,
        config: XbarCell1t1rDetailConfig,
        policy: XbarCell1t1rDetailPolicy,
        inst_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
    ) -> None:
        super().__init__(config=config, policy=policy, inst_shape=inst_shape, dtype=dtype, T__K=T__K)

        self.register_buffer(
            "state_to_g_map__uS",
            torch.tensor(config.state_to_g_map__uS, dtype=dtype),
            persistent=False,
        )
        self.w_states = len(config.state_to_g_map__uS)

        self.rram = Rram(
            config=config.rram_config,
            policy=policy.rram_policy,
            inst_shape=inst_shape,
            dtype=dtype,
            T__K=T__K,
            g_max__uS=config.rram_g_max__uS,
        )
        self.nmos = Nmos(
            config=config.nmos_config,
            policy=policy.nmos_policy,
            inst_shape=inst_shape,
            dtype=dtype,
            T__K=T__K,
            W__um=config.access_nmos_W__um,
            L__um=config.access_nmos_L__um,
        )

        self.n_newton = config.n_newton

    # -----------------------------------------------------------------
    # Snapshot / programming
    # -----------------------------------------------------------------

    def snapshot(
        self,
        *,
        control: Tensor,
        shape: tuple[int, ...],
        multi_coords: tuple[Tensor, ...] | None,
        t_elapsed: float,
    ) -> XbarCell1t1rDetailSnap:
        """Bundle RRAM / NMOS device snaps with the WL control drive.

        Args:
            control: Word-line drive voltage [V] at the NMOS gate;
                broadcasts to ``[..., col, row]``.
            shape: Per-call broadcast shape ``(*leading, col, row)`` the
                RRAM / NMOS snaps fill their tensor fields at.
            multi_coords: Advanced-index tuple selecting a chunk's
                positions from the broadcast view; forwarded to the RRAM /
                NMOS snaps. ``None`` returns the full view.
            t_elapsed: Time elapsed since programming [s]; reserved for
                time-dependent device read state.

        Returns:
            Per-call detailed 1T1R cell snap.
        """
        del t_elapsed  # no time-dependent read state in this cell
        rram_snap = self.rram.snapshot(shape=shape, multi_coords=multi_coords)
        nmos_snap = self.nmos.snapshot(shape=shape, multi_coords=multi_coords)
        return XbarCell1t1rDetailSnap(rram=rram_snap, nmos=nmos_snap, v_wl__V=control)

    def program(self, w_state_idx: Tensor) -> None:
        """Program the RRAM cells from one state-index tensor.

        Args:
            w_state_idx: State-index tensor in ``[0, w_states - 1]`` at
                ``self._inst_shape``.
        """
        target_g__uS = self.state_to_g_map__uS[w_state_idx.long()]
        self.rram.program(target_g__uS, t_elapsed=0.0)

    # -----------------------------------------------------------------
    # Branch solve
    # -----------------------------------------------------------------

    def _solve_vx(
        self,
        v_bl: Tensor,
        v_sl: Tensor,
        snap: XbarCell1t1rDetailSnap,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Condense the access node ``V_X`` and read off the branch quantities.

        Pade current-divider seed for ``V_X`` followed by ``n_newton``
        unrolled Newton steps on ``F_X = I_NMOS(V_X) - I_RRAM(V_X)``.

        Returns ``(i_r, i_n, di_dvbl__uS, di_dvsl__uS, v_x)`` where
        ``i_r`` is the RRAM current (drained from BL), ``i_n`` is the NMOS
        current (delivered to SL), and the two conductances are the signed
        terminal derivatives of the condensed branch. At cell convergence
        ``i_r == i_n``; their difference is the access-node KCL residual.
        """
        v_wl = snap.v_wl__V
        rram_snap = snap.rram
        nmos_snap = snap.nmos

        # --- Pade current-divider seed for V_X ---

        # First-order split of the BL-to-SL drop across the NMOS output
        # conductance and the programmed RRAM conductance, evaluated at the
        # terminal operating point.
        v_cell_bl_to_sl = v_bl - v_sl
        g_rram_seed = rram_snap.g__uS
        dc_nmos_seed = self.nmos.solve_dc(v_wl, v_bl, v_sl, nmos_snap)
        v_rram_drop_init = dc_nmos_seed.did_dvd__uS * v_cell_bl_to_sl / (dc_nmos_seed.did_dvd__uS + g_rram_seed)
        v_x = v_bl - v_rram_drop_init

        # --- Unrolled per-cell Newton on F_X = I_NMOS - I_RRAM ---

        for _ in range(self.n_newton):
            dc_nmos = self.nmos.solve_dc(v_wl, v_x, v_sl, nmos_snap)
            dc_rram = self.rram.solve_dc(v_bl - v_x, rram_snap)
            f_cell = dc_nmos.ids__uA - dc_rram.i__uA
            df_dvx = dc_nmos.did_dvd__uS + dc_rram.di_dv__uS
            v_x = v_x - f_cell / df_dvx

        # --- Final cell evaluation + signed terminal conductances ---

        dc_nmos = self.nmos.solve_dc(v_wl, v_x, v_sl, nmos_snap)
        dc_rram = self.rram.solve_dc(v_bl - v_x, rram_snap)
        i_n = dc_nmos.ids__uA
        i_r = dc_rram.i__uA
        # Series condensation of the NMOS (V_X = drain) and RRAM
        # conductances at the eliminated access node. ``did_dvd >= 0`` and
        # ``di_dv >= 0`` so ``di_dvbl >= 0``; ``did_dvs <= 0`` so
        # ``di_dvsl <= 0``.
        denom = dc_nmos.did_dvd__uS + dc_rram.di_dv__uS
        di_dvbl__uS = dc_nmos.did_dvd__uS * dc_rram.di_dv__uS / denom
        di_dvsl__uS = dc_nmos.did_dvs__uS * dc_rram.di_dv__uS / denom
        return i_r, i_n, di_dvbl__uS, di_dvsl__uS, v_x

    def solve_branch(
        self,
        v_bl: Tensor,
        v_sl: Tensor,
        snap: XbarCell1t1rDetailSnap,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Condensed branch solve: ``(i__uA, di_dvbl__uS, di_dvsl__uS)``."""
        i_r, _i_n, di_dvbl__uS, di_dvsl__uS, _v_x = self._solve_vx(v_bl, v_sl, snap)
        return i_r, di_dvbl__uS, di_dvsl__uS

    def solve_dc(
        self,
        v_bl: Tensor,
        v_sl: Tensor,
        snap: XbarCell1t1rDetailSnap,
    ) -> XbarCell1t1rDcop:
        """Full branch working point including the condensed ``V_X``.

        Emits the access-node KCL residual at the converged ``V_X`` to
        :class:`XbarCell1t1rDetailProber`, demand-gated: the residual is
        computed and the payload built only when a prober is active.
        """
        i_r, i_n, di_dvbl__uS, di_dvsl__uS, v_x = self._solve_vx(v_bl, v_sl, snap)
        if XbarCell1t1rDetailProber.active():
            XbarCell1t1rDetailProber.submit(XbarCell1t1rDetailObservation(cell__uA=(i_n - i_r).abs()))
        return XbarCell1t1rDcop(
            i__uA=i_r,
            di_dvbl__uS=di_dvbl__uS,
            di_dvsl__uS=di_dvsl__uS,
            v_x__V=v_x,
        )
