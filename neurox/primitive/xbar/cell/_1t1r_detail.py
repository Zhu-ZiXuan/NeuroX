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


@dataclass(frozen=True)
class XbarCell1t1rDetailObservation:
    """Per-cell access-node KCL residual of a detailed 1T1R branch solve.

    Attributes:
        cell__uA: ``|I_NMOS - I_RRAM|`` per cell at the condensed ``V_X``.
            Shape: ``[..., col, row]``.
    """

    cell__uA: Tensor

    def detach(self) -> Self:
        return replace(self, cell__uA=self.cell__uA.detach())


class XbarCell1t1rDetailProber(Prober[XbarCell1t1rDetailObservation]):
    """Capture detailed-cell access-node KCL residuals."""

    _active_stack: ClassVar[list[Prober[XbarCell1t1rDetailObservation]]] = []

    @classmethod
    def _stack(cls) -> list[Prober[XbarCell1t1rDetailObservation]]:
        return cls._active_stack


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
        newton_iter_num: Number of unrolled per-cell Newton steps on ``V_X``
            after the Pade current-divider seed.
    """

    rram_config: RramConfig
    nmos_config: MosfetConfig

    state_to_g_map__uS: tuple[float, ...]

    access_nmos_W__um: float
    access_nmos_L__um: float

    rram_g_max__uS: float

    newton_iter_num: int

    def validate(self) -> None:
        super().validate()

        # --- Access transistor and RRAM window ---

        self._require_pos(self.access_nmos_W__um, "access_nmos_W__um")
        self._require_pos(self.access_nmos_L__um, "access_nmos_L__um")
        if not (self.rram_g_max__uS > self.rram_config.g_min__uS):
            raise ValueError(
                f"require: rram_g_max__uS ({self.rram_g_max__uS}) > rram_config.g_min__uS ({self.rram_config.g_min__uS})"
            )

        # --- State map ---

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

        # --- Solver ---

        self._require_pos(self.newton_iter_num, "newton_iter_num")


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


@XbarCell1t1r.register_neurox_module(
    config_type=XbarCell1t1rDetailConfig,
    policy_type=XbarCell1t1rDetailPolicy,
)
class XbarCell1t1rDetail(XbarCell1t1r[XbarCell1t1rDetailConfig, XbarCell1t1rDetailPolicy, XbarCell1t1rDetailSnap]):
    """Series access-NMOS and RRAM cell with a condensed BL-to-SL branch.

    Args:
        config: Detailed 1T1R configuration.
        policy: Detailed 1T1R nonideality policy.
        inst_shape: Per-instance shape ``(*prefix, col, row)``.
        dtype: Tensor dtype for internal buffers.
        T__K: Operating temperature.
    """

    # --- Immutable model buffers ---

    _state_to_g_map__uS: Tensor

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
            "_state_to_g_map__uS",
            torch.tensor(config.state_to_g_map__uS, dtype=dtype),
            persistent=False,
        )
        self.w_state_num = len(config.state_to_g_map__uS)
        self._newton_iter_num = config.newton_iter_num

        self._init_children(dtype=dtype, T__K=T__K)

    def _init_children(self, *, dtype: torch.dtype, T__K: float) -> None:
        """Construct the storage and access devices."""
        config = self.config
        policy = self.policy
        self.rram = Rram(
            config=config.rram_config,
            policy=policy.rram_policy,
            inst_shape=self.inst_shape,
            dtype=dtype,
            T__K=T__K,
            g_max__uS=config.rram_g_max__uS,
        )
        self.nmos = Nmos(
            config=config.nmos_config,
            policy=policy.nmos_policy,
            inst_shape=self.inst_shape,
            dtype=dtype,
            T__K=T__K,
            W__um=config.access_nmos_W__um,
            L__um=config.access_nmos_L__um,
        )

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
        del t_elapsed
        rram_snap = self.rram.snapshot(shape=shape, multi_coords=multi_coords)
        nmos_snap = self.nmos.snapshot(shape=shape, multi_coords=multi_coords)
        return XbarCell1t1rDetailSnap(rram=rram_snap, nmos=nmos_snap, v_wl__V=control)

    def program(self, w_state_idx: Tensor) -> None:
        """Program the RRAM cells from one state-index tensor.

        Args:
            w_state_idx: State-index tensor in ``[0, w_state_num - 1]`` at
                ``self.inst_shape``.
        """
        target_g__uS = self._state_to_g_map__uS[w_state_idx.long()]
        self.rram.program(target_g__uS, t_elapsed=0.0)

    def _solve_vx(
        self,
        v_bl: Tensor,
        v_sl: Tensor,
        snap: XbarCell1t1rDetailSnap,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Condense the access node ``V_X``.

        Args:
            v_bl: Bit-line node voltage [V].
            v_sl: Source-line node voltage [V].
            snap: Per-call detailed-cell snapshot.

        Returns:
            Tuple ``(i_rram, i_nmos, di_dvbl, di_dvsl, v_x)``.
        """
        v_wl = snap.v_wl__V
        rram_snap = snap.rram
        nmos_snap = snap.nmos

        # --- 1: initialize V_X with a Pade current divider ---

        # First-order split of the BL-to-SL drop across the NMOS output
        # conductance and the programmed RRAM conductance, evaluated at the
        # terminal operating point.
        v_cell_bl_to_sl = v_bl - v_sl
        g_rram_seed = rram_snap.g__uS
        dc_nmos_seed = self.nmos.solve_dc(v_wl, v_bl, v_sl, nmos_snap)
        v_rram_drop_init = dc_nmos_seed.did_dvd__uS * v_cell_bl_to_sl / (dc_nmos_seed.did_dvd__uS + g_rram_seed)
        v_x = v_bl - v_rram_drop_init

        # --- 2: solve F_X = I_NMOS - I_RRAM with Newton iterations ---

        for _ in range(self._newton_iter_num):
            dc_nmos = self.nmos.solve_dc(v_wl, v_x, v_sl, nmos_snap)
            dc_rram = self.rram.solve_dc(v_bl - v_x, rram_snap)
            f_cell = dc_nmos.ids__uA - dc_rram.i__uA
            df_dvx = dc_nmos.did_dvd__uS + dc_rram.di_dv__uS
            v_x = v_x - f_cell / df_dvx

        # --- 3: evaluate the final current and terminal derivatives ---

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
        """Return the branch working point including condensed ``V_X``."""
        i_r, i_n, di_dvbl__uS, di_dvsl__uS, v_x = self._solve_vx(v_bl, v_sl, snap)
        if XbarCell1t1rDetailProber.active():
            XbarCell1t1rDetailProber.submit(XbarCell1t1rDetailObservation(cell__uA=(i_n - i_r).abs()))
        return XbarCell1t1rDcop(
            i__uA=i_r,
            di_dvbl__uS=di_dvbl__uS,
            di_dvsl__uS=di_dvsl__uS,
            v_x__V=v_x,
        )
