"""Detailed 1T1R cell — nonlinear device models condensed by a per-cell Newton.

See Also:
    docs/reference/primitive/xbar/cell/1t1r_detail.md
"""

from __future__ import annotations

import torch
from torch import Tensor

from neurox.common import RecordBase, RecorderBase
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

from .x1t1r import (
    XbarCell1t1r,
    XbarCell1t1rConfig,
    XbarCell1t1rDcop,
    XbarCell1t1rPolicy,
    XbarCell1t1rSnap,
)


class XbarCell1t1rDetailRecord(RecordBase):
    cell__uA: Tensor
    """`|I_NMOS - I_RRAM|` per cell at the condensed V_X. Shape: `[..., col, row]`."""


class XbarCell1t1rDetailProber(RecorderBase[XbarCell1t1rDetailRecord]):
    """Capture detailed-cell access-node KCL residuals."""


class XbarCell1t1rDetailConfig(XbarCell1t1rConfig):
    rram_config: RramConfig
    nmos_config: MosfetConfig

    state_to_g_map__uS: tuple[float, ...]
    """State-index to target-conductance lookup table, strictly increasing,
    with both endpoints inside `[rram_config.g_min__uS, rram_g_max__uS]`. Its
    length is the cell's weight-state count."""

    access_nmos_W__um: float
    access_nmos_L__um: float

    rram_g_max__uS: float
    """Upper end of the programmable conductance window."""

    newton_iter_num: int
    """Unrolled per-cell Newton steps on V_X after the Pade current-divider seed."""

    def validate(self) -> None:
        super().validate()

        # --- Access transistor and RRAM window ---

        self._require_pos(self.access_nmos_W__um, "access_nmos_W__um")
        self._require_pos(self.access_nmos_L__um, "access_nmos_L__um")
        self._require_gt(self.rram_g_max__uS, "rram_g_max__uS", self.rram_config.g_min__uS)

        # --- State map ---

        self._require_min_len(self.state_to_g_map__uS, "state_to_g_map__uS", 2)
        self._require_increasing(self.state_to_g_map__uS, "state_to_g_map__uS")
        self._require_ge(self.state_to_g_map__uS[0], "state_to_g_map__uS[0]", self.rram_config.g_min__uS)
        self._require_le(self.state_to_g_map__uS[-1], "state_to_g_map__uS[-1]", self.rram_g_max__uS)

        # --- Solver ---

        self._require_pos(self.newton_iter_num, "newton_iter_num")


class XbarCell1t1rDetailPolicy(XbarCell1t1rPolicy):
    rram_policy: RramPolicy
    nmos_policy: MosfetPolicy


class XbarCell1t1rDetailSnap(XbarCell1t1rSnap):
    rram: RramSnap
    nmos: MosfetSnap


@XbarCell1t1r.register_neurox_module(
    config_type=XbarCell1t1rDetailConfig,
    policy_type=XbarCell1t1rDetailPolicy,
)
class XbarCell1t1rDetail(XbarCell1t1r[XbarCell1t1rDetailConfig, XbarCell1t1rDetailPolicy, XbarCell1t1rDetailSnap]):
    """Series access-NMOS and RRAM cell with a condensed BL-to-SL branch."""

    _state_to_g_map__uS: Tensor  # Shape: [w_state_num]

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
        self._newton_iter_num = config.newton_iter_num

        self._init_children(dtype=dtype, T__K=T__K)

    def _init_children(self, *, dtype: torch.dtype, T__K: float) -> None:
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

    @property
    def w_state_num(self) -> int:
        return len(self.config.state_to_g_map__uS)

    def snapshot(
        self,
        *,
        control: Tensor,
        shape: tuple[int, ...],
        t_elapsed: float,
    ) -> XbarCell1t1rDetailSnap:
        """Bundle RRAM / NMOS device snaps with the WL control drive.

        `t_elapsed` is accepted and unused, reserved for time-dependent device
        read state.
        """
        del t_elapsed
        rram_snap = self.rram.snapshot(shape=shape)
        nmos_snap = self.nmos.snapshot(shape=shape)
        return XbarCell1t1rDetailSnap(rram=rram_snap, nmos=nmos_snap, v_wl__V=control)

    def program(self, w_state_idx: Tensor) -> None:
        """Program the RRAM cells from one state-index tensor.

        Args:
            w_state_idx: State-index tensor in `[0, w_state_num - 1]`.
                Shape: `[*inst_shape]`.
        """
        target_g__uS = self._state_to_g_map__uS[w_state_idx.long()]
        self.rram.program(target_g__uS, t_elapsed=0.0)

    def _solve_vx(
        self,
        v_bl__V: Tensor,
        v_sl__V: Tensor,
        snap: XbarCell1t1rDetailSnap,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Condense the access node V_X.

        Args:
            v_bl__V: Bit-line node voltage.
            v_sl__V: Source-line node voltage.
            snap: Per-call detailed-cell snapshot.

        Returns:
            `(i_rram__uA, i_nmos__uA, di_dvbl__uS, di_dvsl__uS, v_x__V)`.
        """
        v_wl__V = snap.v_wl__V
        rram_snap = snap.rram
        nmos_snap = snap.nmos

        # --- 1: initialize V_X with a Pade current divider ---

        # First-order split of the BL-to-SL drop across the NMOS output
        # conductance and the programmed RRAM conductance, evaluated at the
        # terminal operating point.
        v_cell_bl_to_sl__V = v_bl__V - v_sl__V
        g_rram_seed__uS = rram_snap.g__uS
        dc_nmos_seed = self.nmos.solve_dc(v_wl__V, v_bl__V, v_sl__V, nmos_snap)
        v_rram_drop_init__V = (
            dc_nmos_seed.did_dvd__uS * v_cell_bl_to_sl__V / (dc_nmos_seed.did_dvd__uS + g_rram_seed__uS)
        )
        v_x__V = v_bl__V - v_rram_drop_init__V

        # --- 2: solve F_X = I_NMOS - I_RRAM with Newton iterations ---

        # A fixed trip count: a residual-driven stop would branch on a tensor
        # value and break traceability.
        for _ in range(self._newton_iter_num):
            dc_nmos = self.nmos.solve_dc(v_wl__V, v_x__V, v_sl__V, nmos_snap)
            dc_rram = self.rram.solve_dc(v_bl__V - v_x__V, rram_snap)
            f_cell__uA = dc_nmos.ids__uA - dc_rram.i__uA
            df_dvx__uS = dc_nmos.did_dvd__uS + dc_rram.di_dv__uS
            v_x__V = v_x__V - f_cell__uA / df_dvx__uS

        # --- 3: evaluate the final current and terminal derivatives ---

        dc_nmos = self.nmos.solve_dc(v_wl__V, v_x__V, v_sl__V, nmos_snap)
        dc_rram = self.rram.solve_dc(v_bl__V - v_x__V, rram_snap)
        # The RRAM leg is the reported branch current; the NMOS leg differs
        # from it by the access-node KCL residual the prober measures.
        i_n__uA = dc_nmos.ids__uA
        i_r__uA = dc_rram.i__uA
        # Series condensation of the NMOS (V_X = drain) and RRAM
        # conductances at the eliminated access node. `did_dvd__uS >= 0` and
        # `di_dv__uS >= 0` so `di_dvbl__uS >= 0`; `did_dvs__uS <= 0` so
        # `di_dvsl__uS <= 0`.
        denom__uS = dc_nmos.did_dvd__uS + dc_rram.di_dv__uS
        di_dvbl__uS = dc_nmos.did_dvd__uS * dc_rram.di_dv__uS / denom__uS
        di_dvsl__uS = dc_nmos.did_dvs__uS * dc_rram.di_dv__uS / denom__uS
        return i_r__uA, i_n__uA, di_dvbl__uS, di_dvsl__uS, v_x__V

    def solve_branch(
        self,
        v_bl__V: Tensor,
        v_sl__V: Tensor,
        snap: XbarCell1t1rDetailSnap,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Condensed branch solve: `(i__uA, di_dvbl__uS, di_dvsl__uS)`."""
        i_r__uA, _i_n__uA, di_dvbl__uS, di_dvsl__uS, _v_x__V = self._solve_vx(v_bl__V, v_sl__V, snap)
        return i_r__uA, di_dvbl__uS, di_dvsl__uS

    def solve_dc(
        self,
        v_bl__V: Tensor,
        v_sl__V: Tensor,
        snap: XbarCell1t1rDetailSnap,
    ) -> XbarCell1t1rDcop:
        """Return the branch working point including the condensed V_X."""
        i_r__uA, i_n__uA, di_dvbl__uS, di_dvsl__uS, v_x__V = self._solve_vx(v_bl__V, v_sl__V, snap)
        if XbarCell1t1rDetailProber.active():
            XbarCell1t1rDetailProber.submit(XbarCell1t1rDetailRecord(cell__uA=(i_n__uA - i_r__uA).abs()))
        return XbarCell1t1rDcop(
            i__uA=i_r__uA,
            di_dvbl__uS=di_dvbl__uS,
            di_dvsl__uS=di_dvsl__uS,
            v_x__V=v_x__V,
        )
