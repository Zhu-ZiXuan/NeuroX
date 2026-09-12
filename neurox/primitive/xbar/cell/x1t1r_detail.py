"""Nonlinear 1T1R cell with residual-driven access-node condensation.

See Also:
    docs/reference/primitive/xbar/cell/1t1r_detail.md
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import ClassVar

import torch
from torch import Tensor

from neurox.common.solving import SolvingState, SolvingTrace, run_solving_loop, run_solving_trace_scan
from neurox.common.torch_compat import torch_assert_async
from neurox.primitive.device.mosfet import MosfetConfig, MosfetPolicy, MosfetSnap, Nmos
from neurox.primitive.device.rram import Rram, RramConfig, RramPolicy, RramSnap

from .x1t1r import XbarCell1t1r, XbarCell1t1rConfig, XbarCell1t1rDcop, XbarCell1t1rPolicy, XbarCell1t1rSnap

# ### Access-node solver ###


class _State(SolvingState):
    v_x__V: Tensor


class _Trace(SolvingTrace):
    """Residuals and thresholds are compared before each access-node update.

    Every field has shape `[..., *history]`, preserving each cell position.
    """

    residual__uA: Tensor
    threshold__uA: Tensor
    dv_x_abs__V: Tensor

    @classmethod
    def empty(cls, shape: tuple[int, ...], *, dtype: torch.dtype, device: torch.device) -> _Trace:
        """Construct one unused access-node observation."""
        return cls(
            residual__uA=torch.full(shape, torch.nan, dtype=dtype, device=device),
            threshold__uA=torch.full(shape, torch.nan, dtype=dtype, device=device),
            dv_x_abs__V=torch.full(shape, torch.nan, dtype=dtype, device=device),
        )


class _Solver:
    """Settle the access node and apply a caller-owned terminal projection.

    `final_fn` receives the terminal loop state once, after convergence checks,
    and returns the requested result. `record_trace=True` permits a capped
    unconverged state and returns the projection alongside its raw history.
    The callback runs under no-grad and must support compiled tensor execution.
    """

    MAX_ITER: ClassVar[int] = 20
    FP32_RTOL: ClassVar[float] = 1.3e-3
    FP32_ATOL: ClassVar[float] = 5.0e-6
    FP64_RTOL: ClassVar[float] = 5.0e-12
    FP64_ATOL: ClassVar[float] = 5.0e-15

    def __init__(
        self,
        *,
        rram: Rram,
        nmos: Nmos,
        dtype: torch.dtype,
    ) -> None:
        self.rram = rram
        self.nmos = nmos

        if dtype == torch.float32:
            self._rtol, self._atol = self.FP32_RTOL, self.FP32_ATOL
        elif dtype == torch.float64:
            self._rtol, self._atol = self.FP64_RTOL, self.FP64_ATOL
        else:
            raise TypeError(f"Access-node solve requires float32 or float64 dtype; got {dtype}")

    @torch.no_grad()
    def solve[ResultT](
        self,
        v_bl__V: Tensor,
        v_sl__V: Tensor,
        *,
        v_wl__V: Tensor,
        nmos_snap: MosfetSnap,
        rram_snap: RramSnap,
        final_fn: Callable[[_State], ResultT],
        record_trace: bool,
        trace_mask: Tensor | None,
    ) -> tuple[ResultT, _Trace | None]:
        state, trace = self._solve_vx(
            v_bl__V,
            v_sl__V,
            v_wl__V=v_wl__V,
            nmos_snap=nmos_snap,
            rram_snap=rram_snap,
            record_trace=record_trace,
            trace_mask=trace_mask,
        )
        return final_fn(state), trace

    def _solve_vx(
        self,
        v_bl__V: Tensor,
        v_sl__V: Tensor,
        *,
        v_wl__V: Tensor,
        nmos_snap: MosfetSnap,
        rram_snap: RramSnap,
        record_trace: bool,
        trace_mask: Tensor | None,
    ) -> tuple[_State, _Trace | None]:

        # --- 1: initialize the access-node voltage ---

        v_x_init__V = torch.where((v_wl__V - v_sl__V) > nmos_snap.vth__V, v_sl__V, v_bl__V)
        init_state = _State(
            v_x__V=v_x_init__V,
            is_active=torch.ones_like(v_x_init__V, dtype=torch.bool),
        )

        # --- 2: settle the access node ---

        def body_fn(current: _State) -> tuple[_State, _Trace]:
            return self._evaluate_vx(
                v_bl__V,
                v_sl__V,
                current.v_x__V,
                v_wl__V=v_wl__V,
                nmos_snap=nmos_snap,
                rram_snap=rram_snap,
                is_active=current.is_active,
            )

        if record_trace:
            return run_solving_trace_scan(
                init_state=init_state,
                body_fn=body_fn,
                default_trace=_Trace.empty(
                    tuple(v_x_init__V.shape), dtype=v_x_init__V.dtype, device=v_x_init__V.device
                ),
                max_iter=self.MAX_ITER,
                strict=False,
                trace_mask=trace_mask,
            )

        def solve_body(current: _State) -> _State:
            next_state, _ = body_fn(current)
            return next_state

        final_state = run_solving_loop(
            init_state=init_state,
            body_fn=solve_body,
            max_iter=self.MAX_ITER,
            strict=True,
        )
        return final_state, None

    def _evaluate_vx(
        self,
        v_bl__V: Tensor,
        v_sl__V: Tensor,
        v_x__V: Tensor,
        *,
        v_wl__V: Tensor,
        nmos_snap: MosfetSnap,
        rram_snap: RramSnap,
        is_active: Tensor,
    ) -> tuple[_State, _Trace]:

        # --- 1: evaluate branch currents and the Newton correction ---

        nmos_dcop = self.nmos.solve_dc(v_wl__V, v_x__V, v_sl__V, nmos_snap)
        rram_dcop = self.rram.solve_dc(v_bl__V - v_x__V, rram_snap)

        i_rram__uA = rram_dcop.i__uA
        i_nmos__uA = nmos_dcop.ids__uA
        f_x__uA = i_nmos__uA - i_rram__uA
        # rram's di/dv = d(i)/d(-vx) = d(-i)/d(vx), so use + here
        dfx_dvx__uS = nmos_dcop.did_dvd__uS + rram_dcop.di_dv__uS
        threshold__uA = self._atol + self._rtol * torch.maximum(i_nmos__uA.abs(), i_rram__uA.abs())
        residual__uA = f_x__uA.abs()
        next_is_active = is_active & (residual__uA > threshold__uA)
        dv_x__V = torch.where(next_is_active, -f_x__uA / dfx_dvx__uS, 0)
        next_v_x__V = v_x__V + dv_x__V

        # --- 2: reject non-finite updates ---

        finite = (
            v_x__V.isfinite()
            & f_x__uA.isfinite()
            & dfx_dvx__uS.isfinite()
            & threshold__uA.isfinite()
            & dv_x__V.isfinite()
            & next_v_x__V.isfinite()
        )
        torch_assert_async(
            finite.all(),
            "Access-node Newton solve produced a non-finite state",
        )

        # --- 3: return the updated state and raw observation ---

        state = _State(
            v_x__V=next_v_x__V,
            is_active=next_is_active,
        )
        trace = _Trace(
            residual__uA=residual__uA,
            threshold__uA=threshold__uA,
            dv_x_abs__V=dv_x__V.abs(),
        )
        return state, trace


# ### Detail 1t1r cell ###


class XbarCell1t1rDetailConfig(XbarCell1t1rConfig):
    rram_config: RramConfig
    nmos_config: MosfetConfig

    state_to_g_map__uS: tuple[float, ...]
    """Strictly increasing programmed conductance by state."""

    access_nmos_W__um: float
    access_nmos_L__um: float

    rram_g_max__uS: float
    """Programmable ceiling above `rram_config.g_min__uS`."""

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


class XbarCell1t1rDetailPolicy(XbarCell1t1rPolicy):
    rram_policy: RramPolicy
    nmos_policy: MosfetPolicy


class XbarCell1t1rDetailSnap(XbarCell1t1rSnap):
    rram_snap: RramSnap
    nmos_snap: MosfetSnap


_Dcop = XbarCell1t1rDcop
_Config = XbarCell1t1rDetailConfig
_Policy = XbarCell1t1rDetailPolicy
_Snap = XbarCell1t1rDetailSnap
XbarCell1t1rDetailTrace = _Trace


@XbarCell1t1r.register_neurox_module(config_type=_Config, policy_type=_Policy)
class XbarCell1t1rDetail(XbarCell1t1r[_Config, _Policy, _Snap]):
    """Nonlinear RRAM-NMOS branch condensed at its access node."""

    # === Functional buffers ===

    # Shape: [w_state]
    _state_to_g_map__uS: Tensor

    def __init__(
        self,
        *,
        config: _Config,
        policy: _Policy,
        inst_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
    ) -> None:
        super().__init__(config=config, policy=policy, inst_shape=inst_shape, dtype=dtype, T__K=T__K)

        self._register_nonpersistent_buffer(
            "_state_to_g_map__uS",
            torch.tensor(config.state_to_g_map__uS, dtype=dtype),
        )

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
        self.solver = _Solver(rram=self.rram, nmos=self.nmos, dtype=dtype)

    @property
    def w_state_num(self) -> int:
        return len(self.config.state_to_g_map__uS)

    def snapshot(
        self,
        *,
        control: Tensor,
        shape: tuple[int, ...],
    ) -> _Snap:
        return _Snap(
            rram_snap=self.rram.snapshot(shape=shape),
            nmos_snap=self.nmos.snapshot(shape=shape),
            v_wl__V=control,
        )

    def program(self, w_state_idx: Tensor) -> None:
        target_g__uS = self._state_to_g_map__uS[w_state_idx.long()]
        self.rram.program(target_g__uS)

    def solve_dc_trace(
        self,
        v_bl__V: Tensor,
        v_sl__V: Tensor,
        snap: _Snap,
        *,
        trace_mask: Tensor | None = None,
    ) -> tuple[_Dcop, _Trace]:
        """Return the terminal DCOP and its Newton trajectory.

        The iteration cap returns its unconverged terminal state so the failure
        remains inspectable through the trajectory. `trace_mask` selects the
        cells included in raw observations; `None` includes every cell and
        neither form changes numerical updates.
        """
        dcop, trace = self._solve_dc_impl(v_bl__V, v_sl__V, snap, record_trace=True, trace_mask=trace_mask)
        if trace is None:
            raise RuntimeError("A traced cell solve returned no trace")
        return dcop, trace

    def solve_dc(
        self,
        v_bl__V: Tensor,
        v_sl__V: Tensor,
        snap: _Snap,
    ) -> _Dcop:
        """Return a converged DCOP without allocating trajectory storage."""
        dcop, _ = self._solve_dc_impl(v_bl__V, v_sl__V, snap, record_trace=False, trace_mask=None)
        return dcop

    def _solve_dc_impl(
        self,
        v_bl__V: Tensor,
        v_sl__V: Tensor,
        snap: _Snap,
        *,
        record_trace: bool,
        trace_mask: Tensor | None,
    ) -> tuple[_Dcop, _Trace | None]:
        return self.solver.solve(
            v_bl__V,
            v_sl__V,
            v_wl__V=snap.v_wl__V,
            nmos_snap=snap.nmos_snap,
            rram_snap=snap.rram_snap,
            final_fn=partial(self._dcop_from_state, v_bl__V=v_bl__V, v_sl__V=v_sl__V, snap=snap),
            record_trace=record_trace,
            trace_mask=trace_mask,
        )

    def _dcop_from_state(
        self,
        state: _State,
        *,
        v_bl__V: Tensor,
        v_sl__V: Tensor,
        snap: _Snap,
    ) -> _Dcop:
        nmos_dcop = self.nmos.solve_dc(snap.v_wl__V, state.v_x__V, v_sl__V, snap.nmos_snap)
        rram_dcop = self.rram.solve_dc(v_bl__V - state.v_x__V, snap.rram_snap)
        dfx_dvx__uS = nmos_dcop.did_dvd__uS + rram_dcop.di_dv__uS
        di_dvbl__uS = nmos_dcop.did_dvd__uS * rram_dcop.di_dv__uS / dfx_dvx__uS
        di_dvsl__uS = nmos_dcop.did_dvs__uS * rram_dcop.di_dv__uS / dfx_dvx__uS

        return _Dcop(
            i__uA=rram_dcop.i__uA,
            di_dvbl__uS=di_dvbl__uS,
            di_dvsl__uS=di_dvsl__uS,
            v_x__V=state.v_x__V,
        )
