"""Residual-driven DC solve for arrays with parallel column BL/SL rails.

See Also:
    docs/reference/primitive/xbar/solver/col_bl_col_sl.md
"""

from __future__ import annotations

__all__ = [
    "ColBlColSlArraySolver",
    "ColBlColSlArrayState",
    "ColBlColSlArrayTrace",
]

from collections.abc import Callable
from typing import ClassVar

import torch
from torch import Tensor
from torch._dynamo.decorators import patch_dynamo_config

from neurox.common.solving import SolvingState, SolvingTrace, run_solving_loop, run_solving_trace_scan
from neurox.common.torch_compat import torch_assert_async

from ._linalg import (
    boundary_inverse_block_tridiagonal_2x2,
    solve_2x2,
    solve_block_tridiagonal_2x2,
)
from ._wire_kcl import dfkcl_dvnode__uS, f_kcl__uA, f_kcl_roundoff__uA
from .clamp_driver import ClampDcop, ClampDriver, ClampSnap
from .resistive_cell import ResistiveCell, ResistiveCellDcop


def _node_jacobian_components__uS(
    g_cell_bl_eff__uS: Tensor,
    g_cell_sl_eff__uS: Tensor,
    bl_g__uS: float,
    sl_g__uS: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    # Jacobian entries use KCL-equation rows and node-voltage columns.
    return (
        dfkcl_dvnode__uS(g_cell_bl_eff__uS, bl_g__uS, dim=-1),
        -g_cell_sl_eff__uS,
        -g_cell_bl_eff__uS,
        dfkcl_dvnode__uS(g_cell_sl_eff__uS, sl_g__uS, dim=-1),
    )


# ### Node solver ###


class _NodeState(SolvingState):
    """Node voltages at the current iterate under fixed port voltages."""

    v_bl_node__V: Tensor
    """Shape: `[..., col, row]`."""
    v_sl_node__V: Tensor
    """Shape: `[..., col, row]`."""


class _NodeTrace(SolvingTrace):
    """Residuals and thresholds are compared before each node update.

    Boolean fields have shape `[..., col, *history]`; residuals, thresholds,
    and updates have shape `[..., col, row, *history]`.
    """

    limited: Tensor
    residual__uA: Tensor
    threshold__uA: Tensor
    dv_bl_node_abs__V: Tensor
    dv_sl_node_abs__V: Tensor

    @classmethod
    def empty(
        cls,
        node_shape: tuple[int, ...],
        *,
        history_shape: tuple[int, ...] = (),
        dtype: torch.dtype,
        device: torch.device,
    ) -> _NodeTrace:
        """Construct unused observations with explicit position and history axes."""

        def flags() -> Tensor:
            return torch.zeros((*node_shape[:-1], *history_shape), dtype=torch.bool, device=device)

        def values() -> Tensor:
            return torch.full((*node_shape, *history_shape), torch.nan, dtype=dtype, device=device)

        return cls(
            limited=flags(),
            residual__uA=values(),
            threshold__uA=values(),
            dv_bl_node_abs__V=values(),
            dv_sl_node_abs__V=values(),
        )


class _NodeSolver[CellSnapT, CellDcopT: ResistiveCellDcop]:
    """Solve wire KCL at fixed port voltages, then apply `final_fn`.

    Each call supplies initial node voltages and participating columns. Node
    voltages have shape `[..., col, row]`; ports and activity have shape
    `[..., col]`. `record_trace=False` requires convergence and returns no
    history. `record_trace=True` permits a capped terminal state and retains
    raw history; `trace_mask` selects observations without changing activity.
    The callback runs once after the terminal check, under no-grad, and must
    support compiled tensor execution.
    """

    MAX_ITER: ClassVar[int] = 20
    MAX_STEP: ClassVar[float] = 0.05
    FP32_RTOL: ClassVar[float] = 1.0e-5
    FP32_ATOL: ClassVar[float] = 5.0e-6
    FP64_RTOL: ClassVar[float] = 5.0e-12
    FP64_ATOL: ClassVar[float] = 5.0e-15

    def __init__(
        self,
        *,
        bl_segment_r__MOhm: float,
        sl_segment_r__MOhm: float,
        cell: ResistiveCell[CellSnapT, CellDcopT],
        dtype: torch.dtype,
    ) -> None:
        self.cell = cell
        self.bl_g__uS = 1.0 / bl_segment_r__MOhm
        self.sl_g__uS = 1.0 / sl_segment_r__MOhm

        if dtype == torch.float32:
            self.rtol = self.FP32_RTOL
            self.atol = self.FP32_ATOL
        elif dtype == torch.float64:
            self.rtol = self.FP64_RTOL
            self.atol = self.FP64_ATOL
        else:
            raise TypeError(f"Wire node solve requires float32 or float64, got {dtype}")

    @torch.no_grad()
    def solve[ResultT](
        self,
        v_bl_node__V: Tensor,
        v_sl_node__V: Tensor,
        v_bl_port__V: Tensor,
        v_sl_port__V: Tensor,
        *,
        cell_snap: CellSnapT,
        final_fn: Callable[[_NodeState], ResultT],
        record_trace: bool,
        trace_mask: Tensor | None,
        is_active: Tensor,
    ) -> tuple[ResultT, _NodeTrace | None]:
        """Return the caller's terminal projection and optional node history."""
        state, trace = self._solve_node(
            v_bl_node__V,
            v_sl_node__V,
            v_bl_port__V,
            v_sl_port__V,
            cell_snap=cell_snap,
            record_trace=record_trace,
            trace_mask=trace_mask,
            is_active=is_active,
        )
        return final_fn(state), trace

    def _solve_node(
        self,
        v_bl_node__V: Tensor,
        v_sl_node__V: Tensor,
        v_bl_port__V: Tensor,
        v_sl_port__V: Tensor,
        *,
        cell_snap: CellSnapT,
        record_trace: bool,
        trace_mask: Tensor | None,
        is_active: Tensor,
    ) -> tuple[_NodeState, _NodeTrace | None]:
        """Settle node voltages while holding both ports fixed."""
        # --- 1: initialize node voltages ---

        init_state = _NodeState(
            v_bl_node__V=v_bl_node__V,
            v_sl_node__V=v_sl_node__V,
            is_active=is_active.clone(),
        )

        # --- 2: settle the wire ladders at fixed port voltages ---

        def body_fn(current: _NodeState) -> tuple[_NodeState, _NodeTrace]:
            return self._evaluate_node(
                v_bl_port__V,
                v_sl_port__V,
                current.v_bl_node__V,
                current.v_sl_node__V,
                is_active=current.is_active,
                cell_snap=cell_snap,
            )

        if record_trace:
            return run_solving_trace_scan(
                init_state=init_state,
                body_fn=body_fn,
                default_trace=_NodeTrace.empty(
                    tuple(v_bl_node__V.shape), dtype=v_bl_node__V.dtype, device=v_bl_node__V.device
                ),
                max_iter=self.MAX_ITER,
                strict=False,
                trace_mask=trace_mask,
            )

        def solve_body(current: _NodeState) -> _NodeState:
            next_state, _ = body_fn(current)
            return next_state

        final_state = run_solving_loop(init_state=init_state, body_fn=solve_body, max_iter=self.MAX_ITER, strict=True)
        return final_state, None

    def _evaluate_node(
        self,
        v_bl_port__V: Tensor,
        v_sl_port__V: Tensor,
        v_bl_node__V: Tensor,
        v_sl_node__V: Tensor,
        is_active: Tensor,
        *,
        cell_snap: CellSnapT,
    ) -> tuple[_NodeState, _NodeTrace]:

        # --- 1: evaluate cell branches and node KCL ---

        # Shape: [..., col, row]
        cell_dcop = self.cell.solve_dc(v_bl_node__V, v_sl_node__V, cell_snap)
        # KCL counts currents leaving each node. The branch leaves BL and enters SL.
        # Shape: [..., col, row]
        f_bl__uA = f_kcl__uA(v_bl_node__V, v_bl_port__V.unsqueeze(-1), self.bl_g__uS, cell_dcop.i__uA, dim=-1)
        f_sl__uA = f_kcl__uA(v_sl_node__V, v_sl_port__V.unsqueeze(-1), self.sl_g__uS, -cell_dcop.i__uA, dim=-1)

        # --- 2: evaluate node stopping thresholds ---

        # Local wire rounding sets a floor even when a cell carries little current.
        # Shape: [..., col, row]
        roundoff__uA = torch.maximum(
            f_kcl_roundoff__uA(v_bl_node__V, v_bl_port__V.unsqueeze(-1), self.bl_g__uS, dim=-1),
            f_kcl_roundoff__uA(v_sl_node__V, v_sl_port__V.unsqueeze(-1), self.sl_g__uS, dim=-1),
        )
        threshold__uA = self.atol + self.rtol * cell_dcop.i__uA.abs() + roundoff__uA

        # --- 3: form the next coupled node correction ---

        # Adjacent rows couple through same-rail wires; each block keeps the BL/SL
        # equations of one row together. The off-block is a constant diagonal pair.
        # Shape: [..., col, row]
        dv_bl_node__V, dv_sl_node__V = solve_block_tridiagonal_2x2(
            diag=_node_jacobian_components__uS(
                cell_dcop.di_dvbl__uS, -cell_dcop.di_dvsl__uS, self.bl_g__uS, self.sl_g__uS
            ),
            rhs=(-f_bl__uA, -f_sl__uA),
            off_diag=(-self.bl_g__uS, -self.sl_g__uS),
        )

        # --- 4: reject non-finite updates ---

        finite = (
            v_bl_node__V.isfinite()
            & v_sl_node__V.isfinite()
            & v_bl_port__V.unsqueeze(-1).isfinite()
            & v_sl_port__V.unsqueeze(-1).isfinite()
            & cell_dcop.i__uA.isfinite()
            & cell_dcop.di_dvbl__uS.isfinite()
            & cell_dcop.di_dvsl__uS.isfinite()
            & f_bl__uA.isfinite()
            & f_sl__uA.isfinite()
            & threshold__uA.isfinite()
            & dv_bl_node__V.isfinite()
            & dv_sl_node__V.isfinite()
        )
        torch_assert_async(finite.all(), "Node Newton solve produced a non-finite state")

        # --- 5: return the updated state and raw observation ---

        # Shape: [..., col]
        residual__uA = torch.maximum(f_bl__uA.abs(), f_sl__uA.abs())
        next_is_active = is_active & (residual__uA > threshold__uA).any(dim=-1)

        applied_dv_bl_node__V = torch.where(
            next_is_active.unsqueeze(-1), dv_bl_node__V.clamp(-self.MAX_STEP, self.MAX_STEP), 0
        )
        applied_dv_sl_node__V = torch.where(
            next_is_active.unsqueeze(-1), dv_sl_node__V.clamp(-self.MAX_STEP, self.MAX_STEP), 0
        )
        state = _NodeState(
            v_bl_node__V=v_bl_node__V + applied_dv_bl_node__V,
            v_sl_node__V=v_sl_node__V + applied_dv_sl_node__V,
            is_active=next_is_active,
        )
        limited = next_is_active & ((dv_bl_node__V.abs() > self.MAX_STEP) | (dv_sl_node__V.abs() > self.MAX_STEP)).any(
            dim=-1
        )
        trace = _NodeTrace(
            limited=limited,
            residual__uA=residual__uA,
            threshold__uA=threshold__uA,
            dv_bl_node_abs__V=applied_dv_bl_node__V.abs(),
            dv_sl_node_abs__V=applied_dv_sl_node__V.abs(),
        )
        return state, trace


# ### Port solver ###


class _PortState(SolvingState):
    """Voltages exposed to the port solve's terminal callback.

    A capped diagnostic state may contain a port update whose nodes have not
    yet been re-solved. Only the ordinary solve guarantees joint convergence.
    """

    v_bl_node__V: Tensor
    """Shape: `[..., col, row]`."""
    v_sl_node__V: Tensor
    """Shape: `[..., col, row]`."""
    v_bl_port__V: Tensor
    """Shape: `[..., col]`."""
    v_sl_port__V: Tensor
    """Shape: `[..., col]`."""


class _PortTrace(SolvingTrace):
    """Port observations with the corresponding node history.

    Port fields have shape `[..., col, *history]`. Nested node fields insert
    their row and node-iteration axes before the port history axes.
    `node_trace` is present throughout a traced solve, including unused steps.
    """

    limited: Tensor
    residual__V: Tensor
    threshold__V: Tensor
    dv_bl_port_abs__V: Tensor
    dv_sl_port_abs__V: Tensor
    node_trace: _NodeTrace | None

    @classmethod
    def empty(
        cls, node_shape: tuple[int, ...], *, node_capacity: int, dtype: torch.dtype, device: torch.device
    ) -> _PortTrace:
        """Construct one unused port observation and its nested node history."""

        def flags() -> Tensor:
            return torch.zeros(node_shape[:-1], dtype=torch.bool, device=device)

        def values() -> Tensor:
            return torch.full(node_shape[:-1], torch.nan, dtype=dtype, device=device)

        return cls(
            limited=flags(),
            residual__V=values(),
            threshold__V=values(),
            dv_bl_port_abs__V=values(),
            dv_sl_port_abs__V=values(),
            node_trace=_NodeTrace.empty(node_shape, history_shape=(node_capacity,), dtype=dtype, device=device),
        )


class _PortSolver[CellSnapT, CellDcopT: ResistiveCellDcop, BLSnapT: ClampSnap, SLSnapT: ClampSnap]:
    """DC solver for parallel column BL/SL rails.

    Cell and driver references are borrowed; their owner handles lifecycle and
    accounting. Calls are stateless and receive per-call snapshots.

    Args:
        bl_segment_r__MOhm: Uniform BL resistance of one lattice link,
            including the boundary link.
        sl_segment_r__MOhm: Uniform SL resistance of one lattice link,
            including the boundary link.
        cell: Condensed two-terminal branch evaluated at every array node.
        bl_driver: Borrowed BL clamp; its owner manages lifecycle and accounting.
        sl_driver: Borrowed SL clamp; its owner manages lifecycle and accounting.
        dtype: Floating dtype selecting the stopping tolerances. Only
            `torch.float32` and `torch.float64` are supported.
    """

    MAX_ITER: ClassVar[int] = 20
    MAX_STEP: ClassVar[float] = 0.10
    FP32_RTOL: ClassVar[float] = 1.0e-5
    FP32_ATOL: ClassVar[float] = 1.0e-7
    FP64_RTOL: ClassVar[float] = 5.0e-12
    FP64_ATOL: ClassVar[float] = 5.0e-15
    WIRE_ROUNDOFF_FACTOR: ClassVar[float] = 2.0

    def __init__(
        self,
        *,
        bl_segment_r__MOhm: float,
        sl_segment_r__MOhm: float,
        cell: ResistiveCell[CellSnapT, CellDcopT],
        bl_driver: ClampDriver[BLSnapT, ClampDcop],
        sl_driver: ClampDriver[SLSnapT, ClampDcop],
        dtype: torch.dtype,
    ) -> None:
        self.cell = cell
        self.bl_driver = bl_driver
        self.sl_driver = sl_driver
        self.bl_segment_r__MOhm = bl_segment_r__MOhm
        self.sl_segment_r__MOhm = sl_segment_r__MOhm

        if dtype == torch.float32:
            self.rtol = self.FP32_RTOL
            self.atol = self.FP32_ATOL
        elif dtype == torch.float64:
            self.rtol = self.FP64_RTOL
            self.atol = self.FP64_ATOL
        else:
            raise TypeError(f"Parallel BL/SL solve requires float32 or float64, got {dtype}")

        # Allow voltage rounding from both wire links attached to an interior node.
        self.roundoff = self.WIRE_ROUNDOFF_FACTOR * torch.finfo(dtype).eps
        self.bl_g__uS = 1.0 / bl_segment_r__MOhm
        self.sl_g__uS = 1.0 / sl_segment_r__MOhm
        self.node_solver = _NodeSolver(
            bl_segment_r__MOhm=bl_segment_r__MOhm,
            sl_segment_r__MOhm=sl_segment_r__MOhm,
            cell=cell,
            dtype=dtype,
        )

    @patch_dynamo_config(capture_scalar_outputs=True)
    @torch.no_grad()
    @torch.compile(dynamic=False)
    def solve[ResultT](
        self,
        *,
        cell_snap: CellSnapT,
        bl_driver_snap: BLSnapT,
        sl_driver_snap: SLSnapT,
        final_fn: Callable[[_PortState], ResultT],
        record_trace: bool,
        trace_mask: Tensor | None,
    ) -> tuple[ResultT, _PortTrace | None]:
        """Return the caller's terminal projection and optional raw history.

        Snaps share the complete runtime leading shape: cells at
        `[..., col, row]` and clamps at `[..., col]`. `record_trace=False`
        requires both loops to converge; `True` permits capped terminal states.
        `trace_mask` selects column observations only. The callback receives
        terminal node and port voltages once, under no-grad, and must support
        compiled tensor execution. It owns the output's structure and meaning.
        """
        # --- 1: initialize every port and node at its nominal rail reference ---

        row_num = self.cell.inst_shape[-1]
        # Materialize reference broadcasts so every loop-carried tensor has the
        # same dense layout as the values returned by its first update.
        # Shape: [..., col]
        v_bl_port__V = bl_driver_snap.v_ref__V.clone(memory_format=torch.contiguous_format)
        v_sl_port__V = sl_driver_snap.v_ref__V.clone(memory_format=torch.contiguous_format)
        # Shape: [..., col, row]
        node_shape = (*v_bl_port__V.shape, row_num)
        v_bl_node__V = v_bl_port__V.unsqueeze(-1).expand(node_shape).clone(memory_format=torch.contiguous_format)
        v_sl_node__V = v_sl_port__V.unsqueeze(-1).expand(node_shape).clone(memory_format=torch.contiguous_format)

        # --- 2: solve the coupled port and node equilibrium ---

        state, trace = self._solve_port(
            v_bl_node__V,
            v_sl_node__V,
            v_bl_port__V,
            v_sl_port__V,
            cell_snap=cell_snap,
            bl_driver_snap=bl_driver_snap,
            sl_driver_snap=sl_driver_snap,
            record_trace=record_trace,
            trace_mask=trace_mask,
        )

        return final_fn(state), trace

    def _solve_port(
        self,
        v_bl_node__V: Tensor,
        v_sl_node__V: Tensor,
        v_bl_port__V: Tensor,
        v_sl_port__V: Tensor,
        *,
        cell_snap: CellSnapT,
        bl_driver_snap: BLSnapT,
        sl_driver_snap: SLSnapT,
        record_trace: bool,
        trace_mask: Tensor | None,
    ) -> tuple[_PortState, _PortTrace | None]:
        """Settle both port voltages, re-solving the nodes after each update."""
        # --- 1: initialize port voltages ---

        init_state = _PortState(
            v_bl_node__V=v_bl_node__V,
            v_sl_node__V=v_sl_node__V,
            v_bl_port__V=v_bl_port__V,
            v_sl_port__V=v_sl_port__V,
            is_active=torch.ones_like(v_bl_port__V, dtype=torch.bool),
        )

        # --- 2: settle the ports, solving the wire ladders after each update ---

        def body_fn(current: _PortState) -> tuple[_PortState, _PortTrace]:
            node_state, node_trace = self.node_solver.solve(
                current.v_bl_node__V,
                current.v_sl_node__V,
                current.v_bl_port__V,
                current.v_sl_port__V,
                cell_snap=cell_snap,
                is_active=current.is_active,
                final_fn=lambda state: state,
                record_trace=record_trace,
                trace_mask=trace_mask,
            )
            return self._evaluate_port(
                node_state,
                node_trace=node_trace,
                v_bl_port__V=current.v_bl_port__V,
                v_sl_port__V=current.v_sl_port__V,
                cell_snap=cell_snap,
                is_active=current.is_active,
                bl_driver_snap=bl_driver_snap,
                sl_driver_snap=sl_driver_snap,
            )

        if record_trace:
            return run_solving_trace_scan(
                init_state=init_state,
                body_fn=body_fn,
                default_trace=_PortTrace.empty(
                    tuple(v_bl_node__V.shape),
                    node_capacity=self.node_solver.MAX_ITER,
                    dtype=v_bl_node__V.dtype,
                    device=v_bl_node__V.device,
                ),
                max_iter=self.MAX_ITER,
                strict=False,
                trace_mask=trace_mask,
            )

        def solve_body(current: _PortState) -> _PortState:
            next_state, _ = body_fn(current)
            return next_state

        final_state = run_solving_loop(init_state=init_state, body_fn=solve_body, max_iter=self.MAX_ITER, strict=True)
        return final_state, None

    def _evaluate_port(
        self,
        node_state: _NodeState,
        *,
        node_trace: _NodeTrace | None,
        v_bl_port__V: Tensor,
        v_sl_port__V: Tensor,
        is_active: Tensor,
        cell_snap: CellSnapT,
        bl_driver_snap: BLSnapT,
        sl_driver_snap: SLSnapT,
    ) -> tuple[_PortState, _PortTrace]:

        # --- 1: evaluate driver targets and port residuals ---

        v_bl_node__V = node_state.v_bl_node__V
        v_sl_node__V = node_state.v_sl_node__V
        i_bl_port__uA = (v_bl_port__V - v_bl_node__V[..., 0]) * self.bl_g__uS
        i_sl_port__uA = (v_sl_port__V - v_sl_node__V[..., 0]) * self.sl_g__uS
        # Shape: [..., col]
        bl_driver_dcop = self.bl_driver.solve_dc(i_bl_port__uA, bl_driver_snap, v_port_init__V=v_bl_port__V)
        sl_driver_dcop = self.sl_driver.solve_dc(i_sl_port__uA, sl_driver_snap, v_port_init__V=v_sl_port__V)
        # Shape: [..., col]
        f_bl_port__V = bl_driver_dcop.v_port__V - v_bl_port__V
        f_sl_port__V = sl_driver_dcop.v_port__V - v_sl_port__V

        # --- 2: compute the port Jacobian and boundary correction ---

        cell_dcop = self.cell.solve_dc(node_state.v_bl_node__V, node_state.v_sl_node__V, cell_snap)
        torch_assert_async(
            cell_dcop.di_dvbl__uS.isfinite().all() & cell_dcop.di_dvsl__uS.isfinite().all(),
            "Port linearization produced a non-finite state",
        )
        u_bl_bl, u_bl_sl, u_sl_bl, u_sl_sl = boundary_inverse_block_tridiagonal_2x2(
            diag=_node_jacobian_components__uS(
                cell_dcop.di_dvbl__uS, -cell_dcop.di_dvsl__uS, self.bl_g__uS, self.sl_g__uS
            ),
            off_diag=(-self.bl_g__uS, -self.sl_g__uS),
        )
        di_bl_dv_bl__uS = self.bl_g__uS * (1.0 - u_bl_bl * self.bl_g__uS)
        di_bl_dv_sl__uS = -self.bl_g__uS * u_bl_sl * self.sl_g__uS
        di_sl_dv_bl__uS = -self.sl_g__uS * u_sl_bl * self.bl_g__uS
        di_sl_dv_sl__uS = self.sl_g__uS * (1.0 - u_sl_sl * self.sl_g__uS)

        # Signed target-voltage response to a boundary-link voltage drop.
        # Its sign sets feedback polarity; its magnitude also scales roundoff below.
        bl_driver_gain = bl_driver_dcop.dvport_di__MOhm * self.bl_g__uS
        sl_driver_gain = sl_driver_dcop.dvport_di__MOhm * self.sl_g__uS
        # Shape: [..., col]
        dv_bl_port__V, dv_sl_port__V = solve_2x2(
            (
                bl_driver_dcop.dvport_di__MOhm * di_bl_dv_bl__uS - 1.0,
                bl_driver_dcop.dvport_di__MOhm * di_bl_dv_sl__uS,
                sl_driver_dcop.dvport_di__MOhm * di_sl_dv_bl__uS,
                sl_driver_dcop.dvport_di__MOhm * di_sl_dv_sl__uS - 1.0,
            ),
            (-f_bl_port__V, -f_sl_port__V),
        )

        # --- 3: evaluate stopping thresholds and assemble the port state ---

        # Shape: [..., col]
        v_bl_port_scale__V = torch.maximum(v_bl_port__V.abs(), v_bl_node__V[..., 0].abs())
        v_sl_port_scale__V = torch.maximum(v_sl_port__V.abs(), v_sl_node__V[..., 0].abs())
        v_port_scale__V = torch.maximum(
            torch.maximum(v_bl_port_scale__V, v_sl_port_scale__V),
            torch.maximum(bl_driver_dcop.v_port__V.abs(), sl_driver_dcop.v_port__V.abs()),
        )
        # For an error allowance, use the magnitude of the driver's response:
        # boundary-current rounding becomes voltage error through dV/dI.
        threshold__V = (
            self.atol
            + self.rtol * v_port_scale__V
            + self.roundoff
            * torch.maximum(bl_driver_gain.abs() * v_bl_port_scale__V, sl_driver_gain.abs() * v_sl_port_scale__V)
        )

        finite = (
            v_bl_port__V.isfinite()
            & v_sl_port__V.isfinite()
            & i_bl_port__uA.isfinite()
            & i_sl_port__uA.isfinite()
            & f_bl_port__V.isfinite()
            & f_sl_port__V.isfinite()
            & bl_driver_dcop.dvport_di__MOhm.isfinite()
            & sl_driver_dcop.dvport_di__MOhm.isfinite()
            & threshold__V.isfinite()
            & dv_bl_port__V.isfinite()
            & dv_sl_port__V.isfinite()
        )
        torch_assert_async(
            finite.all(),
            "Port Newton solve produced a non-finite state",
        )

        # Shape: [..., col]
        residual__V = torch.maximum(f_bl_port__V.abs(), f_sl_port__V.abs())
        next_is_active = is_active & (residual__V > threshold__V)

        applied_dv_bl_port__V = torch.where(next_is_active, dv_bl_port__V.clamp(-self.MAX_STEP, self.MAX_STEP), 0)
        applied_dv_sl_port__V = torch.where(next_is_active, dv_sl_port__V.clamp(-self.MAX_STEP, self.MAX_STEP), 0)
        state = _PortState(
            v_bl_node__V=v_bl_node__V,
            v_sl_node__V=v_sl_node__V,
            v_bl_port__V=v_bl_port__V + applied_dv_bl_port__V,
            v_sl_port__V=v_sl_port__V + applied_dv_sl_port__V,
            is_active=next_is_active,
        )
        limited = next_is_active & ((dv_bl_port__V.abs() > self.MAX_STEP) | (dv_sl_port__V.abs() > self.MAX_STEP))
        trace = _PortTrace(
            limited=limited,
            residual__V=residual__V,
            threshold__V=threshold__V,
            dv_bl_port_abs__V=applied_dv_bl_port__V.abs(),
            dv_sl_port_abs__V=applied_dv_sl_port__V.abs(),
            node_trace=node_trace,
        )
        return state, trace


# ### Public Aliases ###


ColBlColSlArraySolver = _PortSolver
ColBlColSlArrayState = _PortState
ColBlColSlArrayTrace = _PortTrace
