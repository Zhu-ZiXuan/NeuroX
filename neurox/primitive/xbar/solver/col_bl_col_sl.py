"""Residual-driven DC solve for arrays with parallel column BL/SL rails.

Grid axes follow row-before-column order. Ports retain both grid axes with
the row extent set to one.

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

from neurox.common.torch_compat import torch_assert_async
from neurox.execution.solving import SolvingState, SolvingTrace, run_solving

from ._linalg import (
    boundary_inverse_block_tridiagonal_2x2,
    solve_2x2,
    solve_block_tridiagonal_2x2,
)
from ._wire_kcl import dfkcl_dvnode__uS, f_kcl__uA, f_kcl_roundoff__uA
from .clamp_driver import ClampDcop, ClampDriver, ClampSnap
from .resistive_cell import ResistiveCell, ResistiveCellDcop


def _node_jacobian_components__uS(
    *,
    g_cell_bl_eff__uS: Tensor,
    g_cell_sl_eff__uS: Tensor,
    bl_g__uS: float,
    sl_g__uS: float,
    row_dim: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    # Jacobian entries use KCL-equation rows and node-voltage columns.
    return (
        dfkcl_dvnode__uS(g_cell_bl_eff__uS, segment_g__uS=bl_g__uS, dim=row_dim),
        -g_cell_sl_eff__uS,
        -g_cell_bl_eff__uS,
        dfkcl_dvnode__uS(g_cell_sl_eff__uS, segment_g__uS=sl_g__uS, dim=row_dim),
    )


# ### Node solver ###


class _NodeState(SolvingState):
    """Node voltages at the current iterate under fixed port voltages."""

    v_bl_node__V: Tensor
    """Shape: `[..., row, col]`."""
    v_sl_node__V: Tensor
    """Shape: `[..., row, col]`."""


class _NodeTrace(SolvingTrace):
    """Residuals and thresholds are compared before each node update.

    Boolean fields have shape `[..., row=1, col, *history]`; residuals, thresholds,
    and updates have shape `[..., row, col, *history]`.
    Construction rejects a non-boolean `limited` field.
    """

    limited: Tensor
    residual__uA: Tensor
    threshold__uA: Tensor
    dv_bl_node_abs__V: Tensor
    dv_sl_node_abs__V: Tensor

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.limited.dtype != torch.bool:
            raise TypeError("Node trace limited must be a boolean mask")

    @classmethod
    def empty(
        cls,
        node_shape: tuple[int, ...],
        *,
        port_shape: tuple[int, ...],
        history_shape: tuple[int, ...] = (),
        dtype: torch.dtype,
        device: torch.device | None = None,
    ) -> _NodeTrace:
        """Construct unused observations with explicit position and history axes."""

        def flags() -> Tensor:
            return torch.zeros((*port_shape, *history_shape), dtype=torch.bool, device=device)

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
    voltages have shape `[..., row, col]`; ports and activity have shape
    `[..., row=1, col]`. `record_trace=False` requires convergence and returns no
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
        row_dim: int,
        col_dim: int,
    ) -> None:
        if (row_dim, col_dim) not in ((-2, -1), (-1, -2)):
            raise ValueError("row_dim and col_dim must order the final two grid axes")
        self.row_dim = row_dim
        self.col_dim = col_dim
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
        *,
        v_bl_node__V: Tensor,
        v_sl_node__V: Tensor,
        v_bl_port__V: Tensor,
        v_sl_port__V: Tensor,
        cell_snap: CellSnapT,
        final_fn: Callable[[_NodeState], ResultT],
        record_trace: bool,
        trace_mask: Tensor | None,
        is_active: Tensor,
    ) -> tuple[ResultT, _NodeTrace | None]:
        """Return the caller's terminal projection and optional node history."""
        state, trace = self._solve_node(
            v_bl_node__V=v_bl_node__V,
            v_sl_node__V=v_sl_node__V,
            v_bl_port__V=v_bl_port__V,
            v_sl_port__V=v_sl_port__V,
            cell_snap=cell_snap,
            record_trace=record_trace,
            trace_mask=trace_mask,
            is_active=is_active,
        )
        return final_fn(state), trace

    def _solve_node(
        self,
        *,
        v_bl_node__V: Tensor,
        v_sl_node__V: Tensor,
        v_bl_port__V: Tensor,
        v_sl_port__V: Tensor,
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
                v_bl_port__V=v_bl_port__V,
                v_sl_port__V=v_sl_port__V,
                v_bl_node__V=current.v_bl_node__V,
                v_sl_node__V=current.v_sl_node__V,
                is_active=current.is_active,
                cell_snap=cell_snap,
            )

        return run_solving(
            init_state=init_state,
            body_fn=body_fn,
            record_trace=record_trace,
            default_trace_fn=lambda: _NodeTrace.empty(
                tuple(v_bl_node__V.shape),
                port_shape=tuple(v_bl_port__V.shape),
                dtype=v_bl_node__V.dtype,
                device=v_bl_node__V.device,
            ),
            max_iter=self.MAX_ITER,
            trace_mask=trace_mask,
        )

    def _evaluate_node(
        self,
        *,
        v_bl_port__V: Tensor,
        v_sl_port__V: Tensor,
        v_bl_node__V: Tensor,
        v_sl_node__V: Tensor,
        is_active: Tensor,
        cell_snap: CellSnapT,
    ) -> tuple[_NodeState, _NodeTrace]:
        row_dim = self.row_dim

        # --- 1: evaluate cell branches and node KCL ---

        # Shape: [..., row, col]
        cell_dcop = self.cell.solve_dc(v_bl__V=v_bl_node__V, v_sl__V=v_sl_node__V, snap=cell_snap)
        # KCL counts currents leaving each node. The branch leaves BL and enters SL.
        # Shape: [..., row, col]
        f_bl__uA = f_kcl__uA(
            v_node__V=v_bl_node__V,
            v_port__V=v_bl_port__V,
            segment_g__uS=self.bl_g__uS,
            i_inject__uA=cell_dcop.i__uA,
            dim=row_dim,
        )
        f_sl__uA = f_kcl__uA(
            v_node__V=v_sl_node__V,
            v_port__V=v_sl_port__V,
            segment_g__uS=self.sl_g__uS,
            i_inject__uA=-cell_dcop.i__uA,
            dim=row_dim,
        )

        # --- 2: evaluate node stopping thresholds ---

        # Local wire rounding sets a floor even when a cell carries little current.
        # Shape: [..., row, col]
        roundoff__uA = torch.maximum(
            f_kcl_roundoff__uA(
                v_node__V=v_bl_node__V, v_port__V=v_bl_port__V, segment_g__uS=self.bl_g__uS, dim=row_dim
            ),
            f_kcl_roundoff__uA(
                v_node__V=v_sl_node__V, v_port__V=v_sl_port__V, segment_g__uS=self.sl_g__uS, dim=row_dim
            ),
        )
        threshold__uA = self.atol + self.rtol * cell_dcop.i__uA.abs() + roundoff__uA

        # --- 3: form the next coupled node correction ---

        # Adjacent rows couple through same-rail wires; each block keeps the BL/SL
        # equations of one row together. The off-block is a constant diagonal pair.
        # Shape: [..., row, col]
        dv_bl_node__V, dv_sl_node__V = solve_block_tridiagonal_2x2(
            diag=_node_jacobian_components__uS(
                g_cell_bl_eff__uS=cell_dcop.di_dvbl__uS,
                g_cell_sl_eff__uS=-cell_dcop.di_dvsl__uS,
                bl_g__uS=self.bl_g__uS,
                sl_g__uS=self.sl_g__uS,
                row_dim=row_dim,
            ),
            rhs=(-f_bl__uA, -f_sl__uA),
            off_diag=(-self.bl_g__uS, -self.sl_g__uS),
            dim=row_dim,
        )

        # --- 4: reject non-finite updates ---

        finite = (
            v_bl_node__V.isfinite()
            & v_sl_node__V.isfinite()
            & v_bl_port__V.isfinite()
            & v_sl_port__V.isfinite()
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

        # Shape: [..., row, col]
        residual__uA = torch.maximum(f_bl__uA.abs(), f_sl__uA.abs())
        # Shape: [..., row, col] -> [..., row=1, col]
        unresolved = (residual__uA > threshold__uA).any(dim=row_dim, keepdim=True)
        next_is_active = is_active & unresolved

        applied_dv_bl_node__V = torch.where(next_is_active, dv_bl_node__V.clamp(-self.MAX_STEP, self.MAX_STEP), 0)
        applied_dv_sl_node__V = torch.where(next_is_active, dv_sl_node__V.clamp(-self.MAX_STEP, self.MAX_STEP), 0)
        state = _NodeState(
            v_bl_node__V=v_bl_node__V + applied_dv_bl_node__V,
            v_sl_node__V=v_sl_node__V + applied_dv_sl_node__V,
            is_active=next_is_active,
        )
        exceeds_step_limit = (dv_bl_node__V.abs() > self.MAX_STEP) | (dv_sl_node__V.abs() > self.MAX_STEP)
        # Shape: [..., row, col] -> [..., row=1, col]
        limited = next_is_active & exceeds_step_limit.any(dim=row_dim, keepdim=True)
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
    """Shape: `[..., row, col]`."""
    v_sl_node__V: Tensor
    """Shape: `[..., row, col]`."""
    v_bl_port__V: Tensor
    """Shape: `[..., row=1, col]`."""
    v_sl_port__V: Tensor
    """Shape: `[..., row=1, col]`."""


class _PortTrace(SolvingTrace):
    """Port observations with the corresponding node history.

    Port fields have shape `[..., row=1, col, *history]`. Nested node fields
    expand the row extent and append a node-iteration axis before port history.
    `node_trace` is present throughout a traced solve, including unused steps.
    Construction rejects a non-boolean `limited` field.
    """

    limited: Tensor
    residual__V: Tensor
    threshold__V: Tensor
    dv_bl_port_abs__V: Tensor
    dv_sl_port_abs__V: Tensor
    node_trace: _NodeTrace | None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.limited.dtype != torch.bool:
            raise TypeError("Port trace limited must be a boolean mask")

    @classmethod
    def empty(
        cls,
        node_shape: tuple[int, ...],
        *,
        port_shape: tuple[int, ...],
        node_capacity: int,
        dtype: torch.dtype,
        device: torch.device | None = None,
    ) -> _PortTrace:
        """Construct one unused port observation and its nested node history."""

        def flags() -> Tensor:
            return torch.zeros(port_shape, dtype=torch.bool, device=device)

        def values() -> Tensor:
            return torch.full(port_shape, torch.nan, dtype=dtype, device=device)

        return cls(
            limited=flags(),
            residual__V=values(),
            threshold__V=values(),
            dv_bl_port_abs__V=values(),
            dv_sl_port_abs__V=values(),
            node_trace=_NodeTrace.empty(
                node_shape, port_shape=port_shape, history_shape=(node_capacity,), dtype=dtype, device=device
            ),
        )


class _PortSolver[CellSnapT, CellDcopT: ResistiveCellDcop, BLSnapT: ClampSnap, SLSnapT: ClampSnap]:
    """DC solver for parallel column BL/SL rails.

    Cell and driver references are borrowed; their owner handles lifecycle and
    accounting. Calls are stateless and receive per-call snapshots. Node grids
    use `[..., row, col]`, as declared by `row_dim` and `col_dim`.

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

    # Column-last layout improves scan input locality and reduces layout-conversion traffic.
    row_dim: ClassVar[int] = -2
    col_dim: ClassVar[int] = -1

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
            row_dim=self.row_dim,
            col_dim=self.col_dim,
        )

    @torch.no_grad()
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
        `[..., row, col]` and clamps at `[..., row=1, col]`. `record_trace=False`
        requires both loops to converge; `True` permits capped terminal states.
        `trace_mask` selects column observations only. The callback receives
        terminal node and port voltages once, under no-grad, and must support
        compiled tensor execution. It owns the output's structure and meaning.
        """
        row_dim = self.row_dim

        # --- 1: initialize every port and node at its nominal rail reference ---

        row_num = self.cell.inst_shape[row_dim]
        # Materialize reference broadcasts so every loop-carried tensor has the
        # same dense layout as the values returned by its first update.
        # Shape: [..., row=1, col]
        v_bl_port__V = bl_driver_snap.v_open__V.clone(memory_format=torch.contiguous_format)
        v_sl_port__V = sl_driver_snap.v_open__V.clone(memory_format=torch.contiguous_format)
        # Shape: [..., row, col]
        node_shape = list(v_bl_port__V.shape)
        node_shape[row_dim] = row_num
        v_bl_node__V = v_bl_port__V.expand(node_shape).clone(memory_format=torch.contiguous_format)
        v_sl_node__V = v_sl_port__V.expand(node_shape).clone(memory_format=torch.contiguous_format)

        # --- 2: solve the coupled port and node equilibrium ---

        state, trace = self._solve_port(
            v_bl_node__V=v_bl_node__V,
            v_sl_node__V=v_sl_node__V,
            v_bl_port__V=v_bl_port__V,
            v_sl_port__V=v_sl_port__V,
            cell_snap=cell_snap,
            bl_driver_snap=bl_driver_snap,
            sl_driver_snap=sl_driver_snap,
            record_trace=record_trace,
            trace_mask=trace_mask,
        )

        return final_fn(state), trace

    def _solve_port(
        self,
        *,
        v_bl_node__V: Tensor,
        v_sl_node__V: Tensor,
        v_bl_port__V: Tensor,
        v_sl_port__V: Tensor,
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
                v_bl_node__V=current.v_bl_node__V,
                v_sl_node__V=current.v_sl_node__V,
                v_bl_port__V=current.v_bl_port__V,
                v_sl_port__V=current.v_sl_port__V,
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

        return run_solving(
            init_state=init_state,
            body_fn=body_fn,
            record_trace=record_trace,
            default_trace_fn=lambda: _PortTrace.empty(
                tuple(v_bl_node__V.shape),
                port_shape=tuple(v_bl_port__V.shape),
                node_capacity=self.node_solver.MAX_ITER,
                dtype=v_bl_node__V.dtype,
                device=v_bl_node__V.device,
            ),
            max_iter=self.MAX_ITER,
            trace_mask=trace_mask,
        )

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
        row_dim = self.row_dim

        # --- 1: evaluate driver targets and port residuals ---

        v_bl_node__V = node_state.v_bl_node__V
        v_sl_node__V = node_state.v_sl_node__V
        # Shape: [..., row=1, col]
        i_bl_port__uA = (v_bl_port__V - v_bl_node__V.narrow(row_dim, 0, 1)) * self.bl_g__uS
        i_sl_port__uA = (v_sl_port__V - v_sl_node__V.narrow(row_dim, 0, 1)) * self.sl_g__uS
        # Shape: [..., row=1, col]
        bl_driver_dcop = self.bl_driver.solve_dc(i_bl_port__uA, snap=bl_driver_snap, v_port_init__V=v_bl_port__V)
        sl_driver_dcop = self.sl_driver.solve_dc(i_sl_port__uA, snap=sl_driver_snap, v_port_init__V=v_sl_port__V)
        # Shape: [..., row=1, col]
        f_bl_port__V = bl_driver_dcop.v_port__V - v_bl_port__V
        f_sl_port__V = sl_driver_dcop.v_port__V - v_sl_port__V

        # --- 2: compute the port Jacobian and boundary correction ---

        cell_dcop = self.cell.solve_dc(v_bl__V=node_state.v_bl_node__V, v_sl__V=node_state.v_sl_node__V, snap=cell_snap)
        torch_assert_async(
            cell_dcop.di_dvbl__uS.isfinite().all() & cell_dcop.di_dvsl__uS.isfinite().all(),
            "Port linearization produced a non-finite state",
        )
        u_bl_bl, u_bl_sl, u_sl_bl, u_sl_sl = boundary_inverse_block_tridiagonal_2x2(
            diag=_node_jacobian_components__uS(
                g_cell_bl_eff__uS=cell_dcop.di_dvbl__uS,
                g_cell_sl_eff__uS=-cell_dcop.di_dvsl__uS,
                bl_g__uS=self.bl_g__uS,
                sl_g__uS=self.sl_g__uS,
                row_dim=row_dim,
            ),
            off_diag=(-self.bl_g__uS, -self.sl_g__uS),
            dim=row_dim,
        )
        di_bl_dv_bl__uS = self.bl_g__uS * (1.0 - u_bl_bl * self.bl_g__uS)
        di_bl_dv_sl__uS = -self.bl_g__uS * u_bl_sl * self.sl_g__uS
        di_sl_dv_bl__uS = -self.sl_g__uS * u_sl_bl * self.bl_g__uS
        di_sl_dv_sl__uS = self.sl_g__uS * (1.0 - u_sl_sl * self.sl_g__uS)

        # Signed target-voltage response to a boundary-link voltage drop.
        # Its sign sets feedback polarity; its magnitude also scales roundoff below.
        bl_driver_gain = bl_driver_dcop.dvport_di__MOhm * self.bl_g__uS
        sl_driver_gain = sl_driver_dcop.dvport_di__MOhm * self.sl_g__uS
        # Shape: [..., row=1, col]
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

        # Shape: [..., row=1, col]
        v_bl_port_scale__V = torch.maximum(v_bl_port__V.abs(), v_bl_node__V.narrow(row_dim, 0, 1).abs())
        v_sl_port_scale__V = torch.maximum(v_sl_port__V.abs(), v_sl_node__V.narrow(row_dim, 0, 1).abs())
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

        # Shape: [..., row=1, col]
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
