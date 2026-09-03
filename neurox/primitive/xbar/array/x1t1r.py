"""Shape-independent 1T1R array with wire parasitics and a DC solver.

See Also:
    docs/reference/primitive/xbar/array/1t1r.md
    docs/system_design/xbar_solve.md
"""

from typing import final

import torch
from torch import Tensor

from neurox.common import ConfigBase, ModuleBase, PolicyBase, TensorDataClassBase
from neurox.primitive.physics import e_cap_excursion__fJ
from neurox.primitive.xbar.cell import (
    XbarCell1t1r,
    XbarCell1t1rConfig,
    XbarCell1t1rDcop,
    XbarCell1t1rPolicy,
    XbarCell1t1rSnap,
)
from neurox.primitive.xbar.solver import (
    ClampDcop,
    ClampDriver,
    ClampSnap,
    ColBlColSlDcop,
    ColBlColSlSolverConfig,
    execute_chunked,
    solve_col_bl_col_sl_dc,
)


class XbarArray1t1rConfig(ConfigBase):
    row_cell_space__um: float
    col_cell_space__um: float

    bl_segment_r__MOhm: float
    sl_segment_r__MOhm: float

    bl_node_c__fF: float
    """Total capacitance to ground seen at each cell's BL node."""
    x_node_c__fF: float
    """Total capacitance to ground seen at each cell's internal access node X."""
    sl_node_c__fF: float
    """Total capacitance to ground seen at each cell's SL node."""
    wl_node_c__fF: float
    """Total capacitance to ground seen at each cell's WL node."""

    cell_config: XbarCell1t1rConfig
    """Its concrete subclass selects the cell model."""
    solver_config: ColBlColSlSolverConfig

    def validate(self) -> None:
        super().validate()

        # --- Layout pitch ---

        self._require_pos(self.row_cell_space__um, "row_cell_space__um")
        self._require_pos(self.col_cell_space__um, "col_cell_space__um")

        # --- Rail links ---

        self._require_pos(self.bl_segment_r__MOhm, "bl_segment_r__MOhm")
        self._require_pos(self.sl_segment_r__MOhm, "sl_segment_r__MOhm")

        # --- Node capacitance ---

        self._require_non_neg(self.bl_node_c__fF, "bl_node_c__fF")
        self._require_non_neg(self.x_node_c__fF, "x_node_c__fF")
        self._require_non_neg(self.sl_node_c__fF, "sl_node_c__fF")
        self._require_non_neg(self.wl_node_c__fF, "wl_node_c__fF")


class XbarArray1t1rPolicy(PolicyBase):
    cell_policy: XbarCell1t1rPolicy
    """Its concrete subclass matches the configured cell model."""
    solve_chunk_size: int
    """Maximum number of leading positions the solver settles per
    chunk, which is the per-chunk memory budget; `0` runs the whole leading in
    one block. A runtime knob, not a chip-preset constant."""

    def validate(self) -> None:
        super().validate()
        self._require_non_neg(self.solve_chunk_size, "solve_chunk_size")


class XbarArray1t1rSteadyState(TensorDataClassBase):
    """Reassembled steady-state array output, one entry per conducting boundary."""

    i_bl_port__uA: Tensor
    """BL port current at the converged operating point. Shape: `[..., col]`."""
    v_bl_clamp__V: Tensor
    """BL clamp voltage at the converged operating point. Shape: `[..., col]`."""
    i_sl_port__uA: Tensor
    """SL port current at the converged operating point. Shape: `[..., col]`."""
    v_sl_drive__V: Tensor
    """SL drive voltage at the converged operating point. Shape: `[..., col]`."""


class XbarArray1t1rSolveProjection[SteadyStateT: XbarArray1t1rSteadyState](TensorDataClassBase):
    """Array-owned result projected from a solved DC operating point."""

    steady_state: SteadyStateT
    energy__fJ: Tensor | None
    """Array capacitive energy, absent when no profiler asks for dynamic
    energy. Shape: `[...]`."""


class _XbarArray1t1rSolveOperands[CellSnapT: XbarCell1t1rSnap, BLSnapT: ClampSnap, SLSnapT: ClampSnap](
    TensorDataClassBase
):
    """Tensor-carrying inputs sliced together for one array solve."""

    cell_snap: CellSnapT
    bl_driver_snap: BLSnapT
    sl_driver_snap: SLSnapT


class XbarArray1t1r[ConfigT: XbarArray1t1rConfig, PolicyT: XbarArray1t1rPolicy](ModuleBase[ConfigT, PolicyT]):
    """Shape-independent 1T1R array with wire parasitics and a DC solver.

    Args:
        vdd__V: Core analog supply behind every array-node capacitance.
    """

    def __init__(
        self,
        *,
        config: ConfigT,
        policy: PolicyT,
        inst_shape: tuple[int, ...],
        row_num: int,
        col_num: int,
        vdd__V: float,
        dtype: torch.dtype,
        T__K: float,
    ) -> None:
        super().__init__(config=config, policy=policy, inst_shape=inst_shape)
        self._row_num = row_num
        self._col_num = col_num
        self._vdd__V = vdd__V

        self._bl_estab_total_c__fF = row_num * (config.bl_node_c__fF + config.x_node_c__fF)
        self._sl_estab_total_c__fF = row_num * config.sl_node_c__fF

        self._init_children(dtype=dtype, T__K=T__K)

    @property
    def _area_per_inst__um2(self) -> float:
        config = self.config
        return self._row_num * config.row_cell_space__um * self._col_num * config.col_cell_space__um

    @property
    def _leakage_per_inst__uW(self) -> float:
        # Every operation returns to zero cell bias, so the array holds no
        # static conduction path between operations.
        return 0.0

    def _init_children(self, *, dtype: torch.dtype, T__K: float) -> None:
        self.cell = XbarCell1t1r.from_config(
            config=self.config.cell_config,
            policy=self.policy.cell_policy,
            inst_shape=(*self.inst_shape, self._col_num, self._row_num),
            dtype=dtype,
            T__K=T__K,
        )

    @property
    def w_state_num(self) -> int:
        """Number of programmable states exposed by each cell."""
        return self.cell.w_state_num

    def program(self, w_state_idx: Tensor) -> None:
        """Write the cells from one state-index tensor.

        The index grid is PHYSICAL: entry `(col, row)` is the state of the
        cell at that intersection. Placement of digits, polarities, or serial
        slots onto physical columns lies outside this array.

        Args:
            w_state_idx: State-index tensor in `[0, w_state_num - 1]`.
                Shape: `[*inst_shape, col, row]`.
        """
        expected_shape = self.cell.inst_shape
        if tuple(w_state_idx.shape) != expected_shape:
            raise ValueError(f"program() expects w_state_idx.shape {expected_shape}; got {tuple(w_state_idx.shape)}")
        self.cell.program(w_state_idx)

    def solve_array[BLSnapT: ClampSnap, BLDcopT: ClampDcop, SLSnapT: ClampSnap, SLDcopT: ClampDcop](
        self,
        *,
        v_wl__V: Tensor,
        wl_phase_dims: tuple[int, ...],
        bl_driver: ClampDriver[BLSnapT, BLDcopT],
        bl_driver_snap: BLSnapT,
        sl_driver: ClampDriver[SLSnapT, SLDcopT],
        sl_driver_snap: SLSnapT,
    ) -> XbarArray1t1rSteadyState:
        """Settle the 1T1R array to DC under an analog WL drive.

        The WL lines declare the call's complete leading shape. One or more
        leading axes group the WL phases that share a held BL/SL rest
        boundary. The array distributes each row drive across its columns before
        snapshotting the cells; both boundary snaps already carry the same
        leading and repeat their nominal references along the phase axes.

        Args:
            v_wl__V: Analog drive, one value per word line.
                Shape: `[..., row]`.
            wl_phase_dims: Axes of `v_wl__V` whose Cartesian product contains
                the WL phases under one held rest boundary. Every axis must be
                leading rather than the final `row_num` axis.
            bl_driver: BL boundary clamp.
            bl_driver_snap: Per-solve BL clamp snap at the full per-call
                shape; its `v_ref__V` is also the ideal BL rest level.
            sl_driver: SL boundary clamp.
            sl_driver_snap: Per-solve SL clamp snap at the full per-call
                shape; its `v_ref__V` is also the ideal SL rest level.

        Returns:
            Both boundaries' per-column port current and clamp voltage at the
            full leading, the column axis being each clamp's own instance
            axis.
        """
        # --- 1: read the canonical leading from the word-line drive ---

        col_num, row_num = self._col_num, self._row_num
        if not wl_phase_dims:
            raise ValueError("wl_phase_dims must name at least one leading axis")
        wl_phase_dims = tuple(dim + v_wl__V.ndim if dim < 0 else dim for dim in wl_phase_dims)
        if any(dim < 0 or dim >= v_wl__V.ndim - 1 for dim in wl_phase_dims):
            raise ValueError("wl_phase_dims must contain only leading axes, not the final row_num axis")
        if len(set(wl_phase_dims)) != len(wl_phase_dims):
            raise ValueError("wl_phase_dims must not contain duplicate axes")
        wl_phase_dims = tuple(sorted(wl_phase_dims))
        leading_shape = tuple(v_wl__V.shape[:-1])

        # --- 2: distribute the word lines and solve over bounded batches ---

        cell_shape = (*leading_shape, col_num, row_num)
        v_wl_grid__V = v_wl__V.unsqueeze(-2).expand(cell_shape)
        cell_snap = self.cell.snapshot(
            control=v_wl_grid__V,
            shape=cell_shape,
            t_elapsed=0.0,
        )
        operands = _XbarArray1t1rSolveOperands(
            cell_snap=cell_snap,
            bl_driver_snap=bl_driver_snap,
            sl_driver_snap=sl_driver_snap,
        )

        def solve_and_project(
            current: _XbarArray1t1rSolveOperands[XbarCell1t1rSnap, BLSnapT, SLSnapT],
        ) -> XbarArray1t1rSolveProjection[XbarArray1t1rSteadyState]:
            dcop = solve_col_bl_col_sl_dc(
                config=self.config.solver_config,
                bl_segment_r__MOhm=self.config.bl_segment_r__MOhm,
                sl_segment_r__MOhm=self.config.sl_segment_r__MOhm,
                cell=self.cell,
                cell_snap=current.cell_snap,
                bl_driver=bl_driver,
                bl_driver_snap=current.bl_driver_snap,
                sl_driver=sl_driver,
                sl_driver_snap=current.sl_driver_snap,
            )
            return self._project_dcop(
                dcop=dcop,
                cell_snap=current.cell_snap,
                bl_driver_snap=current.bl_driver_snap,
                sl_driver_snap=current.sl_driver_snap,
            )

        projection = execute_chunked(
            chunk_size=self.policy.solve_chunk_size,
            leading_shape=leading_shape,
            operands=operands,
            run=solve_and_project,
        )

        # --- 3: record aggregate energy and expose the steady state ---

        if projection.energy__fJ is not None:
            # Every prefix excluding the phase axes is one BL/SL hold window:
            # all phase excursions plus one establishment of its rest state.
            phase_energy__fJ = projection.energy__fJ.sum(dim=wl_phase_dims)
            v_bl_rest__V = bl_driver_snap.v_ref__V
            v_sl_rest__V = sl_driver_snap.v_ref__V
            for dim in reversed(wl_phase_dims):
                v_bl_rest__V = v_bl_rest__V.select(dim, 0)
                v_sl_rest__V = v_sl_rest__V.select(dim, 0)
            rest_energy__fJ = self._rest_cap_energy__fJ(
                v_bl_rest__V=v_bl_rest__V,
                v_sl_rest__V=v_sl_rest__V,
            )
            self._record_dynamic_energy(phase_energy__fJ + rest_energy__fJ)
        return projection.steady_state

    @final
    @torch.compile(dynamic=False)
    def _project_dcop(
        self,
        *,
        dcop: ColBlColSlDcop[XbarCell1t1rDcop],
        cell_snap: XbarCell1t1rSnap,
        bl_driver_snap: ClampSnap,
        sl_driver_snap: ClampSnap,
    ) -> XbarArray1t1rSolveProjection[XbarArray1t1rSteadyState]:
        return self._project_dcop_impl(
            dcop=dcop,
            cell_snap=cell_snap,
            bl_driver_snap=bl_driver_snap,
            sl_driver_snap=sl_driver_snap,
        )

    def _project_dcop_impl(
        self,
        *,
        dcop: ColBlColSlDcop[XbarCell1t1rDcop],
        cell_snap: XbarCell1t1rSnap,
        bl_driver_snap: ClampSnap,
        sl_driver_snap: ClampSnap,
    ) -> XbarArray1t1rSolveProjection[XbarArray1t1rSteadyState]:
        """Project a converged DC operating point before releasing its grids.

        Args:
            dcop: Converged solver DCOP for the current batch.
            cell_snap: Current batch's per-solve cell snap.
            bl_driver_snap: Current batch's BL clamp snap.
            sl_driver_snap: Current batch's SL clamp snap.

        Returns:
            The steady state and, while a profiler asks for it, array energy.
        """
        energy__fJ = None
        if self._is_dynamic_energy_profile_active():
            energy__fJ = self._cap_energy__fJ(
                solver_dcop=dcop,
                cell_snap=cell_snap,
                bl_driver_snap=bl_driver_snap,
                sl_driver_snap=sl_driver_snap,
            )
        return XbarArray1t1rSolveProjection(
            steady_state=XbarArray1t1rSteadyState(
                i_bl_port__uA=dcop.i_bl_driver__uA,
                v_bl_clamp__V=dcop.v_bl_clamp__V,
                i_sl_port__uA=dcop.i_sl_driver__uA,
                v_sl_drive__V=dcop.v_sl_drive__V,
            ),
            energy__fJ=energy__fJ,
        )

    def _cap_energy__fJ(
        self,
        *,
        solver_dcop: ColBlColSlDcop[XbarCell1t1rDcop],
        cell_snap: XbarCell1t1rSnap,
        bl_driver_snap: ClampSnap,
        sl_driver_snap: ClampSnap,
    ) -> Tensor:
        """Cap energy of one WL phase under a held BL/SL rest boundary.

        The rest state is declared, never solved: both rails at the ideal
        levels their boundaries state, the word lines at ground, and each
        cell's internal access node at its own bit-line level.

        Args:
            solver_dcop: Converged DCOP of the current batch.
            cell_snap: Current batch's cell snap, carrying the WL drive.
            bl_driver_snap: Current batch's BL clamp snap.
            sl_driver_snap: Current batch's SL clamp snap.

        Returns:
            Array energy [fJ].
            Shape: `[...]`.
        """
        config = self.config
        vdd__V = self._vdd__V

        # Shape: [..., col] -> [..., col, 1]
        v_bl_rest__V = bl_driver_snap.v_ref__V.unsqueeze(-1)
        # Shape: [..., col] -> [..., col, 1]
        v_sl_rest__V = sl_driver_snap.v_ref__V.unsqueeze(-1)

        # Shape: [..., col, row]
        cap_e__fJ = (
            e_cap_excursion__fJ(vdd__V, config.bl_node_c__fF, solver_dcop.v_bl_node__V - v_bl_rest__V)
            + e_cap_excursion__fJ(vdd__V, config.x_node_c__fF, solver_dcop.cell.v_x__V - v_bl_rest__V)
            + e_cap_excursion__fJ(vdd__V, config.sl_node_c__fF, solver_dcop.v_sl_node__V - v_sl_rest__V)
            + e_cap_excursion__fJ(vdd__V, config.wl_node_c__fF, cell_snap.v_wl__V)
        )
        # Shape: [..., col, row] -> [...]
        return cap_e__fJ.sum(dim=(-2, -1))

    def _rest_cap_energy__fJ(self, *, v_bl_rest__V: Tensor, v_sl_rest__V: Tensor) -> Tensor:
        """Cap energy of establishing and releasing one held BL/SL rest state."""
        vdd__V = self._vdd__V

        # Shape: [..., col] -> [...]
        bl_estab_e__fJ = e_cap_excursion__fJ(vdd__V, self._bl_estab_total_c__fF, v_bl_rest__V).sum(dim=-1)
        # Shape: [..., col] -> [...]
        sl_estab_e__fJ = e_cap_excursion__fJ(vdd__V, self._sl_estab_total_c__fF, v_sl_rest__V).sum(dim=-1)
        # Shape: [...]
        return bl_estab_e__fJ + sl_estab_e__fJ
