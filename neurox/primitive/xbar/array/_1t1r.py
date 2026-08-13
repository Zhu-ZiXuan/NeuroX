"""Shape-independent pure-array core for a 1T1R crossbar tile.

See also:
    docs/reference/primitive/xbar/array/1t1r.md
    docs/internals/primitive/xbar/array/1t1r.md
"""

from dataclasses import dataclass
from enum import StrEnum
from typing import TypeVar

import torch
from torch import Tensor

from neurox.common import ConfigBase, ModuleBase, PolicyBase
from neurox.primitive.physics import e_cap_excursion__fJ
from neurox.primitive.xbar.cell import (
    XbarCell1t1r,
    XbarCell1t1rConfig,
    XbarCell1t1rDcop,
    XbarCell1t1rPolicy,
    XbarCell1t1rSnap,
)
from neurox.primitive.xbar.solver import (
    ChunkedSolver,
    ClampDriver,
    ClampSnap,
    NestedParallelRailSolver,
    NestedParallelRailSolverConfig,
    SolverDcop,
)

BLSnapT = TypeVar("BLSnapT", bound=ClampSnap)
SLSnapT = TypeVar("SLSnapT", bound=ClampSnap)


class XbarArray1t1rOperationMode(StrEnum):
    """Scan organization the array's per-solve capacitive billing follows.

    The mode selects energy evaluation only; solving and programming are
    organization-blind.
    """

    WL_IN_BL_SCAN = "wl_in_bl_scan"
    """Held input on the word lines, scanned bit-line boundary; every node
    rests at ground, so one solve is a full excursion."""
    BL_IN_WL_SCAN = "bl_in_wl_scan"
    """Held input on the bit-line boundary, scanned word lines; the conduction
    path rests at the held level, so one solve is the excursion away from it
    and the hold's establishment spreads over one row scan."""


class XbarArray1t1rConfig(ConfigBase):
    """Shape-independent physical knobs for a 1T1R pure-array core.

    The tile is a uniform lattice of identical cell seats: one pitch per axis,
    one resistance per rail link, and one capacitance per node of each line.
    """

    row_cell_space__um: float
    col_cell_space__um: float

    bl_segment_r__MOhm: float
    sl_segment_r__MOhm: float

    bl_node_c__fF: float
    """Total node-to-ground capacitance at each cell's BL node: the cell
    junction plus that node's share of the bit line."""
    x_node_c__fF: float
    """Total node-to-ground capacitance at each cell's internal access node X."""
    sl_node_c__fF: float
    """Total node-to-ground capacitance at each cell's SL node: the cell
    junction plus that node's share of the source line."""
    wl_node_c__fF: float
    """Total node-to-ground capacitance at each cell's WL node: the
    access-device gate load plus that node's share of the word line."""

    cell_config: XbarCell1t1rConfig
    """Its concrete subclass selects the cell model."""
    solver_config: NestedParallelRailSolverConfig

    def validate(self) -> None:
        super().validate()

        # --- Layout pitch ---

        for field in ("row_cell_space__um", "col_cell_space__um"):
            self._require_pos(getattr(self, field), field)

        # --- Rail links ---

        for field in ("bl_segment_r__MOhm", "sl_segment_r__MOhm"):
            self._require_pos(getattr(self, field), field)

        # --- Node capacitance ---

        for field in ("bl_node_c__fF", "x_node_c__fF", "sl_node_c__fF", "wl_node_c__fF"):
            self._require_non_neg(getattr(self, field), field)


class XbarArray1t1rPolicy(PolicyBase):
    """Composite nonideality policy for a 1T1R pure-array core."""

    cell_policy: XbarCell1t1rPolicy
    """Its concrete subclass matches the configured cell model."""
    solve_chunk_size: int
    """Maximum number of broadcast-leading instances the solver settles per
    chunk, which is the per-chunk memory budget; `0` runs the whole leading in
    one block. A runtime knob, not a chip-preset constant."""

    def validate(self) -> None:
        super().validate()
        self._require_non_neg(self.solve_chunk_size, "solve_chunk_size")


@dataclass(frozen=True)
class XbarArray1t1rSteadyState:
    """Reassembled steady-state array output, one entry per conducting boundary."""

    i_bl_port__uA: Tensor
    """BL port current at the converged operating point. Shape: `[..., col_num]`."""
    v_bl_clamp__V: Tensor
    """BL clamp voltage at the converged operating point. Shape: `[..., col_num]`."""
    i_sl_port__uA: Tensor
    """SL port current at the converged operating point. Shape: `[..., col_num]`."""
    v_sl_drive__V: Tensor
    """SL drive voltage at the converged operating point. Shape: `[..., col_num]`."""


@dataclass(frozen=True)
class XbarArray1t1rChunkMeasure:
    """What survives one solved chunk of this array."""

    i_bl_port__uA: Tensor
    """BL port current at the converged operating point. Shape: `[..., col_num]`."""
    v_bl_clamp__V: Tensor
    """BL clamp voltage at the converged operating point. Shape: `[..., col_num]`."""
    i_sl_port__uA: Tensor
    """SL port current at the converged operating point. Shape: `[..., col_num]`."""
    v_sl_drive__V: Tensor
    """SL drive voltage at the converged operating point. Shape: `[..., col_num]`."""
    energy__fJ: Tensor | None
    """Array capacitive energy, absent when no profiler asks for dynamic
    energy. Shape: `[...]`."""


class XbarArray1t1r(ModuleBase[XbarArray1t1rConfig, XbarArray1t1rPolicy]):
    """Shape-independent 1T1R array with wire parasitics and a DC solver.

    Args:
        config: Concrete configuration dataclass.
        policy: Composite nonideality policy.
        inst_shape: Per-instance replication shape.
        row_num: Number of array rows.
        col_num: Number of array columns.
        operation_mode: Scan organization the capacitive billing follows.
        v_dd_wl__V: Word-line driver rail — the supply behind the WL node
            capacitance.
        v_dd_bl__V: Bit-line driver rail — the supply behind the BL, access
            and SL node capacitance.
        dtype: Tensor dtype for internal buffers.
        T__K: Operating temperature.
    """

    def __init__(
        self,
        *,
        config: XbarArray1t1rConfig,
        policy: XbarArray1t1rPolicy,
        inst_shape: tuple[int, ...],
        row_num: int,
        col_num: int,
        operation_mode: XbarArray1t1rOperationMode,
        v_dd_wl__V: float,
        v_dd_bl__V: float,
        dtype: torch.dtype,
        T__K: float,
    ) -> None:
        super().__init__(config=config, policy=policy, inst_shape=inst_shape)
        self._row_num = row_num
        self._col_num = col_num
        self._operation_mode = operation_mode
        self._v_dd_wl__V = v_dd_wl__V
        self._v_dd_bl__V = v_dd_bl__V

        self._init_children(dtype=dtype, T__K=T__K)

    @property
    def _area_per_inst__um2(self) -> float:
        config = self.config
        return self._row_num * config.row_cell_space__um * self._col_num * config.col_cell_space__um

    @property
    def _leakage_per_inst__uW(self) -> float:
        # Both organizations rest at zero cell bias, so the tile holds no
        # static conduction path; conduction under drive is the DC solve's,
        # billed by the macro.
        return 0.0

    def _sample_fabricate_mismatch(self) -> None:
        """No local static state; the cell fabricates via the traversal."""

    def _init_children(self, *, dtype: torch.dtype, T__K: float) -> None:
        """Construct the cell model and the chunked numerical solver."""
        self.cell = XbarCell1t1r.from_config(
            config=self.config.cell_config,
            policy=self.policy.cell_policy,
            inst_shape=self.weight_grid_shape,
            dtype=dtype,
            T__K=T__K,
        )
        self._solver: ChunkedSolver[XbarArray1t1rChunkMeasure] = ChunkedSolver(
            NestedParallelRailSolver(config=self.config.solver_config),
            chunk_size=self.policy.solve_chunk_size,
        )

    @property
    def w_state_num(self) -> int:
        """Number of programmable states exposed by each cell."""
        return self.cell.w_state_num

    @property
    def weight_grid_shape(self) -> tuple[int, ...]:
        """Shape of the weight grid (per-cell state array): `(*inst_shape, col, row)`."""
        return (*self.inst_shape, self._col_num, self._row_num)

    def program(self, w_state_idx: Tensor) -> None:
        """Write the cells from one state-index tensor.

        The index grid is PHYSICAL: entry `(col, row)` is the state of the
        cell at that intersection. Any placement of digits, polarities, or
        serial slots onto physical columns belongs to the caller.

        Args:
            w_state_idx: State-index tensor in `[0, w_state_num - 1]` at
                `self.weight_grid_shape`.
                Shape: `[*inst_shape, col_num, row_num]`.
        """
        if tuple(w_state_idx.shape) != self.weight_grid_shape:
            raise ValueError(
                f"program() expects w_state_idx.shape {self.weight_grid_shape}; got {tuple(w_state_idx.shape)}"
            )
        self.cell.program(w_state_idx)

    def solve_array(
        self,
        v_wl: Tensor,
        *,
        bl_driver: ClampDriver[BLSnapT],
        bl_driver_snap: BLSnapT,
        sl_driver: ClampDriver[SLSnapT],
        sl_driver_snap: SLSnapT,
    ) -> XbarArray1t1rSteadyState:
        """Settle the 1T1R array to DC under an analog WL drive.

        The call's leading is the weight-grid prefix `(*inst_shape,)`
        broadcast against the WL drive, so a caller's own batch axes stay in
        front of this array's instance axes. No boundary shape is normalized
        here: the snaps arrive at the full per-call shape and the WL drive as
        a full cell grid, a row-uniform drive being an expanded row vector.

        Args:
            v_wl: Analog WL drive [V], one value per cell gate.
                Shape: `[..., col_num, row_num]`.
            bl_driver: BL boundary clamp.
            bl_driver_snap: Per-solve BL clamp snap at the full per-call
                shape; its `v_ref__V` is also the ideal BL rest level.
            sl_driver: SL boundary clamp.
            sl_driver_snap: Per-solve SL clamp snap at the full per-call
                shape; its `v_ref__V` is also the ideal SL rest level.

        Returns:
            Both boundaries' per-column port current [uA] and clamp voltage
            [V] at the full leading, the column axis being each clamp's own
            instance axis.
        """

        # --- 1: read the call's broadcast leading ---

        col_num, row_num = self._col_num, self._row_num
        leading = tuple(torch.broadcast_shapes(self.weight_grid_shape[:-2], v_wl.shape[:-2]))

        # --- 2: snapshot the cell and fold the solve chunk by chunk ---

        # A cell's control is the voltage at its own gate, so the drive
        # travels to the cell exactly as it arrives, on the cell grid.
        cell_snap = self.cell.snapshot(
            control=v_wl,
            shape=(*leading, col_num, row_num),
            t_elapsed=0.0,
        )

        measured = self._solver.solve_dc(
            leading=leading,
            bl_segment_r__MOhm=self.config.bl_segment_r__MOhm,
            sl_segment_r__MOhm=self.config.sl_segment_r__MOhm,
            cell=self.cell,
            cell_snap=cell_snap,
            bl_driver=bl_driver,
            bl_driver_snap=bl_driver_snap,
            sl_driver=sl_driver,
            sl_driver_snap=sl_driver_snap,
            measure=self._measure_chunk,
        )

        # --- 3: record aggregate energy ---

        if measured.energy__fJ is not None:
            # The cell's finer (column, row) axes are already folded by the
            # mode's energy function; the collector sums this array's own work
            # and instance axes past the caller's leading dims.
            # Shape: [*leading]
            self._record_dynamic_energy(measured.energy__fJ)
        return self._assemble_steady_state(measured)

    def _assemble_steady_state(self, measured: XbarArray1t1rChunkMeasure) -> XbarArray1t1rSteadyState:
        """Reassemble the folded measurement into this array's steady state.

        Args:
            measured: Folded measurement at the call's full leading.

        Returns:
            The steady state the caller reads the boundaries off.
        """
        return XbarArray1t1rSteadyState(
            i_bl_port__uA=measured.i_bl_port__uA,
            v_bl_clamp__V=measured.v_bl_clamp__V,
            i_sl_port__uA=measured.i_sl_port__uA,
            v_sl_drive__V=measured.v_sl_drive__V,
        )

    def _measure_chunk(
        self,
        *,
        dcop: SolverDcop[XbarCell1t1rDcop],
        cell_snap: XbarCell1t1rSnap,
        bl_driver_snap: ClampSnap,
        sl_driver_snap: ClampSnap,
        **_other_operands: object,
    ) -> XbarArray1t1rChunkMeasure:
        """Fold one solved chunk down to the state that outlives it.

        Args:
            dcop: This chunk's converged solver DCOP.
            cell_snap: This chunk's slice of the per-solve cell snap.
            bl_driver_snap: This chunk's slice of the BL clamp snap.
            sl_driver_snap: This chunk's slice of the SL clamp snap.
            _other_operands: The remaining sliced snaps and tensors, which
                this array's own measurement does not read.

        Returns:
            Both port states and, while a profiler asks for it, the chunk's
            array energy.
        """
        energy__fJ = None
        if self._is_dynamic_energy_profile_active():
            # Shape: [chunk, col_num, row_num] -> [chunk]
            if self._operation_mode is XbarArray1t1rOperationMode.WL_IN_BL_SCAN:
                energy__fJ = self._energy_wl_in_bl_scan__fJ(solver_dcop=dcop, cell_snap=cell_snap)
            else:
                energy__fJ = self._energy_bl_in_wl_scan__fJ(
                    solver_dcop=dcop,
                    cell_snap=cell_snap,
                    bl_driver_snap=bl_driver_snap,
                    sl_driver_snap=sl_driver_snap,
                )
        return XbarArray1t1rChunkMeasure(
            i_bl_port__uA=dcop.i_bl_driver,
            v_bl_clamp__V=dcop.v_bl_clamp,
            i_sl_port__uA=dcop.i_sl_driver,
            v_sl_drive__V=dcop.v_sl_drive,
            energy__fJ=energy__fJ,
        )

    def _energy_wl_in_bl_scan__fJ(
        self,
        *,
        solver_dcop: SolverDcop[XbarCell1t1rDcop],
        cell_snap: XbarCell1t1rSnap,
    ) -> Tensor:
        """Cap energy of one solve under a scanned bit-line boundary.

        The rest state is ground everywhere, so the bill is one full excursion
        of the four node totals of every cell and no hold to establish.

        Args:
            solver_dcop: Converged DCOP of this chunk.
            cell_snap: This chunk's cell snap, carrying the WL drive.

        Returns:
            Array energy [fJ].
            Shape: `[...]`.
        """
        config = self.config

        # The conduction-path nodes ride the BL driver's rail and the gate
        # rides the word-line driver's.
        # Shape: [..., col_num, row_num] -> [...]
        return (
            e_cap_excursion__fJ(self._v_dd_bl__V, config.bl_node_c__fF, solver_dcop.v_bl_node)
            + e_cap_excursion__fJ(self._v_dd_bl__V, config.x_node_c__fF, solver_dcop.cell.v_x__V)
            + e_cap_excursion__fJ(self._v_dd_bl__V, config.sl_node_c__fF, solver_dcop.v_sl_node)
            + e_cap_excursion__fJ(self._v_dd_wl__V, config.wl_node_c__fF, cell_snap.v_wl__V)
        ).sum(dim=(-2, -1))

    def _energy_bl_in_wl_scan__fJ(
        self,
        *,
        solver_dcop: SolverDcop[XbarCell1t1rDcop],
        cell_snap: XbarCell1t1rSnap,
        bl_driver_snap: ClampSnap,
        sl_driver_snap: ClampSnap,
    ) -> Tensor:
        """Cap energy of one solve under a held bit-line boundary.

        The rest state is declared, never solved: both rails at the ideal
        levels their boundaries state, the word lines at ground, and each
        cell's internal access node at its own bit-line level. The bill is the
        four node totals of every cell against those levels, plus the rest
        state's own establishment spread evenly over the `row_num` solves one
        hold covers.

        Args:
            solver_dcop: Converged DCOP of this chunk.
            cell_snap: This chunk's cell snap, carrying the WL drive.
            bl_driver_snap: This chunk's BL clamp snap.
            sl_driver_snap: This chunk's SL clamp snap.

        Returns:
            Array energy [fJ].
            Shape: `[...]`.
        """
        config = self.config
        v_bl__V = solver_dcop.v_bl_node
        v_sl__V = solver_dcop.v_sl_node
        # Shape: [..., col_num] -> [..., col_num, 1]
        v_bl_rest__V = bl_driver_snap.v_ref__V.unsqueeze(-1)
        # Shape: [..., col_num] -> [..., col_num, 1]
        v_sl_rest__V = sl_driver_snap.v_ref__V.unsqueeze(-1)

        # --- 1: bill every node's displacement away from its rest level ---

        # Shape: [..., col_num, row_num] -> [...]
        e_node__fJ = (
            e_cap_excursion__fJ(self._v_dd_bl__V, config.bl_node_c__fF, v_bl__V - v_bl_rest__V)
            + e_cap_excursion__fJ(self._v_dd_bl__V, config.x_node_c__fF, solver_dcop.cell.v_x__V - v_bl_rest__V)
            + e_cap_excursion__fJ(self._v_dd_bl__V, config.sl_node_c__fF, v_sl__V - v_sl_rest__V)
            + e_cap_excursion__fJ(self._v_dd_wl__V, config.wl_node_c__fF, cell_snap.v_wl__V)
        ).sum(dim=(-2, -1))

        # --- 2: bill establishing the rest state, amortized over its scan ---

        # The other leg of the same excursion, from ground up to the rest
        # profile; the word line rests at ground and carries no term here.
        # Every cell of a column rests at that column's declared levels, so
        # the per-cell bills are laid out on the grid before the fold.
        # Shape: [..., col_num, 1] -> [..., col_num, row_num]
        v_bl_rest_cell__V = v_bl_rest__V.expand_as(v_bl__V)
        # Shape: [..., col_num, 1] -> [..., col_num, row_num]
        v_sl_rest_cell__V = v_sl_rest__V.expand_as(v_sl__V)
        # Shape: [..., col_num, row_num] -> [...]
        e_rest__fJ = (
            e_cap_excursion__fJ(self._v_dd_bl__V, config.bl_node_c__fF, v_bl_rest_cell__V)
            + e_cap_excursion__fJ(self._v_dd_bl__V, config.x_node_c__fF, v_bl_rest_cell__V)
            + e_cap_excursion__fJ(self._v_dd_bl__V, config.sl_node_c__fF, v_sl_rest_cell__V)
        ).sum(dim=(-2, -1))

        return e_node__fJ + e_rest__fJ / self._row_num
