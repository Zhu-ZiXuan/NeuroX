"""DC solver for a crossbar array with PARALLEL BL/SL rails.

Each column's cells share one BL and one SL; columns do not interact. The two
array rails (BL, SL) run side by side along the row axis; the gate/control
line is a driven boundary, so the columns are independent and batched.

Each rail is a UNIFORM ladder — one link resistance joins every pair of
adjacent nodes and the same link joins the clamp driver to the node at index
0 — so a whole rail enters the solve as one scalar.

Probing is per-iteration: an active `ColBlColSlProber` makes the solve emit the
residual pair of every outer clamp event and of every inner Newton step, on top
of the terminal record carrying the converged operating point. The emission is a
compile-time specialization of the solve body — an active prober compiles a
second graph whose submit sites graph-break, correct but slower, while the
unprobed graph stays break-free.

See Also:
    docs/reference/primitive/xbar/solver/col_bl_col_sl.md
    docs/system_design/xbar_solve.md
"""

from __future__ import annotations

from typing import Any, Final, final

import torch
import torch.nn.functional as F
from torch import Tensor

from neurox.common import ConfigBase, DcopBase, RecordBase, RecorderBase
from neurox.primitive.xbar.cell import XbarCell, XbarCellDcop, XbarCellSnap

from ._linalg import block_solve, solve_block_tridiagonal_2x2_uniform
from ._wire_kcl import f_kcl__uA, g_self__uS, i_drive__uA
from .clamp import ClampDcop, ClampDriver, ClampSnap

# The axis the wire ladders run along in `[..., col_num, row_num]`.
_WIRE_DIM: Final = -1


class ColBlColSlDcop[CellDcopT: XbarCellDcop](DcopBase):
    i_bl_driver__uA: Tensor
    """BL driver current. Shape: `[..., num_col]`."""
    i_sl_driver__uA: Tensor
    """SL driver current. Shape: `[..., num_col]`."""
    v_bl_node__V: Tensor
    """BL node voltages. Shape: `[..., num_col, num_row]`."""
    v_sl_node__V: Tensor
    """SL node voltages. Shape: `[..., num_col, num_row]`."""
    cell: CellDcopT
    """Condensed cell DC working point at the converged node voltages,
    including the internal node voltage."""
    v_bl_clamp__V: Tensor
    """BL clamp voltages. Shape: `[..., num_col]`."""
    v_sl_drive__V: Tensor
    """SL drive voltages. Shape: `[..., num_col]`."""


class ColBlColSlRecord[CellDcopT: XbarCellDcop](RecordBase):
    """One event in the nested DC-solve trajectory.

    `inner == 0` identifies an outer clamp event; `inner >= 1` identifies an
    inner Newton step. The sole event at `outer == n_outer` is terminal and
    carries only `dcop`. Every residual belongs to the pre-step iterate at
    which it was evaluated, while the terminal event carries the state the
    final step produced and no residuals.
    """

    outer: int
    """Outer step the record was emitted from; `n_outer` for the terminal one."""
    inner: int
    """`0` for that outer step's clamp event, `j + 1` for its `j`-th inner step."""

    # === Outer clamp event ===

    f_bl_clamp__V: Tensor | None
    """BL outer residual, target minus clamp. Shape: `[..., num_col]`."""
    f_sl_clamp__V: Tensor | None
    """SL outer residual, target minus drive. Shape: `[..., num_col]`."""

    # === Inner Newton step ===

    f_bl_kcl__uA: Tensor | None
    """BL wire KCL residual per node. Shape: `[..., num_col, num_row]`."""
    f_sl_kcl__uA: Tensor | None
    """SL wire KCL residual per node. Shape: `[..., num_col, num_row]`."""

    # === Terminal ===

    dcop: ColBlColSlDcop[CellDcopT] | None
    """Converged operating point on the terminal record alone. Being a
    dataclass, the record's field walk reaches its tensors, including the
    nested cell working point."""


class ColBlColSlProber(RecorderBase[ColBlColSlRecord[XbarCellDcop]]):
    """Collect the iteration trajectory of every solve run inside its context.

    Probing is not free and not silent: it forces the solve to emit at every
    iteration, so it changes what a solve costs while leaving what it returns
    bit-identical.

    Args:
        min_outer: Lowest outer step this recorder keeps; a record from an
            earlier outer step is dropped on arrival, which is how a long
            solve's trailing iterations are collected without holding the
            whole trajectory. `0` keeps everything.
        sync_device: Device a clean exit parks the collected records on; `None`
            leaves each record where it was recorded.
    """

    __min_outer: int

    def __init__(self, *, min_outer: int, sync_device: torch.device | None = None) -> None:
        super().__init__(sync_device=sync_device)
        self.__min_outer = min_outer

    @property
    @final
    def min_outer(self) -> int:
        """Lowest outer step this recorder keeps."""
        return self.__min_outer

    @classmethod
    def _submit_impl(cls, record: ColBlColSlRecord[XbarCellDcop]) -> None:
        """Collect one trajectory record unless its outer step is below `min_outer`.

        The gate sits here rather than at the emit site: the solve emits its
        whole trajectory blind to who is listening, so what a recorder does not
        want is turned away on arrival, before the book grows.

        Args:
            record: Record the solve emitted.
        """
        recorder = cls.current()
        if recorder is None or record.outer < recorder.min_outer:
            return
        cls._submit_record(record)


class ColBlColSlSolverConfig(ConfigBase):
    n_outer: int
    """Outer Newton iterations on the per-column clamp voltage. Each step
    takes one implicit-Jacobian Newton step on V_clamp and then runs `n_inner`
    inner array Newton steps at the updated V_clamp."""
    n_inner: int
    """Inner Newton iterations on the wire / cell coupled state at a frozen
    V_clamp boundary."""

    def validate(self) -> None:
        super().validate()
        self._require_pos(self.n_outer, "n_outer")
        self._require_pos(self.n_inner, "n_inner")


# Per-iteration |dV| damping bounds are fixed properties of the two Newton
# methods rather than configurable inputs.
_MAX_OUTER_STEP__V: float = 0.10
_MAX_INNER_STEP__V: float = 0.05


def solve_col_bl_col_sl_dc[
    CellSnapT: XbarCellSnap,
    CellDcopT: XbarCellDcop,
    BLSnapT: ClampSnap,
    BLDcopT: ClampDcop,
    SLSnapT: ClampSnap,
    SLDcopT: ClampDcop,
](
    *,
    config: ColBlColSlSolverConfig,
    bl_segment_r__MOhm: float,
    sl_segment_r__MOhm: float,
    cell: XbarCell[Any, Any, CellSnapT, CellDcopT],
    cell_snap: CellSnapT,
    bl_driver: ClampDriver[BLSnapT, BLDcopT],
    bl_driver_snap: BLSnapT,
    sl_driver: ClampDriver[SLSnapT, SLDcopT],
    sl_driver_snap: SLSnapT,
) -> ColBlColSlDcop[CellDcopT]:
    """Settle one parallel BL/SL tile by block Gauss-Seidel and Newton steps.

    Args:
        config: Fixed outer and inner iteration counts.
        bl_segment_r__MOhm: BL rail resistance of one lattice link.
        sl_segment_r__MOhm: SL rail resistance of one lattice link.
        cell: Condensed cell branch model.
        cell_snap: Per-solve cell snap bundling the device snaps and the
            per-cell word-line drive at `[..., col, row]`.
        bl_driver: BL clamp driver.
        bl_driver_snap: Per-solve BL driver snap.
        sl_driver: SL clamp driver.
        sl_driver_snap: Per-solve SL driver snap.

    Returns:
        Complete steady-state solution for the current VMM.
    """
    record = ColBlColSlProber.active()
    dcop = _solve_col_bl_col_sl_dc_impl(
        n_outer=config.n_outer,
        n_inner=config.n_inner,
        bl_segment_r__MOhm=bl_segment_r__MOhm,
        sl_segment_r__MOhm=sl_segment_r__MOhm,
        cell=cell,
        cell_snap=cell_snap,
        bl_driver=bl_driver,
        bl_driver_snap=bl_driver_snap,
        sl_driver=sl_driver,
        sl_driver_snap=sl_driver_snap,
        record=record,
    )
    if record:
        ColBlColSlProber.submit(
            ColBlColSlRecord(
                outer=config.n_outer,
                inner=0,
                f_bl_clamp__V=None,
                f_sl_clamp__V=None,
                f_bl_kcl__uA=None,
                f_sl_kcl__uA=None,
                dcop=dcop,
            )
        )
    return dcop


# A Python-typed argument is part of the cache key by value, so each distinct
# segment resistance compiles its own graph: sweeping one exhausts the
# recompile budget and drops the leaf back to eager without raising.
@torch.compile(dynamic=False)
def _solve_col_bl_col_sl_dc_impl[
    CellSnapT: XbarCellSnap,
    CellDcopT: XbarCellDcop,
    BLSnapT: ClampSnap,
    BLDcopT: ClampDcop,
    SLSnapT: ClampSnap,
    SLDcopT: ClampDcop,
](
    *,
    n_outer: int,
    n_inner: int,
    bl_segment_r__MOhm: float,
    sl_segment_r__MOhm: float,
    cell: XbarCell[Any, Any, CellSnapT, CellDcopT],
    cell_snap: CellSnapT,
    bl_driver: ClampDriver[BLSnapT, BLDcopT],
    bl_driver_snap: BLSnapT,
    sl_driver: ClampDriver[SLSnapT, SLDcopT],
    sl_driver_snap: SLSnapT,
    record: bool,
) -> ColBlColSlDcop[CellDcopT]:
    """Run the fixed-shape nested solve, emitting a trajectory on `record`."""
    # --- 1: read the lattice link as a conductance ---

    # One uS is exactly one reciprocal MOhm, so no unit factor enters.
    bl_g__uS = 1.0 / bl_segment_r__MOhm
    sl_g__uS = 1.0 / sl_segment_r__MOhm

    # --- 2: initialize the cell at the reference clamps ---

    v_bl_seed__V = bl_driver_snap.v_ref__V
    v_sl_seed__V = sl_driver_snap.v_ref__V
    # Shape: [..., num_col, num_row]
    i_cell__uA, _g_bl_init__uS, _g_sl_init__uS = cell.solve_branch(
        v_bl_seed__V.unsqueeze(-1), v_sl_seed__V.unsqueeze(-1), cell_snap
    )

    # --- 3: initialize the clamp voltages ---

    # Shape: [..., num_col, num_row] -> [..., num_col]
    i_bl_seed__uA = i_cell__uA.sum(dim=-1)
    # Shape: [..., num_col, num_row] -> [..., num_col]
    i_sl_seed__uA = -i_cell__uA.sum(dim=-1)
    # Shape: [..., num_col]
    v_bl_clamp__V = bl_driver.solve_dc(i_bl_seed__uA, bl_driver_snap, v_clamp_init__V=None).v_clamp__V
    v_sl_drive__V = sl_driver.solve_dc(i_sl_seed__uA, sl_driver_snap, v_clamp_init__V=None).v_clamp__V

    # --- 4: initialize wire nodes with first-order IR drop ---

    # The grid lift of the clamp boundaries is carried through the outer
    # loop and refreshed on every clamp update.
    # Shape: [..., num_col] -> [..., num_col, 1]
    v_bl_clamp_grid__V = v_bl_clamp__V.unsqueeze(-1)
    v_sl_drive_grid__V = v_sl_drive__V.unsqueeze(-1)

    # Shape: [..., num_col, num_row]
    v_bl_node__V, v_sl_node__V = _wire_ir_drop_seed(
        i_cell__uA,
        v_bl_clamp_grid__V,
        v_sl_drive_grid__V,
        bl_segment_r__MOhm,
        sl_segment_r__MOhm,
    )

    # --- 5: solve coupled clamps and wire nodes ---

    max_inner_step__V = _MAX_INNER_STEP__V
    max_outer_step__V = _MAX_OUTER_STEP__V

    for outer_step in range(n_outer):
        # The preceding inner solve ends by updating its nodes, after its last
        # branch evaluation. Refresh at those current nodes so the implicit
        # clamp Jacobian never carries conductances from the pre-update state.
        i_cell__uA, di_dvbl__uS, di_dvsl__uS = cell.solve_branch(v_bl_node__V, v_sl_node__V, cell_snap)

        # Coupled 2×2 Newton step on `(V_BL_clamp, V_SL_drive)`.
        # K = ∂V_node[0]/∂V_clamp captures cross-rail cell coupling.
        g_cell_bl_eff__uS = di_dvbl__uS
        g_cell_sl_eff__uS = -di_dvsl__uS
        # Shape: [..., num_col, 2, 2]
        k_inner_2x2 = _compute_k_inner_coupled_2x2(
            g_cell_bl_eff__uS,
            g_cell_sl_eff__uS,
            bl_g__uS,
            sl_g__uS,
        )

        # Port current through the driver's own link, and the BL / SL
        # clamp-driver targets it implies.
        # Shape: [..., num_col]
        i_bl_port__uA = (v_bl_clamp__V - v_bl_node__V.select(-1, 0)) * bl_g__uS
        i_sl_port__uA = (v_sl_drive__V - v_sl_node__V.select(-1, 0)) * sl_g__uS
        bl_clamp_dcop = bl_driver.solve_dc(
            i_bl_port__uA,
            bl_driver_snap,
            v_clamp_init__V=v_bl_clamp__V,
        )
        sl_clamp_dcop = sl_driver.solve_dc(
            i_sl_port__uA,
            sl_driver_snap,
            v_clamp_init__V=v_sl_drive__V,
        )
        v_bl_target__V = bl_clamp_dcop.v_clamp__V
        bl_dvclamp_di__MOhm = bl_clamp_dcop.dvclamp_di__MOhm
        v_sl_target__V = sl_clamp_dcop.v_clamp__V
        sl_dvclamp_di__MOhm = sl_clamp_dcop.dvclamp_di__MOhm

        # Outer Newton on F_outer(V_clamp) = V_target(V_clamp) - V_clamp:
        # each clamp driver maps the port current through its own link to
        # a target clamp, and the 2×2 ∂F/∂V_clamp couples the driver slope,
        # the port-current sensitivity, and K_inner's V_node[0] response.
        # Shape: [..., num_col]
        f_bl_clamp__V = v_bl_target__V - v_bl_clamp__V
        f_sl_clamp__V = v_sl_target__V - v_sl_drive__V

        if record:
            ColBlColSlProber.submit(
                ColBlColSlRecord(
                    outer=outer_step,
                    inner=0,
                    f_bl_clamp__V=f_bl_clamp__V,
                    f_sl_clamp__V=f_sl_clamp__V,
                    f_bl_kcl__uA=None,
                    f_sl_kcl__uA=None,
                    dcop=None,
                )
            )

        # MOhm times uS is dimensionless, so the driver slope scaled by
        # its rail link conductance carries no unit into the Jacobian.
        dg_bl = bl_dvclamp_di__MOhm * bl_g__uS
        dg_sl = sl_dvclamp_di__MOhm * sl_g__uS
        k00 = k_inner_2x2[..., 0, 0]
        k01 = k_inner_2x2[..., 0, 1]
        k10 = k_inner_2x2[..., 1, 0]
        k11 = k_inner_2x2[..., 1, 1]
        df_row0 = torch.stack(
            [dg_bl * (1.0 - k00) - 1.0, -dg_bl * k01],
            dim=-1,
        )
        df_row1 = torch.stack(
            [-dg_sl * k10, dg_sl * (1.0 - k11) - 1.0],
            dim=-1,
        )
        # Shape: [..., num_col, 2, 2]
        df_outer = torch.stack([df_row0, df_row1], dim=-2)
        # Shape: [..., num_col, 2]
        f_outer__V = torch.stack([f_bl_clamp__V, f_sl_clamp__V], dim=-1)
        # Solve 2×2 system per column: δ = -inv(df_outer) · f_outer.
        delta_2__V = block_solve(df_outer, -f_outer__V.unsqueeze(-1)).squeeze(-1)
        delta_bl__V = delta_2__V[..., 0].clamp(min=-max_outer_step__V, max=max_outer_step__V)
        delta_sl__V = delta_2__V[..., 1].clamp(min=-max_outer_step__V, max=max_outer_step__V)
        v_bl_clamp__V = v_bl_clamp__V + delta_bl__V
        v_sl_drive__V = v_sl_drive__V + delta_sl__V

        # Solve the inner wire state at the updated clamp voltages.
        # Shape: [..., num_col] -> [..., num_col, 1]
        v_bl_clamp_grid__V = v_bl_clamp__V.unsqueeze(-1)
        v_sl_drive_grid__V = v_sl_drive__V.unsqueeze(-1)

        for inner_step in range(n_inner):
            # The first step shares the outer evaluation because updating the
            # clamps does not change the cell's node-voltage inputs.
            if inner_step > 0:
                i_cell__uA, di_dvbl__uS, di_dvsl__uS = cell.solve_branch(
                    v_bl_node__V,
                    v_sl_node__V,
                    cell_snap,
                )
            # Shape: [..., num_col, num_row]
            g_cell_bl_eff__uS = di_dvbl__uS
            g_cell_sl_eff__uS = -di_dvsl__uS
            # Shape: [..., num_col, num_row]
            f_bl_kcl__uA = f_kcl__uA(v_bl_node__V, v_bl_clamp_grid__V, bl_g__uS, i_cell__uA, dim=_WIRE_DIM)
            f_sl_kcl__uA = f_kcl__uA(v_sl_node__V, v_sl_drive_grid__V, sl_g__uS, -i_cell__uA, dim=_WIRE_DIM)

            if record:
                ColBlColSlProber.submit(
                    ColBlColSlRecord(
                        outer=outer_step,
                        inner=inner_step + 1,
                        f_bl_clamp__V=None,
                        f_sl_clamp__V=None,
                        f_bl_kcl__uA=f_bl_kcl__uA,
                        f_sl_kcl__uA=f_sl_kcl__uA,
                        dcop=None,
                    )
                )

            dv_bl_node__V, dv_sl_node__V = _wire_newton_coupled_block2x2(
                f_bl_kcl__uA,
                f_sl_kcl__uA,
                g_cell_bl_eff__uS,
                g_cell_sl_eff__uS,
                bl_g__uS,
                sl_g__uS,
            )
            dv_bl_node__V = dv_bl_node__V.clamp(min=-max_inner_step__V, max=max_inner_step__V)
            dv_sl_node__V = dv_sl_node__V.clamp(min=-max_inner_step__V, max=max_inner_step__V)
            v_bl_node__V = v_bl_node__V + dv_bl_node__V
            v_sl_node__V = v_sl_node__V + dv_sl_node__V

    # --- 6: refresh the cell and boundary currents ---

    # The last inner step moved the nodes after the cell was last
    # evaluated, so the returned working point is re-solved at the final
    # node voltages rather than carried over from inside the loop.
    cell_dcop = cell.solve_dc(v_bl_node__V, v_sl_node__V, cell_snap)

    i_bl_driver__uA = i_drive__uA(v_bl_node__V, v_bl_clamp_grid__V, bl_g__uS, dim=_WIRE_DIM)
    i_sl_driver__uA = i_drive__uA(v_sl_node__V, v_sl_drive_grid__V, sl_g__uS, dim=_WIRE_DIM)

    return ColBlColSlDcop(
        i_bl_driver__uA=i_bl_driver__uA,
        i_sl_driver__uA=i_sl_driver__uA,
        v_bl_node__V=v_bl_node__V,
        v_sl_node__V=v_sl_node__V,
        cell=cell_dcop,
        v_bl_clamp__V=v_bl_clamp__V,
        v_sl_drive__V=v_sl_drive__V,
    )


def _wire_ir_drop_seed(
    i_cell__uA: Tensor,
    v_bl_clamp_grid__V: Tensor,
    v_sl_drive_grid__V: Tensor,
    bl_segment_r__MOhm: float,
    sl_segment_r__MOhm: float,
) -> tuple[Tensor, Tensor]:
    """First-order IR-drop seed for the wire ladders.

    Every link carries the same resistance, so the drop accumulated down
    a ladder is that one resistance times the running sum of the currents
    its links carry. The two rails share that running sum: BL drains the
    cell current and SL injects the very same current back.

    Args:
        i_cell__uA: Signed cell branch currents.
            Shape: `[..., num_col, num_row]`.
        v_bl_clamp_grid__V: BL clamp voltages on the cell grid.
            Shape: `[..., num_col, 1]`.
        v_sl_drive_grid__V: SL drive voltages on the cell grid.
            Shape: `[..., num_col, 1]`.
        bl_segment_r__MOhm: BL rail resistance of one lattice link.
        sl_segment_r__MOhm: SL rail resistance of one lattice link.

    Returns:
        Initial BL and SL node voltages `(v_bl_node__V, v_sl_node__V)`.
        Shape: `[..., num_col, num_row]`.
    """
    # Current in the link that feeds node k: everything drawn at k and beyond.
    # Shape: [..., num_col, num_row]
    i_link__uA = torch.flip(torch.cumsum(torch.flip(i_cell__uA, [-1]), -1), [-1])
    # Shape: [..., num_col, num_row]
    i_cumulative__uA = torch.cumsum(i_link__uA, dim=-1)
    return (
        v_bl_clamp_grid__V - bl_segment_r__MOhm * i_cumulative__uA,
        v_sl_drive_grid__V + sl_segment_r__MOhm * i_cumulative__uA,
    )


def _g_diag_blocks__uS(
    g_cell_bl_eff__uS: Tensor,
    g_cell_sl_eff__uS: Tensor,
    bl_segment_g__uS: float,
    sl_segment_g__uS: float,
) -> Tensor:
    """Per-row 2×2 diagonal blocks of the coupled BL/SL wire Jacobian.

    The rail entries are each ladder's per-node self-conductance; the cross
    entries are the cell branch alone, which is what couples the two rails at
    a node. The off-diagonal blocks of the same Jacobian are not built here —
    they are the constant `diag(-g_BL, -g_SL)` the block solver takes as a
    pair of scalars.

    Args:
        g_cell_bl_eff__uS: BL-side cell derivatives.
            Shape: `[..., num_col, num_row]`.
        g_cell_sl_eff__uS: Negated SL-side cell derivatives.
            Shape: `[..., num_col, num_row]`.
        bl_segment_g__uS: BL rail conductance of one lattice link.
        sl_segment_g__uS: SL rail conductance of one lattice link.

    Returns:
        Diagonal blocks, rail-major within each block.
        Shape: `[..., num_col, num_row, 2, 2]`.
    """
    # Shape: [..., num_col, num_row]
    bl_diag_node__uS = g_self__uS(g_cell_bl_eff__uS, bl_segment_g__uS, dim=_WIRE_DIM)
    sl_diag_node__uS = g_self__uS(g_cell_sl_eff__uS, sl_segment_g__uS, dim=_WIRE_DIM)
    # ∂F_BL/∂V_SL on the top row, ∂F_SL/∂V_BL on the bottom.
    # Shape: [..., num_col, num_row, 2, 2]
    return torch.stack(
        (
            torch.stack((bl_diag_node__uS, -g_cell_sl_eff__uS), dim=-1),
            torch.stack((-g_cell_bl_eff__uS, sl_diag_node__uS), dim=-1),
        ),
        dim=-2,
    )


def _wire_newton_coupled_block2x2(
    f_bl_kcl__uA: Tensor,
    f_sl_kcl__uA: Tensor,
    g_cell_bl_eff__uS: Tensor,
    g_cell_sl_eff__uS: Tensor,
    bl_segment_g__uS: float,
    sl_segment_g__uS: float,
) -> tuple[Tensor, Tensor]:
    """Coupled BL/SL wire Newton step at frozen V_clamp / V_SL_drive.

    Args:
        f_bl_kcl__uA: BL KCL residuals.
            Shape: `[..., num_col, num_row]`.
        f_sl_kcl__uA: SL KCL residuals.
            Shape: `[..., num_col, num_row]`.
        g_cell_bl_eff__uS: BL-side cell derivatives.
            Shape: `[..., num_col, num_row]`.
        g_cell_sl_eff__uS: Negated SL-side cell derivatives.
            Shape: `[..., num_col, num_row]`.
        bl_segment_g__uS: BL rail conductance of one lattice link.
        sl_segment_g__uS: SL rail conductance of one lattice link.

    Returns:
        BL and SL Newton voltage steps `(dv_bl_node__V, dv_sl_node__V)`.
        Shape: `[..., num_col, num_row]`.
    """
    # Shape: [..., num_col, num_row, 2, 2]
    diag_blocks__uS = _g_diag_blocks__uS(g_cell_bl_eff__uS, g_cell_sl_eff__uS, bl_segment_g__uS, sl_segment_g__uS)
    # Shape: [..., num_col, num_row, 2]
    rhs__uA = torch.stack((-f_bl_kcl__uA, -f_sl_kcl__uA), dim=-1)

    # The block solver claims the row axis as its N axis and the per-node
    # rail pair as its 2x2 block, so the column dim sits in its leading
    # batch. Neighbouring rows couple through their shared rail link
    # alone, which is the constant off-block it takes as two scalars.
    # Shape: [..., num_col, num_row, 2]
    delta__V = solve_block_tridiagonal_2x2_uniform(
        diag_blocks__uS,
        rhs__uA,
        off_block=(-bl_segment_g__uS, -sl_segment_g__uS),
    )
    return delta__V[..., 0], delta__V[..., 1]


def _compute_k_inner_coupled_2x2(
    g_cell_bl_eff__uS: Tensor,
    g_cell_sl_eff__uS: Tensor,
    bl_segment_g__uS: float,
    sl_segment_g__uS: float,
) -> Tensor:
    """Compute the 2x2 `K_inner = ∂V_node[0]/∂V_clamp` per column.

    A clamp reaches the array through its own rail link alone, so its
    forcing is that link's conductance at row 0 and nothing anywhere else.
    The frozen array being linear in that forcing, each solve runs on the
    bare row-0 unit vector and the link conductance scales the extracted
    row-0 response.

    Args:
        g_cell_bl_eff__uS: BL-side cell derivatives.
            Shape: `[..., num_col, num_row]`.
        g_cell_sl_eff__uS: Negated SL-side cell derivatives.
            Shape: `[..., num_col, num_row]`.
        bl_segment_g__uS: BL rail conductance of one lattice link.
        sl_segment_g__uS: SL rail conductance of one lattice link.

    Returns:
        Dimensionless clamp-to-port-node sensitivity.
        Shape: `[..., num_col, 2, 2]`.
    """
    # Shape: [..., num_col, num_row, 2, 2]
    diag_blocks__uS = _g_diag_blocks__uS(g_cell_bl_eff__uS, g_cell_sl_eff__uS, bl_segment_g__uS, sl_segment_g__uS)
    off_block = (-bl_segment_g__uS, -sl_segment_g__uS)

    # Unit forcing at row 0, one rail at a time.
    # Shape: [..., num_col, num_row]
    zeros_node = torch.zeros_like(g_cell_bl_eff__uS)
    unit_row0 = F.pad(torch.ones_like(g_cell_bl_eff__uS[..., :1]), (0, g_cell_bl_eff__uS.shape[-1] - 1))

    # Shape: [..., num_col, num_row, 2]
    u_bl = solve_block_tridiagonal_2x2_uniform(
        diag_blocks__uS,
        torch.stack((unit_row0, zeros_node), dim=-1),
        off_block=off_block,
    )
    u_sl = solve_block_tridiagonal_2x2_uniform(
        diag_blocks__uS,
        torch.stack((zeros_node, unit_row0), dim=-1),
        off_block=off_block,
    )

    # K[:, 0] is the BL-forced row-0 response, K[:, 1] the SL-forced one,
    # each carrying the conductance of the link that forced it.
    # Shape: [..., num_col, 2]
    k_col_bl = u_bl.select(-2, 0) * bl_segment_g__uS
    k_col_sl = u_sl.select(-2, 0) * sl_segment_g__uS
    return torch.stack((k_col_bl, k_col_sl), dim=-1)
