"""Dedicated WH-2T1R array — the kernel 1T1R array plus its T2 path.

Geometry follows the solver convention `[..., col_num, row_num]`: `col_num`
bit-line columns carrying the per-column BL input voltages, `row_num` word-line
rows of which one is driven per solve. The divider solve and the BL/SL wire
ladders are the kernel array's; this extension adds the transpose-bitline (TBL)
current and its node-capacitance excursion.

See Also:
    docs/reference/primitive/xbar/array/1t1r.md
    docs/system_design/xbar_solve.md
"""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor

from neurox.primitive.xbar.array import (
    XbarArray1t1r,
    XbarArray1t1rConfig,
    XbarArray1t1rPolicy,
    XbarArray1t1rScanMode,
    XbarArray1t1rSolveProjection,
    XbarArray1t1rSteadyState,
)
from neurox.primitive.xbar.cell import XbarCell1t1rDcop, XbarCell1t1rSnap
from neurox.primitive.xbar.solver import ClampDcop, ClampDriver, ClampSnap, ColBlColSlDcop

from .cell import (
    Ye2023Jssc2t1rCell,
    Ye2023Jssc2t1rCellConfig,
    Ye2023Jssc2t1rCellPolicy,
    Ye2023Jssc2t1rCellSnap,
)


class Ye2023Jssc2t1rArrayConfig(XbarArray1t1rConfig):
    cell_config: Ye2023Jssc2t1rCellConfig

    tbl_node_c__fF: float
    """Effective total capacitance to ground seen at each TBL-connected cell
    site, including the T2 terminal and TBL interconnect parasitics."""

    weight_radix: tuple[int, ...]
    """Per-plane place values of the weight-bearing planes, LSB-first. Non-empty; every
    entry a positive int."""
    redundant_radix: tuple[int, ...]
    """Per-plane place values of the non-weight planes, laid out after the weight planes.
    Every entry a positive int.

    A redundant plane is physically present but carries no weight: it is programmed
    all-HRS and its columns are held at input 0, so it contributes its radix-weighted
    share of the row leakage floor and nothing else. The paper's redundant-slice
    mapping algorithm itself is not modeled.
    """

    def validate(self) -> None:
        super().validate()

        self._require_non_neg(self.tbl_node_c__fF, "tbl_node_c__fF")
        self._require_non_empty(self.weight_radix, "weight_radix")
        for plane, m in enumerate(self.weight_radix):
            self._require_pos(m, f"weight_radix[{plane}]")
        for plane, m in enumerate(self.redundant_radix):
            self._require_pos(m, f"redundant_radix[{plane}]")


class Ye2023Jssc2t1rArrayPolicy(XbarArray1t1rPolicy):
    cell_policy: Ye2023Jssc2t1rCellPolicy


class Ye2023Jssc2t1rSteadyState(XbarArray1t1rSteadyState):
    """Kernel steady state plus the summed T2 compute current."""

    i_tbl__uA: Tensor
    """Summed T2 compute current per leading instance, already reduced over columns and rows.
    Shape: `[...]`.
    """


class Ye2023Jssc2t1rArray(XbarArray1t1r[Ye2023Jssc2t1rArrayConfig, Ye2023Jssc2t1rArrayPolicy]):
    """Kernel 1T1R array extended by the WH-2T1R transpose-bitline lookup sum.

    The scan mode is fixed: the BL boundary holds the input pattern while
    the word lines are scanned one row per solve. The kernel bills its BL, X,
    SL, and WL nodes under `XbarArray1t1rScanMode.BL_IN_WL_SCAN`; this extension
    bills the selected row's TBL nodes for one ground-to-clamp excursion.

    The TBL sum is the cell's per-cell T2 current weighted by the place value of
    its column: which rows contribute and which calibration operating point each
    cell sits at are the CELL's own reading of its gate drive and its solved `V_X`,
    so this array carries no threshold of its own.

    A place value is ANALOG and physically a device-width ratio: the radix-scaled
    currents share one transpose bit line, so no digital shift-add of planes exists
    anywhere in the scheme, and the multiplier scales the floor and leakage table
    entries exactly as it scales the on-current. T2 in sub-threshold saturation is a
    near-ideal current source, so what reaches the readout is that sum whatever the
    line drops. The TBL therefore needs no DC solve, while its known clamp excursion
    still determines its capacitive energy.

    Args:
        col_num: Number of physical BL columns; divisible by
            `len(weight_radix) + len(redundant_radix)`, since the place values are
            laid out plane-major over equal column groups.
    """

    cell: Ye2023Jssc2t1rCell

    # === Functional buffers ===

    _radix_per_col: Tensor  # Shape: [col_num]

    def __init__(
        self,
        *,
        config: Ye2023Jssc2t1rArrayConfig,
        policy: Ye2023Jssc2t1rArrayPolicy,
        inst_shape: tuple[int, ...],
        row_num: int,
        col_num: int,
        v_tbl__V: float,
        vdd__V: float,
        dtype: torch.dtype,
        T__K: float,
    ) -> None:
        all_radix = (*config.weight_radix, *config.redundant_radix)
        # A remainder would silently truncate the per-column place-value vector.
        if col_num % len(all_radix) != 0:
            raise ValueError(
                f"require: col_num ({col_num}) % (len(weight_radix) + len(redundant_radix)) ({len(all_radix)}) == 0"
            )
        cell_config = config.cell_config
        if not isinstance(cell_config, Ye2023Jssc2t1rCellConfig):
            raise TypeError(
                f"require: cell_config a Ye2023Jssc2t1rCellConfig; got {type(cell_config).__name__} "
                "— the TBL projection calls the cell's I_T2 surface"
            )

        super().__init__(
            config=config,
            policy=policy,
            inst_shape=inst_shape,
            row_num=row_num,
            col_num=col_num,
            scan_mode=XbarArray1t1rScanMode.BL_IN_WL_SCAN,
            vdd__V=vdd__V,
            dtype=dtype,
            T__K=T__K,
        )
        self._v_tbl__V = v_tbl__V

        # Plane-major place values: each radix spans an equal group of columns.
        self._register_nonpersistent_buffer(
            "_radix_per_col",
            torch.tensor(all_radix, dtype=dtype).repeat_interleave(col_num // len(all_radix)),
        )

    def _init_children(self, *, dtype: torch.dtype, T__K: float) -> None:
        self.cell = Ye2023Jssc2t1rCell(
            config=self.config.cell_config,
            policy=self.policy.cell_policy,
            inst_shape=(*self.inst_shape, self._col_num, self._row_num),
            dtype=dtype,
            T__K=T__K,
        )

    def solve_array[BLSnapT: ClampSnap, BLDcopT: ClampDcop, SLSnapT: ClampSnap, SLDcopT: ClampDcop](
        self,
        v_wl__V: Tensor,
        *,
        bl_driver: ClampDriver[BLSnapT, BLDcopT],
        bl_driver_snap: BLSnapT,
        sl_driver: ClampDriver[SLSnapT, SLDcopT],
        sl_driver_snap: SLSnapT,
    ) -> Ye2023Jssc2t1rSteadyState:
        """Settle the WH-2T1R array to DC and compute the T2 lookup sum.

        Identical to the kernel solve — same nodes, same snaps, same bounded
        execution — with the T2 lookup projected before the grid-shaped DCOP
        is released, so only the reduced lookup current survives.

        Args:
            v_wl__V: Analog WL drive, one value per word line.
                Shape: `[..., row_num]`.
            bl_driver: BL boundary clamp in the structural `ClampDriver` role.
            bl_driver_snap: Per-solve BL clamp snap at the full per-call shape; its
                `v_ref__V` is the ideal BL rest level, which is also the per-column
                input level the solve drives the branches from.
            sl_driver: SL boundary clamp in the structural `ClampDriver` role.
            sl_driver_snap: Per-solve SL clamp snap at the full per-call shape.

        Returns:
            The kernel steady state plus the summed T2 current.
        """
        return cast(
            Ye2023Jssc2t1rSteadyState,
            super().solve_array(
                v_wl__V,
                bl_driver=bl_driver,
                bl_driver_snap=bl_driver_snap,
                sl_driver=sl_driver,
                sl_driver_snap=sl_driver_snap,
            ),
        )

    def _energy_bl_in_wl_scan__fJ(
        self,
        *,
        solver_dcop: ColBlColSlDcop[XbarCell1t1rDcop],
        cell_snap: XbarCell1t1rSnap,
        bl_driver_snap: ClampSnap,
        sl_driver_snap: ClampSnap,
    ) -> Tensor:
        energy__fJ = super()._energy_bl_in_wl_scan__fJ(
            solver_dcop=solver_dcop,
            cell_snap=cell_snap,
            bl_driver_snap=bl_driver_snap,
            sl_driver_snap=sl_driver_snap,
        )
        return energy__fJ + self._vdd__V * self.config.tbl_node_c__fF * self._col_num * abs(self._v_tbl__V)

    def _project_dcop(
        self,
        *,
        dcop: ColBlColSlDcop[XbarCell1t1rDcop],
        cell_snap: XbarCell1t1rSnap,
        bl_driver_snap: ClampSnap,
        sl_driver_snap: ClampSnap,
    ) -> XbarArray1t1rSolveProjection[Ye2023Jssc2t1rSteadyState]:
        """Project the kernel result and the T2 lookup sum from one DCOP.

        Args:
            dcop: Converged solver DCOP; its cell working point carries
                the `V_X` the cell reads its operating point off.
            cell_snap: Per-solve cell snap carrying the per-cell WL drive.
            bl_driver_snap: BL clamp snap.
            sl_driver_snap: SL clamp snap.

        Returns:
            The extended steady state and the kernel-plus-TBL array energy.
        """
        base = super()._project_dcop(
            dcop=dcop,
            cell_snap=cell_snap,
            bl_driver_snap=bl_driver_snap,
            sl_driver_snap=sl_driver_snap,
        )
        cell_snap = cast(Ye2023Jssc2t1rCellSnap, cell_snap)

        # Row selection and operating point are BOTH the cell's own call; the
        # array contributes the place values and the reduction. The grid-shaped
        # intermediates die with this DCOP; only the reduction leaves.
        # Shape: [..., col_num, row_num] -> [...]
        i_tbl__uA = (self.cell.i_t2__uA(dcop.cell, cell_snap) * self._radix_per_col.view(-1, 1)).sum(dim=(-2, -1))

        state = base.steady_state
        return XbarArray1t1rSolveProjection(
            steady_state=Ye2023Jssc2t1rSteadyState(
                i_bl_port__uA=state.i_bl_port__uA,
                v_bl_clamp__V=state.v_bl_clamp__V,
                i_sl_port__uA=state.i_sl_port__uA,
                v_sl_drive__V=state.v_sl_drive__V,
                i_tbl__uA=i_tbl__uA,
            ),
            energy__fJ=base.energy__fJ,
        )
