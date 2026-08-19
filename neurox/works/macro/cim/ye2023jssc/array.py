"""Dedicated WH-2T1R array — the kernel 1T1R array plus the I_T2 lookup sum.

Geometry follows the solver convention `[..., col_num, row_num]`: `col_num`
bit-line columns carrying the per-column BL input voltages, `row_num` word-line
rows of which one is driven per solve. Everything below the lookup — the divider
solve, the wire ladders, the capacitive billing — is the kernel array's; this
extension adds the transpose-bitline (TBL) current alone.

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
    XbarArray1t1rOperationMode,
    XbarArray1t1rPolicy,
    XbarArray1t1rSolveProjection,
    XbarArray1t1rSteadyState,
)
from neurox.primitive.xbar.cell import XbarCell1t1rDcop, XbarCell1t1rSnap
from neurox.primitive.xbar.solver import ClampDcop, ClampDriver, ClampSnap, ColBlColSlDcop

# Import triggers the cell's registry registration so `from_config` dispatches.
from .cell import Ye2023Jssc2t1rCell, Ye2023Jssc2t1rCellConfig, Ye2023Jssc2t1rCellSnap


class Ye2023Jssc2t1rArrayConfig(XbarArray1t1rConfig):
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
    v_bl_in1__V: float
    """BL voltage driven for input bit 1; input bit 0 drives 0 V."""

    def validate(self) -> None:
        super().validate()

        self._require_non_empty(self.weight_radix, "weight_radix")
        for plane, m in enumerate(self.weight_radix):
            self._require_pos(m, f"weight_radix[{plane}]")
        for plane, m in enumerate(self.redundant_radix):
            self._require_pos(m, f"redundant_radix[{plane}]")

        self._require_pos(self.v_bl_in1__V, "v_bl_in1__V")


class Ye2023Jssc2t1rArrayPolicy(XbarArray1t1rPolicy):
    pass


class Ye2023Jssc2t1rSteadyState(XbarArray1t1rSteadyState):
    """Kernel steady state plus the summed T2 compute current."""

    i_tbl__uA: Tensor
    """Summed T2 compute current per leading instance, already reduced over columns and rows.
    Shape: `[...]`.
    """


class Ye2023Jssc2t1rArray(XbarArray1t1r):
    """Kernel 1T1R array extended by the WH-2T1R transpose-bitline lookup sum.

    The scan organization is fixed: the BL boundary holds the input pattern while
    the word lines are scanned one row per solve, so the array is always a
    `XbarArray1t1rOperationMode.BL_IN_WL_SCAN` one and its capacitive billing is
    the kernel's for that mode.

    The TBL sum is the cell's per-cell T2 current weighted by the place value of
    its column: which rows contribute and which calibration operating point each
    cell sits at are the CELL's own reading of its gate drive and its solved `V_X`,
    so this array carries no threshold of its own.

    A place value is ANALOG and physically a device-width ratio: the radix-scaled
    currents share one transpose bit line, so no digital shift-add of planes exists
    anywhere in the scheme, and the multiplier scales the floor and leakage table
    entries exactly as it scales the on-current. T2 in sub-threshold saturation is a
    near-ideal current source, so what reaches the readout is that sum whatever the
    line drops: the transpose bit line gets no solve, and no capacitance either — it
    is co-driven with its word line and then held at the readout clamp's DC level.

    Args:
        col_num: Number of physical BL columns; divisible by
            `len(weight_radix) + len(redundant_radix)`, since the place values are
            laid out plane-major over equal column groups.
    """

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
        v_dd_wl__V: float,
        v_dd_bl__V: float,
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
            operation_mode=XbarArray1t1rOperationMode.BL_IN_WL_SCAN,
            v_dd_wl__V=v_dd_wl__V,
            v_dd_bl__V=v_dd_bl__V,
            dtype=dtype,
            T__K=T__K,
        )

        # Plane-major place values: each radix spans an equal group of columns.
        self.register_buffer(
            "_radix_per_col",
            torch.tensor(all_radix, dtype=dtype).repeat_interleave(col_num // len(all_radix)),
            persistent=False,
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
            v_wl__V: Analog WL drive, one value per cell gate.
                Shape: `[..., col_num, row_num]`.
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
            The extended steady state and the kernel array energy.
        """
        base = super()._project_dcop(
            dcop=dcop,
            cell_snap=cell_snap,
            bl_driver_snap=bl_driver_snap,
            sl_driver_snap=sl_driver_snap,
        )

        # Row selection and operating point are BOTH the cell's own call; the
        # array contributes the place values and the reduction. The grid-shaped
        # intermediates die with this DCOP; only the reduction leaves.
        cell = cast(Ye2023Jssc2t1rCell, self.cell)
        cell_snap = cast(Ye2023Jssc2t1rCellSnap, cell_snap)
        # Shape: [..., col_num, row_num] -> [...]
        i_tbl__uA = (cell.i_t2__uA(dcop.cell, cell_snap) * self._radix_per_col.view(-1, 1)).sum(dim=(-2, -1))

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
