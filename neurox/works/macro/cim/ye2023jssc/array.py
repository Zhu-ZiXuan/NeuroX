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

import torch
from torch import Tensor

from neurox.primitive.xbar.array import (
    XbarArray1t1r,
    XbarArray1t1rChunkMeasure,
    XbarArray1t1rConfig,
    XbarArray1t1rOperationMode,
    XbarArray1t1rPolicy,
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

        if len(self.weight_radix) == 0:
            raise ValueError("require: weight_radix must be non-empty")
        for plane, m in enumerate(self.weight_radix):
            if not isinstance(m, int):
                raise TypeError(f"weight_radix entry at plane {plane} must be an int; got {type(m).__name__}")
            if m <= 0:
                raise ValueError(f"require: every weight_radix entry a positive int; got {m} at plane {plane}")
        for plane, m in enumerate(self.redundant_radix):
            if not isinstance(m, int):
                raise TypeError(f"redundant_radix entry at plane {plane} must be an int; got {type(m).__name__}")
            if m <= 0:
                raise ValueError(f"require: every redundant_radix entry a positive int; got {m} at plane {plane}")

        self._require_pos(self.v_bl_in1__V, "v_bl_in1__V")


class Ye2023Jssc2t1rArrayPolicy(XbarArray1t1rPolicy):
    pass


class Ye2023Jssc2t1rSteadyState(XbarArray1t1rSteadyState):
    """Kernel steady state plus the summed T2 compute current."""

    i_tbl__uA: Tensor
    """Summed T2 compute current per leading instance, already reduced over columns and rows.
    Shape: `[...]`.
    """


class Ye2023Jssc2t1rChunkMeasure(XbarArray1t1rChunkMeasure):
    """Kernel chunk measurement plus the chunk's lookup sum."""

    i_tbl__uA: Tensor
    """Place-value-weighted T2 lookup sum, already reduced over columns and rows.
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
                "— the TBL measurement calls the cell's I_T2 surface"
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

        Identical to the kernel solve — same nodes, same snaps, same chunking —
        with the T2 lookup folded into the same per-chunk measurement, so the
        grid-shaped lookup dies with its chunk and only the row sum survives.

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
        steady = super().solve_array(
            v_wl__V,
            bl_driver=bl_driver,
            bl_driver_snap=bl_driver_snap,
            sl_driver=sl_driver,
            sl_driver_snap=sl_driver_snap,
        )
        assert isinstance(steady, Ye2023Jssc2t1rSteadyState)
        return steady

    def _assemble_steady_state(self, measured: XbarArray1t1rChunkMeasure) -> Ye2023Jssc2t1rSteadyState:
        """Carry the lookup sum out alongside the two boundaries."""
        assert isinstance(measured, Ye2023Jssc2t1rChunkMeasure)
        return Ye2023Jssc2t1rSteadyState(
            i_bl_port__uA=measured.i_bl_port__uA,
            v_bl_clamp__V=measured.v_bl_clamp__V,
            i_sl_port__uA=measured.i_sl_port__uA,
            v_sl_drive__V=measured.v_sl_drive__V,
            i_tbl__uA=measured.i_tbl__uA,
        )

    def _measure_chunk(
        self,
        *,
        dcop: ColBlColSlDcop[XbarCell1t1rDcop],
        cell_snap: XbarCell1t1rSnap,
        bl_driver_snap: ClampSnap,
        sl_driver_snap: ClampSnap,
        **_other_operands: object,
    ) -> Ye2023Jssc2t1rChunkMeasure:
        """Fold the kernel measurement and add this chunk's I_T2 row sum.

        Args:
            dcop: This chunk's converged solver DCOP; its cell working point carries
                the `V_X` the cell reads its operating point off.
            cell_snap: This chunk's slice of the per-solve cell snap, carrying the
                per-cell WL drive.
            bl_driver_snap: This chunk's slice of the BL clamp snap.
            sl_driver_snap: This chunk's slice of the SL clamp snap.
            _other_operands: The remaining sliced snaps and tensors, which this
                array's own measurement does not read.

        Returns:
            The kernel measurement and the chunk's lookup sum.
        """
        base = super()._measure_chunk(
            dcop=dcop,
            cell_snap=cell_snap,
            bl_driver_snap=bl_driver_snap,
            sl_driver_snap=sl_driver_snap,
        )

        # Row selection and operating point are BOTH the cell's own call; the
        # array contributes the place values and the reduction. The grid-shaped
        # intermediates die with the chunk; only the reduction leaves.
        cell = self.cell
        assert isinstance(cell, Ye2023Jssc2t1rCell)
        assert isinstance(cell_snap, Ye2023Jssc2t1rCellSnap)
        # Shape: [chunk, col_num, row_num] -> [chunk]
        i_tbl__uA = (cell.i_t2__uA(dcop.cell, cell_snap) * self._radix_per_col.view(-1, 1)).sum(dim=(-2, -1))

        return Ye2023Jssc2t1rChunkMeasure(
            i_bl_port__uA=base.i_bl_port__uA,
            v_bl_clamp__V=base.v_bl_clamp__V,
            i_sl_port__uA=base.i_sl_port__uA,
            v_sl_drive__V=base.v_sl_drive__V,
            energy__fJ=base.energy__fJ,
            i_tbl__uA=i_tbl__uA,
        )
