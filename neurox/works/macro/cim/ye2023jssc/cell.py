"""WH-2T1R lookup cell — the linear 1T1R divider plus a per-state I_T2 table.

See Also:
    docs/reference/primitive/xbar/cell/1t1r_linear.md
"""

from __future__ import annotations

import torch
from torch import Tensor

from neurox.primitive.xbar.cell import (
    XbarCell1t1rDcop,
    XbarCell1t1rLinear,
    XbarCell1t1rLinearConfig,
    XbarCell1t1rLinearPolicy,
    XbarCell1t1rLinearSnap,
)


class Ye2023Jssc2t1rCellConfig(XbarCell1t1rLinearConfig):
    i_t2_table__uA: tuple[tuple[float, ...], ...]
    """Unit-scale (m = 1) T2 compute current indexed `[operating_point][weight_state]`.

    Row 0 is the off-cell floor sampled at `V_X = 0`, row 1 the state-dependent drive
    currents sampled at `V_X > 0`; the state axis has length `w_state_num` (state 0 =
    HRS, 1 = LRS). Every entry is finite and >= 0.
    """

    def validate(self) -> None:
        super().validate()

        self._require_len(self.i_t2_table__uA, "i_t2_table__uA", 2)
        for point, row in enumerate(self.i_t2_table__uA):
            self._require_same_len(
                row,
                f"i_t2_table__uA[{point}]",
                self.g_cell_off_table__uS,
                "g_cell_off_table__uS",
            )
            for state_idx, entry in enumerate(row):
                self._require_non_neg(entry, f"i_t2_table__uA[{point}][{state_idx}]")


class Ye2023Jssc2t1rCellPolicy(XbarCell1t1rLinearPolicy):
    pass


class Ye2023Jssc2t1rCellSnap(XbarCell1t1rLinearSnap):
    i_t2_floor__uA: Tensor
    """Unit-scale (m = 1) T2 current at the floor operating point `V_X = 0`, pre-selected
    for the programmed state.
    Shape: `[..., col, row]`.
    """
    i_t2_drive__uA: Tensor
    """Unit-scale (m = 1) T2 current at the drive operating point `V_X > 0`, pre-selected
    for the programmed state.
    Shape: `[..., col, row]`.
    """


class Ye2023Jssc2t1rCell(
    XbarCell1t1rLinear[
        Ye2023Jssc2t1rCellConfig,
        Ye2023Jssc2t1rCellPolicy,
    ]
):
    """WH-2T1R lookup cell: linear 1T1R divider plus a per-state I_T2 table.

    The T2 compute current is selected at `program` time and read back by
    `i_t2__uA` at unit slice scale.

    The cell has two transistors on separate paths. T1, an I/O device held half-on
    as a resistor, sits in series with the RRAM, and the divider they form puts the
    programmed state on the internal node `V_X` — that path is the inherited linear
    1T1R branch. T2, a core device in sub-threshold saturation, converts `V_X` into
    the compute current a transpose bit line collects, and only that conversion is
    tabulated. Two operating points suffice because the composing design drives the
    bit line at exactly two levels, one per input bit.

    Args:
        inst_shape: Per-instance cell-grid shape `(..., col, row)`.
    """

    # === Functional buffers ===

    _i_t2_table__uA: Tensor  # Shape: [2, w_state_num]

    # === Programmed state ===

    _i_t2_floor__uA: Tensor  # Shape: [*inst_shape]
    _i_t2_drive__uA: Tensor  # Shape: [*inst_shape]

    def __init__(
        self,
        *,
        config: Ye2023Jssc2t1rCellConfig,
        policy: Ye2023Jssc2t1rCellPolicy,
        inst_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
    ) -> None:
        super().__init__(config=config, policy=policy, inst_shape=inst_shape, dtype=dtype, T__K=T__K)

        self._register_nonpersistent_buffer(
            "_i_t2_table__uA",
            torch.tensor(config.i_t2_table__uA, dtype=dtype),
        )

    def program(self, w_state_idx: Tensor) -> None:
        """Program the divider and pre-select the per-state I_T2.

        Args:
            w_state_idx: State-index tensor in `[0, w_state_num - 1]`.
                Shape: `[*inst_shape]`.
        """
        super().program(w_state_idx)
        idx = w_state_idx.long()
        self._i_t2_floor__uA = self._i_t2_table__uA[0][idx]
        self._i_t2_drive__uA = self._i_t2_table__uA[1][idx]

    def snapshot(
        self,
        *,
        control: Tensor,
        shape: tuple[int, ...],
        t_elapsed: float,
    ) -> Ye2023Jssc2t1rCellSnap:
        """Bundle the divider snap with the pre-selected per-state I_T2.

        Args:
            control: Per-cell word-line drive voltage [V].
                Shape: `[..., col, row]`.
            shape: Per-call broadcast shape `(..., col, row)` the per-cell fields fill.
            t_elapsed: Time elapsed since programming [s]; unused — the lookup model
                holds no time-dependent read state.

        Returns:
            Per-call WH-2T1R cell snap.
        """
        del t_elapsed

        def view(buf: Tensor) -> Tensor:
            return buf.expand(shape) if shape else buf

        return Ye2023Jssc2t1rCellSnap(
            v_wl__V=control,
            g_cell_on__uS=view(self._g_cell_on__uS),
            g_cell_off__uS=view(self._g_cell_off__uS),
            vx_ratio_on=view(self._vx_ratio_on),
            vx_ratio_off=view(self._vx_ratio_off),
            i_t2_floor__uA=view(self._i_t2_floor__uA),
            i_t2_drive__uA=view(self._i_t2_drive__uA),
        )

    def i_t2__uA(self, dcop: XbarCell1t1rDcop, snap: Ye2023Jssc2t1rCellSnap) -> Tensor:
        """Unit-scale (m = 1) per-cell T2 compute current.

        Two independent branches decide the current. A cell contributes at all only
        while its own row pair is driven: the scheme drives a selected row's word line
        and transpose bit line TOGETHER and grounds both lines of every other row, so an
        unselected cell's T2 drain is undriven and its TBL contribution is exactly zero.
        A driven cell then sits at whichever calibration operating point its solved
        `V_X` places it on, floor or drive.

        Args:
            dcop: Converged branch working point of this cell, whose `v_x__V` selects
                the calibration point.
                Shape: `[..., col, row]`.
            snap: Per-call snap of the programmed state.

        Returns:
            Unit-scale T2 current.
            Shape: `[..., col, row]`.
        """
        on = snap.v_wl__V > self.config.v_wl_on_threshold__V
        return torch.where(on, torch.where(dcop.v_x__V > 0.0, snap.i_t2_drive__uA, snap.i_t2_floor__uA), 0.0)
