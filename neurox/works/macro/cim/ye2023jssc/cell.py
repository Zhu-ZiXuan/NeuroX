"""WH-2T1R lookup cell — the linear 1T1R divider plus a per-state I_T2 table.

See also:
    docs/works/macro/cim/ye2023jssc/model.md
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from neurox.primitive.xbar.cell import (
    XbarCell1t1r,
    XbarCell1t1rDcop,
    XbarCell1t1rLinear,
    XbarCell1t1rLinearConfig,
    XbarCell1t1rLinearPolicy,
    XbarCell1t1rLinearSnap,
)


class Ye2023Jssc2t1rCellConfig(XbarCell1t1rLinearConfig):
    """Physical knobs for the WH-2T1R lookup cell.

    Attributes:
        i_t2_table__uA: Unit-scale (m = 1) T2 compute current [uA] indexed
            ``[operating_point][weight_state]``: the T2 current is
            ``I_T2 = f(V_X)`` sampled at two calibration operating points —
            index 0 the floor at ``V_X = 0`` and index 1 the drive point at
            ``V_X > 0`` — over a state axis of length ``w_state_num``
            (state 0 = HRS, 1 = LRS). Row ``[1]`` holds the state-dependent
            drive currents and row ``[0]`` the off-cell floor. All entries
            finite and >= 0.
    """

    i_t2_table__uA: tuple[tuple[float, ...], ...]

    def validate(self) -> None:
        super().validate()

        w_state_num = len(self.g_cell_off_table__uS)
        if len(self.i_t2_table__uA) != 2:
            raise ValueError(
                f"require: len(i_t2_table__uA) ({len(self.i_t2_table__uA)}) == 2 (floor / drive operating point)"
            )
        for point, row in enumerate(self.i_t2_table__uA):
            if len(row) != w_state_num:
                raise ValueError(f"require: len(i_t2_table__uA[{point}]) ({len(row)}) == w_state_num ({w_state_num})")
            for state_idx, entry in enumerate(row):
                if not (math.isfinite(entry) and entry >= 0):
                    raise ValueError(
                        f"require: every i_t2_table__uA entry finite and >= 0; "
                        f"got {entry} at (operating point {point}, state {state_idx})"
                    )


class Ye2023Jssc2t1rCellPolicy(XbarCell1t1rLinearPolicy):
    """Empty nonideality policy for the deterministic WH-2T1R lookup cell."""


@dataclass(frozen=True, kw_only=True)
class Ye2023Jssc2t1rCellSnap(XbarCell1t1rLinearSnap):
    """Per-call snap of a WH-2T1R cell's programmed state.

    Attributes:
        i_t2_floor__uA: Unit-scale (m = 1) T2 current [uA] at the floor
            operating point (``V_X = 0``), pre-selected for the programmed
            state.
            Shape: ``[..., col, row]``.
        i_t2_drive__uA: Unit-scale (m = 1) T2 current [uA] at the drive
            operating point (``V_X > 0``), pre-selected for the programmed
            state.
            Shape: ``[..., col, row]``.
    """

    i_t2_floor__uA: Tensor
    i_t2_drive__uA: Tensor


@XbarCell1t1r.register_neurox_module(
    config_type=Ye2023Jssc2t1rCellConfig,
    policy_type=Ye2023Jssc2t1rCellPolicy,
)
class Ye2023Jssc2t1rCell(XbarCell1t1rLinear):
    """WH-2T1R lookup cell: linear 1T1R divider plus a per-state I_T2 table.

    The T2 compute current is selected at :meth:`program` time and read back by
    :meth:`i_t2__uA` at unit slice scale.

    Args:
        config: WH-2T1R cell configuration.
        policy: WH-2T1R cell policy.
        inst_shape: Per-instance shape ``(..., col, row)``.
        dtype: Tensor dtype for internal buffers.
        T__K: Operating temperature.
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

        self.register_buffer(
            "_i_t2_table__uA",
            torch.tensor(config.i_t2_table__uA, dtype=dtype),
            persistent=False,
        )

    def program(self, w_state_idx: Tensor) -> None:
        """Program the divider and pre-select the per-state I_T2.

        Args:
            w_state_idx: State-index tensor in ``[0, w_state_num - 1]``.
                Shape: ``[*inst_shape]``.
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
                Shape: ``[..., col, row]``.
            shape: Per-call broadcast shape ``(..., col, row)`` the
                per-cell fields fill.
            t_elapsed: Time elapsed since programming [s]; unused — the
                lookup model holds no time-dependent read state.

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
        """Unit-scale (m = 1) per-cell T2 compute current [uA].

        Two independent branches decide the current:

        (a) Is this cell's row pair driven? The scheme drives a selected row's
            word line and transpose bit line TOGETHER and holds both lines of
            every other row at ground, so an unselected cell's T2 drain is
            undriven and its TBL contribution is exactly zero. The observable
            is the cell's own WL terminal against its calibration threshold.
        (b) Which calibration operating point the steady state sits at. The T2
            current is ``I_T2 = f(V_X)`` sampled at two points: ``V_X = 0`` is
            the floor point and ``V_X > 0`` the drive point. The ``0`` is the
            table's own definition anchor, not a tunable.

        Args:
            dcop: Converged branch working point of this cell, whose ``v_x__V``
                places the steady state on one of the two calibration points.
                Shape: ``[..., col, row]``.
            snap: Per-call snap from :meth:`snapshot`.

        Returns:
            Unit-scale T2 current [uA].
            Shape: ``[..., col, row]``.
        """
        on = snap.v_wl__V > self._v_wl_on_threshold__V
        return torch.where(on, torch.where(dcop.v_x__V > 0.0, snap.i_t2_drive__uA, snap.i_t2_floor__uA), 0.0)
