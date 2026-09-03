"""WH-2T1R array.

See Also:
    docs/reference/primitive/xbar/array/1t1r.md
    docs/system_design/xbar_solve.md
"""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor

from neurox.primitive.physics import e_cap_excursion__fJ
from neurox.primitive.xbar.array import (
    XbarArray1t1r,
    XbarArray1t1rConfig,
    XbarArray1t1rPolicy,
    XbarArray1t1rSolveProjection,
    XbarArray1t1rSteadyState,
)
from neurox.primitive.xbar.cell import XbarCell1t1rDcop, XbarCell1t1rSnap
from neurox.primitive.xbar.solver import ClampDcop, ClampDriver, ClampSnap, ColBlColSlDcop

from .cell import (
    Ye2023Jssc2t1rCell,
    Ye2023Jssc2t1rCellConfig,
    Ye2023Jssc2t1rCellPolicy,
)


class Ye2023Jssc2t1rArrayConfig(XbarArray1t1rConfig):
    cell_config: Ye2023Jssc2t1rCellConfig

    t2_gate_unit_c__fF: float
    """T2 gate capacitance at unit width."""
    tbl_node_unit_c__fF: float
    """Effective total capacitance to ground at a unit-width T2's TBL node."""

    def validate(self) -> None:
        super().validate()

        self._require_non_neg(self.t2_gate_unit_c__fF, "t2_gate_unit_c__fF")
        self._require_non_neg(self.tbl_node_unit_c__fF, "tbl_node_unit_c__fF")


class Ye2023Jssc2t1rArrayPolicy(XbarArray1t1rPolicy):
    cell_policy: Ye2023Jssc2t1rCellPolicy


class Ye2023Jssc2t1rSteadyState(XbarArray1t1rSteadyState):
    i_tbl_by_row__uA: Tensor
    """Selected TBL current at each physical row. Shape: `[..., row]`."""


class Ye2023Jssc2t1rArray(XbarArray1t1r[Ye2023Jssc2t1rArrayConfig, Ye2023Jssc2t1rArrayPolicy]):
    """WH-2T1R array."""

    cell: Ye2023Jssc2t1rCell

    # === Functional buffers ===

    _t2_multipliers: Tensor  # Shape: [col]
    _t2_gate_c_by_col__fF: Tensor  # Shape: [col]

    def __init__(
        self,
        *,
        config: Ye2023Jssc2t1rArrayConfig,
        policy: Ye2023Jssc2t1rArrayPolicy,
        inst_shape: tuple[int, ...],
        row_num: int,
        col_num: int,
        t2_multipliers: tuple[int, ...],
        v_tbl__V: float,
        vdd__V: float,
        dtype: torch.dtype,
        T__K: float,
    ) -> None:
        if len(t2_multipliers) != col_num:
            raise ValueError(f"require: len(t2_multipliers) ({len(t2_multipliers)}) == col_num ({col_num})")
        total_t2_multiplier = sum(t2_multipliers)
        super().__init__(
            config=config,
            policy=policy,
            inst_shape=inst_shape,
            row_num=row_num,
            col_num=col_num,
            vdd__V=vdd__V,
            dtype=dtype,
            T__K=T__K,
        )
        self._i_tbl_leak__uA = self.cell.i_t2_leak__uA * total_t2_multiplier
        self._cap_energy_per_active_tbl__fJ = e_cap_excursion__fJ(
            vdd__V,
            config.tbl_node_unit_c__fF * total_t2_multiplier,
            v_tbl__V,
        )
        self._register_nonpersistent_buffer("_t2_multipliers", torch.tensor(t2_multipliers, dtype=dtype))
        self._register_nonpersistent_buffer("_t2_gate_c_by_col__fF", config.t2_gate_unit_c__fF * self._t2_multipliers)

    def _init_children(self, *, dtype: torch.dtype, T__K: float) -> None:
        self.cell = Ye2023Jssc2t1rCell(
            config=self.config.cell_config,
            policy=self.policy.cell_policy,
            inst_shape=(*self.inst_shape, self._col_num, self._row_num),
            dtype=dtype,
            T__K=T__K,
        )

    @property
    def i_tbl_leak__uA(self) -> float:
        return self._i_tbl_leak__uA

    def solve_array[BLSnapT: ClampSnap, BLDcopT: ClampDcop, SLSnapT: ClampSnap, SLDcopT: ClampDcop](
        self,
        *,
        v_wl__V: Tensor,
        wl_phase_dims: tuple[int, ...],
        bl_driver: ClampDriver[BLSnapT, BLDcopT],
        bl_driver_snap: BLSnapT,
        sl_driver: ClampDriver[SLSnapT, SLDcopT],
        sl_driver_snap: SLSnapT,
    ) -> Ye2023Jssc2t1rSteadyState:
        """Settle the array and return its BL/SL/TBL state."""
        result = super().solve_array(
            v_wl__V=v_wl__V,
            wl_phase_dims=wl_phase_dims,
            bl_driver=bl_driver,
            bl_driver_snap=bl_driver_snap,
            sl_driver=sl_driver,
            sl_driver_snap=sl_driver_snap,
        )
        return cast(Ye2023Jssc2t1rSteadyState, result)

    def _cap_energy__fJ(
        self,
        *,
        solver_dcop: ColBlColSlDcop[XbarCell1t1rDcop],
        cell_snap: XbarCell1t1rSnap,
        bl_driver_snap: ClampSnap,
        sl_driver_snap: ClampSnap,
    ) -> Tensor:
        array_energy__fJ = super()._cap_energy__fJ(
            solver_dcop=solver_dcop,
            cell_snap=cell_snap,
            bl_driver_snap=bl_driver_snap,
            sl_driver_snap=sl_driver_snap,
        )
        # Shape: [..., col, row] -> [...]
        t2_gate_energy__fJ = e_cap_excursion__fJ(
            self._vdd__V,
            self._t2_gate_c_by_col__fF.unsqueeze(-1),
            solver_dcop.cell.v_x__V - bl_driver_snap.v_ref__V.unsqueeze(-1),
        ).sum(dim=(-2, -1))
        # Shape: [..., col, row] -> [..., row] -> [...]
        active_tbl_num = self.cell.is_wl_on(cell_snap).select(-2, 0).sum(dim=-1)
        # Shape: [...]
        tbl_energy__fJ = active_tbl_num.to(t2_gate_energy__fJ.dtype) * self._cap_energy_per_active_tbl__fJ
        return array_energy__fJ + t2_gate_energy__fJ + tbl_energy__fJ

    def _rest_cap_energy__fJ(self, *, v_bl_rest__V: Tensor, v_sl_rest__V: Tensor) -> Tensor:
        array_energy__fJ = super()._rest_cap_energy__fJ(
            v_bl_rest__V=v_bl_rest__V,
            v_sl_rest__V=v_sl_rest__V,
        )
        # Shape: [..., col] -> [...]
        t2_gate_energy__fJ = e_cap_excursion__fJ(
            self._vdd__V,
            self._row_num * self._t2_gate_c_by_col__fF,
            v_bl_rest__V,
        ).sum(dim=-1)
        return array_energy__fJ + t2_gate_energy__fJ

    def _project_dcop_impl(
        self,
        *,
        dcop: ColBlColSlDcop[XbarCell1t1rDcop],
        cell_snap: XbarCell1t1rSnap,
        bl_driver_snap: ClampSnap,
        sl_driver_snap: ClampSnap,
    ) -> XbarArray1t1rSolveProjection[Ye2023Jssc2t1rSteadyState]:
        """Reduce the T2 cell currents before releasing the full DCOP."""
        base_projection = super()._project_dcop_impl(
            dcop=dcop,
            cell_snap=cell_snap,
            bl_driver_snap=bl_driver_snap,
            sl_driver_snap=sl_driver_snap,
        )

        # Shape: [..., col, row]
        i_t2__uA = self.cell.i_t2_unit__uA(dcop.cell) * self._t2_multipliers.unsqueeze(-1)
        # Only TBLs paired with active WLs are connected to the readout.
        # Shape: [..., col, row] -> [..., row]
        i_tbl_by_row__uA = i_t2__uA.sum(dim=-2)
        # Shape: [..., row]
        i_tbl_by_row__uA = i_tbl_by_row__uA.where(self.cell.is_wl_on(cell_snap).select(-2, 0), 0.0)

        state = base_projection.steady_state
        return XbarArray1t1rSolveProjection(
            steady_state=Ye2023Jssc2t1rSteadyState(
                i_bl_port__uA=state.i_bl_port__uA,
                v_bl_clamp__V=state.v_bl_clamp__V,
                i_sl_port__uA=state.i_sl_port__uA,
                v_sl_drive__V=state.v_sl_drive__V,
                i_tbl_by_row__uA=i_tbl_by_row__uA,
            ),
            energy__fJ=base_projection.energy__fJ,
        )
