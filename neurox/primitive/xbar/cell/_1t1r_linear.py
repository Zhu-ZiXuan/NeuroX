"""Linearized 1T1R cell — table-driven per-state chord conductance and
divider drop fraction, no devices.

See also:
    docs/reference/primitive/xbar/cell/1t1r_linear.md
    docs/internals/primitive/xbar/cell/1t1r_linear.md
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from ._1t1r import (
    XbarCell1t1r,
    XbarCell1t1rConfig,
    XbarCell1t1rDcop,
    XbarCell1t1rPolicy,
    XbarCell1t1rSnap,
)


class XbarCell1t1rLinearConfig(XbarCell1t1rConfig):
    """Physical knobs for the linearized (table-driven) 1T1R cell."""

    g_cell_off_table__uS: tuple[float, ...]
    """Per-state BL-to-SL branch chord conductance `g_cell = I / (v_bl_op -
    v_sl_op)` at the calibration operating point with the WL off, indexed by
    weight state. Its length (>= 1) is the cell's weight-state count and all
    four tables share it; entries finite and >= 0."""
    g_cell_on_table__uS: tuple[float, ...]
    """The same chord conductance with the WL on."""
    vx_ratio_off_table: tuple[float, ...]
    """Per-state dimensionless BL-side drop fraction `vx_ratio = (v_bl_op -
    V_X) / (v_bl_op - v_sl_op)` with the WL off; entries finite and in
    `[0, 1]`."""
    vx_ratio_on_table: tuple[float, ...]
    """The same drop fraction with the WL on."""

    v_wl_on_threshold__V: float
    """Analog WL level above which the access device counts as on."""

    def validate(self) -> None:
        super().validate()

        w_state_num = len(self.g_cell_off_table__uS)
        if w_state_num < 1:
            raise ValueError(f"require: len(g_cell_off_table__uS) ({w_state_num}) >= 1")
        for name in ("g_cell_on_table__uS", "vx_ratio_off_table", "vx_ratio_on_table"):
            table: tuple[float, ...] = getattr(self, name)
            if len(table) != w_state_num:
                raise ValueError(f"require: len({name}) ({len(table)}) == len(g_cell_off_table__uS) ({w_state_num})")
        for name in ("g_cell_off_table__uS", "g_cell_on_table__uS"):
            for state_idx, entry in enumerate(getattr(self, name)):
                if not (math.isfinite(entry) and entry >= 0):
                    raise ValueError(f"require: every {name} entry finite and >= 0; got {entry} at state {state_idx}")
        for name in ("vx_ratio_off_table", "vx_ratio_on_table"):
            for state_idx, entry in enumerate(getattr(self, name)):
                if not (math.isfinite(entry) and 0.0 <= entry <= 1.0):
                    raise ValueError(
                        f"require: every {name} entry finite and in [0, 1]; got {entry} at state {state_idx}"
                    )


class XbarCell1t1rLinearPolicy(XbarCell1t1rPolicy):
    """Empty nonideality policy for the deterministic linear cell."""


@dataclass(frozen=True, kw_only=True)
class XbarCell1t1rLinearSnap(XbarCell1t1rSnap):
    """Per-call snap of a linearized 1T1R cell's programmed state."""

    g_cell_on__uS: Tensor
    """Branch chord conductance with the WL on. Shape: `[..., col, row]`."""
    g_cell_off__uS: Tensor
    """Branch chord conductance with the WL off. Shape: `[..., col, row]`."""
    vx_ratio_on: Tensor
    """BL-side drop fraction with the WL on. Shape: `[..., col, row]`."""
    vx_ratio_off: Tensor
    """BL-side drop fraction with the WL off. Shape: `[..., col, row]`."""


@XbarCell1t1r.register_neurox_module(
    config_type=XbarCell1t1rLinearConfig,
    policy_type=XbarCell1t1rLinearPolicy,
)
class XbarCell1t1rLinear(XbarCell1t1r[XbarCell1t1rLinearConfig, XbarCell1t1rLinearPolicy, XbarCell1t1rLinearSnap]):
    """Table-driven linearized 1T1R cell.

    Args:
        config: Linearized 1T1R configuration.
        policy: Linearized 1T1R policy.
        inst_shape: Per-instance shape `(..., col, row)`.
        dtype: Tensor dtype for internal buffers.
        T__K: Operating temperature.
    """

    # === Functional buffers ===

    _g_cell_off_table__uS: Tensor  # Shape: [w_state_num]
    _g_cell_on_table__uS: Tensor  # Shape: [w_state_num]
    _vx_ratio_off_table: Tensor  # Shape: [w_state_num]
    _vx_ratio_on_table: Tensor  # Shape: [w_state_num]

    # === Programmed state ===

    _g_cell_off__uS: Tensor  # Shape: [*inst_shape]
    _g_cell_on__uS: Tensor  # Shape: [*inst_shape]
    _vx_ratio_off: Tensor  # Shape: [*inst_shape]
    _vx_ratio_on: Tensor  # Shape: [*inst_shape]

    def __init__(
        self,
        *,
        config: XbarCell1t1rLinearConfig,
        policy: XbarCell1t1rLinearPolicy,
        inst_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
    ) -> None:
        super().__init__(config=config, policy=policy, inst_shape=inst_shape, dtype=dtype, T__K=T__K)

        self.register_buffer(
            "_g_cell_off_table__uS",
            torch.tensor(config.g_cell_off_table__uS, dtype=dtype),
            persistent=False,
        )
        self.register_buffer(
            "_g_cell_on_table__uS",
            torch.tensor(config.g_cell_on_table__uS, dtype=dtype),
            persistent=False,
        )
        self.register_buffer(
            "_vx_ratio_off_table",
            torch.tensor(config.vx_ratio_off_table, dtype=dtype),
            persistent=False,
        )
        self.register_buffer(
            "_vx_ratio_on_table",
            torch.tensor(config.vx_ratio_on_table, dtype=dtype),
            persistent=False,
        )

        self._v_wl_on_threshold__V = config.v_wl_on_threshold__V

    @property
    def w_state_num(self) -> int:
        return len(self.config.g_cell_off_table__uS)

    def program(self, w_state_idx: Tensor) -> None:
        """Program per-cell branch parameters from state indices.

        Args:
            w_state_idx: State-index tensor in `[0, w_state_num - 1]`.
                Shape: `[*inst_shape]`.
        """
        if tuple(w_state_idx.shape) != self.inst_shape:
            raise ValueError(f"program() expects w_state_idx.shape {self.inst_shape}; got {tuple(w_state_idx.shape)}")
        idx = w_state_idx.long()
        if bool((idx < 0).any()) or bool((idx >= self.w_state_num).any()):
            raise ValueError(f"program() expects state indices in [0, {self.w_state_num}); got out-of-range entries")
        self._g_cell_off__uS = self._g_cell_off_table__uS[idx]
        self._g_cell_on__uS = self._g_cell_on_table__uS[idx]
        self._vx_ratio_off = self._vx_ratio_off_table[idx]
        self._vx_ratio_on = self._vx_ratio_on_table[idx]

    def snapshot(
        self,
        *,
        control: Tensor,
        shape: tuple[int, ...],
        t_elapsed: float,
    ) -> XbarCell1t1rLinearSnap:
        """Bundle the programmed branch parameters with the WL control drive.

        Args:
            control: Word-line drive voltage [V] at each cell's gate.
                Shape: `[..., col, row]`.
            shape: Per-call broadcast shape `(..., col, row)` the
                branch-parameter fields fill.
            t_elapsed: Time elapsed since programming [s]; unused, the linear
                model holding no time-dependent read state.

        Returns:
            Per-call linearized 1T1R cell snap.
        """
        del t_elapsed

        def view(buf: Tensor) -> Tensor:
            return buf.expand(shape) if shape else buf

        return XbarCell1t1rLinearSnap(
            v_wl__V=control,
            g_cell_on__uS=view(self._g_cell_on__uS),
            g_cell_off__uS=view(self._g_cell_off__uS),
            vx_ratio_on=view(self._vx_ratio_on),
            vx_ratio_off=view(self._vx_ratio_off),
        )

    def _select_branch_params(self, snap: XbarCell1t1rLinearSnap) -> tuple[Tensor, Tensor]:
        """WL-switched `(g_cell [uS], vx_ratio)` of the linear branch."""
        on = snap.v_wl__V > self._v_wl_on_threshold__V
        g_cell = torch.where(on, snap.g_cell_on__uS, snap.g_cell_off__uS)
        vx_ratio = torch.where(on, snap.vx_ratio_on, snap.vx_ratio_off)
        return g_cell, vx_ratio

    def solve_branch(
        self,
        v_bl: Tensor,
        v_sl: Tensor,
        snap: XbarCell1t1rLinearSnap,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Closed-form branch solve: `(i__uA, di_dvbl__uS, di_dvsl__uS)`."""
        g_cell, _vx_ratio = self._select_branch_params(snap)
        i__uA = g_cell * (v_bl - v_sl)
        return i__uA, g_cell, -g_cell

    def solve_dc(
        self,
        v_bl: Tensor,
        v_sl: Tensor,
        snap: XbarCell1t1rLinearSnap,
    ) -> XbarCell1t1rDcop:
        """Full branch working point including the divider V_X."""
        g_cell, vx_ratio = self._select_branch_params(snap)
        dv = v_bl - v_sl
        i__uA = g_cell * dv
        v_x = v_bl - vx_ratio * dv
        return XbarCell1t1rDcop(
            i__uA=i__uA,
            di_dvbl__uS=g_cell,
            di_dvsl__uS=-g_cell,
            v_x__V=v_x,
        )
