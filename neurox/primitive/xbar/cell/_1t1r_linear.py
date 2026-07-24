"""Linearized 1T1R cell — table-driven per-state chord conductance and
divider drop fraction, no devices.

See also:
    docs/reference/primitive/xbar/cell/_1t1r/cell_linear.md
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
    """Physical knobs for the linearized (table-driven) 1T1R cell.

    Attributes:
        g_cell_off_table__uS: Per-w_state total BL-to-SL branch chord
            conductance ``g_cell = I / (v_bl_op - v_sl_op)`` at the
            calibration operating point with the WL off, indexed by the
            weight-state index. The length defines the cell's
            weight-state count (all four tables share it, >= 1); all
            entries finite and >= 0 (zero is a cut-off branch's honest
            leakage value — array nonsingularity is carried by the wire
            conductances).
        g_cell_on_table__uS: The same chord conductance with the WL on.
            Same length and constraints as ``g_cell_off_table__uS``.
        vx_ratio_off_table: Per-w_state dimensionless BL-side drop
            fraction ``vx_ratio = (v_bl_op - V_X) / (v_bl_op - v_sl_op)``
            with the WL off, i.e. ``V_X = V_BL - vx_ratio * (V_BL -
            V_SL)`` — equivalently ``R_BL / (R_BL + R_SL)`` of the
            branch divider. Same length as ``g_cell_off_table__uS``; all
            entries finite and in ``[0, 1]``.
        vx_ratio_on_table: The same drop fraction with the WL on. Same
            length and constraints as ``vx_ratio_off_table``.
        v_wl_on_threshold__V: Analog WL level above which the access
            device counts as on.
    """

    g_cell_off_table__uS: tuple[float, ...]
    g_cell_on_table__uS: tuple[float, ...]
    vx_ratio_off_table: tuple[float, ...]
    vx_ratio_on_table: tuple[float, ...]

    v_wl_on_threshold__V: float

    def validate(self) -> None:
        super().validate()
        self.validate_tables()

    def validate_tables(self) -> None:
        w_states = len(self.g_cell_off_table__uS)
        if w_states < 1:
            raise ValueError(f"require: len(g_cell_off_table__uS) ({w_states}) >= 1")
        for name in ("g_cell_on_table__uS", "vx_ratio_off_table", "vx_ratio_on_table"):
            table: tuple[float, ...] = getattr(self, name)
            if len(table) != w_states:
                raise ValueError(f"require: len({name}) ({len(table)}) == len(g_cell_off_table__uS) ({w_states})")
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
    """Per-call snap of a linearized 1T1R cell's programmed state.

    Attributes:
        g_cell_on__uS: Branch chord conductance at WL on. Shape:
            ``[..., col, row]`` (chunk-sliced).
        g_cell_off__uS: Branch chord conductance at WL off. Same shape.
        vx_ratio_on: BL-side drop fraction at WL on. Same shape.
        vx_ratio_off: BL-side drop fraction at WL off. Same shape.
    """

    g_cell_on__uS: Tensor
    g_cell_off__uS: Tensor
    vx_ratio_on: Tensor
    vx_ratio_off: Tensor


@XbarCell1t1r.register_key(XbarCell1t1rLinearConfig)
class XbarCell1t1rLinear(XbarCell1t1r[XbarCell1t1rLinearConfig, XbarCell1t1rLinearPolicy, XbarCell1t1rLinearSnap]):
    """Table-driven linearized 1T1R cell.

    Args:
        config: Linearized 1T1R configuration.
        policy: Linearized 1T1R policy.
        inst_shape: Per-instance shape ``(*prefix, col, row)``.
        dtype: Tensor dtype for internal buffers.
        T__K: Operating temperature.
    """

    _g_cell_off_table__uS: Tensor
    _g_cell_on_table__uS: Tensor
    _vx_ratio_off_table: Tensor
    _vx_ratio_on_table: Tensor
    g_cell_on__uS: Tensor
    g_cell_off__uS: Tensor
    vx_ratio_on: Tensor
    vx_ratio_off: Tensor

    def __init__(
        self,
        *,
        config: XbarCell1t1rLinearConfig,
        policy: XbarCell1t1rLinearPolicy,
        inst_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
    ) -> None:
        if not isinstance(policy, XbarCell1t1rLinearPolicy):
            raise TypeError(f"XbarCell1t1rLinear requires an XbarCell1t1rLinearPolicy; got {type(policy).__name__}")
        super().__init__(config=config, policy=policy, inst_shape=inst_shape, dtype=dtype, T__K=T__K)

        for name, table in (
            ("_g_cell_off_table__uS", config.g_cell_off_table__uS),
            ("_g_cell_on_table__uS", config.g_cell_on_table__uS),
            ("_vx_ratio_off_table", config.vx_ratio_off_table),
            ("_vx_ratio_on_table", config.vx_ratio_on_table),
        ):
            self.register_buffer(name, torch.tensor(table, dtype=dtype), persistent=False)
        for name in ("g_cell_on__uS", "g_cell_off__uS", "vx_ratio_on", "vx_ratio_off"):
            self.register_buffer(name, torch.zeros((), dtype=dtype), persistent=False)

        self.w_states = len(config.g_cell_off_table__uS)

        self.v_wl_on_threshold__V = config.v_wl_on_threshold__V

    def program(self, w_state_idx: Tensor) -> None:
        """Program per-cell branch parameters from state indices.

        Args:
            w_state_idx: State-index tensor in ``[0, w_states - 1]`` at
                ``self.inst_shape``.
        """
        if tuple(w_state_idx.shape) != self.inst_shape:
            raise ValueError(f"program() expects w_state_idx.shape {self.inst_shape}; got {tuple(w_state_idx.shape)}")
        idx = w_state_idx.long()
        if bool((idx < 0).any()) or bool((idx >= self.w_states).any()):
            raise ValueError(f"program() expects state indices in [0, {self.w_states}); got out-of-range entries")
        self.g_cell_off__uS = self._g_cell_off_table__uS[idx]
        self.g_cell_on__uS = self._g_cell_on_table__uS[idx]
        self.vx_ratio_off = self._vx_ratio_off_table[idx]
        self.vx_ratio_on = self._vx_ratio_on_table[idx]

    def snapshot(
        self,
        *,
        control: Tensor,
        shape: tuple[int, ...],
        multi_coords: tuple[Tensor, ...] | None,
        t_elapsed: float,
    ) -> XbarCell1t1rLinearSnap:
        """Bundle the programmed branch parameters with the WL control drive.

        Args:
            control: Word-line drive voltage [V]; broadcasts to
                ``[..., col, row]``.
            shape: Per-call broadcast shape ``(*leading, col, row)`` the
                branch-parameter fields fill.
            multi_coords: Advanced-index tuple selecting a chunk's
                positions from the broadcast view; ``None`` returns the
                full view.
            t_elapsed: Time elapsed since programming [s]; unused — the
                linear model holds no time-dependent read state.

        Returns:
            Per-call linearized 1T1R cell snap.
        """
        del t_elapsed

        def view(buf: Tensor) -> Tensor:
            expanded = buf.expand(shape) if shape else buf
            return expanded if multi_coords is None else expanded[multi_coords]

        return XbarCell1t1rLinearSnap(
            v_wl__V=control,
            g_cell_on__uS=view(self.g_cell_on__uS),
            g_cell_off__uS=view(self.g_cell_off__uS),
            vx_ratio_on=view(self.vx_ratio_on),
            vx_ratio_off=view(self.vx_ratio_off),
        )

    def _branch_params(self, snap: XbarCell1t1rLinearSnap) -> tuple[Tensor, Tensor]:
        """WL-switched ``(g_cell [uS], vx_ratio)`` of the linear branch."""
        on = snap.v_wl__V > self.v_wl_on_threshold__V
        g_cell = torch.where(on, snap.g_cell_on__uS, snap.g_cell_off__uS)
        vx_ratio = torch.where(on, snap.vx_ratio_on, snap.vx_ratio_off)
        return g_cell, vx_ratio

    def solve_branch(
        self,
        v_bl: Tensor,
        v_sl: Tensor,
        snap: XbarCell1t1rLinearSnap,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Closed-form branch solve: ``(i__uA, di_dvbl__uS, di_dvsl__uS)``."""
        g_cell, _vx_ratio = self._branch_params(snap)
        i__uA = g_cell * (v_bl - v_sl)
        return i__uA, g_cell, -g_cell

    def solve_dc(
        self,
        v_bl: Tensor,
        v_sl: Tensor,
        snap: XbarCell1t1rLinearSnap,
    ) -> XbarCell1t1rDcop:
        """Full branch working point including the divider ``V_X``.

        The linear divider's internal KCL is exact by construction, so the
        cell carries no residual concept.
        """
        g_cell, vx_ratio = self._branch_params(snap)
        dv = v_bl - v_sl
        i__uA = g_cell * dv
        v_x = v_bl - vx_ratio * dv
        return XbarCell1t1rDcop(
            i__uA=i__uA,
            di_dvbl__uS=g_cell,
            di_dvsl__uS=-g_cell,
            v_x__V=v_x,
        )
