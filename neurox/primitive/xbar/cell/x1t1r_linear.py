"""Linearized 1T1R cell — table-driven per-state chord conductance and divider drop fraction, no devices.

See Also:
    docs/reference/primitive/xbar/cell/1t1r_linear.md
"""

from __future__ import annotations

import torch
from torch import Tensor

from .x1t1r import (
    XbarCell1t1r,
    XbarCell1t1rConfig,
    XbarCell1t1rDcop,
    XbarCell1t1rPolicy,
    XbarCell1t1rSnap,
)


class XbarCell1t1rLinearConfig(XbarCell1t1rConfig):
    g_cell_off_table__uS: tuple[float, ...]
    """Per-state BL-to-SL branch chord conductance `g_cell__uS = I / (v_bl_op__V
    - v_sl_op__V)` at the calibration operating point with the WL off, indexed by
    weight state. Its length (>= 1) is the cell's weight-state count and all
    four tables share it; entries finite and >= 0."""
    g_cell_on_table__uS: tuple[float, ...]
    """The same chord conductance with the WL on."""
    vx_ratio_off_table: tuple[float, ...]
    """Per-state dimensionless BL-side drop fraction `vx_ratio = (v_bl_op__V -
    V_X) / (v_bl_op__V - v_sl_op__V)` with the WL off; entries finite and in
    `[0, 1]`."""
    vx_ratio_on_table: tuple[float, ...]
    """The same drop fraction with the WL on."""

    v_wl_on_threshold__V: float
    """Analog WL level above which the access device counts as on."""

    def validate(self) -> None:
        super().validate()

        self._require_non_empty(self.g_cell_off_table__uS, "g_cell_off_table__uS")
        self._require_same_len(
            self.g_cell_on_table__uS,
            "g_cell_on_table__uS",
            self.g_cell_off_table__uS,
            "g_cell_off_table__uS",
        )
        self._require_same_len(
            self.vx_ratio_off_table,
            "vx_ratio_off_table",
            self.g_cell_off_table__uS,
            "g_cell_off_table__uS",
        )
        self._require_same_len(
            self.vx_ratio_on_table,
            "vx_ratio_on_table",
            self.g_cell_off_table__uS,
            "g_cell_off_table__uS",
        )

        for state_idx, entry in enumerate(self.g_cell_off_table__uS):
            self._require_non_neg(entry, f"g_cell_off_table__uS[{state_idx}]")
        for state_idx, entry in enumerate(self.g_cell_on_table__uS):
            self._require_non_neg(entry, f"g_cell_on_table__uS[{state_idx}]")
        for state_idx, entry in enumerate(self.vx_ratio_off_table):
            self._require_in_closed_interval(entry, f"vx_ratio_off_table[{state_idx}]", 0.0, 1.0)
        for state_idx, entry in enumerate(self.vx_ratio_on_table):
            self._require_in_closed_interval(entry, f"vx_ratio_on_table[{state_idx}]", 0.0, 1.0)


class XbarCell1t1rLinearPolicy(XbarCell1t1rPolicy):
    pass


class XbarCell1t1rLinearSnap(XbarCell1t1rSnap):
    g_cell_on__uS: Tensor
    """Branch chord conductance with the WL on."""
    g_cell_off__uS: Tensor
    """Branch chord conductance with the WL off."""
    vx_ratio_on: Tensor
    """BL-side drop fraction with the WL on."""
    vx_ratio_off: Tensor
    """BL-side drop fraction with the WL off."""


@XbarCell1t1r.register_neurox_module(config_type=XbarCell1t1rLinearConfig, policy_type=XbarCell1t1rLinearPolicy)
class XbarCell1t1rLinear[ConfigT: XbarCell1t1rLinearConfig, PolicyT: XbarCell1t1rLinearPolicy](
    XbarCell1t1r[ConfigT, PolicyT, XbarCell1t1rLinearSnap]
):
    # === Functional buffers ===

    _g_cell_off_table__uS: Tensor  # Shape: [w_state]
    _g_cell_on_table__uS: Tensor  # Shape: [w_state]
    _vx_ratio_off_table: Tensor  # Shape: [w_state]
    _vx_ratio_on_table: Tensor  # Shape: [w_state]

    # === Programmed state ===

    _g_cell_off__uS: Tensor  # Shape: [*inst_shape]
    _g_cell_on__uS: Tensor  # Shape: [*inst_shape]
    _vx_ratio_off: Tensor  # Shape: [*inst_shape]
    _vx_ratio_on: Tensor  # Shape: [*inst_shape]

    def __init__(
        self,
        *,
        config: ConfigT,
        policy: PolicyT,
        inst_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
    ) -> None:
        super().__init__(config=config, policy=policy, inst_shape=inst_shape, dtype=dtype, T__K=T__K)

        self._register_nonpersistent_buffer(
            "_g_cell_off_table__uS",
            torch.tensor(config.g_cell_off_table__uS, dtype=dtype),
        )
        self._register_nonpersistent_buffer(
            "_g_cell_on_table__uS",
            torch.tensor(config.g_cell_on_table__uS, dtype=dtype),
        )
        self._register_nonpersistent_buffer(
            "_vx_ratio_off_table",
            torch.tensor(config.vx_ratio_off_table, dtype=dtype),
        )
        self._register_nonpersistent_buffer(
            "_vx_ratio_on_table",
            torch.tensor(config.vx_ratio_on_table, dtype=dtype),
        )

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
        # Gather every table here so the branch solve holds no index lookup.
        self._g_cell_off__uS = self._g_cell_off_table__uS[idx]
        self._g_cell_on__uS = self._g_cell_on_table__uS[idx]
        self._vx_ratio_off = self._vx_ratio_off_table[idx]
        self._vx_ratio_on = self._vx_ratio_on_table[idx]

    def is_wl_on(self, snap: XbarCell1t1rSnap) -> Tensor:
        """Return whether the linear model selects its WL-on tables."""
        return snap.v_wl__V > self.config.v_wl_on_threshold__V

    def solve_dc(
        self,
        v_bl__V: Tensor,
        v_sl__V: Tensor,
        snap: XbarCell1t1rLinearSnap,
    ) -> XbarCell1t1rDcop:
        """Full branch working point including the divider V_X."""
        on = self.is_wl_on(snap)
        g_cell__uS = torch.where(on, snap.g_cell_on__uS, snap.g_cell_off__uS)
        vx_ratio = torch.where(on, snap.vx_ratio_on, snap.vx_ratio_off)
        dv__V = v_bl__V - v_sl__V
        return XbarCell1t1rDcop(
            i__uA=g_cell__uS * dv__V,
            di_dvbl__uS=g_cell__uS,
            di_dvsl__uS=-g_cell__uS,
            v_x__V=v_bl__V - vx_ratio * dv__V,
        )

    def snapshot(
        self,
        *,
        control: Tensor,
        shape: tuple[int, ...],
        t_elapsed: float,
    ) -> XbarCell1t1rLinearSnap:
        """Bundle the programmed branch parameters with the WL control drive.

        `t_elapsed` is unused: the linear model holds no time-dependent read
        state.
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
