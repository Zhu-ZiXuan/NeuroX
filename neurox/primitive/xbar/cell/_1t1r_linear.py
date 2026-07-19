"""Linearized 1T1R cell — table-driven per-state sub-conductances, no devices.

See also:
    docs/reference/primitive/xbar/cell/_1t1r/cell_linear.md
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ._1t1r import (
    XbarCell1t1r,
    XbarCell1t1rConfig,
    XbarCell1t1rDcop,
    XbarCell1t1rPolicy,
    XbarCell1t1rResiduals,
    XbarCell1t1rSnap,
)
from .base import XbarCell

# ---------------------------------------------------------------------------
# Config / policy / result containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class XbarCell1t1rLinearConfig(XbarCell1t1rConfig):
    """Physical knobs for the linearized (table-driven) 1T1R cell.

    Attributes:
        g_bl_table__uS: Per-w_state BL-side (RRAM-slot) effective
            conductance, one ``(off, on)`` pair per row indexed by the
            WL on/off level. The row count defines the cell's weight-state
            count; all entries positive.
        g_sl_table__uS: Per-w_state SL-side (access-slot) effective
            conductance; same shape and constraints as
            ``g_bl_table__uS``.
        v_wl_on_threshold__V: Analog WL level above which the access
            device counts as on.
    """

    g_bl_table__uS: tuple[tuple[float, float], ...]
    g_sl_table__uS: tuple[tuple[float, float], ...]

    v_wl_on_threshold__V: float

    def validate(self) -> None:
        super().validate()
        self.validate_tables()

    def validate_tables(self) -> None:
        if len(self.g_sl_table__uS) != len(self.g_bl_table__uS):
            raise ValueError(
                f"require: len(g_sl_table__uS) ({len(self.g_sl_table__uS)}) == "
                f"len(g_bl_table__uS) ({len(self.g_bl_table__uS)})"
            )
        for name, table in (
            ("g_bl_table__uS", self.g_bl_table__uS),
            ("g_sl_table__uS", self.g_sl_table__uS),
        ):
            for row_idx, row in enumerate(table):
                if len(row) != 2:
                    raise ValueError(f"require: len({name}[{row_idx}]) ({len(row)}) == 2 (off, on)")
                for entry in row:
                    if not (entry > 0):
                        raise ValueError(f"require: every {name} entry > 0; got {entry} in row {row_idx}")


@dataclass(frozen=True)
class XbarCell1t1rLinearPolicy(XbarCell1t1rPolicy):
    """Empty nonideality policy for the linearized 1T1R cell.

    The linear model is deterministic: every nonideality it represents
    is baked into its conductance tables at calibration time.
    """


@dataclass(frozen=True, kw_only=True)
class XbarCell1t1rLinearSnap(XbarCell1t1rSnap):
    """Per-call snap of a linearized 1T1R cell's programmed state.

    Attributes:
        g_bl_on__uS: BL-side conductance at WL on. Shape:
            ``[..., col, row]`` (chunk-sliced).
        g_bl_off__uS: BL-side conductance at WL off. Same shape.
        g_sl_on__uS: SL-side conductance at WL on. Same shape.
        g_sl_off__uS: SL-side conductance at WL off. Same shape.
    """

    g_bl_on__uS: Tensor
    g_bl_off__uS: Tensor
    g_sl_on__uS: Tensor
    g_sl_off__uS: Tensor


# ---------------------------------------------------------------------------
# Cell
# ---------------------------------------------------------------------------


@XbarCell.register_key(XbarCell1t1rLinearConfig)
class XbarCell1t1rLinear(XbarCell1t1r):
    """Table-driven linearized 1T1R cell with a closed-form branch.

    Owns no device children. ``program`` gathers the per-state
    ``(off, on)`` sub-conductance tables once into instance-shaped
    buffers; the branch solve is a pure elementwise series combination
    switched by the WL threshold.
    """

    _g_bl_table__uS: Tensor
    _g_sl_table__uS: Tensor
    g_bl_on__uS: Tensor
    g_bl_off__uS: Tensor
    g_sl_on__uS: Tensor
    g_sl_off__uS: Tensor

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

        self.register_buffer(
            "_g_bl_table__uS",
            torch.tensor(config.g_bl_table__uS, dtype=dtype),
            persistent=False,
        )
        self.register_buffer(
            "_g_sl_table__uS",
            torch.tensor(config.g_sl_table__uS, dtype=dtype),
            persistent=False,
        )
        # Programmed at the w_layout shape by ``program``; 0-d until then.
        for name in ("g_bl_on__uS", "g_bl_off__uS", "g_sl_on__uS", "g_sl_off__uS"):
            self.register_buffer(name, torch.zeros((), dtype=dtype), persistent=False)

        self.w_states = len(config.g_bl_table__uS)

        self.v_wl_on_threshold__V = config.v_wl_on_threshold__V

    # -----------------------------------------------------------------
    # Snapshot / programming
    # -----------------------------------------------------------------

    def program(self, w_state_idx: Tensor) -> None:
        """Materialize the per-cell sub-conductance buffers from state indices.

        Gathers the ``(off, on)`` conductance tables by state index once,
        keeping the branch-solve hot path gather-free.

        Args:
            w_state_idx: State-index tensor in ``[0, w_states - 1]`` at
                ``self._inst_shape``.
        """
        if tuple(w_state_idx.shape) != self._inst_shape:
            raise ValueError(f"program() expects w_state_idx.shape {self._inst_shape}; got {tuple(w_state_idx.shape)}")
        idx = w_state_idx.long()
        if bool((idx < 0).any()) or bool((idx >= self.w_states).any()):
            raise ValueError(f"program() expects state indices in [0, {self.w_states}); got out-of-range entries")
        g_bl = self._g_bl_table__uS[idx]
        g_sl = self._g_sl_table__uS[idx]
        self.g_bl_off__uS = g_bl[..., 0]
        self.g_bl_on__uS = g_bl[..., 1]
        self.g_sl_off__uS = g_sl[..., 0]
        self.g_sl_on__uS = g_sl[..., 1]

    def snapshot(
        self,
        *,
        control: Tensor,
        shape: tuple[int, ...],
        multi_coords: tuple[Tensor, ...] | None,
        t_elapsed: float,
    ) -> XbarCell1t1rLinearSnap:
        """Bundle the programmed sub-conductances with the WL control drive.

        Deterministic — the empty policy holds no draws; the programmed
        buffers are broadcast to ``shape`` and chunk-selected by
        ``multi_coords``.

        Args:
            control: Word-line drive voltage [V]; broadcasts to
                ``[..., col, row]``.
            shape: Per-call broadcast shape ``(*leading, col, row)`` the
                conductance fields fill.
            multi_coords: Advanced-index tuple selecting a chunk's
                positions from the broadcast view; ``None`` returns the
                full view.
            t_elapsed: Time elapsed since programming [s]; unused — the
                linear model holds no time-dependent read state.

        Returns:
            Per-call linearized 1T1R cell snap.
        """
        del t_elapsed  # no time-dependent read state in this cell

        def view(buf: Tensor) -> Tensor:
            expanded = buf.expand(shape) if shape else buf
            return expanded if multi_coords is None else expanded[multi_coords]

        return XbarCell1t1rLinearSnap(
            v_wl__V=control,
            g_bl_on__uS=view(self.g_bl_on__uS),
            g_bl_off__uS=view(self.g_bl_off__uS),
            g_sl_on__uS=view(self.g_sl_on__uS),
            g_sl_off__uS=view(self.g_sl_off__uS),
        )

    # -----------------------------------------------------------------
    # Branch solve
    # -----------------------------------------------------------------

    def _branch_conductances(self, snap: XbarCell1t1rLinearSnap) -> tuple[Tensor, Tensor, Tensor]:
        """WL-switched ``(g_bl, g_sl, g_series)`` of the linear branch [uS]."""
        on = snap.v_wl__V > self.v_wl_on_threshold__V
        g_bl = torch.where(on, snap.g_bl_on__uS, snap.g_bl_off__uS)
        g_sl = torch.where(on, snap.g_sl_on__uS, snap.g_sl_off__uS)
        g_series = g_bl * g_sl / (g_bl + g_sl)
        return g_bl, g_sl, g_series

    def solve_branch(
        self,
        v_bl: Tensor,
        v_sl: Tensor,
        snap: XbarCell1t1rLinearSnap,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Closed-form branch solve: ``(i__uA, di_dvbl__uS, di_dvsl__uS)``."""
        _g_bl, _g_sl, g_series = self._branch_conductances(snap)
        i__uA = g_series * (v_bl - v_sl)
        return i__uA, g_series, -g_series

    def solve_dc(
        self,
        v_bl: Tensor,
        v_sl: Tensor,
        snap: XbarCell1t1rLinearSnap,
        compute_residuals: bool = False,
    ) -> XbarCell1t1rDcop:
        """Full branch working point including the divider ``V_X``."""
        g_bl, _g_sl, g_series = self._branch_conductances(snap)
        i__uA = g_series * (v_bl - v_sl)
        v_x = v_bl - i__uA / g_bl
        residuals: XbarCell1t1rResiduals | None
        # Internal KCL is exact by construction in the linear divider.
        residuals = XbarCell1t1rResiduals(cell__uA=torch.zeros_like(i__uA)) if compute_residuals else None
        return XbarCell1t1rDcop(
            i__uA=i__uA,
            di_dvbl__uS=g_series,
            di_dvsl__uS=-g_series,
            residuals=residuals,
            v_x__V=v_x,
        )
