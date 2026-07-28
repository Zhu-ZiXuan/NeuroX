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
    XbarCell1t1rSnap,
)


class Ye2023Jssc2t1rCellConfig(XbarCell1t1rLinearConfig):
    """Physical knobs for the WH-2T1R lookup cell.

    Attributes:
        i_t2_table__uA: Unit-scale (m = 1) T2 compute current [uA] indexed
            ``[input_bit][weight_state]``: ``input_bit`` in ``{0, 1}`` and a
            state axis of length ``w_state_num`` (state 0 = HRS, 1 = LRS).
            Row ``[1]`` holds the state-dependent on-currents and row ``[0]``
            the off-cell floor at input 0. All entries finite and >= 0.
    """

    i_t2_table__uA: tuple[tuple[float, ...], ...]

    def validate(self) -> None:
        super().validate()

        w_state_num = len(self.g_cell_off_table__uS)
        if len(self.i_t2_table__uA) != 2:
            raise ValueError(f"require: len(i_t2_table__uA) ({len(self.i_t2_table__uA)}) == 2 (input bit 0 / 1)")
        for input_bit, row in enumerate(self.i_t2_table__uA):
            if len(row) != w_state_num:
                raise ValueError(
                    f"require: len(i_t2_table__uA[{input_bit}]) ({len(row)}) == w_state_num ({w_state_num})"
                )
            for state_idx, entry in enumerate(row):
                if not (math.isfinite(entry) and entry >= 0):
                    raise ValueError(
                        f"require: every i_t2_table__uA entry finite and >= 0; "
                        f"got {entry} at (input {input_bit}, state {state_idx})"
                    )


class Ye2023Jssc2t1rCellPolicy(XbarCell1t1rLinearPolicy):
    """Empty nonideality policy for the deterministic WH-2T1R lookup cell."""


@dataclass(frozen=True, kw_only=True)
class Ye2023Jssc2t1rCellSnap(XbarCell1t1rLinearSnap):
    """Per-call snap of a WH-2T1R cell's programmed state.

    Attributes:
        i_t2_in0__uA: Unit-scale (m = 1) T2 current [uA] at input bit 0,
            pre-selected for the programmed state. Shape: ``[..., col, row]``
            (chunk-sliced).
        i_t2_in1__uA: Unit-scale (m = 1) T2 current [uA] at input bit 1. Same
            shape.
    """

    i_t2_in0__uA: Tensor
    i_t2_in1__uA: Tensor


@XbarCell1t1r.register_neurox_module(
    config_type=Ye2023Jssc2t1rCellConfig,
    policy_type=Ye2023Jssc2t1rCellPolicy,
)
class Ye2023Jssc2t1rCell(XbarCell1t1rLinear):
    """WH-2T1R lookup cell: linear 1T1R divider plus a per-state I_T2 table.

    The T2 compute current is selected at :meth:`program` time and read back by
    :meth:`lookup_i_t2` at unit slice scale.

    Args:
        config: WH-2T1R cell configuration.
        policy: WH-2T1R cell policy.
        inst_shape: Per-instance shape ``(*prefix, col, row)``.
        dtype: Tensor dtype for internal buffers.
        T__K: Operating temperature.
    """

    # --- Immutable model buffers ---

    _i_t2_table__uA: Tensor

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
            w_state_idx: State-index tensor in ``[0, w_state_num - 1]`` at
                ``self.inst_shape``.
        """
        super().program(w_state_idx)
        idx = w_state_idx.long()
        self._i_t2_in0__uA = self._i_t2_table__uA[0][idx]
        self._i_t2_in1__uA = self._i_t2_table__uA[1][idx]

    def snapshot(
        self,
        *,
        control: Tensor,
        shape: tuple[int, ...],
        multi_coords: tuple[Tensor, ...] | None,
        t_elapsed: float,
    ) -> Ye2023Jssc2t1rCellSnap:
        """Bundle the divider snap with the pre-selected per-state I_T2.

        Args:
            control: Word-line drive voltage [V]; broadcasts to
                ``[..., col, row]``.
            shape: Per-call broadcast shape ``(*leading, col, row)`` the
                per-cell fields fill.
            multi_coords: Advanced-index tuple selecting a chunk's
                positions from the broadcast view; ``None`` returns the
                full view.
            t_elapsed: Time elapsed since programming [s]; unused — the
                lookup model holds no time-dependent read state.

        Returns:
            Per-call WH-2T1R cell snap.
        """
        del t_elapsed

        def view(buf: Tensor) -> Tensor:
            expanded = buf.expand(shape) if shape else buf
            return expanded if multi_coords is None else expanded[multi_coords]

        return Ye2023Jssc2t1rCellSnap(
            v_wl__V=control,
            g_cell_on__uS=view(self._g_cell_on__uS),
            g_cell_off__uS=view(self._g_cell_off__uS),
            vx_ratio_on=view(self._vx_ratio_on),
            vx_ratio_off=view(self._vx_ratio_off),
            i_t2_in0__uA=view(self._i_t2_in0__uA),
            i_t2_in1__uA=view(self._i_t2_in1__uA),
        )

    def compute_dynamic_energy(
        self,
        v_bl: Tensor,
        v_sl: Tensor,
        dcop: XbarCell1t1rDcop,
        snap: XbarCell1t1rSnap,
    ) -> Tensor:
        """Per-cell PER-ACCESS capacitance switching energy [fJ].

        Sums the WL gate load of a driven row and the X-node dip-recharge, both
        gated on the WL-on threshold.

        Args:
            v_bl: Bit-line node voltage [V]. Shape: ``[..., col, row]``.
            v_sl: Source-line node voltage [V]; unused.
            dcop: Converged DCOP; unused — the dip is closed-form in
                ``vx_ratio_on``.
            snap: Per-call snap from :meth:`snapshot`, carrying the preset
                per-state divider ratios.

        Returns:
            Per-cell per-access switching energy [fJ]. Shape: ``[..., col, row]``.
        """
        del v_sl, dcop
        assert isinstance(snap, XbarCell1t1rLinearSnap)
        config = self.config
        wl_on = snap.v_wl__V > self._v_wl_on_threshold__V
        e_wl__fJ = config.c_wl__fF * snap.v_wl__V.square()
        e_x_dip__fJ = config.c_x__fF * v_bl.square() * snap.vx_ratio_on * wl_on
        return e_wl__fJ + e_x_dip__fJ

    def lookup_i_t2(self, input_high: Tensor, snap: Ye2023Jssc2t1rCellSnap) -> Tensor:
        """Unit-scale (m = 1) per-cell T2 compute current [uA].

        Args:
            input_high: Per-cell input bit; ``True`` selects the input-1 current
                and ``False`` the input-0 floor. Broadcasts to
                ``snap.i_t2_in1__uA`` shape.
            snap: Per-call snap from :meth:`snapshot`.

        Returns:
            Unit-scale T2 current [uA]. Shape: ``[..., col, row]``.
        """
        return torch.where(input_high, snap.i_t2_in1__uA, snap.i_t2_in0__uA)
