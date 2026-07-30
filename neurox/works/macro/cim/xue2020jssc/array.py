"""Serial-column 1T1R array core — a column-MUX-serialized cell grid and DC solve.

The ``col_num`` physical columns partition into ``serial`` MUX slots of
``gn * polarity * w_digit`` driver lanes each, seated by the macro-supplied slot map.
One broadcast DC solve settles every physical column exactly once, the serial
axis riding the leading batch (``solve_chunk_size`` applies), and the outputs
stay in the ``[..., serial, gn, polarity, w_digit]`` (slot, driver-lane) layout.

An off column (one outside its active slot) is DEFINED as grounded, so its node
voltages, its current, and its cap energy are identically zero and it is never
materialized: summing the solved ``[..., serial, ...]`` entries IS the full
physical total.

A reporter leaf billing capacitance only — no conduction; the macro bills the
input branch.

See also:
    docs/works/macro/cim/xue2020jssc/model.md
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields, replace
from typing import TypeVar

import torch
from torch import Tensor

from neurox.common import ModuleBase
from neurox.primitive.xbar.array import XbarArray1t1rConfig, XbarArray1t1rPolicy
from neurox.primitive.xbar.cell import XbarCell1t1r, XbarCell1t1rDcop, XbarCell1t1rSnap
from neurox.primitive.xbar.solver import (
    ClampDriver,
    NestedParallelRailSolver,
    SolverDcop,
    classify_leading_positions,
    iter_chunks,
    reassemble_chunks,
)
from neurox.primitive.xbar.solver.clamp import ClampSnap

BLSnapT = TypeVar("BLSnapT", bound=ClampSnap)
SLSnapT = TypeVar("SLSnapT", bound=ClampSnap)
SnapT = TypeVar("SnapT", bound=ClampSnap)

# Structured driver-lane trailing rank: (gn, polarity, w_digit).
_LANE_NDIM = 3


def _flatten_lane_snap(snap: SnapT) -> SnapT:
    """Flatten a snap's structured lane trailing into one flat column axis.

    The boundary drivers snapshot at their true inst trailing ``[..., serial,
    gn, polarity, w_digit]`` while the kernel solver consumes a flat column axis. Only
    fields of at least ``_LANE_NDIM`` dims carry the per-call broadcast shape;
    lower-rank constant fields (e.g. a 0-d slope) pass through untouched.
    """
    changes = {
        f.name: value.flatten(-_LANE_NDIM)
        for f in fields(snap)
        if isinstance(value := getattr(snap, f.name), Tensor) and value.ndim >= _LANE_NDIM
    }
    return replace(snap, **changes)


@dataclass(frozen=True)
class SerialColumnSteadyState:
    """Steady-state array output in the (slot, driver-lane) layout.

    Attributes:
        i_bl_port__uA: BL port current at the converged operating point.
            Shape: ``[..., serial, gn, polarity, w_digit]``.
        v_bl_clamp__V: BL clamp voltage at the converged operating point.
            Shape: ``[..., serial, gn, polarity, w_digit]``.
    """

    i_bl_port__uA: Tensor
    v_bl_clamp__V: Tensor


class SerialColumnXbarArray(ModuleBase[XbarArray1t1rConfig, XbarArray1t1rPolicy]):
    """Column-MUX serialized 1T1R array with wire parasitics and a DC solver.

    Args:
        config: 1T1R pure-array configuration (cell + wire + solver knobs).
        policy: Composite pure-array policy (cell policy + solve chunk knob).
        inst_shape: Per-instance replication shape.
        row_num: Number of array rows.
        col_num: Number of PHYSICAL columns; must equal ``slot_map.numel()``.
        slot_map: Integer bijection from a (slot, gn, polarity, w_digit) seat
            to its physical column index — the macro-computed placement of
            every physical column into its (slot, driver-lane) seat.
            Shape: ``[serial, gn, polarity, w_digit]``.
        dtype: Tensor dtype for internal buffers.
        T__K: Operating temperature.
    """

    # === Functional buffers ===

    _slot_map: Tensor  # Shape: [serial, gn, polarity, w_digit]

    # === Circuit constant buffers ===

    _bl_segment_r__MOhm: Tensor  # Shape: [row_num]
    _sl_segment_r__MOhm: Tensor  # Shape: [row_num]
    _bl_segment_g__uS: Tensor  # Shape: [row_num]
    _sl_segment_g__uS: Tensor  # Shape: [row_num]
    _bl_segment_c__fF: Tensor  # Shape: [row_num]
    _sl_segment_c__fF: Tensor  # Shape: [row_num]
    _latency_per_op__ns: Tensor  # Shape: []

    def __init__(
        self,
        *,
        config: XbarArray1t1rConfig,
        policy: XbarArray1t1rPolicy,
        inst_shape: tuple[int, ...],
        row_num: int,
        col_num: int,
        slot_map: Tensor,
        dtype: torch.dtype,
        T__K: float,
    ) -> None:
        if not (row_num > 1):
            raise ValueError(f"require: row_num ({row_num}) > 1")
        if not (col_num > 1):
            raise ValueError(f"require: col_num ({col_num}) > 1")
        if slot_map.ndim != 4:
            raise ValueError(f"require: slot_map.ndim ({slot_map.ndim}) == 4 — (slot, gn, 2, w_digit)")
        if slot_map.numel() != col_num:
            raise ValueError(f"require: slot_map.numel() ({slot_map.numel()}) == col_num ({col_num})")
        flat_map = slot_map.reshape(-1).long()
        if not torch.equal(torch.sort(flat_map).values, torch.arange(col_num)):
            raise ValueError("require: slot_map is a bijection onto [0, col_num)")

        super().__init__(config=config, policy=policy, inst_shape=inst_shape)
        self._row_num = row_num
        self._col_num = col_num
        self._serial_num = int(slot_map.shape[0])
        self._lane_shape = tuple(slot_map.shape[1:])
        self._active_col_num = math.prod(self._lane_shape)
        if not (self._active_col_num > 1):
            raise ValueError(f"require: active column count per slot ({self._active_col_num}) > 1")
        self.register_buffer("_slot_map", slot_map.long(), persistent=False)

        self._init_children(dtype=dtype, T__K=T__K)
        self._register_model_buffers(dtype=dtype)

        # WL wire cap seen by one held row across the FULL physical column
        # span — the per-plane billing coefficient, independent of `serial`.
        self._c_wl_wire_per_row__fF = config.wl_first_c__fF + (col_num - 1) * config.wl_segment_c__fF

    @property
    def _area_per_inst__um2(self) -> float:
        return self.config.area_per_inst__um2

    @property
    def _leakage_per_inst__uW(self) -> float:
        return self.config.leakage_per_inst__uW

    def _init_children(self, *, dtype: torch.dtype, T__K: float) -> None:
        """Construct the cell model (in seat order) and numerical solver."""
        self.cell = XbarCell1t1r.from_config(
            config=self.config.cell_config,
            policy=self.policy.cell_policy,
            inst_shape=self._cell_grid_shape,
            dtype=dtype,
            T__K=T__K,
        )
        self._solver = NestedParallelRailSolver(config=self.config.solver_config)

    def _register_model_buffers(self, *, dtype: torch.dtype) -> None:
        """Register fixed wire and PPA tensors."""
        config = self.config
        # Segment index zero connects the driver to the first cell.
        bl_segment_r__MOhm = torch.tensor(
            [config.bl_first_r__MOhm] + [config.bl_segment_r__MOhm] * (self._row_num - 1),
            dtype=dtype,
        )
        sl_segment_r__MOhm = torch.tensor(
            [config.sl_first_r__MOhm] + [config.sl_segment_r__MOhm] * (self._row_num - 1),
            dtype=dtype,
        )
        bl_segment_c__fF = torch.tensor(
            [config.bl_first_c__fF] + [config.bl_segment_c__fF] * (self._row_num - 1),
            dtype=dtype,
        )
        sl_segment_c__fF = torch.tensor(
            [config.sl_first_c__fF] + [config.sl_segment_c__fF] * (self._row_num - 1),
            dtype=dtype,
        )
        self.register_buffer("_bl_segment_r__MOhm", bl_segment_r__MOhm, persistent=False)
        self.register_buffer("_sl_segment_r__MOhm", sl_segment_r__MOhm, persistent=False)
        self.register_buffer("_bl_segment_g__uS", 1.0 / bl_segment_r__MOhm, persistent=False)
        self.register_buffer("_sl_segment_g__uS", 1.0 / sl_segment_r__MOhm, persistent=False)
        self.register_buffer("_bl_segment_c__fF", bl_segment_c__fF, persistent=False)
        self.register_buffer("_sl_segment_c__fF", sl_segment_c__fF, persistent=False)
        self.register_buffer(
            "_latency_per_op__ns",
            torch.tensor(self.config.latency_per_op__ns, dtype=dtype),
            persistent=False,
        )

    def _sample_fabricate_mismatch(self) -> None:
        pass

    @property
    def w_state_num(self) -> int:
        """Number of programmable states exposed by each cell."""
        return self.cell.w_state_num

    @property
    def weight_grid_shape(self) -> tuple[int, ...]:
        """Shape of the PHYSICAL program layout: ``(*inst_shape, col_num, row_num)``."""
        return (*self.inst_shape, self._col_num, self._row_num)

    @property
    def _cell_grid_shape(self) -> tuple[int, ...]:
        """Seat-order cell grid: ``(*inst_shape, serial, gn * polarity * w_digit, row)``."""
        return (*self.inst_shape, self._serial_num, self._active_col_num, self._row_num)

    def _seat_weight(self, w_state_idx: Tensor) -> Tensor:
        """Reorder physical columns into the seat-order cell grid.

        Args:
            w_state_idx: State-index tensor at ``self.weight_grid_shape``.
                Shape: ``[*inst_shape, col_num, row_num]``.

        Returns:
            State indices at ``self._cell_grid_shape``.
            Shape: ``[*inst_shape, serial, act, row_num]``.
        """
        # Shape: [*inst_shape, col_num, row] -> [*inst_shape, serial * act, row]
        w_seated = w_state_idx.index_select(-2, self._slot_map.reshape(-1))
        # Shape: [*inst_shape, serial * act, row] -> [*inst_shape, serial, act, row]
        return w_seated.unflatten(-2, (self._serial_num, self._active_col_num))

    def program(self, w_state_idx: Tensor) -> None:
        """Write the cells from one PHYSICAL-layout state-index tensor.

        The slot map places each physical column into its (slot, lane) seat;
        the cell grid holds the seated order the solve consumes directly.

        Args:
            w_state_idx: State-index tensor in ``[0, w_state_num - 1]``, whose
                shape must match ``self.weight_grid_shape``.
                Shape: ``[*inst_shape, col_num, row_num]``.
        """
        if tuple(w_state_idx.shape) != self.weight_grid_shape:
            raise ValueError(
                f"program() expects w_state_idx.shape {self.weight_grid_shape}; got {tuple(w_state_idx.shape)}"
            )
        self.cell.program(self._seat_weight(w_state_idx))

    @torch.compiler.disable(
        recursive=False,
        reason="eager chunk loop; the fixed-shape per-chunk solver body is compiled separately",
    )
    def solve_array(
        self,
        v_wl: Tensor,
        *,
        bl_driver: ClampDriver[BLSnapT],
        bl_v_ref__V: Tensor,
        sl_driver: ClampDriver[SLSnapT],
        sl_v_ref__V: Tensor,
    ) -> SerialColumnSteadyState:
        """Settle every physical column to DC under an analog WL drive.

        The serial slot axis rides the leading batch: one broadcast solve
        settles each physical column exactly once, in its (slot, lane) seat.
        The drivers snapshot at their true inst trailing
        ``[..., serial, gn, polarity, w_digit]``.

        Args:
            v_wl: Analog WL drive [V] — one plane, held across all serial
                slots.
                Shape: ``[..., row_num]``.
            bl_driver: BL boundary clamp (structural ``ClampDriver`` role).
            bl_v_ref__V: BL-clamp reference voltage.
            sl_driver: SL boundary clamp (structural ``ClampDriver`` role).
            sl_v_ref__V: SL-drive reference voltage.

        Returns:
            :class:`SerialColumnSteadyState` carrying the per-lane BL port
            current [uA] and BL clamp voltage [V] at full leading, both in
            the ``[..., serial, gn, polarity, w_digit]`` layout.
        """

        # --- 1: infer the broadcast-leading shape (serial rides the leading) ---

        # Shape: [..., row] -> [..., serial=1, act=1, row]
        v_wl_grid = v_wl.unsqueeze(-2).unsqueeze(-2)
        g_shape = self._cell_grid_shape
        full_shape = torch.broadcast_shapes(g_shape, v_wl_grid.shape)
        *batch_list, col_num, row_num = full_shape
        leading = tuple(batch_list)  # leading[-1] is the serial axis
        cell_trailing = (col_num, row_num)
        col_trailing = (col_num,)

        # --- 2: classify serial and parallel leading positions ---

        a_positions, _b_positions = classify_leading_positions(
            x_shape=tuple(v_wl_grid.shape),
            g_shape=g_shape,
            leading_rank=len(leading),
        )

        # Shape: [..., 1, 1, row] -> [*leading, 1, row]
        v_wl_full = v_wl_grid.expand(*leading, 1, row_num)

        # --- 3: solve and measure each chunk ---

        i_bl_port_chunks: list[Tensor] = []
        v_bl_clamp_chunks: list[Tensor] = []
        chunk_energies: list[Tensor] = []
        global_indices: list[Tensor] = []
        record_dynamic_energy = self._is_dynamic_energy_profile_active()

        for spec in iter_chunks(
            leading=leading,
            chunk_size=self.policy.solve_chunk_size,
            device=v_wl.device,
        ):
            mc = spec.multi_coords
            # Shape: [chunk, 1, row]
            v_wl_chunk = v_wl_full[mc] if mc else v_wl_full
            cell_snap = self.cell.snapshot(
                control=v_wl_chunk,
                shape=(*leading, *cell_trailing),
                multi_coords=mc,
                t_elapsed=0.0,
            )
            # Drivers snapshot at true inst trailing [..., serial, gn, polarity,
            # w_digit] (serial = leading[-1]); the solver consumes the flat
            # column axis.
            bl_snap = _flatten_lane_snap(
                bl_driver.snapshot(v_ref__V=bl_v_ref__V, shape=(*leading, *self._lane_shape), multi_coords=mc)
            )
            sl_snap = _flatten_lane_snap(
                sl_driver.snapshot(v_ref__V=sl_v_ref__V, shape=(*leading, *self._lane_shape), multi_coords=mc)
            )

            solver_dcop_chunk = self._solver.solve_dc(
                bl_segment_r__MOhm=self._bl_segment_r__MOhm,
                sl_segment_r__MOhm=self._sl_segment_r__MOhm,
                bl_segment_g__uS=self._bl_segment_g__uS,
                sl_segment_g__uS=self._sl_segment_g__uS,
                cell=self.cell,
                cell_snap=cell_snap,
                bl_driver=bl_driver,
                bl_driver_snap=bl_snap,
                sl_driver=sl_driver,
                sl_driver_snap=sl_snap,
            )
            if record_dynamic_energy:
                chunk_energy__fJ = self._compute_cap_energy__fJ(
                    solver_dcop=solver_dcop_chunk,
                    cell_snap=cell_snap,
                )
                chunk_energies.append(chunk_energy__fJ[: spec.valid_size] if leading else chunk_energy__fJ)
            i_bl_port_chunks.append(
                solver_dcop_chunk.i_bl_driver[: spec.valid_size] if leading else solver_dcop_chunk.i_bl_driver
            )
            v_bl_clamp_chunks.append(
                solver_dcop_chunk.v_bl_clamp[: spec.valid_size] if leading else solver_dcop_chunk.v_bl_clamp
            )
            global_indices.append(spec.flat_global_idx)

        # --- 4: reassemble the leading dimensions ---

        # Shape: [*leading, act]
        i_bl_port__uA = reassemble_chunks(i_bl_port_chunks, global_indices, leading, col_trailing)
        # Shape: [*leading, act]
        v_bl_clamp__V = reassemble_chunks(v_bl_clamp_chunks, global_indices, leading, col_trailing)

        # --- 5: record aggregate energy and latency ---

        serial_round_count = math.prod(leading[p] for p in a_positions) if a_positions else 1
        latency__ns = self._latency_per_op__ns * serial_round_count
        if record_dynamic_energy:
            # Per-activation BL/SL wire + cell node caps, serial included.
            # Shape: [*leading]
            array_energy__fJ = reassemble_chunks(chunk_energies, global_indices, leading, ())
            self._record_dynamic_energy(array_energy__fJ)
            # WL wire cap once per PLANE: the row input is held across all
            # serial slots, so this term is independent of `serial` and uses
            # the full physical column span.
            # Shape: [..., row] -> [...]
            e_wl_wire_cap__fJ = self._c_wl_wire_per_row__fF * v_wl.square().sum(dim=-1)
            self._record_dynamic_energy(e_wl_wire_cap__fJ)
        self._record_latency(latency__ns)
        return SerialColumnSteadyState(
            i_bl_port__uA=i_bl_port__uA.unflatten(-1, self._lane_shape),
            v_bl_clamp__V=v_bl_clamp__V.unflatten(-1, self._lane_shape),
        )

    def _compute_cap_energy__fJ(
        self,
        *,
        solver_dcop: SolverDcop[XbarCell1t1rDcop],
        cell_snap: XbarCell1t1rSnap,
    ) -> Tensor:
        """BL/SL wire-segment + cell node cap energy for one solved chunk.

        Full-cycle ``C·V²`` per solved (slot, lane) entry — the grounded-off
        convention (each column charges and discharges once per plane, in its
        slot). The per-cell node caps (BL / X / SL / WL gate) come from the
        cell (the cell computes, the array logs); across the slot partition
        every physical cell appears exactly once per plane, so the WL gate
        term is per-plane by construction. The WL WIRE cap is NOT here — it
        is billed once per plane in :meth:`solve_array`.

        Args:
            solver_dcop: Inner array solver's converged DCOP, carrying the
                BL / SL node voltages and the condensed cell DCOP.
            cell_snap: Per-solve cell snap bundling the device snaps and the
                WL control drive ``[..., 1, row_num]``.

        Returns:
            Cap energy [fJ], one entry per leading position, serial included.
            Shape: ``[...]``.
        """

        v_bl__V = solver_dcop.v_bl_node
        v_sl__V = solver_dcop.v_sl_node
        v_bl_clamp__V = solver_dcop.v_bl_clamp
        v_sl_drive__V = solver_dcop.v_sl_drive

        # --- 1: compute wire-capacitance energy ---

        # Shape: [..., act] -> [..., act, row_num]
        v_bl_left__V = torch.cat((v_bl_clamp__V.unsqueeze(-1), v_bl__V[..., :-1]), dim=-1)
        # Shape: [..., act] -> [..., act, row_num]
        v_sl_left__V = torch.cat((v_sl_drive__V.unsqueeze(-1), v_sl__V[..., :-1]), dim=-1)

        # Shape: [..., act, row_num] -> [...]
        bl_seg_q__V2 = (v_bl_left__V.square() + v_bl_left__V * v_bl__V + v_bl__V.square()) / 3.0
        e_bl_wire_cap__fJ = (self._bl_segment_c__fF * bl_seg_q__V2).sum(dim=(-2, -1))

        # Shape: [..., act, row_num] -> [...]
        sl_seg_q__V2 = (v_sl_left__V.square() + v_sl_left__V * v_sl__V + v_sl__V.square()) / 3.0
        e_sl_wire_cap__fJ = (self._sl_segment_c__fF * sl_seg_q__V2).sum(dim=(-2, -1))

        # --- 2: compute cell-capacitance energy ---

        # Shape: [..., act, row_num] -> [...]
        e_cell__fJ = self.cell.compute_dynamic_energy(v_bl__V, v_sl__V, solver_dcop.cell, cell_snap).sum(dim=(-2, -1))

        return e_bl_wire_cap__fJ + e_sl_wire_cap__fJ + e_cell__fJ
