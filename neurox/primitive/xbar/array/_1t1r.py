"""Shape-independent pure-array core for a 1T1R crossbar tile.

See also:
    docs/reference/primitive/xbar/array/_1t1r/array.md
"""

import math
from dataclasses import dataclass
from typing import TypeVar

import torch
from torch import Tensor

from neurox.primitive.xbar.array.base import XbarArray, XbarArrayConfig, XbarArrayPolicy
from neurox.primitive.xbar.cell import (
    XbarCell1T1R,
    XbarCell1T1RConfig,
    XbarCell1T1RDCOP,
    XbarCell1T1RPolicy,
    XbarCell1T1RSnap,
)
from neurox.primitive.xbar.solver import (
    ClampDriver,
    Solver,
    SolverConfig,
    SolverDCOP,
    classify_leading_positions,
    iter_chunks,
    reassemble_chunks,
)
from neurox.primitive.xbar.solver.clamp import ClampSnap

BLSnapT = TypeVar("BLSnapT", bound=ClampSnap)
SLSnapT = TypeVar("SLSnapT", bound=ClampSnap)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class XbarArray1T1RConfig(XbarArrayConfig):
    """Shape-independent physical knobs for a 1T1R pure-array core.

    Attributes:
        wl_pulse_length__ns: Word-line pulse length.
        row_first_space__um: Row pitch from the driver to the first cell.
        row_cell_space__um: Row pitch between adjacent cells.
        col_first_space__um: Column pitch from the driver to the first cell.
        col_cell_space__um: Column pitch between adjacent cells.
        bl_first_r__MOhm: BL driver-to-first-cell segment resistance.
        bl_first_c__fF: BL driver-to-first-cell segment capacitance.
        bl_segment_r__MOhm: BL cell-to-cell segment resistance.
        bl_segment_c__fF: BL cell-to-cell segment capacitance.
        sl_first_r__MOhm: SL driver-to-first-cell segment resistance.
        sl_first_c__fF: SL driver-to-first-cell segment capacitance.
        sl_segment_r__MOhm: SL cell-to-cell segment resistance.
        sl_segment_c__fF: SL cell-to-cell segment capacitance.
        wl_first_r__MOhm: WL driver-to-first-cell segment resistance.
        wl_first_c__fF: WL driver-to-first-cell segment capacitance.
        wl_segment_r__MOhm: WL cell-to-cell segment resistance.
        wl_segment_c__fF: WL cell-to-cell segment capacitance.
        cell_config: 1T1R cell configuration. Owns the RRAM / access-NMOS
            device configs, sizing, parasitic-cap densities, programming
            map, and per-cell branch-solve knobs.
        solver_config: DC-solver fixed numerical knobs. Concrete subclass
            of :class:`SolverConfig` picks which solver implementation the
            core instantiates.
        area_per_inst__um2: Array (cell array + wire infra) silicon area
            per fabricated tile instance. Device-side (RRAM / NMOS) area
            must be folded into this field by the caller.
        leakage_per_inst__uW: Array static leakage per fabricated tile
            instance. Same scope as ``area_per_inst__um2``.
        latency_per_op__ns: Array-side per-VMM latency that the
            profiler attributes the dynamic-energy event to.
    """

    wl_pulse_length__ns: float

    row_first_space__um: float
    row_cell_space__um: float
    col_first_space__um: float
    col_cell_space__um: float

    bl_first_r__MOhm: float
    bl_first_c__fF: float
    bl_segment_r__MOhm: float
    bl_segment_c__fF: float

    sl_first_r__MOhm: float
    sl_first_c__fF: float
    sl_segment_r__MOhm: float
    sl_segment_c__fF: float

    wl_first_r__MOhm: float
    wl_first_c__fF: float
    wl_segment_r__MOhm: float
    wl_segment_c__fF: float

    cell_config: XbarCell1T1RConfig
    solver_config: SolverConfig

    latency_per_op__ns: float

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        self.validate_wl_pulse()
        self.validate_layout_pitch()
        self.validate_wire_segments()
        self.validate_ppa()

    def validate_ppa(self) -> None:
        super().validate_ppa()
        self._require_non_neg(self.latency_per_op__ns, "latency_per_op__ns")

    def validate_wl_pulse(self) -> None:
        self._require_non_neg(self.wl_pulse_length__ns, "wl_pulse_length__ns")

    def validate_layout_pitch(self) -> None:
        for field in (
            "row_first_space__um",
            "row_cell_space__um",
            "col_first_space__um",
            "col_cell_space__um",
        ):
            self._require_pos(getattr(self, field), field)

    def validate_wire_segments(self) -> None:
        for field in (
            "bl_first_r__MOhm",
            "bl_first_c__fF",
            "bl_segment_r__MOhm",
            "bl_segment_c__fF",
            "sl_first_r__MOhm",
            "sl_first_c__fF",
            "sl_segment_r__MOhm",
            "sl_segment_c__fF",
            "wl_first_r__MOhm",
            "wl_first_c__fF",
            "wl_segment_r__MOhm",
            "wl_segment_c__fF",
        ):
            self._require_pos(getattr(self, field), field)


# ---------------------------------------------------------------------------
# Nonideality policy
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class XbarArray1T1RPolicy(XbarArrayPolicy):
    """Composite nonideality policy for a 1T1R pure-array core.

    Attributes:
        cell: 1T1R cell nonideality policy (RRAM + access-NMOS).
        solve_chunk_size: Maximum number of broadcast-leading instances
            ``solve_array`` solves per chunk — the per-chunk peak-memory
            budget. ``0`` runs the whole leading in one block; any
            positive value forces the memory-bounded chunked path,
            splitting the leading into contiguous slices of at most this
            many instances. Runtime knob, not a chip-preset constant.
    """

    cell: XbarCell1T1RPolicy
    solve_chunk_size: int


# ---------------------------------------------------------------------------
# Array steady-state
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class XbarArraySteadyState:
    """Reassembled steady-state array output consumed by the xbar readout.

    Attributes:
        i_bl_port__uA: BL port current at the converged operating point.
            Shape: ``[..., num_line]``.
        v_bl_clamp__V: BL clamp voltage at the converged operating point,
            a warm-start seed for the xbar's I→V readout solve.
            Shape: ``[..., num_line]``.
    """

    i_bl_port__uA: Tensor
    v_bl_clamp__V: Tensor


# ---------------------------------------------------------------------------
# Array
# ---------------------------------------------------------------------------


class XbarArray1T1R(XbarArray[XbarArray1T1RConfig, XbarArray1T1RPolicy]):
    """Shape-independent 1T1R pure array: cells, wire parasitics, and solver."""

    config: XbarArray1T1RConfig
    cell: XbarCell1T1R
    bl_segment_r__MOhm: Tensor
    sl_segment_r__MOhm: Tensor
    bl_segment_g__uS: Tensor
    sl_segment_g__uS: Tensor
    bl_segment_c__fF: Tensor
    sl_segment_c__fF: Tensor

    def __init__(
        self,
        *,
        config: XbarArray1T1RConfig,
        policy: XbarArray1T1RPolicy,
        name: str,
        w_layout_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
    ) -> None:
        """Construct one shape-independent 1T1R pure-array core.

        Args:
            config: Concrete configuration dataclass.
            policy: Composite nonideality policy.
            name: Hierarchical instance name used by the profiler.
            w_layout_shape: Per-instance state-index tensor shape
                ``(*prefix, phys_col_num, row_num)`` that
                ``program(...)`` will receive.
            dtype: Tensor dtype for internal buffers.
            T__K: Operating temperature.
        """
        if len(w_layout_shape) < 2:
            raise ValueError(
                f"w_layout_shape must have at least 2 trailing dims (phys_col_num, row_num); got {w_layout_shape}"
            )
        *prefix, phys_col_num, row_num = w_layout_shape
        if not (phys_col_num > 1):
            raise ValueError(f"require: phys_col_num ({phys_col_num}) > 1")
        if not (row_num > 1):
            raise ValueError(f"require: row_num ({row_num}) > 1")

        super().__init__(config=config, policy=policy, name=name, inst_shape=tuple(prefix))
        self._area_per_inst__um2 = config.area_per_inst__um2
        self._leakage_per_inst__uW = config.leakage_per_inst__uW
        self.dtype = dtype
        self.T__K = T__K
        self._w_layout_shape = tuple(w_layout_shape)

        self.cell = XbarCell1T1R(
            config=config.cell_config,
            policy=policy.cell,
            inst_shape=self._w_layout_shape,
            dtype=dtype,
            T__K=T__K,
        )

        self.w_states = self.cell.w_states

        self.c_wl_wire_per_row__fF = config.wl_first_c__fF + (phys_col_num - 1) * config.wl_segment_c__fF

        # Per-line segment buffers; index 0 is the driver-to-first segment.
        bl_segment_r__MOhm = torch.tensor(
            [config.bl_first_r__MOhm] + [config.bl_segment_r__MOhm] * (row_num - 1),
            dtype=dtype,
        )
        sl_segment_r__MOhm = torch.tensor(
            [config.sl_first_r__MOhm] + [config.sl_segment_r__MOhm] * (row_num - 1),
            dtype=dtype,
        )
        bl_segment_c__fF = torch.tensor(
            [config.bl_first_c__fF] + [config.bl_segment_c__fF] * (row_num - 1),
            dtype=dtype,
        )
        sl_segment_c__fF = torch.tensor(
            [config.sl_first_c__fF] + [config.sl_segment_c__fF] * (row_num - 1),
            dtype=dtype,
        )
        self.register_buffer("bl_segment_r__MOhm", bl_segment_r__MOhm, persistent=False)
        self.register_buffer("sl_segment_r__MOhm", sl_segment_r__MOhm, persistent=False)
        self.register_buffer("bl_segment_g__uS", 1.0 / bl_segment_r__MOhm, persistent=False)
        self.register_buffer("sl_segment_g__uS", 1.0 / sl_segment_r__MOhm, persistent=False)
        self.register_buffer("bl_segment_c__fF", bl_segment_c__fF, persistent=False)
        self.register_buffer("sl_segment_c__fF", sl_segment_c__fF, persistent=False)

        # This 1T1R grid is [..., col, row] with the wire ladder along the
        # last (row) axis → canonical series-last (series_axis = -1).
        self.solver = Solver.from_config(config=config.solver_config, series_axis=-1)

        self.fabricated_col_num = phys_col_num
        self.fabricated_row_num = row_num

    # -----------------------------------------------------------------
    # Geometry
    # -----------------------------------------------------------------

    @property
    def weight_grid_shape(self) -> tuple[int, ...]:
        """Shape of the weight grid (RRAM conductance array): ``(*inst, phys_col, row)``."""
        return tuple(self.cell.rram.g__uS.shape)

    # -----------------------------------------------------------------
    # Programming
    # -----------------------------------------------------------------

    def program(self, w_state_idx: Tensor) -> None:
        """Write the cells from one state-index tensor.

        Args:
            w_state_idx: State-index tensor in ``[0, w_states - 1]``,
                shape must match ``self._w_layout_shape =
                (*prefix, phys_col_num, row_num)``.
        """
        if tuple(w_state_idx.shape) != self._w_layout_shape:
            raise ValueError(
                f"program() expects w_state_idx.shape {self._w_layout_shape}; got {tuple(w_state_idx.shape)}"
            )
        self.cell.program(w_state_idx)

    # -----------------------------------------------------------------
    # Array steady-state solve (plain forward)
    # -----------------------------------------------------------------

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
    ) -> XbarArraySteadyState:
        """Settle the 1T1R array to DC under an analog WL drive.

        Settle the array boundary clamps to DC in chunked Newton
        sub-solves, accumulate per-VMM dynamic energy, emit one energy +
        one latency profile event, and return the reassembled BL port
        current + BL clamp voltage the xbar's I→V readout consumes.

        Args:
            v_wl: Analog WL drive [V] (the xbar already ran the WL DAC).
                Shape: ``[..., row_num]``.
            bl_driver: BL boundary clamp (structural ``ClampDriver`` role).
            bl_v_ref__V: BL-clamp reference tap, a 0-d scalar the xbar
                snapshotted once and broadcasts onto every chunk grid.
            sl_driver: SL boundary clamp (structural ``ClampDriver`` role).
            sl_v_ref__V: SL-drive reference tap, a 0-d scalar.

        Returns:
            :class:`XbarArraySteadyState` carrying the per-column BL port
            current [uA] and BL clamp voltage [V], both at full leading.
        """

        # --- Infer the broadcast leading ---

        # ``v_wl`` trailing is ``[row]``; ``v_wl_grid`` adds a size-1
        # WL-fanout dim at -2 so ``v_wl``'s ``row`` aligns with ``g``'s
        # ``row`` and the ``phys_col`` slot opens for the solver-side
        # broadcast against the RRAM grid.
        v_wl_grid = v_wl.unsqueeze(-2)
        g_shape = self.cell.rram.g__uS.shape
        full_shape = torch.broadcast_shapes(g_shape, v_wl_grid.shape)
        *batch_list, phys_col_num, row_num = full_shape
        leading = tuple(batch_list)
        cell_trailing = (phys_col_num, row_num)
        line_trailing = (phys_col_num,)

        # --- Classify leading positions (for the serial latency count) ---

        a_positions, _b_positions = classify_leading_positions(
            x_shape=tuple(v_wl_grid.shape),
            g_shape=tuple(g_shape),
            leading_rank=len(leading),
        )

        # Broadcast the WL drive to the full leading so the solver sees
        # ``(*leading, 1, row)``.
        v_wl_full = v_wl_grid.expand(*leading, 1, row_num)

        # --- Per-chunk loop: sample → solve → energy ---

        i_bl_port_chunks: list[Tensor] = []
        v_bl_clamp_chunks: list[Tensor] = []
        chunk_energies: list[Tensor] = []
        global_indices: list[Tensor] = []

        for spec in iter_chunks(
            leading=leading,
            chunk_size=self.policy.solve_chunk_size,
            device=v_wl.device,
        ):
            mc = spec.multi_coords
            v_wl_chunk = v_wl_full[mc] if mc else v_wl_full  # (chunk_size, 1, row)
            cell_snap = self.cell.snapshot(
                control=v_wl_chunk,
                shape=(*leading, *cell_trailing),
                multi_coords=mc,
                t_elapsed=0.0,
            )
            bl_snap = bl_driver.snapshot(v_ref__V=bl_v_ref__V, shape=(*leading, *line_trailing), multi_coords=mc)
            sl_snap = sl_driver.snapshot(v_ref__V=sl_v_ref__V, shape=(*leading, *line_trailing), multi_coords=mc)

            solver_dcop_chunk = self.solver.solve_dc(
                bl_segment_r__MOhm=self.bl_segment_r__MOhm,
                sl_segment_r__MOhm=self.sl_segment_r__MOhm,
                bl_segment_g__uS=self.bl_segment_g__uS,
                sl_segment_g__uS=self.sl_segment_g__uS,
                cell=self.cell,
                cell_snap=cell_snap,
                bl_driver=bl_driver,
                bl_driver_snap=bl_snap,
                sl_driver=sl_driver,
                sl_driver_snap=sl_snap,
                compute_residuals=False,
            )
            chunk_energies.append(
                self._compute_array_energy__fJ(
                    solver_dcop=solver_dcop_chunk,
                    cell_snap=cell_snap,
                )
            )
            i_bl_port_chunks.append(solver_dcop_chunk.i_bl_driver)
            v_bl_clamp_chunks.append(solver_dcop_chunk.v_bl_clamp)
            global_indices.append(spec.flat_global_idx)

        # --- Reassemble outputs ---

        i_bl_port__uA = reassemble_chunks(i_bl_port_chunks, global_indices, leading, line_trailing)
        v_bl_clamp__V = reassemble_chunks(v_bl_clamp_chunks, global_indices, leading, line_trailing)
        array_energy__fJ = reassemble_chunks(chunk_energies, global_indices, leading, ())

        # --- Emit one energy + one latency event for this VMM ---

        # Serial is the x-side broadcast (a_positions); the inst-side
        # (b_positions) is parallel physical hardware and must not enter
        # the per-op-latency serial count.
        serial_op_count = math.prod(leading[p] for p in a_positions) if a_positions else 1
        latency__ns = torch.tensor(
            self.config.latency_per_op__ns * serial_op_count,
            device=array_energy__fJ.device,
            dtype=array_energy__fJ.dtype,
        )
        self._log_dynamic_energy(array_energy__fJ)
        self._log_latency(latency__ns)
        return XbarArraySteadyState(i_bl_port__uA=i_bl_port__uA, v_bl_clamp__V=v_bl_clamp__V)

    # -----------------------------------------------------------------
    # Dynamic-energy aggregation
    # -----------------------------------------------------------------

    def _compute_array_energy__fJ(
        self,
        *,
        solver_dcop: SolverDCOP[XbarCell1T1RDCOP],
        cell_snap: XbarCell1T1RSnap,
    ) -> Tensor:
        """Per-VMM array-internal energy. Shape: [...batch...].

        Sums the core-owned terms — DC conduction at the rail clamps plus
        wire-segment (BL / SL) and WL-line capacitive cycling — with the
        per-cell device-capacitance switching energy delegated to
        :meth:`XbarCell.dynamic_energy`.

        Args:
            solver_dcop: Inner array solver's converged DCOP, carrying the
                BL / SL node voltages, the condensed cell DCOP, and the
                per-column port currents.
            cell_snap: Per-solve cell snap bundling the device
                snaps and the WL control drive ``[..., 1, row_num]``.
        """

        # WL drive recovered from the cell snap; drop the WL-fanout
        # slot so the WL-wire term sums to ``[...]``.
        v_wl__V = cell_snap.v_wl__V.squeeze(-2)
        v_bl__V = solver_dcop.v_bl_node
        v_sl__V = solver_dcop.v_sl_node
        v_bl_clamp__V = solver_dcop.v_bl_clamp
        v_sl_drive__V = solver_dcop.v_sl_drive
        pulse__ns = self.config.wl_pulse_length__ns

        # --- DC conduction ---

        # Shape: [..., phys_col_num] -> [...]
        array_power__uW = (v_bl_clamp__V * solver_dcop.i_bl_driver).sum(dim=-1) + (
            v_sl_drive__V * solver_dcop.i_sl_driver
        ).sum(dim=-1)
        e_dc_cond__fJ = array_power__uW * pulse__ns

        # --- Capacitive cycling ---

        # Shape: [..., phys_col_num] -> [..., phys_col_num, row_num]
        v_bl_left__V = torch.cat((v_bl_clamp__V.unsqueeze(-1), v_bl__V[..., :-1]), dim=-1)
        # Shape: [..., phys_col_num] -> [..., phys_col_num, row_num]
        v_sl_left__V = torch.cat((v_sl_drive__V.unsqueeze(-1), v_sl__V[..., :-1]), dim=-1)

        # Shape: [..., row_num] -> [...]
        e_wl_wire_cap__fJ = (self.c_wl_wire_per_row__fF * v_wl__V.square()).sum(dim=-1)

        # Shape: [..., phys_col_num, row_num] -> [...]
        bl_seg_q__V2 = (v_bl_left__V.square() + v_bl_left__V * v_bl__V + v_bl__V.square()) / 3.0
        e_bl_wire_cap__fJ = (self.bl_segment_c__fF * bl_seg_q__V2).sum(dim=(-2, -1))

        # Shape: [..., phys_col_num, row_num] -> [...]
        sl_seg_q__V2 = (v_sl_left__V.square() + v_sl_left__V * v_sl__V + v_sl__V.square()) / 3.0
        e_sl_wire_cap__fJ = (self.sl_segment_c__fF * sl_seg_q__V2).sum(dim=(-2, -1))

        # --- Per-cell device-capacitance switching energy ---

        # Shape: [..., phys_col_num, row_num] -> [...]
        e_cell__fJ = self.cell.dynamic_energy(v_bl__V, v_sl__V, solver_dcop.cell, cell_snap).sum(dim=(-2, -1))

        return e_dc_cond__fJ + e_wl_wire_cap__fJ + e_bl_wire_cap__fJ + e_sl_wire_cap__fJ + e_cell__fJ
