"""Dedicated WH-2T1R array — a chunk-fused BL/SL divider solve and I_T2 lookup sum.

Geometry follows the solver convention ``[..., num_col, num_row]``: ``num_col``
bit-line columns carrying the per-column BL input voltages, ``num_row`` word-line
rows of which one is driven per solve.

See also:
    docs/works/macro/cim/ye2023jssc/model.md
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TypeVar

import torch
from torch import Tensor

from neurox.primitive.xbar.array import (
    XbarArray,
    XbarArray1t1rConfig,
    XbarArray1t1rPolicy,
    XbarArraySteadyState,
)
from neurox.primitive.xbar.cell import XbarCell1t1r, XbarCell1t1rDcop, XbarCell1t1rLinearConfig
from neurox.primitive.xbar.solver import (
    ClampDriver,
    NestedParallelRailSolver,
    SolverDcop,
    classify_leading_positions,
    iter_chunks,
    reassemble_chunks,
)
from neurox.primitive.xbar.solver.clamp import ClampSnap

# Import triggers the cell's registry registration so ``from_config`` dispatches.
from .cell import Ye2023Jssc2t1rCell, Ye2023Jssc2t1rCellSnap

BLSnapT = TypeVar("BLSnapT", bound=ClampSnap)
SLSnapT = TypeVar("SLSnapT", bound=ClampSnap)


class Ye2023Jssc2t1rArrayConfig(XbarArray1t1rConfig):
    """Physical knobs for the WH-2T1R dedicated array.

    Attributes:
        weight_radix: Per-plane place values of the weight-bearing planes,
            LSB-first (index 0 = the least significant plane). Non-empty; every
            entry a positive int.
        redundant_radix: Per-plane place values of the non-weight planes, laid
            out after the weight planes. Every entry a positive int.
        v_bl_in1__V: BL voltage driven for input bit 1; input bit 0 drives 0 V.
        v_bl_in_threshold__V: BL level above which a column counts as
            input-high, in ``(0, v_bl_in1__V)``.
    """

    weight_radix: tuple[int, ...]
    redundant_radix: tuple[int, ...]
    v_bl_in1__V: float
    v_bl_in_threshold__V: float

    def validate(self) -> None:
        super().validate()

        if len(self.weight_radix) == 0:
            raise ValueError("require: weight_radix must be non-empty")
        for plane, m in enumerate(self.weight_radix):
            if not (isinstance(m, int) and m > 0):
                raise ValueError(f"require: every weight_radix entry a positive int; got {m} at plane {plane}")
        for plane, m in enumerate(self.redundant_radix):
            if not (isinstance(m, int) and m > 0):
                raise ValueError(f"require: every redundant_radix entry a positive int; got {m} at plane {plane}")

        self._require_pos(self.v_bl_in1__V, "v_bl_in1__V")
        if not (0.0 < self.v_bl_in_threshold__V < self.v_bl_in1__V):
            raise ValueError(
                f"require: 0 < v_bl_in_threshold__V ({self.v_bl_in_threshold__V}) < v_bl_in1__V ({self.v_bl_in1__V})"
            )


class Ye2023Jssc2t1rArrayPolicy(XbarArray1t1rPolicy):
    """Nonideality policy for the WH-2T1R dedicated array; the scheme adds no fields."""


@dataclass(frozen=True)
class Ye2023Jssc2t1rSteadyState:
    """Reassembled steady-state output of the WH-2T1R array.

    Attributes:
        i_bl_port__uA: BL port (driver-boundary) current [uA] at the converged
            operating point.
            Shape: ``[..., num_col]``.
        v_bl_clamp__V: BL clamp voltage [V] at the converged operating point.
            Shape: ``[..., num_col]``.
        i_tbl__uA: Summed T2 compute current [uA] per leading instance, already
            reduced over columns and rows.
            Shape: ``[...]``.
    """

    i_bl_port__uA: Tensor
    v_bl_clamp__V: Tensor
    i_tbl__uA: Tensor


class Ye2023Jssc2t1rArray(XbarArray[Ye2023Jssc2t1rArrayConfig, Ye2023Jssc2t1rArrayPolicy]):
    """Dedicated WH-2T1R array: a fused BL/SL divider solve and I_T2 lookup sum.

    Args:
        config: WH-2T1R array configuration.
        policy: WH-2T1R array policy.
        inst_shape: Per-instance replication shape (prefix only).
        row_num: Number of word-line rows; > 1.
        col_num: Number of physical BL columns; > 1 and divisible by
            ``len(weight_radix) + len(redundant_radix)``, since the place values
            are laid out plane-major over equal column groups.
        dtype: Tensor dtype for internal buffers.
        T__K: Operating temperature.
        enable_latency_record: Whether ``solve`` emits a latency event.
    """

    # === Functional buffers ===

    _radix_per_col__unit: Tensor  # Shape: [col_num]

    # === Circuit constant buffers ===

    _bl_segment_r__MOhm: Tensor  # Shape: [row_num]
    _sl_segment_r__MOhm: Tensor  # Shape: [row_num]
    _bl_segment_g__uS: Tensor  # Shape: [row_num]
    _sl_segment_g__uS: Tensor  # Shape: [row_num]
    _latency_per_op__ns: Tensor  # Shape: []

    def __init__(
        self,
        *,
        config: Ye2023Jssc2t1rArrayConfig,
        policy: Ye2023Jssc2t1rArrayPolicy,
        inst_shape: tuple[int, ...],
        row_num: int,
        col_num: int,
        dtype: torch.dtype,
        T__K: float,
        enable_latency_record: bool = True,
    ) -> None:
        if not (col_num > 1):
            raise ValueError(f"require: col_num ({col_num}) > 1")
        if not (row_num > 1):
            raise ValueError(f"require: row_num ({row_num}) > 1")
        plane_num = len(config.weight_radix) + len(config.redundant_radix)
        if col_num % plane_num != 0:
            raise ValueError(
                f"require: col_num ({col_num}) % (len(weight_radix) + len(redundant_radix)) ({plane_num}) == 0"
            )

        super().__init__(
            config=config,
            policy=policy,
            inst_shape=inst_shape,
            enable_latency_record=enable_latency_record,
        )
        self._row_num = row_num
        self._col_num = col_num
        self._v_bl_in_threshold__V = config.v_bl_in_threshold__V
        cell_config = config.cell_config
        assert isinstance(cell_config, XbarCell1t1rLinearConfig)
        self._v_wl_on_threshold__V = cell_config.v_wl_on_threshold__V

        self._init_children(dtype=dtype, T__K=T__K)
        self._register_model_buffers(dtype=dtype)

        self._c_wl_wire_per_row__fF = config.wl_first_c__fF + (col_num - 1) * config.wl_segment_c__fF

    @property
    def _area_per_inst__um2(self) -> float:
        return self.config.area_per_inst__um2

    @property
    def _leakage_per_inst__uW(self) -> float:
        return self.config.leakage_per_inst__uW

    def _init_children(self, *, dtype: torch.dtype, T__K: float) -> None:
        """Construct the WH-2T1R lookup cell and the nested DC solver."""
        cell = XbarCell1t1r.from_config(
            config=self.config.cell_config,
            policy=self.policy.cell_policy,
            inst_shape=self.weight_grid_shape,
            dtype=dtype,
            T__K=T__K,
        )
        assert isinstance(cell, Ye2023Jssc2t1rCell)
        self.cell = cell
        self._solver = NestedParallelRailSolver(config=self.config.solver_config)

    def _register_model_buffers(self, *, dtype: torch.dtype) -> None:
        """Register fixed wire, place-value, and PPA tensors."""
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
        self.register_buffer("_bl_segment_r__MOhm", bl_segment_r__MOhm, persistent=False)
        self.register_buffer("_sl_segment_r__MOhm", sl_segment_r__MOhm, persistent=False)
        self.register_buffer("_bl_segment_g__uS", 1.0 / bl_segment_r__MOhm, persistent=False)
        self.register_buffer("_sl_segment_g__uS", 1.0 / sl_segment_r__MOhm, persistent=False)

        # Plane-major place values: each radix spans an equal group of columns.
        all_radix = (*config.weight_radix, *config.redundant_radix)
        radix_per_col = torch.tensor(all_radix, dtype=dtype).repeat_interleave(self._col_num // len(all_radix))
        self.register_buffer("_radix_per_col__unit", radix_per_col, persistent=False)

        self.register_buffer(
            "_latency_per_op__ns",
            torch.tensor(config.latency_per_op__ns, dtype=dtype),
            persistent=False,
        )

    @property
    def w_state_num(self) -> int:
        """Number of programmable states exposed by each cell."""
        return self.cell.w_state_num

    @property
    def weight_grid_shape(self) -> tuple[int, ...]:
        """Shape of the weight grid: ``(*inst_shape, col, row)``."""
        return (*self.inst_shape, self._col_num, self._row_num)

    def program(self, w_state_idx: Tensor) -> None:
        """Write the cells from one state-index tensor.

        Args:
            w_state_idx: State-index tensor in ``[0, w_state_num - 1]``; must
                match ``self.weight_grid_shape``.
                Shape: ``[*inst_shape, col_num, row_num]``.
        """
        if tuple(w_state_idx.shape) != self.weight_grid_shape:
            raise ValueError(
                f"program() expects w_state_idx.shape {self.weight_grid_shape}; got {tuple(w_state_idx.shape)}"
            )
        self.cell.program(w_state_idx)

    def _sample_fabricate_mismatch(self) -> None:
        """No local static state; the cell fabricates via the traversal."""

    def solve_array(
        self,
        v_wl: Tensor,
        *,
        bl_driver: ClampDriver[BLSnapT],
        bl_v_ref__V: Tensor,
        sl_driver: ClampDriver[SLSnapT],
        sl_v_ref__V: Tensor,
    ) -> XbarArraySteadyState:
        """Not supported; use :meth:`solve`, which returns the T2-current-bearing state."""
        raise NotImplementedError("Ye2023Jssc2t1rArray uses solve(), which returns Ye2023Jssc2t1rSteadyState")

    @torch.compiler.disable(
        recursive=False,
        reason="eager chunk loop; the fixed-shape per-chunk solver body is compiled separately",
    )
    def solve(
        self,
        v_wl: Tensor,
        *,
        bl_driver: ClampDriver[BLSnapT],
        bl_v_ref__V: Tensor,
        sl_driver: ClampDriver[SLSnapT],
        sl_v_ref__V: Tensor,
    ) -> Ye2023Jssc2t1rSteadyState:
        """Settle the WH-2T1R array to DC and compute the T2 lookup sum.

        The leading axes are solved in chunks of ``policy.solve_chunk_size``;
        every per-chunk intermediate, the internal node voltage included, is
        released with its chunk.

        Args:
            v_wl: Analog WL drive [V], shape ``[..., row_num]``; a row
                counts as selected above the cell's WL-on threshold.
            bl_driver: BL boundary clamp (structural ``ClampDriver`` role).
            bl_v_ref__V: PER-COLUMN BL input voltages [V], broadcastable to
                ``[..., num_col]``, laid out plane-major over
                ``weight_radix`` then ``redundant_radix``.
            sl_driver: SL boundary clamp (structural ``ClampDriver`` role).
            sl_v_ref__V: SL drive reference [V] (0-d scalar).

        Returns:
            :class:`Ye2023Jssc2t1rSteadyState` with the per-column BL port
            current, the BL clamp voltage, and the summed T2 current.
        """

        # --- 1: infer the broadcast-leading shape ---

        # Shape: [..., row] -> [..., 1, row]
        v_wl_grid = v_wl.unsqueeze(-2)
        # Shape: [..., num_col] -> [..., num_col, 1]
        bl_ref_grid = bl_v_ref__V.unsqueeze(-1) if bl_v_ref__V.ndim else bl_v_ref__V
        g_shape = self.weight_grid_shape
        full_shape = torch.broadcast_shapes(g_shape, tuple(v_wl_grid.shape), tuple(bl_ref_grid.shape))
        *batch_list, col_num, row_num = full_shape
        leading = tuple(batch_list)
        cell_trailing = (col_num, row_num)
        col_trailing = (col_num,)

        # --- 2: classify serial (output) and parallel leading positions ---

        a_positions, _b_positions = classify_leading_positions(
            x_shape=tuple(v_wl_grid.shape),
            g_shape=g_shape,
            leading_rank=len(leading),
        )

        # Shape: [..leading.., 1, row] -> [*leading, 1, row]
        v_wl_full = v_wl_grid.expand(*leading, 1, row_num)
        # Shape: [..leading.., num_col] -> [*leading, num_col]
        bl_ref_full = bl_v_ref__V.expand(*leading, col_num) if leading else bl_v_ref__V.expand(col_num)

        # --- 3: solve + measure + lookup each chunk ---

        i_bl_port_chunks: list[Tensor] = []
        v_bl_clamp_chunks: list[Tensor] = []
        i_tbl_chunks: list[Tensor] = []
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
            # Shape: [chunk, num_col]
            bl_ref_chunk = bl_ref_full[mc] if mc else bl_ref_full

            cell_snap = self.cell.snapshot(
                control=v_wl_chunk,
                shape=(*leading, *cell_trailing),
                multi_coords=mc,
                t_elapsed=0.0,
            )
            bl_snap = bl_driver.snapshot(v_ref__V=bl_v_ref__V, shape=(*leading, *col_trailing), multi_coords=mc)
            sl_snap = sl_driver.snapshot(v_ref__V=sl_v_ref__V, shape=(*leading, *col_trailing), multi_coords=mc)

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

            # Place-value-weighted I_T2 lookup, keyed on (input bit, weight
            # state) alone: it reads no solved quantity of the divider solve.
            # Shape: [chunk, num_col]
            input_high = bl_ref_chunk > self._v_bl_in_threshold__V
            # Shape: [chunk, num_col, row]
            i_t2_unit__uA = self.cell.lookup_i_t2(input_high.unsqueeze(-1), cell_snap)
            i_t2_scaled__uA = i_t2_unit__uA * self._radix_per_col__unit.view(-1, 1)
            # The WL-on mask selects the driven row; sum over col AND row.
            # Shape: [chunk, 1, row]
            wl_on = v_wl_chunk > self._v_wl_on_threshold__V
            # Shape: [chunk]
            i_tbl_chunk__uA = (i_t2_scaled__uA * wl_on).sum(dim=(-2, -1))

            if record_dynamic_energy:
                chunk_energy__fJ = self._compute_array_energy__fJ(
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
            i_tbl_chunks.append(i_tbl_chunk__uA[: spec.valid_size] if leading else i_tbl_chunk__uA)
            global_indices.append(spec.flat_global_idx)

        # --- 4: reassemble the leading dimensions ---

        # Shape: [*leading, num_col]
        i_bl_port__uA = reassemble_chunks(i_bl_port_chunks, global_indices, leading, col_trailing)
        # Shape: [*leading, num_col]
        v_bl_clamp__V = reassemble_chunks(v_bl_clamp_chunks, global_indices, leading, col_trailing)
        # Shape: [*leading]
        i_tbl__uA = reassemble_chunks(i_tbl_chunks, global_indices, leading, ())

        # --- 5: record aggregate energy (per-access caps) and optional latency ---

        if record_dynamic_energy:
            # Shape: [*leading]
            array_energy__fJ = reassemble_chunks(chunk_energies, global_indices, leading, ())
            self._record_dynamic_energy(array_energy__fJ)
        if self.enable_latency_record:
            serial_round_count = math.prod(leading[p] for p in a_positions) if a_positions else 1
            self._record_latency(self._latency_per_op__ns * serial_round_count)

        return Ye2023Jssc2t1rSteadyState(
            i_bl_port__uA=i_bl_port__uA,
            v_bl_clamp__V=v_bl_clamp__V,
            i_tbl__uA=i_tbl__uA,
        )

    def _compute_array_energy__fJ(
        self,
        *,
        solver_dcop: SolverDcop[XbarCell1t1rDcop],
        cell_snap: Ye2023Jssc2t1rCellSnap,
    ) -> Tensor:
        """Compute the PER-ACCESS array capacitive energy for one solve.

        Sums the selected row's WL wire segments and the per-cell terms the cell
        model owns.

        Args:
            solver_dcop: Inner array solver's converged DCOP.
            cell_snap: Per-solve cell snap bundling the device snaps and the WL
                control drive ``[..., 1, row]``.

        Returns:
            Array energy [fJ], shape ``[...]``.
        """
        e_wl_wire_cap__fJ = (self._c_wl_wire_per_row__fF * cell_snap.v_wl__V.square()).sum(dim=(-2, -1))
        e_cell__fJ = self.cell.compute_dynamic_energy(
            solver_dcop.v_bl_node,
            solver_dcop.v_sl_node,
            solver_dcop.cell,
            cell_snap,
        ).sum(dim=(-2, -1))

        return e_wl_wire_cap__fJ + e_cell__fJ
