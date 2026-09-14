"""Shape-independent 1T1R array with wire parasitics and a DC solver.

See Also:
    docs/reference/primitive/xbar/array/1t1r.md
"""

from __future__ import annotations

from typing import ClassVar

import torch
from torch import Tensor

from neurox.common.dataclass_mixin import PyTreeDataClassMixin, TensorDataClassMixin
from neurox.common.module import ConfigBase, DcopBase, ModuleBase, PolicyBase
from neurox.execution.chunking import run_chunked
from neurox.primitive.physics import e_cap_excursion__fJ
from neurox.primitive.xbar.cell import (
    XbarCell1t1r,
    XbarCell1t1rConfig,
    XbarCell1t1rDcop,
    XbarCell1t1rPolicy,
    XbarCell1t1rSnap,
)
from neurox.primitive.xbar.solver import (
    ClampDcop,
    ClampDriver,
    ClampSnap,
    ColBlColSlArraySolver,
    ColBlColSlArrayState,
    ColBlColSlArrayTrace,
)


class XbarArray1t1rConfig(ConfigBase):
    row_cell_space__um: float
    col_cell_space__um: float

    bl_segment_r__MOhm: float
    sl_segment_r__MOhm: float

    bl_node_c__fF: float
    """Total capacitance to ground seen at each cell's BL node."""
    x_node_c__fF: float
    """Total capacitance to ground seen at each cell's internal access node X."""
    sl_node_c__fF: float
    """Total capacitance to ground seen at each cell's SL node."""
    wl_node_c__fF: float
    """Total capacitance to ground seen at each cell's WL node."""

    cell_config: XbarCell1t1rConfig
    """Its concrete subclass selects the cell model."""

    def validate(self) -> None:
        super().validate()

        # --- Layout pitch ---

        self._require_pos(self.row_cell_space__um, "row_cell_space__um")
        self._require_pos(self.col_cell_space__um, "col_cell_space__um")

        # --- Rail links ---

        self._require_pos(self.bl_segment_r__MOhm, "bl_segment_r__MOhm")
        self._require_pos(self.sl_segment_r__MOhm, "sl_segment_r__MOhm")

        # --- Node capacitance ---

        self._require_non_neg(self.bl_node_c__fF, "bl_node_c__fF")
        self._require_non_neg(self.x_node_c__fF, "x_node_c__fF")
        self._require_non_neg(self.sl_node_c__fF, "sl_node_c__fF")
        self._require_non_neg(self.wl_node_c__fF, "wl_node_c__fF")


class XbarArray1t1rPolicy(PolicyBase):
    cell_policy: XbarCell1t1rPolicy
    """Its concrete subclass matches the configured cell model."""
    solve_chunk_size: int
    """Maximum leading positions per chunk; `0` solves all positions in one chunk."""

    def validate(self) -> None:
        super().validate()
        self._require_non_neg(self.solve_chunk_size, "solve_chunk_size")


class XbarArray1t1rDcop(DcopBase, PyTreeDataClassMixin):
    i_bl_port__uA: Tensor
    """Current sourced from the BL driver into the array.
    Shape: `[..., row=1, col]`."""
    v_bl_port__V: Tensor
    """Shape: `[..., row=1, col]`."""
    i_sl_port__uA: Tensor
    """Current sourced from the SL driver into the array.
    Shape: `[..., row=1, col]`."""
    v_sl_port__V: Tensor
    """Shape: `[..., row=1, col]`."""


_Config = XbarArray1t1rConfig
_Policy = XbarArray1t1rPolicy
_Dcop = XbarArray1t1rDcop
_State = ColBlColSlArrayState
_Trace = ColBlColSlArrayTrace
_CellSnap = XbarCell1t1rSnap
_CellDcop = XbarCell1t1rDcop


class _Inputs[BLSnapT: ClampSnap, SLSnapT: ClampSnap](TensorDataClassMixin):
    cell_snap: _CellSnap
    bl_driver_snap: BLSnapT
    sl_driver_snap: SLSnapT


class _Outputs(TensorDataClassMixin, PyTreeDataClassMixin):
    dcop: _Dcop
    energy__fJ: Tensor | None
    trace: _Trace | None


class XbarArray1t1r[BLSnapT: ClampSnap, SLSnapT: ClampSnap](ModuleBase):
    """Shape-independent 1T1R array with wire parasitics and a DC solver.

    The final two grid axes follow the solver's `row_dim` and `col_dim`,
    in row, column order. Boundary snaps and DCOPs retain the row axis at
    extent one. Chunking applies only to the leading positions.

    Args:
        vdd__V: Core analog supply behind every array-node capacitance.
        bl_driver: Externally owned BL clamp, forwarded to the solver without
            registering it as an array child. Its owner handles device placement,
            fabrication, snapshot sampling, and energy accounting.
        sl_driver: Externally owned SL clamp, with the same ownership contract.
    """

    row_dim: ClassVar[int] = ColBlColSlArraySolver.row_dim
    col_dim: ClassVar[int] = ColBlColSlArraySolver.col_dim

    config: _Config
    policy: _Policy

    def __init__(
        self,
        *,
        config: _Config,
        policy: _Policy,
        inst_shape: tuple[int, ...],
        row_num: int,
        col_num: int,
        vdd__V: float,
        bl_driver: ClampDriver[BLSnapT, ClampDcop],
        sl_driver: ClampDriver[SLSnapT, ClampDcop],
        dtype: torch.dtype,
    ) -> None:
        super().__init__(config=config, policy=policy, inst_shape=inst_shape)
        self._row_num = row_num
        self._col_num = col_num
        self._vdd__V = vdd__V

        grid_shape = [0, 0]
        grid_shape[self.row_dim] = row_num
        grid_shape[self.col_dim] = col_num
        self._grid_shape = tuple(grid_shape)

        self._bl_estab_total_c__fF = row_num * (config.bl_node_c__fF + config.x_node_c__fF)
        self._sl_estab_total_c__fF = row_num * config.sl_node_c__fF

        self._init_children(dtype=dtype)
        self.solver = ColBlColSlArraySolver(
            bl_segment_r__MOhm=config.bl_segment_r__MOhm,
            sl_segment_r__MOhm=config.sl_segment_r__MOhm,
            cell=self.cell,
            bl_driver=bl_driver,
            sl_driver=sl_driver,
            dtype=dtype,
        )

    @property
    def _area_per_inst__um2(self) -> float:
        config = self.config
        return self._row_num * config.row_cell_space__um * self._col_num * config.col_cell_space__um

    @property
    def _leakage_per_inst__uW(self) -> float:
        # Every operation returns to zero cell bias, so the array holds no
        # static conduction path between operations.
        return 0.0

    def _init_children(self, *, dtype: torch.dtype) -> None:
        self.cell = XbarCell1t1r.from_config(
            config=self.config.cell_config,
            policy=self.policy.cell_policy,
            # Shape: [*inst_shape, row, col]
            inst_shape=(*self.inst_shape, *self._grid_shape),
            dtype=dtype,
        )

    @property
    def w_state_num(self) -> int:
        """Number of programmable states exposed by each cell."""
        return self.cell.w_state_num

    @torch.no_grad()
    def program(self, w_state_idx: Tensor) -> None:
        """Write the cells from one state-index tensor.

        Entry `(row, col)` programs the cell at that physical intersection.

        Args:
            w_state_idx: State-index tensor in `[0, w_state_num - 1]`.
                Shape: `[*inst_shape, row, col]`.
        """
        expected_shape = self.cell.inst_shape
        if tuple(w_state_idx.shape) != expected_shape:
            raise ValueError(f"program() expects w_state_idx.shape {expected_shape}; got {tuple(w_state_idx.shape)}")
        self.cell.program(w_state_idx)

    @torch.no_grad()
    def solve_dc(
        self,
        *,
        v_wl__V: Tensor,
        leading_shape: tuple[int, ...],
        wl_phase_dims: tuple[int, ...],
        bl_driver_snap: BLSnapT,
        sl_driver_snap: SLSnapT,
    ) -> _Dcop:
        """Return the converged array DC operating point.

        One or more leading axes group the WL phases that share a held BL/SL rest
        boundary. The caller supplies both grid axes in the array's axis order,
        with a singleton column axis for WL. The array broadcasts each row drive
        across its columns before snapshotting the cells; both boundary snaps carry the same
        leading and repeat their zero-load voltages along the phase axes.

        Args:
            v_wl__V: Analog drive, one value per word line. Its leading axes
                broadcast to `leading_shape`.
                Shape: `[..., row, col=1]`.
            leading_shape: Complete per-call leading shape, including WL phase axes.
            wl_phase_dims: Axes of the full `[*leading_shape, row, col]` layout whose Cartesian product contains
                the WL phases under one held rest boundary. Every axis must be
                leading rather than either of the final two grid axes.
            bl_driver_snap: Per-solve BL clamp snap at the full per-call
                shape `[..., row=1, col]`; its `v_open__V` is also the zero-load BL rest level.
            sl_driver_snap: Per-solve SL clamp snap at the full per-call
                shape `[..., row=1, col]`; its `v_open__V` is also the zero-load SL rest level.

        Returns:
            The array's converged DC operating point at both boundaries.

        Raises:
            ValueError: The WL grid shape or phase axes violate the input contract.
            RuntimeError: A numerical state is invalid, or an internal update
                cap is reached while a position remains unconverged.
        """
        dcop, _ = self._solve_dc_impl(
            v_wl__V=v_wl__V,
            leading_shape=leading_shape,
            wl_phase_dims=wl_phase_dims,
            bl_driver_snap=bl_driver_snap,
            sl_driver_snap=sl_driver_snap,
            record_trace=False,
        )
        return dcop

    @torch.no_grad()
    def solve_dc_trace(
        self,
        *,
        v_wl__V: Tensor,
        leading_shape: tuple[int, ...],
        wl_phase_dims: tuple[int, ...],
        bl_driver_snap: BLSnapT,
        sl_driver_snap: SLSnapT,
    ) -> tuple[_Dcop, _Trace]:
        """Settle the array while retaining raw convergence traces.

        Input shapes and phase axes follow `solve_dc`.
        The returned DC operating point may be computed from a capped
        unconverged solver state; the trace reports its convergence status.
        """
        dcop, trace = self._solve_dc_impl(
            v_wl__V=v_wl__V,
            leading_shape=leading_shape,
            wl_phase_dims=wl_phase_dims,
            bl_driver_snap=bl_driver_snap,
            sl_driver_snap=sl_driver_snap,
            record_trace=True,
        )
        if trace is None:
            raise RuntimeError("A traced array solve returned no trace")
        return dcop, trace

    def _solve_dc_impl(
        self,
        *,
        v_wl__V: Tensor,
        leading_shape: tuple[int, ...],
        wl_phase_dims: tuple[int, ...],
        bl_driver_snap: BLSnapT,
        sl_driver_snap: SLSnapT,
        record_trace: bool,
    ) -> tuple[_Dcop, _Trace | None]:
        record_energy = self._is_dynamic_energy_profile_active()
        row_num = self._row_num
        row_dim = self.row_dim
        col_dim = self.col_dim

        # --- 1: resolve phase axes in the complete solve layout ---

        if not wl_phase_dims:
            raise ValueError("wl_phase_dims must name at least one leading axis")
        solve_ndim = len(leading_shape) + 2
        wl_phase_dims = tuple(dim + solve_ndim if dim < 0 else dim for dim in wl_phase_dims)
        if any(dim < 0 or dim >= len(leading_shape) for dim in wl_phase_dims):
            raise ValueError("wl_phase_dims must contain only leading axes, not the final two grid axes")
        if len(set(wl_phase_dims)) != len(wl_phase_dims):
            raise ValueError("wl_phase_dims must not contain duplicate axes")
        wl_phase_dims = tuple(sorted(wl_phase_dims))
        if v_wl__V.shape[row_dim] != row_num:
            raise ValueError(f"v_wl__V row axis must be row_num {row_num}; got {v_wl__V.shape[row_dim]}")
        if v_wl__V.shape[col_dim] != 1:
            raise ValueError("v_wl__V must carry a singleton column axis")

        # --- 2: prepare full-call snapshots ---

        cell_shape = (*leading_shape, *self._grid_shape)
        v_wl_grid__V = v_wl__V.expand(cell_shape)
        cell_snap = self.cell.snapshot(control=v_wl_grid__V, shape=cell_shape)

        # --- 3: execute the numerical graph and submit its energy ---

        dcop, energy__fJ, trace = self._solve_dc_chunking(
            leading_shape=leading_shape,
            wl_phase_dims=wl_phase_dims,
            cell_snap=cell_snap,
            bl_driver_snap=bl_driver_snap,
            sl_driver_snap=sl_driver_snap,
            record_energy=record_energy,
            record_trace=record_trace,
        )

        if energy__fJ is not None:
            self._record_dynamic_energy(energy__fJ)

        return dcop, trace

    @torch.compile(dynamic=False, fullgraph=True)
    def _solve_dc_chunking(
        self,
        *,
        leading_shape: tuple[int, ...],
        wl_phase_dims: tuple[int, ...],
        cell_snap: _CellSnap,
        bl_driver_snap: BLSnapT,
        sl_driver_snap: SLSnapT,
        record_energy: bool,
        record_trace: bool,
    ) -> tuple[_Dcop, Tensor | None, _Trace | None]:

        # --- 1: solve independent chunks and reassemble their outputs ---

        def solve_chunk(inputs: _Inputs[BLSnapT, SLSnapT]) -> _Outputs:
            def final_fn(state: _State) -> tuple[_Dcop, Tensor | None]:
                cell_dcop = self.cell.solve_dc(state.v_bl_node__V, state.v_sl_node__V, inputs.cell_snap)
                energy__fJ = None
                if record_energy:
                    energy__fJ = self._energy_from_state(
                        state,
                        cell_snap=inputs.cell_snap,
                        bl_driver_snap=inputs.bl_driver_snap,
                        sl_driver_snap=inputs.sl_driver_snap,
                        cell_dcop=cell_dcop,
                    )
                dcop = self._dcop_from_state(
                    state,
                    cell_snap=inputs.cell_snap,
                    bl_driver_snap=inputs.bl_driver_snap,
                    sl_driver_snap=inputs.sl_driver_snap,
                    cell_dcop=cell_dcop,
                )
                return dcop, energy__fJ

            (dcop, energy__fJ), trace = self.solver.solve(
                cell_snap=inputs.cell_snap,
                bl_driver_snap=inputs.bl_driver_snap,
                sl_driver_snap=inputs.sl_driver_snap,
                final_fn=final_fn,
                record_trace=record_trace,
                trace_mask=None,
            )
            return _Outputs(dcop=dcop, energy__fJ=energy__fJ, trace=trace)

        # Only the PyTree structure matters here; empty leaves avoid full-size
        # allocations before chunking. Recording flags fix optional structure at trace time.
        trace_template = None
        if record_trace:
            trace_template = _Trace.empty(
                (0, 0),
                port_shape=(0, 0),
                node_capacity=0,
                dtype=cell_snap.v_wl__V.dtype,
                device=cell_snap.v_wl__V.device,
            )
        output_template = _Outputs(
            dcop=self._dcop_template(like=cell_snap.v_wl__V),
            energy__fJ=cell_snap.v_wl__V.new_empty(0) if record_energy else None,
            trace=trace_template,
        )
        result = run_chunked(
            operands=_Inputs(
                cell_snap=cell_snap,
                bl_driver_snap=bl_driver_snap,
                sl_driver_snap=sl_driver_snap,
            ),
            output_template=output_template,
            body_fn=solve_chunk,
            leading_shape=leading_shape,
            expected_chunk_size=self.policy.solve_chunk_size,
            device=cell_snap.v_wl__V.device,
        )

        # --- 2: aggregate phase and rest energy for each hold window ---

        energy__fJ = None
        phase_energy__fJ = result.energy__fJ
        if phase_energy__fJ is not None:
            phase_energy__fJ = phase_energy__fJ.sum(dim=wl_phase_dims)
            v_bl_rest__V = bl_driver_snap.v_open__V
            v_sl_rest__V = sl_driver_snap.v_open__V
            # Remove later axes first so earlier phase indices keep their meaning.
            for dim in reversed(wl_phase_dims):
                v_bl_rest__V = v_bl_rest__V.select(dim, 0)
                v_sl_rest__V = v_sl_rest__V.select(dim, 0)
            rest_energy__fJ = self._rest_cap_energy__fJ(
                v_bl_rest__V=v_bl_rest__V,
                v_sl_rest__V=v_sl_rest__V,
            )
            energy__fJ = phase_energy__fJ + rest_energy__fJ

        return result.dcop, energy__fJ, result.trace

    def _dcop_template(self, *, like: Tensor) -> _Dcop:
        """Return the PyTree structure produced by `_dcop_from_state`.

        Tensor values and metadata are unused. Overrides match their Dcop fields.
        """
        empty = like.new_empty(0)
        return _Dcop(
            i_bl_port__uA=empty,
            v_bl_port__V=empty,
            i_sl_port__uA=empty,
            v_sl_port__V=empty,
        )

    def _dcop_from_state(
        self,
        state: _State,
        *,
        cell_snap: _CellSnap,
        bl_driver_snap: BLSnapT,
        sl_driver_snap: SLSnapT,
        cell_dcop: _CellDcop,
    ) -> _Dcop:
        """Compute the electrical output while preserving the state's leading shape.

        Snapshots and `cell_dcop` describe the same physical positions as `state`.
        The cell DCOP is evaluated at the supplied node voltages. A traced solve
        may supply a capped unconverged state. Overrides return their own Dcop
        and remain pure compiled tensor calculations.
        """
        row_dim = self.row_dim

        v_bl_node__V = state.v_bl_node__V
        v_sl_node__V = state.v_sl_node__V
        v_bl_port__V = state.v_bl_port__V
        v_sl_port__V = state.v_sl_port__V

        i_bl_port__uA = (v_bl_port__V - v_bl_node__V.narrow(row_dim, 0, 1)) * self.solver.bl_g__uS
        i_sl_port__uA = (v_sl_port__V - v_sl_node__V.narrow(row_dim, 0, 1)) * self.solver.sl_g__uS

        return _Dcop(
            i_bl_port__uA=i_bl_port__uA,
            v_bl_port__V=v_bl_port__V,
            i_sl_port__uA=i_sl_port__uA,
            v_sl_port__V=v_sl_port__V,
        )

    def _energy_from_state(
        self,
        state: _State,
        *,
        cell_snap: _CellSnap,
        bl_driver_snap: BLSnapT,
        sl_driver_snap: SLSnapT,
        cell_dcop: _CellDcop,
    ) -> Tensor:
        """Compute one WL phase's capacitive energy under a held rest boundary.

        Snapshots and `cell_dcop` describe the same positions as `state`; the cell
        DCOP is evaluated at its node voltages. A traced solve may supply a capped
        unconverged state. Overrides preserve the leading shape and remain pure
        compiled tensor calculations. Phase aggregation and rest establishment
        are accounted separately.

        Returns:
            Energy per leading position, in fJ.
            Shape: `[...]`.
        """
        config = self.config
        vdd__V = self._vdd__V
        array_dims = (self.row_dim, self.col_dim)

        # Shape: [..., row=1, col]
        v_bl_rest__V = bl_driver_snap.v_open__V
        # Shape: [..., row=1, col]
        v_sl_rest__V = sl_driver_snap.v_open__V

        # Shape: [..., row, col] -> [...]
        bl_node_e__fJ = e_cap_excursion__fJ(
            vdd__V, config.bl_node_c__fF, v_rest__V=v_bl_rest__V, v_work__V=state.v_bl_node__V
        ).sum(dim=array_dims)
        x_node_e__fJ = e_cap_excursion__fJ(
            vdd__V, config.x_node_c__fF, v_rest__V=v_bl_rest__V, v_work__V=cell_dcop.v_x__V
        ).sum(dim=array_dims)
        sl_node_e__fJ = e_cap_excursion__fJ(
            vdd__V, config.sl_node_c__fF, v_rest__V=v_sl_rest__V, v_work__V=state.v_sl_node__V
        ).sum(dim=array_dims)
        wl_node_e__fJ = e_cap_excursion__fJ(
            vdd__V, config.wl_node_c__fF, v_rest__V=0.0, v_work__V=cell_snap.v_wl__V
        ).sum(dim=array_dims)
        # Shape: [...]
        return bl_node_e__fJ + x_node_e__fJ + sl_node_e__fJ + wl_node_e__fJ

    def _rest_cap_energy__fJ(self, *, v_bl_rest__V: Tensor, v_sl_rest__V: Tensor) -> Tensor:
        vdd__V = self._vdd__V
        array_dims = (self.row_dim, self.col_dim)

        # Shape: [..., row=1, col] -> [...]
        bl_estab_e__fJ = e_cap_excursion__fJ(
            vdd__V, self._bl_estab_total_c__fF, v_rest__V=0.0, v_work__V=v_bl_rest__V
        ).sum(dim=array_dims)
        sl_estab_e__fJ = e_cap_excursion__fJ(
            vdd__V, self._sl_estab_total_c__fF, v_rest__V=0.0, v_work__V=v_sl_rest__V
        ).sum(dim=array_dims)
        # Shape: [...]
        return bl_estab_e__fJ + sl_estab_e__fJ
