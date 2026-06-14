"""Shape-independent physical core for a 1T1R crossbar tile.

See also:
    docs/dev/modules/xbar/_1t1r/circuit_core.md
"""

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from neurox.analog import (
    Driver,
    DriverConfig,
    DriverPolicy,
)
from neurox.analog.dac import DAC, DACConfig, DACPolicy
from neurox.analog.tia import TIA, TIAConfig, TIAPolicy
from neurox.common.circuit import CircuitBase, CircuitConfig
from neurox.device import (
    NMOS,
    RRAM,
    NMOSConfig,
    NMOSPolicy,
    RRAMConfig,
    RRAMPolicy,
)

from ._chunking import classify_leading_positions, iter_chunks, reassemble_chunks
from .solver import Solver1T1R, Solver1T1RConfig, Solver1T1RDCOP

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class CircuitCore1T1RConfig(CircuitConfig):
    """Shape-independent physical knobs for a 1T1R core.

    Attributes:
        wl_pulse_length__ns: Word-line pulse length [ns].
        row_first_space__um: Row pitch from the driver to the first cell [um].
        row_cell_space__um: Row pitch between adjacent cells [um].
        col_first_space__um: Column pitch from the driver to the first cell [um].
        col_cell_space__um: Column pitch between adjacent cells [um].
        bl_first_r__MOhm: BL driver-to-first-cell segment resistance [MOhm].
        bl_first_c__fF: BL driver-to-first-cell segment capacitance [fF].
        bl_segment_r__MOhm: BL cell-to-cell segment resistance [MOhm].
        bl_segment_c__fF: BL cell-to-cell segment capacitance [fF].
        sl_first_r__MOhm: SL driver-to-first-cell segment resistance [MOhm].
        sl_first_c__fF: SL driver-to-first-cell segment capacitance [fF].
        sl_segment_r__MOhm: SL cell-to-cell segment resistance [MOhm].
        sl_segment_c__fF: SL cell-to-cell segment capacitance [fF].
        wl_first_r__MOhm: WL driver-to-first-cell segment resistance [MOhm].
        wl_first_c__fF: WL driver-to-first-cell segment capacitance [fF].
        wl_segment_r__MOhm: WL cell-to-cell segment resistance [MOhm].
        wl_segment_c__fF: WL cell-to-cell segment capacitance [fF].
        access_nmos_W__um: Access-NMOS width [um].
        access_nmos_L__um: Access-NMOS length [um].
        c_gs_per_um__fF: Access-NMOS gate-to-source capacitance per unit width
            [fF/um].
        c_gd_per_um__fF: Access-NMOS gate-to-drain capacitance per unit width
            [fF/um].
        c_db_per_um__fF: Access-NMOS drain-to-body capacitance per unit width
            [fF/um].
        rram_g_max__uS: Maximum programmable RRAM conductance [uS].
        state_to_g_map__uS: State-index to target-conductance lookup
            table [uS]. Strictly increasing; endpoints must lie inside
            ``[rram_config.g_min__uS, rram_g_max__uS]``.
        rram_config: RRAM device configuration.
        nmos_config: NMOS device configuration.
        tia_config: BL clamp-driver configuration.
        sl_driver_config: SL driver configuration.
        wl_dac_config: WL DAC configuration.
        solver_config: DC-solver fixed numerical knobs. Concrete subclass
            of :class:`Solver1T1RConfig` (``NestedSolver1T1RConfig`` or
            ``FullJacobianSolver1T1RConfig``) picks which solver
            implementation the core instantiates via
            ``Solver1T1R.from_config(...)``.
        area_per_inst__um2: Core (cell array + wire infra) silicon area
            per fabricated tile instance [um²]. **Excludes** the owned
            ``CircuitBase`` children (TIA / drivers / DAC), which roll
            up separately via the composite-aggregation rule in
            ``docs/dev/architecture/profiler_and_ppa.md``. Device-side
            contributions (RRAM / NMOS) are not separately rolled up —
            their physical area must be folded into this field by the
            caller (devices do not inherit ``CircuitBase``).
        leakage_per_inst__uW: Core static leakage per fabricated tile
            instance [uW]. Same scope as ``area_per_inst__um2``.
        latency_per_op__ns: Core-side per-VMM latency [ns] that the
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

    access_nmos_W__um: float
    access_nmos_L__um: float

    c_gs_per_um__fF: float
    c_gd_per_um__fF: float
    c_db_per_um__fF: float

    rram_g_max__uS: float
    state_to_g_map__uS: tuple[float, ...]

    rram_config: RRAMConfig
    nmos_config: NMOSConfig
    tia_config: TIAConfig
    sl_driver_config: DriverConfig
    wl_dac_config: DACConfig
    solver_config: Solver1T1RConfig

    latency_per_op__ns: float

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        self.validate_wl_pulse()
        self.validate_layout_pitch()
        self.validate_wire_segments()
        self.validate_access_nmos()
        self.validate_parasitics()
        self.validate_rram_window()
        self.validate_state_map()
        self.validate_ppa()

    def validate_ppa(self) -> None:
        super().validate_ppa()
        self._require_nonneg(self.latency_per_op__ns, "latency_per_op__ns")

    def validate_wl_pulse(self) -> None:
        self._require_nonneg(self.wl_pulse_length__ns, "wl_pulse_length__ns")

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

    def validate_access_nmos(self) -> None:
        self._require_pos(self.access_nmos_W__um, "access_nmos_W__um")
        self._require_pos(self.access_nmos_L__um, "access_nmos_L__um")

    def validate_parasitics(self) -> None:
        self._require_nonneg(self.c_gs_per_um__fF, "c_gs_per_um__fF")
        self._require_nonneg(self.c_gd_per_um__fF, "c_gd_per_um__fF")
        self._require_nonneg(self.c_db_per_um__fF, "c_db_per_um__fF")

    def validate_rram_window(self) -> None:
        if not (self.rram_g_max__uS > self.rram_config.g_min__uS):
            raise ValueError(
                f"require: rram_g_max__uS ({self.rram_g_max__uS}) > rram_config.g_min__uS ({self.rram_config.g_min__uS})"
            )

    def validate_state_map(self) -> None:
        self._require_min_length(self.state_to_g_map__uS, 2, "state_to_g_map__uS")
        self._require_strictly_increasing(self.state_to_g_map__uS, "state_to_g_map__uS")
        if self.state_to_g_map__uS[0] < self.rram_config.g_min__uS:
            raise ValueError(
                f"require: state_to_g_map__uS[0] ({self.state_to_g_map__uS[0]}) >= "
                f"rram_config.g_min__uS ({self.rram_config.g_min__uS})"
            )
        if self.state_to_g_map__uS[-1] > self.rram_g_max__uS:
            raise ValueError(
                f"require: state_to_g_map__uS[-1] ({self.state_to_g_map__uS[-1]}) <= "
                f"rram_g_max__uS ({self.rram_g_max__uS})"
            )


# ---------------------------------------------------------------------------
# Nonideality policy
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CircuitCore1T1RPolicy:
    """Composite nonideality policy for a 1T1R circuit core.

    Attributes:
        rram: RRAM cell-array nonideality policy.
        nmos: Cell access-NMOS nonideality policy.
        tia: BL clamp-driver (TIA) nonideality policy.
        sl_driver: SL driver nonideality policy.
        wl_dac: WL DAC nonideality policy.
        solve_chunk_size_x: Chunk size along the **A subset** of the
            ``solve_dc`` broadcast leading (x-side positions
            ``*x_batch`` / ``M`` / ``Sa``). ``0`` disables chunking on
            this axis. Runtime knob (depends on GPU memory budget /
            throughput target), not a chip-preset constant.
        solve_chunk_size_inst: Chunk size along the **B subset** —
            the instance positions (``Sw`` / ``Tr`` / matched ``Tc``).
            ``0`` disables chunking on this axis.

    Both chunk knobs are zero by default behaviour: when both are ``0``
    ``cim_read`` runs the single-block path; either non-zero forces the
    memory-bounded nested chunked path.

    Solvers have **no Policy** — all their knobs are fixed numerical
    constants and live on :class:`CircuitCore1T1RConfig.solver_config`.
    """

    rram: RRAMPolicy
    nmos: NMOSPolicy
    tia: TIAPolicy
    sl_driver: DriverPolicy
    wl_dac: DACPolicy
    solve_chunk_size_x: int
    solve_chunk_size_inst: int


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------


class CircuitCore1T1R(CircuitBase[CircuitCore1T1RConfig]):
    """Shape-independent 1T1R core: devices, boundary circuits, and solver."""

    config: CircuitCore1T1RConfig
    state_to_g_map__uS: Tensor
    bl_segment_r__MOhm: Tensor
    sl_segment_r__MOhm: Tensor
    bl_segment_g__uS: Tensor
    sl_segment_g__uS: Tensor
    bl_segment_c__fF: Tensor
    sl_segment_c__fF: Tensor

    def __init__(
        self,
        *,
        config: CircuitCore1T1RConfig,
        policy: CircuitCore1T1RPolicy,
        name: str,
        w_layout_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
    ) -> None:
        """Construct one shape-independent 1T1R core.

        Args:
            config: Concrete configuration dataclass.
            policy: Composite nonideality policy.
            name: Hierarchical instance name used by the profiler.
            w_layout_shape: Per-instance state-index tensor shape
                ``(*prefix, phys_col_num, row_num)`` that
                ``program(...)`` will receive.
            dtype: Tensor dtype for internal buffers.
            T__K: Operating temperature [K].
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

        super().__init__(config=config, name=name, inst_shape=tuple(prefix))
        self.policy = policy
        self.dtype = dtype
        self.T__K = T__K
        self._w_layout_shape = tuple(w_layout_shape)

        sub_prefix = name + "."
        self.rram = RRAM(
            config=config.rram_config,
            policy=policy.rram,
            inst_shape=self._w_layout_shape,
            dtype=dtype,
            T__K=T__K,
            g_max__uS=config.rram_g_max__uS,
        )
        self.nmos = NMOS(
            config=config.nmos_config,
            policy=policy.nmos,
            inst_shape=self._w_layout_shape,
            dtype=dtype,
            T__K=T__K,
            W__um=config.access_nmos_W__um,
            L__um=config.access_nmos_L__um,
        )
        self.tia = TIA.from_config(
            config=config.tia_config,
            policy=policy.tia,
            name=f"{sub_prefix}tia",
            inst_shape=(*prefix, phys_col_num),
            dtype=dtype,
            T__K=T__K,
        )
        access_W__um = config.access_nmos_W__um
        self.c_gs_per_cell__fF = config.c_gs_per_um__fF * access_W__um
        self.c_gd_per_cell__fF = config.c_gd_per_um__fF * access_W__um
        self.c_db_per_cell__fF = config.c_db_per_um__fF * access_W__um

        self.sl_driver = Driver(
            config=config.sl_driver_config,
            policy=policy.sl_driver,
            name=f"{sub_prefix}sl_driver",
            inst_shape=(*prefix, phys_col_num),
            dtype=dtype,
            T__K=T__K,
        )
        self.wl_dac = DAC.from_config(
            config=config.wl_dac_config,
            policy=policy.wl_dac,
            name=f"{sub_prefix}wl_dac",
            inst_shape=(*prefix, row_num),
            dtype=dtype,
            T__K=T__K,
        )

        self.register_buffer(
            "state_to_g_map__uS",
            torch.tensor(config.state_to_g_map__uS, dtype=dtype),
            persistent=False,
        )

        self.w_states = len(config.state_to_g_map__uS)
        self.x_states = self.wl_dac.code_max + 1

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

        self.solver = Solver1T1R.from_config(
            config=config.solver_config,
            rram=self.rram,
            nmos=self.nmos,
            bl_driver=self.tia,
            sl_driver=self.sl_driver,
        )

        self.fabricated_col_num = phys_col_num
        self.fabricated_row_num = row_num

    # -----------------------------------------------------------------
    # Programming
    # -----------------------------------------------------------------

    def program(self, w_state_idx: Tensor) -> None:
        """Write the RRAM cells from one state-index tensor.

        Args:
            w_state_idx: State-index tensor in ``[0, w_states - 1]``,
                shape must match ``self._w_layout_shape =
                (*prefix, phys_col_num, row_num)``.
        """
        if tuple(w_state_idx.shape) != self._w_layout_shape:
            raise ValueError(
                f"program() expects w_state_idx.shape {self._w_layout_shape}; got {tuple(w_state_idx.shape)}"
            )
        target_g__uS = self.state_to_g_map__uS[w_state_idx.long()]
        self.rram.program(target_g__uS, t_elapsed=0.0)

    # -----------------------------------------------------------------
    # CIM read (plain forward)
    # -----------------------------------------------------------------

    def cim_read(self, x: Tensor) -> Tensor:
        """Drive the 1T1R array with a WL input and return the BL clamp voltage.

        Plain forward: drive WL via the DAC, settle the array+TIA to DC
        in chunked Newton sub-solves, accumulate per-VMM dynamic energy,
        emit one energy + one latency profile event, and return the BL
        clamp voltage that the downstream readout will sample. The
        chunked sub-solves are an internal memory-bounding detail —
        from the caller's view this is a single forward call.

        Args:
            x: WL DAC input-code tensor. Shape: [..., row_num].

        Returns:
            BL clamp voltage at the converged operating point [V].
            Shape: ``[..., phys_col_num]``.
        """

        # --- Infer the broadcast leading ---

        # ``x_code`` stays raw — its trailing is ``[row]``, matching the
        # WL DAC's natural shape contract (no synthetic fanout dim that
        # has nothing to do with the DAC's own structure). ``x_grid``
        # adds a size-1 WL-fanout dim at -2 so ``x``'s ``row`` aligns
        # with ``g``'s ``row`` and the ``phys_col`` slot opens for the
        # solver-side broadcast against the RRAM grid. The split keeps
        # each consumer working in the shape space that makes physical
        # sense for it (DAC: (*leading, row); solver: (*leading, 1, row)).
        x_code = x
        x_grid = x_code.unsqueeze(-2)
        full_shape = torch.broadcast_shapes(self.rram.g__uS.shape, x_grid.shape)
        *batch_list, phys_col_num, row_num = full_shape
        leading = tuple(batch_list)
        rram_trailing = (phys_col_num, row_num)
        tia_trailing = (phys_col_num,)
        col_trailing = (phys_col_num,)

        cx = self.policy.solve_chunk_size_x
        ci = self.policy.solve_chunk_size_inst

        # --- Classify leading positions and convert DAC once ---

        a_positions, b_positions = classify_leading_positions(
            x_shape=tuple(x_grid.shape),
            g_shape=tuple(self.rram.g__uS.shape),
            leading_rank=len(leading),
        )

        # DAC convert runs on the broadcast-to-full-leading view of
        # ``x_code`` — no WL fanout slot. The expand is zero-copy but
        # converting against the realised full leading is required so
        # that (a) per-instance per-op dynamic energy is counted once
        # per instance (not once per x_batch), and (b)
        # ``drive_thermal__V`` samples an independent noise tensor per
        # instance instead of having one x-batch noise pattern aliased
        # across every inst position. The fanout slot is added back
        # AFTER convert so the solver sees ``(*leading, 1, row)``.
        # Each VMM still logs exactly one DAC energy+latency event pair.
        x_dac_input = x_code.expand(*leading, row_num)
        v_wl_dac = self.wl_dac.convert(x_dac_input)
        v_wl_full = v_wl_dac.unsqueeze(-2)

        # --- Per-chunk loop: sample → solve → TIA clamp → energy ---
        # Only the small per-chunk tensors needed for reassembly +
        # per-chunk energy are retained. The heavy ``solver_dcop_chunk``
        # (carrying ``v_bl_node`` / ``v_sl_node`` / ``v_x_node`` /
        # ``i_cell`` sized ``(chunk_size, phys_col, row)``) lives only
        # within one loop iteration and is released by Python's GC
        # before the next chunk starts — preserving chunking's peak-
        # memory contract.

        v_out_phys_chunks: list[Tensor] = []
        chunk_energies: list[Tensor] = []
        global_indices: list[Tensor] = []

        for spec in iter_chunks(
            leading=leading,
            a_positions=a_positions,
            b_positions=b_positions,
            chunk_size_x=cx,
            chunk_size_inst=ci,
            device=x.device,
        ):
            mc = spec.multi_coords
            rram_snap = self.rram.snapshot(shape=(*leading, *rram_trailing), multi_coords=mc)
            nmos_snap = self.nmos.snapshot(shape=(*leading, *rram_trailing), multi_coords=mc)
            bl_snap = self.tia.snapshot(shape=(*leading, *tia_trailing), multi_coords=mc)
            sl_snap = self.sl_driver.snapshot(shape=(*leading, *tia_trailing), multi_coords=mc)
            v_wl_chunk = v_wl_full[mc] if mc else v_wl_full  # (chunk_size, 1, row)

            solver_dcop_chunk = self.solver.solve_dc(
                v_wl_drive__V=v_wl_chunk,
                bl_segment_r__MOhm=self.bl_segment_r__MOhm,
                sl_segment_r__MOhm=self.sl_segment_r__MOhm,
                bl_segment_g__uS=self.bl_segment_g__uS,
                sl_segment_g__uS=self.sl_segment_g__uS,
                rram_snapshot=rram_snap,
                nmos_snapshot=nmos_snap,
                bl_driver_snapshot=bl_snap,
                sl_driver_snapshot=sl_snap,
                compute_residuals=False,
            )
            clamp_dcop_chunk = self.tia.solve_dc(
                solver_dcop_chunk.i_bl_driver,
                bl_snap,
                v_clamp_init__V=solver_dcop_chunk.v_bl_clamp,
            )
            chunk_energies.append(
                self._compute_array_energy__fJ(
                    solver_dcop=solver_dcop_chunk,
                    v_wl_drive=v_wl_chunk.squeeze(-2),
                )
            )
            v_out_phys_chunks.append(clamp_dcop_chunk.v_out__V)
            global_indices.append(spec.flat_global_idx)
            # solver_dcop_chunk / clamp_dcop_chunk go out of scope at
            # iteration end → heavy per-cell tensors freed before the
            # next chunk allocates its own.

        # --- Reassemble outputs ---

        v_out_phys = reassemble_chunks(v_out_phys_chunks, global_indices, leading, col_trailing)
        array_energy__fJ = reassemble_chunks(chunk_energies, global_indices, leading, ())

        # --- Emit one energy + one latency event for this VMM ---

        # Serial is the x-side broadcast (a_positions); the inst-side
        # (b_positions) is parallel physical hardware and must not enter
        # the per-op-latency serial count — same convention as every
        # other emitting leaf, where parallel multiplicity divides out
        # of ``numel(output)``. Here a/b is broadcast-determined so we
        # use ``classify_leading_positions``'s explicit split rather
        # than dividing by a static ``inst_count``.
        serial_op_count = math.prod(leading[p] for p in a_positions) if a_positions else 1
        latency__ns = torch.tensor(
            self.config.latency_per_op__ns * serial_op_count,
            device=array_energy__fJ.device,
            dtype=array_energy__fJ.dtype,
        )
        self._log_dynamic_energy(array_energy__fJ)
        self._log_latency(latency__ns)
        return v_out_phys

    # -----------------------------------------------------------------
    # Dynamic-energy aggregation
    # -----------------------------------------------------------------

    def _compute_array_energy__fJ(
        self,
        *,
        solver_dcop: Solver1T1RDCOP,
        v_wl_drive: Tensor,
    ) -> Tensor:
        """Per-VMM array-internal energy [fJ]. Shape: [...batch...].

        Args:
            solver_dcop: Inner array solver's converged DCOP, carrying
                the per-cell node voltages and per-column port currents.
            v_wl_drive: WL drive voltages [V] from the WL DAC,
                shape ``[..., row_num]``.
        """

        v_wl__V = v_wl_drive
        v_bl__V = solver_dcop.v_bl_node
        v_sl__V = solver_dcop.v_sl_node
        v_x__V = solver_dcop.v_x_node
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

        # Shape: [..., phys_col_num, row_num] -> [...]
        e_rram_top__fJ = (self.rram.c_top__fF * v_bl__V.square()).sum(dim=(-2, -1))
        e_rram_bot__fJ = (self.rram.c_bot__fF * v_x__V.square()).sum(dim=(-2, -1))
        e_nmos_db__fJ = (self.c_db_per_cell__fF * v_x__V.square()).sum(dim=(-2, -1))

        # Shape: [..., row_num] -> [..., 1, row_num]
        v_wl_grid__V = v_wl__V.unsqueeze(-2)
        # Shape: [..., phys_col_num, row_num] -> [...]
        e_nmos_gs__fJ = (self.c_gs_per_cell__fF * (v_wl_grid__V - v_sl__V).square()).sum(dim=(-2, -1))
        e_nmos_gd__fJ = (self.c_gd_per_cell__fF * (v_wl_grid__V - v_x__V).square()).sum(dim=(-2, -1))

        return (
            e_dc_cond__fJ
            + e_wl_wire_cap__fJ
            + e_bl_wire_cap__fJ
            + e_sl_wire_cap__fJ
            + e_rram_top__fJ
            + e_rram_bot__fJ
            + e_nmos_db__fJ
            + e_nmos_gs__fJ
            + e_nmos_gd__fJ
        )
