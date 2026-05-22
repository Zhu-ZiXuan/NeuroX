"""Shape-independent physical core for a 1T1R crossbar tile.

See also:
    docs/dev/modules/xbar/_1t1r/circuit_core.md
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from neurox.analog import (
    Driver,
    DriverConfig,
)
from neurox.analog.dac import DAC, DACConfig
from neurox.analog.tia import TIA, TIAConfig
from neurox.common.mixin import FabricateMixin, ValidateMixin
from neurox.device import NMOS, RRAM, NMOSConfig, RRAMConfig

from .newton_raphson_solver import NewtonRaphsonSolver1T1R

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class CircuitCore1T1RConfig(ValidateMixin):
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
            ``[rram_cfg.g_min__uS, rram_g_max__uS]``.
        rram_cfg: RRAM device configuration.
        nmos_cfg: NMOS device configuration.
        tia_cfg: BL clamp-driver configuration.
        sl_driver_cfg: SL driver configuration.
        wl_dac_cfg: WL DAC configuration.
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
    state_to_g_map__uS: list[float]

    rram_cfg: RRAMConfig
    nmos_cfg: NMOSConfig
    tia_cfg: TIAConfig
    sl_driver_cfg: DriverConfig
    wl_dac_cfg: DACConfig

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
        if not (self.rram_g_max__uS > self.rram_cfg.g_min__uS):
            raise ValueError(
                f"require: rram_g_max__uS ({self.rram_g_max__uS}) > rram_cfg.g_min__uS ({self.rram_cfg.g_min__uS})"
            )

    def validate_state_map(self) -> None:
        self._require_min_length(self.state_to_g_map__uS, 2, "state_to_g_map__uS")
        self._require_strictly_increasing(self.state_to_g_map__uS, "state_to_g_map__uS")
        if self.state_to_g_map__uS[0] < self.rram_cfg.g_min__uS:
            raise ValueError(
                f"require: state_to_g_map__uS[0] ({self.state_to_g_map__uS[0]}) >= "
                f"rram_cfg.g_min__uS ({self.rram_cfg.g_min__uS})"
            )
        if self.state_to_g_map__uS[-1] > self.rram_g_max__uS:
            raise ValueError(
                f"require: state_to_g_map__uS[-1] ({self.state_to_g_map__uS[-1]}) <= "
                f"rram_g_max__uS ({self.rram_g_max__uS})"
            )


# ---------------------------------------------------------------------------
# DC operating-point container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Core1T1RDCOP:
    """Per-VMM DC operating point of a fabricated 1T1R core.

    Attributes:
        i_bl_driver: BL driver current [uA]. Shape: [..., phys_col_num].
        i_sl_driver: SL driver current [uA]. Shape: [..., phys_col_num].
        v_bl_node: BL node voltages [V]. Shape: [..., phys_col_num, row_num].
        v_sl_node: SL node voltages [V]. Shape: [..., phys_col_num, row_num].
        v_x_node: Access-NMOS drain voltages [V]. Shape: [..., phys_col_num, row_num].
        i_cell: Cell currents [uA]. Shape: [..., phys_col_num, row_num].
        v_bl_clamp: BL clamp voltages [V]. Shape: [..., phys_col_num].
        v_sl_drive: SL drive voltages [V]. Shape: [..., phys_col_num].
        v_wl_drive: WL drive voltages [V] as produced by the WL DAC.
            Shape: [..., row_num].
        v_out_phys: BL clamp output voltage at the converged port
            current [V]. Shape: [..., phys_col_num].
    """

    i_bl_driver: Tensor
    i_sl_driver: Tensor
    v_bl_node: Tensor
    v_sl_node: Tensor
    v_x_node: Tensor
    i_cell: Tensor
    v_bl_clamp: Tensor
    v_sl_drive: Tensor
    v_wl_drive: Tensor
    v_out_phys: Tensor


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------


class CircuitCore1T1R(FabricateMixin, nn.Module):
    """Shape-independent 1T1R core: devices, boundary circuits, and solver."""

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
        cfg: CircuitCore1T1RConfig,
        name: str,
        w_layout_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
    ) -> None:
        """Construct one shape-independent 1T1R core.

        Args:
            cfg: Concrete configuration dataclass.
            name: Hierarchical instance name used by the profiler.
            w_layout_shape: Per-instance state-index tensor shape
                ``(*prefix, phys_col_num, row_num)`` that
                ``program(...)`` will receive.
            dtype: Tensor dtype for internal buffers.
            T__K: Operating temperature [K].
        """
        super().__init__()

        if len(w_layout_shape) < 2:
            raise ValueError(
                f"w_layout_shape must have at least 2 trailing dims (phys_col_num, row_num); got {w_layout_shape}"
            )
        *prefix, phys_col_num, row_num = w_layout_shape
        if not (phys_col_num > 1):
            raise ValueError(f"require: phys_col_num ({phys_col_num}) > 1")
        if not (row_num > 1):
            raise ValueError(f"require: row_num ({row_num}) > 1")

        self._neurox_name = name
        self.cfg = cfg
        self.dtype = dtype
        self.T__K = T__K
        self._w_layout_shape = tuple(w_layout_shape)
        self._inst_shape = tuple(prefix)

        sub_prefix = name + "."
        self.rram = RRAM(
            cfg=cfg.rram_cfg,
            inst_shape=self._w_layout_shape,
            dtype=dtype,
            T__K=T__K,
            g_max__uS=cfg.rram_g_max__uS,
        )
        self.nmos = NMOS(
            cfg=cfg.nmos_cfg,
            inst_shape=self._w_layout_shape,
            dtype=dtype,
            T__K=T__K,
            W__um=cfg.access_nmos_W__um,
            L__um=cfg.access_nmos_L__um,
        )
        self.tia = TIA.from_config(
            cfg=cfg.tia_cfg,
            name=f"{sub_prefix}tia",
            inst_shape=(phys_col_num,),
            dtype=dtype,
            T__K=T__K,
        )
        access_W__um = cfg.access_nmos_W__um
        self.c_gs_per_cell__fF = cfg.c_gs_per_um__fF * access_W__um
        self.c_gd_per_cell__fF = cfg.c_gd_per_um__fF * access_W__um
        self.c_db_per_cell__fF = cfg.c_db_per_um__fF * access_W__um

        self.sl_driver = Driver(
            cfg=cfg.sl_driver_cfg,
            name=f"{sub_prefix}sl_driver",
            inst_shape=(phys_col_num,),
            dtype=dtype,
            T__K=T__K,
        )
        self.wl_dac = DAC.from_config(
            cfg=cfg.wl_dac_cfg,
            name=f"{sub_prefix}wl_dac",
            inst_shape=(row_num,),
            dtype=dtype,
            T__K=T__K,
        )

        self.register_buffer(
            "state_to_g_map__uS",
            torch.tensor(cfg.state_to_g_map__uS, dtype=dtype),
            persistent=False,
        )

        self.w_states = len(cfg.state_to_g_map__uS)
        self.x_states = self.wl_dac.code_max + 1

        self.c_wl_wire_per_row__fF = cfg.wl_first_c__fF + (phys_col_num - 1) * cfg.wl_segment_c__fF

        # Per-line segment buffers; index 0 is the driver-to-first segment.
        bl_segment_r__MOhm = torch.tensor(
            [cfg.bl_first_r__MOhm] + [cfg.bl_segment_r__MOhm] * (row_num - 1),
            dtype=dtype,
        )
        sl_segment_r__MOhm = torch.tensor(
            [cfg.sl_first_r__MOhm] + [cfg.sl_segment_r__MOhm] * (row_num - 1),
            dtype=dtype,
        )
        bl_segment_c__fF = torch.tensor(
            [cfg.bl_first_c__fF] + [cfg.bl_segment_c__fF] * (row_num - 1),
            dtype=dtype,
        )
        sl_segment_c__fF = torch.tensor(
            [cfg.sl_first_c__fF] + [cfg.sl_segment_c__fF] * (row_num - 1),
            dtype=dtype,
        )
        self.register_buffer("bl_segment_r__MOhm", bl_segment_r__MOhm, persistent=False)
        self.register_buffer("sl_segment_r__MOhm", sl_segment_r__MOhm, persistent=False)
        self.register_buffer("bl_segment_g__uS", 1.0 / bl_segment_r__MOhm, persistent=False)
        self.register_buffer("sl_segment_g__uS", 1.0 / sl_segment_r__MOhm, persistent=False)
        self.register_buffer("bl_segment_c__fF", bl_segment_c__fF, persistent=False)
        self.register_buffer("sl_segment_c__fF", sl_segment_c__fF, persistent=False)

        self.solver = NewtonRaphsonSolver1T1R(
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
    # DC solve
    # -----------------------------------------------------------------

    def solve_dc(self, x: Tensor) -> Core1T1RDCOP:
        """Run one DC solve on the fabricated 1T1R core.

        Args:
            x: WL DAC input-code tensor. Shape: [..., row_num].

        Returns:
            Core DC operating point for the current VMM.
        """

        # --- Insert the WL fanout dim and infer the broadcast execution shape ---

        # ``Tensor.expand`` cannot insert a dim mid-rank; the WL fanout dim
        # must be unsqueezed in before broadcasting against the RRAM grid.
        # Shape: [..., row_num] -> [..., 1, row_num]
        x = x.unsqueeze(-2)
        full_shape = torch.broadcast_shapes(self.rram.g__uS.shape, x.shape)
        *batch, _phys_col_num, row_num = full_shape

        # --- Sample runtime non-idealities ---

        rram_snapshot = self.rram.snapshot(shape=full_shape)
        nmos_snapshot = self.nmos.snapshot(shape=full_shape)
        bl_driver_snapshot = self.tia.snapshot(shape=(*batch, self.fabricated_col_num))
        sl_driver_snapshot = self.sl_driver.snapshot(shape=(*batch, self.fabricated_col_num))

        # --- Convert WL DAC codes into the per-cell gate-drive voltage ---

        v_wl_drive__V = self.wl_dac.convert(x)

        # --- Solve the array DC operating point ---

        solver_dcop = self.solver.solve_dc(
            v_wl_drive__V=v_wl_drive__V,
            bl_segment_r__MOhm=self.bl_segment_r__MOhm,
            sl_segment_r__MOhm=self.sl_segment_r__MOhm,
            bl_segment_g__uS=self.bl_segment_g__uS,
            sl_segment_g__uS=self.sl_segment_g__uS,
            rram_snapshot=rram_snapshot,
            nmos_snapshot=nmos_snapshot,
            bl_driver_snapshot=bl_driver_snapshot,
            sl_driver_snapshot=sl_driver_snapshot,
        )

        # --- Recover the BL output clamp voltage ---

        clamp_dcop = self.tia.solve_dc(
            solver_dcop.i_bl_driver,
            bl_driver_snapshot,
            v_clamp_init__V=solver_dcop.v_bl_clamp,
        )
        v_out_phys = clamp_dcop.v_out__V

        # --- Assemble the core DCOP ---

        core_dcop = Core1T1RDCOP(
            i_bl_driver=solver_dcop.i_bl_driver,
            i_sl_driver=solver_dcop.i_sl_driver,
            v_bl_node=solver_dcop.v_bl_node,
            v_sl_node=solver_dcop.v_sl_node,
            v_x_node=solver_dcop.v_x_node,
            i_cell=solver_dcop.i_cell,
            v_bl_clamp=solver_dcop.v_bl_clamp,
            v_sl_drive=solver_dcop.v_sl_drive,
            v_wl_drive=v_wl_drive__V.squeeze(-2),
            v_out_phys=v_out_phys,
        )

        # --- Accumulate analog-side dynamic energy ---

        array_energy__fJ = self._compute_array_energy__fJ(core_dcop)
        self.tia._log_dynamic(array_energy__fJ, self.tia.cfg.latency_per_op__ns)

        return core_dcop

    # -----------------------------------------------------------------
    # Dynamic-energy aggregation
    # -----------------------------------------------------------------

    def _compute_array_energy__fJ(self, dcop: Core1T1RDCOP) -> Tensor:
        """Per-VMM array-internal energy [fJ]. Shape: [...batch...]."""

        v_wl__V = dcop.v_wl_drive
        v_bl__V = dcop.v_bl_node
        v_sl__V = dcop.v_sl_node
        v_x__V = dcop.v_x_node
        v_bl_clamp__V = dcop.v_bl_clamp
        v_sl_drive__V = dcop.v_sl_drive
        pulse__ns = self.cfg.wl_pulse_length__ns

        # --- DC conduction ---

        # Shape: [..., phys_col_num] -> [...]
        array_power__uW = (v_bl_clamp__V * dcop.i_bl_driver).sum(dim=-1) + (
            v_sl_drive__V * dcop.i_sl_driver
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
