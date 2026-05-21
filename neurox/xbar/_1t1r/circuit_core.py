"""Shape-independent physical core for a 1T1R crossbar tile.

See also:
    docs/dev/modules/xbar/_1t1r/circuit_core.md
"""

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
from torch import Tensor

from neurox.analog import (
    Decoder,
    DecoderConfig,
    Driver,
    DriverConfig,
)
from neurox.analog.dac import DAC, DACConfig
from neurox.analog.tia import TIA, TIAConfig
from neurox.common.validate import ValidateMixin
from neurox.device import NMOS, RRAM, NMOSConfig, RRAMConfig

from .newton_raphson_solver import NewtonRaphsonSolver1T1R, Solver1T1RDCOP

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class CircuitCore1T1RConfig(ValidateMixin):
    """Shape-independent physical knobs for a 1T1R core.

    Attributes:
        wl_pulse_length__ns: Word-line pulse length [ns].
        sl_topology: Source-line sharing topology.
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
        wl_decoder_cfg: WL decoder configuration.
        wl_dac_cfg: WL DAC configuration.
    """

    wl_pulse_length__ns: float
    sl_topology: Literal["row_shared", "col_shared"]

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
    wl_decoder_cfg: DecoderConfig
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

    Carries the solver's array-internal state plus the post-clamp BL
    output voltage. Fields are flattened from the solver and TIA results;
    callers read them directly without further nesting.

    Attributes:
        i_bl_driver: BL driver current [uA]. Shape: [..., phys_col_num].
        i_sl_driver: SL driver current [uA]. Shape: [..., row_num].
        v_bl_node: BL node voltages [V]. Shape: [..., phys_col_num, row_num].
        v_sl_node: SL node voltages [V]. Shape: [..., phys_col_num, row_num].
        v_x_node: Access-NMOS drain voltages [V]. Shape: [..., phys_col_num, row_num].
        i_cell: Cell currents [uA]. Shape: [..., phys_col_num, row_num].
        v_bl_clamp: BL clamp voltages [V]. Shape: [..., phys_col_num].
        v_sl_drive: SL drive voltages [V]. Shape: [..., row_num].
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
    v_out_phys: Tensor


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------


class CircuitCore1T1R(nn.Module):
    """Shape-independent 1T1R core: devices, boundary circuits, and solver."""

    state_to_g_map__uS: Tensor

    def __init__(
        self,
        *,
        cfg: CircuitCore1T1RConfig,
        name: str,
        T__K: float,
        dtype: torch.dtype,
    ) -> None:
        """Construct one shape-independent 1T1R core.

        Args:
            cfg: Core configuration.
            name: Profiler/debug name.
            T__K: Operating temperature [K].
            dtype: Tensor dtype for internal buffers.
        """
        super().__init__()

        self._neurox_name = name
        self.cfg = cfg
        self.dtype = dtype
        self.T__K = T__K

        # Build the owned children from the embedded member configs.
        prefix = name + "."
        self.rram = RRAM(cfg=cfg.rram_cfg, T__K=T__K, dtype=dtype, g_max__uS=cfg.rram_g_max__uS)
        self.nmos = NMOS(
            cfg=cfg.nmos_cfg,
            T__K=T__K,
            dtype=dtype,
            W__um=cfg.access_nmos_W__um,
            L__um=cfg.access_nmos_L__um,
        )
        self.tia = TIA.from_config(
            cfg=cfg.tia_cfg,
            name=f"{prefix}tia",
            T__K=T__K,
            dtype=dtype,
        )
        # Per-cell access-transistor parasitic caps for energy accounting.
        access_W__um = cfg.access_nmos_W__um
        self._c_gs__fF = cfg.c_gs_per_um__fF * access_W__um
        self._c_gd__fF = cfg.c_gd_per_um__fF * access_W__um
        self._c_db__fF = cfg.c_db_per_um__fF * access_W__um

        self.sl_driver = Driver(cfg=cfg.sl_driver_cfg, name=f"{prefix}sl_driver", T__K=T__K, dtype=dtype)
        self.wl_decoder = Decoder(cfg=cfg.wl_decoder_cfg, name=f"{prefix}wl_decoder", T__K=T__K, dtype=dtype)
        self.wl_dac = DAC.from_config(
            cfg=cfg.wl_dac_cfg,
            name=f"{prefix}wl_dac",
            T__K=T__K,
            dtype=dtype,
        )

        self.v_dd_wl__V = float(self.wl_dac.code_to_signal[1].item())

        self.register_buffer(
            "state_to_g_map__uS",
            torch.tensor(cfg.state_to_g_map__uS, dtype=dtype),
            persistent=False,
        )

        self.w_states = len(cfg.state_to_g_map__uS)
        self.x_states = 2

        # Shape-dependent capacitances and the solver are set up in fabricate().
        self.solver: NewtonRaphsonSolver1T1R

        self.fabricated_row_num = 0
        self.fabricated_col_num = 0
        self.c_wl_per_row__fF = 0.0
        self.c_bl_per_node__fF = 0.0
        self.c_x_per_cell__fF = 0.0
        self.c_gd_per_cell__fF = 0.0

    # -----------------------------------------------------------------
    # Fabrication
    # -----------------------------------------------------------------

    def fabricate(self, w_state_idx: Tensor) -> None:
        """Program the core from one state-index tensor.

        Args:
            w_state_idx: State-index tensor in ``[0, w_states - 1]``.
                Shape: [..., phys_col_num, row_num].
        """
        *_, phys_col_num, row_num = w_state_idx.shape
        if not (phys_col_num > 1):
            raise ValueError(f"require: phys_col_num ({phys_col_num}) > 1")
        if not (row_num > 1):
            raise ValueError(f"require: row_num ({row_num}) > 1")
        if self.cfg.sl_topology != "row_shared":
            raise NotImplementedError(
                f"sl_topology={self.cfg.sl_topology!r} is reserved for a future "
                "build; only 'row_shared' is implemented today"
            )

        target_g__uS = self.state_to_g_map__uS[w_state_idx.long()]
        self.rram.program(target_g__uS, t_elapsed=0.0)
        self.nmos.fabricate(w_state_idx.shape)
        self.tia.fabricate((phys_col_num,))

        # Energy-model capacitance scalars: total wire cap = first + (N-1) * segment.
        wl_wire_cap__fF = self.cfg.wl_first_c__fF + (phys_col_num - 1) * self.cfg.wl_segment_c__fF
        self.c_wl_per_row__fF = wl_wire_cap__fF + phys_col_num * self._c_gs__fF
        bl_wire_total__fF = self.cfg.bl_first_c__fF + (row_num - 1) * self.cfg.bl_segment_c__fF
        self.c_bl_per_node__fF = bl_wire_total__fF / row_num + self.rram.c_top__fF
        self.c_x_per_cell__fF = self._c_db__fF + self.rram.c_bot__fF
        self.c_gd_per_cell__fF = self._c_gd__fF

        # Per-line segment resistances. Index 0 is the driver-to-first segment.
        bl_segment_r__MOhm = torch.tensor(
            [self.cfg.bl_first_r__MOhm] + [self.cfg.bl_segment_r__MOhm] * (row_num - 1),
            dtype=self.dtype,
        )
        sl_segment_r__MOhm = torch.tensor(
            [self.cfg.sl_first_r__MOhm] + [self.cfg.sl_segment_r__MOhm] * (phys_col_num - 1),
            dtype=self.dtype,
        )
        # Bind the DC solver.
        self.solver = NewtonRaphsonSolver1T1R(
            rram=self.rram,
            nmos=self.nmos,
            bl_driver=self.tia,
            sl_driver=self.sl_driver,
            bl_segment_r__MOhm=bl_segment_r__MOhm,
            sl_segment_r__MOhm=sl_segment_r__MOhm,
        )

        self.fabricated_col_num = phys_col_num
        self.fabricated_row_num = row_num

    # -----------------------------------------------------------------
    # DC solve
    # -----------------------------------------------------------------

    def solve_dc(self, x: Tensor) -> Core1T1RDCOP:
        """Run one DC solve on the fabricated 1T1R core.

        Args:
            x: Binary word-line logic tensor. Shape: [..., row_num].

        Returns:
            Core DC operating point for the current VMM.
        """

        # --- Infer the broadcast execution shape ---

        # Execution shape is determined by the programmed RRAM layout and the input.
        full_shape = torch.broadcast_shapes(self.rram.g__uS.shape, x.shape)
        *batch, _phys_col_num, row_num = full_shape

        # --- Sample runtime non-idealities ---

        rram_snapshot = self.rram.snapshot(shape=full_shape)
        nmos_snapshot = self.nmos.snapshot(shape=full_shape)
        bl_driver_snapshot = self.tia.snapshot(shape=(*batch, self.fabricated_col_num))
        sl_driver_snapshot = self.sl_driver.snapshot(shape=(*batch, row_num))

        # --- Expand the logical WL input into solver layout ---

        # Shape: [..., row_num] -> [..., 1, row_num]
        x = x.expand(*batch, 1, row_num)
        # Shape: [..., 1, row_num] -> [..., row_num]
        wl_logic = x.squeeze(-2)
        v_wl_drive__V = self.wl_dac.convert(x)

        # --- Solve the array DC operating point ---

        solver_dcop = self.solver.solve_dc(
            v_wl_drive__V,
            rram_snapshot=rram_snapshot,
            nmos_snapshot=nmos_snapshot,
            bl_driver_snapshot=bl_driver_snapshot,
            sl_driver_snapshot=sl_driver_snapshot,
        )

        # --- Accumulate analog-side dynamic energy ---

        array_energy__fJ = self._compute_array_energy__fJ(
            solver_dcop=solver_dcop,
            wl_logic=wl_logic,
        )
        self.tia._log_dynamic(array_energy__fJ, self.tia.cfg.latency_per_op__ns)

        # --- Recover the BL output clamp voltage ---

        # Re-evaluate the BL clamp at the converged port current to recover ``v_out``.
        clamp_dcop = self.tia.solve_dc(
            solver_dcop.i_bl_driver,
            bl_driver_snapshot,
            v_clamp_init__V=solver_dcop.v_bl_clamp,
        )
        # Shape: [..., phys_col_num]
        v_out_phys = clamp_dcop.v_out__V

        return Core1T1RDCOP(
            i_bl_driver=solver_dcop.i_bl_driver,
            i_sl_driver=solver_dcop.i_sl_driver,
            v_bl_node=solver_dcop.v_bl_node,
            v_sl_node=solver_dcop.v_sl_node,
            v_x_node=solver_dcop.v_x_node,
            i_cell=solver_dcop.i_cell,
            v_bl_clamp=solver_dcop.v_bl_clamp,
            v_sl_drive=solver_dcop.v_sl_drive,
            v_out_phys=v_out_phys,
        )

    # -----------------------------------------------------------------
    # Dynamic-energy aggregation
    # -----------------------------------------------------------------

    def _compute_array_energy__fJ(
        self,
        solver_dcop: Solver1T1RDCOP,
        wl_logic: Tensor,
    ) -> Tensor:
        """Compute per-VMM array-internal dynamic energy.

        Args:
            solver_dcop: Converged solver DC operating point for one VMM.
            wl_logic: Binary WL logic tensor. Shape: [..., row_num].

        Returns:
            Array-internal dynamic energy [fJ]. Shape: [...].
        """

        # --- Solver outputs ---

        # Shape: [..., phys_col_num]
        i_bl_driver__uA = solver_dcop.i_bl_driver
        # Shape: [..., row_num]
        i_sl_driver__uA = solver_dcop.i_sl_driver
        # Shape: [..., phys_col_num, row_num]
        v_bl_node__V = solver_dcop.v_bl_node
        # Shape: [..., phys_col_num, row_num]
        v_x_node__V = solver_dcop.v_x_node
        # Shape: [..., phys_col_num]
        v_bl_clamp__V = solver_dcop.v_bl_clamp
        # Shape: [..., row_num]
        v_sl_drive__V = solver_dcop.v_sl_drive

        # --- Broadcast setup ---

        v_dd_wl = self.v_dd_wl__V
        # Shape: [..., row_num] -> [..., 1, row_num]
        v_wl_state1__V = (v_dd_wl * wl_logic).unsqueeze(-2)
        # Shape: [..., phys_col_num] -> [..., phys_col_num, 1]
        v_clamp__V = v_bl_clamp__V.unsqueeze(-1)

        # Every per-term ``e_*__fJ`` below collapses to ``[*batch]``;
        # leading shape comments only flag the changed (reduced) axes.

        # --- E_thermal: steady-state Joule dissipation over the WL pulse ---

        # Supply-power balance:
        #   P = V_bl · I_bl_into_array + V_sl · I_sl_into_array
        # (driver-into-wire sign convention).  Unit: uW · ns = fJ.
        # Shape: [...]
        array_power__uW = torch.sum(v_bl_clamp__V * i_bl_driver__uA, dim=-1) + torch.sum(
            v_sl_drive__V * i_sl_driver__uA, dim=-1
        )
        e_thermal__fJ = array_power__uW * self.cfg.wl_pulse_length__ns

        # --- E_WL_drive: wire + C_gs ground caps at WL ---

        # Inactive rows contribute zero (V_WL^(1) = 0 for binary DAC).
        # Unit: fF · V² = fJ.
        # Shape: [...]
        e_wl_ground__fJ = wl_logic.sum(dim=-1) * self.c_wl_per_row__fF * v_dd_wl * v_dd_wl

        # --- E_BL_recover at Node X: C_db + c_bot ground caps ---

        # E_per_cell = V_BL_clamp · C_X · (V_BL_clamp - V_X^(1)).
        delta_v_x__V = v_clamp__V - v_x_node__V
        # Shape: [...]
        e_bl_x_ground__fJ = (v_clamp__V.squeeze(-1) * self.c_x_per_cell__fF * delta_v_x__V.sum(dim=-1)).sum(dim=-1)

        # --- E_BL_recover at BL nodes: wire + c_top ground caps ---

        # E_per_node = V_BL_clamp · C_BL_node · (V_BL_clamp - V_BL^(1)).
        delta_v_bl__V = v_clamp__V - v_bl_node__V
        # Shape: [...]
        e_bl_node__fJ = (v_clamp__V * self.c_bl_per_node__fF * delta_v_bl__V).sum(dim=(-2, -1))

        # --- E_Cgd: Miller-coupled gate-drain cap ---

        # Combined WL + BL supply energy per cell:
        #   E = C_gd · (V_WL^(1) + V_BL_clamp) · (V_WL^(1) + V_BL_clamp - V_X^(1))
        sum_voltage__V = v_wl_state1__V + v_clamp__V
        delta_v_miller__V = sum_voltage__V - v_x_node__V
        # Shape: [...]
        e_cgd__fJ = (sum_voltage__V * self.c_gd_per_cell__fF * delta_v_miller__V).sum(dim=(-2, -1))

        return e_thermal__fJ + e_wl_ground__fJ + e_bl_x_ground__fJ + e_bl_node__fJ + e_cgd__fJ
