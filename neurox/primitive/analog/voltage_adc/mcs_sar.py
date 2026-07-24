"""V_cm-based (Merged Capacitor Switching, MCS) differential SAR voltage ADC.

See also:
    docs/reference/primitive/analog/voltage_adc/mcs_sar.md
"""

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from neurox.primitive.nonideality import (
    apply_gaussian,
    apply_lsb_jitter,
    apply_pelgrom_mismatch,
)
from neurox.primitive.physical_constant import K_BOLTZMANN__J_per_K

from .base import DifferentialVoltageAdc, DifferentialVoltageAdcConfig, DifferentialVoltageAdcPolicy


@dataclass(frozen=True)
class McsSarDifferentialVoltageAdcConfig(DifferentialVoltageAdcConfig):
    """Immutable design-parameter config for :class:`McsSarDifferentialVoltageAdc`.

    Attributes:
        max_bits: Physical bit width; active array carries
            ``max_bits - 1`` binary-weighted caps + a dummy cap
            (MSB-free design).
        clk_period__ns: SAR comparator clock period; latency at
            ``bits`` active bits is ``(bits + 1) · clk_period``.
        c_unit__fF: CDAC unit capacitance.
        cap_mismatch_sigma_relative: Per-unit-cap relative Pelgrom
            σ.
        comparator_offset_sigma__V: Static Gaussian σ on the
            comparator threshold.
        comparator_thermal_noise_sigma__V: Per-cycle Gaussian σ
            for thermal comparator noise.
        e_bootstrap__fJ: Per-conversion bootstrapped sampling-switch
            overhead.
        e_constant_per_bit__fJ: Per-cycle SAR strobe / logic / control
            overhead; charged ``bits`` times per conversion.
    """

    # --- Topology ---
    max_bits: int

    # --- Timing ---
    clk_period__ns: float

    # --- CDAC unit ---
    c_unit__fF: float

    # --- Cap mismatch (Pelgrom) ---
    cap_mismatch_sigma_relative: float

    # --- Comparator static offset ---
    comparator_offset_sigma__V: float

    # --- Comparator thermal noise (per-cycle) ---
    comparator_thermal_noise_sigma__V: float

    # --- Energy ---
    e_bootstrap__fJ: float
    e_constant_per_bit__fJ: float

    def validate(self) -> None:
        super().validate()
        self.validate_topology()
        self.validate_timing()
        self.validate_cdac()
        self.validate_comparator()
        self.validate_energy()
        self.validate_ppa()

    def validate_topology(self) -> None:
        if not (self.max_bits >= 2):
            raise ValueError(f"require: max_bits ({self.max_bits}) >= 2")

    def validate_timing(self) -> None:
        self._require_pos(self.clk_period__ns, "clk_period__ns")

    def validate_cdac(self) -> None:
        self._require_pos(self.c_unit__fF, "c_unit__fF")
        self._require_non_neg(self.cap_mismatch_sigma_relative, "cap_mismatch_sigma_relative")

    def validate_comparator(self) -> None:
        self._require_non_neg(self.comparator_offset_sigma__V, "comparator_offset_sigma__V")
        self._require_non_neg(self.comparator_thermal_noise_sigma__V, "comparator_thermal_noise_sigma__V")

    def validate_energy(self) -> None:
        self._require_non_neg(self.e_bootstrap__fJ, "e_bootstrap__fJ")
        self._require_non_neg(self.e_constant_per_bit__fJ, "e_constant_per_bit__fJ")


@dataclass(frozen=True)
class McsSarDifferentialVoltageAdcPolicy(DifferentialVoltageAdcPolicy):
    """Per-source toggles selecting which McsSarDifferentialVoltageAdc nonidealities are active.

    Attributes:
        cap_mismatch: Apply ``cap_mismatch_sigma_relative`` at fabricate time.
        comparator_offset: Apply ``comparator_offset_sigma__V`` at fabricate time.
        comparator_thermal_noise: Apply ``comparator_thermal_noise_sigma__V`` per SAR cycle.
        sampling_thermal_noise: Apply kT/C sampling thermal noise on the held top plates.
    """

    cap_mismatch: bool
    comparator_offset: bool
    comparator_thermal_noise: bool
    sampling_thermal_noise: bool


@DifferentialVoltageAdc.register_key(McsSarDifferentialVoltageAdcConfig)
class McsSarDifferentialVoltageAdc(
    DifferentialVoltageAdc[McsSarDifferentialVoltageAdcConfig, McsSarDifferentialVoltageAdcPolicy]
):
    """V_cm-based (MCS) differential SAR voltage ADC.

    Args:
        config: Concrete configuration dataclass.
        policy: Per-source nonideality enable flags.
        inst_shape: Per-instance fabrication shape.
        dtype: Tensor dtype for internal buffers.
        T__K: Operating temperature.
    """

    nominal_c__fF: Tensor
    nominal_comparator_offset__V: Tensor
    c_p__fF: Tensor
    c_n__fF: Tensor
    comparator_offset__V: Tensor

    def __init__(
        self,
        *,
        config: McsSarDifferentialVoltageAdcConfig,
        policy: McsSarDifferentialVoltageAdcPolicy,
        inst_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
    ) -> None:
        super().__init__(
            config=config,
            policy=policy,
            inst_shape=inst_shape,
            dtype=dtype,
            T__K=T__K,
        )
        if not (T__K > 0.0):
            raise ValueError(f"McsSarDifferentialVoltageAdc T__K ({T__K}) must be > 0")

        self._area_per_inst__um2 = config.area_per_inst__um2
        self._leakage_per_inst__uW = config.leakage_per_inst__uW
        self.T__K = T__K
        self.dtype = dtype

        self.comparator_noise_sigma__V = config.comparator_thermal_noise_sigma__V * math.sqrt(T__K / 300.0)

        self.n_caps = config.max_bits

        c_unit = config.c_unit__fF
        nominal_c__fF = torch.tensor(
            [c_unit] + [c_unit * (2**k) for k in range(config.max_bits - 1)],
            dtype=dtype,
        )
        self.register_buffer("nominal_c__fF", nominal_c__fF, persistent=False)
        self.register_buffer(
            "nominal_comparator_offset__V",
            torch.zeros((), dtype=dtype),
            persistent=False,
        )

        self.register_buffer(
            "c_p__fF",
            self.nominal_c__fF.clone(),
            persistent=False,
        )
        self.register_buffer(
            "c_n__fF",
            self.nominal_c__fF.clone(),
            persistent=False,
        )
        self.register_buffer(
            "comparator_offset__V",
            self.nominal_comparator_offset__V.clone(),
            persistent=False,
        )

        # Precompute per-resolution Python int tables so the runtime path
        # ``convert(...)`` never evaluates ``1 << bits`` against the
        # SymInt that dynamo derives from ``bits``
        # (dynamo's SymInt lshift lowering currently mishandles it).
        # ``unsigned_max_table[b] = 2**b - 1`` clamps the offset-binary code
        # range; ``zero_offset_table[b] = 2**(b-1)`` is the raw code that
        # represents analog zero, exposed via :meth:`zero_offset` for the
        # consumer to subtract affinely (the ADC returns the raw code and
        # does NOT fold the offset in).
        self._unsigned_max_table: tuple[int, ...] = tuple(
            ((1 << b) - 1) if b >= 1 else 0 for b in range(config.max_bits + 1)
        )
        self._zero_offset_table: tuple[int, ...] = tuple(
            (1 << (b - 1)) if b >= 1 else 0 for b in range(config.max_bits + 1)
        )

    # --- runtime-mode introspection ---

    def unsigned_range(self, bits: int) -> tuple[int, int]:
        """Raw offset-binary code endpoints at ``bits`` — ``(0, 2 ** bits - 1)``.

        The CDAC's code count is ``2 ** bits`` by construction.
        """
        if not (1 <= bits <= self.max_bits):
            raise ValueError(f"bits {bits} outside [1, {self.max_bits}]")
        return 0, self._unsigned_max_table[bits]

    def zero_offset(self, bits: int) -> int:
        """Offset-binary zero code at ``bits`` — ``2 ** (bits - 1)``."""
        if not (1 <= bits <= self.max_bits):
            raise ValueError(f"bits {bits} outside [1, {self.max_bits}]")
        return self._zero_offset_table[bits]

    @property
    def max_bits(self) -> int:
        """Physical CDAC bit width — the maximum ``bits`` value."""
        return self.config.max_bits

    # --- fabricate (static non-idealities) ---

    def _sample_fabricate_mismatch(self) -> None:
        """Resample independent differential cap arrays and comparator offset."""

        # Two independently-sampled cap arrays for the differential CDAC.
        policy = self.policy
        self.c_p__fF = apply_pelgrom_mismatch(
            self.nominal_c__fF.clone().expand(*self.inst_shape, self.n_caps),
            self.config.cap_mismatch_sigma_relative,
            unit=self.config.c_unit__fF,
            floor=0.1 * self.config.c_unit__fF,
            enabled=policy.cap_mismatch,
        )
        self.c_n__fF = apply_pelgrom_mismatch(
            self.nominal_c__fF.clone().expand(*self.inst_shape, self.n_caps),
            self.config.cap_mismatch_sigma_relative,
            unit=self.config.c_unit__fF,
            floor=0.1 * self.config.c_unit__fF,
            enabled=policy.cap_mismatch,
        )
        self.comparator_offset__V = apply_gaussian(
            self.nominal_comparator_offset__V.clone().expand(self.inst_shape),
            self.config.comparator_offset_sigma__V,
            enabled=policy.comparator_offset,
        )

    # --- convert ---

    def _convert_impl(
        self,
        v_pos__V: Tensor,
        v_neg__V: Tensor,
        *,
        v_ref__V: Tensor,
        bits: int,
    ) -> Tensor:
        """V_cm-based (MCS) differential SAR conversion.

        Args:
            v_pos__V: Positive-side input voltage.
            v_neg__V: Negative-side input voltage, same shape.
            v_ref__V: The owner-preselected single reference tap, shape
                ``(*inst,)``. The ADC is mode-blind.
            bits: Active resolution [bits].

        Returns:
            Raw offset-binary code tensor in ``[0, 2 ** bits - 1]``
            (see :meth:`unsigned_range`). The zero point
            (:meth:`zero_offset`) is subtracted consumer-side, not here.
        """
        self._validate_runtime_args(bits)

        v_cm__V = 0.5 * v_ref__V

        c_p__fF = self.c_p__fF
        c_n__fF = self.c_n__fF
        c_p_total__fF = c_p__fF.sum(dim=-1)
        c_n_total__fF = c_n__fF.sum(dim=-1)

        # --- 1. Sample and hold ---

        # sample: bottom (drive): V_in, top (drive): V_cm
        # hold: bottom (drive): V_cm, top (float): 2*V_cm-V_in
        v_p_top__V = 2 * v_cm__V - v_pos__V
        v_n_top__V = 2 * v_cm__V - v_neg__V

        # sample thermal noise on each held top plate (per-leg kT/C).
        kt__fJ = K_BOLTZMANN__J_per_K * self.T__K * 1e15
        v_p_top__V = apply_gaussian(
            v_p_top__V, torch.sqrt(kt__fJ / c_p_total__fF), enabled=self.policy.sampling_thermal_noise
        )
        v_n_top__V = apply_gaussian(
            v_n_top__V, torch.sqrt(kt__fJ / c_n_total__fF), enabled=self.policy.sampling_thermal_noise
        )

        # Sample energy: input source charges the bottom-plate caps from V_cm to V_in.
        e_p_sample__fJ = c_p_total__fF * v_pos__V * torch.clamp_min(v_pos__V - v_cm__V, 0)
        e_n_sample__fJ = c_n_total__fF * v_neg__V * torch.clamp_min(v_neg__V - v_cm__V, 0)
        e_sample__fJ = e_p_sample__fJ + e_n_sample__fJ + self.config.e_bootstrap__fJ

        # --- 2. MSB decision (free, no cap switch) ---

        # neg cap top to comparator Vin+, pos cap top to comparator Vin-
        last_bit = self._compare(v_pos__V=v_n_top__V, v_neg__V=v_p_top__V)
        code = last_bit.to(torch.int32)

        # --- 3. Hoist loop-invariant per-bit constants ---

        # The SAR loop reads c_p/c_n at idx = (max_bits - bits + 1) + k for
        # k = bits-2 .. 0; slicing once gives a [..., bits-1] table the loop
        # can index by k directly. v_p_step, v_n_step, the switch-energy and
        # c_diff increments depend only on these caps + v_ref / v_cm
        # (runtime-input-independent), so they are precomputed here.
        cap_lo = self.config.max_bits - bits + 1
        c_p_used__fF = c_p__fF[..., cap_lo : self.config.max_bits]
        c_n_used__fF = c_n__fF[..., cap_lo : self.config.max_bits]
        c_p_total_e__fF = c_p_total__fF.unsqueeze(-1)
        c_n_total_e__fF = c_n_total__fF.unsqueeze(-1)
        v_p_step_table__V = v_cm__V * c_p_used__fF / c_p_total_e__fF
        v_n_step_table__V = v_cm__V * c_n_used__fF / c_n_total_e__fF
        e_step_p_table__fJ = 0.5 * v_ref__V**2 * c_p_used__fF * (1 - c_p_used__fF / c_p_total_e__fF)
        e_step_n_table__fJ = 0.5 * v_ref__V**2 * c_n_used__fF * (1 - c_n_used__fF / c_n_total_e__fF)
        c_diff_step_table__fF = c_p_used__fF - c_n_used__fF

        # --- 4. SAR loop (per-cycle comparator noise preserved) ---

        for k in range(bits - 2, -1, -1):
            v_p_step__V = v_p_step_table__V[..., k]
            v_n_step__V = v_n_step_table__V[..., k]
            v_p_top__V = torch.where(last_bit, v_p_top__V + v_p_step__V, v_p_top__V - v_p_step__V)
            v_n_top__V = torch.where(last_bit, v_n_top__V - v_n_step__V, v_n_top__V + v_n_step__V)
            # neg cap top to comparator Vin+, pos cap top to comparator Vin-
            last_bit = self._compare(v_pos__V=v_n_top__V, v_neg__V=v_p_top__V)
            code = (code << 1) | last_bit.to(torch.int32)

        # --- 5. Vectorised side-channel updates ---

        shifts = torch.arange(1, bits, device=code.device, dtype=code.dtype)
        bit_seq = ((code.unsqueeze(-1) >> shifts) & 1).to(torch.bool)
        e_detect__fJ = (
            torch.where(bit_seq, e_step_p_table__fJ, e_step_n_table__fJ).sum(dim=-1)
            + bits * self.config.e_constant_per_bit__fJ
        )
        c_diff__fF = (torch.where(bit_seq, -0.5, 0.5) * c_diff_step_table__fF).sum(dim=-1)

        # --- 6. Reset: dissipate residual differential charge ---

        e_reset__fJ = torch.abs(0.5 * v_ref__V**2 * c_diff__fF)

        e_dynamic__fJ = e_sample__fJ + e_detect__fJ + e_reset__fJ

        # --- 7. Final code: optional stochastic LSB jitter ---

        # The SAR loop output is in [0, 2**bits - 1] by construction (each
        # iter ORs in a 0/1 bit), and ``apply_lsb_jitter`` clamps back
        # into that range after the +1 overflow case — so no extra clamp
        # is needed here.
        code = apply_lsb_jitter(
            code,
            unsigned_max=self._unsigned_max_table[bits],
            enabled=self.training,
        )
        # McsSarDifferentialVoltageAdc: code carries no extra parallel trailing beyond
        # inst_shape; serial count via the position-invariant numel
        # rule. Per-op latency is parametric in the runtime bit width:
        # one sample cycle + `bits` SAR comparisons → (bits + 1) clocks.
        serial_op_count = max(1, code.numel() // max(self.inst_count, 1))
        per_op_latency__ns = (bits + 1) * self.config.clk_period__ns
        latency__ns = torch.tensor(
            per_op_latency__ns * serial_op_count,
            device=code.device,
            dtype=e_dynamic__fJ.dtype,
        )
        self._log_dynamic_energy(e_dynamic__fJ)
        self._log_latency(latency__ns)
        # Raw offset-binary code; the zero point is subtracted consumer-side.
        return code

    def _compare(self, v_pos__V: Tensor, v_neg__V: Tensor) -> Tensor:
        """Strobe the differential comparator.

        Adds per-cycle thermal noise to the differential voltage and
        compares against ``comparator_offset__V``.

        Args:
            v_pos__V: Positive-side top-plate voltage.
            v_neg__V: Negative-side top-plate voltage.

        Returns:
            Bool tensor; ``True`` means the positive leg won.
        """
        v_diff__V = apply_gaussian(
            v_pos__V - v_neg__V,
            self.comparator_noise_sigma__V,
            enabled=self.policy.comparator_thermal_noise,
        )
        return v_diff__V > self.comparator_offset__V

    # --- shared helpers ---

    def _validate_runtime_args(self, bits: int) -> None:
        """Validate the per-call bit width against the config bound."""
        if not (1 <= bits <= self.config.max_bits):
            raise ValueError(f"bits {bits} outside [1, {self.config.max_bits}]")
