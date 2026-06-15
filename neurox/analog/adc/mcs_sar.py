"""V_cm-based (Merged Capacitor Switching, MCS) differential SAR ADC.

See also:
    docs/modules/analog/adc/mcs_sar.md
"""

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from neurox.common.nonideality import (
    apply_gaussian,
    apply_lsb_jitter,
    apply_pelgrom_mismatch,
)
from neurox.common.physical_constant import K_BOLTZMANN__J_per_K

from .base import ADC, ADCConfig, AdcOperationPoint, ADCPolicy


@dataclass(frozen=True)
class McsSarAdcConfig(ADCConfig):
    """Immutable design-parameter config for :class:`McsSarAdc`.

    Per-op latency is parametric: each ``convert`` call computes
    ``(adc_operation_point.adc_bits + 1) · clk_period__ns`` and feeds
    it to ``_log_latency`` — there is no ``latency_per_op__ns`` field
    because the value is not knowable until the runtime op point is
    chosen.

    Attributes:
        max_bits: Physical bit width; active array carries
            ``max_bits - 1`` binary-weighted caps + a dummy cap
            (MSB-free design).
        v_refs__V: Supported reference voltages [V], strictly
            decreasing — ``v_refs__V[0]`` is the calibration anchor.
        clk_period__ns: SAR comparator clock period [ns]; latency at
            ``bits`` active bits is ``(bits + 1) · clk_period``.
        c_unit__fF: CDAC unit capacitance [fF].
        cap_mismatch_sigma_relative: Per-unit-cap relative Pelgrom
            sigma.
        comparator_offset_sigma__V: Static Gaussian sigma on the
            comparator threshold [V].
        comparator_thermal_noise_sigma__V: Per-cycle Gaussian sigma
            for thermal comparator noise [V].
        e_bootstrap__fJ: Per-conversion bootstrapped sampling-switch
            overhead [fJ].
        e_constant_per_bit__fJ: Per-cycle SAR strobe / logic / control
            overhead [fJ]; charged ``bits`` times per conversion.
    """

    # --- Topology ---
    max_bits: int

    # --- References + timing ---
    v_refs__V: tuple[float, ...]
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
        self.validate_refs()
        self.validate_timing()
        self.validate_cdac()
        self.validate_comparator()
        self.validate_energy()
        self.validate_ppa()

    def validate_topology(self) -> None:
        if not (self.max_bits >= 2):
            raise ValueError(f"require: max_bits ({self.max_bits}) >= 2")

    def validate_refs(self) -> None:
        self._require_min_length(self.v_refs__V, 1, "v_refs__V")
        for i, v in enumerate(self.v_refs__V):
            self._require_pos(v, f"v_refs__V[{i}]")
        self._require_strictly_decreasing(self.v_refs__V, "v_refs__V")

    def validate_timing(self) -> None:
        self._require_pos(self.clk_period__ns, "clk_period__ns")

    def validate_cdac(self) -> None:
        self._require_pos(self.c_unit__fF, "c_unit__fF")
        self._require_nonneg(self.cap_mismatch_sigma_relative, "cap_mismatch_sigma_relative")

    def validate_comparator(self) -> None:
        self._require_nonneg(self.comparator_offset_sigma__V, "comparator_offset_sigma__V")
        self._require_nonneg(self.comparator_thermal_noise_sigma__V, "comparator_thermal_noise_sigma__V")

    def validate_energy(self) -> None:
        self._require_nonneg(self.e_bootstrap__fJ, "e_bootstrap__fJ")
        self._require_nonneg(self.e_constant_per_bit__fJ, "e_constant_per_bit__fJ")


@dataclass(frozen=True)
class McsSarAdcPolicy(ADCPolicy):
    """Per-source toggles selecting which McsSarAdc nonidealities are active.

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


@ADC.register_key(McsSarAdcConfig)
class McsSarAdc(ADC):
    """V_cm-based (MCS) differential SAR ADC.

    Args:
        config: Concrete configuration dataclass.
        policy: Per-source nonideality enable flags.
        name: Hierarchical instance name used by the profiler.
        dtype: Tensor dtype for internal buffers.
        T__K: Operating temperature [K].
    """

    config: McsSarAdcConfig
    nominal_c__fF: Tensor
    nominal_comparator_offset__V: Tensor
    c_p__fF: Tensor
    c_n__fF: Tensor
    comparator_offset__V: Tensor

    def __init__(
        self,
        *,
        config: McsSarAdcConfig,
        policy: McsSarAdcPolicy,
        name: str,
        inst_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
    ) -> None:
        super().__init__(
            config=config,
            policy=policy,
            name=name,
            inst_shape=inst_shape,
            dtype=dtype,
            T__K=T__K,
        )
        if not (T__K > 0.0):
            raise ValueError(f"McsSarAdc T__K ({T__K}) must be > 0")

        self.policy = policy
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
        # SymInt that dynamo derives from ``adc_operation_point.adc_bits``
        # (dynamo's SymInt lshift lowering currently mishandles it).
        # ``unsigned_max_table[b] = 2**b - 1`` clamps offset-binary code
        # range; ``zero_offset_table[b] = 2**(b-1)`` is the offset-binary
        # → two's-complement bias subtracted in ``return code - offset``
        # (semantically, an MSB flip; subtraction is the implementation
        # that preserves the int32 storage representation of negatives).
        self._unsigned_max_table: tuple[int, ...] = tuple(
            ((1 << b) - 1) if b >= 1 else 0 for b in range(config.max_bits + 1)
        )
        self._zero_offset_table: tuple[int, ...] = tuple(
            (1 << (b - 1)) if b >= 1 else 0 for b in range(config.max_bits + 1)
        )

    # --- runtime-mode introspection ---

    def available_modes(self) -> tuple[float, ...]:
        """V_ref values the configured CDAC supports, in index order."""
        return self.config.v_refs__V

    @property
    def mode_num(self) -> int:
        """Number of operating points — one per supported V_ref."""
        return len(self.config.v_refs__V)

    def signed_range(self, adc_bits: int) -> tuple[int, int]:
        """Canonical SAR signed-bit endpoints at ``adc_bits``.

        The CDAC's code count is ``2 ** adc_bits`` by construction, so
        the realisable signed range is exactly
        ``(-2 ** (adc_bits - 1), 2 ** (adc_bits - 1) - 1)``.
        """
        if not (1 <= adc_bits <= self.max_bits):
            raise ValueError(f"adc_bits {adc_bits} outside [1, {self.max_bits}]")
        half = 1 << (adc_bits - 1)
        return -half, half - 1

    @property
    def max_bits(self) -> int:
        """Physical CDAC bit width — the maximum ``adc_bits`` value."""
        return self.config.max_bits

    # --- fabricate (static non-idealities) ---

    def _sample_fabricate_mismatch(self) -> None:
        """Resample independent differential cap arrays and comparator offset."""
        config = self.config
        inst_shape = self._inst_shape

        # Two independently-sampled cap arrays for the differential CDAC.
        policy = self.policy
        self.c_p__fF = apply_pelgrom_mismatch(
            self.nominal_c__fF.clone().expand(*inst_shape, self.n_caps),
            config.cap_mismatch_sigma_relative,
            unit=config.c_unit__fF,
            floor=0.1 * config.c_unit__fF,
            enabled=policy.cap_mismatch,
        )
        self.c_n__fF = apply_pelgrom_mismatch(
            self.nominal_c__fF.clone().expand(*inst_shape, self.n_caps),
            config.cap_mismatch_sigma_relative,
            unit=config.c_unit__fF,
            floor=0.1 * config.c_unit__fF,
            enabled=policy.cap_mismatch,
        )
        self.comparator_offset__V = apply_gaussian(
            self.nominal_comparator_offset__V.clone().expand(inst_shape),
            config.comparator_offset_sigma__V,
            enabled=policy.comparator_offset,
        )

    # --- convert ---

    def convert(
        self,
        v_pos__V: Tensor,
        v_neg__V: Tensor,
        *,
        adc_operation_point: AdcOperationPoint,
    ) -> Tensor:
        """V_cm-based (MCS) differential SAR conversion.

        Args:
            v_pos__V: Positive-side input voltage.
            v_neg__V: Negative-side input voltage, same shape.
            adc_operation_point: Runtime operating point. ``adc_operation_point.adc_mode`` selects V_ref;
                ``adc_operation_point.adc_bits`` sets active resolution.

        Returns:
            Code tensor in ``[0, 2 ** adc_operation_point.adc_bits - 1]``.
        """
        self._validate_runtime_args(adc_operation_point)
        bits = adc_operation_point.adc_bits

        config = self.config
        v_ref__V = config.v_refs__V[adc_operation_point.adc_mode]
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

        # sample thermal noise: sigma_V² = k_B · T / C_total.
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
        e_sample__fJ = e_p_sample__fJ + e_n_sample__fJ + config.e_bootstrap__fJ

        # --- 2. MSB decision (free, no cap switch) ---

        # neg cap top to comparator Vin+, pos cap top to comparator Vin-
        last_bit = self._compare(v_pos__V=v_n_top__V, v_neg__V=v_p_top__V)
        code = last_bit.to(torch.int32)

        # --- 3. Hoist loop-invariant per-bit constants ---

        # The SAR loop reads c_p/c_n at idx = (max_bits - bits + 1) + k for
        # k = bits-2 .. 0; slicing once gives a [..., bits-1] table the loop
        # can index by k directly. v_p_step, v_n_step, the switch-energy and
        # c_diff increments all depend only on these caps + v_ref / v_cm
        # (runtime-input-independent), so they are precomputed here. Keeping
        # only the where / compare / shift inside the loop body shortens the
        # unrolled inductor graph and is the precondition for lifting the
        # ``@torch.compiler.disable`` on ``Offset1T1RXbar.vec_mat_mul``.
        cap_lo = config.max_bits - bits + 1
        c_p_used__fF = c_p__fF[..., cap_lo : config.max_bits]
        c_n_used__fF = c_n__fF[..., cap_lo : config.max_bits]
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
            + bits * config.e_constant_per_bit__fJ
        )
        c_diff__fF = (torch.where(bit_seq, -0.5, 0.5) * c_diff_step_table__fF).sum(dim=-1)

        # --- 6. Reset: dissipate residual differential charge ---

        e_reset__fJ = torch.abs(0.5 * v_ref__V**2 * c_diff__fF)

        e_dynamic__fJ = e_sample__fJ + e_detect__fJ + e_reset__fJ

        # --- 7. Final code: optional stochastic LSB jitter ---

        # The SAR loop output is in [0, 2**bits - 1] by construction (each
        # iter ORs in a 0/1 bit), and ``apply_lsb_jitter`` clamps back
        # into that range after the +1 overflow case — so no extra clamp
        # is needed here. Both tail-end constants come from the
        # per-resolution Python int tables prepared in ``__init__``;
        # this keeps the compiled graph free of ``1 << <SymInt>`` ops.
        code = apply_lsb_jitter(
            code,
            unsigned_max=self._unsigned_max_table[bits],
            enabled=self.training,
        )
        # McsSarAdc: code carries no extra parallel trailing beyond
        # inst_shape; serial count via the position-invariant numel
        # rule. Per-op latency is parametric in the runtime bit width:
        # one sample cycle + `bits` SAR comparisons → (bits + 1) clocks.
        serial_op_count = max(1, code.numel() // max(self.inst_count, 1))
        per_op_latency__ns = (bits + 1) * config.clk_period__ns
        latency__ns = torch.tensor(
            per_op_latency__ns * serial_op_count,
            device=code.device,
            dtype=e_dynamic__fJ.dtype,
        )
        self._log_dynamic_energy(e_dynamic__fJ)
        self._log_latency(latency__ns)
        # Offset-binary → two's-complement bias (semantically: MSB flip).
        return code - self._zero_offset_table[bits]

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

    def _validate_runtime_args(self, adc_operation_point: AdcOperationPoint) -> None:
        """Validate per-call ``adc_operation_point``."""
        config = self.config
        bits = adc_operation_point.adc_bits
        mode = adc_operation_point.adc_mode
        if not (0 <= mode < len(config.v_refs__V)):
            raise ValueError(f"mode {mode} outside [0, {len(config.v_refs__V)})")
        if not (1 <= bits <= config.max_bits):
            raise ValueError(f"bits {bits} outside [1, {config.max_bits}]")
