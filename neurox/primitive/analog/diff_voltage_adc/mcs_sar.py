"""V_cm-based (Merged Capacitor Switching, MCS) differential SAR voltage ADC.

See Also:
    docs/reference/primitive/analog/diff_voltage_adc/mcs_sar.md
"""

import math

import torch
from torch import Tensor

from neurox.primitive.nonideality import (
    apply_gaussian,
    apply_pelgrom_mismatch,
)
from neurox.primitive.physics import K_BOLTZMANN__J_per_K

from .base import DiffVadc, DiffVadcConfig, DiffVadcPolicy


class McsSarDiffVadcConfig(DiffVadcConfig):
    latency_per_bit__ns: float
    """SAR comparator clock period; latency at `active_bits` is
    `(active_bits + 1) · latency_per_bit`."""
    c_unit__fF: float
    """CDAC unit capacitance the binary weights multiply."""
    cap_mismatch_sigma_relative: float
    """Per-unit-cap relative Pelgrom σ."""
    comparator_offset_sigma__V: float
    """Static Gaussian σ on the comparator threshold."""
    comparator_thermal_noise_sigma__V: float
    """Per-cycle Gaussian σ for thermal comparator noise, quoted at 300 K."""
    energy_per_op__fJ: float
    """Data-independent energy charged once per conversion."""
    energy_per_bit__fJ: float
    """Per-cycle SAR strobe / logic / control overhead; charged `bits` times
    per conversion."""

    def validate(self) -> None:
        super().validate()

        # --- Topology and timing ---

        self._require_ge(self.bits, "bits", 2)
        self._require_pos(self.latency_per_bit__ns, "latency_per_bit__ns")

        # --- CDAC and comparator ---

        self._require_pos(self.c_unit__fF, "c_unit__fF")
        self._require_non_neg(self.cap_mismatch_sigma_relative, "cap_mismatch_sigma_relative")
        self._require_non_neg(self.comparator_offset_sigma__V, "comparator_offset_sigma__V")
        self._require_non_neg(self.comparator_thermal_noise_sigma__V, "comparator_thermal_noise_sigma__V")

        # --- Energy ---

        self._require_non_neg(self.energy_per_op__fJ, "energy_per_op__fJ")
        self._require_non_neg(self.energy_per_bit__fJ, "energy_per_bit__fJ")


class McsSarDiffVadcPolicy(DiffVadcPolicy):
    cap_mismatch: bool
    """Apply `cap_mismatch_sigma_relative` at fabricate time."""
    comparator_offset: bool
    """Apply `comparator_offset_sigma__V` at fabricate time."""
    comparator_thermal_noise: bool
    """Apply `comparator_thermal_noise_sigma__V` per SAR cycle."""
    sampling_thermal_noise: bool
    """Apply kT/C sampling thermal noise on the held top plates."""


@DiffVadc.register_neurox_module(
    config_type=McsSarDiffVadcConfig,
    policy_type=McsSarDiffVadcPolicy,
)
class McsSarDiffVadc(DiffVadc[McsSarDiffVadcConfig, McsSarDiffVadcPolicy]):
    """V_cm-based (MCS) differential SAR voltage ADC.

    The CDAC swings against one full-scale reference, so this converter's
    injected bank is single-tap: the sole tap sets `V_cm = V_ref / 2` and the
    per-bit switching energy.

    """

    # === Nominal buffers ===

    _nominal_c__fF: Tensor  # Shape: [cap_num]
    _nominal_comparator_offset__V: Tensor  # Shape: []

    # === Fabricated state ===

    _c_p__fF: Tensor  # Shape: [*inst_shape, cap_num]
    _c_n__fF: Tensor  # Shape: [*inst_shape, cap_num]
    _comparator_offset__V: Tensor  # Shape: [*inst_shape]

    def __init__(
        self,
        *,
        config: McsSarDiffVadcConfig,
        policy: McsSarDiffVadcPolicy,
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
            raise ValueError(f"McsSarDiffVadc T__K ({T__K}) must be > 0")

        self._T__K = T__K

        self._comparator_noise_sigma__V = config.comparator_thermal_noise_sigma__V * math.sqrt(T__K / 300.0)

        self._register_fabrication_buffers(dtype=dtype)

    def latency__ns(self, *, active_bits: int) -> float:
        """One conversion — the sample cycle plus one comparator cycle per bit.

        The SAR cycles run sequentially inside the one converter, all on the
        comparator clock, so the window is `(active_bits + 1) * clk_period`.

        Raises:
            ValueError: `active_bits` is outside `[1, bits]`.
        """
        self._check_active_bits(active_bits)
        return self.config.latency_per_bit__ns * (active_bits + 1)

    def _register_fabrication_buffers(self, *, dtype: torch.dtype) -> None:
        config = self.config
        c_unit = config.c_unit__fF
        nominal_c__fF = torch.tensor(
            [c_unit] + [c_unit * (2**k) for k in range(config.bits - 1)],
            dtype=dtype,
        )
        self._register_nonpersistent_buffer("_nominal_c__fF", nominal_c__fF)
        self._register_nonpersistent_buffer(
            "_nominal_comparator_offset__V",
            torch.zeros((), dtype=dtype),
        )

    def _sample_fabrication_variation(self) -> None:
        # Two independently-sampled cap arrays for the differential CDAC. The
        # floor keeps a Gaussian tail from sampling a non-positive cap, which
        # the step tables and the kT/C sigma both divide by.
        policy = self.policy
        self._c_p__fF = apply_pelgrom_mismatch(
            self._nominal_c__fF.clone().expand(*self.inst_shape, self.config.bits),
            self.config.cap_mismatch_sigma_relative,
            unit=self.config.c_unit__fF,
            floor=0.1 * self.config.c_unit__fF,
            enabled=policy.cap_mismatch,
        )
        self._c_n__fF = apply_pelgrom_mismatch(
            self._nominal_c__fF.clone().expand(*self.inst_shape, self.config.bits),
            self.config.cap_mismatch_sigma_relative,
            unit=self.config.c_unit__fF,
            floor=0.1 * self.config.c_unit__fF,
            enabled=policy.cap_mismatch,
        )
        self._comparator_offset__V = apply_gaussian(
            self._nominal_comparator_offset__V.clone().expand(self.inst_shape),
            self.config.comparator_offset_sigma__V,
            enabled=policy.comparator_offset,
        )

    def _convert_impl(
        self,
        v_pos__V: Tensor,
        v_neg__V: Tensor,
        *,
        v_refs__V: Tensor,
        active_bits: int,
    ) -> Tensor:
        """V_cm-based (MCS) differential SAR conversion.

        Args:
            v_pos__V: Positive-side input voltage.
            v_neg__V: Negative-side input voltage, at the same shape.
            v_refs__V: Injected reference taps; the CDAC swings against one
                full-scale reference, so the single tap is read off the last
                axis and the leading dims broadcast against the inputs.
                Shape: `[..., 1]`.
            active_bits: Active conversion resolution in `[1, bits]`.

        Returns:
            Raw offset-binary code tensor valued in `[0, 2 ** active_bits - 1]`, one
            code per `v_pos__V` element.

        Raises:
            ValueError: `active_bits` is outside `[1, bits]`, or `v_refs__V` does
                not hold exactly one tap on its last axis.
        """
        self._validate_runtime_args(v_refs__V)

        # Shape: [..., 1] -> [...]
        v_ref__V = v_refs__V[..., 0]
        v_cm__V = 0.5 * v_ref__V

        c_p__fF = self._c_p__fF
        c_n__fF = self._c_n__fF
        # Shape: [..., cap_num] -> [...]
        c_p_total__fF = c_p__fF.sum(dim=-1)
        # Shape: [..., cap_num] -> [...]
        c_n_total__fF = c_n__fF.sum(dim=-1)

        # --- 1: sample and hold ---

        # sample: bottom (drive): V_in, top (drive): V_cm
        # hold: bottom (drive): V_cm, top (float): 2*V_cm-V_in
        v_p_top__V = 2 * v_cm__V - v_pos__V
        v_n_top__V = 2 * v_cm__V - v_neg__V

        # sample thermal noise on each held top plate (per-leg kT/C).
        kt__fJ = K_BOLTZMANN__J_per_K * self._T__K * 1e15
        v_p_top__V = apply_gaussian(
            v_p_top__V, torch.sqrt(kt__fJ / c_p_total__fF), enabled=self.policy.sampling_thermal_noise
        )
        v_n_top__V = apply_gaussian(
            v_n_top__V, torch.sqrt(kt__fJ / c_n_total__fF), enabled=self.policy.sampling_thermal_noise
        )

        # --- 2: resolve the MSB without capacitor switching ---

        # neg cap top to comparator Vin+, pos cap top to comparator Vin-
        last_bit = self._compare(v_pos__V=v_n_top__V, v_neg__V=v_p_top__V)
        code = last_bit.to(torch.int32)

        # --- 3: precompute loop-invariant per-bit constants ---

        # The active capacitor slice is indexed directly by the SAR bit.
        cap_lo = self.bits - active_bits + 1
        # Shape: [..., cap_num] -> [..., active_bits-1]
        c_p_used__fF = c_p__fF[..., cap_lo : self.bits]
        # Shape: [..., cap_num] -> [..., active_bits-1]
        c_n_used__fF = c_n__fF[..., cap_lo : self.bits]
        # Shape: [...] -> [..., 1]
        c_p_total_e__fF = c_p_total__fF.unsqueeze(-1)
        # Shape: [...] -> [..., 1]
        c_n_total_e__fF = c_n_total__fF.unsqueeze(-1)
        v_p_step_table__V = v_cm__V * c_p_used__fF / c_p_total_e__fF
        v_n_step_table__V = v_cm__V * c_n_used__fF / c_n_total_e__fF

        # --- 4: run the SAR decisions ---

        for k in range(active_bits - 2, -1, -1):
            # Shape: [..., active_bits-1] -> [...]
            v_p_step__V = v_p_step_table__V[..., k]
            # Shape: [..., active_bits-1] -> [...]
            v_n_step__V = v_n_step_table__V[..., k]
            v_p_top__V = torch.where(last_bit, v_p_top__V + v_p_step__V, v_p_top__V - v_p_step__V)
            v_n_top__V = torch.where(last_bit, v_n_top__V - v_n_step__V, v_n_top__V + v_n_step__V)
            # neg cap top to comparator Vin+, pos cap top to comparator Vin-
            last_bit = self._compare(v_pos__V=v_n_top__V, v_neg__V=v_p_top__V)
            code = (code << 1) | last_bit.to(torch.int32)

        # --- 5: record dynamic energy when requested ---

        if self._is_dynamic_energy_profile_active():
            e_p_sample__fJ = c_p_total__fF * v_pos__V * torch.clamp_min(v_pos__V - v_cm__V, 0)
            e_n_sample__fJ = c_n_total__fF * v_neg__V * torch.clamp_min(v_neg__V - v_cm__V, 0)
            e_sample__fJ = e_p_sample__fJ + e_n_sample__fJ + self.config.energy_per_op__fJ

            e_step_p_table__fJ = 0.5 * v_ref__V**2 * c_p_used__fF * (1 - c_p_used__fF / c_p_total_e__fF)
            e_step_n_table__fJ = 0.5 * v_ref__V**2 * c_n_used__fF * (1 - c_n_used__fF / c_n_total_e__fF)
            c_diff_step_table__fF = c_p_used__fF - c_n_used__fF

            shifts = torch.arange(1, active_bits, device=code.device, dtype=code.dtype)
            # Shape: [...] -> [..., active_bits-1]
            bit_seq = ((code.unsqueeze(-1) >> shifts) & 1).to(torch.bool)
            # Shape: [..., active_bits-1] -> [...]
            e_detect__fJ = (
                torch.where(bit_seq, e_step_p_table__fJ, e_step_n_table__fJ).sum(dim=-1)
                + active_bits * self.config.energy_per_bit__fJ
            )
            # Shape: [..., active_bits-1] -> [...]
            c_diff__fF = (torch.where(bit_seq, -0.5, 0.5) * c_diff_step_table__fF).sum(dim=-1)
            e_reset__fJ = torch.abs(0.5 * v_ref__V**2 * c_diff__fF)
            self._record_dynamic_energy(e_sample__fJ + e_detect__fJ + e_reset__fJ)

        return code

    def _compare(self, v_pos__V: Tensor, v_neg__V: Tensor) -> Tensor:
        """Strobe the differential comparator.

        Adds per-cycle thermal noise to the differential top-plate voltage and
        compares against the fabricated comparator offset.

        Returns:
            Bool tensor; `True` means the positive leg won.
        """
        v_diff__V = apply_gaussian(
            v_pos__V - v_neg__V,
            self._comparator_noise_sigma__V,
            enabled=self.policy.comparator_thermal_noise,
        )
        return v_diff__V > self._comparator_offset__V

    def _validate_runtime_args(self, v_refs__V: Tensor) -> None:
        # One full-scale reference feeds the CDAC, so the bank is single-tap.
        tap_num = int(v_refs__V.shape[-1]) if v_refs__V.ndim else 0
        if tap_num != 1:
            raise ValueError(f"require: v_refs__V tap_num ({tap_num}) == 1")
