"""Offset-coded readout chain: switch-cap banks, mux, and ADC.

See also:
    docs/modules/xbar/readout/offset_switchcap_mux_adc.md
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from neurox.analog.adc import ADC, ADCConfig, AdcOperationPoint, ADCPolicy
from neurox.analog.analog_mux import AnalogMux, AnalogMuxConfig, AnalogMuxPolicy
from neurox.analog.switch_cap import SwitchCap, SwitchCapConfig, SwitchCapPolicy

from .base import ReadOut, ReadOutConfig, ReadOutPolicy


@dataclass(frozen=True, kw_only=True)
class OffsetSwitchCapMuxAdcReadOutConfig(ReadOutConfig):
    """Config for :class:`OffsetSwitchCapMuxAdcReadOut`.

    Attributes:
        data_switchcap_config: Per-digit data-leg switch-cap bank config.
        ref_switchcap_config: Per-group ref-leg switch-cap bank config.
        analog_mux_config: Analog mux config.
        adc_config: Inner ADC config.
    """

    data_switchcap_config: SwitchCapConfig
    ref_switchcap_config: SwitchCapConfig
    analog_mux_config: AnalogMuxConfig
    adc_config: ADCConfig


@dataclass(frozen=True)
class OffsetSwitchCapMuxAdcReadOutPolicy(ReadOutPolicy):
    """Composite policy for :class:`OffsetSwitchCapMuxAdcReadOut`.

    Attributes:
        data_switchcap: Data-leg switch-cap bank nonideality policy.
        ref_switchcap: Reference-leg switch-cap bank nonideality policy.
        analog_mux: Analog mux nonideality policy.
        bl_adc: Inner ADC nonideality policy.
    """

    data_switchcap: SwitchCapPolicy
    ref_switchcap: SwitchCapPolicy
    analog_mux: AnalogMuxPolicy
    bl_adc: ADCPolicy


@ReadOut.register_key(OffsetSwitchCapMuxAdcReadOutConfig)
class OffsetSwitchCapMuxAdcReadOut(ReadOut):
    """Offset-coded readout: data and ref switch-cap banks, mux, and ADC."""

    config: OffsetSwitchCapMuxAdcReadOutConfig

    def __init__(
        self,
        *,
        config: OffsetSwitchCapMuxAdcReadOutConfig,
        policy: OffsetSwitchCapMuxAdcReadOutPolicy,
        name: str,
        inst_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
        data_num: int,
        digit_weights: tuple[float, ...],
    ) -> None:
        super().__init__(
            config=config,
            policy=policy,
            name=name,
            inst_shape=inst_shape,
            dtype=dtype,
            T__K=T__K,
            data_num=data_num,
            digit_weights=digit_weights,
        )
        if len(inst_shape) < 1:
            raise ValueError(f"inst_shape must be (*prefix, group_num), got {inst_shape}")
        if data_num <= 0:
            raise ValueError(f"require: data_num ({data_num}) > 0")
        if len(digit_weights) < 1:
            raise ValueError(f"require: len(digit_weights) ({len(digit_weights)}) >= 1")

        self.policy = policy
        self.T__K = T__K
        self.dtype = dtype
        self.data_num = data_num
        self.digit_weights = digit_weights

        prefix = name + "."
        self.data_switchcap = SwitchCap(
            config=config.data_switchcap_config,
            policy=policy.data_switchcap,
            name=f"{prefix}data_switchcap",
            inst_shape=(*inst_shape, data_num),
            dtype=dtype,
            T__K=T__K,
            cap_weights=digit_weights,
        )
        self.ref_switchcap = SwitchCap(
            config=config.ref_switchcap_config,
            policy=policy.ref_switchcap,
            name=f"{prefix}ref_switchcap",
            inst_shape=inst_shape,
            dtype=dtype,
            T__K=T__K,
            cap_weights=(1.0,),
        )
        self.analog_mux = AnalogMux(
            config=config.analog_mux_config,
            policy=policy.analog_mux,
            name=f"{prefix}analog_mux",
            inst_shape=(*inst_shape, 1),
            dtype=dtype,
            T__K=T__K,
        )
        self.bl_adc = ADC.from_config(
            config=config.adc_config,
            policy=policy.bl_adc,
            name=f"{prefix}bl_adc",
            inst_shape=(*inst_shape, 1),
            dtype=dtype,
            T__K=T__K,
        )

    @property
    def adc_mode_num(self) -> int:
        """Number of supported ADC operating points; valid ``adc_mode`` values are ``[0, mode_num)``."""
        return self.bl_adc.mode_num

    @property
    def adc_max_bits(self) -> int:
        """Maximum supported ``adc_bits`` value."""
        return self.bl_adc.max_bits

    def readout(
        self,
        v_data_grouped__V: Tensor,
        v_ref_grouped__V: Tensor,
        *,
        adc_operation_point: AdcOperationPoint,
    ) -> Tensor:
        """Run one VMM through data S/H → ref S/H → MUX → ADC.

        Args:
            v_data_grouped__V: Per-data per-digit voltages [V],
                shape ``(*runtime, group_num, data_num, digit_num)``.
            v_ref_grouped__V: Per-reference-column voltages [V],
                shape ``(*runtime, group_num)``.
            adc_operation_point: Runtime ADC operating point.

        Returns:
            Integer ADC code tensor.
        """
        v_pos_pre_mux__V = self.data_switchcap.sample_and_accumulate(v_data_grouped__V)
        data_num = v_pos_pre_mux__V.shape[-1]

        v_ref_bank__V = v_ref_grouped__V.unsqueeze(-1)
        v_ref_sampled__V = self.ref_switchcap.sample_and_accumulate(v_ref_bank__V)

        v_neg_pre_mux__V = v_ref_sampled__V.unsqueeze(-1).expand(*v_ref_sampled__V.shape, data_num)

        v_pos__V, v_neg__V = self.analog_mux.transport(v_pos_pre_mux__V, v_neg_pre_mux__V)

        code = self.bl_adc.convert(
            v_pos__V=v_pos__V,
            v_neg__V=v_neg__V,
            adc_operation_point=adc_operation_point,
        )

        # ReadOut orch overhead only — children with their own dynamic
        # profile model (SwitchCap / AnalogMux / ADC) emitted their own
        # events inside the chain above. Code shape =
        # (*serial, *inst_shape, data_num). Energy and latency are
        # independent: emit each only when its own config knob is > 0
        # so a latency-only or energy-only orch path records correctly.
        # Serial via the position-invariant numel rule; ReadOut's
        # parallel multiplier is ``inst_count * data_num`` (each data
        # column has its own switch-cap + mux + ADC chain in parallel).
        parallel_count = self.inst_count * self.data_num
        serial_op_count = max(1, code.numel() // max(parallel_count, 1))
        if self.config.energy_per_op__fJ > 0.0:
            dynamic_energy__fJ = torch.full_like(code, self.config.energy_per_op__fJ, dtype=torch.float32)
            self._log_dynamic_energy(dynamic_energy__fJ)
        if self.config.latency_per_op__ns > 0.0:
            latency__ns = torch.tensor(
                self.config.latency_per_op__ns * serial_op_count,
                device=code.device,
                dtype=torch.float32,
            )
            self._log_latency(latency__ns)

        return code
