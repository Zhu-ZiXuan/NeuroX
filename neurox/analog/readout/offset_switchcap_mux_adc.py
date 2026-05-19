"""Offset-coded readout chain: switch-cap banks, mux, and ADC.

See also:
    docs/dev/modules/analog/readout/offset_switchcap_mux_adc.md
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from neurox.analog.adc import ADC, ADCConfig
from neurox.analog.analog_mux import AnalogMux, AnalogMuxConfig
from neurox.analog.switch_cap import SwitchCap, SwitchCapConfig

from .base import ReadOut, ReadOutConfig


@dataclass(frozen=True, kw_only=True)
class OffsetSwitchCapMuxAdcReadOutConfig(ReadOutConfig):
    """Config for :class:`OffsetSwitchCapMuxAdcReadOut`.

    Attributes:
        data_switchcap_cfg: Per-digit data-leg switch-cap bank config.
        ref_switchcap_cfg: Per-group ref-leg switch-cap bank config.
        analog_mux_cfg: Analog mux config.
        adc_cfg: Inner ADC config.
    """

    data_switchcap_cfg: SwitchCapConfig
    ref_switchcap_cfg: SwitchCapConfig
    analog_mux_cfg: AnalogMuxConfig
    adc_cfg: ADCConfig


@ReadOut.register_config(OffsetSwitchCapMuxAdcReadOutConfig)
class OffsetSwitchCapMuxAdcReadOut(ReadOut):
    """Offset-coded readout: data and ref switch-cap banks, mux, and ADC."""

    def __init__(
        self,
        *,
        cfg: OffsetSwitchCapMuxAdcReadOutConfig,
        name: str,
        T__K: float,
        dtype: torch.dtype,
        stochastic: bool | None,
        data_num: int,
        digit_weights: tuple[float, ...],
    ) -> None:
        super().__init__(
            cfg=cfg,
            name=name,
            T__K=T__K,
            dtype=dtype,
            stochastic=stochastic,
            data_num=data_num,
            digit_weights=digit_weights,
        )
        if data_num <= 0:
            raise ValueError(f"require: data_num ({data_num}) > 0")
        if len(digit_weights) < 1:
            raise ValueError(f"require: len(digit_weights) ({len(digit_weights)}) >= 1")

        self.cfg = cfg
        self.T__K = T__K
        self.dtype = dtype
        self.stochastic: bool | None = stochastic
        self.data_num: int = data_num
        self.digit_weights: tuple[float, ...] = digit_weights

        prefix = name + "."
        self.data_switchcap = SwitchCap(
            cfg=cfg.data_switchcap_cfg,
            name=f"{prefix}data_switchcap",
            T__K=T__K,
            dtype=dtype,
            cap_weights=digit_weights,
        )
        self.ref_switchcap = SwitchCap(
            cfg=cfg.ref_switchcap_cfg,
            name=f"{prefix}ref_switchcap",
            T__K=T__K,
            dtype=dtype,
            cap_weights=(1.0,),
        )
        self.analog_mux = AnalogMux(
            cfg=cfg.analog_mux_cfg,
            name=f"{prefix}analog_mux",
            T__K=T__K,
            dtype=dtype,
        )
        self.bl_adc: ADC = ADC.from_config(
            cfg=cfg.adc_cfg,
            name=f"{prefix}bl_adc",
            T__K=T__K,
            dtype=dtype,
            stochastic=stochastic,
        )

    @property
    def area_per_inst__um2(self) -> float:
        """Silicon area of this readout's own orchestration logic [um^2]."""
        return self.cfg.area_per_inst__um2

    @property
    def leakage_per_inst__uW(self) -> float:
        """Static leakage of this readout's own orchestration logic [uW]."""
        return self.cfg.leakage_per_inst__uW

    def latency_per_op__ns(self, *, adc_bits: int) -> float:
        """Per-VMM pipeline latency [ns]: ``orch + data_sc + ref_sc + mux + bl_adc(adc_bits)``.

        Args:
            adc_bits: Active ADC bit width.
        """
        return (
            self.cfg.latency_per_op__ns
            + self.data_switchcap.latency_per_op__ns
            + self.ref_switchcap.latency_per_op__ns
            + self.analog_mux.latency_per_op__ns
            + self.bl_adc.latency_per_op__ns(bits=adc_bits)
        )

    def fabricate(self, shape: tuple[int, ...]) -> None:
        """Sample static per-instance state over ``shape`` (re-callable).

        Args:
            shape: Per-instance fabrication shape
                ``(*prefix, group_num)``.
        """
        if len(shape) < 1:
            raise ValueError(f"shape must be (*prefix, group_num), got {shape}")

        self.data_switchcap.fabricate((*shape, self.data_num))
        self.ref_switchcap.fabricate(shape)

        self.analog_mux.fabricate((*shape, 1))
        self.bl_adc.fabricate((*shape, 1))

        self._record_inst_count(shape)

    def readout(
        self,
        v_data_grouped__V: Tensor,
        v_ref_grouped__V: Tensor,
        *,
        adc_mode: int,
        adc_bits: int,
    ) -> Tensor:
        """Run one VMM through data S/H → ref S/H → MUX → ADC.

        Args:
            v_data_grouped__V: Per-data per-digit voltages [V],
                shape ``(*runtime, group_num, data_num, digit_num)``.
            v_ref_grouped__V: Per-reference-column voltages [V],
                shape ``(*runtime, group_num)``.
            adc_mode: ADC operating-point index.
            adc_bits: ADC bit width.

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
            mode=adc_mode,
            bits=adc_bits,
        )

        if self.cfg.energy_per_op__fJ > 0.0:
            self._log_dynamic(self.cfg.energy_per_op__fJ, self.cfg.latency_per_op__ns)

        return code
