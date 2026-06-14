"""Regression: readout orch overhead emits energy and latency events independently.

After R10 split into ``_log_dynamic_energy`` / ``_log_latency``, the
readout's per-op orch overhead is gated **per-quantity**: an
``energy_per_op__fJ = 0, latency_per_op__ns > 0`` config still emits a
latency event for the readout itself; an ``energy > 0, latency = 0``
emits only the energy event. The previous single-``if`` gate dropped
the latency-only case silently.
"""

from __future__ import annotations

import pytest
import torch

from neurox.analog.adc import AdcOperationPoint
from neurox.analog.adc.general import GeneralADCConfig, GeneralADCPolicy
from neurox.analog.analog_mux import AnalogMuxConfig, AnalogMuxPolicy
from neurox.analog.switch_cap import SwitchCapConfig, SwitchCapPolicy
from neurox.common.profiler import NeuroxProfiler
from neurox.xbar.readout.offset_switchcap_mux_adc import (
    OffsetSwitchCapMuxAdcReadOut,
    OffsetSwitchCapMuxAdcReadOutConfig,
    OffsetSwitchCapMuxAdcReadOutPolicy,
)


def _zero_switchcap_config() -> SwitchCapConfig:
    return SwitchCapConfig(
        c_unit__fF=1.0,
        cap_mismatch_sigma_relative=0.0,
        energy_per_sample_overhead__fJ=1.0,
        latency_per_op__ns=0.5,
        leakage_per_inst__uW=0.0,
        area_per_inst__um2=0.0,
    )


def _zero_mux_config() -> AnalogMuxConfig:
    return AnalogMuxConfig(
        mux_gain=1.0,
        mux_noise_cm_sigma__V=0.0,
        mux_noise_dm_sigma__V=0.0,
        energy_per_access__fJ=1.0,
        latency_per_op__ns=0.5,
        leakage_per_inst__uW=0.0,
        area_per_inst__um2=0.0,
    )


def _zero_adc_config() -> GeneralADCConfig:
    return GeneralADCConfig(
        boundaries=(-1.0, 0.0, 1.0),
        sampling_noise__V=0.0,
        comparator_noise__V=0.0,
        drive_thermal__V=0.0,
        drive_value=0.0,
        input_transform="linear",
        energy_per_op__fJ=1.0,
        latency_per_op__ns=0.5,
        leakage_per_inst__uW=0.0,
        area_per_inst__um2=0.0,
    )


def _build_readout(*, energy: float, latency: float) -> OffsetSwitchCapMuxAdcReadOut:
    cfg = OffsetSwitchCapMuxAdcReadOutConfig(
        data_switchcap_config=_zero_switchcap_config(),
        ref_switchcap_config=_zero_switchcap_config(),
        analog_mux_config=_zero_mux_config(),
        adc_config=_zero_adc_config(),
        energy_per_op__fJ=energy,
        latency_per_op__ns=latency,
        leakage_per_inst__uW=0.0,
        area_per_inst__um2=0.0,
    )
    policy = OffsetSwitchCapMuxAdcReadOutPolicy(
        data_switchcap=SwitchCapPolicy(cap_mismatch=False, sampling_thermal_noise=False),
        ref_switchcap=SwitchCapPolicy(cap_mismatch=False, sampling_thermal_noise=False),
        analog_mux=AnalogMuxPolicy(mux_noise_cm=False, mux_noise_dm=False),
        bl_adc=GeneralADCPolicy(sampling_noise=False, comparator_noise=False, drive_thermal=False),
    )
    readout = OffsetSwitchCapMuxAdcReadOut(
        config=cfg,
        policy=policy,
        name="ro",
        inst_shape=(1,),
        dtype=torch.float32,
        T__K=300.0,
        data_num=2,
        digit_weights=(1.0,),
    )
    readout.eval()
    return readout


def _drive_one_vmm(readout: OffsetSwitchCapMuxAdcReadOut) -> None:
    # data shape: (*, group_num=1, data_num=2, digit_num=1)
    v_data = torch.zeros(1, 2, 1, dtype=torch.float32)
    v_ref = torch.zeros(1, dtype=torch.float32)
    readout.readout(
        v_data, v_ref, adc_operation_point=AdcOperationPoint(adc_mode=0, adc_bits=2)
    )


def _self_events(events: list, name: str) -> list:
    return [e for e in events if e.qualified_name == name]


def test_readout_emits_latency_when_energy_is_zero() -> None:
    """Regression: pre-R10 the readout had one shared ``if energy > 0`` gate;
    a latency-only orch (energy=0, latency>0) was silently dropped."""
    readout = _build_readout(energy=0.0, latency=2.0)
    with NeuroxProfiler() as p:
        _drive_one_vmm(readout)
    assert len(_self_events(p.energy_events, "ro")) == 0
    assert len(_self_events(p.latency_events, "ro")) == 1


def test_readout_emits_energy_when_latency_is_zero() -> None:
    readout = _build_readout(energy=3.0, latency=0.0)
    with NeuroxProfiler() as p:
        _drive_one_vmm(readout)
    assert len(_self_events(p.energy_events, "ro")) == 1
    assert len(_self_events(p.latency_events, "ro")) == 0


def test_readout_skips_self_emission_when_both_zero() -> None:
    readout = _build_readout(energy=0.0, latency=0.0)
    with NeuroxProfiler() as p:
        _drive_one_vmm(readout)
    assert len(_self_events(p.energy_events, "ro")) == 0
    assert len(_self_events(p.latency_events, "ro")) == 0


def test_readout_emits_both_when_both_nonzero() -> None:
    readout = _build_readout(energy=5.0, latency=4.0)
    with NeuroxProfiler() as p:
        _drive_one_vmm(readout)
    assert len(_self_events(p.energy_events, "ro")) == 1
    assert len(_self_events(p.latency_events, "ro")) == 1
