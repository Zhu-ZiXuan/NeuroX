"""RS-CSA transfer and energy ownership."""

import dataclasses

import pytest
import torch

from neurox import Profiler, Reporter, stamp_names
from neurox.works.macro.cim.ye2023jssc.rscsa import RsCsaIadc, RsCsaIadcPolicy

from ._utils import ADC_BITS, T2_SIGNAL__uA, adc_config


def test_rscsa_quantizes_signal_and_records_owned_energy() -> None:
    adc = RsCsaIadc(
        config=adc_config(),
        policy=RsCsaIadcPolicy(comparator_offset=False),
        inst_shape=(),
        dtype=torch.float64,
        T__K=300.0,
    )
    adc.fabricate()
    stamp_names(adc)
    reporter = Reporter(adc)
    i_in = torch.tensor([0.0, T2_SIGNAL__uA, 15 * T2_SIGNAL__uA])
    i_ref = torch.tensor([T2_SIGNAL__uA])

    with Profiler(leading_rank=1) as profiler:
        code = adc.convert(i_in, i_ref, active_bits=4)

    assert torch.equal(code, torch.tensor([0, 1, 15]))
    assert adc.latency__ns(active_bits=4) == 8.0
    e_switching__fJ = i_in.numel() * ADC_BITS * adc.config.energy_per_bit__fJ
    assert reporter.by_name(profiler) == {"": pytest.approx(e_switching__fJ)}


def test_rscsa_initialization_changes_latency_but_not_dynamic_energy() -> None:
    init__ns = 3.0
    adc = RsCsaIadc(
        config=dataclasses.replace(adc_config(), init__ns=init__ns),
        policy=RsCsaIadcPolicy(comparator_offset=False),
        inst_shape=(),
        dtype=torch.float64,
        T__K=300.0,
    )
    adc.fabricate()
    stamp_names(adc)
    reporter = Reporter(adc)
    i_in = torch.tensor([0.0, T2_SIGNAL__uA, 15 * T2_SIGNAL__uA])
    i_ref = torch.tensor([T2_SIGNAL__uA])

    with Profiler(leading_rank=1) as profiler:
        adc.convert(i_in, i_ref, active_bits=ADC_BITS)

    assert adc.latency__ns(active_bits=ADC_BITS) == 11.0
    switching__fJ = i_in.numel() * ADC_BITS * adc.config.energy_per_bit__fJ
    assert reporter.by_name(profiler) == {"": pytest.approx(switching__fJ)}
