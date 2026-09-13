"""RS-CSA single-reference selection composed with SAR conversion."""

from __future__ import annotations

import torch

from neurox.works.macro.cim.ye2023jssc.rscsa import RsCsaIadc, RsCsaIadcPolicy

from ._utils import T2_SIGNAL__uA, adc_config


def test_single_reference_drives_the_sar_search() -> None:
    adc = RsCsaIadc(
        config=adc_config(),
        policy=RsCsaIadcPolicy(comparator_offset=False),
        inst_shape=(),
        dtype=torch.float64,
    )
    adc.fabricate()
    i_in = torch.tensor([0.0, T2_SIGNAL__uA, 15 * T2_SIGNAL__uA])
    i_ref = torch.tensor([T2_SIGNAL__uA])

    code = adc.convert(i_in, i_ref, active_bits=4)

    assert torch.equal(code, torch.tensor([0, 1, 15]))
