"""WH-2T1R cell behavior."""

from dataclasses import replace

import pytest
import torch

from neurox.primitive.xbar.cell import XbarCell1t1rDcop
from neurox.works.macro.cim.ye2023jssc.cell import Ye2023Jssc2t1rCell, Ye2023Jssc2t1rCellPolicy

from ._utils import T2_LEAK__uA, T2_SIGNAL__uA, cell_config


def test_cell_requires_binary_weight_states() -> None:
    config = cell_config()
    with pytest.raises(ValueError, match=r"len\(g_cell_off_table__uS\).*== 2"):
        replace(
            config,
            g_cell_off_table__uS=(0.001, 0.001, 0.001),
            g_cell_on_table__uS=(1.0, 5.0, 10.0),
            vx_ratio_off_table=(0.0, 0.0, 0.0),
            vx_ratio_on_table=(0.5, 0.5, 0.5),
        ).validate()


def test_t2_current_uses_solved_internal_voltage() -> None:
    cell = Ye2023Jssc2t1rCell(
        config=cell_config(),
        policy=Ye2023Jssc2t1rCellPolicy(),
        inst_shape=(2,),
        dtype=torch.float64,
        T__K=300.0,
    )
    dcop = XbarCell1t1rDcop(
        i__uA=torch.zeros(2),
        di_dvbl__uS=torch.zeros(2),
        di_dvsl__uS=torch.zeros(2),
        v_x__V=torch.tensor([0.1, 0.2]),
    )
    assert torch.equal(cell.i_t2_unit__uA(dcop), torch.tensor([T2_LEAK__uA, T2_LEAK__uA + T2_SIGNAL__uA]))
    assert cell.i_t2_leak__uA == T2_LEAK__uA
