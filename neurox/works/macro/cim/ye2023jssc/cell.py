"""WH-2T1R lookup cell.

See Also:
    docs/reference/primitive/xbar/cell/1t1r_linear.md
"""

from __future__ import annotations

import torch
from torch import Tensor

from neurox.primitive.xbar.cell import (
    XbarCell1t1rDcop,
    XbarCell1t1rLinear,
    XbarCell1t1rLinearConfig,
    XbarCell1t1rLinearPolicy,
)


class Ye2023Jssc2t1rCellConfig(XbarCell1t1rLinearConfig):
    v_x_on_threshold__V: float
    """Internal-node threshold separating the HRS and LRS T2 lookup values."""

    i_t2_leak__uA: float
    """Unit-width T2 leakage while its TBL is selected."""
    i_t2_unit_signal__uA: float
    """Unit-width T2 signal current in LRS, excluding leakage."""

    def validate(self) -> None:
        super().validate()

        self._require_len(self.g_cell_off_table__uS, "g_cell_off_table__uS", 2)
        self._require_non_neg(self.v_x_on_threshold__V, "v_x_on_threshold__V")
        self._require_non_neg(self.i_t2_leak__uA, "i_t2_leak__uA")
        self._require_non_neg(self.i_t2_unit_signal__uA, "i_t2_unit_signal__uA")


class Ye2023Jssc2t1rCellPolicy(XbarCell1t1rLinearPolicy):
    pass


_Config = Ye2023Jssc2t1rCellConfig
_Policy = Ye2023Jssc2t1rCellPolicy
_Dcop = XbarCell1t1rDcop


class Ye2023Jssc2t1rCell(XbarCell1t1rLinear):
    """WH-2T1R cell."""

    config: _Config
    policy: _Policy

    def __init__(
        self,
        *,
        config: _Config,
        policy: _Policy,
        inst_shape: tuple[int, ...],
        dtype: torch.dtype,
    ) -> None:
        super().__init__(config=config, policy=policy, inst_shape=inst_shape, dtype=dtype)

    @property
    def i_t2_leak__uA(self) -> float:
        return self.config.i_t2_leak__uA

    @torch.no_grad()
    def i_t2_unit__uA(self, dcop: _Dcop) -> Tensor:
        """Return the unit-width T2 current at the solved internal voltage."""
        config = self.config
        driven = dcop.v_x__V > config.v_x_on_threshold__V
        return driven * config.i_t2_unit_signal__uA + self.i_t2_leak__uA
