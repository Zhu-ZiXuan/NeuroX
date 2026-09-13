"""Triple-margin current-mode successive-approximation ADC."""

from __future__ import annotations

import torch
from torch import Tensor

from neurox.primitive.analog.current_adc import (
    SarIadc,
    SarIadcConfig,
    SarIadcPolicy,
)
from neurox.primitive.physics import e_charge__fJ, q_conduction__fC


class TmcsaConfig(SarIadcConfig):
    t_ph2__ns: float
    """PH2 conduction duration of one decision step."""

    t_ph3__ns: float
    """PH3 conduction duration of one decision step."""

    energy_per_bit__fJ: float
    """Data-independent switching energy of one output bit per instance."""

    def validate(self) -> None:
        super().validate()
        self._require_non_neg(self.t_ph2__ns, "t_ph2__ns")
        self._require_non_neg(self.t_ph3__ns, "t_ph3__ns")
        self._require_le(self.t_ph2__ns + self.t_ph3__ns, "t_ph2__ns + t_ph3__ns", self.latency_per_bit__ns)
        self._require_non_neg(self.energy_per_bit__fJ, "energy_per_bit__fJ")


class TmcsaPolicy(SarIadcPolicy):
    pass


_Config = TmcsaConfig
_Policy = TmcsaPolicy


class Tmcsa(SarIadc):
    """TMCSA with PH2/PH3 energy accounting."""

    config: _Config
    policy: _Policy

    def __init__(
        self,
        *,
        config: _Config,
        policy: _Policy,
        inst_shape: tuple[int, ...],
        vdd__V: float,
        dtype: torch.dtype,
    ) -> None:
        if not (vdd__V >= 0.0):
            raise ValueError(f"require: vdd__V ({vdd__V}) >= 0")
        super().__init__(
            config=config,
            policy=policy,
            inst_shape=inst_shape,
            dtype=dtype,
        )
        self._vdd__V = vdd__V

    def _compute_bit_dynamic_energy__fJ(
        self,
        i_in__uA: Tensor,
        i_refs__uA: Tensor,
        trial_code: Tensor,
        *,
        bit_position: int,
    ) -> Tensor:
        del bit_position
        config = self.config
        i_ref__uA = self._select_reference(i_refs__uA, trial_code)
        i_common__uA = i_in__uA + i_ref__uA
        q_ph2__fC = q_conduction__fC(3.0 * i_common__uA, config.t_ph2__ns)
        q_ph3__fC = q_conduction__fC(2.0 * i_common__uA, config.t_ph3__ns)
        return e_charge__fJ(self._vdd__V, q_ph2__fC + q_ph3__fC) + config.energy_per_bit__fJ
