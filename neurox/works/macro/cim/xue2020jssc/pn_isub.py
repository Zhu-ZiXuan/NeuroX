"""PN-ISUB polarity subtractor — one current subtractor + sign comparator per CIM-IO.

The SINWP-SC combined PWG / NWG lane currents subtract into a single-ended
magnitude `I_SUB = |I_P - I_N|` plus a sign decision (`N > P`), recovering the
polarity of the sign-magnitude weight encoding for the TMCSA magnitude
quantization. A reporter leaf: it self-bills the three ISUB internal replica legs
and the comparator decision constant.
"""

from __future__ import annotations

from torch import Tensor

from neurox.common import ConfigBase, ModuleBase, PolicyBase


class PnIsubConfig(ConfigBase):
    e_per_op__fJ: float
    """Data-independent comparator energy, billed once per sign decision — per (slot, IO) entry."""

    def validate(self) -> None:
        self._require_non_neg(self.e_per_op__fJ, "e_per_op__fJ")


class PnIsubPolicy(PolicyBase):
    pass


class PnIsub(ModuleBase[PnIsubConfig, PnIsubPolicy]):
    """PN-ISUB subtractor bank: polarity subtraction, sign decision, magnitude out.

    One instance per CIM-IO. Leading dims of the forward tensors are anonymous
    broadcast batch (the serial slot axis rides them).

    Args:
        inst_shape: Fabrication shape `(*inst_shape, gn)` — one subtractor per CIM-IO.
        vdd__V: Supply rail the three replica legs conduct across.
    """

    def __init__(
        self,
        *,
        config: PnIsubConfig,
        policy: PnIsubPolicy,
        inst_shape: tuple[int, ...],
        vdd__V: float,
    ) -> None:
        if not (vdd__V >= 0.0):
            raise ValueError(f"require: vdd__V ({vdd__V}) >= 0")
        super().__init__(config=config, policy=policy, inst_shape=inst_shape)
        self._vdd__V = vdd__V

    @property
    def _area_per_inst__um2(self) -> float:
        return 0.0

    @property
    def _leakage_per_inst__uW(self) -> float:
        return 0.0

    def forward(self, i_p__uA: Tensor, i_n__uA: Tensor, *, window__ns: float) -> tuple[Tensor, Tensor]:
        """Subtract the polarity lane currents into magnitude + sign.

        Args:
            i_p__uA: Combined PWG lane current.
                Shape: `[..., serial, gn]`.
            i_n__uA: Combined NWG lane current.
                Shape: `[..., serial, gn]`.
            window__ns: Conduction window of the three rail branches — the
                macro-injected live-bit tail.

        Returns:
            The single-ended magnitude `|I_P - I_N|` and the boolean sign, `True`
            where `I_N > I_P`.
            Shape: `[..., serial, gn]`.
        """
        i_sub_abs__uA = (i_p__uA - i_n__uA).abs()
        sign = i_n__uA > i_p__uA
        if self._is_dynamic_energy_profile_active():
            # The three rail branches over the injected window plus the comparator
            # decision constant, per (slot, IO) entry. The collector sums the slot
            # and CIM-IO axes past the caller's leading dims.
            # Shape: [..., serial, gn]
            e__fJ = self._vdd__V * window__ns * (i_p__uA + i_n__uA + i_sub_abs__uA) + self.config.e_per_op__fJ
            self._record_dynamic_energy(e__fJ)
        return i_sub_abs__uA, sign
