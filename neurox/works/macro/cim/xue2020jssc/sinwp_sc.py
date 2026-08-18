"""SINWP-SC input-radix combine — one switched-capacitor unit per (CIM-IO, polarity) lane.

The K per-input-bit DSWCT output currents are weighted by the LSB-first
input-radix combine ratios `s_k` and summed over the bit axis into the per-lane
pre-subtraction current `I_DL_PN`. A reporter leaf: it self-bills its held/live
mirror-leg conduction and hold-cap cycling.
"""

from __future__ import annotations

import torch
from torch import Tensor

from neurox.common import ConfigBase, ModuleBase, PolicyBase

_POLARITY_NUM = 2  # P (PWG), N (NWG) lane per CIM-IO


class SinwpScConfig(ConfigBase):
    """Physical knobs and static PPA seat of one SINWP-SC combine unit."""

    c_hold__fF: float
    """Sample-and-hold capacitance per combine leg, cycled once per instance per (slot, bit) event."""
    area_per_inst__um2: float
    leakage_per_inst__uW: float

    def validate(self) -> None:
        self._require_non_neg(self.c_hold__fF, "c_hold__fF")
        self._require_non_neg(self.area_per_inst__um2, "area_per_inst__um2")
        self._require_non_neg(self.leakage_per_inst__uW, "leakage_per_inst__uW")


class SinwpScPolicy(PolicyBase):
    """Source-free nonideality policy for the deterministic SINWP-SC combine."""


class SinwpSc(ModuleBase[SinwpScConfig, SinwpScPolicy]):
    """SINWP-SC switched-capacitor input-radix combine — one unit per (IO, polarity) lane.

    Args:
        inst_shape: Fabrication shape `(*inst_shape, gn, polarity)` — one combine
            unit per (CIM-IO, polarity) lane.
        bit_ratios: LSB-first per-input-bit combine ratios `s_k`, in the module's
            working dtype.
            Shape: `[x_bits]`.
        v_dd__V: Supply rail every billed branch conducts across.
    """

    # === Functional buffers ===

    _bit_ratios: Tensor  # Shape: [x_bits]

    def __init__(
        self,
        *,
        config: SinwpScConfig,
        policy: SinwpScPolicy,
        inst_shape: tuple[int, ...],
        bit_ratios: Tensor,
        v_dd__V: float,
    ) -> None:
        if len(inst_shape) < 2 or inst_shape[-1] != _POLARITY_NUM:
            raise ValueError(
                f"require: inst_shape ({inst_shape}) trailing (gn, {_POLARITY_NUM}) — one unit per (IO, polarity) lane"
            )
        if bit_ratios.ndim != 1 or bit_ratios.numel() < 1:
            raise ValueError(f"require: bit_ratios is a non-empty 1-D tensor; got shape {tuple(bit_ratios.shape)}")
        if not bool((bit_ratios > 0.0).all()):
            raise ValueError(f"require: every bit_ratios entry > 0; got {bit_ratios.tolist()}")
        if not (v_dd__V >= 0.0):
            raise ValueError(f"require: v_dd__V ({v_dd__V}) >= 0")

        super().__init__(config=config, policy=policy, inst_shape=inst_shape)
        self._v_dd__V = v_dd__V
        self.register_buffer("_bit_ratios", bit_ratios.detach().clone(), persistent=False)

    @property
    def _area_per_inst__um2(self) -> float:
        return self.config.area_per_inst__um2

    @property
    def _leakage_per_inst__uW(self) -> float:
        return self.config.leakage_per_inst__uW

    @property
    def input_bit_num(self) -> int:
        """Number of input bits K — the combine-ratio count."""
        return self._bit_ratios.numel()

    def forward(self, i__uA: Tensor, *, window_per_bit__ns: Tensor) -> Tensor:
        """Combine the per-bit currents over the input-radix ratios.

        Args:
            i__uA: Per-input-bit lane currents, bit axis at dim -4 and LSB first.
                Every leading axis is anonymous broadcast batch.
                Shape: `[..., x_bits, serial, gn, polarity]`.
            window_per_bit__ns: Per-input-bit conduction window — the
                sample-and-hold suffix-sum window the macro injects per call.
                Shape: `[x_bits]`.

        Returns:
            Combined lane current.
            Shape: `[..., serial, gn, polarity]`.
        """
        n_bits = self.input_bit_num
        if i__uA.ndim < 4:
            raise ValueError(
                f"forward() expects i__uA with >= 4 dims [..., x_bits, serial, gn, {_POLARITY_NUM}]; got {i__uA.ndim}"
            )
        if tuple(i__uA.shape[-2:]) != self.inst_shape[-2:]:
            raise ValueError(
                f"forward() expects i__uA trailing (gn, {_POLARITY_NUM}) {self.inst_shape[-2:]}; "
                f"got {tuple(i__uA.shape[-2:])}"
            )
        if i__uA.shape[-4] != n_bits:
            raise ValueError(f"forward() expects x_bits ({n_bits}) at dim -4; got {i__uA.shape[-4]}")
        if tuple(window_per_bit__ns.shape) != (n_bits,):
            raise ValueError(
                f"forward() expects window_per_bit__ns.shape ({(n_bits,)}); got {tuple(window_per_bit__ns.shape)}"
            )

        # Branch-tensor law: materialize the per-leg sink currents FIRST —
        # each mirror leg carries the s_k-scaled copy, not the interface
        # current — then value AND energy consume the same tensor.
        # Shape: [x_bits, serial=1, gn=1, polarity=1]
        bit_ratios = self._bit_ratios.view(n_bits, 1, 1, 1)
        # Shape: [..., x_bits, serial, gn, polarity]
        i_leg__uA = i__uA * bit_ratios

        if self._is_dynamic_energy_profile_active():
            # Neither branch reduces its own axes: the collector sums the bit,
            # slot and (gn, polarity) lane axes past the caller's leading dims.
            # Rail conduction: signed per-bit LEG current (not |I|) over the
            # suffix-sum hold window.
            # Shape: [x_bits] -> [x_bits, serial=1, gn=1, polarity=1]
            window_view__ns = window_per_bit__ns.view(n_bits, 1, 1, 1)
            # Shape: [..., x_bits, serial, gn, polarity]
            e_conduction = (self._v_dd__V * window_view__ns) * i_leg__uA
            self._record_dynamic_energy(e_conduction)
            # Hold-cap cycling: one c_hold * v_dd**2 event per (slot x bit) per
            # (IO, polarity) lane — a constant, so the expanded view holds no
            # storage and only the caller's leading dims are materialized.
            # Shape: [] -> [..., x_bits, serial, gn, polarity]
            e_hold__fJ = torch.full(
                (), self.config.c_hold__fF * self._v_dd__V**2, dtype=torch.float32, device=i__uA.device
            )
            self._record_dynamic_energy(e_hold__fJ.expand(i__uA.shape))

        # Temporal weighted sum over bits (input radix), LSB first — the sum
        # of the SAME materialized legs the billing consumed.
        # Shape: [..., x_bits, serial, gn, polarity] -> [..., serial, gn, polarity]
        return i_leg__uA.sum(dim=-4)
