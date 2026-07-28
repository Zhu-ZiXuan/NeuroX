"""SINWP-SC input-radix combine — one switched-capacitor unit per (CIM-IO, P/N) lane.

The K per-input-bit DSWCT output currents are weighted by the LSB-first
input-radix combine ratios ``s_k`` and summed over the bit axis into the
per-lane pre-subtraction current ``I_DL_PN``. A reporter leaf: it self-bills
its held/live mirror-leg conduction and hold-cap cycling, and emits no latency
event (the macro is the sole latency emitter).

See also:
    docs/works/macro/cim/xue2020jssc/model.md
"""

from __future__ import annotations

import math

from torch import Tensor

from neurox.common import ConfigBase, ModuleBase, PolicyBase

_POLARITY_NUM = 2  # P (PWG), N (NWG) lane per CIM-IO


class SinwpScConfig(ConfigBase):
    """Immutable configuration for :class:`SinwpSc`.

    Attributes:
        c_hold__fF: Sample-and-hold capacitance per combine leg; cycled once
            per (slot x bit) event per instance at the supply rail.
        area_per_inst__um2: Silicon area per fabricated instance.
        leakage_per_inst__uW: Static leakage per instance.
    """

    c_hold__fF: float
    area_per_inst__um2: float
    leakage_per_inst__uW: float

    def validate(self) -> None:
        self._require_non_neg(self.c_hold__fF, "c_hold__fF")
        self._require_non_neg(self.area_per_inst__um2, "area_per_inst__um2")
        self._require_non_neg(self.leakage_per_inst__uW, "leakage_per_inst__uW")


class SinwpScPolicy(PolicyBase):
    """Source-free nonideality policy for the deterministic SINWP-SC combine."""


class SinwpSc(ModuleBase[SinwpScConfig, SinwpScPolicy]):
    """SINWP-SC switched-capacitor input-radix combine — one unit per (IO, P/N) lane.

    Args:
        config: SINWP-SC configuration (cap knob + static PPA seat).
        policy: Source-free policy.
        inst_shape: Per-instance fabrication shape ``(*inst, gn, 2)`` — one
            combine unit per (CIM-IO, polarity) lane.
        bit_ratios: LSB-first per-input-bit combine ratios ``s_k``, 1-D tensor
            of length ``x_bits``, in the module's working dtype.
        v_dd__V: Supply-rail voltage [V] every billed branch conducts across.
    """

    # --- Immutable model buffers ---

    _bit_ratios: Tensor

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
                f"require: inst_shape ({inst_shape}) trailing (gn, {_POLARITY_NUM}) — one unit per (IO, P/N) lane"
            )
        if bit_ratios.ndim != 1 or bit_ratios.numel() < 1:
            raise ValueError(f"require: bit_ratios is a non-empty 1-D tensor; got shape {tuple(bit_ratios.shape)}")
        if not bool((bit_ratios > 0.0).all()):
            raise ValueError(f"require: every bit_ratios entry > 0; got {bit_ratios.tolist()}")
        if not (v_dd__V >= 0.0):
            raise ValueError(f"require: v_dd__V ({v_dd__V}) >= 0")

        super().__init__(config=config, policy=policy, inst_shape=inst_shape)
        self._area_per_inst__um2 = config.area_per_inst__um2
        self._leakage_per_inst__uW = config.leakage_per_inst__uW
        self._v_dd__V = v_dd__V
        self.register_buffer("_bit_ratios", bit_ratios.detach().clone(), persistent=False)

    def _sample_fabricate_mismatch(self) -> None:
        pass

    @property
    def input_bit_num(self) -> int:
        """Number of input bits K — the combine-ratio count."""
        return self._bit_ratios.numel()

    def forward(self, i__uA: Tensor, *, window_per_bit__ns: Tensor) -> Tensor:
        """Combine the per-bit currents over the input-radix ratios.

        Args:
            i__uA: Per-input-bit lane currents, shape
                ``[..., x_bits, serial, gn, 2]`` (bit axis at dim -4, LSB
                first). Every leading axis is anonymous broadcast batch.
            window_per_bit__ns: Per-input-bit conduction window, shape
                ``[x_bits]`` — the sample-and-hold suffix-sum window the macro
                injects per call.

        Returns:
            Combined lane current [uA], shape ``[..., serial, gn, 2]``.
        """
        n_bits = self.input_bit_num
        if i__uA.ndim < 4:
            raise ValueError(f"forward() expects i__uA with >= 4 dims [..., x_bits, serial, gn, 2]; got {i__uA.ndim}")
        if tuple(i__uA.shape[-2:]) != self.inst_shape[-2:]:
            raise ValueError(
                f"forward() expects i__uA trailing (gn, 2) {self.inst_shape[-2:]}; got {tuple(i__uA.shape[-2:])}"
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
        # Shape: [x_bits, serial=1, gn=1, P/N=1]
        bit_ratios = self._bit_ratios.view(n_bits, 1, 1, 1)
        # Shape: [..., x_bits, serial, gn, 2]
        i_leg__uA = i__uA * bit_ratios

        if self._is_dynamic_energy_profile_active():
            # Rail conduction: signed per-bit LEG-current sum (not |I|) over
            # the suffix-sum hold window.
            # Shape: [..., x_bits, serial, gn, 2] -> [..., x_bits]
            i_leg_per_bit = i_leg__uA.sum(dim=(-3, -2, -1))
            # Shape: [..., x_bits] -> [...]
            e_conduction = self._v_dd__V * (i_leg_per_bit * window_per_bit__ns).sum(dim=-1)
            self._record_dynamic_energy(e_conduction)
            # Hold-cap cycling: one c_hold * v_dd**2 event per (slot x bit)
            # per (IO, P/N) lane.
            event_count = math.prod(i__uA.shape[-4:-2]) * math.prod(self.inst_shape[-2:])
            e_cap = i__uA.new_full(
                i__uA.shape[:-4],
                self.config.c_hold__fF * self._v_dd__V**2 * event_count,
            )
            self._record_dynamic_energy(e_cap)

        # Temporal weighted sum over bits (input radix), LSB first — the sum
        # of the SAME materialized legs the billing consumed.
        # Shape: [..., x_bits, serial, gn, 2] -> [..., serial, gn, 2]
        return i_leg__uA.sum(dim=-4)
