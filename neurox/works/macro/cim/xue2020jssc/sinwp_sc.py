"""SINWP-SC input- and weight-radix combine."""

from __future__ import annotations

from torch import Tensor

from neurox.common import ConfigBase, ModuleBase, PolicyBase

_POLARITY_NUM = 2  # P (PWG), N (NWG) lane per CIM-IO


class SinwpScConfig(ConfigBase):
    pass


class SinwpScPolicy(PolicyBase):
    pass


class SinwpSc(ModuleBase[SinwpScConfig, SinwpScPolicy]):
    """SINWP-SC switched-capacitor input-radix combine — one unit per (IO, polarity) lane.

    Args:
        inst_shape: Fabrication shape `(*inst_shape, gn, polarity)` — one combine
            unit per (CIM-IO, polarity) lane.
        bit_ratios: LSB-first per-input-bit combine ratios `s_k`, in the module's
            working dtype.
            Shape: `[x_bits]`.
        vdd__V: Supply rail every billed branch conducts across.
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
        vdd__V: float,
    ) -> None:
        if len(inst_shape) < 2 or inst_shape[-1] != _POLARITY_NUM:
            raise ValueError(
                f"require: inst_shape ({inst_shape}) trailing (gn, {_POLARITY_NUM}) — one unit per (IO, polarity) lane"
            )
        if bit_ratios.ndim != 1 or bit_ratios.numel() < 1:
            raise ValueError(f"require: bit_ratios is a non-empty 1-D tensor; got shape {tuple(bit_ratios.shape)}")
        if not bool((bit_ratios > 0.0).all()):
            raise ValueError(f"require: every bit_ratios entry > 0; got {bit_ratios.tolist()}")
        if not (vdd__V >= 0.0):
            raise ValueError(f"require: vdd__V ({vdd__V}) >= 0")

        super().__init__(config=config, policy=policy, inst_shape=inst_shape)
        self._vdd__V = vdd__V
        self._register_nonpersistent_buffer("_bit_ratios", bit_ratios.detach().clone())

    @property
    def _area_per_inst__um2(self) -> float:
        return 0.0

    @property
    def _leakage_per_inst__uW(self) -> float:
        return 0.0

    @property
    def input_bit_num(self) -> int:
        """Number of input bits K — the combine-ratio count."""
        return self._bit_ratios.numel()

    def forward(self, i__uA: Tensor, *, window_per_bit__ns: tuple[float, ...]) -> Tensor:
        """Combine the per-bit currents over the input-radix ratios.

        Args:
            i__uA: Per-input-bit, per-weight-digit lane currents. Both radix
                axes are LSB first; the input-bit axis is at dim -5.
                Every leading axis is anonymous broadcast batch.
                Shape: `[..., x_bits, serial, gn, polarity, w_digit]`.
            window_per_bit__ns: Per-input-bit conduction window — the
                sample-and-hold suffix-sum window the macro injects per call.

        Returns:
            Combined lane current.
            Shape: `[..., serial, gn, polarity]`.
        """
        n_bits = self.input_bit_num
        if i__uA.ndim < 5:
            raise ValueError(
                f"forward() expects i__uA with >= 5 dims "
                f"[..., x_bits, serial, gn, {_POLARITY_NUM}, w_digit]; got {i__uA.ndim}"
            )
        if tuple(i__uA.shape[-3:-1]) != self.inst_shape[-2:]:
            raise ValueError(
                f"forward() expects i__uA trailing (gn, {_POLARITY_NUM}) {self.inst_shape[-2:]}; "
                f"got {tuple(i__uA.shape[-3:-1])}"
            )
        if i__uA.shape[-5] != n_bits:
            raise ValueError(f"forward() expects x_bits ({n_bits}) at dim -5; got {i__uA.shape[-5]}")
        if len(window_per_bit__ns) != n_bits:
            raise ValueError(f"forward() expects {n_bits} window_per_bit__ns entries; got {len(window_per_bit__ns)}")

        # Branch-tensor law: materialize the per-leg sink currents FIRST —
        # each mirror leg carries the s_k-scaled copy, not the interface
        # current — then value AND energy consume the same tensor.
        # Shape: [x_bits, serial=1, gn=1, polarity=1, w_digit=1]
        bit_ratios = self._bit_ratios.view(n_bits, 1, 1, 1, 1)
        # Shape: [..., x_bits, serial, gn, polarity, w_digit]
        i_leg__uA = i__uA * bit_ratios

        if self._is_dynamic_energy_profile_active():
            # Rail conduction: signed per-bit LEG current (not |I|) over the
            # suffix-sum hold window.
            # Fold the internal bit axis with its Python-scalar windows;
            # torch.compile unrolls this fixed-length loop.
            # Shape: [..., x_bits, serial, gn, polarity, w_digit]
            #     -> [..., serial, gn, polarity, w_digit]
            e_conduction = i_leg__uA.select(-5, 0) * window_per_bit__ns[0]
            for bit, bit_window__ns in enumerate(window_per_bit__ns[1:], start=1):
                e_conduction = e_conduction + i_leg__uA.select(-5, bit) * bit_window__ns
            e_conduction = self._vdd__V * e_conduction
            self._record_dynamic_energy(e_conduction)

        # Sum input-bit and weight-digit place values into one lane current.
        # Shape: [..., x_bits, serial, gn, polarity, w_digit]
        #     -> [..., serial, gn, polarity]
        return i_leg__uA.sum(dim=(-5, -1))
