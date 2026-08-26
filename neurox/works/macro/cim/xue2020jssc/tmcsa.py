"""TMCSA phase-resolved conversion billing — one billing unit per CIM-IO.

The VALUE conversion stays in the kernel
`neurox.primitive.analog.current_adc.SarIadc`, which the macro builds
energy-silent; this reporter leaf bills the conversion energy from the magnitude
current, the raw unsigned codes and the reference ladder after `convert` returns,
resolving each binary-search step into the paper's PH2/PH3 conduction phases plus
one data-independent switching event. The
per-step reference path is recovered from the final code through a structural tap
LUT built at init; the ladder itself is passed per call.
"""

from __future__ import annotations

import torch
from torch import Tensor

from neurox.common import ConfigBase, ModuleBase, PolicyBase


class TmcsaConfig(ConfigBase):
    t_ph2__ns: float
    """PH2 conduction duration of one decision step."""
    t_ph3__ns: float
    """PH3 conduction duration of one decision step."""
    e_per_step__fJ: float
    """Data-independent switching energy of one sensing step, per TMCSA instance."""
    area_per_inst__um2: float
    leakage_per_inst__uW: float

    def validate(self) -> None:
        self._require_non_neg(self.t_ph2__ns, "t_ph2__ns")
        self._require_non_neg(self.t_ph3__ns, "t_ph3__ns")
        self._require_non_neg(self.e_per_step__fJ, "e_per_step__fJ")
        self._require_non_neg(self.area_per_inst__um2, "area_per_inst__um2")
        self._require_non_neg(self.leakage_per_inst__uW, "leakage_per_inst__uW")


class TmcsaPolicy(PolicyBase):
    pass


class Tmcsa(ModuleBase[TmcsaConfig, TmcsaPolicy]):
    """TMCSA phase-resolved conversion billing bank — one unit per CIM-IO.

    Billing-only: the value conversion lives in the kernel SAR current ADC.
    Leading dims of the forward tensors are anonymous broadcast batch (the serial
    slot axis rides them).

    Args:
        inst_shape: Fabrication shape `(*inst_shape, gn)` — one billing unit per CIM-IO.
        vdd__V: Supply rail the phase branches conduct across.
    """

    # === Functional buffers ===

    _ref_tap_lut: Tensor  # Shape: [2**max_bits, max_bits]

    # === Circuit constant buffers ===

    _t_ph2__ns: Tensor  # Shape: []
    _t_ph3__ns: Tensor  # Shape: []

    def __init__(
        self,
        *,
        config: TmcsaConfig,
        policy: TmcsaPolicy,
        inst_shape: tuple[int, ...],
        max_bits: int,
        vdd__V: float,
        dtype: torch.dtype,
    ) -> None:
        if max_bits < 1:
            raise ValueError(f"require: max_bits ({max_bits}) >= 1")
        if not (vdd__V >= 0.0):
            raise ValueError(f"require: vdd__V ({vdd__V}) >= 0")
        super().__init__(config=config, policy=policy, inst_shape=inst_shape)
        self._max_bits = max_bits
        self._vdd__V = vdd__V
        self.register_buffer("_t_ph2__ns", torch.tensor(config.t_ph2__ns, dtype=dtype), persistent=False)
        self.register_buffer("_t_ph3__ns", torch.tensor(config.t_ph3__ns, dtype=dtype), persistent=False)
        self.register_buffer("_ref_tap_lut", self._build_ref_tap_lut(self.max_bits), persistent=False)

    @property
    def _area_per_inst__um2(self) -> float:
        return self.config.area_per_inst__um2

    @property
    def _leakage_per_inst__uW(self) -> float:
        return self.config.leakage_per_inst__uW

    @staticmethod
    def _build_ref_tap_lut(bits: int) -> Tensor:
        """Structural code -> per-step reference-tap index LUT.

        The SAR binary-search tap sequence is a bijection of the final unsigned
        code: at step `s` (1-based) only the top `s - 1` final bits are resolved,
        so `tap(s, c) = ((c >> (B-s+1)) << (B-s+1)) + 2^(B-s) - 1` replays the
        kernel converter's own tap selection exactly.
        """
        codes = torch.arange(1 << bits, dtype=torch.long)
        lut = torch.empty((1 << bits, bits), dtype=torch.long)
        for step in range(bits):  # 0-based step; s = step + 1
            shift = bits - step
            prefix = codes >> shift
            lut[:, step] = (prefix << shift) + (1 << (shift - 1)) - 1
        return lut

    @property
    def max_bits(self) -> int:
        """Maximum conversion step count B."""
        return self._max_bits

    def forward(self, i_sub__uA: Tensor, code: Tensor, adc_refs_mode__uA: Tensor, *, bits: int) -> None:
        """Bill phase conduction and fixed switching for one completed conversion.

        Args:
            i_sub__uA: Converted magnitude current — the ISUB output copy the
                TMCSA sinks.
                Shape: `[..., serial, gn]`.
            code: Raw unsigned codes (long) the kernel converter returned for
                `i_sub__uA`.
                Shape: `[..., serial, gn]`.
            adc_refs_mode__uA: Per-instance MAX-BITS reference ladder of the
                selected mode, taps ascending on the last axis, its leading dims
                right-broadcasting against `i_sub__uA` — passed per call, since the
                LUT is structural and the refs are runtime.
                Shape: `[..., 2**max_bits - 1]`.
            bits: Resolution this conversion ran at, in `[1, max_bits]`. A `b`-bit
                conversion truncates the max-bits binary search after its FIRST `b`
                steps and lands on the max-bits code right-shifted by
                `max_bits - b`; the billing therefore takes the leading `b` phase
                windows and looks the reference path up at the re-shifted code.

        Raises:
            ValueError: A shape or tap-count mismatch, or `bits` outside
                `[1, max_bits]`.
        """
        if i_sub__uA.ndim < 1 or (self.inst_shape and i_sub__uA.shape[-1] != self.inst_shape[-1]):
            raise ValueError(
                f"forward() expects i_sub__uA trailing (gn,) = ({self.inst_shape[-1] if self.inst_shape else 1},); "
                f"got shape {tuple(i_sub__uA.shape)}"
            )
        if tuple(code.shape) != tuple(i_sub__uA.shape):
            raise ValueError(
                f"forward() expects code.shape == i_sub__uA.shape ({tuple(i_sub__uA.shape)}); got {tuple(code.shape)}"
            )
        max_bits = self.max_bits
        if not (1 <= bits <= max_bits):
            raise ValueError(f"require: bits ({bits}) in [1, max_bits ({max_bits})]")
        n_taps = int(adc_refs_mode__uA.shape[-1])
        want_taps = (1 << max_bits) - 1
        if n_taps != want_taps:
            raise ValueError(f"forward() expects adc_refs_mode__uA n_taps ({n_taps}) == 2**max_bits - 1 ({want_taps})")

        if not self._is_dynamic_energy_profile_active():
            return

        # Shape: [..., n_taps] -> [..., serial, gn, n_taps]
        ref_b = torch.broadcast_to(adc_refs_mode__uA, (*i_sub__uA.shape, n_taps))
        # The per-step selected reference-path current, recovered from the
        # final code through the structural LUT: the code re-enters the
        # max-bits ladder shifted back up, the leading steps are the ones run.
        # Shape: [..., serial, gn, bits]
        i_ref_path__uA = torch.gather(ref_b, -1, self._ref_tap_lut[:, :bits][code.long() << (max_bits - bits)])

        # Branch-tensor law: materialize BOTH phase branch currents per step.
        # Shape: [..., serial, gn, bits]
        i_common__uA = i_sub__uA.unsqueeze(-1) + i_ref_path__uA
        i_ph2__uA = 3.0 * i_common__uA  # PH2: inputs (1x each) + internal P3/P4 (2x each)
        i_ph3__uA = 2.0 * i_common__uA  # PH3: internal only; the 2x splits into two 1x sinks
        # Shape: [..., serial, gn, bits] -> [..., serial, gn]
        e_conduction__fJ = self._vdd__V * (i_ph2__uA * self._t_ph2__ns + i_ph3__uA * self._t_ph3__ns).sum(dim=-1)
        e__fJ = e_conduction__fJ + bits * self.config.e_per_step__fJ
        # The SAR step axis is already summed above; the collector sums the slot
        # and CIM-IO axes past the caller's leading dims.
        self._record_dynamic_energy(e__fJ)
