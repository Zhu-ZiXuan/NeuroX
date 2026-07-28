"""Reference-Subtracting Current Sense Amplifier (RS-CSA) readout.

A current-domain SAR ADC that quantizes uniformly over an owner-supplied
decision ladder after subtracting a static compensation current.

See also:
    docs/works/macro/cim/ye2023jssc/model.md
"""

from __future__ import annotations

import torch
from torch import Tensor

from neurox.common.quant import floor_bucketize
from neurox.primitive.analog.current_adc.base import Iadc, IadcConfig, IadcPolicy


class RsCsaIadcConfig(IadcConfig):
    """Physical knobs for the Reference-Subtracting CSA readout.

    Attributes:
        bits: Physical resolution [bits]; the phase set always runs at this
            width, and it is the finest resolution a conversion may request.
        i_lsb__uA: Quantizer LSB current [uA] — one code step of the
            owner-supplied max-bits decision ladder.
        ref_radix: The ``bits`` reference weights, MSB..LSB, strictly descending
            positive ints (e.g. ``(8, 4, 2, 1)``); compare phase ``i`` sizes its
            reference current as ``ref_radix[i] * i_lsb__uA``.
        v_rail__V: Supply rail the comparator input mirror conducts across.
        t_phase__ns: Phase durations [ns], ``bits + 1`` entries: the
            compensation phase first, then one compare phase per bit, MSB-first.
        t4_intrinsic__ns: Delay [ns] from the last compare phase's start to that
            phase's comparator output latching, in ``[0, t_phase__ns[-1]]``.
        mirror_scale: Dimensionless comparator-side mirror scale — the fraction
            of the compared branch current the comparator input mirror draws
            from ``v_rail__V`` during a compare phase.
        e_fixed_per_op__fJ: Code-independent per-conversion baseline energy [fJ].
    """

    bits: int
    i_lsb__uA: float
    ref_radix: tuple[int, ...]
    v_rail__V: float
    t_phase__ns: tuple[float, ...]
    t4_intrinsic__ns: float
    mirror_scale: float
    e_fixed_per_op__fJ: float

    def validate(self) -> None:
        super().validate()

        # --- Quantizer ---

        self._require_pos(self.bits, "bits")
        self._require_pos(self.i_lsb__uA, "i_lsb__uA")
        if len(self.ref_radix) != self.bits:
            raise ValueError(f"require: len(ref_radix) ({len(self.ref_radix)}) == bits ({self.bits})")
        self._require_decreasing(self.ref_radix, "ref_radix")
        for r in self.ref_radix:
            self._require_pos(r, "ref_radix")

        # --- Timing ---

        if len(self.t_phase__ns) != self.bits + 1:
            raise ValueError(
                f"require: len(t_phase__ns) ({len(self.t_phase__ns)}) == bits + 1 ({self.bits + 1}) "
                f"(PH0 + one compare phase per bit)"
            )
        for t in self.t_phase__ns:
            self._require_non_neg(t, "t_phase__ns")
        if not (0.0 <= self.t4_intrinsic__ns <= self.t_phase__ns[-1]):
            raise ValueError(
                f"require: t4_intrinsic__ns ({self.t4_intrinsic__ns}) in [0, last t_phase__ns ({self.t_phase__ns[-1]})]"
            )

        # --- Energy ---

        self._require_non_neg(self.v_rail__V, "v_rail__V")
        self._require_non_neg(self.mirror_scale, "mirror_scale")
        self._require_non_neg(self.e_fixed_per_op__fJ, "e_fixed_per_op__fJ")


class RsCsaIadcPolicy(IadcPolicy):
    """Empty nonideality policy for the deterministic RS-CSA readout."""


class RsCsaIadc(Iadc[RsCsaIadcConfig, RsCsaIadcPolicy]):
    """Reference-Subtracting CSA: uniform current quantizer with a static offset.

    ``i_ph0_comp__uA`` is supplied by the owner at construction, so this class is
    built directly rather than through the ``Iadc`` config-policy registry.

    Args:
        config: Concrete configuration dataclass.
        policy: Per-source nonideality flags.
        inst_shape: Per-instance fabrication shape.
        dtype: Tensor dtype for internal buffers.
        T__K: Operating temperature.
        i_ph0_comp__uA: Static PH0 compensation current [uA] the readout
            subtracts once per conversion; non-negative.
        enable_latency_record: Whether conversions emit latency events.
    """

    # --- Immutable model buffers ---

    _t_conversion__ns: Tensor

    def __init__(
        self,
        *,
        config: RsCsaIadcConfig,
        policy: RsCsaIadcPolicy,
        inst_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
        i_ph0_comp__uA: float,
        enable_latency_record: bool = True,
    ) -> None:
        super().__init__(
            config=config,
            policy=policy,
            inst_shape=inst_shape,
            dtype=dtype,
            T__K=T__K,
            enable_latency_record=enable_latency_record,
        )
        if not (i_ph0_comp__uA >= 0.0):
            raise ValueError(f"require: i_ph0_comp__uA ({i_ph0_comp__uA}) >= 0")
        self._area_per_inst__um2 = config.area_per_inst__um2
        self._leakage_per_inst__uW = config.leakage_per_inst__uW
        self._i_ph0_comp__uA = i_ph0_comp__uA
        # The window ends at the last comparator latch, which cuts the last
        # compare phase short of its nominal boundary.
        self.register_buffer(
            "_t_conversion__ns",
            torch.tensor(sum(config.t_phase__ns[:-1]) + config.t4_intrinsic__ns, dtype=dtype),
            persistent=False,
        )

    def _sample_fabricate_mismatch(self) -> None:
        """No local static state — the RS-CSA model is deterministic."""

    @property
    def max_bits(self) -> int:
        """Physical resolution — the largest ``bits`` a call may request."""
        return self.config.bits

    @property
    def i_ph0_comp__uA(self) -> float:
        """Static PH0 compensation current [uA] subtracted once per conversion."""
        return self._i_ph0_comp__uA

    @property
    def t_conversion__ns(self) -> Tensor:
        """Conversion window [ns] (0-d), closing at the last compare phase's latch."""
        return self._t_conversion__ns

    def unsigned_range(self, bits: int) -> tuple[int, int]:
        """Unsigned code endpoints at ``bits`` — ``(0, 2 ** bits - 1)``."""
        self._check_bits(bits)
        return 0, (1 << bits) - 1

    def _check_bits(self, bits: int) -> None:
        """Require a resolution the physical phase set resolves.

        Raises:
            ValueError: ``bits`` is outside ``[1, config.bits]``.
        """
        if not (1 <= bits <= self.config.bits):
            raise ValueError(f"require: bits ({bits}) in [1, config.bits ({self.config.bits})]")

    def _convert_impl(
        self,
        i_in__uA: Tensor,
        i_refs__uA: Tensor,
        *,
        bits: int,
    ) -> Tensor:
        """Digitise a magnitude current: subtract the static offset, then quantize.

        The analog machine is one fixed operating point: it runs its whole phase
        set and window whatever ``bits`` is asked for. A resolution below
        ``config.bits`` is realized by the owner decimating the decision ladder,
        which widens the bin to ``2 ** (config.bits - bits)`` current steps and
        drops the code's low bits.

        Args:
            i_in__uA: Non-negative magnitude current [uA]. Shape: arbitrary.
            i_refs__uA: Ascending decision ladder, ``[*R, 2 ** bits - 1]`` taps
                spaced by ``2 ** (config.bits - bits) * i_lsb``; ``[*R]``
                right-broadcasts against ``i_in__uA``.
            bits: Conversion resolution [bits] in ``[1, config.bits]``.

        Returns:
            Unsigned integer code [int16] in ``[0, 2 ** bits - 1]``, shaped
            like ``i_in__uA``.
        """
        config = self.config
        self._check_bits(bits)
        n_taps = int(i_refs__uA.shape[-1])
        if n_taps != (1 << bits) - 1:
            raise ValueError(f"require: i_refs__uA n_taps ({n_taps}) == 2**bits - 1 ({(1 << bits) - 1})")

        # --- PH0: static leakage compensation (operand-independent) ---

        i_comp__uA = (i_in__uA - self._i_ph0_comp__uA).clamp(min=0.0)

        # --- Compare phases: UNIFORM quantize over the owner-supplied ladder ---

        code = floor_bucketize(
            i_comp__uA,
            i_refs__uA,
            out_dtype=torch.int16,
            training=False,
            lsb=config.i_lsb__uA * (1 << (config.bits - bits)),
        )
        max_code = (1 << bits) - 1
        code = code.clamp(min=0, max=max_code)

        # --- DATA-DEPENDENT energy: E_fixed + E_code ---

        if self._is_dynamic_energy_profile_active():
            e__fJ = torch.full_like(i_in__uA, config.e_fixed_per_op__fJ)
            # Compare phase i draws min(residue, reference) scaled by
            # `mirror_scale` across the rail for that phase's whole duration.
            i_residue__uA = i_comp__uA
            for phase, radix in enumerate(config.ref_radix, start=1):
                i_ref__uA = radix * config.i_lsb__uA
                e__fJ = e__fJ + (config.mirror_scale * config.v_rail__V * config.t_phase__ns[phase]) * (
                    i_residue__uA.clamp(max=i_ref__uA)
                )
                # Cumulative subtraction: the residue drops only where the bit resolves 1.
                i_residue__uA = torch.where(
                    i_residue__uA >= i_ref__uA,
                    i_residue__uA - i_ref__uA,
                    i_residue__uA,
                )
            self._record_dynamic_energy(e__fJ)

        # --- FIXED conversion window, no early termination ---

        if self.enable_latency_record:
            self._record_latency(self._t_conversion__ns * self._count_serial_rounds(i_in__uA.numel()))

        return code
