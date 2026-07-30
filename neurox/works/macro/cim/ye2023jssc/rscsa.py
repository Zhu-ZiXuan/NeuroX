"""Reference-Subtracting Current Sense Amplifier (RS-CSA) readout.

A current-domain SAR ADC that takes ONE owner-supplied reference current, scales
it internally by its compare phases' binary weights, and quantizes uniformly over
the resulting decision ladder after subtracting a static compensation current.

See also:
    docs/works/macro/cim/ye2023jssc/model.md
"""

from __future__ import annotations

import torch
from torch import Tensor

from neurox.primitive.analog.current_adc.base import Iadc, IadcConfig, IadcPolicy


class RsCsaIadcConfig(IadcConfig):
    """Physical knobs for the Reference-Subtracting CSA readout.

    Attributes:
        bits: Physical resolution [bits] — the phase set holds one compare phase
            per bit, and it is the finest resolution a conversion may request. It
            also fixes the compare phases' binary reference weights: phase ``p``
            (1-based, MSB-first) weighs ``2 ** (bits - p)``, so the reachable
            decision ladder is the uniform ``c * I_ref`` set for
            ``c = 1 .. 2 ** bits - 1``. The quantizer is uniform over a binary
            code space, so no other radix is representable.
        v_rail__V: Supply rail the comparator input mirror conducts across.
        t_phase__ns: Phase durations [ns], ``bits + 1`` entries: the
            compensation phase first, then one compare phase per bit, MSB-first.
            A conversion at ``b`` bits runs the compensation phase and the first
            ``b`` compare phases.
        t4_intrinsic__ns: Delay [ns] from the LAST EXECUTED compare phase's start
            to that phase's comparator output latching; it fits inside every
            compare phase, so it lies in ``[0, min(t_phase__ns[1:])]``.
        mirror_scale: Dimensionless comparator-side mirror scale — the fraction
            of the compared branch current the comparator input mirror draws
            from ``v_rail__V`` during a compare phase.
        e_fixed_per_op__fJ: Code-independent per-conversion baseline energy [fJ]
            over the full-resolution window; a lowered resolution prorates it by
            the executed-window ratio.
    """

    bits: int
    v_rail__V: float
    t_phase__ns: tuple[float, ...]
    t4_intrinsic__ns: float
    mirror_scale: float
    e_fixed_per_op__fJ: float

    def validate(self) -> None:
        super().validate()

        # --- Quantizer ---

        self._require_pos(self.bits, "bits")

        # --- Timing ---

        if len(self.t_phase__ns) != self.bits + 1:
            raise ValueError(
                f"require: len(t_phase__ns) ({len(self.t_phase__ns)}) == bits + 1 ({self.bits + 1}) "
                f"(PH0 + one compare phase per bit)"
            )
        for t in self.t_phase__ns:
            self._require_non_neg(t, "t_phase__ns")
        # The latch offset closes WHICHEVER compare phase runs last, so it must
        # fit inside the shortest of them.
        t_compare_min = min(self.t_phase__ns[1:])
        if not (0.0 <= self.t4_intrinsic__ns <= t_compare_min):
            raise ValueError(
                f"require: t4_intrinsic__ns ({self.t4_intrinsic__ns}) in [0, shortest compare phase ({t_compare_min})]"
            )
        if not (sum(self.t_phase__ns[:-1]) + self.t4_intrinsic__ns > 0.0):
            raise ValueError("require: the full-resolution conversion window (t_phase + t4_intrinsic) > 0")

        # --- Energy ---

        self._require_non_neg(self.v_rail__V, "v_rail__V")
        self._require_non_neg(self.mirror_scale, "mirror_scale")
        self._require_non_neg(self.e_fixed_per_op__fJ, "e_fixed_per_op__fJ")


class RsCsaIadcPolicy(IadcPolicy):
    """Empty nonideality policy for the deterministic RS-CSA readout."""


class RsCsaIadc(Iadc[RsCsaIadcConfig, RsCsaIadcPolicy]):
    """Reference-Subtracting CSA: uniform current quantizer with a static offset.

    The circuit takes ONE reference current and scales it by its compare phases'
    binary weights, so a call injects a single reference tap and the ladder is
    built here; nothing outside the converter states its tap count.

    The converter runs the compensation phase plus ONE compare phase per
    requested bit, so its window, its per-phase energy, and its latency all
    follow the EXECUTED phases; only the code-independent baseline energy is
    apportioned, by the executed-window ratio.

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

    # === Functional buffers ===

    _ladder_weights: Tensor  # Shape: [2**max_bits - 1]

    # === Circuit constant buffers ===

    _t_conversion__ns: Tensor  # Shape: [max_bits]

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
        self._i_ph0_comp__uA = i_ph0_comp__uA
        # A conversion at ``b`` bits runs the compensation phase and the first
        # ``b`` compare phases; the window ends at the last EXECUTED compare
        # phase's comparator latch, which cuts that phase short of its nominal
        # boundary.
        window__ns = [sum(config.t_phase__ns[:b]) + config.t4_intrinsic__ns for b in range(1, config.bits + 1)]
        self.register_buffer(
            "_t_conversion__ns",
            torch.tensor(window__ns, dtype=dtype),
            persistent=False,
        )
        # The code-independent baseline is prorated by the executed-window ratio.
        self._e_fixed_scale = tuple(t / window__ns[-1] for t in window__ns)
        # The compare phases weigh the ONE injected reference by 2**(bits - p),
        # so together they resolve every tap of the uniform ladder below.
        self.register_buffer(
            "_ladder_weights",
            torch.arange(1, 1 << config.bits, dtype=dtype),
            persistent=False,
        )

    @property
    def _area_per_inst__um2(self) -> float:
        return self.config.area_per_inst__um2

    @property
    def _leakage_per_inst__uW(self) -> float:
        return self.config.leakage_per_inst__uW

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

    def t_conversion__ns(self, bits: int) -> Tensor:
        """Return the executed conversion window [ns] (0-d) of a ``bits`` conversion.

        The window spans the compensation phase, the first ``bits - 1`` compare
        phases in full, and the last executed compare phase up to its comparator
        latch.

        Args:
            bits: Conversion resolution [bits] in ``[1, max_bits]``.

        Raises:
            ValueError: ``bits`` is outside ``[1, max_bits]``.
        """
        self._check_bits(bits)
        return self._t_conversion__ns[bits - 1]

    def unsigned_range(self, bits: int) -> tuple[int, int]:
        """Unsigned code endpoints at ``bits`` — ``(0, 2 ** bits - 1)``."""
        self._check_bits(bits)
        return 0, (1 << bits) - 1

    def _convert_impl(
        self,
        i_in__uA: Tensor,
        i_refs__uA: Tensor,
        *,
        bits: int,
    ) -> Tensor:
        """Digitise a magnitude current: subtract the static offset, then quantize.

        The whole ladder stays wired at every resolution, so a request below
        ``max_bits`` widens the effective bin to ``2 ** (max_bits - bits)``
        current steps without moving the transfer: the code is taken at full
        resolution and its unresolved low bits are dropped, which is exactly what
        the first ``bits`` compare phases resolve.

        Energy and latency follow the EXECUTED phases: the per-phase compare term
        sums over the first ``bits`` references, the code-independent baseline is
        prorated by the executed-window ratio, and the window is the one
        :meth:`t_conversion__ns` reports for ``bits``.

        Args:
            i_in__uA: Non-negative magnitude current [uA]. The conversions
                serialized on one converter are the middle axes, and the instance
                axes are the last ones, since the energy is billed at this
                layout.
                Shape: ``[*caller_leading, *middle, *inst_shape]``.
            i_refs__uA: The ONE reference current [uA] the compare phases scale,
                with its leading dims right-broadcasting against ``i_in__uA``. It
                is this circuit's single reference input, so a deeper tap axis is
                rejected.
                Shape: ``[..., 1]``.
            bits: Conversion resolution [bits] in ``[1, max_bits]``.

        Returns:
            Unsigned integer code [int16] in ``[0, 2 ** bits - 1]``, one code per
            ``i_in__uA`` element.
            Shape: ``[*caller_leading, *middle, *inst_shape]``.

        Raises:
            ValueError: ``i_refs__uA`` carries more than one tap.
        """
        config = self.config
        max_bits = self.max_bits

        # --- The single reference input this converter's circuit takes ---

        n_taps = int(i_refs__uA.shape[-1])
        if n_taps != 1:
            raise ValueError(
                f"require: i_refs__uA n_taps ({n_taps}) == 1 — the RS-CSA takes ONE reference current "
                "and scales it by its own compare-phase weights"
            )
        # Shape: [..., 1] -> [...]
        i_ref__uA = i_refs__uA[..., 0]

        # --- PH0: static leakage compensation (operand-independent) ---

        i_comp__uA = (i_in__uA - self._i_ph0_comp__uA).clamp(min=0.0)

        # --- Compare phases: UNIFORM quantize over the self-scaled ladder ---

        # The binary phase weights span the uniform tap set ``c * I_ref``, so the
        # full-resolution code is the number of taps the compensated current
        # clears; a current sitting exactly on a tap clears it. The ladder length
        # bounds the code by 2**max_bits - 1, so no clamp is needed.
        # Shape: [..., 1] * [2**max_bits - 1] -> [..., 2**max_bits - 1]
        ladder__uA = i_refs__uA * self._ladder_weights
        code = (i_comp__uA.unsqueeze(-1) >= ladder__uA).sum(dim=-1).to(torch.int16)

        # --- DATA-DEPENDENT energy: E_fixed + E_code, over the EXECUTED phases ---

        if self._is_dynamic_energy_profile_active():
            # The baseline holds for the executed window alone.
            e__fJ = torch.full_like(i_in__uA, config.e_fixed_per_op__fJ * self._e_fixed_scale[bits - 1])
            # Compare phase p draws min(residue, reference) scaled by
            # `mirror_scale` across the rail for that phase's whole duration.
            # Only the first `bits` phases run; the residue recursion truncates
            # with them.
            i_residue__uA = i_comp__uA
            for phase in range(1, bits + 1):
                # Phase p resolves bit max_bits - p, so it weighs the residue
                # against that bit's place value on the ONE reference.
                i_phase_ref__uA = (1 << (max_bits - phase)) * i_ref__uA
                e__fJ = e__fJ + (config.mirror_scale * config.v_rail__V * config.t_phase__ns[phase]) * (
                    i_residue__uA.clamp(max=i_phase_ref__uA)
                )
                # Cumulative subtraction: the residue drops only where the bit resolves 1.
                i_residue__uA = torch.where(
                    i_residue__uA >= i_phase_ref__uA,
                    i_residue__uA - i_phase_ref__uA,
                    i_residue__uA,
                )
            self._record_dynamic_energy(e__fJ)

        # --- EXECUTED conversion window ---

        if self.enable_latency_record:
            self._record_latency(self._t_conversion__ns[bits - 1] * self._count_serial_rounds(i_in__uA.numel()))

        # Bit width is internal: the full-resolution code drops its unresolved
        # low bits. The shift keeps the int16 code dtype.
        return code >> (max_bits - bits)
