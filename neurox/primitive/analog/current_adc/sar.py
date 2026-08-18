"""Triple-margin current-mode successive-approximation ADC.

See Also:
    docs/reference/primitive/analog/current_adc/sar.md
"""

from __future__ import annotations

import torch
from torch import Tensor

from neurox.primitive.nonideality import apply_gaussian

from .base import Iadc, IadcConfig, IadcPolicy


class SarIadcConfig(IadcConfig):
    bits: int
    """Physical (maximum) magnitude resolution [bits]; a `convert` call
    requests any resolution in `[1, bits]`."""
    margin_gain: float
    """Deterministic triple-margin pre-gain applied to the clean
    `i_in - i_ref` before the latch. Both offset sources are added after this
    gain, so an effective offset is `sigma / margin_gain`."""

    e_fixed_per_op__fJ: float
    """Data-independent per-step energy constant folding sampling, latch,
    coupling, and reference-selector switching; billed once per
    binary-search step."""
    v_rail__V: float
    """Supply rail the input and selected reference conduct across during each
    comparison step."""
    t_conduct_per_step__ns: tuple[float, ...]
    """Per-step conduction window, one entry per binary-search step (length
    >= `bits`; a longer tuple is tolerated and only the first `bits` entries
    are drawn). All-zero reduces to the pure fixed-energy model."""

    step_latency__ns: tuple[float, ...]
    """Per-step decision latency, exactly `bits` entries — the search tree has
    no step beyond the physical resolution, and an owner reads the whole tuple
    as the full-resolution sensing duration."""

    comparator_offset_sigma__uA: float
    """Input-referred SA offset σ, as a current-domain margin perturbation."""
    coupling_mismatch_sigma__uA: float
    """Residual coupling-driven offset σ, as a current-domain margin
    perturbation."""

    def validate(self) -> None:
        super().validate()

        # --- Quantizer ---

        self._require_pos(self.bits, "bits")
        self._require_pos(self.margin_gain, "margin_gain")

        # --- Energy and timing ---

        self._require_non_neg(self.e_fixed_per_op__fJ, "e_fixed_per_op__fJ")
        self._require_non_neg(self.v_rail__V, "v_rail__V")
        self._require_min_len(self.t_conduct_per_step__ns, "t_conduct_per_step__ns", self.bits)
        for step, t in enumerate(self.t_conduct_per_step__ns):
            self._require_non_neg(t, f"t_conduct_per_step__ns[{step}]")

        # Exactly one entry per search step: a trailing entry the search never
        # executes would inflate every window derived from the tuple.
        self._require_len(self.step_latency__ns, "step_latency__ns", self.bits)
        for step, latency in enumerate(self.step_latency__ns):
            self._require_non_neg(latency, f"step_latency__ns[{step}]")

        # --- Nonidealities ---

        self._require_non_neg(self.comparator_offset_sigma__uA, "comparator_offset_sigma__uA")
        self._require_non_neg(self.coupling_mismatch_sigma__uA, "coupling_mismatch_sigma__uA")


class SarIadcPolicy(IadcPolicy):
    comparator_offset: bool
    """Inject `comparator_offset_sigma__uA` at fabricate time."""
    coupling_mismatch: bool
    """Inject `coupling_mismatch_sigma__uA` at fabricate time."""


@Iadc.register_neurox_module(
    config_type=SarIadcConfig,
    policy_type=SarIadcPolicy,
)
class SarIadc(Iadc[SarIadcConfig, SarIadcPolicy]):
    """Triple-margin current ADC using a binary search over injected references.

    The search tree is wired for `config.bits`, so this converter reads exactly
    `2 ** bits - 1` ascending taps off the injected ladder's last axis. That
    count is this circuit's own property: a shorter ladder is not rejected up
    front, it fails in the tap gather.

    Args:
        inst_shape: Fabricated shared-sense-lane shape.
        enable_energy_record: Whether conversions emit dynamic-energy events.
            The value conversion is unaffected when disabled.
    """

    # === Nominal buffers ===

    _nominal_comparator_offset__uA: Tensor  # Shape: []
    _nominal_coupling_offset__uA: Tensor  # Shape: []

    # === Fabricated state ===

    _comparator_offset__uA: Tensor  # Shape: [*inst_shape]
    _coupling_offset__uA: Tensor  # Shape: [*inst_shape]

    def __init__(
        self,
        *,
        config: SarIadcConfig,
        policy: SarIadcPolicy,
        inst_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
        enable_energy_record: bool = True,
    ) -> None:
        super().__init__(
            config=config,
            policy=policy,
            inst_shape=inst_shape,
            dtype=dtype,
            T__K=T__K,
        )
        self.enable_energy_record = enable_energy_record
        self._register_fabrication_buffers(dtype=dtype)

    @property
    def _area_per_inst__um2(self) -> float:
        return self.config.area_per_inst__um2

    @property
    def _leakage_per_inst__uW(self) -> float:
        return self.config.leakage_per_inst__uW

    def latency__ns(self, *, bits: int) -> float:
        """One conversion — the summed decision latencies of the executed steps.

        The binary-search steps run sequentially inside the one sense lane, and
        a call at `bits` executes the first `bits` of them.
        """
        self._check_bits(bits)
        return sum(self.config.step_latency__ns[:bits])

    def _register_fabrication_buffers(self, *, dtype: torch.dtype) -> None:
        self.register_buffer("_nominal_comparator_offset__uA", torch.zeros((), dtype=dtype), persistent=False)
        self.register_buffer("_nominal_coupling_offset__uA", torch.zeros((), dtype=dtype), persistent=False)

    @property
    def max_bits(self) -> int:
        return self.config.bits

    def unsigned_range(self, bits: int) -> tuple[int, int]:
        """Unsigned magnitude code endpoints at `bits` — `(0, 2 ** bits - 1)`."""
        self._check_bits(bits)
        return 0, (1 << bits) - 1

    def _sample_fabrication_variation(self) -> None:
        self._comparator_offset__uA = apply_gaussian(
            self._nominal_comparator_offset__uA.clone().expand(self.inst_shape),
            self.config.comparator_offset_sigma__uA,
            enabled=self.policy.comparator_offset,
        )
        self._coupling_offset__uA = apply_gaussian(
            self._nominal_coupling_offset__uA.clone().expand(self.inst_shape),
            self.config.coupling_mismatch_sigma__uA,
            enabled=self.policy.coupling_mismatch,
        )

    def _col_to_lane(self, n_col: int, device: torch.device) -> Tensor:
        """Map logical columns to contiguous lanes by `(column * n_lane) // n_col`."""
        n_lane = self.inst_shape[-1] if self.inst_shape else 1
        # Shape: [n_col]
        return (torch.arange(n_col, device=device) * n_lane) // n_col

    def _convert_impl(
        self,
        i_in__uA: Tensor,
        i_refs__uA: Tensor,
        *,
        bits: int,
    ) -> Tensor:
        """Quantize `i_in` with a `bits`-step TRUNCATED binary search.

        The search tree is always the max-bits one over the full ladder: the
        first compare sits at tap `2 ** (max_bits - 1) - 1` regardless of
        `bits`, and a `bits`-bit conversion stops after the first `bits`
        levels. Those levels resolve the max-bits code's leading `bits` bits,
        so the returned code is the max-bits code right-shifted by
        `max_bits - bits`. Energy follows the executed steps.

        Args:
            i_in__uA: Unsigned magnitude current.
                Shape: `[..., n_col]`.
            i_refs__uA: Per-instance reference ladder with the taps on the last
                axis and the leading dims broadcasting right-aligned against
                `i_in__uA`. The gather indexes taps in max-bits numbering, so
                this converter takes `n_ref = 2 ** max_bits - 1` ascending taps.
                Shape: `[..., n_ref]`.
            bits: Conversion resolution [bits], in `[1, max_bits]`.

        Returns:
            Unsigned magnitude code [long] in `[0, 2 ** bits - 1]`.
            Shape: `[..., n_col]`.
        """
        max_bits = self.max_bits
        n_taps = int(i_refs__uA.shape[-1])
        # Shape: [..., n_ref] -> [..., n_col, n_ref]
        ref_b = torch.broadcast_to(i_refs__uA, (*i_in__uA.shape, n_taps))

        margin_gain = self.config.margin_gain
        e_fixed = self.config.e_fixed_per_op__fJ
        v_rail = self.config.v_rail__V
        t_conduct = self.config.t_conduct_per_step__ns

        code = torch.zeros_like(i_in__uA, dtype=torch.long)
        record_energy = self.enable_energy_record and self._is_dynamic_energy_profile_active()
        e_dyn__fJ = torch.zeros_like(i_in__uA) if record_energy else None

        # The fabricated lane offset remains fixed throughout the binary search.
        lane = self._col_to_lane(i_in__uA.shape[-1], i_in__uA.device)
        # Shape: [..., n_lane] -> [..., n_col]
        offset__uA = (self._comparator_offset__uA + self._coupling_offset__uA).index_select(-1, lane)

        for step in range(bits):
            i_ref__uA = self._select_ref(ref_b, code, step, max_bits)

            clean_margin__uA = i_in__uA - i_ref__uA
            bit = (margin_gain * clean_margin__uA + offset__uA) > 0.0
            code = self._set_bit(code, step, bit, max_bits)

            if e_dyn__fJ is not None:
                # V * uA * ns = fJ.
                e_dyn__fJ = (
                    e_dyn__fJ
                    + e_fixed
                    + v_rail * (i_in__uA + i_ref__uA) * t_conduct[step]
                    + self._compute_input_dynamic_energy__fJ(i_in__uA, i_ref__uA)
                )

        if e_dyn__fJ is not None:
            self._record_dynamic_energy(e_dyn__fJ)

        # The executed levels sit at the TOP of the max-bits code; the
        # unresolved trailing levels are dropped.
        return code >> (max_bits - bits)

    def _compute_input_dynamic_energy__fJ(self, i_in__uA: Tensor, i_ref__uA: Tensor) -> Tensor:
        """Additional per-step conduction energy [fJ], one value per input element."""
        return torch.zeros_like(i_in__uA)

    def _select_ref(self, ref_b: Tensor, code: Tensor, step: int, max_bits: int) -> Tensor:
        """Mid-point reference [uA] for `step`, data-dependent on resolved bits.

        Args:
            ref_b: Per-instance threshold ladder with the taps on the last
                axis, already broadcast against the input.
                Shape: `[..., n_col, n_ref]`.
            code: Partial magnitude code in max-bits numbering, with the high
                `step` bits set.
                Shape: `[..., n_col]`.
            step: Zero-based search level (0 is the max-bits MSB).
            max_bits: Bit width the full ladder resolves.

        Returns:
            Selected reference current [uA], one tap per `code` element.
            Shape: `[..., n_col]`.
        """
        # Threshold index: prefix * 2^(max_bits-step) + 2^(max_bits-step-1) - 1.
        shift = max_bits - step
        prefix = code >> shift
        idx = (prefix << shift) + (1 << (shift - 1)) - 1
        # Shape: [..., n_col, n_ref] -> [..., n_col]
        return torch.gather(ref_b, -1, idx.unsqueeze(-1)).squeeze(-1)

    def _set_bit(self, code: Tensor, step: int, bit: Tensor, max_bits: int) -> Tensor:
        """Write the `step`-th search level (MSB-first) into the max-bits `code`."""
        bit_pos = max_bits - 1 - step
        return code | (bit.long() << bit_pos)
