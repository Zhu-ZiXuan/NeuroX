"""Triple-margin current-mode successive-approximation ADC.

A single current-mode sense amplifier is time-multiplexed over ``bits``
sequential comparisons — a binary search across ``2 ** bits - 1``
caller-supplied mid-point reference levels — producing an ``bits``-bit
**unsigned** magnitude code from a single-ended magnitude current
``i_in__uA``. The sign is handled outside the ADC by the caller.

Each single comparison mirrors ``i_in`` and the step's reference ``i_ref``
into the sense amplifier, then a deterministic ``margin_gain`` pre-gain
amplifies the clean current
difference ``i_in - i_ref`` before the latch resolves its sign. The
input-referred comparator offset is a current-domain margin perturbation added
**after** the ``margin_gain`` pre-gain, so the effective offset is divided by
``margin_gain`` (the triple-margin benefit): a raw ``sigma`` offset acts as
``sigma / margin_gain`` at the decision.

The mid-point thresholds arrive per call as ``i_refs__uA``, a **per-instance**
tap ladder ``[*R, n_ref]`` with ``n_ref = 2 ** bits - 1`` as the last axis
(taps ascending), and the resolution ``bits`` arrives per call too. The leading
``[*R]`` broadcasts (right-aligned) against ``i_in__uA``, so each ADC instance
reads its own ladder. The caller (the composing macro) has already selected the
operating mode's row, so mode is invisible to the ADC and switching modes costs
no per-conversion energy. The ADC self-holds no reference — the caller's
reference block is the single ladder source, so a single-ended and a
differential current ADC are parallel classes reading the same injected taps,
never a wrapper pattern.

The binary-search steps are an internal Python loop — the sub-comparisons are not
separate profiled leaves, so the whole conversion emits exactly **one**
dynamic-energy event and (when ``record_latency`` is set) **one** latency event
(``sum(step_latency__ns[:bits])``). ``record_latency=False`` suppresses only the
latency event; the dynamic energy — including the per-step ``t_conduct_per_step``
conduction — is emitted regardless.

Dynamic energy per sensing step is the per-step switching constant
``e_fixed_per_op`` (sampling, latch, coupling, reference-selector switching)
plus the current-domain conduction a current ADC necessarily draws while it
compares — its input and the selected reference conduct across ``v_rail__V`` for
``t_conduct_per_step__ns[s]``: ``e_step = e_fixed_per_op +
v_rail__V * (i_in + i_ref_step) * t_conduct_per_step__ns[s] +
_input_dynamic_energy__fJ(i_in, i_ref_step)``. The last term is an overridable
hook (base zero) for structures outside this parameterized form. All-zero
``t_conduct_per_step__ns`` reduces to the pure fixed-energy model
(``bits * e_fixed_per_op`` per element). The unity input and reference legs
sourced upstream are billed there; this term is the ADC's own comparison
conduction.

The single sense amplifier is heavily time-shared: it is multiplexed over the
``bits`` binary-search steps and across a whole set of columns. Its fabricated
``inst_shape`` is the real shared sense-lane count, NOT one per column. Dynamic
energy stays per-column (the forward tensor already sums all columns);
``inst_shape`` sets only the PPA multiplicity and the serial-latency
time-multiplex factor.

See also:
    docs/reference/primitive/analog/current_adc/sar.md
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from neurox.primitive.nonideality import apply_gaussian

from .base import SingleEndedCurrentAdc, SingleEndedCurrentAdcConfig, SingleEndedCurrentAdcPolicy


@dataclass(frozen=True)
class SarSingleEndedCurrentAdcConfig(SingleEndedCurrentAdcConfig):
    """Physical knobs for the triple-margin current-mode SAR ADC.

    The mid-point threshold ladder is NOT a config field: it arrives per call
    as ``i_refs__uA`` (the caller's reference block is the single source), so
    the config carries only the fixed structural and energy knobs.

    Attributes:
        bits: Physical (maximum) magnitude resolution [bits]; a ``convert``
            call requests any resolution in ``[1, bits]``.
        margin_gain: Deterministic triple-margin pre-gain applied to the clean
            ``i_in - i_ref`` before the latch. The input-referred offset is added
            after this gain, so the effective offset is ``offset_sigma / margin_gain``.
        e_fixed_per_op__fJ: Data-independent per-step energy constant [fJ]
            folding sampling, latch, coupling, and reference-selector switching;
            billed once per binary-search step.
        v_rail__V: Supply rail the input and selected reference conduct across
            during each comparison step [V]; drives the per-step conduction
            energy term.
        t_conduct_per_step__ns: Per-step conduction window [ns], one entry per
            binary-search step (length ``>= bits``; a longer list is tolerated
            and only the first ``bits`` entries are drawn). All-zero reduces to
            the pure fixed-energy model.
        step_latency__ns: Per-step decision latency [ns], one entry per
            binary-search step (length ``>= bits``; a longer list is
            tolerated and only the first requested-resolution entries are
            summed). The conversion latency is the sum of the first ``bits``
            (the requested resolution) entries.
        comparator_offset_sigma__uA: Input-referred SA offset sigma [uA] — a
            current-domain margin perturbation added to the clean ``i_in - i_ref``
            after the ``margin_gain`` pre-gain (effective ``sigma / margin_gain``).
        coupling_mismatch_sigma__uA: Residual coupling-driven offset sigma [uA] — a
            current-domain margin perturbation added after the ``margin_gain``
            pre-gain (effective ``sigma / margin_gain``).
        mirror_mismatch_sigma_relative: Relative multiplicative sigma on the
            mirror ratios (gated by ``mirror_mismatch``).
        area_per_inst__um2: SA silicon area per fabricated shared sense-lane
            instance [um²].
        leakage_per_inst__uW: SA static leakage per fabricated shared sense-lane
            instance [uW].
    """

    bits: int
    margin_gain: float

    e_fixed_per_op__fJ: float
    v_rail__V: float
    t_conduct_per_step__ns: tuple[float, ...]

    step_latency__ns: tuple[float, ...]

    # --- Nonideality sigmas (gated by policy) ---
    comparator_offset_sigma__uA: float
    coupling_mismatch_sigma__uA: float
    mirror_mismatch_sigma_relative: float

    def validate(self) -> None:
        self.validate_quantizer()
        self.validate_energy()
        self.validate_timing()
        self.validate_nonideality()
        self.validate_ppa()

    def validate_quantizer(self) -> None:
        self._require_pos(self.bits, "bits")
        self._require_pos(self.margin_gain, "margin_gain")

    def validate_energy(self) -> None:
        self._require_non_neg(self.e_fixed_per_op__fJ, "e_fixed_per_op__fJ")
        self._require_non_neg(self.v_rail__V, "v_rail__V")
        # One conduction window per binary-search step; a longer list is allowed
        # (only the first bits entries are drawn) so a shared shipped list can
        # cover several bit widths. Each entry non-negative; all-zero = pure
        # fixed-energy.
        if len(self.t_conduct_per_step__ns) < self.bits:
            raise ValueError(
                f"require: len(t_conduct_per_step__ns) ({len(self.t_conduct_per_step__ns)}) >= bits ({self.bits})"
            )
        for t in self.t_conduct_per_step__ns:
            self._require_non_neg(t, "t_conduct_per_step__ns")

    def validate_timing(self) -> None:
        # One entry per binary-search step; a longer list is tolerated. A
        # convert call sums only the first requested-resolution entries, so a
        # shared shipped list can cover several bit widths up to the physical
        # bits without overbilling. Require at least bits so every legal
        # runtime resolution has a latency window.
        if len(self.step_latency__ns) < self.bits:
            raise ValueError(f"require: len(step_latency__ns) ({len(self.step_latency__ns)}) >= bits ({self.bits})")
        for latency in self.step_latency__ns:
            self._require_non_neg(latency, "step_latency__ns")

    def validate_nonideality(self) -> None:
        self._require_non_neg(self.comparator_offset_sigma__uA, "comparator_offset_sigma__uA")
        self._require_non_neg(self.coupling_mismatch_sigma__uA, "coupling_mismatch_sigma__uA")
        self._require_non_neg(self.mirror_mismatch_sigma_relative, "mirror_mismatch_sigma_relative")


@dataclass(frozen=True)
class SarSingleEndedCurrentAdcPolicy(SingleEndedCurrentAdcPolicy):
    """Per-source toggles selecting which SarSingleEndedCurrentAdc nonidealities are active.

    The ``all_off`` preset disables every toggle so the SA quantizer reduces to
    the ideal triple-margin binary search against the injected ``i_refs__uA``.

    Attributes:
        comparator_offset: Inject ``comparator_offset_sigma__uA`` as a
            current-domain margin perturbation added **after** ``margin_gain``
            (effective offset ``sigma / margin_gain`` — the triple-margin benefit).
        replica_threshold_variation: Track cell-current sigma on each reference
            level. Wired but inert, which uses the nominal config tuple.
        mirror_mismatch: Perturb the mirror ratios by ``mirror_mismatch_sigma_relative``.
        coupling_mismatch: Inject ``coupling_mismatch_sigma__uA`` as a residual
            current-domain margin perturbation added after ``margin_gain``
            (effective ``sigma / margin_gain``).
    """

    comparator_offset: bool
    replica_threshold_variation: bool
    mirror_mismatch: bool
    coupling_mismatch: bool


@SingleEndedCurrentAdc.register_key(SarSingleEndedCurrentAdcConfig)
class SarSingleEndedCurrentAdc(SingleEndedCurrentAdc[SarSingleEndedCurrentAdcConfig, SarSingleEndedCurrentAdcPolicy]):
    """Triple-margin current-mode SA time-multiplexed over an ``bits``-step binary search.

    A single current-mode SA realizes an ``bits``-bit unsigned magnitude ADC by
    comparing ``i_in`` against ``bits`` binary-search-selected mid-point
    references out of the ``2 ** bits - 1`` taps of the per-call **per-instance**
    ``i_refs__uA`` ladder the caller supplies — shape ``[*R, n_ref]`` with the
    taps on the last axis and ``[*R]`` broadcasting against ``i_in`` (the caller
    has already selected the mode's row; mode switching costs no per-conversion
    energy). The sub-comparisons are not separate profiled leaves, so
    :meth:`convert` aggregates the steps and emits exactly one dynamic-energy
    event and, when ``record_latency`` is set, one latency event.
    """

    comparator_offset__uA: Tensor
    coupling_offset__uA: Tensor

    def __init__(
        self,
        *,
        config: SarSingleEndedCurrentAdcConfig,
        policy: SarSingleEndedCurrentAdcPolicy,
        inst_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
        record_latency: bool = True,
    ) -> None:
        """Construct one triple-margin current-mode SA quantizer.

        Args:
            config: Concrete configuration dataclass.
            policy: Per-source nonideality enable flags.
            inst_shape: Per-instance fabrication shape ``(*prefix, n_lane)`` — one
                time-multiplexed SA per shared sense lane, NOT one per column. The
                forward ``convert`` still processes per-column tensors (functional
                codes unchanged); ``inst_shape`` only sets the PPA multiplicity and
                the serial-latency time-multiplex factor.
            dtype: Tensor dtype for internal buffers.
            T__K: Operating temperature [K].
            record_latency: Emit the per-conversion latency event when set. When
                ``False`` the latency event is suppressed (the composing macro
                bills the shared cycle latency itself); the dynamic energy —
                including the ``t_conduct_per_step`` conduction — is unaffected.
        """
        super().__init__(
            config=config,
            policy=policy,
            inst_shape=inst_shape,
            dtype=dtype,
            T__K=T__K,
            record_latency=record_latency,
        )
        self._area_per_inst__um2 = config.area_per_inst__um2
        self._leakage_per_inst__uW = config.leakage_per_inst__uW
        self.dtype = dtype
        self.T__K = T__K

        # --- Static per-lane input-referred SA offsets (fabricate-once) ---
        # Both are current-domain margin offsets at the real shared-lane count
        # (``inst_shape[-1]``), HELD CONSTANT across the binary-search steps;
        # OFF => 0 (no perturbation). ``convert`` reads them gathered to the
        # forward logical-column axis.
        self.register_buffer("comparator_offset__uA", torch.zeros(inst_shape, dtype=dtype), persistent=False)
        self.register_buffer("coupling_offset__uA", torch.zeros(inst_shape, dtype=dtype), persistent=False)

    # --- runtime-mode introspection ---

    @property
    def max_bits(self) -> int:
        """Physical magnitude resolution — the maximum ``bits`` a call may request."""
        return self.config.bits

    def unsigned_range(self, bits: int) -> tuple[int, int]:
        """Unsigned magnitude code endpoints at ``bits`` — ``(0, 2 ** bits - 1)``."""
        if not (1 <= bits <= self.max_bits):
            raise ValueError(f"bits {bits} outside [1, {self.max_bits}]")
        return 0, (1 << bits) - 1

    # --- fabricate (static non-idealities) ---

    def _sample_fabricate_mismatch(self) -> None:
        """Sample the static per-lane SA input-referred offsets at ``inst_shape`` (re-callable).

        Draws each shared-lane comparator input-referred offset and coupling-mismatch
        residual ONCE as a fixed current-domain offset, held constant across every
        binary-search step; both are 0 when their policy toggle is off. ``convert``
        reads these buffers gathered to the forward logical-column axis.
        """
        self.comparator_offset__uA = apply_gaussian(
            torch.zeros_like(self.comparator_offset__uA),
            self.config.comparator_offset_sigma__uA,
            enabled=self.policy.comparator_offset,
        )
        self.coupling_offset__uA = apply_gaussian(
            torch.zeros_like(self.coupling_offset__uA),
            self.config.coupling_mismatch_sigma__uA,
            enabled=self.policy.coupling_mismatch,
        )
        # TODO(domain-author): the mirror-ratio mismatch (``mirror_mismatch`` /
        # ``mirror_mismatch_sigma_relative``) and the ``replica_threshold_variation``
        # reference tracking are not modeled yet; the per-instance sampling law is
        # a domain decision.

    def _col_to_lane(self, n_col: int, device: torch.device) -> Tensor:
        """Fixed logical-column -> shared-sense-lane gather index.

        The single SA is time-shared across a whole sense lane: its fabricated
        offsets live at the real lane count ``n_lane = inst_shape[-1]`` while the
        forward tensor carries ``n_col`` logical columns. The ``n_col`` columns
        partition into ``n_lane`` contiguous equal blocks, one per lane, so
        column ``c`` maps to lane ``(c * n_lane) // n_col`` — every column a lane
        time-serves shares that lane's single static offset.
        """
        n_lane = self._inst_shape[-1] if self._inst_shape else 1
        # Shape: [n_col]
        return (torch.arange(n_col, device=device) * n_lane) // n_col

    # --- Conversion ---

    def _convert_impl(
        self,
        i_in__uA: Tensor,
        i_refs__uA: Tensor,
        *,
        bits: int,
    ) -> Tensor:
        """Quantize ``i_in`` to an unsigned magnitude code via ``bits``-step binary search.

        Consumes the per-call per-instance reference ladder and resolution:
        ``i_refs__uA`` is the ``[*R, n_ref]`` ladder (``n_ref = 2 ** bits - 1``
        taps ascending on the last axis, ``[*R]`` broadcasting right-aligned
        against ``i_in``) the caller has already selected for the operating mode,
        and ``bits`` (in ``[1, config.bits]``) drives the binary search. Each
        step selects a reference (data-dependent on the bits resolved so far),
        forms the clean margin ``i_in - i_ref``, applies the ``margin_gain``
        pre-gain and adds the input-referred offset **after** it, and resolves
        one bit. The step energies are aggregated and one dynamic-energy event —
        plus, when ``record_latency`` is set, one latency event
        (``sum(step_latency__ns[:bits])``) — is emitted on completion. The
        sub-comparisons are a method-internal Python loop, never separate leaves.

        Args:
            i_in__uA: Unsigned magnitude current [uA]. Shape: ``[..., n_col]``.
            i_refs__uA: Per-instance reference ladder [uA], ``[*R, n_ref]`` with
                ``n_ref = 2 ** bits - 1`` taps ascending on the last axis and
                ``[*R]`` broadcasting right-aligned against ``i_in__uA``; supplied
                per call by the caller (mode row already selected).
            bits: Conversion resolution [bits], in ``[1, config.bits]``.

        Returns:
            Unsigned magnitude code [long] in ``[0, 2 ** bits - 1]``. Shape:
            ``[..., n_col]``. The sign is combined by the caller.
        """
        if not (1 <= bits <= self.config.bits):
            raise ValueError(f"require: bits ({bits}) in [1, config.bits ({self.config.bits})]")
        n_taps = int(i_refs__uA.shape[-1])
        if n_taps != (1 << bits) - 1:
            raise ValueError(f"require: i_refs__uA n_taps ({n_taps}) == 2**bits - 1 ({(1 << bits) - 1})")
        # Shape: [*R, n_ref] -> [..., n_col, n_ref]
        ref_b = torch.broadcast_to(i_refs__uA, (*i_in__uA.shape, n_taps))

        margin_gain = self.config.margin_gain
        e_fixed = self.config.e_fixed_per_op__fJ
        v_rail = self.config.v_rail__V
        t_conduct = self.config.t_conduct_per_step__ns

        code = torch.zeros_like(i_in__uA, dtype=torch.long)
        e_dyn__fJ = torch.zeros_like(i_in__uA)

        # Static input-referred SA offset gathered ONCE to the forward
        # logical-column axis and HELD CONSTANT across all binary-search steps:
        # the sum of the comparator and coupling offsets (0 when off). It is
        # added AFTER the margin_gain pre-gain, so the effective decision offset
        # is sigma / margin_gain (the triple-margin benefit).
        lane = self._col_to_lane(i_in__uA.shape[-1], i_in__uA.device)
        # Shape: [*prefix, n_lane] -> [*prefix, n_col]
        offset__uA = (self.comparator_offset__uA + self.coupling_offset__uA).index_select(-1, lane)

        for step in range(bits):
            i_ref__uA = self._select_ref(ref_b, code, step, bits)

            clean_margin__uA = i_in__uA - i_ref__uA
            bit = (margin_gain * clean_margin__uA + offset__uA) > 0.0
            code = self._set_bit(code, step, bit, bits)

            # --- Per-step dynamic energy ---
            # Fixed switching constant plus the current-domain conduction a
            # current ADC necessarily draws while it compares: its input and the
            # selected reference conduct across v_rail__V for the step window
            # (V * uA * ns = fJ), plus an overridable hook (base zero) for
            # structures outside this parameterized form. The unity input and
            # reference legs sourced upstream are billed there.
            e_dyn__fJ = (
                e_dyn__fJ
                + e_fixed
                + v_rail * (i_in__uA + i_ref__uA) * t_conduct[step]
                + self._input_dynamic_energy__fJ(i_in__uA, i_ref__uA)
            )

        # --- One aggregated energy event (always) + one latency event (gated) ---

        self._log_dynamic_energy(e_dyn__fJ)
        if self.record_latency:
            latency__ns = torch.tensor(
                sum(self.config.step_latency__ns[:bits]) * self._serial_op_count(i_in__uA),
                device=i_in__uA.device,
                dtype=e_dyn__fJ.dtype,
            )
            self._log_latency(latency__ns)

        return code

    def _input_dynamic_energy__fJ(self, i_in__uA: Tensor, i_ref__uA: Tensor) -> Tensor:
        """Extra per-step data-dependent conduction energy [fJ]; base zero.

        Escape-hatch hook for a structural subclass whose comparison draws
        current outside the parameterized ``v_rail__V * (i_in + i_ref) *
        t_conduct`` form (subclass only for structural diversity; coefficient
        diversity stays in config). Called once per binary-search step with the
        step's input and selected reference; the base returns a zero broadcast
        to ``i_in__uA`` so the parameterized term stands alone.
        """
        return torch.zeros_like(i_in__uA)

    # --- Binary-search helpers (method-internal; not leaves) ---

    def _select_ref(self, ref_b: Tensor, code: Tensor, step: int, bits: int) -> Tensor:
        """Mid-point reference [uA] for ``step``, data-dependent on resolved bits.

        Binary search over the ladder's ``2 ** bits - 1`` nominal mid-point
        thresholds: the partial code from the bits resolved so far (MSB-first)
        indexes the reference for the current step. Step 0 selects the central
        threshold; each later step bisects the surviving sub-interval. The lookup
        is a per-element gather along the last (tap) axis of the broadcast
        per-instance ladder, so every logical column reads its own tap.

        Args:
            ref_b: Per-instance threshold ladder broadcast to ``[..., n_col,
                2 ** bits - 1]`` (taps on the last axis).
            code: Partial magnitude code with the high ``step`` bits set.
                Shape: ``[..., n_col]``.
            step: Zero-based binary-search step (``0`` is the MSB).
            bits: Conversion resolution [bits] for this call.

        Returns:
            Selected reference current [uA] broadcast to ``code``'s shape.
        """
        # High `step` bits of `code` are resolved (MSB-first); they live in
        # bit positions [bits - step, bits - 1]. The threshold index of the
        # interval boundary tested at this step is `prefix*2^(bits-step) + 2^(bits-step-1) - 1`.
        shift = bits - step
        prefix = code >> shift
        idx = (prefix << shift) + (1 << (shift - 1)) - 1
        # Shape: [..., n_col, n_ref] -> [..., n_col]
        return torch.gather(ref_b, -1, idx.unsqueeze(-1)).squeeze(-1)

    def _set_bit(self, code: Tensor, step: int, bit: Tensor, bits: int) -> Tensor:
        """Write the ``step``-th magnitude bit (MSB-first) into ``code``."""
        bit_pos = bits - 1 - step
        return code | (bit.long() << bit_pos)

    def _serial_op_count(self, out: Tensor) -> int:
        """Serial op count for the latency multiplier.

        The per-column SA hardware is parallel (the ``inst_count`` trailing
        positions divide out); the serial latency tracks the broadcast-leading
        instances, matching every other emitting forward leaf.
        """
        return max(1, out.numel() // max(self.inst_count, 1))
