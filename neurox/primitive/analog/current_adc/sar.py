"""Triple-margin current-mode successive-approximation ADC.

A single current-mode sense amplifier is time-multiplexed over ``n_bits``
sequential comparisons — a binary search across ``2 ** n_bits - 1`` nominal
mid-point reference levels — producing an ``n_bits``-bit **unsigned** magnitude
code from a single-ended magnitude current ``i_in__uA``. The sign is handled
outside the ADC by the caller.

Each single comparison mirrors ``i_in`` and the step's reference ``i_ref``
through input mirrors (sized ``input_mirror_ratio`` times the reference legs),
then a deterministic ``margin_gain`` pre-gain amplifies the clean current
difference ``i_in - i_ref`` before the latch resolves its sign. The
input-referred comparator offset is a current-domain margin perturbation added
**after** the ``margin_gain`` pre-gain, so the effective offset is divided by
``margin_gain`` (the triple-margin benefit): a raw ``sigma`` offset acts as
``sigma / margin_gain`` at the decision.

The nominal mid-point thresholds ``ref_levels__uA`` are a config tuple; the ADC
reads them directly and self-holds no external reference.

The binary-search steps are an internal Python loop — the sub-comparisons are not
separate profiled leaves, so the whole conversion emits exactly **one**
dynamic-energy event and **one** latency event (``sum(step_latency__ns)``).

Dynamic energy is the data-dependent regeneration the sense amplifier's sized
mirror controls, plus one data-independent per-op constant: per sensing step
``e_dyn_step = v_rail_sa * n * (i_in_pos + i_ref_pos) * t_eff + e_fixed_per_op``
with ``n = input_mirror_ratio`` (the regeneration legs). The control-based
attribution bills only the currents whose magnitude the ADC controls: the unity
input and reference legs are owned/billed by the upstream blocks that source
them, so neither is re-billed here. ``e_fixed_per_op`` folds the data-independent
sampling, latch, coupling, and reference-selector switching into a single per-op
constant.

The single sense amplifier is heavily time-shared: it is multiplexed over the
``n_bits`` binary-search steps and across a whole set of columns. Its fabricated
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

from neurox.primitive.analog.adc_common import AdcOperationPoint
from neurox.primitive.nonideality import apply_gaussian

from .base import CurrentAdc, CurrentAdcConfig, CurrentAdcPolicy


@dataclass(frozen=True)
class SarCurrentAdcConfig(CurrentAdcConfig):
    """Physical knobs for the triple-margin current-mode SAR ADC.

    Attributes:
        n_bits: Output magnitude resolution [bits].
        margin_gain: Deterministic triple-margin pre-gain applied to the clean
            ``i_in - i_ref`` before the latch. The input-referred offset is added
            after this gain, so the effective offset is ``offset_sigma / margin_gain``.
        input_mirror_ratio: Regeneration mirror ratio relative to the unity legs;
            sets the ``input_mirror_ratio * I`` replica legs the SA rails feed.
        ref_levels__uA: ``2 ** n_bits - 1`` nominal mid-point thresholds [uA],
            strictly increasing.
        v_rail_sa__V: SA-leg overdrive ``V_DD_SA - V_node`` [V] the regenerated
            ``n``-path replica currents are pulled across.
        t_eff__ns: Effective conduction time [ns] the ``n``-path regeneration
            current is drawn over.
        e_fixed_per_op__fJ: Single data-independent per-op energy constant [fJ]
            folding sampling, latch, coupling, and reference-selector switching.
        step_latency__ns: Per-step decision latency, one entry per binary-search
            step [ns]; the conversion latency is their sum.
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

    n_bits: int
    margin_gain: float
    input_mirror_ratio: float
    ref_levels__uA: tuple[float, ...]

    v_rail_sa__V: float
    t_eff__ns: float
    e_fixed_per_op__fJ: float

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
        self._require_pos(self.n_bits, "n_bits")
        self._require_pos(self.margin_gain, "margin_gain")
        self._require_pos(self.input_mirror_ratio, "input_mirror_ratio")
        self._require_min_length(self.ref_levels__uA, 2**self.n_bits - 1, "ref_levels__uA")
        self._require_increasing(self.ref_levels__uA, "ref_levels__uA")

    def validate_energy(self) -> None:
        self._require_non_neg(self.v_rail_sa__V, "v_rail_sa__V")
        self._require_non_neg(self.t_eff__ns, "t_eff__ns")
        self._require_non_neg(self.e_fixed_per_op__fJ, "e_fixed_per_op__fJ")

    def validate_timing(self) -> None:
        self._require_min_length(self.step_latency__ns, self.n_bits, "step_latency__ns")
        for latency in self.step_latency__ns:
            self._require_non_neg(latency, "step_latency__ns")

    def validate_nonideality(self) -> None:
        self._require_non_neg(self.comparator_offset_sigma__uA, "comparator_offset_sigma__uA")
        self._require_non_neg(self.coupling_mismatch_sigma__uA, "coupling_mismatch_sigma__uA")
        self._require_non_neg(self.mirror_mismatch_sigma_relative, "mirror_mismatch_sigma_relative")


@dataclass(frozen=True)
class SarCurrentAdcPolicy(CurrentAdcPolicy):
    """Per-source toggles selecting which SarCurrentAdc nonidealities are active.

    The ``all_off`` preset disables every toggle so the SA quantizer reduces to
    the ideal triple-margin binary search against the nominal ``ref_levels__uA``.

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


@CurrentAdc.register_key(SarCurrentAdcConfig)
class SarCurrentAdc(CurrentAdc):
    """Triple-margin current-mode SA time-multiplexed over an ``n_bits``-step binary search.

    A single current-mode SA realizes an ``n_bits``-bit unsigned magnitude ADC by
    comparing ``i_in`` against ``n_bits`` binary-search-selected mid-point
    references out of the ``2 ** n_bits - 1`` nominal ``ref_levels__uA``. The
    sub-comparisons are not separate profiled leaves, so :meth:`convert`
    aggregates the steps and emits exactly one dynamic-energy + one latency event.
    """

    config: SarCurrentAdcConfig
    policy: SarCurrentAdcPolicy
    ref_levels__uA: Tensor
    comparator_offset__uA: Tensor
    coupling_offset__uA: Tensor

    def __init__(
        self,
        *,
        config: SarCurrentAdcConfig,
        policy: SarCurrentAdcPolicy,
        inst_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
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
        """
        super().__init__(
            config=config,
            policy=policy,
            inst_shape=inst_shape,
            dtype=dtype,
            T__K=T__K,
        )
        self._area_per_inst__um2 = config.area_per_inst__um2
        self._leakage_per_inst__uW = config.leakage_per_inst__uW
        self.dtype = dtype
        self.T__K = T__K

        self.register_buffer(
            "ref_levels__uA",
            torch.tensor(config.ref_levels__uA, dtype=dtype),
            persistent=False,
        )

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
        """Physical magnitude resolution — the maximum ``adc_bits`` value."""
        return self.config.n_bits

    def unsigned_range(self, adc_bits: int) -> tuple[int, int]:
        """Unsigned magnitude code endpoints at ``adc_bits`` — ``(0, 2 ** adc_bits - 1)``."""
        if not (1 <= adc_bits <= self.max_bits):
            raise ValueError(f"adc_bits {adc_bits} outside [1, {self.max_bits}]")
        return 0, (1 << adc_bits) - 1

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
        return (torch.arange(n_col, device=device) * n_lane) // n_col

    # --- Conversion ---

    def convert(self, i_in__uA: Tensor, *, adc_operation_point: AdcOperationPoint) -> Tensor:
        """Quantize ``i_in`` to an unsigned magnitude code via ``n_bits``-step binary search.

        Runs an ``n_bits``-step binary search over the ``2 ** n_bits - 1`` nominal
        mid-point ``ref_levels__uA``: each step selects a reference (data-dependent
        on the bits resolved so far), forms the clean margin ``i_in - i_ref``,
        applies the ``margin_gain`` pre-gain and adds the input-referred offset
        **after** it, and resolves one bit. The step energies are aggregated and
        one dynamic-energy + one latency event (``sum(step_latency__ns)``) is
        emitted on completion. The sub-comparisons are a method-internal Python
        loop, never separate leaves.

        Args:
            i_in__uA: Unsigned magnitude current [uA]. Shape: ``[..., n_col]``.
            adc_operation_point: Runtime operating point (reserved; thresholds come
                from ``ref_levels__uA``).

        Returns:
            Unsigned magnitude code [long] in ``[0, 2 ** n_bits - 1]``. Shape:
            ``[..., n_col]``. The sign is combined by the caller.
        """
        del adc_operation_point  # reserved; thresholds come from ref_levels__uA

        n_bits = self.config.n_bits
        margin_gain = self.config.margin_gain
        mirror_ratio = self.config.input_mirror_ratio
        v_rail_sa = self.config.v_rail_sa__V
        t_eff = self.config.t_eff__ns
        e_fixed = self.config.e_fixed_per_op__fJ

        code = torch.zeros_like(i_in__uA, dtype=torch.long)
        e_dyn__fJ = torch.zeros_like(i_in__uA)

        i_in_pos__uA = i_in__uA.clamp_min(0.0)

        # Static input-referred SA offset gathered ONCE to the forward
        # logical-column axis and HELD CONSTANT across all binary-search steps:
        # the sum of the comparator and coupling offsets (0 when off). It is
        # added AFTER the margin_gain pre-gain, so the effective decision offset
        # is sigma / margin_gain (the triple-margin benefit).
        lane = self._col_to_lane(i_in__uA.shape[-1], i_in__uA.device)
        offset__uA = (self.comparator_offset__uA + self.coupling_offset__uA).index_select(-1, lane)

        for step in range(n_bits):
            i_ref__uA = self._select_ref(code, step)
            i_ref_pos__uA = i_ref__uA.clamp_min(0.0)

            clean_margin__uA = i_in__uA - i_ref__uA
            bit = (margin_gain * clean_margin__uA + offset__uA) > 0.0
            code = self._set_bit(code, step, bit)

            # --- Per-step dynamic energy ---
            # Control-based attribution: the ADC bills ONLY the "n"-path
            # regeneration its sized mirror controls (n = input_mirror_ratio),
            # drawn across the SA-leg overdrive over the effective conduction
            # time. The unity input / reference legs are owned/billed by the
            # upstream blocks that source them, so they are NOT re-billed. The
            # data-independent per-op constant e_fixed_per_op is broadcast onto
            # the per-column term each step.
            e_dyn_step__fJ = v_rail_sa * mirror_ratio * (i_in_pos__uA + i_ref_pos__uA) * t_eff + e_fixed
            e_dyn__fJ = e_dyn__fJ + e_dyn_step__fJ

        # --- One aggregated energy + one latency event for the whole convert ---

        latency__ns = torch.tensor(
            sum(self.config.step_latency__ns) * self._serial_op_count(i_in__uA),
            device=i_in__uA.device,
            dtype=e_dyn__fJ.dtype,
        )
        self._log_dynamic_energy(e_dyn__fJ)
        self._log_latency(latency__ns)

        return code

    # --- Binary-search helpers (method-internal; not leaves) ---

    def _select_ref(self, code: Tensor, step: int) -> Tensor:
        """Mid-point reference [uA] for ``step``, data-dependent on resolved bits.

        Binary search over the ``2 ** n_bits - 1`` nominal mid-point thresholds:
        the partial code from the bits resolved so far (MSB-first) indexes the
        reference for the current step. Step 0 selects the central threshold;
        each later step bisects the surviving sub-interval.

        Args:
            code: Partial magnitude code with the high ``step`` bits set.
                Shape: ``[..., n_col]``.
            step: Zero-based binary-search step (``0`` is the MSB).

        Returns:
            Selected reference current [uA] broadcast to ``code``'s shape.
        """
        n_bits = self.config.n_bits
        # High `step` bits of `code` are resolved (MSB-first); they live in
        # bit positions [n_bits - step, n_bits - 1]. The threshold index of the
        # interval boundary tested at this step is `prefix*2^(n_bits-step) + 2^(n_bits-step-1) - 1`.
        shift = n_bits - step
        prefix = code >> shift
        idx = (prefix << shift) + (1 << (shift - 1)) - 1
        return self.ref_levels__uA[idx]

    def _set_bit(self, code: Tensor, step: int, bit: Tensor) -> Tensor:
        """Write the ``step``-th magnitude bit (MSB-first) into ``code``."""
        bit_pos = self.config.n_bits - 1 - step
        return code | (bit.long() << bit_pos)

    def _serial_op_count(self, out: Tensor) -> int:
        """Serial op count for the latency multiplier.

        The per-column SA hardware is parallel (the ``inst_count`` trailing
        positions divide out); the serial latency tracks the broadcast-leading
        instances, matching every other emitting forward leaf.
        """
        return max(1, out.numel() // max(self.inst_count, 1))
