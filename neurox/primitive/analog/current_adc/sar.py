"""Triple-margin current-mode successive-approximation ADC.

See also:
    docs/reference/primitive/analog/current_adc/sar.md
"""

from __future__ import annotations

import torch
from torch import Tensor

from neurox.primitive.nonideality import apply_gaussian

from .base import Iadc, IadcConfig, IadcPolicy


class SarIadcConfig(IadcConfig):
    """Physical knobs for the triple-margin current-mode SAR ADC.

    Attributes:
        bits: Physical (maximum) magnitude resolution [bits]; a ``convert``
            call requests any resolution in ``[1, bits]``.
        margin_gain: Deterministic triple-margin pre-gain applied to the clean
            ``i_in - i_ref`` before the latch. The input-referred offset is added
            after this gain, so the effective offset is ``offset_sigma / margin_gain``.
        e_fixed_per_op__fJ: Data-independent per-step energy constant
            folding sampling, latch, coupling, and reference-selector switching;
            billed once per binary-search step.
        v_rail__V: Supply rail the input and selected reference conduct across
            during each comparison step; drives the per-step conduction
            energy term.
        t_conduct_per_step__ns: Per-step conduction window, one entry per
            binary-search step (length ``>= bits``; a longer list is tolerated
            and only the first ``bits`` entries are drawn). All-zero reduces to
            the pure fixed-energy model.
        step_latency__ns: Per-step decision latency, one entry per
            binary-search step (length ``>= bits``; a longer list is
            tolerated and only the first requested-resolution entries are
            summed). The conversion latency is the sum of the first ``bits``
            (the requested resolution) entries.
        comparator_offset_sigma__uA: Input-referred SA offset sigma — a
            current-domain margin perturbation added to the clean ``i_in - i_ref``
            after the ``margin_gain`` pre-gain (effective ``sigma / margin_gain``).
        coupling_mismatch_sigma__uA: Residual coupling-driven offset sigma — a
            current-domain margin perturbation added after the ``margin_gain``
            pre-gain (effective ``sigma / margin_gain``).
        area_per_inst__um2: SA silicon area per fabricated shared sense-lane
            instance.
        leakage_per_inst__uW: SA static leakage per fabricated shared sense-lane
            instance.
    """

    bits: int
    margin_gain: float

    e_fixed_per_op__fJ: float
    v_rail__V: float
    t_conduct_per_step__ns: tuple[float, ...]

    step_latency__ns: tuple[float, ...]

    comparator_offset_sigma__uA: float
    coupling_mismatch_sigma__uA: float

    def validate(self) -> None:
        super().validate()

        # --- Quantizer ---

        self._require_pos(self.bits, "bits")
        self._require_pos(self.margin_gain, "margin_gain")

        # --- Energy and timing ---

        self._require_non_neg(self.e_fixed_per_op__fJ, "e_fixed_per_op__fJ")
        self._require_non_neg(self.v_rail__V, "v_rail__V")
        if len(self.t_conduct_per_step__ns) < self.bits:
            raise ValueError(
                f"require: len(t_conduct_per_step__ns) ({len(self.t_conduct_per_step__ns)}) >= bits ({self.bits})"
            )
        for t in self.t_conduct_per_step__ns:
            self._require_non_neg(t, "t_conduct_per_step__ns")

        if len(self.step_latency__ns) < self.bits:
            raise ValueError(f"require: len(step_latency__ns) ({len(self.step_latency__ns)}) >= bits ({self.bits})")
        for latency in self.step_latency__ns:
            self._require_non_neg(latency, "step_latency__ns")

        # --- Nonidealities ---

        self._require_non_neg(self.comparator_offset_sigma__uA, "comparator_offset_sigma__uA")
        self._require_non_neg(self.coupling_mismatch_sigma__uA, "coupling_mismatch_sigma__uA")


class SarIadcPolicy(IadcPolicy):
    """Per-source toggles selecting which SarIadc nonidealities are active.

    Attributes:
        comparator_offset: Inject ``comparator_offset_sigma__uA`` as a
            current-domain margin perturbation added **after** ``margin_gain``
            (effective offset ``sigma / margin_gain`` — the triple-margin benefit).
        coupling_mismatch: Inject ``coupling_mismatch_sigma__uA`` as a residual
            current-domain margin perturbation added after ``margin_gain``
            (effective ``sigma / margin_gain``).
    """

    comparator_offset: bool
    coupling_mismatch: bool


@Iadc.register_neurox_module(
    config_type=SarIadcConfig,
    policy_type=SarIadcPolicy,
)
class SarIadc(Iadc[SarIadcConfig, SarIadcPolicy]):
    """Triple-margin current ADC using a binary search over injected references.

    Args:
        config: Concrete configuration dataclass.
        policy: Per-source nonideality flags.
        inst_shape: Fabricated shared-sense-lane shape.
        dtype: Tensor dtype for internal buffers.
        T__K: Operating temperature.
        enable_latency_record: Whether conversions emit latency events.
    """

    # --- Immutable PPA buffers ---

    _step_latency__ns: Tensor

    # --- Fabrication source buffers ---

    _nominal_comparator_offset__uA: Tensor
    _nominal_coupling_offset__uA: Tensor

    def __init__(
        self,
        *,
        config: SarIadcConfig,
        policy: SarIadcPolicy,
        inst_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
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
        self._area_per_inst__um2 = config.area_per_inst__um2
        self._leakage_per_inst__uW = config.leakage_per_inst__uW
        self.register_buffer(
            "_step_latency__ns",
            torch.tensor(config.step_latency__ns, dtype=dtype),
            persistent=False,
        )
        self._register_fabrication_buffers(dtype=dtype)

    def _register_fabrication_buffers(self, *, dtype: torch.dtype) -> None:
        """Register immutable tensors used as fabrication sources."""
        self.register_buffer("_nominal_comparator_offset__uA", torch.zeros((), dtype=dtype), persistent=False)
        self.register_buffer("_nominal_coupling_offset__uA", torch.zeros((), dtype=dtype), persistent=False)

    @property
    def max_bits(self) -> int:
        """Physical magnitude resolution — the maximum ``bits`` a call may request."""
        return self.config.bits

    def unsigned_range(self, bits: int) -> tuple[int, int]:
        """Unsigned magnitude code endpoints at ``bits`` — ``(0, 2 ** bits - 1)``."""
        if not (1 <= bits <= self.max_bits):
            raise ValueError(f"bits {bits} outside [1, {self.max_bits}]")
        return 0, (1 << bits) - 1

    def _sample_fabricate_mismatch(self) -> None:
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
        """Map logical columns to contiguous shared-sense lanes.

        Args:
            n_col: Number of logical columns.
            device: Device for the returned index tensor.

        Returns:
            Lane indices given by ``(column * n_lane) // n_col``.
        """
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
        """Quantize ``i_in`` with a ``bits``-step binary search.

        Args:
            i_in__uA: Unsigned magnitude current. Shape: ``[..., n_col]``.
            i_refs__uA: Per-instance reference ladder, ``[*R, n_ref]`` with
                ``n_ref = 2 ** bits - 1`` taps ascending on the last axis and
                ``[*R]`` broadcasting right-aligned against ``i_in__uA``.
            bits: Conversion resolution [bits], in ``[1, config.bits]``.

        Returns:
            Unsigned magnitude code [long] in ``[0, 2 ** bits - 1]``,
            shape ``[..., n_col]``.
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
        e_dyn__fJ = torch.zeros_like(i_in__uA) if self._is_dynamic_energy_profile_active() else None

        # The fabricated lane offset remains fixed throughout the binary search.
        lane = self._col_to_lane(i_in__uA.shape[-1], i_in__uA.device)
        # Shape: [*prefix, n_lane] -> [*prefix, n_col]
        offset__uA = (self._comparator_offset__uA + self._coupling_offset__uA).index_select(-1, lane)

        for step in range(bits):
            i_ref__uA = self._select_ref(ref_b, code, step, bits)

            clean_margin__uA = i_in__uA - i_ref__uA
            bit = (margin_gain * clean_margin__uA + offset__uA) > 0.0
            code = self._set_bit(code, step, bit, bits)

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
        if self.enable_latency_record:
            latency__ns = self._step_latency__ns[:bits].sum() * self._count_serial_rounds(i_in__uA.numel())
            self._record_latency(latency__ns)

        return code

    def _compute_input_dynamic_energy__fJ(self, i_in__uA: Tensor, i_ref__uA: Tensor) -> Tensor:
        """Return additional per-step conduction energy.

        Args:
            i_in__uA: Input current.
            i_ref__uA: Selected reference current.

        Returns:
            Additional energy [fJ], shaped like ``i_in__uA``.
        """
        return torch.zeros_like(i_in__uA)

    def _select_ref(self, ref_b: Tensor, code: Tensor, step: int, bits: int) -> Tensor:
        """Mid-point reference [uA] for ``step``, data-dependent on resolved bits.

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
        # Threshold index: prefix * 2^(bits-step) + 2^(bits-step-1) - 1.
        shift = bits - step
        prefix = code >> shift
        idx = (prefix << shift) + (1 << (shift - 1)) - 1
        # Shape: [..., n_col, n_ref] -> [..., n_col]
        return torch.gather(ref_b, -1, idx.unsqueeze(-1)).squeeze(-1)

    def _set_bit(self, code: Tensor, step: int, bit: Tensor, bits: int) -> Tensor:
        """Write the ``step``-th magnitude bit (MSB-first) into ``code``."""
        bit_pos = bits - 1 - step
        return code | (bit.long() << bit_pos)
