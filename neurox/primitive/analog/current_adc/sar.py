"""Current-mode successive-approximation ADC.

See Also:
    docs/reference/primitive/analog/current_adc/sar.md
"""

from __future__ import annotations

import torch
from torch import Tensor

from neurox.primitive.nonideality import apply_gaussian

from .base import Iadc, IadcConfig, IadcPolicy


class SarIadcConfig(IadcConfig):
    # === Nonidealities ===

    comparator_offset_sigma__uA: float
    """Static input-referred comparator-offset σ. Positive offset raises the
    reference-side threshold."""

    # === Timing ===

    latency_per_bit__ns: float
    """Scheduled duration of one bit-decision phase."""

    def validate(self) -> None:
        super().validate()

        # --- Nonidealities ---

        self._require_non_neg(self.comparator_offset_sigma__uA, "comparator_offset_sigma__uA")

        # --- Timing ---

        self._require_non_neg(self.latency_per_bit__ns, "latency_per_bit__ns")


class SarIadcPolicy(IadcPolicy):
    comparator_offset: bool
    """Inject `comparator_offset_sigma__uA` at fabricate time."""


_Config = SarIadcConfig
_Policy = SarIadcPolicy


@Iadc.register_neurox_impl(config_type=_Config, policy_type=_Policy)
class SarIadc(Iadc):
    """Current ADC using a binary search over injected references.

    The search tree is wired for `config.bits`, so this converter reads exactly
    `2 ** bits - 1` ascending taps off the injected ladder's last axis. That
    count is this circuit's own property: a shorter ladder is not rejected up
    front, it fails in the tap gather.

    Args:
        inst_shape: Fabricated ADC-instance shape.
    """

    config: _Config
    policy: _Policy

    # === Nominal buffers ===

    _nominal_comparator_offset__uA: Tensor  # Shape: []

    # === Fabricated state ===

    _comparator_offset__uA: Tensor  # Shape: [*inst_shape]

    def __init__(
        self,
        *,
        config: _Config,
        policy: _Policy,
        inst_shape: tuple[int, ...],
        dtype: torch.dtype,
    ) -> None:
        super().__init__(
            config=config,
            policy=policy,
            inst_shape=inst_shape,
            dtype=dtype,
        )
        self._register_fabrication_buffers(dtype=dtype)

    def latency__ns(self, *, active_bits: int) -> float:
        """One conversion — the summed decision latencies of the executed steps.

        The binary-search steps run sequentially inside the one sense lane, and
        a call executes the first `active_bits` of them.
        """
        self._check_active_bits(active_bits)
        return active_bits * self.config.latency_per_bit__ns

    def _register_fabrication_buffers(self, *, dtype: torch.dtype) -> None:
        self._register_nonpersistent_buffer("_nominal_comparator_offset__uA", torch.zeros((), dtype=dtype))

    def _sample_fabrication_variation(self) -> None:
        self._comparator_offset__uA = apply_gaussian(
            self._nominal_comparator_offset__uA.clone().expand(self.inst_shape),
            self.config.comparator_offset_sigma__uA,
            enabled=self.policy.comparator_offset,
        )

    @torch.compile(dynamic=False, fullgraph=True)
    def _convert_impl(
        self,
        i_in__uA: Tensor,
        i_refs__uA: Tensor,
        *,
        active_bits: int,
        record_energy: bool,
    ) -> tuple[Tensor, Tensor | None]:
        """Quantize `i_in` with a truncated binary search.

        The search tree is always the configured `bits`-level one over the
        full ladder. A conversion stops after its first `active_bits` levels
        while retaining each decision in its full-width bit position. The
        result is compacted only at the return boundary.

        Args:
            i_in__uA: Unsigned magnitude current.
            i_refs__uA: Per-instance reference ladder with the taps on the last
                axis and the leading dims broadcasting right-aligned against
                `i_in__uA`. Reference selection uses full-width tap numbering, so
                this converter takes `2 ** bits - 1` ascending taps.
                Shape: `[..., tap]`.
            active_bits: Active conversion resolution in `[1, bits]`.
            record_energy: Whether to compute dynamic energy.

        Returns:
            Unsigned magnitude codes [long] in `[0, 2 ** active_bits - 1]`
            and optional per-output dynamic energy [fJ].
        """
        # --- 1: initialize full-width conversion state ---

        code = torch.zeros_like(i_in__uA, dtype=torch.int)
        e_dyn__fJ: Tensor | None = None

        # --- 2: resolve the active bits from most to least significant ---

        for bit_position in range(self.bits - 1, self.bits - active_bits - 1, -1):
            trial_code = code | (1 << bit_position)
            i_ref__uA = self._select_reference(i_refs__uA, trial_code)
            accept_trial = i_in__uA - i_ref__uA >= self._comparator_offset__uA
            code = torch.where(accept_trial, trial_code, code)

            if record_energy:
                e_bit__fJ = self._compute_bit_dynamic_energy__fJ(
                    i_in__uA,
                    i_refs__uA,
                    trial_code,
                    bit_position=bit_position,
                )
                if e_bit__fJ is not None:
                    e_dyn__fJ = e_bit__fJ if e_dyn__fJ is None else e_dyn__fJ + e_bit__fJ

        # --- 3: compact the retained decisions ---

        dropped_bits = self.bits - active_bits
        return code >> dropped_bits, e_dyn__fJ

    def _select_reference(self, i_refs__uA: Tensor, trial_code: Tensor) -> Tensor:
        """Select the decision reference for `trial_code`."""
        tap_num = i_refs__uA.shape[-1]
        # Gather needs every conversion position present in the ladder's leading axes.
        # Shape: [..., tap] -> [..., tap]
        i_ref_lut__uA = torch.broadcast_to(i_refs__uA, (*trial_code.shape, tap_num))
        # Shape: [...] -> [..., tap=1]
        tap_index = (trial_code - 1).unsqueeze(-1)
        # Shape: [..., tap] -> [...]
        return torch.gather(i_ref_lut__uA, -1, tap_index).squeeze(-1)

    def _compute_bit_dynamic_energy__fJ(
        self,
        i_in__uA: Tensor,
        i_refs__uA: Tensor,
        trial_code: Tensor,
        *,
        bit_position: int,
    ) -> Tensor | None:
        del i_in__uA, i_refs__uA, trial_code, bit_position
        return None
