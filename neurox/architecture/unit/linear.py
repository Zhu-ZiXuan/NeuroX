"""LinearUnit operator ABC and the ideal linear reference unit.

See also:
    docs/internals/architecture/unit/linear.md
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from torch import Tensor

from neurox.architecture.unit.base import UnitBase
from neurox.primitive.analog.adc_common import AdcOperationPoint


class LinearUnit(UnitBase, ABC):
    """Operator ABC for an exact-integer drop-in replacement of ``F.linear``.

    The integer bias belonging to ``F.linear``'s algorithmic scope is
    programmed alongside the weight and added in the int64 accumulation
    domain. :meth:`linear` runs the inherited lowering template; every
    leading input dim is a broadcast batch dim that rides through
    untouched.
    """

    @abstractmethod
    def program(self, weight: Tensor, bias: Tensor | None = None) -> None:
        """Write the unit's static weight state and optional integer bias.

        Args:
            weight: Integer weight tensor of shape ``(*prefix, N, K)``.
            bias: Optional integer bias tensor of shape ``(N,)``, added in
                the int64 accumulation domain by :meth:`linear`; ``None``
                clears any programmed bias.
        """
        raise NotImplementedError

    def _activation_to_planes(self, input: Tensor) -> Tensor:
        # Shape: [..., K] -> [..., 1, K]
        return input.unsqueeze(-2)

    def _undo_aggregation(self, output: Tensor) -> Tensor:
        # Shape: [..., 1, N] -> [..., N]
        return output.squeeze(-2)

    @torch.no_grad()
    def linear(self, input: Tensor, *, adc_operation_point: AdcOperationPoint) -> Tensor:
        """Execute one integer linear operator against the programmed state.

        Matches ``torch.nn.functional.linear`` shape semantics: the
        trailing axis is the contraction axis and every leading dim is a
        broadcast batch dim passed through untouched. The programmed
        integer bias, if any, is added in the int64 accumulation domain.

        Args:
            input: Integer activation tensor with trailing ``[K]``.
            adc_operation_point: Runtime ADC operating point.

        Returns:
            Integer pre-requantize output tensor with trailing ``[N]``;
            leading dims mirror ``input``.
        """
        y = self._lower_matmul(input, adc_operation_point=adc_operation_point)
        int_bias = self.int_bias
        if int_bias is not None:
            # Shape: [..., N] + [N] -> [..., N]  (broadcast add)
            y = y + int_bias
        return y


# Imported after the operator ABC definition: loading ``...cim.base``
# executes the ``cim`` package __init__, whose leaves import ``LinearUnit``
# from this module.
from neurox.architecture.unit.cim.base import CimUnit, CimUnitConfig, CimUnitPolicy  # noqa: E402


@dataclass(frozen=True, kw_only=True)
class IdealLinearUnitConfig(CimUnitConfig):
    """Configuration for :class:`IdealLinearUnit`.

    Bring-up / reference use only — see :class:`IdealLinearUnit`. A no-PPA
    reference: the inherited ``area_per_inst__um2`` / ``leakage_per_inst__uW``
    are supplied as ``0.0`` at construction.

    Attributes:
        x_value_range: Inclusive integer activation range.
        w_value_range: Inclusive integer weight range.
    """

    x_value_range: tuple[int, int]
    w_value_range: tuple[int, int]


@dataclass(frozen=True)
class IdealLinearUnitPolicy(CimUnitPolicy):
    """Empty policy — :class:`IdealLinearUnit` has no nonidealities to toggle."""


@CimUnit.register_key(IdealLinearUnitConfig)
class IdealLinearUnit(LinearUnit, CimUnit):
    """Degenerate ``CimUnit``: stores the integer weight and runs the exact integer matmul against it.

    No xbar tile, no slicing, no transcoding. ``dtype``, ``T__K``, and
    ``ideal_xbar`` are accepted for API uniformity and ignored.

    Reference, not hardware: a value-domain / lossless upper-bound baseline
    with no tile, ADC, fabrication, or PPA. Use it for flow bring-up and to
    isolate QAT issues from analog modelling — never as a stand-in for a
    physical macro in a production accuracy or PPA study.

    Args:
        config: Concrete configuration dataclass.
        policy: Empty :class:`IdealLinearUnitPolicy` marker.
        w_logical_shape: Logical weight shape ``(*prefix, N, K)`` bound to ``program(...)``.
        dtype: Tensor dtype for internal buffers.
        T__K: Operating temperature.
        ideal_xbar: Accepted for API uniformity and ignored (no xbar tile to swap).
    """

    config: IdealLinearUnitConfig
    nominal_weight: Tensor
    weight: Tensor

    def __init__(
        self,
        *,
        config: IdealLinearUnitConfig,
        policy: IdealLinearUnitPolicy,
        w_logical_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
        ideal_xbar: bool,
    ) -> None:
        super().__init__(
            config=config,
            policy=policy,
            w_logical_shape=w_logical_shape,
            dtype=dtype,
            T__K=T__K,
            ideal_xbar=ideal_xbar,
        )
        self.config = config
        self._area_per_inst__um2 = config.area_per_inst__um2
        self._leakage_per_inst__uW = config.leakage_per_inst__uW
        self._inst_shape = self._w_logical_shape[:-2]

        # fp32-exact fast-path eligibility: CUDA has no integer-matmul
        # kernel. With every per-element product and every partial sum
        # bounded by ``K * max|x| * max|w| < 2^24``, IEEE fp32 matmul
        # accumulation (the framework default; TF32 disabled) reproduces
        # the int64 contraction bit-exactly for range-conformant operands.
        x_lo, x_hi = config.x_value_range
        w_lo, w_hi = config.w_value_range
        max_dot_abs = self._w_logical_shape[-1] * max(abs(x_lo), abs(x_hi)) * max(abs(w_lo), abs(w_hi))
        self._fp32_exact: bool = max_dot_abs < 2**24

        # 0-d nominal weight: broadcasts to a zero-weight matmul before any
        # ``program(...)`` call.
        self.register_buffer("nominal_weight", torch.zeros((), dtype=torch.int32), persistent=False)
        self.register_buffer("weight", self.nominal_weight.clone(), persistent=False)
        self._init_int_bias_slot()

    # --- value-range / ADC surface ---

    @property
    def w_value_range(self) -> tuple[int, int]:
        return self.config.w_value_range

    @property
    def x_value_range(self) -> tuple[int, int]:
        return self.config.x_value_range

    @property
    def adc_mode_num(self) -> int:
        return 1

    @property
    def adc_max_bits(self) -> int:
        # ``0`` is the sentinel meaning no output quantization is applied.
        return 0

    def adc_rescale_factor(self, adc_operation_point: AdcOperationPoint) -> float:
        """Rescale factor for ``adc_operation_point``; always ``1.0`` (no ADC)."""
        del adc_operation_point  # accepted for API uniformity
        return 1.0

    # --- lifecycle ---

    def program(self, weight: Tensor, bias: Tensor | None = None) -> None:
        """Write the unit's static weight state and optional integer bias.

        Args:
            weight: Integer weight tensor whose shape matches the unit's
                ``w_logical_shape``.
            bias: Optional integer bias tensor of shape ``(N,)``; ``None``
                clears any programmed bias.
        """
        if tuple(weight.shape) != self._w_logical_shape:
            raise ValueError(f"program() expects weight.shape {self._w_logical_shape}; got {tuple(weight.shape)}")
        self.weight = weight
        self._program_int_bias(bias, channels=self._w_logical_shape[-2])

    @torch.no_grad()
    def _matmul(self, input: Tensor, *, adc_operation_point: AdcOperationPoint) -> Tensor:
        del adc_operation_point  # accepted for API uniformity
        weight = self.weight
        if self._fp32_exact:
            # Bound checked in ``__init__`` against the config value
            # ranges; the cast back to int64 is lossless.
            # Shape: [..., M, K] @ [*prefix, K, N] -> [..., M, N]
            out = torch.matmul(input.to(torch.float32), weight.to(torch.float32).transpose(-2, -1))
            return out.to(torch.int64)
        # Shape: [..., M, K] @ [*prefix, K, N] -> [..., M, N]
        return torch.matmul(input.to(torch.int64), weight.to(torch.int64).transpose(-2, -1))
