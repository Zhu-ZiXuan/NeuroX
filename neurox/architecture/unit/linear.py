"""LinearUnit operator ABC and the ideal linear reference unit.

See also:
    docs/internals/architecture/unit/linear.md
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor

from neurox.architecture.unit.base import UnitBase


class LinearUnit(UnitBase, ABC):
    """Interface for an integer ``torch.nn.functional.linear`` replacement."""

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
    def linear(self, input: Tensor, *, adc_mode: int, adc_bits: int) -> Tensor:
        """Execute one integer linear operator against the programmed state.

        Args:
            input: Integer activation tensor with trailing ``[K]``.
            adc_mode: Runtime ADC operating-point index.
            adc_bits: Runtime ADC resolution.

        Returns:
            Integer pre-requantize output tensor with trailing ``[N]``;
            leading dims mirror ``input``.
        """
        y = self._lower_matmul(input, adc_mode=adc_mode, adc_bits=adc_bits)
        int_bias = self.int_bias
        if int_bias is not None:
            # Shape: [..., N] + [N] -> [..., N]
            y = y + int_bias
        return y


# Deferred to avoid a circular import with CIM unit implementations.
from neurox.architecture.unit.cim.base import CimUnit, CimUnitConfig, CimUnitPolicy  # noqa: E402


class IdealLinearUnitConfig(CimUnitConfig):
    """Configuration for :class:`IdealLinearUnit`.

    Attributes:
        x_value_range: Inclusive integer activation range.
        w_value_range: Inclusive integer weight range.
    """

    x_value_range: tuple[int, int]
    w_value_range: tuple[int, int]


class IdealLinearUnitPolicy(CimUnitPolicy):
    """Empty policy — :class:`IdealLinearUnit` has no nonidealities to toggle."""


@CimUnit.register_key(IdealLinearUnitConfig)
class IdealLinearUnit(LinearUnit, CimUnit[IdealLinearUnitConfig, IdealLinearUnitPolicy]):
    """Exact integer linear unit without output quantization.

    Args:
        config: Concrete configuration dataclass.
        policy: Nonideality policy.
        w_logical_shape: Logical weight shape ``(*prefix, N, K)`` bound to ``program(...)``.
        dtype: Requested tensor dtype; it does not affect exact integer execution.
        T__K: Operating temperature.
        ideal_xbar: Accepted without changing this already ideal unit.
    """

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
        self._area_per_inst__um2 = config.area_per_inst__um2
        self._leakage_per_inst__uW = config.leakage_per_inst__uW

        # fp32-exact fast-path eligibility: CUDA has no integer-matmul
        # kernel. With every per-element product and every partial sum
        # bounded by ``K * max|x| * max|w| < 2^24``, IEEE fp32 matmul
        # accumulation (the framework default; TF32 disabled) reproduces
        # the int64 contraction bit-exactly for range-conformant operands.
        x_lo, x_hi = config.x_value_range
        w_lo, w_hi = config.w_value_range
        max_dot_abs = self._w_logical_shape[-1] * max(abs(x_lo), abs(x_hi)) * max(abs(w_lo), abs(w_hi))
        self._fp32_exact: bool = max_dot_abs < 2**24

        # A scalar zero provides a valid pre-programming weight.
        self.register_buffer("nominal_weight", torch.zeros((), dtype=torch.int32), persistent=False)
        self.register_buffer("weight", self.nominal_weight.clone(), persistent=False)
        self._init_int_bias_slot()

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
        return 0

    def adc_rescale_factor(self, *, adc_mode: int, adc_bits: int) -> float:
        del adc_mode, adc_bits
        return 1.0

    def program(self, weight: Tensor, bias: Tensor | None = None) -> None:
        if tuple(weight.shape) != self._w_logical_shape:
            raise ValueError(f"program() expects weight.shape {self._w_logical_shape}; got {tuple(weight.shape)}")
        self.weight = weight
        self._program_int_bias(bias, channels=self._w_logical_shape[-2])

    @torch.no_grad()
    def _matmul(self, input: Tensor, *, adc_mode: int, adc_bits: int) -> Tensor:
        del adc_mode, adc_bits
        weight = self.weight
        if self._fp32_exact:
            # Shape: [..., M, K] @ [*prefix, K, N] -> [..., M, N]
            out = torch.matmul(input.to(torch.float32), weight.to(torch.float32).transpose(-2, -1))
            return out.to(torch.int64)
        # Shape: [..., M, K] @ [*prefix, K, N] -> [..., M, N]
        return torch.matmul(input.to(torch.int64), weight.to(torch.int64).transpose(-2, -1))
