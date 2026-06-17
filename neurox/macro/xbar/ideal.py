"""Ideal XbarMacro: lossless integer-matmul reference (no xbar tile).

See also:
    docs/reference/macro/xbar/ideal.md
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from neurox.analog.adc import AdcOperationPoint

from .base import XbarMacro, XbarMacroConfig, XbarMacroPolicy


@dataclass(frozen=True)
class IdealXbarMacroConfig(XbarMacroConfig):
    """Configuration for :class:`IdealXbarMacro`.

    Bring-up / reference use only — see :class:`IdealXbarMacro`.

    Attributes:
        x_value_range: Inclusive integer activation range.
        w_value_range: Inclusive integer weight range.
    """

    x_value_range: tuple[int, int]
    w_value_range: tuple[int, int]


@dataclass(frozen=True)
class IdealXbarMacroPolicy(XbarMacroPolicy):
    """Empty policy — :class:`IdealXbarMacro` has no nonidealities to toggle."""


@XbarMacro.register_key(IdealXbarMacroConfig)
class IdealXbarMacro(XbarMacro):
    """Degenerate ``XbarMacro``: stores the integer weight and runs ``torch.matmul`` against it.

    No xbar tile, no slicing, no transcoding. ``dtype``, ``T__K``, and
    ``ideal_xbar`` are accepted for API uniformity and ignored.

    Reference, not hardware: a value-domain / lossless upper-bound baseline
    with no tile, ADC, fabrication, or PPA. Use it for flow bring-up and to
    isolate QAT issues from analog modelling — never as a stand-in for a
    physical macro in a production accuracy or PPA study.

    Args:
        config: Concrete configuration dataclass.
        policy: Empty :class:`IdealXbarMacroPolicy` marker.
        name: Hierarchical instance name used by the profiler.
        w_logical_shape: Logical weight shape ``(*prefix, N, K)`` bound to ``program(...)``.
        dtype: Tensor dtype for internal buffers.
        T__K: Operating temperature [K].
        ideal_xbar: When True, the macro replaces its physical xbar with the lossless ideal twin returned by xbar.to_ideal().
    """

    config: IdealXbarMacroConfig
    nominal_weight: Tensor
    weight: Tensor

    def __init__(
        self,
        *,
        config: IdealXbarMacroConfig,
        policy: IdealXbarMacroPolicy,
        name: str,
        w_logical_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
        ideal_xbar: bool,
    ) -> None:
        super().__init__(
            config=config,
            policy=policy,
            name=name,
            w_logical_shape=w_logical_shape,
            dtype=dtype,
            T__K=T__K,
            ideal_xbar=ideal_xbar,
        )
        self.config = config
        self._inst_shape = self._w_logical_shape[:-2]

        # 0-d nominal weight: broadcasts to a zero-weight matmul before any
        # ``program(...)`` call.
        self.register_buffer("nominal_weight", torch.zeros((), dtype=torch.int32), persistent=False)
        self.register_buffer("weight", self.nominal_weight.clone(), persistent=False)

    # --- value-range / ADC surface ---

    @property
    def w_value_range(self) -> tuple[int, int]:
        """Inclusive integer weight range accepted by the macro."""
        return self.config.w_value_range

    @property
    def x_value_range(self) -> tuple[int, int]:
        """Inclusive integer activation range accepted by the macro."""
        return self.config.x_value_range

    @property
    def adc_mode_num(self) -> int:
        """Number of supported ADC operating points; valid ``adc_mode`` values are ``[0, mode_num)``."""
        return 1

    @property
    def adc_max_bits(self) -> int:
        """Maximum supported ``adc_bits`` value."""
        # ``0`` is the sentinel meaning no output quantization is applied.
        return 0

    def adc_rescale_factor(self, adc_operation_point: AdcOperationPoint) -> float:
        """Rescale factor for ``adc_operation_point``; raises ``KeyError`` if uncalibrated."""
        del adc_operation_point  # accepted for API uniformity
        return 1.0

    # --- lifecycle ---

    def program(self, weight: Tensor) -> None:
        """Write the macro's static weight state from one logical weight tensor.

        Args:
            weight: Integer weight tensor whose shape matches
                ``self._w_logical_shape``.
        """
        if tuple(weight.shape) != self._w_logical_shape:
            raise ValueError(f"program() expects weight.shape {self._w_logical_shape}; got {tuple(weight.shape)}")
        self.weight = weight

    @torch.no_grad()
    def matmul(self, input: Tensor, *, adc_operation_point: AdcOperationPoint) -> Tensor:
        """Execute one integer matrix multiply against the programmed weight state.

        Matches ``torch.matmul`` semantics (pure matmul, no bias). Bias add
        and requantize live in the operator layer.

        Args:
            input: Integer activation tensor. Shape: ``[..., M, K]``.
            adc_operation_point: Runtime ADC operating point.

        Returns:
            Integer pre-requantize output tensor. Shape: ``[..., M, N]``.
        """
        del adc_operation_point  # accepted for API uniformity
        weight = self.weight
        return torch.matmul(input.to(torch.int64), weight.to(torch.int64).transpose(-2, -1))
