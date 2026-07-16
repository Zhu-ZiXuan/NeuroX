"""Ideal CimUnit: lossless integer-matmul reference (no xbar tile).

See also:
    docs/reference/architecture/unit/cim/ideal.md
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from neurox.primitive.analog.adc_common import AdcOperationPoint

from .base import CimUnit, CimUnitConfig, CimUnitPolicy


@dataclass(frozen=True)
class IdealCimUnitConfig(CimUnitConfig):
    """Configuration for :class:`IdealCimUnit`.

    Bring-up / reference use only — see :class:`IdealCimUnit`. A no-PPA
    reference: the inherited ``area_per_inst__um2`` / ``leakage_per_inst__uW``
    are supplied as ``0.0`` at construction.

    Attributes:
        x_value_range: Inclusive integer activation range.
        w_value_range: Inclusive integer weight range.
    """

    x_value_range: tuple[int, int]
    w_value_range: tuple[int, int]


@dataclass(frozen=True)
class IdealCimUnitPolicy(CimUnitPolicy):
    """Empty policy — :class:`IdealCimUnit` has no nonidealities to toggle."""


@CimUnit.register_key(IdealCimUnitConfig)
class IdealCimUnit(CimUnit):
    """Degenerate ``CimUnit``: stores the integer weight and runs ``torch.matmul`` against it.

    No xbar tile, no slicing, no transcoding. ``dtype``, ``T__K``, and
    ``ideal_xbar`` are accepted for API uniformity and ignored.

    Reference, not hardware: a value-domain / lossless upper-bound baseline
    with no tile, ADC, fabrication, or PPA. Use it for flow bring-up and to
    isolate QAT issues from analog modelling — never as a stand-in for a
    physical macro in a production accuracy or PPA study.

    Args:
        config: Concrete configuration dataclass.
        policy: Empty :class:`IdealCimUnitPolicy` marker.
        w_logical_shape: Logical weight shape ``(*prefix, N, K)`` bound to ``program(...)``.
        dtype: Tensor dtype for internal buffers.
        T__K: Operating temperature.
        ideal_xbar: Accepted for API uniformity and ignored (no xbar tile to swap).
    """

    config: IdealCimUnitConfig
    nominal_weight: Tensor
    weight: Tensor

    def __init__(
        self,
        *,
        config: IdealCimUnitConfig,
        policy: IdealCimUnitPolicy,
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

        # 0-d nominal weight: broadcasts to a zero-weight matmul before any
        # ``program(...)`` call.
        self.register_buffer("nominal_weight", torch.zeros((), dtype=torch.int32), persistent=False)
        self.register_buffer("weight", self.nominal_weight.clone(), persistent=False)

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

    def program(self, weight: Tensor) -> None:
        if tuple(weight.shape) != self._w_logical_shape:
            raise ValueError(f"program() expects weight.shape {self._w_logical_shape}; got {tuple(weight.shape)}")
        self.weight = weight

    @torch.no_grad()
    def matmul(self, input: Tensor, *, adc_operation_point: AdcOperationPoint) -> Tensor:
        del adc_operation_point  # accepted for API uniformity
        weight = self.weight
        return torch.matmul(input.to(torch.int64), weight.to(torch.int64).transpose(-2, -1))
