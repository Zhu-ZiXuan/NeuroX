"""Ideal XbarMacro: lossless integer-matmul reference (no xbar tile).

See also:
    docs/dev/modules/macro/xbar/ideal.md
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .base import XbarMacro, XbarMacroConfig


@dataclass(frozen=True)
class IdealXbarMacroConfig(XbarMacroConfig):
    """Configuration for :class:`IdealXbarMacro`.

    Attributes:
        x_value_range: Inclusive integer activation range.
        w_value_range: Inclusive integer weight range.
    """

    x_value_range: tuple[int, int]
    w_value_range: tuple[int, int]


@XbarMacro.register_key(IdealXbarMacroConfig)
class IdealXbarMacro(XbarMacro):
    """Lossless reference replacement for any :class:`XbarMacro`.

    No xbar tile, no slicing, no transcoding — just stores the integer
    weight and computes ``torch.matmul`` against it. Useful as the
    noise-free ground truth when isolating QAT issues from analog / IO
    modelling, and as the contract conformance baseline for the rest of
    the family.

    Args:
        cfg: :class:`IdealXbarMacroConfig`.
        name: Hierarchical instance name used by the profiler.
        w_logical_shape: Logical weight shape ``(*prefix, N, K)`` bound to ``program(...)``.
        dtype: Tensor dtype for internal buffers. Accepted for API
            uniformity; ignored by the integer matmul.
        T__K: Operating temperature [K]. Accepted for API uniformity; ignored.
        ideal_xbar: Accepted for API uniformity; ignored (no xbar to swap).
    """

    cfg: IdealXbarMacroConfig
    nominal_weight: Tensor
    weight: Tensor

    def __init__(
        self,
        *,
        cfg: IdealXbarMacroConfig,
        name: str,
        w_logical_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
        ideal_xbar: bool,
    ) -> None:
        super().__init__(
            cfg=cfg,
            name=name,
            w_logical_shape=w_logical_shape,
            dtype=dtype,
            T__K=T__K,
            ideal_xbar=ideal_xbar,
        )
        self.cfg = cfg
        self._inst_shape = self._w_logical_shape[:-2]

        # 0-d nominal weight: broadcasts to a zero-weight matmul before any
        # ``program(...)`` call.
        self.register_buffer("nominal_weight", torch.zeros((), dtype=torch.int32), persistent=False)
        self.register_buffer("weight", self.nominal_weight.clone(), persistent=False)

        self._log_static()

    # --- value-range / rescale ---

    @property
    def w_value_range(self) -> tuple[int, int]:
        """Inclusive integer weight range accepted by the macro."""
        return self.cfg.w_value_range

    @property
    def x_value_range(self) -> tuple[int, int]:
        """Inclusive integer activation range accepted by the macro."""
        return self.cfg.x_value_range

    @property
    def output_rescale_factor(self) -> float:
        """Ratio of the ideal partial-product max to the actual tile output max."""
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
    @torch.compile(dynamic=True)
    def matmul(self, input: Tensor) -> Tensor:
        """Execute one integer matrix multiply against the programmed weight state.

        Matches ``torch.matmul`` semantics (pure matmul, no bias). Bias add
        and requantize live in the operator layer.

        Args:
            input: Integer activation tensor. Shape: ``[..., M, K]``.

        Returns:
            Integer pre-requantize output tensor. Shape: ``[..., M, N]``.
        """
        weight = self.weight
        return torch.matmul(input.to(torch.int64), weight.to(torch.int64).transpose(-2, -1))
