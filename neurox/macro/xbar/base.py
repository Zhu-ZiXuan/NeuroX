"""Abstract base for xbar-backed macros.

See also:
    docs/dev/architecture/xbar_macro.md
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from neurox.common.registry_dispatch import RegistryDispatchMixin
from neurox.common.validate import ValidateMixin
from neurox.profiler import ProfiledModule
from neurox.xbar import Xbar, XbarConfig


@dataclass(frozen=True)
class XbarMacroConfig(ValidateMixin):
    """Abstract config base for :class:`XbarMacro` subclasses.

    Attributes:
        xbar_cfg: Owned physical-xbar config; the macro constructs the
            xbar from this field via ``Xbar.from_config(...)``.
    """

    xbar_cfg: XbarConfig

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Run all ``validate_*`` checks."""


class XbarMacro(nn.Module, ProfiledModule, RegistryDispatchMixin[type["XbarMacroConfig"], "XbarMacro"], ABC):
    """Abstract base for xbar-backed quantised-MAC macros.

    Args:
        cfg: Concrete subclass config.
        name: Hierarchical profiler name.
        T__K: Operating temperature in kelvin.
        dtype: Analog forward-path dtype.
        ideal_xbar: When ``True``, the macro replaces its physical xbar
            with the lossless ideal twin returned by ``xbar.to_ideal()``.
    """

    xbar: Xbar

    def __init__(
        self,
        *,
        cfg: XbarMacroConfig,
        name: str,
        T__K: float,
        dtype: torch.dtype,
        ideal_xbar: bool,
    ) -> None:
        nn.Module.__init__(self)
        ProfiledModule.__init__(self, name)
        prefix = f"{name}." if name else ""
        xbar = Xbar.from_config(
            cfg=cfg.xbar_cfg,
            name=f"{prefix}xbar",
            T__K=T__K,
            dtype=dtype,
        )
        self.xbar = xbar.to_ideal() if ideal_xbar else xbar

    @classmethod
    def from_config(
        cls,
        *,
        cfg: XbarMacroConfig,
        name: str,
        T__K: float,
        dtype: torch.dtype,
        ideal_xbar: bool,
    ) -> XbarMacro:
        """Build the concrete impl registered for ``type(cfg)``."""
        impl = cls._lookup_impl(type(cfg))
        return impl(
            cfg=cfg,
            name=name,
            T__K=T__K,
            dtype=dtype,
            ideal_xbar=ideal_xbar,
        )

    # --- value-range / rescale contract ---

    @property
    @abstractmethod
    def w_value_range(self) -> tuple[int, int]:
        """Inclusive algorithm-side integer weight range."""
        raise NotImplementedError

    @property
    @abstractmethod
    def x_value_range(self) -> tuple[int, int]:
        """Inclusive algorithm-side integer activation range."""
        raise NotImplementedError

    @property
    @abstractmethod
    def output_rescale_factor(self) -> float:
        """Ratio of ideal integer partial-product max to actual tile output max."""
        raise NotImplementedError

    # --- lifecycle ---

    @abstractmethod
    def fabricate(self, weight: Tensor) -> None:
        """Prepare the macro for one logical weight tensor (re-callable).

        Args:
            weight: Integer weight tensor. Shape: ``[..., N, K]``.
        """
        raise NotImplementedError

    @abstractmethod
    def matmul(
        self,
        input: Tensor,
        weight: Tensor,
        bias: Tensor | None,
        rescale_multiplier: Tensor,
        rescale_rshift: Tensor,
        output_zero_point: Tensor | None,
    ) -> Tensor:
        """Execute one integer matrix multiply through the macro.

        Args:
            input: Integer activation tensor. Shape: ``[..., M, K]``.
            weight: Integer weight tensor. Shape: ``[..., N, K]``.
            bias: Optional integer bias tensor. Shape: ``[..., N]``.
            rescale_multiplier: Per-output fixed-point multiplier.
            rescale_rshift: Per-output right-shift amount.
            output_zero_point: Optional output zero point.

        Returns:
            Integer output tensor. Shape: ``[..., M, N]``.
        """
        raise NotImplementedError

    # --- shared tensor utility ---

    @staticmethod
    def chunk_pad_along(
        t: Tensor,
        *,
        axis: int,
        chunk_size: int,
        pad_value: int,
    ) -> Tensor:
        """Right-pad ``t`` along ``axis`` to a multiple of ``chunk_size``, then
        split that axis into ``(num_chunks, chunk_size)``.

        Args:
            t: Input tensor.
            axis: Axis to chunk; negative indices count from the end.
            chunk_size: Chunk size along ``axis``; must be ``>= 1``.
            pad_value: Fill value for the padding region.

        Returns:
            Tensor where ``axis`` becomes ``num_chunks`` and a new
            ``chunk_size`` axis is inserted immediately after it.
        """
        if chunk_size < 1:
            raise ValueError(f"require: chunk_size ({chunk_size}) >= 1")
        if axis < 0:
            axis += t.ndim
        if not (0 <= axis < t.ndim):
            raise ValueError(f"require: 0 <= axis ({axis}) < ndim ({t.ndim})")
        n = t.size(axis)
        num_chunks = (n + chunk_size - 1) // chunk_size
        pad_amount = num_chunks * chunk_size - n
        if pad_amount > 0:
            # F.pad indexes from the last dim; pad axis only on the high side.
            pad_spec = [0, 0] * (t.ndim - axis - 1) + [0, pad_amount]
            t = F.pad(t, pad_spec, value=pad_value)
        return t.unflatten(axis, (num_chunks, chunk_size))  # type: ignore[no-any-return]
