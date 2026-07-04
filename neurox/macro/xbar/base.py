"""Abstract base for the XbarMacro family.

See also:
    docs/reference/macro/README.md
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from neurox.analog.adc import AdcOperationPoint
from neurox.common.mixin import FabricateMixin, ProfileMixin, RegistryMixin, ValidateMixin
from neurox.xbar import Xbar, XbarConfig, XbarPolicy


@dataclass(frozen=True)
class XbarMacroConfig(ValidateMixin):
    """Abstract config root for the :class:`XbarMacro` registry."""

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Run all ``validate_*`` checks."""


@dataclass(frozen=True)
class XbarMacroPolicy:
    """Abstract marker base for XbarMacro-family nonideality policies."""


class XbarMacro(FabricateMixin, nn.Module, ProfileMixin, RegistryMixin[type["XbarMacroConfig"], "XbarMacro"], ABC):
    """Abstract base for the XbarMacro family.

    Args:
        config: Concrete configuration dataclass.
        policy: Composite nonideality policy.
        name: Hierarchical instance name used by the profiler.
        w_logical_shape: Logical weight shape ``(*prefix, N, K)`` bound to ``program(...)``.
        dtype: Tensor dtype for internal buffers.
        T__K: Operating temperature.
        ideal_xbar: Hint accepted for API uniformity. Consumed by
            xbar-using subclasses (swaps the physical xbar for its ideal
            twin); degenerate members ignore it.
    """

    config: XbarMacroConfig
    policy: XbarMacroPolicy

    def __init__(
        self,
        *,
        config: XbarMacroConfig,
        policy: XbarMacroPolicy,
        name: str,
        w_logical_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
        ideal_xbar: bool,
    ) -> None:
        nn.Module.__init__(self)
        ProfileMixin.__init__(self, name)
        if len(w_logical_shape) < 2:
            raise ValueError(f"w_logical_shape must have at least 2 trailing dims (N, K); got {w_logical_shape}")
        self.config = config
        self.policy = policy
        self._w_logical_shape = tuple(w_logical_shape)
        self._inst_shape = ()
        self._macro_dtype = dtype
        self._macro_T__K = T__K
        self._ideal_xbar = ideal_xbar
        self._macro_name = name

    @classmethod
    def from_config(
        cls,
        *,
        config: XbarMacroConfig,
        policy: XbarMacroPolicy,
        name: str,
        w_logical_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
        ideal_xbar: bool,
    ) -> XbarMacro:
        """Build the concrete impl registered for ``type(config)``."""
        impl = cls._lookup_impl(type(config))
        return impl(
            config=config,
            policy=policy,
            name=name,
            w_logical_shape=w_logical_shape,
            dtype=dtype,
            T__K=T__K,
            ideal_xbar=ideal_xbar,
        )

    def _sample_fabricate_mismatch(self) -> None:
        pass  # container: child mismatch is sampled through the cascade

    # --- PPA contract (macros aggregate via children) ---

    @property
    def area_per_inst__um2(self) -> float:
        """Silicon area per instance."""
        return 0.0

    @property
    def leakage_per_inst__uW(self) -> float:
        """Static leakage per instance."""
        return 0.0

    # --- value-range contract ---

    @property
    @abstractmethod
    def w_value_range(self) -> tuple[int, int]:
        """Inclusive integer weight range accepted by the macro."""
        raise NotImplementedError

    @property
    @abstractmethod
    def x_value_range(self) -> tuple[int, int]:
        """Inclusive integer activation range accepted by the macro."""
        raise NotImplementedError

    # --- ADC operating-point surface ---

    @property
    @abstractmethod
    def adc_mode_num(self) -> int:
        """Number of supported ADC operating points; valid ``adc_mode`` values are ``[0, adc_mode_num)``."""
        raise NotImplementedError

    @property
    @abstractmethod
    def adc_max_bits(self) -> int:
        """Maximum supported ``adc_bits`` value."""
        raise NotImplementedError

    @abstractmethod
    def adc_rescale_factor(self, adc_operation_point: AdcOperationPoint) -> float:
        """Rescale factor for ``adc_operation_point``; raises ``KeyError`` if uncalibrated."""
        raise NotImplementedError

    # --- lifecycle ---

    @abstractmethod
    def program(self, weight: Tensor) -> None:
        """Write the macro's static weight state from one logical weight tensor.

        Args:
            weight: Integer weight tensor whose shape matches
                ``self._w_logical_shape``.
        """
        raise NotImplementedError

    @abstractmethod
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
        raise NotImplementedError

    # --- xbar construction helper for xbar-using subclasses ---

    def _build_xbar(
        self,
        *,
        xbar_config: XbarConfig,
        xbar_policy: XbarPolicy,
        inst_shape: tuple[int, ...],
    ) -> Xbar:
        """Construct the owned xbar at a derived per-instance multiplicity.

        Args:
            xbar_config: Subclass-owned xbar configuration (the base does not
                require its concrete config to carry one).
            xbar_policy: Subclass-owned xbar nonideality policy.
                If ``ideal_xbar`` is true the policy is discarded in favor
                of an empty :class:`IdealXbarPolicy`.
            inst_shape: Per-instance multiplicity prefix; the xbar
                derives the trailing ``(col_num, w_digit_count,
                row_num)`` dims from its own config.

        Returns:
            The xbar (physical or ideal twin per ``ideal_xbar``).
        """
        prefix = f"{self._macro_name}." if self._macro_name else ""
        xbar = Xbar.from_config(
            config=xbar_config,
            policy=xbar_policy,
            name=f"{prefix}xbar",
            inst_shape=inst_shape,
            dtype=self._macro_dtype,
            T__K=self._macro_T__K,
        )
        return xbar.to_ideal() if self._ideal_xbar else xbar

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
        return t.unflatten(axis, (num_chunks, chunk_size))
