"""Abstract base for the CimEngine family.

See also:
    docs/internals/architecture/unit/cim/engine/base.md
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar, Generic, TypeVar

import torch
import torch.nn.functional as F
from torch import Tensor

from neurox.common import ConfigBase, ModuleBase, PolicyBase
from neurox.common.encoding import Encoding
from neurox.common.mixin import RegistryMixin
from neurox.primitive.digital import AccumulatorConfig
from neurox.primitive.macro.cim import CimMacro, CimMacroConfig, CimMacroPolicy


@dataclass(frozen=True)
class CimEngineConfig(ConfigBase, ABC):
    """Abstract config root for the :class:`CimEngine` registry.

    The concrete subclass type selects the engine variant through the
    ``_neurox_class`` discriminator when nested inside a unit config.

    Attributes:
        cim_macro_config: Owned physical-xbar config.
        w_encoding: Signed-digit encoding for the weight transcoder / slicer.
        phase_accumulator_config: Sub-phase-axis per-tile-port accumulator config.
        col_accumulator_config: Tc-axis cross-tile accumulator config.
    """

    cim_macro_config: CimMacroConfig
    w_encoding: Encoding

    phase_accumulator_config: AccumulatorConfig
    col_accumulator_config: AccumulatorConfig

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Run all ``validate_*`` checks."""


@dataclass(frozen=True)
class CimEnginePolicy(PolicyBase):
    """Composite policy the owning unit assembles for its engine in code.

    Attributes:
        cim_macro_policy: Embedded xbar nonideality policy.
    """

    cim_macro_policy: CimMacroPolicy


ConfigT = TypeVar("ConfigT", bound=CimEngineConfig)


class CimEngine(
    ModuleBase[ConfigT, CimEnginePolicy],
    RegistryMixin[type["CimEngineConfig"], "CimEngine"],
    Generic[ConfigT],
    ABC,
):
    """Abstract base for the CIM execution engines.

    An engine is the workload-agnostic slicing / tiling / macro-cycle /
    aggregation pipeline behind a CIM unit: it owns the xbar and the
    digital reduction primitives and exposes the ``torch.matmul``-shaped
    integer contract the unit delegates to. It is a container with no own
    PPA (``is_profile_target`` is ``False``); its children self-report.

    Args:
        config: Concrete configuration dataclass.
        policy: Composite nonideality policy assembled by the owning unit.
        w_logical_shape: Logical weight shape ``(*prefix, N, K)`` bound to ``program(...)``.
        dtype: Tensor dtype for internal buffers.
        T__K: Operating temperature.
        ideal_xbar: Swap the physical xbar for its ideal twin.
    """

    is_profile_target: ClassVar[bool] = False

    xbar: CimMacro
    _active_row_mask: Tensor
    _w_value_range: tuple[int, int]
    _x_value_range: tuple[int, int]
    _xbar_inst_rank: int
    _sub_phase_num: int
    _sub_phase_dim: int
    _n_logical: int
    _w_parallel_size: int
    _row_tile_num: int

    def __init__(
        self,
        *,
        config: ConfigT,
        policy: CimEnginePolicy,
        w_logical_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
        ideal_xbar: bool,
    ) -> None:
        ModuleBase.__init__(self, config=config, policy=policy, inst_shape=())
        if len(w_logical_shape) < 2:
            raise ValueError(f"w_logical_shape must have at least 2 trailing dims (N, K); got {w_logical_shape}")
        self._w_logical_shape = tuple(w_logical_shape)
        self._macro_dtype = dtype
        self._macro_T__K = T__K
        self._ideal_xbar = ideal_xbar

    @classmethod
    def from_config(
        cls,
        *,
        config: CimEngineConfig,
        policy: CimEnginePolicy,
        w_logical_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
        ideal_xbar: bool,
    ) -> CimEngine:
        """Build the concrete impl registered for ``type(config)``."""
        impl = cls._lookup_impl(type(config))
        return impl(
            config=config,
            policy=policy,
            w_logical_shape=w_logical_shape,
            dtype=dtype,
            T__K=T__K,
            ideal_xbar=ideal_xbar,
        )

    def _sample_fabricate_mismatch(self) -> None:
        pass  # container: child mismatch is sampled through the cascade

    # --- shared engine backend: xbar + tiling + sub-phase machinery ---

    def _init_engine_backend(
        self,
        *,
        inst_shape: tuple[int, ...],
        n_logical: int,
        k_logical: int,
        w_parallel_size: int,
        row_tile_num: int,
    ) -> None:
        """Build the owned xbar plus shared tiling and sub-phase machinery.

        Args:
            inst_shape: Per-instance multiplicity prefix for the owned xbar.
            n_logical: Logical output width ``N`` before tile padding.
            k_logical: Logical contraction width ``K`` before tile padding;
                fixes how many sub-phases carry real (non-padding) rows.
            w_parallel_size: Parallel weight-instance count (``prod(w_batch)``).
            row_tile_num: Output tile count ``Tr``.
        """
        self.xbar = self._build_cim_macro(
            xbar_config=self.config.cim_macro_config,
            xbar_policy=self.policy.cim_macro_policy,
            inst_shape=inst_shape,
        )
        self._n_logical = n_logical
        self._w_parallel_size = w_parallel_size
        self._row_tile_num = row_tile_num
        self._xbar_inst_rank = len(inst_shape)
        self._sub_phase_dim = -(len(inst_shape) + 2)
        row_num = self.xbar.row_num
        max_rows = self.xbar.max_active_rows
        # Skip sub-phases that would read only zero-padded rows. A tile spans at
        # most row_num rows, but when the logical contraction dim k_logical fits
        # in a single tile the widest tile carries only k_logical real rows
        # (rows beyond it are zero padding in every tile). The widest tile's
        # real-row count is therefore min(k_logical, row_num): a full tile
        # (k_logical > row_num) spans row_num, a single short tile spans
        # k_logical. Sub-phases past that extent read only padding and
        # contributed exactly 0, so dropping them is bit-identical while cutting
        # per-op / latency energy for narrow (K < row_num) layers.
        real_row_extent = min(k_logical, row_num)
        # Ceil so a non-divisible geometry still covers every real row: the last
        # kept sub-phase reads the short remainder block (< max_rows live rows).
        self._sub_phase_num = -(-real_row_extent // max_rows)
        # Static row -> sub-phase WL mask; zero-fill = WL off. Shape: [P, row_num]
        # Row r belongs to sub-phase r // max_rows; rows whose block index reaches
        # P are pure padding and stay off in every plane. Every real row lands in
        # exactly one sub-phase and the short final block leaves its own rows the
        # only ones live in that plane.
        self.register_buffer(
            "_active_row_mask",
            torch.arange(row_num) // max_rows == torch.arange(self._sub_phase_num).unsqueeze(-1),
            persistent=False,
        )

    def _unroll_sub_phase(self, x: Tensor) -> Tensor:
        """Expand WL planes over the hardware sub-phase axis.

        The P axis is inserted immediately LEFT of the xbar's inst-aligned
        trailing block (right of all other batch axes); always present,
        size 1 when ``active_row_num == row_num``. Rows outside a plane's
        active window are zeroed (WL off).
        """
        inst_rank = self._xbar_inst_rank
        # Shape: [P, row_num] -> [P, 1*inst_rank, row_num]
        mask = self._active_row_mask.reshape(-1, *(1,) * inst_rank, self.xbar.row_num)
        # Shape: [..., *span, row_num] -> [..., P, *span, row_num]
        return torch.where(mask, x.unsqueeze(max(-(inst_rank + 2), -(x.ndim + 1))), x.new_zeros(()))

    # --- value-range contract ---

    @property
    def w_value_range(self) -> tuple[int, int]:
        """Inclusive integer weight range accepted by the engine."""
        return self._w_value_range

    @property
    def x_value_range(self) -> tuple[int, int]:
        """Inclusive integer activation range accepted by the engine."""
        return self._x_value_range

    # --- ADC operating-point surface ---

    @property
    def adc_mode_num(self) -> int:
        """Number of supported ADC operating points; valid ``adc_mode`` values are ``[0, adc_mode_num)``."""
        return self.xbar.adc_mode_num

    @property
    def adc_max_bits(self) -> int:
        """Maximum supported ``adc_bits`` value."""
        return self.xbar.adc_max_bits

    def adc_rescale_factor(self, *, adc_mode: int, adc_bits: int) -> float:
        """Rescale factor for ``(adc_mode, adc_bits)``; raises ``KeyError`` if uncalibrated."""
        return self.xbar.adc_rescale_factor(adc_mode=adc_mode, adc_bits=adc_bits)

    # --- lifecycle ---

    def program(self, weight: Tensor) -> None:
        """Write the xbar's static weight state from one logical weight tensor.

        Args:
            weight: Integer weight tensor whose shape matches
                ``self._w_logical_shape``.
        """
        if tuple(weight.shape) != self._w_logical_shape:
            raise ValueError(f"program() expects weight.shape {self._w_logical_shape}; got {tuple(weight.shape)}")
        self.xbar.program(self._organize_w(weight))

    @abstractmethod
    def _organize_w(self, weight: Tensor) -> Tensor:
        """Map a logical weight tensor ``[..., N, K]`` into xbar-native layout."""
        raise NotImplementedError

    @abstractmethod
    def _organize_x(self, x: Tensor) -> Tensor:
        """Map a logical activation tensor ``[..., M, K]`` into xbar-native layout."""
        raise NotImplementedError

    @abstractmethod
    def matmul(self, input: Tensor, *, adc_mode: int, adc_bits: int) -> Tensor:
        """Execute one integer matrix multiply against the programmed weight state.

        Matches ``torch.matmul`` semantics (pure matmul, no bias).
        Internally, the engine expands WL planes over its hardware
        sub-phase axis (``P = ceil(min(K, row_num) / max_active_rows)``,
        always present, immediately left of the xbar's inst-aligned block,
        skipping the trailing all-padding blocks of a narrow layer); the
        xbar returns per-plane codes with primitive trailing
        ``[col_num]`` and leading order preserved; the engine sums
        exactly its own sub-phase axis (``phase_accumulator``) before
        the leaf-specific slice/tile reductions (``col_accumulator`` on
        ``Tc``, shift-adders on ``Sa``/``Sw``).

        Args:
            input: Integer activation tensor. Shape: ``[..., M, K]``.
            adc_mode: Runtime ADC operating-point index.
            adc_bits: Runtime ADC resolution.

        Returns:
            Integer pre-requantize output tensor. Shape: ``[..., M, N]``.
        """
        raise NotImplementedError

    # --- repr ---

    def extra_repr(self) -> str:
        return (
            f"xbar={type(self.xbar).__name__}, "
            f"row_num={self.xbar.row_num}, col_num={self.xbar.col_num}, "
            f"w_value_range={self.w_value_range}, x_value_range={self.x_value_range}"
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.extra_repr()})"

    # --- xbar construction helper for concrete engines ---

    def _build_cim_macro(
        self,
        *,
        xbar_config: CimMacroConfig,
        xbar_policy: CimMacroPolicy,
        inst_shape: tuple[int, ...],
    ) -> CimMacro:
        """Construct the owned xbar at a derived per-instance multiplicity.

        Args:
            xbar_config: Engine-owned xbar configuration.
            xbar_policy: Engine-owned xbar nonideality policy.
                If ``ideal_xbar`` is true the policy is discarded in favor
                of an empty :class:`IdealCimMacroPolicy`.
            inst_shape: Per-instance multiplicity prefix; the xbar
                derives the trailing ``(col_num, w_digit_count,
                row_num)`` dims from its own config.

        Returns:
            The xbar (physical or ideal twin per ``ideal_xbar``).
        """
        xbar = CimMacro.from_config(
            config=xbar_config,
            policy=xbar_policy,
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
