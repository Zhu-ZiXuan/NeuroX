"""Logical crossbar macro bridging a physical xbar tile and a full [N, K] weight matrix.

This module implements ``XbarMacro``, the algorithm-facing minimum
hardware compute unit.  It receives algorithm-side integer
activations and weights, runs them through a single unified
mapper, then through the xbar primitive (per-tile analog VMM +
ADC readout), then folds the per-tile outputs back into a single
integer result via the digital aggregation pipeline.

Composition
-----------
The macro is **composed**, not subclassed.  Mapping is owned by
one unified :class:`~neurox.mapper.xbar.XbarMapper` instance
which internally holds a shared tiler + an activation-side
slicer + a weight-side slicer.

* ``xbar``: :class:`neurox.xbar.Xbar` — VMM primitive.  Exposes
  six primitive capabilities:
  - primitive value / digit-grid capabilities:
    ``x_range``, ``w_digit_count``, ``w_digit_radix``,
    ``w_digit_range``
  - primitive geometry capabilities:
    ``col_num``, ``row_num``
  Here ``col_num`` / ``row_num`` size the tile geometry, not its
  value grid.
* ``mapper``: :class:`neurox.mapper.xbar.XbarMapper` — full
  mapping semantics owner (tiler + slicers + reshape).
* ``col_accumulator``, ``w_shift_adder``, ``x_shift_adder``,
  ``requantizer``: digital aggregation modules.

Ownership split:

* the **mapper** owns all static mapping strategy (slice counts,
  encoding choices, tile policy) on its sub-components.
* the **macro** owns the xbar and the mapper.  At every call it
  snapshots the xbar's primitive capabilities to local variables
  and passes the **same full set** as explicit keyword arguments
  into every mapper call.  No ``**dict`` splat — every cross-class
  call site documents the full argument list inline.  The macro
  never selects a subset; a different mapper drops in with no
  macro-side changes.

Tiling and encoding
-------------------
The weight pipeline (inside the mapper) is **slice-first-then-
digitize**:

1. The weight slicer decomposes the algorithm weight into
   ``slice_num`` xbar-word slices (``Sw``), each containing
   ``digit_count`` xbar-internal digit slots.
2. The tiler chops the result along ``N`` and ``K`` into
   xbar tiles of ``(data_num, row_num)``.
3. The mapper reshuffles into the macro canonical layout.

The activation pipeline (inside the mapper) is **serialise +
tile**:

    weight [Bw, N, K]  ->  [Bw, M=1, Tc, Tr, Sa=1, Sw, data_num, digit_num, row_num]
    input  [Bx, M, K]  ->  [Bx, M, Tc, Tr=1, Sa, Sw=1, row_num]

Structural singleton broadcast axes — kept so the two paths share
a single broadcast contract:

* weight: ``M = 1`` (no per-sample weight state) and ``Sa = 1``
  (weights don't depend on activation digit).
* activation: ``Tr = 1`` (activation invariant along output
  tile) and ``Sw = 1`` (no macro-external activation slicing).

The xbar primitive sees only the trailing dims of these full
macro-side layouts: ``[data_num, digit_num, row_num]`` for
``Xbar.fabricate`` and ``[row_num]`` for ``Xbar.vec_mat_mul``.

Digital aggregation
-------------------
The xbar output carries the macro-side trailing
``[..., Sa, Sw, data_num]`` (one ``data_num`` value per output
cell, one slot per activation digit, one slot per weight slice).
``_digital_aggregate`` (wrapped in ``@torch.compile``) collapses
those axes back to the algorithm-side ``N``:

1. ``x_shift_adder``  — reduce ``Sa`` at radix
   ``mapper.x_slice_radix(...)``.
2. ``w_shift_adder``  — reduce ``Sw`` at radix
   ``mapper.w_slice_radix(...)``.
3. ``column_accumulator`` — reduce ``Tc`` (K-split accumulation).
4. Flatten ``(Tr, data_num) -> N``, trim padding, add bias.

ADC rescale correction
----------------------
The operator (``neurox.operator.linear.derive_layer_int_params``)
folds the xbar's ``output_rescale_factor`` into the
``(rescale_multiplier, rescale_rshift, bias_int)`` triple at
checkpoint / HAT-forward time, so ``matmul`` collapses to a single
post-aggregation ``Requantizer`` call.

Dim symbols (algorithm vs. tile vs. xbar primitive)
---------------------------------------------------
Runtime input axes:
    Bx, M             — caller's batch dims and per-call sample axis.

Mapping result axes:
    Tc, Tr            — input / output tile counts.
    Sa                — activation digit count (serialising).
    Sw                — macro-external weight slice count.

Xbar primitive axes:
    data_num          — xbar-native data-parallel count per fab.
    digit_num         — xbar-internal digit count per xbar-word.
    row_num           — xbar input-direction cell count per tile.

Row / column terminology is decoupled from any specific tile
topology — see :class:`neurox.xbar.Xbar`'s class docstring.
"""

import math

import torch
import torch.nn as nn
from torch import Tensor

from neurox.digital import Accumulator, Requantizer, ShiftAdder
from neurox.mapper.xbar import XbarMapper
from neurox.xbar import Xbar


class XbarMacro(nn.Module):
    """Logical macro wrapping a stateless physical xbar array.

    ``XbarMacro`` is a pure **orchestrator**.  It composes a
    pre-built xbar with a pre-built unified mapper
    (:class:`~neurox.mapper.xbar.XbarMapper`) plus four digital
    aggregation modules, and exposes the algorithm-facing
    ``fabricate`` / ``matmul`` lifecycle.

    Algorithm-side value ranges (:attr:`x_value_range`,
    :attr:`w_value_range`) are pure forwards that read the xbar's
    primitive capabilities once and hand them to the mapper as
    explicit kwargs.

    In training mode ``matmul`` re-fabricates from the current
    weight tensor on every forward pass.  In eval mode it uses the
    cached physical state registered by the last ``fabricate`` call.

    ``@torch.compile`` is assumed to be applied by the caller for
    end-to-end fusion; ``_digital_aggregate`` additionally carries
    its own ``@torch.compile`` to fuse the reduction.

    Args:
        xbar: Pre-built physical crossbar simulator (the VMM
            primitive).  Source of the six primitive capabilities
            (value-domain + geometry) the macro dispatches to its
            mapper.
        mapper: Pre-built unified mapper owning its tiler +
            slicers + strategy parameters.
        col_accumulator: Pre-built digital accumulator for the
            ``Tc`` (K-tile) reduction.
        w_shift_adder: Pre-built shift-adder for the ``Sw``
            (macro-external weight slice) reduction.
        x_shift_adder: Pre-built shift-adder for the ``Sa``
            (activation digit) reduction.
        requantizer: Pre-built fixed-point requantizer applied
            after digital aggregation.
    """

    def __init__(
        self,
        *,
        xbar: Xbar,
        mapper: XbarMapper,
        col_accumulator: Accumulator,
        w_shift_adder: ShiftAdder,
        x_shift_adder: ShiftAdder,
        requantizer: Requantizer,
    ) -> None:
        super().__init__()

        self.xbar = xbar
        # Mapper is a pure tensor reshaper — assigned as a plain
        # attribute so ``nn.Module.__setattr__`` doesn't register
        # it as a child module (it isn't an ``nn.Module``).
        self.mapper = mapper

        self.col_accumulator = col_accumulator
        self.w_shift_adder = w_shift_adder
        self.x_shift_adder = x_shift_adder
        self.requantizer = requantizer

        self._serial_op_num: int = 0
        self._w_parallel_size: int = 0
        self._x_shape_cached: tuple[int, ...] = ()

    def extra_repr(self) -> str:
        """One-line summary of this macro shown by ``print(model)``."""
        return (
            f"xbar={type(self.xbar).__name__}, "
            f"row_num={self.xbar.row_num}, col_num={self.xbar.col_num}, "
            f"w_value_range={self.w_value_range}, x_value_range={self.x_value_range}"
        )

    def __repr__(self) -> str:
        """Compact repr that hides macro internals from ``print(model)``."""
        return f"{type(self).__name__}({self.extra_repr()})"

    @property
    def w_value_range(self) -> tuple[int, int]:
        """Inclusive algorithm-side integer weight range — forwards to the mapper."""
        return self.mapper.w_value_range(
            x_range=self.xbar.x_range,
            col_num=self.xbar.col_num,
            row_num=self.xbar.row_num,
            w_digit_count=self.xbar.w_digit_count,
            w_digit_radix=self.xbar.w_digit_radix,
            w_digit_range=self.xbar.w_digit_range,
        )

    @property
    def x_value_range(self) -> tuple[int, int]:
        """Inclusive algorithm-side integer activation range — forwards to the mapper."""
        return self.mapper.x_value_range(
            x_range=self.xbar.x_range,
            col_num=self.xbar.col_num,
            row_num=self.xbar.row_num,
            w_digit_count=self.xbar.w_digit_count,
            w_digit_radix=self.xbar.w_digit_radix,
            w_digit_range=self.xbar.w_digit_range,
        )

    @property
    def output_rescale_factor(self) -> float:
        """Ratio of ideal integer partial-product max to actual tile output max."""
        return self.xbar.output_rescale_factor

    def fabricate(self, weight: Tensor) -> None:
        """Map weight to tiles, transcode into digits, and program the xbar.

        Snapshots the xbar's primitive capabilities to local
        variables and passes the **full** set as explicit keyword
        arguments to :meth:`XbarMapper.map_w`, then forwards the
        resulting xbar-native digit tensor to :meth:`Xbar.fabricate`.

        Args:
            weight: Integer weight tensor. Shape: [..., N, K].
        """
        mapped = self.mapper.map_w(
            weight,
            x_range=self.xbar.x_range,
            col_num=self.xbar.col_num,
            row_num=self.xbar.row_num,
            w_digit_count=self.xbar.w_digit_count,
            w_digit_radix=self.xbar.w_digit_radix,
            w_digit_range=self.xbar.w_digit_range,
        )
        self.xbar.fabricate(mapped.w_xbar)

        # Strip the trailing ``data_num, digit_num, row_num`` to
        # leave ``[Bw, M=1, Tc, Tr, Sa=1, Sw]``.  The digit
        # axis is absorbed by the xbar fabricate call.
        *w_batch, _M, _col_tile_num, row_tile_num, _Sa, w_slice_num = mapped.w_xbar.shape[:-3]

        self._w_parallel_size = math.prod(w_batch)
        # Per-physical-instance counts pushed down to each profiled
        # digital child so ``NeuroxProfiler.analyze_static`` can sum
        # module-local totals without the macro re-aggregating.
        N = mapped.logical_out_dim
        self.col_accumulator._record_inst_count(self._w_parallel_size * w_slice_num * row_tile_num)
        self.w_shift_adder._record_inst_count(self._w_parallel_size * row_tile_num)
        self.x_shift_adder._record_inst_count(self._w_parallel_size * row_tile_num)
        self.requantizer._record_inst_count(self._w_parallel_size * N)

    def _eval_tiles(
        self,
        x: Tensor,
        N: int,
        bias: Tensor | None,
        x_slice_radix: int,
        w_slice_radix: int,
    ) -> Tensor:
        """Run the analog pipeline then reduce the digital side."""
        y = self.xbar.vec_mat_mul(x)
        y = self._digital_aggregate(y, N, bias, x_slice_radix, w_slice_radix)
        return y

    @torch.compile
    def _digital_aggregate(
        self,
        y: Tensor,
        N: int,
        bias: Tensor | None,
        x_slice_radix: int,
        w_slice_radix: int,
    ) -> Tensor:
        """Fused digital reduction: Sa -> Sw -> Tc, then flatten + bias.

        Both shift-adder radixes flow in as Python-int arguments
        from :meth:`matmul`, which queries the mapper once per
        call.  Specialising on these values lets ``torch.compile``
        constant-fold the shifts.
        """
        # Shape: y [..., M, Tc, Tr, Sa, Sw, data_num] -> [..., M, Tc, Tr, Sw, data_num].
        y = self.x_shift_adder.operate(y, x_slice_radix, dim=-3)
        # Shape: y [..., M, Tc, Tr, Sw, data_num] -> [..., M, Tc, Tr, data_num].
        y = self.w_shift_adder.operate(y, w_slice_radix, dim=-2)
        # Shape: y [..., M, Tc, Tr, data_num] -> [..., M, Tr, data_num].
        y = self.col_accumulator.operate(y, dim=-3)
        # Shape: y [..., M, Tr, data_num] -> [..., M, Tr * data_num] -> [..., M, N].
        y = y.flatten(start_dim=-2)[..., :N]
        if bias is not None:
            y = y + bias
        return y

    @torch.no_grad()
    def matmul(
        self,
        input: Tensor,
        weight: Tensor,
        bias: Tensor | None,
        rescale_multiplier: Tensor,
        rescale_rshift: Tensor,
        output_zero_point: Tensor | None,
    ) -> Tensor:
        """Compute an integer matmul end-to-end through the crossbar macro.

        Eager Python orchestrator; inner hot paths
        (``xbar.vec_mat_mul`` and ``_digital_aggregate``) carry their
        own ``@torch.compile`` decorators.

        Args:
            input: Integer activation tensor (int32). Shape: [*batch_x, M, K].
            weight: Integer weight tensor. Shape: [*batch_w, N, K].
            bias: Optional integer bias (int32). Shape: [*batch_w, N].
            rescale_multiplier: Per-channel int32 fixed-point multiplier.
            rescale_rshift: Per-channel int32 right-shift amount.
            output_zero_point: Output zero-point offset (int32), or ``None``.

        Returns:
            Output tensor (matching input dtype).
        """
        N = weight.shape[-2]
        x_dtype = input.dtype

        # Snapshot the xbar's primitive capabilities once per call to
        # local variables; every mapper call below threads the same
        # full set as explicit keyword arguments.  The macro never
        # decides which subset a mapper consumes — a different mapper
        # drops in without changing this method.
        x_range = self.xbar.x_range
        col_num = self.xbar.col_num
        row_num = self.xbar.row_num
        w_digit_count = self.xbar.w_digit_count
        w_digit_radix = self.xbar.w_digit_radix
        w_digit_range = self.xbar.w_digit_range

        if self.training:
            self.xbar.fabricate(
                self.mapper.map_w(
                    weight,
                    x_range=x_range,
                    col_num=col_num,
                    row_num=row_num,
                    w_digit_count=w_digit_count,
                    w_digit_radix=w_digit_radix,
                    w_digit_range=w_digit_range,
                ).w_xbar,
            )

        x_mapped = self.mapper.map_x(
            input,
            x_range=x_range,
            col_num=col_num,
            row_num=row_num,
            w_digit_count=w_digit_count,
            w_digit_radix=w_digit_radix,
            w_digit_range=w_digit_range,
        )
        x = x_mapped.x_xbar
        if not self.training and self._x_shape_cached != x.shape:
            self._x_shape_cached = x.shape
            x_slice_num = self._x_shape_cached[-4]
            batch_M_prod = math.prod(self._x_shape_cached[:-6])
            self._serial_op_num = batch_M_prod * x_slice_num // self._w_parallel_size

        # Query slice radixes once per call; pass into the compiled
        # aggregation as Python ints so torch.compile can specialise.
        x_slice_radix = self.mapper.x_slice_radix(
            x_range=x_range,
            col_num=col_num,
            row_num=row_num,
            w_digit_count=w_digit_count,
            w_digit_radix=w_digit_radix,
            w_digit_range=w_digit_range,
        )
        w_slice_radix = self.mapper.w_slice_radix(
            x_range=x_range,
            col_num=col_num,
            row_num=row_num,
            w_digit_count=w_digit_count,
            w_digit_radix=w_digit_radix,
            w_digit_range=w_digit_range,
        )

        y = self._eval_tiles(x, N, bias, x_slice_radix, w_slice_radix)
        y = self.requantizer.operate(y, rescale_multiplier, rescale_rshift, output_zero_point)
        return y.to(x_dtype)
