"""Ideal crossbar: tile-level lossless VMM with output quantization.

:class:`IdealXbar` is the lossless reference for any physical xbar at
the **tile** level: after the macro has sliced and tiled the matrix,
the mapper hands each tile to :meth:`fabricate(w)` as an xbar-native
digit tensor with primitive core tail
``[data_num, digit_num, row_num]``.  The ideal tile materialises the
LSB-first per-digit weight vector ``(1, r, r^2, ..., r^(D-1))`` from
the configured :attr:`w_digit_radix` and collapses the digit axis
with that vector, runs one exact integer dot-product, and
floor-quantises to the same ``output_rescale_factor`` grid the
physical xbar would impose.

The ideal tile knows nothing about offset coding, reference columns,
or the analog-domain digit-shift-add — every encoding detail
belongs to the physical xbar.  The "ideal" in :class:`IdealXbar`
refers to the analog side: every non-ideality (IR drop, noise,
finite gain, ...) is dropped, but the integer-output grid and the
input contract are preserved so the upstream macro / mapper drops in
unchanged.

For a macro-level reference that skips tiling entirely, use
:class:`neurox.macro.ideal.IdealMacro`.
"""

import torch
from torch import Tensor

from neurox.common.quant import stochastic_floor_to_int

from .base import Xbar, XbarConfig


class IdealXbar(Xbar):
    """Tile-level ideal VMM with output quantization.

    ``fabricate(w)`` stores the xbar-native digit tensor verbatim;
    ``vec_mat_mul(x)`` reconstructs the logical weight by collapsing
    the digit axis with the radix-derived weight vector
    :attr:`digit_weights`, runs an exact integer dot product, and
    quantises via :attr:`Xbar.output_rescale_factor`.

    Args:
        cfg: Base :class:`XbarConfig` carrying tile geometry, runtime
            ADC operating point, the ``(adc_mode, adc_bits) -> rf``
            lookup, and PPA.
        x_range: Inclusive logical input range — ``(0, 1)`` for the
            current 1T1R family.
        w_digit_count: Number of digits per ``w`` the fabricate
            input tensor carries.  Mirrors the physical xbar's
            :attr:`Xbar.w_digit_count`.
        w_digit_radix: Positional base ``r`` of the in-tile digit
            combination.  The ideal tile materialises the LSB-first
            weight vector ``(1, r, r^2, ..., r^(D-1))`` once at
            ``__init__`` and uses it to collapse the digit axis at
            VMM time.
        w_digit_range: Inclusive integer range a single digit
            accepts.  Reported through :attr:`Xbar.w_digit_range`;
            the ideal tile does not enforce it at fabricate time.
        stochastic: Optional override for stochastic-floor rounding.
            ``None`` (default) defers to ``self.training``.
    """

    weight: Tensor
    digit_weights: Tensor

    def __init__(
        self,
        cfg: XbarConfig,
        *,
        name: str = "",
        x_range: tuple[int, int],
        w_digit_count: int,
        w_digit_radix: int,
        w_digit_range: tuple[int, int],
        stochastic: bool | None = None,
    ) -> None:
        super().__init__(cfg, name=name)
        if w_digit_count <= 0:
            raise ValueError(f"require: w_digit_count ({w_digit_count}) > 0")
        if w_digit_radix <= 1:
            raise ValueError(f"require: w_digit_radix ({w_digit_radix}) > 1")
        self._x_range = x_range
        self._w_digit_count = w_digit_count
        self._w_digit_radix = w_digit_radix
        self._w_digit_range = w_digit_range
        self._stochastic_override = stochastic

        # Placeholder until ``fabricate`` programs the tile.
        self.register_buffer("weight", torch.empty(0, dtype=torch.int32), persistent=False)
        # Materialise the LSB-first digit weight vector
        # ``(1, r, r^2, ..., r^(D-1))`` from the radix.  Kept as a
        # float buffer so ``module.to(device)`` migrates it with
        # the rest of the xbar; the actual dtype follows the input
        # ``x`` at VMM time.
        digit_weights = torch.tensor(
            [w_digit_radix**k for k in range(w_digit_count)],
            dtype=torch.int32,
        )
        self.register_buffer("digit_weights", digit_weights, persistent=False)

    @property
    def x_range(self) -> tuple[int, int]:
        return self._x_range

    @property
    def w_digit_count(self) -> int:
        return self._w_digit_count

    @property
    def w_digit_radix(self) -> int:
        return self._w_digit_radix

    @property
    def w_digit_range(self) -> tuple[int, int]:
        return self._w_digit_range

    def fabricate(self, w: Tensor) -> None:
        """Register the xbar-native digit tensor as the tile weight.

        Args:
            w: Digit tensor, integer-valued.  Primitive core tail
                ``[data_num, digit_num, row_num]`` (full caller
                layout is the macro's
                ``[Bw, M=1, Tc, Tr, Sa=1, Sw, data_num, digit_num,
                row_num]``).
        """
        self._record_xbar_inst_count(w)
        self.register_buffer("weight", w, persistent=False)

    @torch.compile
    def vec_mat_mul(self, x: Tensor) -> Tensor:
        """Ideal VMM + output quantization.

        Implements the generic xbar primitive contract — see
        :meth:`Xbar.vec_mat_mul` — so ``x`` need only carry the
        primitive trailing ``[row_num]`` and the output carries the
        primitive trailing ``[data_num]``.  Any macro-side leading
        axes (``Bx``, ``M``, ``Tc``, ``Sa``, ``Sw``) ride along as
        opaque broadcast batch dims.

        The ideal tile collapses the xbar-internal digit axis by an
        LSB-first weighted sum, broadcasts the activation against
        the recombined tile weight, and floor-quantises by
        :attr:`Xbar.output_rescale_factor`.  No analog non-ideality;
        no profiler event emitted.

        Args:
            x: Activation tensor with primitive trailing
                ``[row_num]``.  Each entry must lie in
                :attr:`x_range`.

        Returns:
            ``code`` with primitive trailing ``[data_num]``.
        """
        # Stored fab tensor: [..., data_num, digit_num, row_num].
        digits = self.weight
        # Collapse the digit axis to recover the logical xbar-word
        # weight tile.  Shape: w_logic -> [..., data_num, row_num].
        digit_weights = self.digit_weights.to(digits.dtype).view(*([1] * (digits.ndim - 2)), -1, 1)
        w_logic = (digits * digit_weights).sum(dim=-2)

        # Implementation-specific: insert a size-1 slot at -2 so
        # ``w_logic``'s data axis broadcasts cleanly against ``x``.
        # Shape: [..., row_num] -> [..., 1, row_num].
        x = x.unsqueeze(-2)

        full_shape = torch.broadcast_shapes(w_logic.shape, x.shape)
        w_logic = w_logic.expand(full_shape)
        x = x.expand(full_shape)
        # Shape: dot -> [..., data_num]
        dot = (x * w_logic).sum(dim=-1)

        rf = self.output_rescale_factor
        code = stochastic_floor_to_int(
            dot.to(torch.float32),
            rf,
            out_dtype=torch.int16,
            training=self.training,
            override=self._stochastic_override,
        )
        return code
