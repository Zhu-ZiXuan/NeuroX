"""Conv2dCimUnit — engine-backed ``F.conv2d`` replacement with a Toeplitz weight mapping.

See also:
    docs/internals/architecture/unit/cim/conv2d.md
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from neurox.architecture.unit.cim.base import (
    CimUnit,
    EngineBackedCimUnit,
    EngineBackedCimUnitConfig,
    EngineBackedCimUnitPolicy,
)
from neurox.architecture.unit.conv2d import Conv2dUnit


@dataclass(frozen=True, kw_only=True)
class Conv2dCimUnitConfig(EngineBackedCimUnitConfig):
    """Configuration for :class:`Conv2dCimUnit`.

    Attributes:
        stride: Output step ``(s_h, s_w)``.
        padding: Zero-pad extent ``(p_h, p_w)`` on each side.
        dilation: Kernel tap spacing ``(d_h, d_w)``.
    """

    stride: tuple[int, int]
    padding: tuple[int, int]
    dilation: tuple[int, int]

    def validate(self) -> None:
        super().validate()
        self.validate_geometry()

    def validate_geometry(self) -> None:
        """Require positive stride / dilation and non-negative padding."""
        self._require_pos(self.stride[0], "stride[0]")
        self._require_pos(self.stride[1], "stride[1]")
        self._require_pos(self.dilation[0], "dilation[0]")
        self._require_pos(self.dilation[1], "dilation[1]")
        self._require_non_neg(self.padding[0], "padding[0]")
        self._require_non_neg(self.padding[1], "padding[1]")


@dataclass(frozen=True)
class Conv2dCimUnitPolicy(EngineBackedCimUnitPolicy):
    """Composite policy for :class:`Conv2dCimUnit`; no fields beyond the inherited set."""


@CimUnit.register_key(Conv2dCimUnitConfig)
class Conv2dCimUnit(Conv2dUnit, EngineBackedCimUnit[Conv2dCimUnitConfig, Conv2dCimUnitPolicy]):
    """CIM unit exposing the conv2d operator through a Toeplitz / input-stationary mapping.

    The engine is programmed with one ``(N', K')`` Toeplitz matrix whose
    columns hold ``W_g`` stride-shifted copies of the kernel, so one
    matmul plane computes ``W_g`` consecutive output columns of one
    output row from a single gathered input strip. There are no
    per-window row masks — the Toeplitz structural zeros do the
    selecting. Every sub-phase is independently driven (no
    sample-and-hold modeling), and conv contains no chunking logic —
    sub-phase chunking is the engine's generic mechanism.

    ``W_g`` is a geometry-derived mapping policy, not a config field:
    the maximal ``W_g`` with ``K' <= xbar row_num`` and
    ``W_g * C_out <= xbar col_num``, floored at ``W_g = 1``; the generic
    engine tiling then splits ``K'`` / ``N'`` as usual. ``W_g = 1`` is
    the single-window im2col degenerate case.

    Strip geometry for weight ``(C_out, C_in, kh, kw)``, stride
    ``(s_h, s_w)``, dilation ``(d_h, d_w)``:
    ``kw_eff = (kw - 1)*d_w + 1``, ``W_strip = kw_eff + (W_g - 1)*s_w``,
    ``K' = C_in*kh*W_strip``, ``N' = W_g*C_out``.
    """

    _kw_eff: int
    _w_g: int
    _w_strip: int
    _k_prime: int
    _n_prime: int

    def __init__(
        self,
        *,
        config: Conv2dCimUnitConfig,
        policy: Conv2dCimUnitPolicy,
        w_logical_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
        ideal_xbar: bool,
    ) -> None:
        if len(w_logical_shape) != 4:
            raise ValueError(f"w_logical_shape must be (C_out, C_in, kh, kw); got {w_logical_shape}")
        c_out, c_in, kh, kw = w_logical_shape
        row_num = config.engine.cim_macro_config.row_num
        col_num = config.engine.cim_macro_config.col_num
        s_w = config.stride[1]
        d_w = config.dilation[1]
        # Toeplitz geometry before super init: the engine build reads
        # ``_engine_w_logical_shape()``.
        self._kw_eff = (kw - 1) * d_w + 1
        # Maximal W_g with K'(g) = C_in*kh*(kw_eff + (g-1)*s_w) <= row_num AND
        # g*C_out <= col_num; floor 1. K'(g) <= row_num  <=>  kw_eff +
        # (g-1)*s_w <= row_num // (C_in*kh)  <=>  g <= 1 +
        # (row_num // (C_in*kh) - kw_eff) // s_w; all quantities integral,
        # s_w >= 1. The floor at 1 covers the case where even the
        # single-window strip exceeds the xbar (K'(1) > row_num or
        # C_out > col_num) — the generic engine tiling then splits K'/N' as
        # usual (im2col degenerate).
        g_k = 1 + (row_num // (c_in * kh) - self._kw_eff) // s_w  # int arithmetic; may be <= 0
        g_n = col_num // c_out  # may be 0
        self._w_g = max(1, min(g_k, g_n))
        self._w_strip = self._kw_eff + (self._w_g - 1) * s_w
        self._k_prime = c_in * kh * self._w_strip
        self._n_prime = self._w_g * c_out
        super().__init__(
            config=config,
            policy=policy,
            w_logical_shape=w_logical_shape,
            dtype=dtype,
            T__K=T__K,
            ideal_xbar=ideal_xbar,
        )
        self._init_conv2d_operator(
            kernel_size=(kh, kw),
            stride=config.stride,
            padding=config.padding,
            dilation=config.dilation,
        )
        w_lo, w_hi = self.engine.w_value_range
        if not (w_lo <= 0 <= w_hi):
            raise ValueError(
                f"require: w_value_range ({(w_lo, w_hi)}) covers 0 — the Toeplitz matrix stores structural zeros"
            )
        if config.padding != (0, 0) or self._w_g > 1:
            x_lo, x_hi = self.engine.x_value_range
            if not (x_lo <= 0 <= x_hi):
                raise ValueError(
                    f"require: x_value_range ({(x_lo, x_hi)}) covers 0 — zero-padding, strip "
                    "right-padding, and last-segment surplus windows inject x = 0 activations"
                )

    def _engine_w_logical_shape(self) -> tuple[int, ...]:
        """Toeplitz matrix shape ``(N', K')`` handed to the engine."""
        return (self._n_prime, self._k_prime)

    def _weight_to_matrix(self, weight: Tensor) -> Tensor:
        """Toeplitz builder. Shape: [C_out, C_in, kh, kw] -> [N', K'] = [W_g*C_out, C_in*kh*W_strip].

        Placement law: entry ``weight[n, ci, i, j]`` of window
        ``g in [0, W_g)`` lands at column ``c = g*C_out + n`` (N' axis,
        row-major: g outer, n inner) and row ``r = (ci*kh + i)*W_strip +
        (g*s_w + j*d_w)`` (K' axis); all other entries are 0. Dilation
        gaps and inter-window gaps stay zero — those zeros ARE the row
        selection.
        """
        c_out, c_in, kh, kw = weight.shape
        w_g, w_strip = self._w_g, self._w_strip
        s_w = self._conv2d_stride[1]
        d_w = self._conv2d_dilation[1]
        device = weight.device
        g = torch.arange(w_g, device=device).view(-1, 1, 1, 1, 1)
        n = torch.arange(c_out, device=device).view(1, -1, 1, 1, 1)
        ci = torch.arange(c_in, device=device).view(1, 1, -1, 1, 1)
        i = torch.arange(kh, device=device).view(1, 1, 1, -1, 1)
        j = torch.arange(kw, device=device).view(1, 1, 1, 1, -1)
        # Shape: broadcast index grids [W_g, C_out, C_in, kh, kw]
        c_idx = (g * c_out + n).expand(w_g, c_out, c_in, kh, kw)
        r_idx = ((ci * kh + i) * w_strip + g * s_w + j * d_w).expand(w_g, c_out, c_in, kh, kw)
        matrix = weight.new_zeros(w_g * c_out, c_in * kh * w_strip)
        # Collision-free scatter: c fixes (g, n); given g, r fixes (ci, i, j)
        # uniquely. Index validity: max c = (W_g-1)*C_out + C_out-1 = N'-1;
        # max r = (C_in*kh - 1)*W_strip + (W_g-1)*s_w + (kw-1)*d_w
        #       = K' - W_strip + (W_g-1)*s_w + kw_eff - 1 = K'-1
        # (since W_strip = kw_eff + (W_g-1)*s_w).
        # Shape: [C_out, C_in, kh, kw] -> [W_g, C_out, C_in, kh, kw] -> flat assign into [N', K']
        matrix[c_idx.flatten(), r_idx.flatten()] = weight.unsqueeze(0).expand(w_g, -1, -1, -1, -1).flatten()
        return matrix

    def program(self, weight: Tensor, bias: Tensor | None = None) -> None:
        """Write the engine's static weight state and the optional integer bias.

        Args:
            weight: Integer weight tensor of shape ``(C_out, C_in, kh, kw)``.
            bias: Optional integer bias tensor of shape ``(C_out,)``;
                ``None`` clears any programmed bias.
        """
        if tuple(weight.shape) != self._w_logical_shape:
            raise ValueError(f"program() expects weight.shape {self._w_logical_shape}; got {tuple(weight.shape)}")
        self.engine.program(self._weight_to_matrix(weight))
        self._program_int_bias(bias, channels=self._w_logical_shape[0])

    def _conv2d_planes(self, input: Tensor, *, out_hw: tuple[int, int]) -> Tensor:
        """Strip gather. Shape: [..., C_in, H, W] -> [..., H_out, T_seg, K'].

        For output row ``ho`` and strip segment ``t`` (``T_seg =
        ceil(W_out / W_g)``), the strip reads padded input rows
        ``h = ho*s_h + i*d_h`` for ``i in [0, kh)`` and padded input
        columns ``w = t*W_g*s_w + s`` for ``s in [0, W_strip)``. The
        gather is dilation-blind along W — it copies the whole strip;
        the Toeplitz rows select the taps. Surplus windows of the last
        segment read only zero-padded columns and are trimmed at fold.
        """
        h_out, w_out = out_hw
        kh = self._conv2d_kernel_size[0]
        s_h, s_w = self._conv2d_stride
        p_h, p_w = self._conv2d_padding
        d_h = self._conv2d_dilation[0]
        w_g, w_strip = self._w_g, self._w_strip
        t_seg = -(-w_out // w_g)
        h_in, w_in = input.shape[-2:]
        # Padded extents actually touched; bottom/right padding beyond the
        # symmetric amount covers the last segment's surplus windows
        # (zero-fill).
        pad_bottom = max(0, (h_out - 1) * s_h + (kh - 1) * d_h + 1 - p_h - h_in)
        pad_right = max(0, (t_seg - 1) * w_g * s_w + w_strip - p_w - w_in)
        x = input
        if p_h or p_w or pad_bottom or pad_right:
            # Shape: [..., C_in, H, W] -> [..., C_in, Hp, Wp]   zero fill
            x = F.pad(x, (p_w, pad_right, p_h, pad_bottom))
        device = x.device
        # h_idx[ho, i] = ho*s_h + i*d_h   Shape: [H_out, kh]
        h_idx = (torch.arange(h_out, device=device) * s_h).view(-1, 1) + (torch.arange(kh, device=device) * d_h).view(
            1, -1
        )
        # w_idx[t, s] = t*W_g*s_w + s    Shape: [T_seg, W_strip]
        w_idx = (torch.arange(t_seg, device=device) * (w_g * s_w)).view(-1, 1) + torch.arange(
            w_strip, device=device
        ).view(1, -1)
        # Shape: [..., C_in, Hp, Wp] -> [..., C_in, H_out, kh, Wp]
        x = x[..., h_idx, :]
        # Shape: [..., C_in, H_out, kh, Wp] -> [..., C_in, H_out, kh, T_seg, W_strip]
        x = x[..., w_idx]
        # Shape: [..., C_in, H_out, kh, T_seg, W_strip] -> [..., H_out, T_seg, C_in, kh, W_strip]
        b = x.ndim - 5
        x = x.permute(*range(b), b + 1, b + 3, b + 0, b + 2, b + 4)
        # Row-major (C_in, kh, W_strip) flatten: plane index (ci*kh + i)*W_strip + s
        # — matches the Toeplitz row law exactly.
        # Shape: [..., H_out, T_seg, C_in, kh, W_strip] -> [..., H_out, T_seg, K']
        return x.flatten(-3)

    def _conv2d_fold(self, output: Tensor, *, out_hw: tuple[int, int]) -> Tensor:
        """Undo the conv serial axes. Shape: [..., H_out, T_seg, N'] -> [..., C_out, H_out, W_out]."""
        _h_out, w_out = out_hw
        # ``unflatten(-1, (W_g, C_out))`` matches the column law c = g*C_out + n (g outer).
        # Shape: [..., H_out, T_seg, W_g*C_out] -> [..., H_out, T_seg, W_g, C_out]
        y = output.unflatten(-1, (self._w_g, self._w_logical_shape[0]))
        # ``t*W_g + g = wo``: the flatten produces the output-column order.
        # Shape: [..., H_out, T_seg, W_g, C_out] -> [..., H_out, T_seg*W_g, C_out]
        y = y.flatten(-3, -2)
        # Trim the last segment's surplus windows (before the template's bias
        # add, so bias lands exactly once per real output element).
        # Shape: [..., H_out, T_seg*W_g, C_out] -> [..., H_out, W_out, C_out]
        y = y[..., :w_out, :]
        # Shape: [..., H_out, W_out, C_out] -> [..., C_out, H_out, W_out]
        return y.movedim(-1, -3)
