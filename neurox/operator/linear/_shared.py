"""Shared int-domain math for the Linear operators.

Two module-level helpers used by every crossbar-backed operator (the
inference :class:`QuantLinear`, the HAT :class:`HATLinear`, and — via
cross-package imports — both Conv2d operators):

- :func:`run_matmul_pipeline` — int matmul → activation-grid clamp →
  float dequantization.
- :func:`derive_layer_int_params` — turns the float quantization
  parameters into the int tensor 5-tuple the macro consumes.

These live in ``linear/_shared.py`` because the matmul kernel is the
canonical NeuroX operator math; conv just regroups its inputs / outputs
around the same kernel.
"""

import torch
from torch import Tensor

from neurox.macro.base import NeuroxMacroQuantMatMul

from ..qat_util import derive_multiplier_and_shift_tensor


def run_matmul_pipeline(
    input_int: Tensor,
    weight_int: Tensor,
    bias_int: Tensor,
    rescale_multiplier: Tensor,
    rescale_rshift: Tensor,
    output_zero_point: Tensor,
    output_scale: Tensor,
    output_qmin: int | Tensor,
    output_qmax: int | Tensor,
    macro: NeuroxMacroQuantMatMul,
) -> Tensor:
    """Execute one int matmul through the crossbar macro, clamp, dequantize.

    ``output_qmin`` / ``output_qmax`` may be Python ints or 0-d int tensors;
    callers inside a ``@torch.compile`` region should pass tensor buffers
    directly to avoid CPU-sync ``.item()`` calls in the compiled graph.

    Args:
        input_int: Integer activation tensor already quantized to the grid.
        weight_int: Integer weight tensor already quantized to ``[-w_qmax, +w_qmax]``.
        bias_int: Folded int32 bias (``round(b / (sx·sw)) - zp_x · sum(w_int)``).
        rescale_multiplier: Per-channel int32 fixed-point multiplier.
        rescale_rshift: Per-channel int32 right-shift.
        output_zero_point: Scalar int32 output zero-point.
        output_scale: Scalar float32 output scale (used for dequantization).
        output_qmin: Post-rescale clamp lower bound (output activation grid min).
        output_qmax: Post-rescale clamp upper bound.
        macro: Crossbar macro running the int matmul + rescale.

    Returns:
        Dequantized float output.  Dynamic energy emits through the
        profiler side channel from every physical leaf inside the
        macro.
    """
    y_int = macro.matmul(
        input_int,
        weight_int,
        bias_int,
        rescale_multiplier,
        rescale_rshift,
        output_zero_point,
    )
    # Clamp to the activation grid to match pt2e's Q/DQ saturation at each
    # layer boundary (see module docstring for rationale).
    y_int = torch.clamp(y_int, output_qmin, output_qmax)
    y_float = (y_int.float() - output_zero_point.float()) * output_scale
    return y_float


def derive_layer_int_params(
    float_weight: Tensor,
    float_bias: Tensor | None,
    input_scale: Tensor,
    input_zero_point: Tensor,
    weight_scale: Tensor,
    output_scale: Tensor,
    w_qmax: int,
    weight_dtype: torch.dtype,
    rescale_factor: float = 1.0,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Build the ``(weight_int, bias_int, multiplier, rshift)`` tuple.

    Shared between :class:`HATLinear` / :class:`HATConv2d` (called
    every forward from the current float weight / bias) and the pt2e
    extractor in ``example/common/pt2e.py`` (called once at export
    time).

    The crossbar emits ``y_agg`` in ADC-code scale — a factor ``rf``
    smaller than the ideal integer MAC would produce, where
    ``rf = N_states / N_codes = macro.output_rescale_factor``.  To
    collapse the old two-step "ADC-correct → bias → requantize" into
    a single macro-side requantize, we fold ``rf`` into the params
    this function returns:

        combined_scale = (s_x * s_w / s_y) * rf
        (multiplier, rshift) = derive_multiplier_and_shift_tensor(combined_scale)
        bias_int = round((bias_fp / (s_x * s_w) - zp_x * sum(w_int)) / rf)

    The macro then computes ``requantize(y_agg + bias_int, ...)``
    in one shot and the math agrees with the unfolded reference up
    to the unavoidable rounding loss in ``bias_int``.  When
    ``rf == 1.0`` (e.g. ``IdealMacro`` or an ADC whose codes already
    cover the ideal state range) the formulas collapse to the
    original scale-only derivation.

    The output integer dtype for ``weight_int`` is supplied explicitly
    by the caller — the helper is pure math and does not own the dtype
    policy.  Callers derive it from ``macro.w_value_range`` via
    ``NeuroxOperator.weight_dtype_for_range`` so the int buffer always
    matches the algorithm-side range the macro publishes (e.g.
    ``w_qmax > 127`` selects ``int16`` instead of saturating to int8).

    Args:
        float_weight: The trained float weight, shape ``[C_out, ...]``.
        float_bias: Optional trained float bias, shape ``[C_out]``.
        input_scale: Activation per-tensor scale, scalar float32.
        input_zero_point: Activation per-tensor zero-point, scalar int32.
        weight_scale: Per-channel symmetric scale, shape ``[C_out]``.
        output_scale: Output per-tensor scale, scalar float32 (used only
            to derive the fixed-point rescale pair).
        w_qmax: Symmetric weight bound for clipping.
        weight_dtype: Target integer dtype for ``weight_int``; must be
            wide enough for ``[-w_qmax, +w_qmax]``.
        rescale_factor: Macro's ``output_rescale_factor``
            (``N_states / N_codes``).  Multiplied into ``(mult, rshift)``
            and divided into the folded bias so the macro's single
            requantize produces correctly-scaled output.

    Returns:
        ``(weight_int, bias_int, multiplier, rshift)`` — all on the
        same device as ``float_weight``; ``weight_int`` has dtype
        ``weight_dtype``.
    """
    device = float_weight.device
    shape = [weight_scale.shape[0]] + [1] * (float_weight.ndim - 1)
    sw = weight_scale.view(shape).to(device)
    weight_int = torch.round(float_weight / sw).clamp(-w_qmax, w_qmax).to(weight_dtype)

    # Bias fold: subtract the activation zero-point cross-term so the int
    # MAC accumulator produces the right numerator for the rescale step,
    # then divide by ``rescale_factor`` to compensate for ``y_agg`` living
    # in ADC-code scale (see function docstring).  One combined
    # ``round(...)`` keeps precision loss to ±0.5 ADC-code units.
    k_axes = tuple(range(1, weight_int.ndim))
    w_sum = weight_int.to(torch.int64).sum(dim=k_axes) if k_axes else weight_int.to(torch.int64)
    sx = input_scale.to(torch.float64).to(device)
    zp_x = input_zero_point.to(torch.float64).to(device)
    sw_f64 = weight_scale.to(torch.float64).to(device)
    rf = rescale_factor if rescale_factor != 0.0 else 1.0
    if float_bias is not None:
        bias_ideal = float_bias.to(torch.float64).to(device) / (sx * sw_f64).clamp(min=1e-30)
    else:
        bias_ideal = torch.zeros(float_weight.shape[0], dtype=torch.float64, device=device)
    folded = torch.round((bias_ideal - zp_x * w_sum.to(torch.float64)) / rf)
    int32_info = torch.iinfo(torch.int32)
    bias_int = folded.clamp(min=int32_info.min, max=int32_info.max).to(torch.int32)

    combined = (sx * sw_f64) * rf / output_scale.to(torch.float64).to(device).clamp(min=1e-30)
    multiplier, rshift = derive_multiplier_and_shift_tensor(combined.to(torch.float32))
    return weight_int, bias_int, multiplier, rshift
