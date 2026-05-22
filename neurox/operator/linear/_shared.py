"""Shared int-domain math for the Linear operators.

- :func:`derive_layer_int_params` — float quantization params → int 5-tuple
  consumed by the macro and the operator. The post-macro requantize +
  clamp + dequant lives as :meth:`NeuroxOperator.run_matmul_pipeline`.
"""

import torch
from torch import Tensor

from neurox.operator.qat_util import derive_multiplier_and_shift_tensor


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

    Folds ``rescale_factor`` (ADC-code-to-ideal-state ratio) into a single requantize::

        combined_scale = (s_x · s_w / s_y) · rf
        (multiplier, rshift) = derive_multiplier_and_shift_tensor(combined_scale)
        bias_int = round((bias_fp / (s_x · s_w) - zp_x · Σ w_int) / rf)

    Args:
        float_weight: Trained float weight, shape ``[C_out, ...]``.
        float_bias: Optional trained float bias, shape ``[C_out]``.
        input_scale: Activation per-tensor scale.
        input_zero_point: Activation per-tensor zero-point.
        weight_scale: Per-channel symmetric scale, shape ``[C_out]``.
        output_scale: Output per-tensor scale.
        w_qmax: Symmetric weight bound for clipping.
        weight_dtype: Target integer dtype for ``weight_int``.
        rescale_factor: Macro's ``output_rescale_factor``; ``1.0`` collapses
            the formulas to the scale-only case.

    Returns:
        ``(weight_int, bias_int, multiplier, rshift)``.
    """
    device = float_weight.device
    shape = [weight_scale.shape[0]] + [1] * (float_weight.ndim - 1)
    sw = weight_scale.view(shape).to(device)
    weight_int = torch.round(float_weight / sw).clamp(-w_qmax, w_qmax).to(weight_dtype)

    # Subtract zero-point cross-term, then divide by ``rescale_factor`` so
    # the int MAC output lands on the right post-rescale grid.
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
