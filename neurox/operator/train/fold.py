"""BatchNorm folding helper for hardware-aware training.

Folds each ``Conv2d → BatchNorm2d`` (or ``Linear → BatchNorm1d``) pair
into the weight / bias of the preceding linear op, then replaces the
BatchNorm module with ``nn.Identity``.  This is the standard
pre-processing step before QAT / HAT: since BN's affine transform is
trivially absorbable into a linear layer's parameters, there's no
reason to keep it as a separate op once its float running stats are
frozen.

Call ``fold_batchnorm(model)`` *before* :func:`neurox.replace_for_hat`
so the HAT replacement sees a BN-free graph.  LeNet has no BN, so
this is a no-op for the current example; it is here for future VGG /
MobileNet examples.

Limitations (by design, keep Python-level simple):

- Only folds when the BN IMMEDIATELY follows a Conv/Linear at the same
  ``nn.Module`` level (via ``named_children``).  Skip-connection or
  separately-scoped BN is left alone.
- Only handles ``nn.BatchNorm2d`` / ``nn.BatchNorm1d``.  ``LayerNorm``
  and ``GroupNorm`` are not fold-compatible (they normalize across
  different axes) and stay as-is.
- Assumes BN is in eval mode or carries valid running stats.  The
  caller should have already finished any BN-updating training
  (typically the float pretraining phase) before folding.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


def _fold_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> None:
    """In-place fold of ``bn`` into ``conv``'s weight and bias."""
    if not bn.track_running_stats or bn.running_mean is None or bn.running_var is None:
        raise RuntimeError(f"cannot fold BN without running stats: {bn}")
    gamma = bn.weight if bn.affine and bn.weight is not None else torch.ones_like(bn.running_var)
    beta = bn.bias if bn.affine and bn.bias is not None else torch.zeros_like(bn.running_var)
    std = torch.sqrt(bn.running_var + bn.eps)
    scale = gamma / std  # shape: [C_out]

    w_shape = [-1, 1, 1, 1]
    with torch.no_grad():
        conv.weight.mul_(scale.view(w_shape).to(conv.weight.dtype))
        if conv.bias is None:
            # Allocate a bias so the folded offset lands somewhere.
            conv.bias = nn.Parameter(torch.zeros(conv.out_channels, dtype=conv.weight.dtype, device=conv.weight.device))
        folded_bias: Tensor = (conv.bias - bn.running_mean) * scale + beta
        conv.bias.copy_(folded_bias.to(conv.bias.dtype))


def _fold_linear_bn(linear: nn.Linear, bn: nn.BatchNorm1d) -> None:
    """In-place fold of ``bn`` into ``linear``'s weight and bias."""
    if not bn.track_running_stats or bn.running_mean is None or bn.running_var is None:
        raise RuntimeError(f"cannot fold BN without running stats: {bn}")
    gamma = bn.weight if bn.affine and bn.weight is not None else torch.ones_like(bn.running_var)
    beta = bn.bias if bn.affine and bn.bias is not None else torch.zeros_like(bn.running_var)
    std = torch.sqrt(bn.running_var + bn.eps)
    scale = gamma / std  # shape: [out_features]
    with torch.no_grad():
        linear.weight.mul_(scale.view(-1, 1).to(linear.weight.dtype))
        if linear.bias is None:
            linear.bias = nn.Parameter(
                torch.zeros(linear.out_features, dtype=linear.weight.dtype, device=linear.weight.device)
            )
        folded_bias: Tensor = (linear.bias - bn.running_mean) * scale + beta
        linear.bias.copy_(folded_bias.to(linear.bias.dtype))


def fold_batchnorm(model: nn.Module) -> nn.Module:
    """Walk ``model`` and fold each Conv+BN / Linear+BN pair.

    The BN modules are replaced with ``nn.Identity`` so downstream
    ``replace_for_hat`` walkers see only Conv / Linear / unknowns.

    Returns the same ``model`` object (mutated in place) for chaining.
    """

    def _walk(module: nn.Module) -> None:
        children = list(module.named_children())
        # We're matching (prev, curr) pairs; use an index so we can swap out both.
        for idx in range(len(children) - 1):
            prev_name, prev_mod = children[idx]
            curr_name, curr_mod = children[idx + 1]
            if isinstance(prev_mod, nn.Conv2d) and isinstance(curr_mod, nn.BatchNorm2d):
                _fold_conv_bn(prev_mod, curr_mod)
                setattr(module, curr_name, nn.Identity())
            elif isinstance(prev_mod, nn.Linear) and isinstance(curr_mod, nn.BatchNorm1d):
                _fold_linear_bn(prev_mod, curr_mod)
                setattr(module, curr_name, nn.Identity())
        # Recurse after potential replacements at this level.
        for _, child in module.named_children():
            _walk(child)

    _walk(model)
    return model
