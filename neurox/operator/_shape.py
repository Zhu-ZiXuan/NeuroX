"""Shape helpers for crossbar-backed operators."""

import torch.nn as nn


def w_logical_shape_for(module: nn.Module) -> tuple[int, ...]:
    """Derive the macro ``w_logical_shape`` for a float ``nn.Linear``/``nn.Conv2d``.

    Args:
        module: A ``nn.Linear`` (``[out_features, in_features]``) or
            ``nn.Conv2d`` (per-group ``[groups, out_per_group,
            in_per_group * kH * kW]``) module.

    Returns:
        Logical weight shape ``(*prefix, N, K)`` consumed by
        ``XbarMacro.from_config(w_logical_shape=...)`` and by the operator
        macro factory.

    Raises:
        TypeError: ``module`` is neither ``nn.Linear`` nor ``nn.Conv2d``.
    """
    if isinstance(module, nn.Linear):
        return (module.out_features, module.in_features)
    if isinstance(module, nn.Conv2d):
        kh, kw = module.kernel_size
        in_per_group = module.in_channels // module.groups
        out_per_group = module.out_channels // module.groups
        return (module.groups, out_per_group, in_per_group * kh * kw)
    raise TypeError(f"w_logical_shape_for: expected nn.Linear or nn.Conv2d, got {type(module).__name__}")
